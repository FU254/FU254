"""Shared worker for controlled equal/CA/RL teacher-weight experiments.

The existing teacher restoration, shared views, optimizer, patient metrics and
student checkpoint format remain the implementation of those operations. LR
pilots train on train/validation only; a selected equal pilot is evaluated by
adding separate artifacts without altering its training result or checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import distillation as single
from . import distillation_suite as records
from . import equal_teacher_distillation as equal
from . import random_student_distillation as random_student
from . import train as supervised
from .equal_teacher_sources import load_teacher, validate_teacher_cohorts
from .equal_teacher_losses import TEACHER_NAMES
from .metrics import aggregate_patient_predictions, binary_metrics, save_evaluation, select_threshold
from .models.vit_tiny_student import build_vit_tiny_student, _strict_load
from .teacher_weight_losses import WeightedTeacherDistillationLoss


EXPERIMENT = "teacher_weight_kd_v1"
CONDITION = "teacher_weight"
LOSS_SETTINGS = deepcopy(equal.LOSS_SETTINGS)
HEAD_SEED_XOR = 0x48454144
PROJECTOR_SEED_XOR = 0x50524F4A
METHODS = ("equal", "ca", "rl")
INITIALIZATIONS = ("random", "imagenet")
PHASES = ("pilot", "main")


def _validate_config(config):
    if config.get("student_initialization") not in INITIALIZATIONS:
        raise ValueError("Declare student_initialization=random or imagenet")
    if config.get("method") not in METHODS or config.get("phase") not in PHASES:
        raise ValueError("Declare method=equal/ca/rl and phase=pilot/main")
    if config["phase"] == "pilot" and config["method"] != "equal":
        raise ValueError("LR pilots use the common equal-teacher reference")
    if type(config.get("seed")) is not int or config["seed"] != 42:
        raise ValueError("This confirmed teacher-weight study uses seed42")
    if single._student_options(config)["mode"] != "full_finetune":
        raise ValueError("The teacher-weight comparison fully trains the same student")
    records._same("fixed three-term KD", config.get("distillation"), LOSS_SETTINGS)
    train = config.get("training", {})
    for key, expected in (("weight_decay", .01), ("drop_path_rate", 0.)):
        value = train.get(key, 0. if key == "drop_path_rate" else None)
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            raise ValueError(f"This comparison requires training.{key}={expected}")
    if train.get("amp") is not False:
        raise ValueError("The teacher-weight comparison uses FP32")
    if config["student_initialization"] == "random" and config.get("student_checkpoint") is not None:
        raise ValueError("Random initialization must not load an ImageNet student file")
    sources = config.get("teachers")
    if (not isinstance(sources, list) or len(sources) != 3
            or [source.get("model") for source in sources] != list(TEACHER_NAMES)):
        raise ValueError("Declare the three fixed RAD-DINO/CheXFound/CheXzero sources in order")
    policy = config.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("policy must be a mapping")
    warmup = policy.get("warmup_epochs", 1)
    if type(warmup) is not int or warmup != 1:
        raise ValueError("All methods share one normalized equal-weight warm-up epoch")
    rate = policy.get("lr", 1e-4)
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("policy.lr must be positive and finite")


def _prepare(config, project):
    config = deepcopy(config)
    _validate_config(config)
    config.setdefault("device", "cuda:0")
    config.setdefault("output_root", "runs/teacher_weight_comparison")
    for key in ("output_root", "student_checkpoint", "cache_dir"):
        if config.get(key) is not None:
            config[key] = str(single._path(config[key], project))
    if config["student_initialization"] == "imagenet":
        if not config.get("student_checkpoint") or not Path(config["student_checkpoint"]).is_file():
            raise FileNotFoundError("ImageNet initialization requires the declared local student checkpoint")
    supervised._device(config)
    loaded = [load_teacher(source, project, cache_dir=config.get("cache_dir")) for source in config["teachers"]]
    split = validate_teacher_cohorts(loaded)
    teachers = {name: item[0] for name, item in zip(TEACHER_NAMES, loaded)}
    infos = [deepcopy(item[4]) for item in loaded]
    for info, teacher in zip(infos, teachers.values()):
        info["feature_dim"] = teacher.adapter.feature_dim
    teacher_config = loaded[0][1]
    config["teacher_checkpoint"] = infos[0]["checkpoint"]
    config["teacher_config"] = deepcopy(teacher_config)
    config["training"] = single._training_settings(teacher_config, config)
    config["evaluation"] = deepcopy(teacher_config["evaluation"])
    anchor = {key: deepcopy(infos[0][key]) for key in (
        "checkpoint", "checkpoint_sha256", "model_metadata", "cohort_hash", "best_val_patient_roc_auc")}
    anchor["selection"] = "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC"
    return config, teachers, split, infos, anchor


def _head_protocol(seed):
    return {"seed": seed ^ HEAD_SEED_XOR, "distribution": "truncated_normal_std_0.02_zero_bias"}


def _build_student(config):
    if config["student_initialization"] == "random":
        student = build_vit_tiny_student(random_seed=config["seed"])
    else:
        student = build_vit_tiny_student(config["student_checkpoint"])
    # The ImageNet encoder and random encoder receive identical task-head
    # tensors. A private CPU stream cannot perturb sampling or augmentation.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"] ^ HEAD_SEED_XOR)
        nn.init.trunc_normal_(student.encoder.head.weight, std=.02)
        if student.encoder.head.bias is not None:
            nn.init.zeros_(student.encoder.head.bias)
    return student


def _build_loss(student, teachers, config):
    # All strategies and both initialization modes start with equal projectors.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"] ^ PROJECTOR_SEED_XOR)
        return WeightedTeacherDistillationLoss(
            {name: teacher.adapter.feature_dim for name, teacher in teachers.items()},
            student_dim=student.feature_dim, method=config["method"], seed=config["seed"],
            **config["distillation"])


def _projector_hash(kd):
    return single._state_hash(kd.projector_state_dict())


def _initial_proof(student, kd, config):
    return {"initial_state_sha256": single._state_hash(student.state_dict()),
        "head_initial_state_sha256": single._state_hash(student.encoder.head.state_dict()),
        "distillation_initial_state_sha256": _projector_hash(kd),
        "head_initialization": _head_protocol(config["seed"])}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return deepcopy(value)


def _stochastic(kd):
    return _cpu_tree(kd.stochastic_state_dict()) if hasattr(kd, "stochastic_state_dict") else None


def _payload(student, kd, config, proof, anchor, geometry, infos, views, cohort_hash, files=None):
    value = single._student_payload(student, config, cohort_hash, CONDITION,
        proof["initial_state_sha256"], anchor, geometry)
    value.update(experiment=EXPERIMENT, method=config["method"], phase=config["phase"],
        student_initialization=config["student_initialization"], teachers=deepcopy(infos),
        teacher_views=deepcopy(views), distillation_state=single._cpu_state(kd),
        distillation_stochastic_state=_stochastic(kd),
        weighting_metadata=kd.weighting_metadata(), **deepcopy(proof))
    if files is not None:
        value["data_files"] = deepcopy(files)
    return value


def _batch_loss(student, teachers, kd, batch, criterion, device, epoch, total_epochs):
    equal._assert_frozen(teachers)
    features = student(batch["image"].to(device), return_features=True)
    references = {}
    with torch.no_grad():
        for name in TEACHER_NAMES:
            references[name] = teachers[name](batch["teacher_images"][name].to(device), return_features=True)
    labels = batch["label"].to(device)
    bce = F.binary_cross_entropy_with_logits(features["logits"], labels, reduction="none")
    hard_losses = criterion.scale * (1 - torch.exp(-bce)).pow(criterion.gamma) * bce
    terms = kd(features["logits"], references, labels,
        student_global=features["global_features"], student_patches=features["local_features"],
        student_grid=features["grid_shape"], epoch=epoch, total_epochs=total_epochs,
        hard_losses=hard_losses.detach())
    hard = hard_losses.mean()
    values = {name: terms[name] for name in ("soft", "global", "local")}
    values.update(hard=hard, kd=terms["total"], total=hard + terms["total"])
    if any(not torch.isfinite(value).all().item() for value in values.values()):
        raise RuntimeError("Non-finite teacher-weight loss; no update was applied")
    values.update(policy_loss=terms.get("policy_loss"), weights=terms["weights"],
        weight_entropy=terms.get("weight_entropy"), reward=terms.get("reward"),
        policy_entropy=terms.get("policy_entropy"), per_teacher=terms["per_teacher"])
    return values


def _optimizer(student, kd, train):
    class Projectors:
        def parameters(self):
            return kd.projector_parameters()
    return single._optimizer(student, Projectors(), train)


def _policy_optimizer(kd, config):
    parameters = list(kd.policy_parameters())
    return torch.optim.Adam(parameters, lr=config.get("policy", {}).get("lr", 1e-4)) if parameters else None


def _apply_gradients(optimizer, parameters, factor, clip):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("Missing or non-finite teacher-weight gradients")
    for gradient in gradients:
        gradient.mul_(factor)
    nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _train_epoch(student, teachers, kd, loader, optimizer, policy_optimizer, criterion, device, train, epoch):
    student.train()
    kd.train()
    equal._assert_frozen(teachers)
    optimizer.zero_grad(set_to_none=True)
    if policy_optimizer is not None:
        policy_optimizer.zero_grad(set_to_none=True)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    actor_parameters = list(kd.policy_parameters())
    totals, samples, window, denominator, steps, actor_steps = {}, 0, 0, None, 0, 0
    diagnostic_counts = {key: 0 for key in ("policy_loss", "reward", "policy_entropy", "weight_entropy")}
    weight_sums = torch.zeros(len(TEACHER_NAMES), dtype=torch.float64)
    per_teacher = {name: {key: 0. for key in ("total", "soft", "global", "local")} for name in TEACHER_NAMES}
    for index, batch in enumerate(loader):
        values = _batch_loss(student, teachers, kd, batch, criterion, device, epoch, train["epochs"])
        count = len(batch["label"])
        if not window:
            denominator = count * train["accumulation_steps"]
        (values["total"] * (count / denominator)).backward()
        if values["policy_loss"] is not None:
            if policy_optimizer is None:
                raise RuntimeError("A policy loss requires a separate policy optimizer")
            (values["policy_loss"] * (count / denominator)).backward()
        window += count
        samples += count
        for key in ("total", "hard", "kd", "soft", "global", "local"):
            totals[key] = totals.get(key, 0.) + values[key].detach().item() * count
        for key in diagnostic_counts:
            if values.get(key) is not None:
                totals[key] = totals.get(key, 0.) + values[key].detach().item() * count
                diagnostic_counts[key] += count
        weights = values["weights"].detach().cpu().double()
        if tuple(weights.shape) != (count, len(TEACHER_NAMES)) or not torch.isfinite(weights).all():
            raise RuntimeError("Teacher weights must be finite [batch,3]")
        weight_sums += weights.sum(0)
        for name in TEACHER_NAMES:
            for key in per_teacher[name]:
                per_teacher[name][key] += values["per_teacher"][name][key].detach().item() * count
        if (index + 1) % train["accumulation_steps"] == 0 or index + 1 == len(loader):
            _apply_gradients(optimizer, parameters, denominator / window, train["gradient_clip"])
            if policy_optimizer is not None and any(p.grad is not None for p in actor_parameters):
                _apply_gradients(policy_optimizer, actor_parameters, denominator / window, train["gradient_clip"])
                actor_steps += 1
            steps += 1
            window = 0
        if (index + 1) % 50 == 0 or index + 1 == len(loader):
            print(f"[train] batch={index + 1}/{len(loader)} optimizer_steps={steps}", flush=True)
    if not steps or not samples:
        raise RuntimeError("No student optimizer update completed")
    equal._assert_frozen(teachers)
    result = {key: totals[key] / samples for key in ("total", "hard", "kd", "soft", "global", "local")}
    result.update({key: totals[key] / count if count else None for key, count in diagnostic_counts.items()})
    result.update(optimizer_steps=steps, policy_optimizer_steps=actor_steps,
        teacher_weights={name: weight_sums[i].item() / samples for i, name in enumerate(TEACHER_NAMES)},
        teacher_losses={name: {key: value / samples for key, value in terms.items()} for name, terms in per_teacher.items()})
    return result


def check(config, project):
    config, teachers, split, infos, anchor = _prepare(config, project)
    device = supervised._device(config)
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    kd = _build_loss(student, teachers, config).to(device)
    proof = _initial_proof(student, kd, config)
    before = {name: single._state_hash(teacher.state_dict()) for name, teacher in teachers.items()}
    for teacher in teachers.values():
        teacher.to(device).eval()
    supervised.set_seed(config["seed"])
    loader = equal._loader(split, "train", teachers, student, config, torch.Generator().manual_seed(config["seed"]))
    batch = next(iter(loader))
    criterion = supervised.FocalLoss(config["training"]["focal_gamma"], config["training"]["focal_scale"])
    # Active epoch exercises learned/sample-conditioned weights in preflight;
    # actual training retains the common equal-weight first epoch.
    values = _batch_loss(student, teachers, kd, batch, criterion, device, 2, config["training"]["epochs"])
    values["total"].backward()
    backbone = [p.grad for name, p in student.named_parameters() if "encoder.head." not in name and p.requires_grad]
    if not any(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0 for g in backbone):
        raise RuntimeError("Student backbone received no finite nonzero preflight gradients")
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in student.parameters()):
        raise RuntimeError("Student preflight produced non-finite gradients")
    for parameter in kd.projector_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item() or not parameter.grad.abs().sum().item() > 0:
            raise RuntimeError("Each teacher global projector must receive finite nonzero gradients")
    actor_verified = False
    if config["method"] == "rl":
        if values["policy_loss"] is None:
            raise RuntimeError("RL preflight did not exercise its active policy")
        values["policy_loss"].backward()
        gradients = [p.grad for p in kd.policy_parameters()]
        if not any(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0 for g in gradients):
            raise RuntimeError("RL policy received no finite nonzero preflight gradients")
        if any(g is not None and not torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("RL policy preflight produced non-finite gradients")
        actor_verified = True
    equal._assert_frozen(teachers)
    for name, teacher in teachers.items():
        if before[name] != single._state_hash(teacher.state_dict()):
            raise RuntimeError("Frozen teacher changed in preflight")
    student.eval()
    with torch.no_grad():
        expected = student(batch["image"].to(device)).detach().cpu()
    directory = single._new_suite(config, "check")
    snapshot = directory / "roundtrip.pt"
    payload = _payload(student, kd, config, proof, anchor, loader.dataset.metadata(), infos,
        loader.dataset.teacher_view_metadata(), supervised._split_hash(split))
    supervised._torch_save(snapshot, payload)
    restored, _ = single.load_student_checkpoint(snapshot, device)
    roundtrip = equal._verify_roundtrip_state(student, restored)
    with torch.no_grad():
        actual = restored(batch["image"].to(device)).detach().cpu()
    roundtrip.update(equal._verify_roundtrip_outputs(expected, actual, len(batch["label"])))
    restored_kd = _build_loss(student, teachers, config).to(device)
    _strict_load(restored_kd, payload["distillation_state"], "All projectors and policy")
    if payload["distillation_stochastic_state"] is not None:
        restored_kd.load_stochastic_state_dict(payload["distillation_stochastic_state"])
    snapshot.unlink()
    result = {"check_passed": True, "training_started": False, "condition": CONDITION,
        "experiment": EXPERIMENT, "student_mode": "full_finetune", "seed": config["seed"],
        "student_initialization": config["student_initialization"], "method": config["method"], "phase": config["phase"],
        "teacher_unchanged": True, "student_backbone_gradient_verified": True,
        "projector_gradients_verified": True, "policy_gradient_verified": actor_verified,
        "config": config, "effective_training": config["training"], "model_metadata": student.training_metadata(),
        "cohort_hash": supervised._split_hash(split), "teacher": anchor, "teachers": infos,
        "shared_view": loader.dataset.metadata(), "teacher_views": loader.dataset.teacher_view_metadata(),
        "losses": {key: values[key].detach().item() for key in ("total", "hard", "kd", "soft", "global", "local")},
        "weights": values["weights"].detach().cpu().tolist(),
        "weighting_metadata": kd.weighting_metadata(), **proof, **roundtrip}
    supervised._json(directory / "check.json", result)
    print(f"Check passed: {config['student_initialization']} / {config['method']} / {config['phase']}", flush=True)
    return directory


def _data_files(run):
    return {name: records._file_hash(Path(run) / name) for name in ("split.csv", "split_metadata.json")}


def _history_record(epoch, started, losses, training_metrics, train_eval_loss, metrics, val_loss, optimizer):
    return {"epoch": epoch, "epoch_seconds": time.monotonic() - started,
        "train_loss": losses["total"], "train_hard_loss": losses["hard"], "train_kd_loss": losses["kd"],
        "train_soft_loss": losses["soft"], "train_global_loss": losses["global"], "train_local_loss": losses["local"],
        "train_eval_loss": train_eval_loss, "train_eval_patient_roc_auc": training_metrics["roc_auc"],
        "train_eval_patient_pr_auc": training_metrics["pr_auc"], "val_loss": val_loss,
        "val_patient_roc_auc": metrics["roc_auc"], "val_patient_pr_auc": metrics["pr_auc"],
        "teacher_weights": losses["teacher_weights"], "teacher_losses": losses["teacher_losses"],
        "weight_entropy": losses["weight_entropy"], "policy_loss": losses["policy_loss"],
        "reward": losses["reward"], "policy_entropy": losses["policy_entropy"],
        "optimizer_steps": losses["optimizer_steps"], "policy_optimizer_steps": losses["policy_optimizer_steps"],
        "learning_rates": [group["lr"] for group in optimizer.param_groups]}


def _run_training(config, teachers, split, infos, anchor):
    run = single._new_suite(config) / CONDITION
    run.mkdir()
    device, train = supervised._device(config), config["training"]
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    kd = _build_loss(student, teachers, config).to(device)
    proof = _initial_proof(student, kd, config)
    before = {name: single._state_hash(teacher.state_dict()) for name, teacher in teachers.items()}
    for teacher in teachers.values():
        teacher.to(device).eval()
    equal._assert_frozen(teachers)
    optimizer = _optimizer(student, kd, train)
    policy_optimizer = _policy_optimizer(kd, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"], eta_min=train["minimum_lr"])
    criterion = supervised.FocalLoss(train["focal_gamma"], train["focal_scale"])
    supervised.set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = equal._loader(split, "train", teachers, student, config, generator)
    val_loader = single._loader(split, "val", teachers["rad_dino"].adapter.preprocess, student.preprocess, config)
    geometry, views = loader.dataset.metadata(), loader.dataset.teacher_view_metadata()
    train_eval_loader = random_student._train_eval_loader(split, geometry, config)
    cohort_hash = supervised._split_hash(split)
    for name, value in (("config", config), ("model_metadata", student.training_metadata()),
            ("teacher", anchor), ("teachers", infos), ("shared_view", geometry),
            ("teacher_views", views), ("split_metadata", split.attrs["split_metadata"]),
            ("weighting_metadata", kd.weighting_metadata())):
        supervised._json(run / f"{name}.json", value)
    split.to_csv(run / "split.csv", index=False)
    files = _data_files(run)
    supervised._json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "FP32", "cohort_hash": cohort_hash})
    print(f"Run directory: {run}", flush=True)
    history, best_auc, bad_epochs, best_model, best_kd, best_rng = [], -math.inf, 0, None, None, None
    for epoch in range(1, train["epochs"] + 1):
        started = time.monotonic()
        print(f"[epoch] {epoch}/{train['epochs']} started; {config['student_initialization']} / {config['method']} / {config['phase']}", flush=True)
        losses = _train_epoch(student, teachers, kd, loader, optimizer, policy_optimizer, criterion, device, train, epoch)
        train_predictions, train_eval_loss = supervised.predict(student, train_eval_loader, device, criterion=criterion)
        train_patients = aggregate_patient_predictions(train_predictions)
        training_metrics = binary_metrics(train_patients.label, train_patients.probability)
        validation, val_loss = supervised.predict(student, val_loader, device, criterion=criterion)
        patients = aggregate_patient_predictions(validation)
        metrics = binary_metrics(patients.label, patients.probability)
        val_auc = metrics["roc_auc"]
        if val_auc is None or not math.isfinite(val_auc):
            raise ValueError("Validation requires both classes for checkpoint selection")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_model, best_kd, best_rng = student.trainable_state_dict(), single._cpu_state(kd), _stochastic(kd)
        else:
            bad_epochs += 1
        history.append(_history_record(epoch, started, losses, training_metrics, train_eval_loss, metrics, val_loss, optimizer))
        print(json.dumps({"method": config["method"], "student_initialization": config["student_initialization"], **history[-1]}), flush=True)
        scheduler.step()
        payload = _payload(student, kd, config, proof, anchor, geometry, infos, views, cohort_hash, files)
        payload.update(epoch=epoch - 1, best_auc=best_auc, bad_epochs=bad_epochs, best_model=best_model,
            best_distillation_state=best_kd, best_distillation_stochastic_state=best_rng,
            optimizer=_cpu_tree(optimizer.state_dict()), scheduler=_cpu_tree(scheduler.state_dict()),
            policy_optimizer=_cpu_tree(policy_optimizer.state_dict()) if policy_optimizer is not None else None,
            rng=supervised._rng_state(generator), history=deepcopy(history))
        supervised._torch_save(run / "last.pt", payload)
        supervised._json(run / "history.json", history)
        if bad_epochs >= train["patience"]:
            break
    if best_model is None:
        raise RuntimeError("No validation-selected student was produced")
    from .training_history import save_history_plots
    save_history_plots([{**row, "train_loss": row["train_eval_loss"]} for row in history], run)
    supervised._json(run / "history_plot_metadata.json", {"train_loss": "train_eval_loss: Focal, eval mode, no random augmentation",
        "val_loss": "val_loss: Focal, eval mode", "optimization_losses": "Preserved separately in history.json"})
    student.load_trainable_state_dict(best_model)
    _strict_load(kd, best_kd, "Validation-selected KD and policy")
    if best_rng is not None:
        kd.load_stochastic_state_dict(best_rng)
    validation = supervised.predict(student, val_loader, device)
    patients = aggregate_patient_predictions(validation)
    threshold = select_threshold(patients.label, patients.probability)
    best_record = max(history, key=lambda row: row["val_patient_roc_auc"])
    payload = _payload(student, kd, config, proof, anchor, geometry, infos, views, cohort_hash, files)
    payload.update(threshold=threshold, best_val_patient_roc_auc=best_auc, best_epoch=best_record["epoch"])
    supervised._torch_save(run / "best.pt", payload)
    restored, _ = single.load_student_checkpoint(run / "best.pt", device)
    equal._verify_roundtrip_state(student, restored)
    validation.to_csv(run / "validation_predictions.csv", index=False)
    supervised._json(run / "selection.json", {"threshold": threshold, "source": "validation patient predictions",
        "best_epoch": best_record["epoch"], "best_val_patient_roc_auc": best_auc})
    equal._assert_frozen(teachers)
    for name, teacher in teachers.items():
        if before[name] != single._state_hash(teacher.state_dict()):
            raise RuntimeError("Teacher changed during student training")
        teacher.to("cpu")
    receipt = {"status": "completed", "experiment": EXPERIMENT, "condition": CONDITION,
        "phase": config["phase"], "method": config["method"], "student_initialization": config["student_initialization"],
        "seed": config["seed"], "student_mode": "full_finetune", "run": str(run),
        "best_val_patient_roc_auc": best_auc, "val_patient_pr_auc": best_record["val_patient_pr_auc"],
        "best_epoch": best_record["epoch"], "threshold": threshold, "cohort_hash": cohort_hash,
        "teacher_unchanged": True, "teachers": infos, "teacher_views": views, "data_files": files,
        "checkpoint_sha256": records._file_hash(run / "best.pt"),
        "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "distillation_trainable_parameters": sum(p.numel() for p in kd.projector_parameters()),
        "policy_trainable_parameters": sum(p.numel() for p in kd.policy_parameters()), **proof}
    receipt["weighting_metadata"] = kd.weighting_metadata()
    # A pilot is a completed train/validation experiment. It never constructs a
    # test loader, reads test pixels, or stores test metrics.
    supervised._json(run / "result.json", receipt)
    if config["phase"] == "main":
        return evaluate_selected(run, config=config)
    print("[pilot completed] Train/validation only; no test evaluation", flush=True)
    return receipt


def run_training(config, project):
    return _run_training(*_prepare(config, project))


def _protocol(saved, current, *, promotion=False):
    records._validate_protocol(saved, current, CONDITION)
    for name in ("student_initialization", "method", "seed"):
        records._same("teacher-weight " + name, saved.get(name), current.get(name))
    records._same("teacher-weight KD", saved.get("distillation"), current.get("distillation"))
    records._same("teacher-weight policy", saved.get("policy", {}), current.get("policy", {}))
    if promotion:
        if saved.get("phase") != "pilot" or current.get("phase") != "main" or saved.get("method") != "equal":
            raise ValueError("Only a selected equal pilot can be promoted to main evaluation")
    else:
        records._same("teacher-weight phase", saved.get("phase"), current.get("phase"))


def _read_run(run, *, checkpoint_loader=None):
    run = Path(run).expanduser().resolve()
    payload = (checkpoint_loader or records._default_checkpoint_loader)(run / "best.pt")
    config = records._read(run / "config.json")
    _validate_config(config)
    records._same("worker experiment", payload.get("experiment"), EXPERIMENT)
    _protocol(payload["config"], config)
    files = _data_files(run)
    records._same("frozen run split files", payload.get("data_files"), files)
    for key in ("model_metadata", "teachers", "teacher_views", "shared_view", "teacher", "weighting_metadata"):
        records._same("saved " + key, records._read(run / (key + ".json")), payload.get(key))
    history = records._read(run / "history.json")
    if not isinstance(history, list) or not history:
        raise ValueError("Completed worker requires epoch history")
    for row in history:
        for key in ("train_eval_loss", "val_loss", "train_eval_patient_roc_auc", "val_patient_roc_auc"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing or invalid comparable history field " + key)
    best = max(history, key=lambda row: row["val_patient_roc_auc"])
    records._same("saved validation best", payload.get("best_val_patient_roc_auc"), best["val_patient_roc_auc"])
    records._same("saved best epoch", payload.get("best_epoch"), best["epoch"])
    return run, config, payload, history, files


def _verify_evaluation_artifacts(run, result, payload):
    records._same("completed evaluation checkpoint", result.get("checkpoint_sha256"), records._file_hash(run / "best.pt"))
    metrics = records._read(run / "test/metrics.json")
    records._same("evaluation patient metrics", metrics.get("patient"), result.get("test_patient"))
    records._same("evaluation interval", metrics.get("bootstrap", {}).get("patient", {}).get("roc_auc"), result.get("roc_auc_ci"))
    records._same("evaluation threshold", result.get("threshold"), payload.get("threshold"))
    for name in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
        if not (run / name).is_file() or not (run / name).stat().st_size:
            raise ValueError("Incomplete evaluation artifact: " + name)


def evaluate_selected(run, config=None, project=None):
    run, saved, payload, history, files = _read_run(run)
    requested = deepcopy(saved) if config is None else deepcopy(config)
    requested["phase"] = "main"
    if requested.get("selected_checkpoint_sha256") is not None:
        records._same("selected pilot checkpoint fingerprint",
            records._file_hash(run / "best.pt"), requested["selected_checkpoint_sha256"])
    # The selected pilot's effective recipe is frozen, even when its job config
    # only lists overrides. Full checks against a main preflight occur in verify.
    if config is not None:
        for key, value in config.get("training", {}).items():
            records._same("selected evaluation training." + key, saved["training"].get(key), value)
        for key in ("student_initialization", "method", "seed", "distillation"):
            records._same("selected evaluation " + key, saved.get(key), config.get(key))
    if saved["phase"] == "pilot" and saved["method"] != "equal":
        raise ValueError("Only equal-teacher pilots can be selected")
    result_path = run / "evaluation_result.json"
    if result_path.is_file():
        result = records._read(result_path)
        records._same("evaluation phase", result.get("phase"), "main")
        _verify_evaluation_artifacts(run, result, payload)
        return result
    device = torch.device(requested.get("device", saved.get("device", "cpu")))
    restored, _ = single.load_student_checkpoint(run / "best.pt", device)
    split = pd.read_csv(run / "split.csv", dtype={"sample_id": str, "patient_id": str, "path": str})
    split.attrs["split_metadata"] = records._read(run / "split_metadata.json")
    records._same("evaluation frozen cohort", supervised._split_hash(split), payload["cohort_hash"])
    evaluation_context = {"checkpoint_sha256": records._file_hash(run / "best.pt"),
        "cohort_hash": payload["cohort_hash"], "threshold": payload["threshold"], "data_files": files}
    context_path = run / "evaluation_context.json"
    if context_path.exists():
        records._same("frozen evaluation context", records._read(context_path), evaluation_context)
    else:
        supervised._json(context_path, evaluation_context)
    complete = all((run / name).is_file() for name in ("test/metrics.json", "test/predictions.csv", "test/patient_predictions.csv"))
    if not complete:
        print("[evaluation] Selected checkpoint reloaded; evaluating test once", flush=True)
        train = saved["training"]
        loader = DataLoader(single.inference_dataset(split.loc[split["split"] == "test"], payload,
            requested.get("cache_dir", saved.get("cache_dir"))), batch_size=train["batch_size"], shuffle=False,
            num_workers=train["num_workers"], pin_memory=device.type == "cuda", worker_init_fn=supervised._seed_worker,
            generator=torch.Generator().manual_seed(saved["seed"] ^ 0x54455354))
        predictions = supervised.predict(restored, loader, device)
        metrics = save_evaluation(predictions, run / "test", threshold=payload["threshold"],
            bootstrap_samples=saved["evaluation"]["bootstrap_samples"], seed=saved["seed"])
    else:
        metrics = records._read(run / "test/metrics.json")
    source = records._read(run / "result.json")
    result = {**source, "phase": "main", "source_phase": saved["phase"],
        "test_patient": metrics["patient"], "roc_auc_ci": metrics["bootstrap"]["patient"].get("roc_auc"),
        "evaluation_context": evaluation_context}
    _verify_evaluation_artifacts(run, result, payload)
    supervised._json(result_path, result)
    return result


def verify_completed_run(run, check, job, *, checkpoint_loader=None):
    run, saved, payload, history, files = _read_run(run, checkpoint_loader=checkpoint_loader)
    current = check["config"]
    phase = job.get("config", {}).get("phase", job.get("phase", current.get("phase")))
    promotion = saved["phase"] == "pilot" and phase == "main"
    _protocol(saved, current, promotion=promotion)
    result = records._read(run / ("evaluation_result.json" if phase == "main" else "result.json"))
    for key, expected in (("status", "completed"), ("experiment", EXPERIMENT), ("condition", CONDITION),
            ("phase", phase), ("seed", saved["seed"]), ("method", saved["method"]),
            ("student_initialization", saved["student_initialization"]), ("teacher_unchanged", True)):
        records._same("worker result " + key, result.get(key), expected)
    for key in ("initial_state_sha256", "head_initial_state_sha256", "distillation_initial_state_sha256", "head_initialization"):
        records._same("preflight " + key, check.get(key), payload.get(key))
        records._same("result " + key, result.get(key), payload.get(key))
    for key in ("model_metadata", "teacher", "teachers", "teacher_views", "shared_view", "cohort_hash"):
        records._same("preflight " + key, check.get(key), payload.get(key))
    for key in ("method", "student_initialization", "phase"):
        records._same("checkpoint " + key, payload.get(key), saved.get(key))
    records._same("result frozen cohort", result.get("cohort_hash"), payload["cohort_hash"])
    records._same("result data files", result.get("data_files"), files)
    records._same("result teacher metadata", result.get("teachers"), payload["teachers"])
    records._same("result teacher views", result.get("teacher_views"), payload["teacher_views"])
    records._same("result checkpoint fingerprint", result.get("checkpoint_sha256"), records._file_hash(run / "best.pt"))
    records._same("result validation selection", result.get("best_val_patient_roc_auc"), payload["best_val_patient_roc_auc"])
    expected = WeightedTeacherDistillationLoss(
        {info["model"]: info["feature_dim"] for info in payload["teachers"]},
        student_dim=payload["model_metadata"]["feature_dim"], method=saved["method"], seed=saved["seed"], **saved["distillation"])
    _strict_load(expected, payload.get("distillation_state"), "Complete teacher-weight projectors and policy")
    records._same("exact teacher-weight algorithm", payload.get("weighting_metadata"), expected.weighting_metadata())
    records._same("preflight teacher-weight algorithm", check.get("weighting_metadata"), expected.weighting_metadata())
    records._same("result teacher-weight algorithm", result.get("weighting_metadata"), expected.weighting_metadata())
    if payload.get("distillation_stochastic_state") is not None:
        expected.load_stochastic_state_dict(payload["distillation_stochastic_state"])
    if phase == "main":
        if current.get("selected_checkpoint_sha256") is not None:
            records._same("selected equal checkpoint", result.get("checkpoint_sha256"), current["selected_checkpoint_sha256"])
        _verify_evaluation_artifacts(run, result, payload)
    else:
        if "test_patient" in result or "roc_auc_ci" in result:
            raise ValueError("Pilot receipt must not contain test metrics")
    return deepcopy(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config")
    parser.add_argument("--action", choices=("check", "train", "evaluate"), required=True)
    parser.add_argument("--run", help="Selected completed pilot/main run for evaluation")
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    config = records._read(single._path(args.config, project)) if args.config else None
    if args.action == "evaluate":
        if not args.run:
            parser.error("--action evaluate requires --run")
        receipt = evaluate_selected(single._path(args.run, project), config=config, project=project)
    else:
        if config is None:
            parser.error("check/train require --config")
        if args.action == "check":
            directory = check(config, project)
            receipt = records._read(directory / "check.json")
            receipt.update(action="check", directory=str(directory))
        else:
            receipt = run_training(config, project)
    if args.result_json:
        supervised._json(single._path(args.result_json, project), receipt)
    return receipt


if __name__ == "__main__":
    main()
