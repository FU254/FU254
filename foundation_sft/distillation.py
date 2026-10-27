"""Paired supervised / soft + global + local single-teacher experiments.

Reuse the existing frozen cohort, image cache, Focal loss and patient evaluation.
This module never changes the original supervised trainer or its checkpoints.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import train as supervised
from .data import CXRDataset
from .distillation_data import SharedViewDataset
from .distillation_losses import SingleTeacherDistillationLoss
from .metrics import (aggregate_patient_predictions, binary_metrics,
                      save_evaluation, select_threshold)
from .models import build_adapter
from .models.common import PreprocessSpec, sha256_file
from .models.vit_tiny_student import build_vit_tiny_student


FORMAT = "single_teacher_kd_v1"
CONDITIONS = ("supervised", "distillation")
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "distillation_rad_dino_vit_tiny.json"


def _path(value, project):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(project) / path).resolve()


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _verify_roundtrip_state(student, restored):
    """Check exact serialization before the restored model runs inference."""
    original_hash = _state_hash(student.state_dict())
    restored_hash = _state_hash(restored.state_dict())
    if original_hash != restored_hash:
        raise ValueError("Complete student state changed during save/reload")
    return {"student_state_exact": True, "student_state_sha256": original_hash,
            "restored_state_sha256": restored_hash}


def _verify_roundtrip_outputs(expected, actual, batch_size):
    """Allow bounded FP32 inference roundoff after an exact state comparison."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Roundtrip batch size must be a positive integer")
    for name, value in (("expected", expected), ("actual", actual)):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or tuple(value.shape) != (batch_size,)):
            raise ValueError(f"Roundtrip {name} logits must have shape [batch_size]")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Roundtrip {name} logits contain non-finite values")
    # The weights are checked separately and exactly. CUDA FP32 inference may
    # vary slightly near zero even with identical restored tensors.
    rtol, atol = 1e-4, 1e-5
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {"roundtrip_max_abs_error": (actual - expected).abs().max().item(),
            "roundtrip_rtol": rtol, "roundtrip_atol": atol}


def resolve_teacher_checkpoint(config, project):
    if config.get("teacher_checkpoint"):
        checkpoint = _path(config["teacher_checkpoint"], project)
    else:
        suite = _path(config["teacher_suite"], project)
        log = suite / "logs" / "rad_dino_lora_seed42_train.log"
        if not log.is_file():
            raise FileNotFoundError(
                f"Teacher training log not found: {log}. "
                "Use --teacher-checkpoint /actual/RAD-DINO/run/best.pt")
        directories = {match.group(1).strip() for line in log.read_text(encoding="utf-8").splitlines()
                       if (match := re.fullmatch(r"Run directory:\s*(.+)", line.strip()))}
        if len(directories) != 1:
            raise ValueError("Teacher log must identify exactly one training run; use --teacher-checkpoint")
        checkpoint = _path(directories.pop(), project) / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Teacher best checkpoint not found: {checkpoint}")
    return checkpoint


def load_teacher(path, device="cpu", cache_dir=None):
    """Restore LoRA + original classifier before freezing metadata-sensitive parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = deepcopy(payload.get("config", {}))
    if (payload.get("format_version") != 1 or config.get("model") != "rad_dino"
            or config.get("mode") != "lora"
            or "best_val_patient_roc_auc" not in payload or "threshold" not in payload):
        raise ValueError("Teacher must be a validation-selected RAD-DINO LoRA best.pt")
    if not Path(config["split_manifest"]).is_file():
        raise FileNotFoundError("The teacher's frozen split manifest is missing; do not create a new split")
    config["device"] = str(device)
    if cache_dir is not None:
        config["cache_dir"] = str(cache_dir)
    split = supervised.prepare(config)
    teacher = supervised._classifier(build_adapter(config), config)
    # The legacy classifier loader checks names but may cast dtype silently.
    # Reject damaged task adapters before spending time on either student run.
    expected = teacher.trainable_state_dict()
    state = payload.get("model")
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Teacher LoRA/classification parameter names differ")
    for name, target in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid teacher task tensor: {name}")
    supervised._load_checkpoint(path, teacher, config, supervised._split_hash(split))
    if any(not torch.isfinite(value).all() for value in teacher.state_dict().values()
           if value.is_floating_point()):
        raise ValueError("Restored teacher contains non-finite weights")
    # _load_checkpoint compares trainable counts and LoRA parameter names.
    # Freezing any earlier would invalidate or incompletely restore the teacher.
    teacher.requires_grad_(False).eval()
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise ValueError("Restored teacher unexpectedly contains gradients")
    return teacher.to(device), config, split, payload


def _training_settings(teacher_config, config):
    settings = deepcopy(teacher_config["training"])
    settings.update({"backbone_lr": 5e-5, "projection_lr": 1e-4})
    settings.update(config.get("training", {}))
    lora = _student_options(config)["mode"] == "lora"
    if lora:
        # The teacher's optimizer LR belongs to a different experiment. Keep
        # the student update LR controlled against the full-finetune backbone.
        settings["lora_lr"] = config.get("training", {}).get("lora_lr", settings["backbone_lr"])
    if settings.get("amp") is not False:
        raise ValueError("This first paired KD protocol requires training.amp=false (FP32)")
    for key in ("epochs", "patience", "batch_size", "accumulation_steps"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("training.num_workers must be nonnegative")
    rates = ("lora_lr" if lora else "backbone_lr", "head_lr", "projection_lr")
    for key in (*rates, "gradient_clip", "focal_scale"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    for key in ("weight_decay", "minimum_lr", "focal_gamma", "rotation_degrees"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be nonnegative and finite")
    if settings["minimum_lr"] > min(settings[key] for key in rates):
        raise ValueError("minimum_lr must not exceed any initial learning rate")
    return settings


def _student_options(config):
    """Validate the update mode before loading any large weights or patient data."""
    settings = config.get("student", {})
    if not isinstance(settings, dict) or set(settings) - {"mode", "rank", "alpha", "train_last_blocks"}:
        raise ValueError("student contains an unsupported update option")
    mode = settings.get("mode", "full_finetune")
    if mode == "partial_finetune":
        if set(settings) - {"mode", "train_last_blocks"}:
            raise ValueError("A partial student accepts only mode and train_last_blocks")
        count = settings.get("train_last_blocks")
        if type(count) is not int or count not in (2, 4, 8):
            raise ValueError("student.train_last_blocks must be 2, 4 or 8")
        return {"mode": mode, "train_last_blocks": count}
    if mode not in ("full_finetune", "lora"):
        raise ValueError("student.mode must be full_finetune, lora or partial_finetune")
    if "train_last_blocks" in settings:
        raise ValueError("train_last_blocks requires student.mode=partial_finetune")
    rank, alpha = settings.get("rank", 8), settings.get("alpha", 16.0)
    if type(rank) is not int or rank <= 0:
        raise ValueError("student.rank must be a positive integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("student.alpha must be positive and finite")
    return {"mode": mode, "rank": rank, "alpha": float(alpha)} if mode == "lora" else {"mode": mode}


def _build_student(config):
    options = _student_options(config)
    if options["mode"] == "full_finetune":
        # Preserve the original factory call and original full metadata schema.
        return build_vit_tiny_student(config["student_checkpoint"])
    return build_vit_tiny_student(config["student_checkpoint"], **options)


def _loader(split, name, teacher_spec, student_spec, config, generator=None, include_teacher=False):
    train = config["training"]
    base = CXRDataset(
        split.loc[split["split"] == name], teacher_spec, training=name == "train",
        seed=config["seed"], horizontal_flip_probability=train["horizontal_flip_probability"],
        rotation_degrees=train["rotation_degrees"], cache_dir=config.get("cache_dir"))
    paired = SharedViewDataset(base, student_size=student_spec.image_size,
                               include_teacher=include_teacher,
                               student_mean=student_spec.mean, student_std=student_spec.std)
    return DataLoader(paired, batch_size=train["batch_size"], shuffle=name == "train",
                      num_workers=train["num_workers"], drop_last=False,
                      pin_memory=str(config["device"]).startswith("cuda"),
                      worker_init_fn=supervised._seed_worker, generator=generator)


def _optimizer(student, kd, train):
    head_ids = {id(parameter) for parameter in student.encoder.head.parameters()}
    head, backbone = [], []
    lora = getattr(student, "training_mode", "full_finetune") == "lora"
    for name, parameter in student.named_parameters():
        if parameter.requires_grad:
            if lora and id(parameter) not in head_ids and not name.endswith((".lora_A", ".lora_B")):
                raise ValueError(f"LoRA student unexpectedly trains an original encoder tensor: {name}")
            (head if id(parameter) in head_ids else backbone).append(parameter)
    if not head or not backbone:
        raise ValueError("Student must train both the encoder and classification head")
    groups = [{"params": backbone, "lr": train["lora_lr" if lora else "backbone_lr"]},
              {"params": head, "lr": train["head_lr"]}]
    if kd is not None:
        projection = [parameter for parameter in kd.parameters() if parameter.requires_grad]
        if projection:
            groups.append({"params": projection, "lr": train["projection_lr"]})
    return torch.optim.AdamW(groups, weight_decay=train["weight_decay"])


def _batch_loss(student, teacher, kd, batch, hard_loss, device):
    images, labels = batch["image"].to(device), batch["label"].to(device)
    if kd is None:
        hard = hard_loss(student(images), labels)
        values = {name: hard.new_zeros(()) for name in ("soft", "global", "local", "kd")}
    else:
        features = student(images, return_features=True)
        with torch.no_grad():
            reference = teacher(batch["teacher_image"].to(device), return_features=True)
        terms = kd(features["logits"], reference["logits"],
                   student_global=features["global_features"], teacher_global=reference["global_features"],
                   student_patches=features["local_features"], teacher_patches=reference["local_features"],
                   student_grid=features["grid_shape"], teacher_grid=reference["grid_shape"])
        hard = hard_loss(features["logits"], labels)
        values = {name: terms[name] for name in ("soft", "global", "local")}
        values["kd"] = terms["total"]
    values.update(hard=hard, total=hard + values["kd"])
    if any(not torch.isfinite(value).item() for value in values.values()):
        raise RuntimeError("Non-finite supervised or distillation loss; no update was applied")
    return values


def _train_epoch(student, teacher, kd, loader, optimizer, hard_loss, device, train):
    student.train()
    teacher.eval()
    if kd is not None:
        kd.train()
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("total", "hard", "kd", "soft", "global", "local")}
    total_images, window_images, denominator, steps = 0, 0, None, 0
    for index, batch in enumerate(loader):
        values = _batch_loss(student, teacher, kd, batch, hard_loss, device)
        count = len(batch["label"])
        if not window_images:
            denominator = count * train["accumulation_steps"]
        (values["total"] * (count / denominator)).backward()
        total_images += count
        window_images += count
        for key, value in values.items():
            totals[key] += value.detach().item() * count
        if (index + 1) % train["accumulation_steps"] == 0 or index + 1 == len(loader):
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                raise RuntimeError("Missing or non-finite FP32 training gradients; update aborted")
            for gradient in gradients:
                gradient.mul_(denominator / window_images)
            nn.utils.clip_grad_norm_(parameters, train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_images = 0
            steps += 1
    if not steps:
        raise ValueError("Empty training loader; no optimizer update completed")
    if any(parameter.grad is not None or parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher must remain frozen and gradient-free")
    return {**{key: value / total_images for key, value in totals.items()}, "optimizer_steps": steps}


def _student_payload(student, config, cohort_hash, condition, initial_hash, teacher_info, geometry):
    payload = {"format_version": FORMAT, "model": student.trainable_state_dict(),
            "model_metadata": student.training_metadata(), "config": deepcopy(config),
            "cohort_hash": cohort_hash, "condition": condition,
            "initial_state_sha256": initial_hash, "teacher": teacher_info,
            "shared_view": geometry}
    if "regularization" in payload["model_metadata"]:
        state = student.stochastic_state_dict()
        if state is None:
            raise ValueError("A DropPath student must save its independent random stream")
        payload["student_stochastic_state"] = state
    return payload


def load_student_checkpoint(path, device="cpu"):
    """Restore a complete student without the old teacher or ImageNet weight file."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT:
        raise ValueError("Not a single-teacher ViT-Tiny student checkpoint")
    metadata = payload["model_metadata"]
    mode = metadata.get("mode", "full_finetune")
    regularization = metadata.get("regularization")
    options = {}
    rate = 0.0
    if regularization is not None:
        if not isinstance(regularization, dict):
            raise ValueError("Saved student regularization must be a dictionary")
        rate = regularization.get("drop_path_rate")
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate < 1:
            raise ValueError("Saved student DropPath rate must be finite and in (0,1)")
        options["drop_path_rate"] = rate
    configured = payload.get("config", {}).get("training", {}).get("drop_path_rate", 0.0)
    if type(configured) not in (int, float) or configured != rate:
        raise ValueError("Checkpoint DropPath configuration and model metadata differ")
    if mode == "full_finetune":
        student = build_vit_tiny_student(initialization=metadata["initialization"], **options)
    elif mode == "lora":
        saved = metadata.get("lora", {})
        options.update(_student_options({"student": {"mode": mode, "rank": saved.get("rank"),
                                                     "alpha": saved.get("alpha")}}))
        student = build_vit_tiny_student(initialization=metadata["initialization"], **options)
    elif mode == "partial_finetune":
        saved = metadata.get("partial_finetune", {})
        if not isinstance(saved, dict):
            raise ValueError("Saved partial student scope must be a dictionary")
        partial = _student_options({"student": {"mode": mode,
            "train_last_blocks": saved.get("train_last_blocks")}})
        if _student_options(payload.get("config", {})) != partial:
            raise ValueError("Checkpoint partial update configuration and metadata differ")
        options.update(partial)
        student = build_vit_tiny_student(initialization=metadata["initialization"], **options)
    else:
        raise ValueError("Unsupported saved student update mode")
    if student.training_metadata() != payload["model_metadata"]:
        raise ValueError("Saved student architecture or initialization metadata differs")
    student.load_trainable_state_dict(payload["model"])
    # Restore a saved CUDA stream against the requested device, never the
    # default GPU. CPU inference keeps CUDA bytes opaque until CUDA is used.
    student = student.to(device)
    if regularization is not None:
        student.load_stochastic_state_dict(payload.get("student_stochastic_state"))
    elif "student_stochastic_state" in payload:
        raise ValueError("A zero-DropPath checkpoint must not contain a stochastic-depth stream")
    return student.eval(), payload


def inference_dataset(cohort, payload, cache_dir=None):
    """Reproduce the student's saved field of view without constructing a teacher.

    Using CXRDataset(student.preprocess) alone would omit the first resize/crop.
    Both resize stages and normalization are part of the deployed student.
    """
    if payload.get("format_version") != FORMAT:
        raise ValueError("Expected a saved single-teacher student preprocessing record")
    geometry = payload["shared_view"]
    teacher_spec = PreprocessSpec(**geometry["teacher_preprocessing"])
    student_spec = geometry["student_preprocessing"]
    base = CXRDataset(cohort, teacher_spec, training=False, cache_dir=cache_dir)
    dataset = SharedViewDataset(base, student_size=student_spec["image_size"], include_teacher=False,
                                student_mean=student_spec["mean"], student_std=student_spec["std"])
    if dataset.metadata() != geometry:
        raise ValueError("Saved student shared-view preprocessing differs from this implementation")
    return dataset


def _run_condition(condition, suite, teacher, split, config, teacher_info, expected_initial=None):
    run = suite / condition
    run.mkdir()
    device, train = supervised._device(config), config["training"]
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    initial_hash = _state_hash(student.state_dict())
    if expected_initial is not None and initial_hash != expected_initial:
        raise ValueError("Supervised and distilled students must start from exactly the same state")
    kd = (SingleTeacherDistillationLoss(student_dim=student.feature_dim,
                                      teacher_dim=teacher.adapter.feature_dim,
                                      **config["distillation"]).to(device)
          if condition == "distillation" else None)
    teacher.to(device if kd is not None else "cpu").eval()
    teacher_before = _state_hash(teacher.state_dict())
    optimizer = _optimizer(student, kd, train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train["epochs"], eta_min=train["minimum_lr"])
    hard_loss = supervised.FocalLoss(train["focal_gamma"], train["focal_scale"])
    # Teacher/projector construction cannot consume the experiment's view RNG.
    supervised.set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = _loader(split, "train", teacher.adapter.preprocess, student.preprocess, config,
                           generator, include_teacher=kd is not None)
    val_loader = _loader(split, "val", teacher.adapter.preprocess, student.preprocess, config)
    geometry = train_loader.dataset.metadata()
    cohort_hash = supervised._split_hash(split)
    supervised._json(run / "config.json", config)
    supervised._json(run / "model_metadata.json", student.training_metadata())
    supervised._json(run / "shared_view.json", geometry)
    supervised._json(run / "teacher.json", teacher_info)
    split.to_csv(run / "split.csv", index=False)
    supervised._json(run / "split_metadata.json", split.attrs["split_metadata"])
    supervised._json(run / "environment.json", {
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "FP32", "cohort_hash": cohort_hash})
    print(f"[{condition}] Run directory: {run}", flush=True)
    history, best_auc, bad_epochs = [], -math.inf, 0
    best_model = best_distillation = None
    for epoch in range(train["epochs"]):
        losses = _train_epoch(student, teacher, kd, train_loader, optimizer, hard_loss, device, train)
        validation, val_loss = supervised.predict(student, val_loader, device, criterion=hard_loss)
        patients = aggregate_patient_predictions(validation)
        metrics = binary_metrics(patients.label, patients.probability)
        val_auc = metrics["roc_auc"]
        if val_auc is None:
            raise ValueError("Validation requires both classes for checkpoint selection")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_model = student.trainable_state_dict()
            best_distillation = _cpu_state(kd) if kd is not None else None
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": losses["total"],
                        "train_hard_loss": losses["hard"], "train_kd_loss": losses["kd"],
                        "train_soft_loss": losses["soft"], "train_global_loss": losses["global"],
                        "train_local_loss": losses["local"], "optimizer_steps": losses["optimizer_steps"],
                        "val_loss": val_loss, "val_patient_roc_auc": val_auc,
                        "val_patient_pr_auc": metrics["pr_auc"],
                        "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        print(json.dumps({"condition": condition, **history[-1]}), flush=True)
        scheduler.step()
        payload = _student_payload(student, config, cohort_hash, condition, initial_hash, teacher_info, geometry)
        payload.update(epoch=epoch, best_auc=best_auc, bad_epochs=bad_epochs,
                       best_model=best_model, best_distillation_state=best_distillation,
                       distillation_state=_cpu_state(kd) if kd is not None else None,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng=supervised._rng_state(generator), history=history)
        supervised._torch_save(run / "last.pt", payload)
        supervised._json(run / "history.json", history)
        if bad_epochs >= train["patience"]:
            break
    if best_model is None:
        raise RuntimeError("No validation-selected student was saved")
    from .training_history import save_history_plots
    save_history_plots(history, run)
    student.load_trainable_state_dict(best_model)
    validation = supervised.predict(student, val_loader, device)
    patients = aggregate_patient_predictions(validation)
    threshold = select_threshold(patients.label, patients.probability)
    payload = _student_payload(student, config, cohort_hash, condition, initial_hash, teacher_info, geometry)
    payload.update(threshold=threshold, best_val_patient_roc_auc=best_auc,
                   distillation_state=best_distillation)
    supervised._torch_save(run / "best.pt", payload)
    # Verify and use the saved complete student, rather than a stale in-memory snapshot.
    restored, restored_payload = load_student_checkpoint(run / "best.pt", device)
    del student
    student = restored
    validation.to_csv(run / "validation_predictions.csv", index=False)
    supervised._json(run / "selection.json", {
        "threshold": threshold, "source": "validation patient predictions",
        "best_val_patient_roc_auc": best_auc})
    if teacher_before != _state_hash(teacher.state_dict()):
        raise RuntimeError("Frozen teacher changed during student training")
    test_loader = DataLoader(
        inference_dataset(split.loc[split["split"] == "test"], restored_payload, config.get("cache_dir")),
        batch_size=train["batch_size"], shuffle=False, num_workers=train["num_workers"],
        pin_memory=device.type == "cuda", worker_init_fn=supervised._seed_worker)
    test = supervised.predict(student, test_loader, device)
    result = save_evaluation(test, run / "test", threshold=restored_payload["threshold"],
                             bootstrap_samples=config["evaluation"]["bootstrap_samples"], seed=config["seed"])
    receipt = {"condition": condition, "status": "completed", "run": str(run),
               "student_mode": _student_options(config)["mode"],
               "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
               "distillation_trainable_parameters": sum(p.numel() for p in kd.parameters() if p.requires_grad) if kd is not None else 0,
               "seed": config["seed"], "initial_state_sha256": initial_hash,
               "cohort_hash": cohort_hash, "best_val_patient_roc_auc": best_auc,
               "threshold": threshold, "test_patient": result["patient"],
               "roc_auc_ci": result["bootstrap"]["patient"].get("roc_auc"),
               "teacher_unchanged": True}
    supervised._json(run / "result.json", receipt)
    print(f"[{condition}] completed", flush=True)
    return receipt


def _resolved(config, project):
    config = deepcopy(config)
    _student_options(config)
    if type(config.get("seed", 42)) is not int or not 0 <= config.get("seed", 42) < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    config.setdefault("seed", 42)
    config.setdefault("device", "cuda:0")
    config.setdefault("output_root", "runs/single_teacher_distillation")
    supervised._device(config)
    config["teacher_checkpoint"] = str(resolve_teacher_checkpoint(config, project))
    for key in ("student_checkpoint", "output_root"):
        config[key] = str(_path(config[key], project))
    if config.get("cache_dir") is not None:
        config["cache_dir"] = str(_path(config["cache_dir"], project))
    return config


def _prepare(config, project):
    # Fail during the short check if an older server checkout lacks the plotting API.
    from .training_history import save_history_plots

    config = _resolved(config, project)
    teacher, teacher_config, split, payload = load_teacher(
        config["teacher_checkpoint"], "cpu", config.get("cache_dir"))
    config["training"] = _training_settings(teacher_config, config)
    config["evaluation"] = deepcopy(teacher_config["evaluation"])
    config["teacher_config"] = teacher_config
    if not Path(config["student_checkpoint"]).is_file():
        raise FileNotFoundError(f"Student ImageNet checkpoint is missing: {config['student_checkpoint']}")
    # Validate all loss settings before committing to a long training run.
    SingleTeacherDistillationLoss(teacher_dim=teacher.adapter.feature_dim, **config["distillation"])
    info = {"checkpoint": config["teacher_checkpoint"],
            "checkpoint_sha256": sha256_file(config["teacher_checkpoint"]),
            "model_metadata": payload["model_metadata"], "cohort_hash": payload["cohort_hash"],
            "best_val_patient_roc_auc": payload["best_val_patient_roc_auc"],
            "selection": "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC"}
    return config, teacher, split, info


def _new_suite(config, kind="suite"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite = Path(config["output_root"]) / ("checks" if kind == "check" else "suites") / f"{kind}_{stamp}_{uuid.uuid4().hex[:8]}"
    suite.mkdir(parents=True)
    return suite


def run_experiments(config, project, conditions=None):
    conditions = tuple(CONDITIONS if conditions is None else conditions)
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Select supervised and/or distillation once each")
    config, teacher, split, info = _prepare(config, project)
    suite = _new_suite(config)
    supervised._json(suite / "config.json", config)
    status = {"status": "running", "teacher": info, "conditions": {}}
    supervised._json(suite / "suite.json", status)
    print(f"Suite directory: {suite}", flush=True)
    initial_hash = None
    for condition in conditions:
        try:
            receipt = _run_condition(condition, suite, teacher, split, config, info, initial_hash)
            initial_hash = receipt["initial_state_sha256"]
            status["conditions"][condition] = receipt
        except Exception as error:
            status["conditions"][condition] = {"status": "failed", "error_type": type(error).__name__,
                                                "error": str(error)}
            status["status"] = "partial" if initial_hash is not None else "failed"
            supervised._json(suite / "suite.json", status)
            raise
        supervised._json(suite / "suite.json", status)
    status["status"] = "completed"
    supervised._json(suite / "suite.json", status)
    import pandas as pd
    pd.DataFrame([_metric_row(value) for value in status["conditions"].values()]).to_csv(
        suite / "comparison.csv", index=False)
    show_metrics(config, project, suite=suite)
    print(f"Workflow completed: {suite}", flush=True)
    return suite


def check(config, project, condition="distillation"):
    if condition not in CONDITIONS:
        raise ValueError("Unknown training condition")
    config, teacher, split, info = _prepare(config, project)
    device = supervised._device(config)
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    initial_hash = _state_hash(student.state_dict())
    teacher.to(device if condition == "distillation" else "cpu").eval()
    before = _state_hash(teacher.state_dict())
    kd = (SingleTeacherDistillationLoss(student_dim=student.feature_dim,
                                      teacher_dim=teacher.adapter.feature_dim,
                                      **config["distillation"]).to(device)
          if condition == "distillation" else None)
    hard_loss = supervised.FocalLoss(config["training"]["focal_gamma"], config["training"]["focal_scale"])
    loader = _loader(split, "train", teacher.adapter.preprocess, student.preprocess, config,
                     torch.Generator().manual_seed(config["seed"]), include_teacher=kd is not None)
    batch = next(iter(loader))
    values = _batch_loss(student, teacher, kd, batch, hard_loss, device)
    values["total"].backward()
    backbone_gradients = [parameter.grad for name, parameter in student.named_parameters()
                          if not name.startswith("encoder.head.") and parameter.grad is not None]
    if not backbone_gradients or not any(gradient.abs().sum().item() > 0 for gradient in backbone_gradients):
        raise RuntimeError("Student backbone received no nonzero training gradients")
    if any(not torch.isfinite(parameter.grad).all() for parameter in list(student.parameters()) + (list(kd.parameters()) if kd is not None else [])
           if parameter.grad is not None):
        raise RuntimeError("Smoke backward produced non-finite gradients")
    if kd is not None and kd.projector is not None and (kd.projector.weight.grad is None or not kd.projector.weight.grad.abs().sum().item()):
        raise RuntimeError("Global projection received no gradients")
    if before != _state_hash(teacher.state_dict()) or any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("Teacher changed in smoke test")
    student.eval()
    with torch.no_grad():
        expected = student(batch["image"].to(device)).detach().cpu()
    directory = _new_suite(config, kind="check")
    snapshot = directory / "roundtrip.pt"
    supervised._torch_save(snapshot, _student_payload(student, config, supervised._split_hash(split),
                            condition, initial_hash, info, loader.dataset.metadata()))
    restored, _ = load_student_checkpoint(snapshot, device)
    roundtrip_state = _verify_roundtrip_state(student, restored)
    with torch.no_grad():
        actual = restored(batch["image"].to(device)).detach().cpu()
    roundtrip_outputs = _verify_roundtrip_outputs(expected, actual, len(batch["label"]))
    snapshot.unlink()
    result = {"check_passed": True, "training_started": False,
              **roundtrip_state, **roundtrip_outputs,
              "teacher_unchanged": True, "student_full_checkpoint_roundtrip": True,
              "teacher_features": teacher.adapter.feature_dim, "student_features": student.feature_dim,
              "losses": {name: value.detach().item() for name, value in values.items()},
              "cohort_hash": supervised._split_hash(split), "shared_view": loader.dataset.metadata(),
              "config": config, "effective_training": config["training"],
              "condition": condition, "student_mode": _student_options(config)["mode"],
              "model_metadata": student.training_metadata(), "initial_state_sha256": initial_hash,
              "teacher": info,
              "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
              "distillation_trainable_parameters": sum(p.numel() for p in kd.parameters() if p.requires_grad) if kd is not None else 0}
    supervised._json(directory / "check.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"config", "effective_training", "teacher", "model_metadata"}},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"Check completed (no training run): {directory}", flush=True)
    return directory


def _metric_row(receipt):
    metrics = receipt["test_patient"]
    ci = receipt.get("roc_auc_ci") or {}
    return {"condition": receipt["condition"], "student_mode": receipt.get("student_mode", "full_finetune"),
            "seed": receipt["seed"],
            "val_auc": receipt["best_val_patient_roc_auc"], "roc_auc": metrics["roc_auc"],
            "roc_ci_lower": ci.get("lower"), "roc_ci_upper": ci.get("upper"),
            "pr_auc": metrics["pr_auc"], "ap": metrics["average_precision"],
            "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"]}


def show_metrics(config, project, suite=None):
    # Reporting is deliberately file-only: no weights, Torch model or patient scans.
    root = _path(config.get("output_root", "runs/single_teacher_distillation"), project)
    suites = [_path(suite, project)] if suite is not None else sorted((root / "suites").glob("suite_*"))
    found = False
    for batch in suites:
        rows = []
        for condition in CONDITIONS:
            result_file = batch / condition / "result.json"
            if result_file.is_file():
                rows.append(_metric_row(json.loads(result_file.read_text(encoding="utf-8"))))
        if not rows:
            continue
        found = True
        print(f"\nBatch: {batch}")
        names = list(rows[0])
        formatted = [[str(row[key]) if key in {"condition", "student_mode", "seed"} else
                      ("-" if row[key] is None else f"{row[key]:.4f}") for key in names] for row in rows]
        widths = [max(len(name), *(len(row[index]) for row in formatted)) for index, name in enumerate(names)]
        print("  ".join(name.ljust(widths[index]) for index, name in enumerate(names)))
        for row in formatted:
            print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    if not found:
        print(f"No completed single-teacher student records under {root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAD-DINO LoRA teacher -> supervised / three-term KD ViT-Tiny students")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Train-view forward/backward + full-student save/reload; no training")
    actions.add_argument("--show-metrics", action="store_true", help="Print all saved student batches without loading teacher/data")
    parser.add_argument("--suite", help="With --show-metrics, print this batch only")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device")
    parser.add_argument("--condition", choices=CONDITIONS,
                        help="Run/check one condition; default training still runs the paired experiment")
    parser.add_argument("--result-json", help="Write a machine-readable check or training receipt")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    config = json.loads(_path(args.config, project).read_text(encoding="utf-8"))
    for name in ("teacher_checkpoint", "student_checkpoint", "cache_dir", "device"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.suite and not args.show_metrics:
        parser.error("--suite is only valid with --show-metrics")
    if args.show_metrics:
        return show_metrics(config, project, suite=args.suite)
    if args.check_only:
        directory = check(config, project, condition=args.condition or "distillation")
        receipt = json.loads((directory / "check.json").read_text(encoding="utf-8"))
        receipt.update(action="check", directory=str(directory))
    else:
        directory = run_experiments(config, project, conditions=[args.condition] if args.condition else None)
        receipt = {"suite": str(directory), **json.loads((directory / "suite.json").read_text(encoding="utf-8"))}
    if args.result_json:
        supervised._json(_path(args.result_json, project), receipt)
    return directory


if __name__ == "__main__":
    main()
