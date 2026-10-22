"""Frozen RAD-DINO / CheXFound / CheXzero, equally weighted, full ViT-Tiny KD.

The single-teacher implementation remains the source of truth for both control
arms. The deployed student retains that implementation's complete checkpoint
and preprocessing format and needs none of the teachers at inference time.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import distillation as single
from . import distillation_suite as records
from . import train as supervised
from .distillation_losses import SingleTeacherDistillationLoss
from .equal_teacher_data import MultiTeacherViewDataset
from .equal_teacher_losses import EqualTeacherDistillationLoss, TEACHER_NAMES
from .equal_teacher_sources import load_teacher, validate_teacher_cohorts
from .metrics import aggregate_patient_predictions, binary_metrics, save_evaluation, select_threshold
from .models.vit_tiny_student import _strict_load


EXPERIMENT = "equal_teacher_kd_v1"
CONDITIONS = ("supervised", "single_teacher", "equal_teacher")
LOSS_SETTINGS = {"temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
                 "local_weight": 1.0, "pool_grid": [7, 7]}


def _verify_roundtrip_state(student, restored):
    original = single._state_hash(student.state_dict())
    actual = single._state_hash(restored.state_dict())
    if original != actual:
        raise ValueError("Complete student state changed during save/reload")
    return {"student_state_exact": True, "student_state_sha256": original,
            "restored_state_sha256": actual}


def _verify_roundtrip_outputs(expected, actual, count):
    for value in (expected, actual):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or tuple(value.shape) != (count,) or not torch.isfinite(value).all().item()):
            raise ValueError("Saved/reloaded student must return finite [batch] logits")
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    return {"roundtrip_max_abs_error": (actual - expected).abs().max().item(),
            "roundtrip_rtol": 1e-4, "roundtrip_atol": 1e-5}


def _prepare(config, project):
    config = deepcopy(config)
    if single._student_options(config)["mode"] != "full_finetune":
        raise ValueError("The confirmed equal-teacher protocol uses full_finetune ViT-Tiny")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("Student seed must be an integer in [0, 2**32)")
    records._same("confirmed KD coefficients", config.get("distillation"), LOSS_SETTINGS)
    sources = config.get("teachers")
    if (not isinstance(sources, list) or len(sources) != 3
            or [source.get("model") for source in sources] != list(TEACHER_NAMES)):
        raise ValueError("Declare exactly RAD-DINO, CheXFound, CheXzero, in that order")
    config.setdefault("device", "cuda:0")
    config.setdefault("output_root", "runs/equal_teacher_distillation")
    for key in ("student_checkpoint", "output_root", "cache_dir"):
        if config.get(key) is not None:
            config[key] = str(single._path(config[key], project))
    supervised._device(config)
    if not Path(config["student_checkpoint"]).is_file():
        raise FileNotFoundError(f"Student ImageNet checkpoint is missing: {config['student_checkpoint']}")
    loaded = [load_teacher(source, project, cache_dir=config.get("cache_dir")) for source in sources]
    split = validate_teacher_cohorts(loaded)
    teachers = {name: item[0] for name, item in zip(TEACHER_NAMES, loaded)}
    infos = [deepcopy(item[4]) for item in loaded]
    for info, teacher in zip(infos, teachers.values()):
        info["feature_dim"] = teacher.adapter.feature_dim
    anchor_config, anchor_payload = loaded[0][1], loaded[0][3]
    config["teacher_checkpoint"] = infos[0]["checkpoint"]
    config["teacher_config"] = anchor_config
    config["training"] = single._training_settings(anchor_config, config)
    config["evaluation"] = deepcopy(anchor_config["evaluation"])
    # Keep this exact legacy schema for strict reuse of the two control arms.
    anchor_info = {key: infos[0][key] for key in (
        "checkpoint", "checkpoint_sha256", "model_metadata", "cohort_hash", "best_val_patient_roc_auc")}
    anchor_info["selection"] = "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC"
    records._same("anchor provenance", anchor_payload["model_metadata"], anchor_info["model_metadata"])
    return config, teachers, split, infos, anchor_info


def _loader(split, name, teachers, student, config, generator=None):
    train = config["training"]
    legacy = single._loader(split, name, teachers["rad_dino"].adapter.preprocess,
                            student.preprocess, config, generator, include_teacher=True)
    paired = MultiTeacherViewDataset(legacy.dataset,
        {name: teacher.adapter.preprocess for name, teacher in teachers.items()})
    return DataLoader(paired, batch_size=train["batch_size"], shuffle=name == "train",
        num_workers=train["num_workers"], drop_last=False,
        pin_memory=str(config["device"]).startswith("cuda"),
        worker_init_fn=supervised._seed_worker, generator=generator)


def _build_loss(student, teachers, config):
    return EqualTeacherDistillationLoss(
        {name: teacher.adapter.feature_dim for name, teacher in teachers.items()},
        student_dim=student.feature_dim, **config["distillation"])


def _assert_frozen(teachers):
    for name, teacher in teachers.items():
        if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
            raise RuntimeError(f"Teacher {name} must remain frozen, in eval mode, and gradient-free")


def _batch_loss(student, teachers, kd, batch, criterion, device):
    _assert_frozen(teachers)
    features = student(batch["image"].to(device), return_features=True)
    references = {}
    with torch.no_grad():
        # Only frozen teacher activations survive; never retain three autograd graphs.
        for name in TEACHER_NAMES:
            references[name] = teachers[name](batch["teacher_images"][name].to(device), return_features=True)
    terms = kd(features["logits"], references,
        student_global=features["global_features"], student_patches=features["local_features"],
        student_grid=features["grid_shape"])
    hard = criterion(features["logits"], batch["label"].to(device))
    values = {key: terms[key] for key in ("soft", "global", "local")}
    values.update(hard=hard, kd=terms["total"], total=hard + terms["total"])
    for name in TEACHER_NAMES:
        for key in ("total", "soft", "global", "local"):
            values[f"{name}_{key}"] = terms["per_teacher"][name][key]
    if any(not torch.isfinite(value).item() for value in values.values()):
        raise RuntimeError("Non-finite equal-teacher loss; no optimizer update applied")
    return values


def _train_epoch(student, teachers, kd, loader, optimizer, criterion, device, train):
    student.train()
    kd.train()
    _assert_frozen(teachers)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    optimizer.zero_grad(set_to_none=True)
    totals, samples, window, denominator, steps = {}, 0, 0, None, 0
    for index, batch in enumerate(loader):
        values = _batch_loss(student, teachers, kd, batch, criterion, device)
        count = len(batch["label"])
        if not window:
            denominator = count * train["accumulation_steps"]
        (values["total"] * (count / denominator)).backward()
        window += count
        samples += count
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * count
        if (index + 1) % train["accumulation_steps"] == 0 or index + 1 == len(loader):
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError("Missing or non-finite equal-teacher FP32 gradients")
            for gradient in gradients:
                gradient.mul_(denominator / window)
            nn.utils.clip_grad_norm_(parameters, train["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            window = 0
        if (index + 1) % 50 == 0:
            print(f"[train] batch={index + 1}/{len(loader)} optimizer_steps={steps}", flush=True)
    if not steps:
        raise ValueError("Empty equal-teacher training loader")
    _assert_frozen(teachers)
    return {**{key: value / samples for key, value in totals.items()}, "optimizer_steps": steps}


def _payload(student, config, cohort_hash, initial, anchor, geometry, infos, views):
    payload = single._student_payload(student, config, cohort_hash, "equal_teacher", initial, anchor, geometry)
    payload.update(experiment=EXPERIMENT, teachers=deepcopy(infos), teacher_views=deepcopy(views))
    return payload


def check(config, project, condition="equal_teacher"):
    if condition not in CONDITIONS:
        raise ValueError("Unknown equal-teacher experiment condition")
    config, teachers, split, infos, anchor = _prepare(config, project)
    device = supervised._device(config)
    supervised.set_seed(config["seed"])
    student = single._build_student(config).to(device)
    initial = single._state_hash(student.state_dict())
    before = {name: single._state_hash(teacher.state_dict()) for name, teacher in teachers.items()}
    for name, teacher in teachers.items():
        teacher.to(device if condition == "equal_teacher" or
                   (condition == "single_teacher" and name == "rad_dino") else "cpu").eval()
    paired = _loader(split, "train", teachers, student, config, torch.Generator().manual_seed(config["seed"]))
    kd = (_build_loss(student, teachers, config).to(device) if condition == "equal_teacher" else
          SingleTeacherDistillationLoss(student_dim=student.feature_dim,
              teacher_dim=teachers["rad_dino"].adapter.feature_dim, **config["distillation"]).to(device)
          if condition == "single_teacher" else None)
    criterion = supervised.FocalLoss(config["training"]["focal_gamma"], config["training"]["focal_scale"])
    supervised.set_seed(config["seed"])
    batch = next(iter(paired))
    values = (_batch_loss(student, teachers, kd, batch, criterion, device) if condition == "equal_teacher" else
              single._batch_loss(student, teachers["rad_dino"], kd, batch, criterion, device))
    values["total"].backward()
    backbone = [p.grad for name, p in student.named_parameters() if "head" not in name and p.requires_grad]
    if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in backbone):
        raise RuntimeError("Preflight produced no finite nonzero student backbone gradient")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
        raise RuntimeError("Preflight produced non-finite student gradients")
    if kd is not None and any(p.grad is None or not torch.isfinite(p.grad).all()
                              or p.grad.abs().sum() == 0 for p in kd.parameters()):
        raise RuntimeError("Every enabled global projector must receive finite nonzero gradients")
    _assert_frozen(teachers)
    for name, teacher in teachers.items():
        if before[name] != single._state_hash(teacher.state_dict()):
            raise RuntimeError(f"Teacher {name} changed in preflight")
    student.eval()
    with torch.no_grad():
        expected = student(batch["image"].to(device)).detach().cpu()
    directory = single._new_suite(config, "check")
    snapshot = directory / "roundtrip.pt"
    payload = _payload(student, config, supervised._split_hash(split), initial, anchor,
                       paired.dataset.metadata(), infos, paired.dataset.teacher_view_metadata())
    payload["condition"] = condition
    supervised._torch_save(snapshot, payload)
    restored, _ = single.load_student_checkpoint(snapshot, device)
    proof = _verify_roundtrip_state(student, restored)
    with torch.no_grad():
        actual = restored(batch["image"].to(device)).detach().cpu()
    proof.update(_verify_roundtrip_outputs(expected, actual, len(batch["label"])))
    snapshot.unlink()
    result = {"check_passed": True, "training_started": False, **proof,
        "condition": condition, "student_mode": "full_finetune", "teacher_unchanged": True,
        "config": config, "effective_training": config["training"],
        "model_metadata": student.training_metadata(), "initial_state_sha256": initial,
        "cohort_hash": supervised._split_hash(split), "teacher": anchor, "teachers": infos,
        "shared_view": paired.dataset.metadata(), "teacher_views": paired.dataset.teacher_view_metadata(),
        "losses": {key: value.detach().item() for key, value in values.items()}}
    supervised._json(directory / "check.json", result)
    print(f"Check passed: {condition}; student backward, frozen teachers, exact checkpoint reload", flush=True)
    return directory


def _run_equal(suite, config, teachers, split, infos, anchor):
    run = Path(suite) / "equal_teacher"
    run.mkdir()
    device, train = supervised._device(config), config["training"]
    supervised.set_seed(config["seed"])
    student = single._build_student(config).to(device)
    initial = single._state_hash(student.state_dict())
    kd = _build_loss(student, teachers, config).to(device)
    before = {name: single._state_hash(teacher.state_dict()) for name, teacher in teachers.items()}
    for teacher in teachers.values():
        teacher.to(device).eval()
    _assert_frozen(teachers)
    optimizer = single._optimizer(student, kd, train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"], eta_min=train["minimum_lr"])
    criterion = supervised.FocalLoss(train["focal_gamma"], train["focal_scale"])
    # Match controls' shuffle and geometric RNG after all projectors are built.
    supervised.set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = _loader(split, "train", teachers, student, config, generator)
    val_loader = single._loader(split, "val", teachers["rad_dino"].adapter.preprocess, student.preprocess, config)
    geometry, views = loader.dataset.metadata(), loader.dataset.teacher_view_metadata()
    cohort_hash = supervised._split_hash(split)
    for name, value in (("config", config), ("model_metadata", student.training_metadata()),
                        ("shared_view", geometry), ("teacher", anchor), ("teachers", infos), ("teacher_views", views),
                        ("split_metadata", split.attrs["split_metadata"])):
        supervised._json(run / f"{name}.json", value)
    split.to_csv(run / "split.csv", index=False)
    supervised._json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "FP32", "cohort_hash": cohort_hash})
    print(f"Run directory: {run}", flush=True)
    history, best_auc, bad_epochs = [], -math.inf, 0
    best_model = best_kd = None
    for epoch in range(train["epochs"]):
        started = time.monotonic()
        print(f"[epoch] {epoch + 1}/{train['epochs']} started", flush=True)
        losses = _train_epoch(student, teachers, kd, loader, optimizer, criterion, device, train)
        validation, val_loss = supervised.predict(student, val_loader, device, criterion=criterion)
        patients = aggregate_patient_predictions(validation)
        metrics = binary_metrics(patients.label, patients.probability)
        val_auc = metrics["roc_auc"]
        if val_auc is None or not math.isfinite(val_auc):
            raise ValueError("Validation must contain both classes for checkpoint selection")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_model, best_kd = student.trainable_state_dict(), single._cpu_state(kd)
        else:
            bad_epochs += 1
        record = {"epoch": epoch + 1, "epoch_seconds": time.monotonic() - started,
            "train_loss": losses["total"], "train_hard_loss": losses["hard"], "train_kd_loss": losses["kd"],
            "train_soft_loss": losses["soft"], "train_global_loss": losses["global"], "train_local_loss": losses["local"],
            "teacher_losses": {name: {key: losses[f"{name}_{key}"] for key in ("total", "soft", "global", "local")}
                               for name in TEACHER_NAMES},
            "optimizer_steps": losses["optimizer_steps"], "val_loss": val_loss,
            "val_patient_roc_auc": val_auc, "val_patient_pr_auc": metrics["pr_auc"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        history.append(record)
        print(json.dumps({"condition": "equal_teacher", **record}), flush=True)
        scheduler.step()
        payload = _payload(student, config, cohort_hash, initial, anchor, geometry, infos, views)
        payload.update(epoch=epoch, best_auc=best_auc, bad_epochs=bad_epochs, best_model=best_model,
            best_distillation_state=best_kd, distillation_state=single._cpu_state(kd),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=supervised._rng_state(generator), history=history)
        supervised._torch_save(run / "last.pt", payload)
        supervised._json(run / "history.json", history)
        if bad_epochs >= train["patience"]:
            break
    if best_model is None:
        raise RuntimeError("No validation-selected student was produced")
    from .training_history import save_history_plots
    save_history_plots(history, run)
    student.load_trainable_state_dict(best_model)
    validation = supervised.predict(student, val_loader, device)
    patients = aggregate_patient_predictions(validation)
    threshold = select_threshold(patients.label, patients.probability)
    payload = _payload(student, config, cohort_hash, initial, anchor, geometry, infos, views)
    payload.update(threshold=threshold, best_val_patient_roc_auc=best_auc, distillation_state=best_kd)
    supervised._torch_save(run / "best.pt", payload)
    restored, restored_payload = single.load_student_checkpoint(run / "best.pt", device)
    _verify_roundtrip_state(student, restored)
    del student
    student = restored
    validation.to_csv(run / "validation_predictions.csv", index=False)
    supervised._json(run / "selection.json", {"threshold": threshold, "source": "validation patient predictions",
        "best_val_patient_roc_auc": best_auc})
    _assert_frozen(teachers)
    for name, teacher in teachers.items():
        if before[name] != single._state_hash(teacher.state_dict()):
            raise RuntimeError(f"Teacher {name} changed during training")
        teacher.to("cpu")
    print("[evaluation] Validation-selected checkpoint reloaded; evaluating test once", flush=True)
    test_loader = DataLoader(single.inference_dataset(split.loc[split["split"] == "test"], restored_payload,
        config.get("cache_dir")), batch_size=train["batch_size"], shuffle=False, num_workers=train["num_workers"],
        pin_memory=device.type == "cuda", worker_init_fn=supervised._seed_worker)
    test = supervised.predict(student, test_loader, device)
    result = save_evaluation(test, run / "test", threshold=threshold,
        bootstrap_samples=config["evaluation"]["bootstrap_samples"], seed=config["seed"])
    receipt = {"condition": "equal_teacher", "experiment": EXPERIMENT, "status": "completed", "run": str(run),
        "student_mode": "full_finetune", "seed": config["seed"], "initial_state_sha256": initial,
        "cohort_hash": cohort_hash, "best_val_patient_roc_auc": best_auc, "threshold": threshold,
        "test_patient": result["patient"], "roc_auc_ci": result["bootstrap"]["patient"].get("roc_auc"),
        "teacher_unchanged": True, "teachers": infos, "teacher_views": views,
        "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "distillation_trainable_parameters": sum(p.numel() for p in kd.parameters() if p.requires_grad)}
    supervised._json(run / "result.json", receipt)
    return receipt


def verify_completed_run(run, check, job, *, checkpoint_loader=None):
    """Validate equal-KD provenance and finished evaluation, including all teachers."""
    run = Path(run).resolve()
    # Legacy verification supplies complete student state/metadata/metrics checks;
    # its generic condition handling accepts the added equal_teacher condition.
    loader = checkpoint_loader or records._default_checkpoint_loader
    payload = loader(run / "best.pt")
    result = records.verify_run(run, check, job, checkpoint_loader=lambda path: payload)
    saved = records._read(run / "config.json")
    records._same("equal KD coefficients", saved.get("distillation"), check["config"].get("distillation"))
    records._same("equal KD experiment", result.get("experiment"), EXPERIMENT)
    infos, views = records._read(run / "teachers.json"), records._read(run / "teacher_views.json")
    records._same("all teacher checkpoint provenance", infos, check["teachers"])
    records._same("all teacher effective views", views, check["teacher_views"])
    records._same("result teacher provenance", result.get("teachers"), infos)
    records._same("result teacher views", result.get("teacher_views"), views)
    records._same("checkpoint experiment", payload.get("experiment"), EXPERIMENT)
    records._same("checkpoint teacher provenance", payload.get("teachers"), infos)
    records._same("checkpoint teacher views", payload.get("teacher_views"), views)
    records._same("checkpoint KD coefficients", payload.get("config", {}).get("distillation"), saved.get("distillation"))
    expected = EqualTeacherDistillationLoss(
        {info["model"]: info["feature_dim"] for info in infos},
        student_dim=check["model_metadata"]["feature_dim"], **saved["distillation"])
    _strict_load(expected, payload.get("distillation_state"), "All three global projectors")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", required=True, help="A resolved per-job config from the equal-teacher coordinator")
    parser.add_argument("--condition", choices=CONDITIONS, default="equal_teacher")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--result-json")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    config = records._read(single._path(args.config, project))
    if args.check_only:
        directory = check(config, project, args.condition)
        receipt = records._read(directory / "check.json")
        receipt.update(action="check", directory=str(directory))
    else:
        config, teachers, split, infos, anchor = _prepare(config, project)
        directory = single._new_suite(config)
        if args.condition == "equal_teacher":
            receipt = _run_equal(directory, config, teachers, split, infos, anchor)
        else:
            mapped = "supervised" if args.condition == "supervised" else "distillation"
            receipt = single._run_condition(mapped, directory, teachers["rad_dino"], split, config, anchor)
            receipt = dict(receipt, source_condition=mapped, condition=args.condition)
    if args.result_json:
        supervised._json(single._path(args.result_json, project), receipt)
    return directory


if __name__ == "__main__":
    main()
