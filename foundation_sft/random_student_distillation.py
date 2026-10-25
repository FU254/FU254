"""Matched A/B experiments with a randomly initialized, fully trained student.

Only student initialization changes relative to the ImageNet experiment. Keep
the frozen RAD-DINO teacher, patient split, shared view and selection protocol.
The original ImageNet trainer and saved results are not modified.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from . import distillation as single
from . import distillation_suite as records
from . import train as supervised
from .distillation_losses import SingleTeacherDistillationLoss
from .metrics import aggregate_patient_predictions, binary_metrics, save_evaluation, select_threshold
from .models.vit_tiny_student import build_vit_tiny_student, _strict_load


EXPERIMENT = "random_student_kd_v1"
CONDITIONS = ("supervised", "distillation")
LOSS_SETTINGS = {"temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
                 "local_weight": 1.0, "pool_grid": [7, 7]}


def _validate_config(config):
    if config.get("student_initialization") != "random":
        raise ValueError("This experiment requires student_initialization='random'")
    if config.get("student_checkpoint") is not None:
        raise ValueError("A random student must not specify an ImageNet student_checkpoint")
    if single._student_options(config)["mode"] != "full_finetune":
        raise ValueError("The confirmed A/B experiment trains all random student parameters")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Student seed must be an integer in [0, 2**32)")
    rate = config.get("training", {}).get("drop_path_rate", 0.0)
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("Student drop_path_rate must be finite and in [0,1)")
    records._same("confirmed three-term distillation", config.get("distillation"), LOSS_SETTINGS)


def _build_student(config):
    _validate_config(config)
    rate = config.get("training", {}).get("drop_path_rate", 0.0)
    if rate:
        return build_vit_tiny_student(random_seed=config["seed"], drop_path_rate=rate)
    return build_vit_tiny_student(random_seed=config["seed"])


def _prepare(config, project):
    from .training_history import save_history_plots

    config = deepcopy(config)
    _validate_config(config)
    config.setdefault("device", "cuda:0")
    config.setdefault("output_root", "runs/random_student_distillation")
    config.setdefault("student_checkpoint", None)
    supervised._device(config)
    config["teacher_checkpoint"] = str(single.resolve_teacher_checkpoint(config, project))
    for name in ("output_root", "cache_dir"):
        if config.get(name) is not None:
            config[name] = str(single._path(config[name], project))
    teacher, teacher_config, split, payload = single.load_teacher(
        config["teacher_checkpoint"], "cpu", config.get("cache_dir"))
    config["training"] = single._training_settings(teacher_config, config)
    config["evaluation"] = deepcopy(teacher_config["evaluation"])
    config["teacher_config"] = teacher_config
    SingleTeacherDistillationLoss(teacher_dim=teacher.adapter.feature_dim, **config["distillation"])
    info = {"checkpoint": config["teacher_checkpoint"],
        "checkpoint_sha256": records._file_hash(config["teacher_checkpoint"]),
        "model_metadata": payload["model_metadata"], "cohort_hash": payload["cohort_hash"],
        "best_val_patient_roc_auc": payload["best_val_patient_roc_auc"],
        "feature_dim": teacher.adapter.feature_dim,
        "selection": "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC"}
    return config, teacher, split, info


def _assert_random_metadata(metadata, seed):
    initialization = metadata.get("initialization", {})
    if (initialization.get("initialization_mode") != "random"
            or initialization.get("pretraining") != "none"
            or initialization.get("seed") != seed
            or any(key in initialization for key in ("checkpoint", "checkpoint_sha256", "official_checkpoint_url"))):
        raise ValueError("Student metadata must identify random initialization and the exact student seed")
    if metadata.get("mode") != "full_finetune":
        raise ValueError("Random A/B students must train the complete encoder and classifier")


def _assert_frozen(teacher):
    if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("RAD-DINO must remain in eval mode, frozen and gradient-free")


def _payload(student, config, cohort_hash, condition, initial, teacher, geometry, distillation_initial=None):
    _assert_random_metadata(student.training_metadata(), config["seed"])
    payload = single._student_payload(student, config, cohort_hash, condition, initial, teacher, geometry)
    payload.update(experiment=EXPERIMENT, student_initialization="random",
                   distillation_initial_state_sha256=distillation_initial)
    return payload


def _roundtrip(student, restored, expected=None, actual=None):
    before, after = single._state_hash(student.state_dict()), single._state_hash(restored.state_dict())
    if before != after:
        raise ValueError("Complete random student state changed during save/reload")
    proof = {"student_state_exact": True, "student_state_sha256": before, "restored_state_sha256": after}
    if expected is not None:
        for value in (expected, actual):
            if (not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.is_floating_point()
                    or not torch.isfinite(value).all().item()):
                raise ValueError("Random student roundtrip requires finite [batch] logits")
        if expected.shape != actual.shape or expected.numel() == 0:
            raise ValueError("Random student roundtrip logit shapes differ")
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        proof.update(roundtrip_max_abs_error=(actual - expected).abs().max().item(),
                     roundtrip_rtol=1e-4, roundtrip_atol=1e-5)
    return proof


def check(config, project, condition="distillation"):
    if condition not in CONDITIONS:
        raise ValueError("Select supervised or distillation")
    config, teacher, split, info = _prepare(config, project)
    device = supervised._device(config)
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    _assert_random_metadata(student.training_metadata(), config["seed"])
    initial = single._state_hash(student.state_dict())
    teacher.to(device if condition == "distillation" else "cpu").eval()
    before = single._state_hash(teacher.state_dict())
    _assert_frozen(teacher)
    kd = (SingleTeacherDistillationLoss(student_dim=student.feature_dim, teacher_dim=teacher.adapter.feature_dim,
        **config["distillation"]).to(device) if condition == "distillation" else None)
    kd_initial = single._state_hash(kd.state_dict()) if kd is not None else None
    criterion = supervised.FocalLoss(config["training"]["focal_gamma"], config["training"]["focal_scale"])
    supervised.set_seed(config["seed"])
    loader = single._loader(split, "train", teacher.adapter.preprocess, student.preprocess, config,
        torch.Generator().manual_seed(config["seed"]), include_teacher=kd is not None)
    batch = next(iter(loader))
    values = single._batch_loss(student, teacher, kd, batch, criterion, device)
    values["total"].backward()
    gradients = [p.grad for name, p in student.named_parameters() if not name.startswith("encoder.head.")]
    if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients):
        raise RuntimeError("Random student backbone received no finite nonzero training gradients")
    parameters = list(student.parameters()) + (list(kd.parameters()) if kd is not None else [])
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("Random student preflight produced non-finite gradients")
    if kd is not None and any(p.grad is None or not torch.isfinite(p.grad).all()
                              or p.grad.abs().sum() == 0 for p in kd.parameters()):
        raise RuntimeError("The global projector received no finite nonzero gradient")
    _assert_frozen(teacher)
    if before != single._state_hash(teacher.state_dict()):
        raise RuntimeError("Teacher changed during random-student preflight")
    student.eval()
    with torch.no_grad():
        expected = student(batch["image"].to(device)).detach().cpu()
    directory = single._new_suite(config, "check")
    snapshot = directory / "roundtrip.pt"
    supervised._torch_save(snapshot, _payload(student, config, supervised._split_hash(split), condition,
        initial, info, loader.dataset.metadata(), kd_initial))
    restored, _ = single.load_student_checkpoint(snapshot, device)
    with torch.no_grad():
        actual = restored(batch["image"].to(device)).detach().cpu()
    proof = _roundtrip(student, restored, expected, actual)
    snapshot.unlink()
    result = {"check_passed": True, "training_started": False, "condition": condition,
        "student_initialization": "random", "student_mode": "full_finetune", **proof,
        "teacher_unchanged": True, "student_full_checkpoint_roundtrip": True,
        "config": config, "effective_training": config["training"], "model_metadata": student.training_metadata(),
        "initial_state_sha256": initial, "cohort_hash": supervised._split_hash(split),
        "distillation_initial_state_sha256": kd_initial,
        "teacher": info, "shared_view": loader.dataset.metadata(),
        "losses": {name: value.detach().item() for name, value in values.items()}}
    supervised._json(directory / "check.json", result)
    print(f"Check passed: {condition}; random initialization, backbone backward, frozen teacher, exact student reload", flush=True)
    return directory


def _train_eval_loader(split, geometry, config):
    """Comparable train/validation Focal losses without random train augmentation."""
    dataset = single.inference_dataset(split.loc[split["split"] == "train"],
        {"format_version": single.FORMAT, "shared_view": geometry}, config.get("cache_dir"))
    settings = config["training"]
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False,
        num_workers=settings["num_workers"], pin_memory=str(config["device"]).startswith("cuda"),
        worker_init_fn=supervised._seed_worker,
        generator=torch.Generator().manual_seed(config["seed"] ^ 0x5A17))


class _ProgressLoader:
    """Expose batch progress while preserving the existing optimizer loop."""

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for index, batch in enumerate(self.loader, 1):
            yield batch
            if index % 50 == 0 or index == len(self.loader):
                print(f"[train] batch={index}/{len(self.loader)}", flush=True)


def _run_condition(condition, suite, teacher, split, config, info, expected_initial=None):
    if condition not in CONDITIONS:
        raise ValueError("Select supervised or distillation")
    run = Path(suite) / condition
    run.mkdir()
    device, train = supervised._device(config), config["training"]
    supervised.set_seed(config["seed"])
    student = _build_student(config).to(device)
    _assert_random_metadata(student.training_metadata(), config["seed"])
    initial = single._state_hash(student.state_dict())
    if expected_initial is not None and initial != expected_initial:
        raise ValueError("A/B must start from exactly the same random student state for each seed")
    kd = (SingleTeacherDistillationLoss(student_dim=student.feature_dim, teacher_dim=teacher.adapter.feature_dim,
        **config["distillation"]).to(device) if condition == "distillation" else None)
    kd_initial = single._state_hash(kd.state_dict()) if kd is not None else None
    teacher.to(device if kd is not None else "cpu").eval()
    _assert_frozen(teacher)
    before = single._state_hash(teacher.state_dict())
    optimizer = single._optimizer(student, kd, train)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"], eta_min=train["minimum_lr"])
    criterion = supervised.FocalLoss(train["focal_gamma"], train["focal_scale"])
    supervised.set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = single._loader(split, "train", teacher.adapter.preprocess, student.preprocess,
        config, generator, include_teacher=kd is not None)
    val_loader = single._loader(split, "val", teacher.adapter.preprocess, student.preprocess, config)
    geometry, cohort_hash = train_loader.dataset.metadata(), supervised._split_hash(split)
    train_eval_loader = _train_eval_loader(split, geometry, config)
    for name, value in (("config", config), ("model_metadata", student.training_metadata()),
            ("teacher", info), ("shared_view", geometry), ("split_metadata", split.attrs["split_metadata"])):
        supervised._json(run / f"{name}.json", value)
    split.to_csv(run / "split.csv", index=False)
    supervised._json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "FP32", "student_initialization": "random", "cohort_hash": cohort_hash})
    print(f"Run directory: {run}", flush=True)
    history, best_auc, bad_epochs, best_model, best_kd = [], -math.inf, 0, None, None
    best_stochastic = None
    for epoch in range(train["epochs"]):
        started = time.monotonic()
        print(f"[epoch] {epoch + 1}/{train['epochs']} started; initialization=random", flush=True)
        losses = single._train_epoch(student, teacher, kd, _ProgressLoader(train_loader), optimizer, criterion, device, train)
        train_predictions, train_eval_loss = supervised.predict(student, train_eval_loader, device, criterion=criterion)
        train_patients = aggregate_patient_predictions(train_predictions)
        train_metrics = binary_metrics(train_patients.label, train_patients.probability)
        validation, val_loss = supervised.predict(student, val_loader, device, criterion=criterion)
        patients = aggregate_patient_predictions(validation)
        metrics = binary_metrics(patients.label, patients.probability)
        val_auc = metrics["roc_auc"]
        if val_auc is None or not math.isfinite(val_auc):
            raise ValueError("Validation requires both classes for checkpoint selection")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_model, best_kd = student.trainable_state_dict(), single._cpu_state(kd) if kd is not None else None
            best_stochastic = getattr(student, "stochastic_state_dict", lambda: None)()
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "epoch_seconds": time.monotonic() - started,
            "train_loss": losses["total"], "train_hard_loss": losses["hard"], "train_kd_loss": losses["kd"],
            "train_soft_loss": losses["soft"], "train_global_loss": losses["global"], "train_local_loss": losses["local"],
            "train_eval_loss": train_eval_loss, "train_eval_patient_roc_auc": train_metrics["roc_auc"],
            "train_eval_patient_pr_auc": train_metrics["pr_auc"], "optimizer_steps": losses["optimizer_steps"],
            "val_loss": val_loss, "val_patient_roc_auc": val_auc, "val_patient_pr_auc": metrics["pr_auc"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        print(json.dumps({"condition": condition, "student_initialization": "random", **history[-1]}), flush=True)
        scheduler.step()
        payload = _payload(student, config, cohort_hash, condition, initial, info, geometry, kd_initial)
        payload.update(epoch=epoch, best_auc=best_auc, bad_epochs=bad_epochs, best_model=best_model,
            best_distillation_state=best_kd, distillation_state=single._cpu_state(kd) if kd is not None else None,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=supervised._rng_state(generator), history=history)
        if best_stochastic is not None:
            payload["best_student_stochastic_state"] = best_stochastic
        supervised._torch_save(run / "last.pt", payload)
        supervised._json(run / "history.json", history)
        if bad_epochs >= train["patience"]:
            break
    if best_model is None:
        raise RuntimeError("No validation-selected random student was produced")
    from .training_history import save_history_plots
    # Plot like-for-like Focal losses; keep the actual optimization/KD totals in history.json.
    save_history_plots([{**row, "train_loss": row["train_eval_loss"]} for row in history], run)
    supervised._json(run / "history_plot_metadata.json", {
        "training_loss": "train_eval_loss: Focal, eval mode, no random augmentation",
        "validation_loss": "val_loss: Focal, eval mode, no random augmentation",
        "optimization_losses": "history.json retains train_loss, train_hard_loss and all KD components"})
    student.load_trainable_state_dict(best_model)
    if best_stochastic is not None:
        student.load_stochastic_state_dict(best_stochastic)
    validation = supervised.predict(student, val_loader, device)
    patients = aggregate_patient_predictions(validation)
    threshold = select_threshold(patients.label, patients.probability)
    payload = _payload(student, config, cohort_hash, condition, initial, info, geometry, kd_initial)
    payload.update(threshold=threshold, best_val_patient_roc_auc=best_auc, distillation_state=best_kd)
    supervised._torch_save(run / "best.pt", payload)
    restored, restored_payload = single.load_student_checkpoint(run / "best.pt", device)
    _roundtrip(student, restored)
    del student
    student = restored
    validation.to_csv(run / "validation_predictions.csv", index=False)
    supervised._json(run / "selection.json", {"threshold": threshold, "source": "validation patient predictions",
        "best_val_patient_roc_auc": best_auc})
    _assert_frozen(teacher)
    if before != single._state_hash(teacher.state_dict()):
        raise RuntimeError("Frozen teacher changed during random-student training")
    teacher.to("cpu")
    print("[evaluation] Random student reloaded; evaluating test once", flush=True)
    test_loader = DataLoader(single.inference_dataset(split.loc[split["split"] == "test"], restored_payload,
        config.get("cache_dir")), batch_size=train["batch_size"], shuffle=False, num_workers=train["num_workers"],
        pin_memory=device.type == "cuda", worker_init_fn=supervised._seed_worker)
    test = supervised.predict(student, test_loader, device)
    result = save_evaluation(test, run / "test", threshold=threshold,
        bootstrap_samples=config["evaluation"]["bootstrap_samples"], seed=config["seed"])
    receipt = {"status": "completed", "experiment": EXPERIMENT, "condition": condition, "run": str(run),
        "student_initialization": "random", "student_mode": "full_finetune", "seed": config["seed"],
        "initial_state_sha256": initial, "cohort_hash": cohort_hash, "best_val_patient_roc_auc": best_auc,
        "distillation_initial_state_sha256": kd_initial,
        "threshold": threshold, "test_patient": result["patient"],
        "roc_auc_ci": result["bootstrap"]["patient"].get("roc_auc"), "teacher_unchanged": True,
        "student_trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "distillation_trainable_parameters": sum(p.numel() for p in kd.parameters() if p.requires_grad) if kd is not None else 0}
    supervised._json(run / "result.json", receipt)
    return receipt


def verify_completed_run(run, check, job, *, checkpoint_loader=None):
    """Accept only complete random students from the exact checked A/B protocol."""
    run = Path(run).resolve()
    loader = checkpoint_loader or records._default_checkpoint_loader
    payload = loader(run / "best.pt")
    saved = records._read(run / "config.json")
    for config in (saved, check["config"], payload.get("config", {})):
        _validate_config(config)
    for metadata in (check["model_metadata"], payload.get("model_metadata", {})):
        _assert_random_metadata(metadata, job["seed"])
    result = records.verify_run(run, check, job, checkpoint_loader=lambda path: payload)
    records._same("random student experiment", result.get("experiment"), EXPERIMENT)
    records._same("checkpoint random experiment", payload.get("experiment"), EXPERIMENT)
    records._same("result random initialization", result.get("student_initialization"), "random")
    records._same("checkpoint random initialization", payload.get("student_initialization"), "random")
    records._same("checkpoint initial projector", payload.get("distillation_initial_state_sha256"),
                  result.get("distillation_initial_state_sha256"))
    if "distillation_initial_state_sha256" in check:
        records._same("initial projector", payload.get("distillation_initial_state_sha256"),
                      check["distillation_initial_state_sha256"])
    if job["condition"] == "distillation":
        expected = SingleTeacherDistillationLoss(student_dim=check["model_metadata"]["feature_dim"],
            teacher_dim=check["teacher"]["feature_dim"], **saved["distillation"])
        _strict_load(expected, payload.get("distillation_state"), "Random student global projector")
    elif payload.get("distillation_state") is not None:
        raise ValueError("A supervised random student must not contain a trained KD projector")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", required=True, help="Resolved job config from run_random_student_experiments.py")
    parser.add_argument("--condition", choices=CONDITIONS, default="distillation")
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
        config, teacher, split, info = _prepare(config, project)
        directory = single._new_suite(config)
        receipt = _run_condition(args.condition, directory, teacher, split, config, info)
    if args.result_json:
        supervised._json(single._path(args.result_json, project), receipt)
    return directory


if __name__ == "__main__":
    main()
