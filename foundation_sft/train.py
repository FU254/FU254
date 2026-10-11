"""Shared prepare, GPU smoke, supervised training, and final evaluation entry."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data import CXRDataset, build_cohort
from .metrics import (aggregate_patient_predictions, binary_metrics,
                      save_evaluation, select_threshold)
from .models import build_adapter
from .models.common import FoundationClassifier
from .split import load_or_create_split


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, scale=0.8):
        super().__init__()
        self.gamma, self.scale = gamma, scale

    def forward(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (self.scale * (1 - torch.exp(-bce)).pow(self.gamma) * bce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as file:
        temporary = Path(file.name)
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def _torch_save(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(config):
    device = torch.device(config["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose a CPU or CUDA device")
    if device.type == "cuda":
        index = device.index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but it is unavailable; select an available GPU or cpu")
    return device


def prepare(config):
    cohort = build_cohort(config)
    split = load_or_create_split(cohort, config["split_manifest"],
                                 seed=config["split_seed"], ratios=config["split_ratios"])
    diagnostics = dict(cohort.attrs.get("cohort_diagnostics", {}))
    diagnostics["splits"] = {
        name: {"images": int((split["split"] == name).sum()),
               "patients": int(split.loc[split["split"] == name, "patient_id"].nunique()),
               "positives": int(split.loc[split["split"] == name, "label"].sum())}
        for name in ("train", "val", "test")
    }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)
    return split


def _classifier(adapter, config):
    train = config["training"]
    return FoundationClassifier(adapter, mode=config["mode"], rank=train["lora_rank"],
                                alpha=train["lora_alpha"], hidden_dims=train["hidden_dims"])


def _loader(split, name, adapter, config, generator=None):
    train = config["training"]
    dataset = CXRDataset(split.loc[split["split"] == name], adapter.preprocess,
                         training=name == "train", seed=config["seed"],
                         horizontal_flip_probability=train["horizontal_flip_probability"],
                         rotation_degrees=train["rotation_degrees"])
    return DataLoader(dataset, batch_size=train["batch_size"], shuffle=name == "train",
                      num_workers=train["num_workers"], drop_last=False,
                      pin_memory=str(config["device"]).startswith("cuda"),
                      worker_init_fn=_seed_worker, generator=generator)


def predict(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            probabilities = logits.sigmoid().cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Non-finite validation/test predictions")
            for sample, patient, label, probability in zip(
                    batch["sample_id"], batch["patient_id"], batch["label"].numpy(), probabilities):
                rows.append({"sample_id": sample, "patient_id": patient,
                             "label": int(label), "probability": float(probability)})
    return pd.DataFrame(rows)


def _train_epoch(model, loader, optimizer, criterion, device, train, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_images, window_images = 0.0, 0, 0
    amp = bool(train["amp"] and device.type == "cuda")
    for index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast() if amp else nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        count = len(labels)
        # Sum sample losses before accumulation, then normalize once by the
        # actual window size: the final short batch/window has the right weight.
        scaler.scale(loss * count).backward()
        window_images += count
        total_images += count
        total_loss += loss.detach().item() * count
        end_window = (index + 1) % train["accumulation_steps"] == 0 or index + 1 == len(loader)
        if end_window:
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(window_images)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                    train["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window_images = 0
    return total_loss / total_images


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                      numpy_state[3], numpy_state[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _split_hash(split):
    metadata = split.attrs.get("split_metadata", {})
    if not metadata.get("_manifest_hash"):
        raise ValueError("Missing frozen patient split fingerprint")
    return str(metadata["_manifest_hash"])


def _checkpoint(model, config, cohort_hash):
    return {"format_version": 1, "model": model.trainable_state_dict(),
            "model_metadata": model.training_metadata(), "cohort_hash": cohort_hash,
            "config": config}


def _load_checkpoint(path, model, config, cohort_hash):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("cohort_hash") != cohort_hash:
        raise ValueError("Checkpoint format or cohort differs from this frozen dataset")
    if state["model_metadata"] != model.training_metadata():
        raise ValueError("Checkpoint encoder, source weights, preprocessing or training mode differs")
    model.load_trainable_state_dict(state["model"])
    return state


def _new_run(config, suffix=""):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config["output_root"]) / config["model"] / config["mode"] / (
        f"seed{config['seed']}_{stamp}_{uuid.uuid4().hex[:8]}{suffix}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_training(config, adapter=None, resume=None):
    set_seed(config["seed"])
    split = prepare(config)
    cohort_hash = _split_hash(split)
    device = _device(config)
    adapter = build_adapter(config) if adapter is None else adapter
    model = _classifier(adapter, config).to(device)
    train = config["training"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = _loader(split, "train", adapter, config, generator)
    val_loader = _loader(split, "val", adapter, config)
    head_params, lora_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (lora_params if name.endswith(("lora_A", "lora_B")) else head_params).append(param)
    groups = [{"params": head_params, "lr": train["head_lr"], "weight_decay": train["weight_decay"]}]
    if lora_params:
        groups.append({"params": lora_params, "lr": train["lora_lr"], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train["epochs"], eta_min=train["minimum_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train["amp"] and device.type == "cuda"))
    criterion = FocalLoss(train["focal_gamma"], train["focal_scale"])
    start_epoch, best_auc, bad_epochs, best_model = 0, -math.inf, 0, None
    history = []
    if resume:
        previous = _load_checkpoint(resume, model, config, cohort_hash)
        if previous["config"]["training"] != train or previous["config"]["seed"] != config["seed"]:
            raise ValueError("Resume requires the same training settings and seed")
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch = previous["epoch"] + 1
        best_auc, bad_epochs = previous["best_auc"], previous["bad_epochs"]
        best_model, history = previous["best_model"], previous["history"]
        _restore_rng(previous["rng"], generator)
    run = _new_run(config)
    _json(run / "config.json", config)
    _json(run / "model_metadata.json", model.training_metadata())
    _json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                                     "device": str(device), "cohort_hash": cohort_hash,
                                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    split.to_csv(run / "split.csv", index=False)
    _json(run / "split_metadata.json", split.attrs["split_metadata"])
    print(f"Run directory: {run.resolve()}", flush=True)
    for epoch in range(start_epoch, train["epochs"]):
        loss = _train_epoch(model, train_loader, optimizer, criterion, device, train, scaler)
        validation = predict(model, val_loader, device)
        patient_validation = aggregate_patient_predictions(validation)
        metrics = binary_metrics(patient_validation.label, patient_validation.probability)
        val_auc = metrics["roc_auc"]
        if val_auc is None:
            raise ValueError("Validation patients contain only one class; model selection is undefined")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            best_model = {name: tensor.clone() for name, tensor in model.trainable_state_dict().items()}
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": loss, "val_patient_roc_auc": val_auc,
                        "val_patient_pr_auc": metrics["pr_auc"],
                        "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        print(json.dumps(history[-1]), flush=True)
        scheduler.step()
        payload = _checkpoint(model, config, cohort_hash)
        payload.update(epoch=epoch, best_auc=best_auc, bad_epochs=bad_epochs,
                       best_model=best_model, optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                       rng=_rng_state(generator), history=history)
        _torch_save(run / "last.pt", payload)
        _json(run / "history.json", history)
        if bad_epochs >= train["patience"]:
            break
    if best_model is None:
        raise RuntimeError("No best checkpoint was produced")
    model.load_trainable_state_dict(best_model)
    validation = predict(model, val_loader, device)
    patients = aggregate_patient_predictions(validation)
    threshold = select_threshold(patients.label, patients.probability)
    best_payload = _checkpoint(model, config, cohort_hash)
    best_payload.update(threshold=threshold, best_val_patient_roc_auc=best_auc)
    _torch_save(run / "best.pt", best_payload)
    # Reload precisely the file used by future inference before touching test.
    _load_checkpoint(run / "best.pt", model, config, cohort_hash)
    validation.to_csv(run / "validation_predictions.csv", index=False)
    test_loader = _loader(split, "test", adapter, config)
    test = predict(model, test_loader, device)
    result = save_evaluation(test, run / "test", threshold=threshold,
                             bootstrap_samples=config["evaluation"]["bootstrap_samples"], seed=config["seed"])
    _json(run / "selection.json", {"threshold": threshold, "source": "validation patient predictions",
                                    "best_val_patient_roc_auc": best_auc})
    print(json.dumps({"test_patient": result["patient"], "run": str(run.resolve())}, ensure_ascii=False), flush=True)
    return run


def smoke(config):
    set_seed(config["seed"])
    split = prepare(config)
    device = _device(config)
    adapter = build_adapter(config)
    model = _classifier(adapter, config).to(device)
    batch = next(iter(_loader(split, "train", adapter, config)))
    model.train()
    features = model(batch["image"].to(device), return_features=True)
    loss = FocalLoss()(features["logits"], batch["label"].to(device))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("Missing or non-finite trainable gradients")
    if config["mode"] == "lora" and any(
            p.grad is None for name, p in model.named_parameters() if name.endswith("lora_B")):
        raise RuntimeError("A selected LoRA projection is disconnected from the classification loss")
    if any(p.grad is not None for p in model.adapter.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen encoder unexpectedly received gradients")
    model.eval()
    with torch.no_grad():
        before = model(batch["image"].to(device)).clone()
    state = model.trainable_state_dict()
    with tempfile.TemporaryDirectory() as temporary:
        file = Path(temporary) / "adapter.pt"
        torch.save(state, file)
        model.load_trainable_state_dict(torch.load(file, map_location="cpu", weights_only=True))
    with torch.no_grad():
        after = model(batch["image"].to(device))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {**model.training_metadata(), "logits_shape": list(features["logits"].shape),
              "global_shape": list(features["global_features"].shape),
              "local_shape": list(features["local_features"].shape),
              "grid_shape": list(features["grid_shape"]), "smoke_passed": True}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def evaluate(config, checkpoint):
    split = prepare(config)
    device = _device(config)
    adapter = build_adapter(config)
    model = _classifier(adapter, config).to(device)
    state = _load_checkpoint(checkpoint, model, config, _split_hash(split))
    if "threshold" not in state:
        raise ValueError("Evaluation requires best.pt with a validation-selected threshold")
    run = _new_run(config, "_evaluation")
    test = predict(model, _loader(split, "test", adapter, config), device)
    save_evaluation(test, run, threshold=state["threshold"],
                    bootstrap_samples=config["evaluation"]["bootstrap_samples"], seed=config["seed"])
    _json(run / "config.json", config)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Model JSON configuration")
    parser.add_argument("--action", choices=("prepare", "smoke", "train", "evaluate"), default="train")
    parser.add_argument("--mode", choices=("linear_probe", "lora"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", help="Trained best.pt for evaluation")
    parser.add_argument("--resume", help="last.pt to resume with identical training settings")
    args = parser.parse_args(argv)
    config = load_config(args.config, {"device": args.device, "mode": args.mode, "seed": args.seed})
    if args.action == "prepare":
        return prepare(config)
    if args.action == "smoke":
        return smoke(config)
    if args.action == "evaluate":
        if not args.checkpoint:
            parser.error("--action evaluate requires --checkpoint")
        return evaluate(config, args.checkpoint)
    return run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
