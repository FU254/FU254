"""Restore the three declared, validation-selected teachers for equal-weight KD.

Only a declared checkpoint or the exact declared suite job is considered. This
module never searches runs by recency or by validation/test score. Imports of
Torch and the training stack are delayed so planning and source checks work on
machines that do not have the training dependencies installed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


TEACHER_MODELS = ("rad_dino", "chexfound", "chexzero")
_IMAGE_SUITE = "runs/foundation_sft/suites/suite_20261013T024024Z_92ac596d"
_VLM_SUITE = "runs/foundation_sft_vlm/suites/suite_20261015T045005Z_2162f5a1"
DEFAULT_TEACHER_SOURCES = tuple(
    {"model": model, "mode": "lora", "seed": 42,
     "suite": _VLM_SUITE if model == "chexzero" else _IMAGE_SUITE,
     "checkpoint": None, "weight": 1.0 / 3.0}
    for model in TEACHER_MODELS
)


def _path(value, project):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Teacher source paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(project) / path).resolve()


def _validate_source(source):
    if not isinstance(source, dict) or source.get("model") not in TEACHER_MODELS:
        raise ValueError("Teacher must be one of rad_dino, chexfound and chexzero")
    if source.get("mode") != "lora" or type(source.get("seed")) is not int or source["seed"] != 42:
        raise ValueError("The selected teacher protocol requires mode=lora and seed=42")
    weight = source.get("weight", 1.0 / 3.0)
    if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isclose(
            weight, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Each selected teacher must have weight 1/3")


def _identity(value, source, description):
    if not isinstance(value, dict) or any(value.get(name) != source[name]
                                          for name in ("model", "mode", "seed")):
        raise ValueError(f"{description} does not match the declared teacher model/mode/seed")


def resolve_teacher_checkpoint(source, project):
    """Resolve one source without choosing among historical runs.

    An explicit ``checkpoint`` takes precedence. Otherwise only the precise
    ``logs/<model>_lora_seed42_train.log`` and matching training receipt under
    the declared ``suite`` are read. If they name different runs, the caller
    must specify a checkpoint explicitly instead of having a run chosen here.
    """
    _validate_source(source)
    if source.get("checkpoint") is not None:
        checkpoint = _path(source["checkpoint"], project)
    else:
        if not source.get("suite"):
            raise ValueError("Declare a teacher checkpoint or a specific source suite")
        suite = _path(source["suite"], project)
        job = f"{source['model']}_lora_seed42"
        log = suite / "logs" / f"{job}_train.log"
        receipt = suite / "receipts" / f"{job}_train.json"
        directories = set()
        if log.is_file():
            directories.update(
                _path(match.group(1).strip(), project)
                for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if (match := re.fullmatch(r"Run directory:\s*(.+)", line.strip()))
            )
        if receipt.is_file():
            with receipt.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            _identity(saved, source, "Training receipt")
            if saved.get("action") != "train" or not saved.get("run_dir"):
                raise ValueError("Teacher receipt must record a successful training run_dir")
            directories.add(_path(saved["run_dir"], project))
        if not log.is_file() and not receipt.is_file():
            raise FileNotFoundError(f"Declared teacher job has no training log or receipt: {suite / job}")
        if len(directories) != 1:
            raise ValueError("Declared teacher job must identify one unambiguous Run directory; "
                             "set checkpoint explicitly if the source was recovered or rerun")
        checkpoint = directories.pop() / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected teacher best checkpoint is missing: {checkpoint}")
    return checkpoint


def _runtime():
    import torch
    from . import train as supervised
    from .models import build_adapter
    return torch, supervised, build_adapter


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def _validate_payload(payload, source):
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Teacher must use the original FoundationClassifier checkpoint format 1")
    _identity(payload.get("config"), source, "Teacher checkpoint")
    if (not _bounded_number(payload.get("threshold"))
            or not _bounded_number(payload.get("best_val_patient_roc_auc"))):
        raise ValueError("Teacher must be a completed validation-selected best checkpoint with threshold")
    if not isinstance(payload.get("model_metadata"), dict):
        raise ValueError("Teacher checkpoint is missing its model metadata")
    if not isinstance(payload.get("cohort_hash"), str) or not payload["cohort_hash"]:
        raise ValueError("Teacher checkpoint is missing its frozen cohort hash")


def load_teacher(source, project, device="cpu", cache_dir=None):
    """Return ``(teacher, config, split, payload, info)`` after strict restore.

    The original model configuration is used to reconstruct the frozen source
    encoder and its trained LoRA/head. Freezing occurs after the legacy loader
    verifies training metadata, which includes trainable parameter counts.
    """
    checkpoint = resolve_teacher_checkpoint(source, project)
    before_hash = _sha256_file(checkpoint)
    torch, supervised, build_adapter = _runtime()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _validate_payload(payload, source)
    saved_config = deepcopy(payload["config"])
    config = deepcopy(saved_config)
    manifest = _path(config.get("split_manifest"), project)
    if not manifest.is_file():
        raise FileNotFoundError(f"Teacher frozen split is missing: {manifest}; do not create a new split")
    config["split_manifest"] = str(manifest)
    config["device"] = str(device)
    if cache_dir is not None:
        config["cache_dir"] = str(_path(cache_dir, project))
    split = supervised.prepare(config)
    cohort_hash = supervised._split_hash(split)
    if payload["cohort_hash"] != cohort_hash:
        raise ValueError("Selected teacher checkpoint belongs to a different frozen patient split")
    teacher = supervised._classifier(build_adapter(config), config)
    expected = teacher.trainable_state_dict()
    state = payload.get("model")
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Teacher trainable state has missing or unexpected parameter keys")
    for name, target in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all().item()):
            raise ValueError(f"Teacher parameter has incompatible shape/dtype or non-finite values: {name}")
    restored = supervised._load_checkpoint(checkpoint, teacher, config, cohort_hash)
    # Protect the two safe loads against a checkpoint being replaced mid-restore.
    if _sha256_file(checkpoint) != before_hash:
        raise ValueError("Teacher checkpoint changed while it was being restored")
    if restored.get("model_metadata") != payload["model_metadata"]:
        raise ValueError("Teacher checkpoint metadata changed while it was being restored")
    if any(not torch.isfinite(value).all().item() for value in teacher.state_dict().values()
           if value.is_floating_point() or value.is_complex()):
        raise ValueError("Restored teacher contains non-finite parameters or buffers")
    model_metadata = deepcopy(teacher.training_metadata())
    teacher.requires_grad_(False).eval()
    if any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()):
        raise ValueError("Selected teacher must be frozen with no residual gradients")
    teacher = teacher.to(device)
    info = {
        "model": source["model"], "mode": source["mode"], "seed": source["seed"],
        "weight": 1.0 / 3.0, "checkpoint": str(checkpoint),
        "checkpoint_sha256": before_hash, "cohort_hash": cohort_hash,
        "config": saved_config, "model_metadata": model_metadata,
        "split_metadata": deepcopy(split.attrs.get("split_metadata", {})),
        "threshold": payload["threshold"],
        "best_val_patient_roc_auc": payload["best_val_patient_roc_auc"],
        "source": deepcopy(source),
    }
    return teacher, config, split, payload, info


def validate_teacher_cohorts(loaded, anchor="rad_dino"):
    """Require identical sample/patient assignments, task and source provenance.

    ``loaded`` contains the five-tuples returned by :func:`load_teacher`. The
    RAD-DINO split is returned as the canonical cohort for the student. This
    comparison does not access predictions, evaluation scores or test metrics.
    """
    loaded = list(loaded)
    by_model = {}
    for item in loaded:
        if not isinstance(item, (tuple, list)) or len(item) != 5:
            raise ValueError("Expected teacher records returned by load_teacher")
        name = item[4].get("model")
        if name in by_model:
            raise ValueError("Each selected teacher must appear exactly once")
        by_model[name] = item
    if set(by_model) != set(TEACHER_MODELS) or anchor not in by_model:
        raise ValueError("The equal-weight experiment requires rad_dino, chexfound and chexzero")
    reference = by_model[anchor]
    config, split, payload = reference[1], reference[2], reference[3]
    metadata = split.attrs.get("split_metadata", {})
    required_columns = ("sample_id", "patient_id", "path", "label", "split")
    optional_columns = ("accession", "dicom_size", "dicom_mtime_ns", "patient_id_source",
                        "label_source_signature", "label_definition", "label_definition_signature")
    if any(name not in split.columns for name in required_columns):
        raise ValueError("Teacher cohort is missing required sample/patient/label/split columns")
    columns = list(required_columns) + [name for name in optional_columns if name in split.columns]
    records = split.loc[:, columns].sort_values("sample_id", kind="stable").reset_index(drop=True)
    for name, item in by_model.items():
        candidate, candidate_split, candidate_payload = item[1], item[2], item[3]
        if candidate_payload.get("cohort_hash") != payload.get("cohort_hash"):
            raise ValueError(f"Teacher {name} has a different frozen cohort fingerprint")
        if candidate_split.attrs.get("split_metadata", {}) != metadata:
            raise ValueError(f"Teacher {name} has different split/label-source provenance")
        if (any(column not in candidate_split.columns for column in columns)
                or any((column in candidate_split.columns) != (column in split.columns)
                       for column in optional_columns)):
            raise ValueError(f"Teacher {name} has different cohort record fields")
        candidate_records = candidate_split.loc[:, columns].sort_values(
            "sample_id", kind="stable").reset_index(drop=True)
        if not candidate_records.equals(records):
            raise ValueError(f"Teacher {name} has different patient assignments, labels or source images")
        for field in ("seed", "split_seed", "split_ratios", "data"):
            if candidate.get(field) != config.get(field):
                raise ValueError(f"Teacher {name} has different task/data/split settings: {field}")
        for field in ("label_mapping", "positive_label", "negative_label", "task", "evaluation"):
            if candidate.get(field) != config.get(field):
                raise ValueError(f"Teacher {name} has different task/label/evaluation settings: {field}")
    return split
