"""JSON configuration with a shared base and explicit local model assets."""

from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULTS = {
    "mode": "lora", "device": "cuda:3", "seed": 42,
    "output_root": "runs/foundation_sft",
    "split_manifest": "runs/foundation_sft/shared_patient_split.csv",
    "split_seed": 42, "split_ratios": [0.7, 0.1, 0.2],
    "data": {
        "excel_path": "/home/cfu/肺功能万能表最终版1_fev1fvc差值标签(1).xlsx",
        "dicom_dir": "/data/users/cfu/DR数据_按文件10份-筛选合成",
        "label_column": "阻塞性通气障碍", "accession_column": "放射编号",
        "patient_column": "auto", "patient_id_source": "auto",
        "dicom_patient_fallback": True, "allow_name_patient_id": False,
        "label_definition": "Existing Excel obstruction label; confirm LLN reference equation and bronchodilator status before formal analysis",
    },
    "training": {
        "epochs": 300, "patience": 30, "batch_size": 8, "accumulation_steps": 8,
        "num_workers": 4, "head_lr": 1e-4, "lora_lr": 1e-4,
        "weight_decay": 1e-4, "minimum_lr": 1e-7,
        "focal_gamma": 2.0, "focal_scale": 0.8,
        "lora_rank": 8, "lora_alpha": 16.0, "hidden_dims": [384, 192],
        "amp": True, "gradient_clip": 1.0,
        "horizontal_flip_probability": 0.0, "rotation_degrees": 10.0,
    },
    "evaluation": {"bootstrap_samples": 1000},
}


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path, seen):
    path = path.expanduser().resolve()
    if path in seen:
        raise ValueError("Circular base_config reference")
    seen.add(path)
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    base = config.pop("base_config", None)
    if base:
        config = _merge(_read(path.parent / base, seen), config)
    return config


def load_config(path, overrides=None):
    config = _merge(DEFAULTS, _read(Path(path), set()))
    config = _merge(config, {k: v for k, v in (overrides or {}).items() if v is not None})
    from .models import SUPPORTED_MODELS
    if config.get("model") not in SUPPORTED_MODELS:
        raise ValueError(f"Config must specify a model from {SUPPORTED_MODELS}")
    if config["mode"] not in {"linear_probe", "lora"}:
        raise ValueError("mode must be linear_probe or lora")
    for name in ("epochs", "patience", "batch_size", "accumulation_steps", "lora_rank"):
        value = config["training"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    workers = config["training"]["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("training.num_workers must be a nonnegative integer")
    for name in ("head_lr", "lora_lr", "lora_alpha", "focal_scale", "gradient_clip"):
        value = config["training"][name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be positive and finite")
    for name in ("weight_decay", "minimum_lr", "focal_gamma"):
        value = config["training"][name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{name} must be nonnegative and finite")
    hidden = config["training"]["hidden_dims"]
    if not isinstance(hidden, list) or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in hidden):
        raise ValueError("training.hidden_dims must be a list of positive integers")
    if config["evaluation"]["bootstrap_samples"] < 0:
        raise ValueError("evaluation.bootstrap_samples must be nonnegative")
    return config
