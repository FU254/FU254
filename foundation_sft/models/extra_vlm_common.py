"""Local asset checks shared by the additional chest-image encoders."""

from pathlib import Path

import torch

from .common import sha256_file


def local_asset(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {key} to a local file before running {config['model']}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {config['model']} asset: {path}. See EXTRA_VLM_SETUP.md")
    checksum = sha256_file(path)
    expected = config.get(key + "_sha256")
    if expected is not None and checksum != expected:
        raise ValueError(f"{config['model']} {key} SHA256 differs from the configured file")
    return path, checksum


def strict_tensors(module, state, label):
    """Check all names, shapes, types and finite values before mutating a model."""
    expected = module.state_dict()
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f"{label} mismatch: missing={sorted(missing)[:12]}, unexpected={sorted(extra)[:12]}")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype):
            raise ValueError(f"{label} shape/dtype mismatch: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"{label} non-finite tensor: {name}")
    module.load_state_dict(state, strict=True)


def require_linear_probe(config):
    if config.get("mode", "linear_probe") != "linear_probe":
        raise ValueError(f"{config['model']} currently supports linear_probe only. "
                         "Its mixed CNN/Transformer structure has no verified attention-LoRA training path.")
