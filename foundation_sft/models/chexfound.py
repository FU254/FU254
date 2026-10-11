"""Official CheXFound ViT-L/16 encoder, loaded from local source and weights.

Architecture and preprocessing follow RPIDIAL/CheXFound. The adapter deliberately
does not call its distributed evaluation setup or download checkpoints. See
https://github.com/RPIDIAL/CheXFound and its notebooks/chexfound_example.ipynb.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .common import (
    BackboneAdapter,
    BackboneFeatures,
    PreprocessSpec,
    load_backbone_weights,
    load_checkpoint,
    sha256_file,
)


OFFICIAL_REPOSITORY = "https://github.com/RPIDIAL/CheXFound"
VERIFIED_SOURCE_REVISION = "7e288d06532f24abedb7d00781185c36727d9d69"
_CONFIG_RELATIVE_PATH = "chexfound/configs/train/vitl16_ibot333_highres512.yaml"
_TRAINING_HEAD_PREFIXES = ("dino_head.", "ibot_head.")


def _import_official_model(repo_dir: Path):
    """Use the real CheXFound package and refuse an unrelated installed copy."""
    package_dir = repo_dir / "chexfound"
    source_file = package_dir / "models" / "vision_transformer.py"
    if not source_file.is_file() or not (package_dir / "layers" / "attention.py").is_file():
        raise FileNotFoundError(
            f"repo_dir must contain the official CheXFound source: {source_file}"
        )
    existing = sys.modules.get("chexfound")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if not existing_file or Path(existing_file).resolve().parent != package_dir:
            raise RuntimeError(
                "A different chexfound package is already imported; use a separate "
                "process for the selected official checkout."
            )
    search_path = str(repo_dir)
    sys.path.insert(0, search_path)
    try:
        module = importlib.import_module("chexfound.models.vision_transformer")
    except ImportError as error:
        raise ImportError(
            "Could not import the official CheXFound encoder. Install only its "
            "required compatible inference packages, without replacing working "
            "PyTorch/CUDA packages or installing all pretraining dependencies."
        ) from error
    finally:
        # Remove exactly the temporary entry; existing caller paths remain intact.
        if sys.path and sys.path[0] == search_path:
            sys.path.pop(0)
        else:
            sys.path.remove(search_path)
    if Path(module.__file__).resolve() != source_file:
        raise RuntimeError("The imported CheXFound encoder does not come from repo_dir")
    return module


def _read_architecture(repo_dir: Path, config_file: Path) -> tuple[dict, int]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("CheXFound architecture configuration requires PyYAML") from error
    defaults_file = repo_dir / "chexfound/configs/ssl_default_config.yaml"
    if not config_file.is_file() or not defaults_file.is_file():
        raise FileNotFoundError(
            f"CheXFound needs its release configuration and defaults: {config_file}"
        )
    defaults = yaml.safe_load(defaults_file.read_text(encoding="utf-8"))
    release = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(defaults, dict) or not isinstance(release, dict):
        raise ValueError("CheXFound configuration must be a YAML mapping")
    student = {**defaults.get("student", {}), **release.get("student", {})}
    crops = {**defaults.get("crops", {}), **release.get("crops", {})}
    if student.get("arch", "").removesuffix("_memeff") != "vit_large":
        raise ValueError("This adapter supports the released CheXFound ViT-L encoder")
    image_size = int(crops["global_crops_size"])
    patch_size = int(student["patch_size"])
    if patch_size != 16 or image_size <= 0 or image_size % patch_size:
        raise ValueError("CheXFound expects ViT-L/16 with a positive patch-aligned image size")
    required = (
        "layerscale", "ffn_layer", "qkv_bias", "proj_bias", "ffn_bias",
        "num_register_tokens", "interpolate_offset", "interpolate_antialias",
    )
    if any(name not in student for name in required):
        raise ValueError("The provided CheXFound defaults lack required architecture fields")
    architecture = {
        "img_size": image_size,
        "patch_size": patch_size,
        "init_values": student["layerscale"],
        "ffn_layer": student["ffn_layer"],
        # Flattening changes only FSDP wrapping, not the encoder mathematics.
        "block_chunks": 0,
        "qkv_bias": student["qkv_bias"],
        "proj_bias": student["proj_bias"],
        "ffn_bias": student["ffn_bias"],
        "num_register_tokens": int(student["num_register_tokens"]),
        "interpolate_offset": student["interpolate_offset"],
        "interpolate_antialias": student["interpolate_antialias"],
    }
    return architecture, image_size


def _normalize_teacher_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the teacher wrapper and documented FSDP chunk indices only."""
    normalized = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith(_TRAINING_HEAD_PREFIXES):
            continue
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        # The official example maps blocks.<chunk>.<global-index> to blocks.<index>.
        key = re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", key)
        if key in normalized:
            raise ValueError(f"Duplicate backbone parameter after checkpoint normalization: {key}")
        normalized[key] = value
    if not normalized:
        raise ValueError("The checkpoint contains no CheXFound backbone parameters")
    return normalized


class CheXFoundAdapter(BackboneAdapter):
    def __init__(self, encoder: nn.Module, preprocess: PreprocessSpec, provenance: dict):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(encoder.embed_dim)
        self.patch_size = int(encoder.patch_size)
        self.preprocess = preprocess
        self.provenance = dict(provenance)

    def forward_features(self, images: torch.Tensor) -> BackboneFeatures:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [B,3,H,W]")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("CheXFound image dimensions must be divisible by its patch size")
        # The upstream xformers path does not support CPU. Its native path is
        # available when xformers is absent or XFORMERS_DISABLED=1 at import time.
        if not images.is_cuda:
            attention_module = importlib.import_module("chexfound.layers.attention")
            if attention_module.XFORMERS_AVAILABLE:
                raise RuntimeError(
                    "For CPU CheXFound validation, set XFORMERS_DISABLED=1 before "
                    "starting Python; the upstream accelerated path requires CUDA."
                )
        features = self.encoder.forward_features(images)
        result = BackboneFeatures(
            global_features=features["x_norm_clstoken"],
            local_features=features["x_norm_patchtokens"],
            grid_shape=(images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size),
        )
        return result.validate()

    def lora_target_names(self) -> list[str]:
        pattern = re.compile(r"^encoder\.blocks\.\d+\.attn\.qkv$")
        names = [
            name for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and pattern.fullmatch(name)
        ]
        if len(names) != int(self.encoder.n_blocks):
            raise RuntimeError("Could not identify every CheXFound attention QKV projection")
        return names


def build_chexfound(config: dict[str, Any]) -> BackboneAdapter:
    if not config.get("repo_dir") or not config.get("checkpoint"):
        raise ValueError(
            "CheXFound requires repo_dir for its official checkout and checkpoint "
            "for the actual released teacher weights; automatic downloads are disabled"
        )
    repo_dir = Path(config["repo_dir"]).expanduser().resolve()
    config_file = Path(config.get("config_file") or repo_dir / _CONFIG_RELATIVE_PATH).expanduser().resolve()
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CheXFound checkpoint not found: {checkpoint}")
    architecture, image_size = _read_architecture(repo_dir, config_file)
    source = _import_official_model(repo_dir)
    encoder = source.vit_large(**architecture)
    state = load_checkpoint(checkpoint, state_key=config.get("state_key", "teacher"))
    state = _normalize_teacher_state(state)
    load_backbone_weights(encoder, state, ignored_prefixes=())
    provenance = {
        "model": "chexfound",
        "official_repository": OFFICIAL_REPOSITORY,
        "verified_source_revision": VERIFIED_SOURCE_REVISION,
        "repo_dir": str(repo_dir),
        "source_sha256": sha256_file(repo_dir / "chexfound/models/vision_transformer.py"),
        "config_file": str(config_file),
        "config_sha256": sha256_file(config_file),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_key": config.get("state_key", "teacher"),
        "architecture": architecture,
        "checkpoint_identity_note": "Local weights must be obtained from the official release.",
    }
    preprocess = PreprocessSpec(
        image_size=image_size,
        resize_mode="shortest_edge_center_crop",
        resize_shortest_edge=image_size,
        interpolation="bicubic",
    )
    return CheXFoundAdapter(encoder, preprocess, provenance)
