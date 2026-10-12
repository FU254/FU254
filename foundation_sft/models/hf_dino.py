"""Local Hugging Face DINOv2 encoders for RAD-DINO and RayDINO.

The archive must contain the actual medical-model weights, config.json and
preprocessor_config.json. A generic Facebook DINOv2 archive is not a replacement
for either medical model. This module never downloads weights or executes
checkpoint-provided Python code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, sha256_file


_MODEL_TYPES = {"dinov2", "dinov2_with_registers"}
_INTERPOLATION = {
    0: "nearest",
    1: "lanczos",
    2: "bilinear",
    3: "bicubic",
    4: "box",
    5: "hamming",
}


def _square_size(value: Any, *, field: str) -> int:
    if isinstance(value, int):
        size = value
    elif isinstance(value, dict) and "height" in value and "width" in value:
        if int(value["height"]) != int(value["width"]):
            raise ValueError(f"{field} must be square for the shared CXR dataset: {value}")
        size = int(value["height"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if int(value[0]) != int(value[1]):
            raise ValueError(f"{field} must be square: {value}")
        size = int(value[0])
    else:
        raise ValueError(f"Cannot determine square {field} from {value!r}")
    if size < 1:
        raise ValueError(f"{field} must be positive")
    return size


def _channel_values(value: Any, *, field: str) -> tuple[float, float, float]:
    if isinstance(value, (float, int)):
        result = (float(value),) * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        result = tuple(float(item) for item in value)
    else:
        raise ValueError(f"The model processor must provide three-channel {field}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Non-finite processor {field}")
    if field == "image_std" and any(item <= 0 for item in result):
        raise ValueError("Processor image_std must be positive")
    return result


def _processor_spec(processor: Any, encoder_config: Any) -> PreprocessSpec:
    """Translate the saved official processor without substituting EVA-X rules."""
    # The shared DICOM decoder supplies intensities in [0, 1]. An HF processor
    # that rescales uint8 by 1/255 consequently needs no second rescaling here.
    if not getattr(processor, "do_rescale", True) or not math.isclose(
        float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
        1.0 / 255.0,
        rel_tol=1e-6,
    ):
        raise ValueError(
            "This HF processor does not use the supported uint8-to-[0,1] "
            "rescaling convention; supply a matching preprocessing adapter."
        )
    if getattr(processor, "do_normalize", True):
        mean = _channel_values(processor.image_mean, field="image_mean")
        std = _channel_values(processor.image_std, field="image_std")
    else:
        mean, std = (0.0,) * 3, (1.0,) * 3

    if not getattr(processor, "do_resize", True):
        raise ValueError("HF processors without resizing need a dedicated dataset adapter")
    size = getattr(processor, "size", None)
    crop = bool(getattr(processor, "do_center_crop", False))
    if crop:
        image_size = _square_size(processor.crop_size, field="processor.crop_size")
        if not isinstance(size, dict) or "shortest_edge" not in size:
            raise ValueError(
                "Center-crop processors must specify size.shortest_edge; "
                "fixed-height resize followed by crop is not silently approximated"
            )
        resize_shortest_edge = int(size["shortest_edge"])
        if resize_shortest_edge < image_size:
            raise ValueError("A shortest-edge resize smaller than the crop needs padding support")
        resize_mode = "shortest_edge_center_crop"
    elif isinstance(size, dict) and "shortest_edge" in size:
        raise ValueError(
            "A shortest-edge-only processor produces variable rectangular shapes; "
            "its batching policy must be specified instead of stretching to a square"
        )
    else:
        image_size = _square_size(size, field="processor.size")
        resize_shortest_edge = None
        resize_mode = "resize"
    interpolation = _INTERPOLATION.get(int(getattr(processor, "resample", 3)))
    if interpolation is None:
        raise ValueError(f"Unsupported HF resampling mode: {processor.resample!r}")
    return PreprocessSpec(
        image_size=image_size,
        mean=mean,
        std=std,
        resize_mode=resize_mode,
        resize_shortest_edge=resize_shortest_edge,
        interpolation=interpolation,
    )


class HFDinoAdapter(BackboneAdapter):
    """Expose normalized CLS and spatial patch features from real HF DINOv2."""

    def __init__(
        self,
        encoder: nn.Module,
        preprocess: PreprocessSpec,
        provenance: dict[str, Any],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = int(encoder.config.hidden_size)
        self.preprocess = preprocess
        self.provenance = dict(provenance)
        model_type = encoder.config.model_type
        if model_type not in _MODEL_TYPES:
            raise ValueError(f"Expected a DINOv2 encoder; got model_type={model_type!r}")
        if int(getattr(encoder.config, "num_channels", 3)) != 3:
            raise ValueError("The shared CXR pipeline requires a three-channel encoder")
        patch_size = encoder.config.patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = tuple(int(value) for value in patch_size)
        if len(self.patch_size) != 2 or min(self.patch_size) < 1:
            raise ValueError(f"Invalid encoder patch_size: {patch_size!r}")
        self.num_register_tokens = (
            int(encoder.config.num_register_tokens)
            if model_type == "dinov2_with_registers"
            else 0
        )
        if model_type == "dinov2" and int(getattr(encoder.config, "num_register_tokens", 0)):
            raise ValueError(
                "A standard Dinov2Model cannot load register-token weights. "
                "Use a correctly converted dinov2_with_registers archive."
            )

    def forward_features(self, images: torch.Tensor) -> BackboneFeatures:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [batch, 3, height, width]")
        grid = (
            images.shape[-2] // self.patch_size[0],
            images.shape[-1] // self.patch_size[1],
        )
        if min(grid) < 1:
            raise ValueError("Image dimensions must each contain at least one patch")
        outputs = self.encoder(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        prefix = 1 + self.num_register_tokens
        expected = prefix + grid[0] * grid[1]
        if tokens.ndim != 3 or tokens.shape[1] != expected:
            raise RuntimeError(
                f"Encoder returned {tuple(tokens.shape)} instead of "
                f"[B, {expected}, D] for grid {grid} and {prefix} prefix tokens"
            )
        return BackboneFeatures(
            global_features=tokens[:, 0],
            local_features=tokens[:, prefix:],
            grid_shape=grid,
        )

    def lora_target_names(self) -> list[str]:
        pattern = re.compile(
            r"^encoder\.encoder\.layer\.\d+\.attention\.attention\.(query|key|value)$"
        )
        names = [
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear) and pattern.fullmatch(name)
        ]
        expected = int(self.encoder.config.num_hidden_layers) * 3
        if len(names) != expected:
            raise RuntimeError(
                f"Expected {expected} DINOv2 Q/K/V projections, found {len(names)}; "
                "the supplied encoder needs an architecture-specific LoRA adapter"
            )
        return names


def build_hf_dino(config: dict[str, Any]) -> BackboneAdapter:
    """Load an explicitly supplied local medical DINOv2 model archive."""
    model_name = config.get("model")
    if model_name not in {"rad_dino", "ray_dino"}:
        raise ValueError("HF DINO adapter model must be 'rad_dino' or 'ray_dino'")
    if not config.get("model_dir"):
        raise ValueError(
            f"{model_name} requires model_dir pointing to a complete local HF "
            "snapshot of its official weights; automatic downloading is disabled"
        )
    model_dir = Path(config["model_dir"]).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HF model directory does not exist: {model_dir}")
    for filename in ("config.json", "preprocessor_config.json"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Required HF snapshot file is missing: {model_dir / filename}")
    source = config.get("checkpoint_source") or config.get("source")
    if model_name == "ray_dino" and not source:
        raise ValueError(
            "RayDINO requires checkpoint_source identifying its actual official "
            "release and any HF conversion. A generic DINOv2 archive is not RayDINO."
        )
    source = source or "microsoft/rad-dino"
    if str(source).lower().startswith("facebook/dinov2"):
        raise ValueError(
            "Generic Facebook DINOv2 weights are not RAD-DINO or RayDINO. "
            "Supply the actual medical-pretrained model archive."
        )
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_type = raw_config.get("model_type")
    if model_type not in _MODEL_TYPES:
        raise ValueError(
            f"Unsupported HF model_type={model_type!r}. Expected a real DINOv2 "
            "encoder archive; custom RayDINO releases require their own verified conversion."
        )
    try:
        from transformers import AutoConfig, AutoImageProcessor
    except ImportError as error:
        raise ImportError(
            "Install the pinned foundation-SFT Transformers dependency. "
            "Do not upgrade the working PyTorch/CUDA environment implicitly."
        ) from error
    local_options = {"local_files_only": True, "trust_remote_code": False}
    try:
        encoder_config = AutoConfig.from_pretrained(str(model_dir), **local_options)
    except (ValueError, KeyError) as error:
        raise ValueError(
            f"Installed Transformers cannot read model_type={model_type!r}; "
            "register-token models require Transformers 4.48.3 or a verified compatible version."
        ) from error
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir), use_fast=False, **local_options
    )
    preprocess = _processor_spec(processor, encoder_config)
    # AutoModel.from_pretrained enumerates unrelated model configurations in
    # Recent Transformers, importing a timm wrapper incompatible with EVA-X's
    # pinned timm 0.4.12. The two concrete verified classes avoid that dependency.
    if model_type == "dinov2":
        from transformers import Dinov2Model as encoder_type
    else:
        from transformers import Dinov2WithRegistersModel as encoder_type
    encoder, loading_info = encoder_type.from_pretrained(
        str(model_dir),
        config=encoder_config,
        output_loading_info=True,
        attn_implementation="eager",
        **local_options,
    )
    problems = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    if problems:
        raise ValueError(
            "The checkpoint does not exactly match its declared DINOv2 encoder; "
            f"refusing partially/randomly initialized weights: {problems}"
        )
    provenance = {
        "model": model_name,
        "model_dir": str(model_dir),
        "checkpoint_source": str(source),
        "model_type": model_type,
        "official_source_verified": False,
        "source_note": (
            "Architecture and complete parameter loading are checked locally. "
            "Medical pretraining identity must be verified against the official release."
        ),
        "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256(
            (model_dir / "preprocessor_config.json").read_bytes()
        ).hexdigest(),
        "weights_sha256": {
            path.name: sha256_file(path)
            for path in sorted(model_dir.glob("*.safetensors"))
            + sorted(model_dir.glob("pytorch_model*.bin"))
        },
        "attention_implementation": "eager",
    }
    return HFDinoAdapter(encoder, preprocess, provenance)
