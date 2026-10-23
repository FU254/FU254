"""Single-CLS DeiT-Tiny/16 student with full or LoRA training, always offline.

Fresh training::

    model = build_vit_tiny_student("deit_tiny_patch16_224-a1311bcf.pth")

    # Optional: freeze the ImageNet base and train QKV LoRA plus a linear head.
    model = build_vit_tiny_student("deit_tiny_patch16_224-a1311bcf.pth", mode="lora")

    # Explicitly train from random weights; no source checkpoint is accessed.
    model = build_vit_tiny_student(random_seed=42)

Evaluation/resume from a saved student needs no original ImageNet file::

    model = build_vit_tiny_student(
        initialization=saved_metadata["initialization"])
    model.load_trainable_state_dict(saved_student_state)

The second form cannot run forward until the complete student state is loaded.
Save ``training_metadata()`` and the complete ``trainable_state_dict()`` together.
For LoRA reconstruction, also pass the saved mode, LoRA rank and alpha to the
factory. Both modes save the entire encoder, so neither needs its source file.
Random initialization is recorded separately from ImageNet initialization and
does not change the caller's Torch random-number state.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import re

import torch
from torch import nn

from .common import BackboneFeatures, PreprocessSpec, inject_lora, sha256_file


ARCHITECTURE = "deit_tiny_patch16_224"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
CHECKPOINT_SHA256_PREFIX = "a1311bcf"
FEATURE_DIM = 192
IMAGE_SIZE = 224
GRID_SHAPE = (14, 14)


def _validate_training(mode, rank, alpha):
    if mode not in ("full_finetune", "lora"):
        raise ValueError("Student mode must be full_finetune or lora")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("Student LoRA rank must be a positive integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Student LoRA alpha must be positive and finite")


def _validate_initialization(initialization):
    """Validate saved source metadata without requiring its old filesystem path."""
    if not isinstance(initialization, Mapping):
        raise ValueError("Student initialization must be saved initialization metadata")
    if initialization.get("initialization_mode") == "random" or initialization.get("pretraining") == "none":
        required = {
            "architecture": ARCHITECTURE,
            "initialization_mode": "random",
            "pretraining": "none",
            "source_num_classes": 1,
            "variant": "single_cls",
        }
        if (set(initialization) != {*required, "seed"}
                or type(initialization.get("source_num_classes")) is not int
                or any(initialization.get(key) != value for key, value in required.items())):
            raise ValueError("Random student initialization metadata differs from the single-CLS architecture")
        _validate_random_seed(initialization.get("seed"))
        return deepcopy(dict(initialization))
    required = {
        "architecture": ARCHITECTURE,
        "official_checkpoint_url": CHECKPOINT_URL,
        "pretraining": "ImageNet-1k",
        "source_num_classes": 1000,
        "variant": "single_cls",
    }
    if any(initialization.get(key) != value for key, value in required.items()):
        raise ValueError("Student initialization does not describe the official single-CLS DeiT-Tiny")
    checksum = initialization.get("checkpoint_sha256")
    if (not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or not checksum.startswith(CHECKPOINT_SHA256_PREFIX)):
        raise ValueError("Student initialization lacks the official checkpoint's full SHA256")
    return deepcopy(dict(initialization))


def _validate_random_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    return seed


def _strict_load(module, state, description):
    """Check every key/tensor before copying any state into a live model."""
    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise ValueError(f"{description} must be a tensor state dictionary")
    expected = module.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f"{description} mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"{description} tensor shape/dtype mismatch: {name}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{description} contains a non-finite tensor: {name}")
    module.load_state_dict(state, strict=True)


def _new_encoder(num_classes):
    try:
        import timm
    except ImportError as error:
        raise ImportError("ViT-Tiny student requires timm (validated with timm==0.9.0)") from error
    encoder = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=num_classes)
    patch = encoder.patch_embed
    if (getattr(encoder, "num_features", None) != FEATURE_DIM
            or getattr(encoder, "num_prefix_tokens", None) != 1
            or getattr(encoder, "global_pool", None) != "token"
            or getattr(encoder, "dist_token", None) is not None
            or getattr(encoder, "head_dist", None) is not None
            or tuple(patch.patch_size) != (16, 16)
            or tuple(patch.img_size) != (IMAGE_SIZE, IMAGE_SIZE)
            or tuple(patch.grid_size) != GRID_SHAPE
            or len(encoder.blocks) != 12
            or any(block.attn.num_heads != 3 for block in encoder.blocks)
            or not isinstance(encoder.head, nn.Linear)
            or encoder.head.in_features != FEATURE_DIM
            or encoder.head.out_features != num_classes):
        raise ValueError("Expected the native single-CLS DeiT-Tiny/16 architecture")
    return encoder


class ViTTinyStudent(nn.Module):
    """One linear output head, with full encoder tuning or QKV LoRA tuning."""

    def __init__(self, encoder, initialization, *, ready, mode="full_finetune", rank=8, alpha=16.0):
        super().__init__()
        _validate_training(mode, rank, alpha)
        self.encoder = encoder
        self.feature_dim = FEATURE_DIM
        self.preprocess = PreprocessSpec(IMAGE_SIZE)
        self.initialization = _validate_initialization(initialization)
        self._ready = ready
        self.training_mode = mode
        self.lora = None
        if mode == "full_finetune":
            self.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)
            targets = inject_lora(self, rank=rank, alpha=alpha)
            self.encoder.head.requires_grad_(True)
            self.lora = {"rank": rank, "alpha": float(alpha), "targets": targets}

    def lora_target_names(self):
        return [f"encoder.blocks.{index}.attn.qkv" for index in range(12)]

    def forward(self, images, return_features=False):
        if not self._ready:
            raise RuntimeError("Load the complete student state before using a reconstructed student")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("ViT-Tiny student expects images with shape [B,3,224,224]")
        # The native timm forward already applies the final encoder norm.
        # Only the first token is CLS; the other 196 are real spatial patches.
        tokens = self.encoder.forward_features(images)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (images.shape[0], 197, FEATURE_DIM):
            raise ValueError("ViT-Tiny encoder must return normalized [B,197,192] tokens")
        features = BackboneFeatures(tokens[:, 0], tokens[:, 1:], GRID_SHAPE).validate()
        logits = self.encoder.forward_head(tokens)
        if tuple(logits.shape) != (images.shape[0], 1):
            raise ValueError("ViT-Tiny classification head must return one logit per image")
        logits = logits.squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def trainable_state_dict(self):
        if not self._ready:
            raise RuntimeError("Cannot save a student before loading its complete state")
        # This is the entire student (including any persistent buffers), not a
        # LoRA delta. Clone CPU tensors so an in-memory best snapshot is stable.
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def load_trainable_state_dict(self, state):
        _strict_load(self, state, "Complete student state")
        self._ready = True

    def training_metadata(self):
        metadata = {
            "model": "vit_tiny_student",
            "architecture": ARCHITECTURE,
            "mode": self.training_mode,
            "feature_dim": self.feature_dim,
            "preprocessing": asdict(self.preprocess),
            "initialization": deepcopy(self.initialization),
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
        # Preserve the exact legacy full-finetune metadata schema.
        if self.lora is not None:
            metadata["lora"] = deepcopy(self.lora)
        return metadata


def build_vit_tiny_student(checkpoint=None, *, initialization=None, random_seed=None,
                           mode="full_finetune", rank=8, alpha=16.0):
    """Build from ImageNet, explicit random weights, or saved metadata, offline.

    ``checkpoint`` is the original 1000-class single-CLS release, verified by
    SHA256 prefix and loaded strictly before replacing its head. ``initialization``
    is the dict from ``training_metadata()``; this reconstruction form requires
    a subsequent strict ``load_trainable_state_dict`` of the complete student.
    ``mode='lora'`` freezes original weights and trains QKV adapters plus the
    one-logit linear head. Pass the saved mode/rank/alpha when reconstructing it.
    ``random_seed`` creates a ready-to-train random student without reading a
    checkpoint. Its seed applies to the encoder, head and any LoRA parameters;
    it leaves the caller's CPU and CUDA random-number states unchanged.
    Exactly one of the three initialization arguments is required. There are
    no automatic downloads or fallbacks from missing ImageNet weights.
    """
    _validate_training(mode, rank, alpha)
    if sum(value is not None for value in (checkpoint, initialization, random_seed)) != 1:
        raise ValueError("Provide either an official checkpoint, saved initialization metadata, or random_seed (exactly one)")
    if random_seed is not None:
        seed = _validate_random_seed(random_seed)
        metadata = {
            "architecture": ARCHITECTURE,
            "initialization_mode": "random",
            "pretraining": "none",
            "seed": seed,
            "source_num_classes": 1,
            "variant": "single_cls",
        }
        # Build on CPU, including LoRA injection. Seeding only the CPU default
        # generator avoids changing CUDA RNG streams used by the caller.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return ViTTinyStudent(_new_encoder(1), metadata, ready=True,
                                  mode=mode, rank=rank, alpha=alpha)
    if initialization is not None:
        initialization = _validate_initialization(initialization)
        return ViTTinyStudent(_new_encoder(1), initialization, ready=False, mode=mode, rank=rank, alpha=alpha)

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Official DeiT-Tiny checkpoint not found: {path}")
    checksum = sha256_file(path)
    if not checksum.startswith(CHECKPOINT_SHA256_PREFIX):
        raise ValueError("DeiT-Tiny checkpoint SHA256 does not match its official source")
    # Never retry using pickle/weights_only=False, even for a matching filename.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("Official DeiT-Tiny checkpoint must contain its 'model' state")
    encoder = _new_encoder(1000)
    _strict_load(encoder, payload["model"], "Official DeiT-Tiny encoder and 1000-class head")
    encoder.reset_classifier(1)
    initialization = {
        "architecture": ARCHITECTURE,
        "official_checkpoint_url": CHECKPOINT_URL,
        "checkpoint": str(path),
        "checkpoint_sha256": checksum,
        "pretraining": "ImageNet-1k",
        "source_num_classes": 1000,
        "variant": "single_cls",
    }
    return ViTTinyStudent(encoder, initialization, ready=True, mode=mode, rank=rank, alpha=alpha)
