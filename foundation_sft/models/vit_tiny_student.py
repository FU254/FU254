"""Single-CLS DeiT-Tiny/16 student with full, partial or LoRA tuning, offline.

Fresh training::

    model = build_vit_tiny_student("deit_tiny_patch16_224-a1311bcf.pth")

    # Optional: freeze the ImageNet base and train QKV LoRA plus a linear head.
    model = build_vit_tiny_student("deit_tiny_patch16_224-a1311bcf.pth", mode="lora")

    # Optional: preserve ImageNet embeddings/early blocks, tune the last four.
    model = build_vit_tiny_student("deit_tiny_patch16_224-a1311bcf.pth",
                                  mode="partial_finetune", train_last_blocks=4)

    # Explicitly train from random weights; no source checkpoint is accessed.
    model = build_vit_tiny_student(random_seed=42)

Evaluation/resume from a saved student needs no original ImageNet file::

    model = build_vit_tiny_student(
        initialization=saved_metadata["initialization"])
    model.load_trainable_state_dict(saved_student_state)

The second form cannot run forward until the complete student state is loaded.
Save ``training_metadata()`` and the complete ``trainable_state_dict()`` together.
For LoRA reconstruction, also pass the saved mode, LoRA rank and alpha to the
factory. For partial reconstruction, pass the saved mode and train_last_blocks.
All modes save the entire encoder, so none needs its source file.
Random initialization is recorded separately from ImageNet initialization and
does not change the caller's Torch random-number state.
"""

from collections.abc import Mapping
from contextlib import contextmanager
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
DROP_PATH_SEED_XOR = 0x44504154


def _validate_drop_path(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError("drop_path_rate must be finite and in [0, 1)")
    return float(value)


def _drop_path_context(rate, initialization, mode):
    if rate and (mode != "full_finetune" or initialization.get("initialization_mode") != "random"):
        raise ValueError("Positive DropPath requires a random, fully trained student")


def _validate_training(mode, rank, alpha, train_last_blocks=None):
    if mode not in ("full_finetune", "partial_finetune", "lora"):
        raise ValueError("Student mode must be full_finetune, partial_finetune or lora")
    if mode == "partial_finetune":
        if type(train_last_blocks) is not int or train_last_blocks not in (2, 4, 8):
            raise ValueError("Partial fine-tuning requires train_last_blocks=2, 4 or 8")
    elif train_last_blocks is not None:
        raise ValueError("train_last_blocks is only valid for partial_finetune")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("Student LoRA rank must be a positive integer")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Student LoRA alpha must be positive and finite")


def _partial_context(mode, initialization):
    if mode == "partial_finetune" and initialization.get("pretraining") != "ImageNet-1k":
        raise ValueError("Partial fine-tuning requires the ImageNet-pretrained student")


def _partial_metadata(train_last_blocks):
    first = 12 - train_last_blocks
    return {
        "train_last_blocks": train_last_blocks,
        "total_blocks": 12,
        "trainable_block_indices": list(range(first, 12)),
        "frozen_block_indices": list(range(first)),
        "frozen_embeddings": ["patch_embed", "cls_token", "pos_embed"],
        "trainable_modules": [f"encoder.blocks.{index}" for index in range(first, 12)]
                             + ["encoder.norm", "encoder.head"],
    }


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


def _new_encoder(num_classes, drop_path_rate=0.0):
    try:
        import timm
    except ImportError as error:
        raise ImportError("ViT-Tiny student requires timm (validated with timm==0.9.0)") from error
    options = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
    encoder = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=num_classes, **options)
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
    if drop_path_rate:
        expected = torch.linspace(0, drop_path_rate, 12).tolist()
        for block, probability in zip(encoder.blocks, expected):
            for name in ("drop_path1", "drop_path2"):
                module = getattr(block, name, None)
                actual = 0.0 if isinstance(module, nn.Identity) else getattr(module, "drop_prob", None)
                if actual != probability:
                    raise ValueError("Expected native linear DropPath schedule over 12 blocks")
    return encoder


class ViTTinyStudent(nn.Module):
    """One output head with full, last-block partial, or QKV LoRA tuning."""

    def __init__(self, encoder, initialization, *, ready, mode="full_finetune", rank=8, alpha=16.0,
                 drop_path_rate=0.0, train_last_blocks=None):
        super().__init__()
        _validate_training(mode, rank, alpha, train_last_blocks)
        self.encoder = encoder
        self.feature_dim = FEATURE_DIM
        self.preprocess = PreprocessSpec(IMAGE_SIZE)
        self.initialization = _validate_initialization(initialization)
        _partial_context(mode, self.initialization)
        self.drop_path_rate = _validate_drop_path(drop_path_rate)
        _drop_path_context(self.drop_path_rate, self.initialization, mode)
        self._drop_path_seed = None
        self._drop_path_cpu_state = self._drop_path_cuda_state = None
        if self.drop_path_rate:
            self._drop_path_seed = self.initialization["seed"] ^ DROP_PATH_SEED_XOR
            self._drop_path_cpu_state = torch.Generator(device="cpu").manual_seed(self._drop_path_seed).get_state()
        self._ready = ready
        self.training_mode = mode
        self.lora = None
        self.partial_finetune = None
        if mode == "full_finetune":
            self.requires_grad_(True)
        elif mode == "partial_finetune":
            if len(self.encoder.blocks) != 12:
                raise ValueError("Partial fine-tuning requires exactly twelve Transformer blocks")
            self.encoder.requires_grad_(False)
            for block in self.encoder.blocks[-train_last_blocks:]:
                block.requires_grad_(True)
            self.encoder.norm.requires_grad_(True)
            self.encoder.head.requires_grad_(True)
            self.partial_finetune = _partial_metadata(train_last_blocks)
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
        with self._stochastic_rng(images):
            return self._forward(images, return_features)

    @contextmanager
    def _stochastic_rng(self, images):
        """Give DropPath its own stream, preserving data/outer model RNG draws."""
        if not self.training or not self.drop_path_rate:
            yield
            return
        if images.device.type not in {"cpu", "cuda"}:
            raise ValueError("Isolated DropPath supports CPU and CUDA training")
        cuda = images.device.type == "cuda"
        index = images.device.index if cuda else None
        if cuda and index is None:
            index = torch.cuda.current_device()
        if cuda and self._drop_path_cuda_state is None:
            self._drop_path_cuda_state = torch.Generator(device=images.device).manual_seed(
                self._drop_path_seed).get_state().cpu()
        if cuda:
            # A CPU-loaded checkpoint may contain a CUDA stream produced by a
            # different runtime. Let the target backend validate its own bytes.
            self._drop_path_cuda_state = self._validated_cuda_state(
                self._drop_path_cuda_state, images.device)
        with torch.random.fork_rng(devices=[index] if cuda else []):
            torch.set_rng_state(self._drop_path_cpu_state)
            if cuda:
                torch.cuda.set_rng_state(self._drop_path_cuda_state, index)
            try:
                yield
            finally:
                self._drop_path_cpu_state = torch.get_rng_state().clone()
                if cuda:
                    self._drop_path_cuda_state = torch.cuda.get_rng_state(index).cpu().clone()

    def _forward(self, images, return_features=False):
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

    def stochastic_state_dict(self):
        """Save streams separately so initial model-weight SHA is unchanged."""
        if not self.drop_path_rate:
            return None
        return {"format_version": 1, "seed": self._drop_path_seed,
                "cpu": self._drop_path_cpu_state.cpu().clone(),
                "cuda": self._drop_path_cuda_state.cpu().clone()
                    if self._drop_path_cuda_state is not None else None}

    def _validated_cuda_state(self, state, device):
        """Validate opaque CUDA state using an independent target-device RNG."""
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("DropPath CUDA RNG validation requires a CUDA target")
        probe = torch.Generator(device=device)
        try:
            probe.set_state(state)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("DropPath CUDA RNG state cannot be restored by this runtime") from error
        if probe.initial_seed() != self._drop_path_seed:
            raise ValueError("DropPath CUDA RNG state belongs to a different seed")
        return probe.get_state().cpu().clone()

    def load_stochastic_state_dict(self, state):
        """Restore future masks without changing the caller's random streams."""
        if not self.drop_path_rate:
            if state is not None:
                raise ValueError("A zero-DropPath student must not have stochastic state")
            return
        if (not isinstance(state, Mapping) or set(state) != {"format_version", "seed", "cpu", "cuda"}
                or type(state["format_version"]) is not int or state["format_version"] != 1
                or type(state["seed"]) is not int or state["seed"] != self._drop_path_seed):
            raise ValueError("DropPath stochastic state metadata differs")
        for name in ("cpu", "cuda"):
            value = state[name]
            if name == "cuda" and value is None:
                continue
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                    or value.device.type != "cpu" or value.layout != torch.strided
                    or value.ndim != 1 or value.numel() == 0):
                raise ValueError("Invalid DropPath " + name + " RNG state")
        probe = torch.Generator(device="cpu")
        try:
            probe.set_state(state["cpu"])
        except RuntimeError as error:
            raise ValueError("Invalid DropPath CPU RNG state") from error
        if probe.initial_seed() != self._drop_path_seed:
            raise ValueError("DropPath CPU RNG state belongs to a different seed")
        cuda_state = state["cuda"]
        device = self.encoder.head.weight.device
        if cuda_state is not None and device.type == "cuda":
            cuda_state = self._validated_cuda_state(cuda_state, device)
        # CPU inference must not initialize an unrelated GPU. Preserve CUDA
        # state opaquely; a future CUDA training forward validates on its target.
        self._drop_path_cpu_state = state["cpu"].clone()
        self._drop_path_cuda_state = cuda_state.clone() if cuda_state is not None else None

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
        if self.partial_finetune is not None:
            metadata["partial_finetune"] = deepcopy(self.partial_finetune)
        if self.drop_path_rate:
            metadata["regularization"] = {"drop_path_rate": self.drop_path_rate,
                "schedule": "linear_12_blocks", "rng": "isolated_torch", "rng_seed": self._drop_path_seed}
        return metadata


def build_vit_tiny_student(checkpoint=None, *, initialization=None, random_seed=None,
                           mode="full_finetune", rank=8, alpha=16.0, drop_path_rate=0.0,
                           train_last_blocks=None):
    """Build from ImageNet, explicit random weights, or saved metadata, offline.

    ``checkpoint`` is the original 1000-class single-CLS release, verified by
    SHA256 prefix and loaded strictly before replacing its head. ``initialization``
    is the dict from ``training_metadata()``; this reconstruction form requires
    a subsequent strict ``load_trainable_state_dict`` of the complete student.
    ``mode='lora'`` freezes original weights and trains QKV adapters plus the
    one-logit linear head. Pass the saved mode/rank/alpha when reconstructing it.
    ``mode='partial_finetune', train_last_blocks=2|4|8`` freezes ImageNet input
    embeddings and early blocks; only the last blocks, final norm and head are
    trained. Pass the same scope when reconstructing a saved complete student.
    ``random_seed`` creates a ready-to-train random student without reading a
    checkpoint. Its seed applies to the encoder, head and any LoRA parameters;
    it leaves the caller's CPU and CUDA random-number states unchanged.
    Exactly one of the three initialization arguments is required. There are
    no automatic downloads or fallbacks from missing ImageNet weights.
    Positive ``drop_path_rate`` uses timm's linear 12-block schedule for random,
    fully trained students. Save its separate stochastic state alongside weights
    when the future training mask sequence must be reproduced.
    """
    _validate_training(mode, rank, alpha, train_last_blocks)
    drop_path_rate = _validate_drop_path(drop_path_rate)
    if sum(value is not None for value in (checkpoint, initialization, random_seed)) != 1:
        raise ValueError("Provide either an official checkpoint, saved initialization metadata, or random_seed (exactly one)")
    if random_seed is not None:
        if mode == "partial_finetune":
            raise ValueError("Partial fine-tuning requires the ImageNet-pretrained student")
        seed = _validate_random_seed(random_seed)
        metadata = {
            "architecture": ARCHITECTURE,
            "initialization_mode": "random",
            "pretraining": "none",
            "seed": seed,
            "source_num_classes": 1,
            "variant": "single_cls",
        }
        _drop_path_context(drop_path_rate, metadata, mode)
        # Build on CPU, including LoRA injection. Seeding only the CPU default
        # generator avoids changing CUDA RNG streams used by the caller.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            encoder = _new_encoder(1, drop_path_rate) if drop_path_rate else _new_encoder(1)
            return ViTTinyStudent(encoder, metadata, ready=True, mode=mode, rank=rank, alpha=alpha,
                                  drop_path_rate=drop_path_rate, train_last_blocks=train_last_blocks)
    if initialization is not None:
        initialization = _validate_initialization(initialization)
        _partial_context(mode, initialization)
        _drop_path_context(drop_path_rate, initialization, mode)
        encoder = _new_encoder(1, drop_path_rate) if drop_path_rate else _new_encoder(1)
        return ViTTinyStudent(encoder, initialization, ready=False, mode=mode, rank=rank, alpha=alpha,
                             drop_path_rate=drop_path_rate, train_last_blocks=train_last_blocks)

    if drop_path_rate:
        raise ValueError("Positive DropPath requires a random, fully trained student")
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
    return ViTTinyStudent(encoder, initialization, ready=True, mode=mode, rank=rank, alpha=alpha,
                         train_last_blocks=train_last_blocks)
