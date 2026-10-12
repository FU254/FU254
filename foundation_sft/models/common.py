"""Encoder contract, trainable heads, and architecture-aware LoRA primitives."""

from dataclasses import asdict, dataclass
from collections import OrderedDict
from pathlib import Path
import hashlib
import math
import pickle
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PreprocessSpec:
    image_size: int
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    resize_mode: str = "resize"
    resize_shortest_edge: int | None = None
    interpolation: str = "bicubic"
    do_hist_equalization: bool = False

    def __post_init__(self):
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int) or self.image_size <= 0 or len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Preprocessing needs a positive image size and three-channel statistics")
        if any(not math.isfinite(v) for v in tuple(self.mean) + tuple(self.std)) or any(v <= 0 for v in self.std):
            raise ValueError("Normalization statistics must be finite with positive standard deviation")
        if not isinstance(self.do_hist_equalization, bool):
            raise ValueError("do_hist_equalization must be a boolean")


@dataclass
class BackboneFeatures:
    global_features: torch.Tensor
    local_features: torch.Tensor
    grid_shape: tuple

    def validate(self):
        if self.global_features.ndim != 2 or self.local_features.ndim != 3:
            raise ValueError("Expected global [B,D] and spatial local [B,N,D] features")
        if self.global_features.shape[0] != self.local_features.shape[0]:
            raise ValueError("Global and local features have different batch sizes")
        if self.global_features.shape[-1] != self.local_features.shape[-1]:
            raise ValueError("Global and local features have different channel dimensions")
        if len(self.grid_shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.grid_shape):
            raise ValueError("A positive two-dimensional spatial grid is required")
        if math.prod(self.grid_shape) != self.local_features.shape[1]:
            raise ValueError("Spatial grid does not match local token count (exclude CLS/register tokens)")
        return self


class BackboneAdapter(nn.Module):
    feature_dim: int
    preprocess: PreprocessSpec
    provenance: dict

    def forward_features(self, images):
        raise NotImplementedError

    def lora_target_names(self):
        """Return exact relative module names of verified attention projections."""
        raise NotImplementedError


class LoRALinear(nn.Module):
    """Preserve linear weights, including encoders that call F.linear directly."""

    def __init__(self, base, rank=8, alpha=16.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA requires nn.Linear, positive rank and positive finite alpha")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self):
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        # Ordinary module calls use the low-rank path without materializing
        # another full projection matrix. EVA-style F.linear calls can still
        # access the equivalent merged projection through the weight property.
        update = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return self.base(inputs) + update * self.scaling


def inject_lora(adapter, rank=8, alpha=16.0):
    targets = list(adapter.lora_target_names())
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Encoder must provide nonempty, unique, verified attention projection names")
    modules = dict(adapter.named_modules())
    for name in targets:
        if not isinstance(modules.get(name), nn.Linear):
            raise ValueError(f"LoRA target {name!r} is not an nn.Linear in this encoder")
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = adapter.get_submodule(parent_name) if parent_name else adapter
        setattr(parent, child_name, LoRALinear(modules[name], rank, alpha))
    return targets


class FoundationClassifier(nn.Module):
    """Frozen linear probe or LoRA + output normalization + configurable MLP."""

    def __init__(self, adapter, mode="lora", rank=8, alpha=16.0, hidden_dims=(384, 192)):
        super().__init__()
        if mode not in {"linear_probe", "lora"}:
            raise ValueError("mode must be linear_probe or lora")
        self.adapter = adapter
        self.mode = mode
        self.adapter.requires_grad_(False)
        self.lora_targets = []
        if mode == "linear_probe":
            self.output_norm = nn.Identity()
            self.head = nn.Linear(adapter.feature_dim, 1)
            self.adapter.eval()
        else:
            self.lora_targets = inject_lora(adapter, rank, alpha)
            self.output_norm = nn.LayerNorm(adapter.feature_dim)
            sizes = [adapter.feature_dim, *hidden_dims, 1]
            if any(size <= 0 for size in sizes):
                raise ValueError("Classification head dimensions must be positive")
            layers = []
            for index, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(in_dim, out_dim))
                if index < len(sizes) - 2:
                    layers.append(nn.ReLU())
            self.head = nn.Sequential(*layers)
        self.rank, self.alpha = rank, alpha

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "linear_probe":
            self.adapter.eval()
        else:
            # Frozen running statistics must not drift outside the saved adapter.
            for module in self.adapter.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, images, return_features=False):
        if self.mode == "linear_probe":
            with torch.no_grad():
                features = self.adapter.forward_features(images).validate()
        else:
            features = self.adapter.forward_features(images).validate()
        logits = self.head(self.output_norm(features.global_features)).squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def trainable_state_dict(self):
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable}

    def load_trainable_state_dict(self, state):
        expected = {name for name, param in self.named_parameters() if param.requires_grad}
        if set(state) != expected:
            raise ValueError(f"Adapter parameter mismatch: missing={sorted(expected-set(state))}, "
                             f"unexpected={sorted(set(state)-expected)}")
        self.load_state_dict(state, strict=False)

    def training_metadata(self):
        return {"mode": self.mode, "rank": self.rank, "alpha": self.alpha,
                "lora_targets": self.lora_targets, "preprocessing": asdict(self.adapter.preprocess),
                "pretrained": self.adapter.provenance,
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_dtype(value, *args):
    dtype = np.dtype(value, *args)
    if dtype.kind not in "biufc" or dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise pickle.UnpicklingError("Only plain numeric NumPy dtypes are permitted in checkpoint metadata")
    return dtype


def _numeric_scalar(dtype, data):
    dtype = _numeric_dtype(dtype)
    if not isinstance(data, bytes) or len(data) != dtype.itemsize:
        raise pickle.UnpicklingError("Invalid numeric NumPy scalar payload")
    return np.core.multiarray.scalar(dtype, data)


def _checkpoint_bytes(value, encoding="latin1", errors="strict"):
    # NumPy's protocol-2 pickle encodes scalar bytes through _codecs.encode.
    # Other encodings could invoke a user-installed codec during deserialization.
    if not isinstance(value, str) or encoding != "latin1" or errors != "strict":
        raise pickle.UnpicklingError("Only literal latin1 bytes are permitted in checkpoint metadata")
    return value.encode("latin1")


def _numeric_checkpoint_globals():
    allowed = {
        "collections.OrderedDict": OrderedDict,
        "__builtin__.set": set,
        "builtins.set": set,
        "numpy.dtype": _numeric_dtype,
        "numpy.core.multiarray.scalar": _numeric_scalar,
        "numpy._core.multiarray.scalar": _numeric_scalar,
        "_codecs.encode": _checkpoint_bytes,
        "torch._utils._rebuild_tensor": torch._utils._rebuild_tensor,
        "torch._utils._rebuild_tensor_v2": torch._utils._rebuild_tensor_v2,
        "torch._utils._rebuild_parameter": torch._utils._rebuild_parameter,
        "torch.storage.UntypedStorage": torch.UntypedStorage,
        "torch.UntypedStorage": torch.UntypedStorage,
    }
    if hasattr(torch._utils, "_rebuild_tensor_v3"):
        allowed["torch._utils._rebuild_tensor_v3"] = torch._utils._rebuild_tensor_v3
    for name in ("float16", "float32", "float64", "bfloat16", "int8", "int16", "int32", "int64",
                 "uint8", "bool", "complex64", "complex128"):
        allowed[f"torch.{name}"] = getattr(torch, name)
    return allowed


class _NumericCheckpointUnpickler(pickle._Unpickler):
    """Local tensor/numeric/set metadata loader for older PyTorch checkpoints.

    Besides an exact GLOBAL allowlist, restrict REDUCE, BUILD and storage IDs.
    In particular, never permit NumPy arrays, nested pickle/storage loaders, or
    extension opcodes that could bypass find_class through pickle's global cache.
    CPython's ordinary FRAME support also handles protocol-4 checkpoints that
    PyTorch's weights-only loader cannot parse, under these same restrictions.
    """

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_globals = _numeric_checkpoint_globals()

    def find_class(self, module, name):
        key = f"{module}.{name}"
        if key not in self.allowed_globals:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {key}")
        return self.allowed_globals[key]

    def load_reduce(self):
        args = self.stack.pop()
        function = self.stack[-1]
        if not any(function is allowed for allowed in self.allowed_globals.values()):
            raise pickle.UnpicklingError("Unrecognized checkpoint reconstruction function")
        self.stack[-1] = function(*args)

    def load_build(self):
        state = self.stack.pop()
        instance = self.stack[-1]
        if isinstance(instance, np.dtype):
            # NumPy dtype's ordinary scalar state contains no fields/subarrays.
            if not isinstance(state, tuple) or len(state) != 8 or any(value is not None for value in state[2:5]):
                raise pickle.UnpicklingError("Structured NumPy dtype metadata is not permitted")
            instance.__setstate__(state)
            _numeric_dtype(instance)
        elif type(instance) is OrderedDict and isinstance(state, dict):
            instance.__dict__.update(state)
        elif type(instance) is torch.Tensor:
            instance.set_(*state)
        elif type(instance) is torch.nn.Parameter:
            instance.__setstate__(state)
        else:
            raise pickle.UnpicklingError("Unsupported checkpoint object state")

    def load_binpersid(self):
        saved_id = self.stack.pop()
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] not in ("storage", b"storage"):
            raise pickle.UnpicklingError("Only tensor-storage persistent IDs are permitted")
        self.append(self.persistent_load(saved_id))

    def reject_object_opcode(self):
        raise pickle.UnpicklingError("Unsupported checkpoint object/extension opcode")

    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.BINPERSID[0]] = load_binpersid
    for opcode in (pickle.INST, pickle.OBJ, pickle.NEWOBJ, pickle.NEWOBJ_EX,
                   pickle.EXT1, pickle.EXT2, pickle.EXT4, pickle.PERSID):
        dispatch[opcode[0]] = reject_object_opcode


def _load_numeric_checkpoint(path):
    # Restrict compatibility loading to Torch ZIP checkpoints. The obsolete tar
    # loader extracts members and is unnecessary for the EVA-X release format.
    if not zipfile.is_zipfile(path):
        raise pickle.UnpicklingError("Numeric metadata compatibility requires a Torch ZIP checkpoint")
    restricted_pickle = SimpleNamespace(
        __name__="foundation_numeric_checkpoint_pickle", Unpickler=_NumericCheckpointUnpickler,
    )
    # weights_only=False selects OUR restricted unpickler, not Python's ordinary
    # pickle module; arbitrary GLOBAL/REDUCE calls remain rejected on Torch 2.0.
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted_pickle)


def load_checkpoint(path, state_key=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        # Older training checkpoints save NumPy metrics and ordinary set
        # metadata outside the tensor state dict. PyTorch 2.0 has no local
        # safe_globals context. Its weights-only loader also lacks protocol-4
        # FRAME (149) support; handle only these known restrictions, never an
        # arbitrary unknown application class or corrupt checkpoint.
        message = str(error)
        metadata_globals = ("numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar", "numpy.dtype",
                            "__builtin__.set", "builtins.set")
        unsupported_frame = "Unsupported operand 149" in message
        if not any(name in message for name in metadata_globals) and not unsupported_frame:
            raise
        state = _load_numeric_checkpoint(path)
    if state_key:
        for part in state_key.split("."):
            state = state[part]
    else:
        for key in ("model", "state_dict", "module"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint is not a tensor state dict; configure the verified state_key")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def load_backbone_weights(module, state, ignored_prefixes=("head.",)):
    state = {key: value for key, value in state.items() if not key.startswith(ignored_prefixes)}
    result = module.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith(ignored_prefixes)]
    unexpected = [key for key in result.unexpected_keys if not key.startswith(ignored_prefixes)]
    if missing or unexpected:
        raise ValueError(f"Pretrained backbone mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")
    return result
