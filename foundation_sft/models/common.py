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


class _InertOmegaConfNode:
    """Data-only stand-in: OmegaConf constructors/resolvers are never imported."""

    __slots__ = ("state",)


_OMEGACONF_NODE_KINDS = {
    "omegaconf.listconfig.ListConfig": "list",
    "omegaconf.dictconfig.DictConfig": "dict",
    "omegaconf.nodes.AnyNode": "scalar",
    "omegaconf.nodes.StringNode": "scalar",
    "omegaconf.nodes.IntegerNode": "scalar",
    "omegaconf.nodes.FloatNode": "scalar",
    "omegaconf.nodes.BooleanNode": "scalar",
    "omegaconf.base.Metadata": "metadata",
    "omegaconf.base.ContainerMetadata": "container_metadata",
}
_OMEGACONF_NODE_CLASSES = {
    name: type("_Inert" + name.rsplit(".", 1)[1], (_InertOmegaConfNode,),
               {"__slots__": (), "kind": kind})
    for name, kind in _OMEGACONF_NODE_KINDS.items()
}


@dataclass(frozen=True)
class _OmegaConfTypeToken:
    name: str


_OMEGACONF_TYPE_TOKENS = {
    name: _OmegaConfTypeToken(name)
    for name in ("dict", "list", "tuple", "str", "bytes", "int", "float", "bool", "Any", "Dict", "List")
}


def _checkpoint_resolver_cache(factory=None):
    # OmegaConf serializes resolver_cache as defaultdict(dict). Keep only its
    # inert dictionary data; neither an arbitrary factory nor a resolver runs.
    if factory is not None and factory is not _OMEGACONF_TYPE_TOKENS["dict"]:
        raise pickle.UnpicklingError("Unsupported OmegaConf resolver-cache factory")
    return {}


def _omegaconf_checkpoint_globals():
    allowed = dict(_OMEGACONF_NODE_CLASSES)
    for name in ("dict", "list", "tuple", "str", "bytes", "int", "float", "bool"):
        allowed["builtins." + name] = _OMEGACONF_TYPE_TOKENS[name]
        allowed["__builtin__." + name] = _OMEGACONF_TYPE_TOKENS[name]
    allowed["__builtin__.long"] = _OMEGACONF_TYPE_TOKENS["int"]
    for name in ("Any", "Dict", "List"):
        allowed["typing." + name] = _OMEGACONF_TYPE_TOKENS[name]
    allowed["collections.defaultdict"] = _checkpoint_resolver_cache
    return allowed


def _valid_omegaconf_state(instance, state):
    if not isinstance(state, dict):
        return False
    kind = instance.kind
    if kind in {"metadata", "container_metadata"}:
        required = {"ref_type", "object_type", "optional", "key", "flags", "flags_root", "resolver_cache"}
        if kind == "container_metadata":
            required.update(("key_type", "element_type"))
        if set(state) != required or not isinstance(state["optional"], bool) or not isinstance(state["flags_root"], bool):
            return False
        for name in ("ref_type", "object_type", "key_type", "element_type"):
            if name in state and state[name] is not None and not isinstance(state[name], _OmegaConfTypeToken):
                return False
        flags = state["flags"]
        if flags is not None and (not isinstance(flags, dict) or any(
                not isinstance(key, str) or not isinstance(value, bool) for key, value in flags.items())):
            return False
        return (state["key"] is None or isinstance(state["key"], (str, int, float, bool))) and isinstance(state["resolver_cache"], dict)
    required = {"_metadata", "_parent", "_content" if kind in {"list", "dict"} else "_val"}
    if not required.issubset(state) or set(state).difference(required | {"_flags_cache"}):
        return False
    metadata = state["_metadata"]
    if not isinstance(metadata, _InertOmegaConfNode) or metadata.kind not in {"metadata", "container_metadata"}:
        return False
    parent = state["_parent"]
    if parent is not None and (not isinstance(parent, _InertOmegaConfNode) or parent.kind not in {"list", "dict"}):
        return False
    cache = state.get("_flags_cache")
    if cache is not None and (not isinstance(cache, dict) or any(
            not isinstance(key, str) or value is not None and not isinstance(value, bool) for key, value in cache.items())):
        return False
    if kind == "scalar":
        return state["_val"] is None or isinstance(state["_val"], (str, bytes, bool, int, float))
    content = state["_content"]
    if content is None:
        return True
    if kind == "list":
        return isinstance(content, list) and all(isinstance(item, _InertOmegaConfNode) for item in content)
    return isinstance(content, dict) and all(
        isinstance(key, (str, bytes, bool, int, float)) and isinstance(value, _InertOmegaConfNode)
        for key, value in content.items()
    )


class _OmegaConfCheckpointUnpickler(_NumericCheckpointUnpickler):
    """Restricted tensor loading with inert OmegaConf 2.3 metadata support.

    Only the reviewed container/scalar-node layouts are accepted. NEWOBJ does
    not call OmegaConf code, BUILD stores validated data in one local slot, and
    REDUCE cannot call metadata types or arbitrary resolver/cache factories.
    """

    dispatch = _NumericCheckpointUnpickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_functions = tuple(self.allowed_globals.values()) + (_checkpoint_resolver_cache,)
        self.allowed_globals.update(_omegaconf_checkpoint_globals())

    def load_newobj(self):
        arguments = self.stack.pop()
        cls = self.stack.pop()
        if arguments != () or not any(cls is allowed for allowed in _OMEGACONF_NODE_CLASSES.values()):
            raise pickle.UnpicklingError("Unsupported checkpoint metadata constructor")
        self.append(object.__new__(cls))

    def load_reduce(self):
        arguments = self.stack.pop()
        function = self.stack[-1]
        if not isinstance(arguments, tuple) or not any(function is allowed for allowed in self.reconstruction_functions):
            raise pickle.UnpicklingError("Unrecognized checkpoint reconstruction function")
        self.stack[-1] = function(*arguments)

    def load_build(self):
        if isinstance(self.stack[-2], _InertOmegaConfNode):
            state = self.stack.pop()
            instance = self.stack[-1]
            if not _valid_omegaconf_state(instance, state):
                raise pickle.UnpicklingError("Unsupported OmegaConf checkpoint metadata state")
            instance.state = state
        else:
            super().load_build()

    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.BUILD[0]] = load_build


def _plain_omegaconf_metadata(payload):
    """Discard parent/type/flag metadata; read values without interpolation."""
    memo, active = {}, set()

    def plain(value):
        if isinstance(value, _InertOmegaConfNode):
            if value.kind not in {"dict", "list", "scalar"} or not hasattr(value, "state"):
                raise pickle.UnpicklingError("OmegaConf checkpoint metadata is not a data value")
            if value.kind == "scalar":
                # In particular, preserve '${...}' literally. No resolver,
                # environment lookup, import, or interpolation is evaluated.
                return value.state["_val"]
            return plain(value.state["_content"])
        if isinstance(value, _OmegaConfTypeToken):
            raise pickle.UnpicklingError("OmegaConf type metadata cannot be a configuration value")
        if type(value) not in (dict, OrderedDict, list, tuple, set):
            return value
        identity = id(value)
        if identity in active:
            raise pickle.UnpicklingError("Cyclic checkpoint configuration values are unsupported")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        if isinstance(value, dict):
            result = OrderedDict() if type(value) is OrderedDict else {}
            for key, item in value.items():
                result[key] = plain(item)
            if type(value) is OrderedDict:
                result.__dict__.update({key: plain(item) for key, item in value.__dict__.items()})
        else:
            items = [plain(item) for item in value]
            result = tuple(items) if type(value) is tuple else set(items) if type(value) is set else items
        active.remove(identity)
        memo[identity] = result
        return result

    return plain(payload)


def _load_omegaconf_checkpoint(path):
    if not zipfile.is_zipfile(path):
        raise pickle.UnpicklingError("OmegaConf metadata compatibility requires a Torch ZIP checkpoint")
    restricted_pickle = SimpleNamespace(
        __name__="foundation_omegaconf_checkpoint_pickle", Unpickler=_OmegaConfCheckpointUnpickler,
    )
    # This selects only our restricted, inert parser, never arbitrary pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted_pickle)
    return _plain_omegaconf_metadata(payload)


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


def load_checkpoint_payload(path, *, allow_omegaconf_metadata=False):
    """Load local tensor/plain-metadata payload without selecting its state dict."""
    if not isinstance(allow_omegaconf_metadata, bool):
        raise ValueError("allow_omegaconf_metadata must be a boolean")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    if allow_omegaconf_metadata:
        # CXR-CLIP stores config after model/optimizer tensors. Going directly
        # to its inert parser avoids loading the large weights twice before
        # discovering ListConfig. It also never consults process-wide user
        # safe_globals for OmegaConf classes.
        state = _load_omegaconf_checkpoint(path)
    else:
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
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint payload must be a nonempty dictionary")
    return state


def load_checkpoint(path, state_key=None):
    state = load_checkpoint_payload(path)
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
