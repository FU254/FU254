"""Read the historical TorchXRayVision DenseNet release without model pickle code.

The release is a legacy ``torch.save(nn.Module)`` file. Its module GLOBALs are
mapped to local, inert records; constructors, module hooks and embedded source
are never evaluated. Tensor reconstruction uses the reviewed numeric checkpoint
parser, independently of installed TorchXRayVision or process-wide safe globals.
"""

from collections import OrderedDict
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile

import torch

from .common import _NumericCheckpointUnpickler


class _InertModule:
    __slots__ = ("state",)


_MODULE_GLOBALS = (
    "torchxrayvision.models.DenseNet",
    "torchxrayvision.models._DenseBlock",
    "torchxrayvision.models._DenseLayer",
    "torchxrayvision.models._Transition",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.activation.ReLU",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.pooling.AvgPool2d",
    "torch.nn.modules.linear.Linear",
)
_MODULE_CLASSES = {
    name: type("_Inert" + name.rsplit(".", 1)[1], (_InertModule,),
               {"__slots__": (), "kind": name.rsplit(".", 1)[1]})
    for name in _MODULE_GLOBALS
}
_STORAGE_GLOBALS = {"torch." + name for name in (
    "DoubleStorage", "FloatStorage", "HalfStorage", "LongStorage", "IntStorage", "ShortStorage",
    "CharStorage", "ByteStorage", "BoolStorage", "BFloat16Storage", "ComplexFloatStorage",
    "ComplexDoubleStorage", "UntypedStorage",
)} | {"torch.storage.UntypedStorage"}
_HOOK_FIELDS = {
    "_backward_hooks", "_backward_pre_hooks", "_forward_hooks", "_forward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
}
_MODULE_FIELDS = {
    "DenseNet": set(), "Sequential": set(), "_DenseBlock": set(), "_Transition": set(),
    "_DenseLayer": {"drop_rate", "memory_efficient"},
    "Conv2d": {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation",
               "transposed", "output_padding", "groups", "padding_mode", "_reversed_padding_repeated_twice"},
    "BatchNorm2d": {"num_features", "eps", "momentum", "affine", "track_running_stats"},
    "ReLU": {"inplace"},
    "MaxPool2d": {"kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"},
    "AvgPool2d": {"kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"},
    "Linear": {"in_features", "out_features"},
}


def _mapping(value):
    return type(value) in (dict, OrderedDict)


def _module_name(value):
    return type(value) is str and bool(value) and "." not in value


def _module_attribute(value):
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in (tuple, list) and all(type(item) in (int, bool) for item in value)


def _valid_module_state(instance, state):
    required = {"training", "_parameters", "_buffers", "_modules"}
    optional = _HOOK_FIELDS | {"_non_persistent_buffers_set", "_is_full_backward_hook", "_compiled_call_impl"}
    if not _mapping(state) or not required.issubset(state) or set(state) - required - optional - _MODULE_FIELDS[instance.kind]:
        return False
    if type(state["training"]) is not bool:
        return False
    for field in ("_parameters", "_buffers"):
        if not _mapping(state[field]) or any(not _module_name(name) or
                value is not None and type(value) not in (torch.Tensor, torch.nn.Parameter)
                for name, value in state[field].items()):
            return False
    if not _mapping(state["_modules"]) or any(not _module_name(name) or
            value is not None and not isinstance(value, _InertModule)
            for name, value in state["_modules"].items()):
        return False
    for field in _HOOK_FIELDS & set(state):
        if not _mapping(state[field]) or state[field]:
            return False
    if "_non_persistent_buffers_set" in state and (type(state["_non_persistent_buffers_set"]) is not set or
            not all(_module_name(name) for name in state["_non_persistent_buffers_set"])):
        return False
    if "_is_full_backward_hook" in state and state["_is_full_backward_hook"] is not None and type(state["_is_full_backward_hook"]) is not bool:
        return False
    if state.get("_compiled_call_impl") is not None:
        return False
    return all(_module_attribute(state[name]) for name in _MODULE_FIELDS[instance.kind] & set(state))


class _XRVCheckpointUnpickler(_NumericCheckpointUnpickler):
    dispatch = _NumericCheckpointUnpickler.dispatch.copy()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_functions = tuple(self.allowed_globals.values())
        self.allowed_globals.update(_MODULE_CLASSES)

    def _checked_find_class(self, module, name):
        # Torch's internal wrapper resolves *Storage before calling find_class.
        # Check the exact namespace first so that bypass cannot accept unknown
        # application globals that merely share a storage class name.
        if f"{module}.{name}" not in self.allowed_globals and f"{module}.{name}" not in _STORAGE_GLOBALS:
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")
        return self.find_class(module, name)

    def load_global(self):
        module = self.readline()[:-1].decode("utf-8")
        name = self.readline()[:-1].decode("utf-8")
        self.append(self._checked_find_class(module, name))

    def load_stack_global(self):
        name = self.stack.pop()
        module = self.stack.pop()
        if type(module) is not str or type(name) is not str:
            raise pickle.UnpicklingError("Invalid checkpoint global name")
        self.append(self._checked_find_class(module, name))

    def load_newobj(self):
        arguments = self.stack.pop()
        cls = self.stack.pop()
        if arguments != () or not any(cls is allowed for allowed in _MODULE_CLASSES.values()):
            raise pickle.UnpicklingError("Unsupported TorchXRayVision checkpoint constructor")
        self.append(object.__new__(cls))

    def _validate_tensor_storage(self, function, arguments):
        if function not in (torch._utils._rebuild_tensor, torch._utils._rebuild_tensor_v2,
                            getattr(torch._utils, "_rebuild_tensor_v3", None)):
            return
        if len(arguments) < 4 or type(arguments[0]) is not torch.storage.TypedStorage:
            raise pickle.UnpicklingError("Invalid checkpoint tensor storage")
        storage, offset, shape, stride = arguments[:4]
        if type(offset) is not int or offset < 0 or type(shape) is not tuple or type(stride) is not tuple or \
                len(shape) != len(stride) or any(type(value) is not int or value < 0 for value in shape + stride):
            raise pickle.UnpicklingError("Invalid checkpoint tensor layout")
        capacity = storage._size()
        if function is getattr(torch._utils, "_rebuild_tensor_v3", None):
            if len(arguments) < 7 or type(arguments[6]) is not torch.dtype:
                raise pickle.UnpicklingError("Invalid checkpoint tensor dtype")
            capacity = storage._untyped_storage.nbytes() // torch._utils._element_size(arguments[6])
        upper = offset if 0 in shape else offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        # set_ otherwise resizes storage for a crafted offset/stride, creating
        # bytes that were never present in the checkpoint.
        if upper > capacity:
            raise pickle.UnpicklingError("Checkpoint tensor layout exceeds its stored data")

    def load_reduce(self):
        arguments = self.stack.pop()
        function = self.stack[-1]
        if type(arguments) is not tuple or not any(function is allowed for allowed in self.reconstruction_functions):
            raise pickle.UnpicklingError("Unrecognized checkpoint reconstruction function")
        self._validate_tensor_storage(function, arguments)
        hook_index = None
        if function is torch._utils._rebuild_parameter:
            hook_index = 2
        elif function in (torch._utils._rebuild_tensor_v2, getattr(torch._utils, "_rebuild_tensor_v3", None)):
            hook_index = 5
        if hook_index is not None and (len(arguments) <= hook_index or
                not _mapping(arguments[hook_index]) or arguments[hook_index]):
            raise pickle.UnpicklingError("Checkpoint tensor hooks are unsupported")
        self.stack[-1] = function(*arguments)

    def load_build(self):
        if isinstance(self.stack[-2], _InertModule):
            state = self.stack.pop()
            instance = self.stack[-1]
            if not _valid_module_state(instance, state):
                raise pickle.UnpicklingError("Unsupported TorchXRayVision checkpoint module state or hooks")
            instance.state = state
        elif type(self.stack[-2]) is OrderedDict:
            state = self.stack.pop()
            if type(state) is not dict or set(state) - {"_metadata"} or not _mapping(state.get("_metadata", {})):
                raise pickle.UnpicklingError("Unsupported checkpoint dictionary metadata state")
            # Only the inert state_dict metadata attribute may be restored;
            # arbitrary attribute state could shadow dict methods such as items.
            self.stack[-1].__dict__.update(state)
        elif type(self.stack[-2]) in (torch.Tensor, torch.nn.Parameter):
            raise pickle.UnpicklingError("Unsupported checkpoint tensor object state")
        else:
            super().load_build()

    def load_binpersid(self):
        saved_id = self.stack.pop()
        if type(saved_id) is tuple and saved_id and saved_id[0] in ("module", b"module"):
            if len(saved_id) != 4 or not any(saved_id[1] is cls for cls in _MODULE_CLASSES.values()) or any(
                    value is not None and type(value) is not str for value in saved_id[2:]):
                raise pickle.UnpicklingError("Unsupported TorchXRayVision module persistent ID")
            # PyTorch normally compares embedded source with live classes, which
            # can write source patches. Discard that metadata here instead.
            self.append(saved_id[1])
        elif type(saved_id) is tuple and saved_id and saved_id[0] in ("storage", b"storage"):
            if len(saved_id) not in (5, 6) or type(saved_id[1]) is not torch.serialization.StorageType or \
                    type(saved_id[2]) not in (str, bytes) or type(saved_id[3]) not in (str, bytes) or \
                    type(saved_id[4]) is not int or saved_id[4] < 0:
                raise pickle.UnpicklingError("Unsupported tensor-storage persistent ID")
            if len(saved_id) == 6 and saved_id[5] is not None:
                view = saved_id[5]
                if type(view) is not tuple or len(view) != 3 or type(view[0]) not in (str, bytes) or any(
                        type(value) is not int or value < 0 for value in view[1:]):
                    raise pickle.UnpicklingError("Unsupported tensor-storage view")
            storage_keys = getattr(self, "storage_keys", None)
            if storage_keys is not None:
                storage_keys.add(saved_id[2])
            self.append(self.persistent_load(saved_id))
        else:
            raise pickle.UnpicklingError("Only reviewed module and tensor-storage persistent IDs are permitted")

    dispatch[pickle.GLOBAL[0]] = load_global
    dispatch[pickle.STACK_GLOBAL[0]] = load_stack_global
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.BINPERSID[0]] = load_binpersid


class _HeaderUnpickler(pickle._Unpickler):
    """Legacy framing is literal data, never tensors or application objects."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def reject_object_opcode(self):
        raise pickle.UnpicklingError("Expected a plain Torch checkpoint header; legacy tar archives are unsupported")

    for opcode in (pickle.GLOBAL, pickle.STACK_GLOBAL, pickle.REDUCE, pickle.BUILD,
                   pickle.INST, pickle.OBJ, pickle.NEWOBJ, pickle.NEWOBJ_EX,
                   pickle.EXT1, pickle.EXT2, pickle.EXT4, pickle.PERSID, pickle.BINPERSID):
        dispatch[opcode[0]] = reject_object_opcode


def _checkpoint_pickle_facade():
    # The flat format allocates storages while parsing tensor IDs and reads their
    # bytes only after a final key table. Missing IDs would leave uninitialized
    # heap memory in otherwise valid tensors. Track every root ID per load and
    # require that final table to cover all of them before Torch returns data.
    storage_keys, framing_reads = set(), 0

    class _LocalUnpickler(_XRVCheckpointUnpickler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.storage_keys = storage_keys

    def load_framing(file, **kwargs):
        nonlocal framing_reads
        framing_reads += 1
        value = _HeaderUnpickler(file, **kwargs).load()
        if framing_reads == 4 and (type(value) is not list or
                not all(type(key) in (str, bytes) for key in value) or
                len(value) != len(set(value)) or set(value) != storage_keys):
            raise pickle.UnpicklingError("Checkpoint storage table does not cover all tensor data")
        return value

    return SimpleNamespace(
        __name__="foundation_xrv_checkpoint_pickle", Unpickler=_LocalUnpickler, load=load_framing,
    )


class _NoFileDescriptor:
    """Keep PyTorch on its flat-format parser, away from legacy tar extraction."""

    def __init__(self, file):
        self.file = file

    def read(self, size=-1):
        return self.file.read(size)

    def readline(self, size=-1):
        return self.file.readline(size)

    def readinto(self, buffer):
        return self.file.readinto(buffer)

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()


def _extract_module_tensors(module):
    if type(module) is not _MODULE_CLASSES["torchxrayvision.models.DenseNet"]:
        raise ValueError("Checkpoint root must be the reviewed TorchXRayVision DenseNet")
    result, active = OrderedDict(), set()

    def visit(current, prefix):
        if id(current) in active or not hasattr(current, "state"):
            raise pickle.UnpicklingError("Cyclic or incomplete TorchXRayVision module state")
        active.add(id(current))
        state = current.state
        # Pickle memo aliases may mutate registries after their BUILD validation.
        if not _valid_module_state(current, state):
            raise pickle.UnpicklingError("Unsupported TorchXRayVision checkpoint module state or hooks")
        nonpersistent = state.get("_non_persistent_buffers_set", set())
        for field in ("_parameters", "_buffers"):
            for name, value in state[field].items():
                if value is not None and not (field == "_buffers" and name in nonpersistent):
                    key = prefix + name
                    if key in result:
                        raise ValueError(f"Duplicate checkpoint tensor: {key}")
                    result[key] = value.detach().cpu()
        for name, child in state["_modules"].items():
            if child is not None:
                visit(child, prefix + name + ".")
        active.remove(id(current))

    visit(module, "")
    if not result:
        raise ValueError("TorchXRayVision checkpoint contains no tensors")
    return result


def load_xrv_densenet_state(path):
    """Return CPU tensors from the reviewed XRV module or a tensor state dict.

    Accept flat Torch legacy files and ordinary Torch ZIP files. Reject older
    tar archives, TorchScript, unknown classes, custom constructors and hooks.
    The caller must verify the desired architecture and artifact provenance.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    restricted_pickle = _checkpoint_pickle_facade()
    with path.open("rb") as file:
        # Match Torch's leading ZIP signature: an embedded ZIP footer inside
        # a flat pickle is not evidence that the outer checkpoint is a ZIP.
        is_zip = file.read(4) == b"PK\x03\x04"
        file.seek(0)
        if is_zip:
            with zipfile.ZipFile(file) as archive:
                if any(name.rsplit("/", 1)[-1] == "constants.pkl" for name in archive.namelist()):
                    raise pickle.UnpicklingError("TorchScript archives are unsupported")
        else:
            file.seek(0)
            try:
                magic = _HeaderUnpickler(file).load()
                version = _HeaderUnpickler(file).load()
            except (pickle.UnpicklingError, EOFError, ValueError, IndexError) as error:
                raise pickle.UnpicklingError("Expected a Torch checkpoint; legacy tar archives are unsupported") from error
            if type(magic) is not int or magic != torch.serialization.MAGIC_NUMBER or type(version) is not int or version != torch.serialization.PROTOCOL_VERSION:
                raise pickle.UnpicklingError("Expected a Torch checkpoint; legacy tar archives are unsupported")
        file.seek(0)
        # weights_only=False selects our local restricted parser, never ordinary
        # unrestricted pickle. No file descriptor means no legacy tar fallback.
        payload = torch.load(_NoFileDescriptor(file), map_location="cpu", weights_only=False,
                             pickle_module=restricted_pickle)
    if isinstance(payload, _InertModule):
        return _extract_module_tensors(payload)
    if _mapping(payload):
        for key in ("state_dict", "model"):
            if key in payload and _mapping(payload[key]):
                payload = payload[key]
                break
        if payload and all(type(key) is str and type(value) in (torch.Tensor, torch.nn.Parameter)
                           for key, value in payload.items()):
            return OrderedDict((key, value.detach().cpu()) for key, value in payload.items())
    raise ValueError("Checkpoint must contain a DenseNet module or a nonempty tensor state dict")
