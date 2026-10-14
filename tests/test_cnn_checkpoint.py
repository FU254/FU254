"""Restricted XRV module loading, genuine tensor roundtrips and hostile pickle."""

from collections import OrderedDict
import builtins
import codecs
import copyreg
import io
import os
from pathlib import Path
import pickle
import sys
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np
import torch
from torch import nn

from foundation_sft.models.cnn_checkpoint import (
    _MODULE_CLASSES, _XRVCheckpointUnpickler, load_xrv_densenet_state,
)
from foundation_sft.models.common import sha256_file


class _ExecuteOnUnpickle:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",)


class _ModuleReduce:
    def __reduce__(self):
        return nn.ReLU, ()


class _TensorWithHooks:
    def __reduce__(self):
        value = torch.arange(4, dtype=torch.float32)
        return torch._utils._rebuild_tensor_v2, (
            value.storage(), 0, (4,), (1,), False, OrderedDict([(1, 1)]),
        )


class _TensorOutOfBounds:
    def __reduce__(self):
        return torch._utils._rebuild_tensor_v2, (
            torch.tensor([123.0, 456.0]).storage(), 3, (1,), (1,), False, OrderedDict(),
        )


class _ParameterWithHooks:
    def __reduce__(self):
        return torch._utils._rebuild_parameter, (torch.ones(4), True, OrderedDict([(1, 1)]))


class _NestedStorageLoader:
    def __init__(self, payload):
        self.payload = payload

    def __reduce__(self):
        return torch.storage._load_from_bytes, (self.payload,)


class _CustomCodec:
    def __reduce__(self):
        return codecs.encode, ("payload", "rot_13")


def _tiny_init(self):
    nn.Module.__init__(self)
    self.features = nn.Sequential(OrderedDict([
        ("conv0", nn.Conv2d(1, 2, kernel_size=3, bias=False)),
        ("norm0", nn.BatchNorm2d(2)),
        ("relu0", nn.ReLU(inplace=True)),
        ("pool0", nn.MaxPool2d(2)),
        ("avgpool", nn.AvgPool2d(2)),
    ]))
    self.classifier = nn.Linear(2, 18)
    self.register_buffer("op_thresh", torch.arange(18, dtype=torch.float32))
    self.register_buffer("temporary", torch.tensor(1.0), persistent=False)


class XRVCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "checkpoint.pt"
        self.marker = self.root / "executed.txt"
        self.state = OrderedDict([
            ("features.conv0.weight", torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)),
            ("features.norm0.num_batches_tracked", torch.tensor(5)),
            ("classifier.weight", torch.arange(36, dtype=torch.float32).reshape(18, 2)),
        ])
        self.state._metadata = OrderedDict([("", {"version": 1})])

    def tearDown(self):
        self.directory.cleanup()

    def save(self, value, *, zipped=False, protocol=2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            torch.save(value, self.path, pickle_protocol=protocol,
                       _use_new_zipfile_serialization=zipped)

    def save_module(self, *, zipped=False, protocol=2, mutate=None):
        # This tiny fixture serializes genuine nn.Module registries with the
        # official XRV global name, without installing or importing XRV.
        package = ModuleType("torchxrayvision")
        package.__path__ = []
        models = ModuleType("torchxrayvision.models")
        models.DenseNet = type("DenseNet", (nn.Module,), {
            "__module__": models.__name__, "__init__": _tiny_init,
        })
        package.models = models
        with mock.patch.dict(sys.modules, {package.__name__: package, models.__name__: models}):
            model = models.DenseNet()
            expected = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())
            if mutate is not None:
                mutate(model)
            self.save(model, zipped=zipped, protocol=protocol)
        return expected

    def assert_tensors(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in actual.items():
            self.assertIs(type(value), torch.Tensor)
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(value, expected[key])

    def test_genuine_tensor_state_dicts_in_zip_and_flat_formats(self):
        for zipped in (False, True):
            for protocol in (2, 3, 4):
                with self.subTest(zipped=zipped, protocol=protocol):
                    self.save(self.state, zipped=zipped, protocol=protocol)
                    self.assert_tensors(load_xrv_densenet_state(self.path), self.state)

    def test_safe_numpy_scalar_set_and_parameter_metadata(self):
        for zipped in (False, True):
            for protocol in (2, 3, 4):
                with self.subTest(zipped=zipped, protocol=protocol):
                    state = OrderedDict(self.state)
                    state["classifier.bias"] = nn.Parameter(torch.ones(18))
                    self.save({"model": state, "best_auc": np.float64(0.875),
                               "missing_keys": {"unused"}}, zipped=zipped, protocol=protocol)
                    self.assert_tensors(load_xrv_densenet_state(self.path), state)

    def test_full_module_extracts_only_persistent_tensors_without_model_import(self):
        original_import = builtins.__import__

        def no_xrv_import(name, *args, **kwargs):
            if name.startswith("torchxrayvision"):
                raise AssertionError("XRV must not be imported while loading")
            return original_import(name, *args, **kwargs)

        for zipped in (False, True):
            for protocol in (2, 3, 4):
                with self.subTest(zipped=zipped, protocol=protocol):
                    expected = self.save_module(zipped=zipped, protocol=protocol)
                    with mock.patch("builtins.__import__", side_effect=no_xrv_import), \
                            mock.patch.object(nn.Module, "__setstate__", side_effect=AssertionError("active BUILD")), \
                            mock.patch.object(nn.Conv2d, "__init__", side_effect=AssertionError("active constructor")):
                        actual = load_xrv_densenet_state(self.path)
                    self.assert_tensors(actual, expected)
                    self.assertNotIn("temporary", actual)
                    self.assertIn("op_thresh", actual)

    def test_hostile_global_is_rejected_without_execution_across_formats_and_protocols(self):
        for zipped in (False, True):
            for protocol in (2, 3, 4):
                with self.subTest(zipped=zipped, protocol=protocol):
                    self.save({"model": self.state, "metadata": np.float64(0.9),
                               "attack": _ExecuteOnUnpickle(self.marker)}, zipped=zipped, protocol=protocol)
                    with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*eval"):
                        load_xrv_densenet_state(self.path)
                    self.assertFalse(self.marker.exists())

    def test_known_module_class_cannot_be_called_through_reduce(self):
        self.save({"model": self.state, "attack": _ModuleReduce()})
        with mock.patch.object(nn.ReLU, "__init__", side_effect=AssertionError("constructor executed")):
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unrecognized checkpoint reconstruction"):
                load_xrv_densenet_state(self.path)

    def test_nonempty_module_hooks_and_unreviewed_fields_are_rejected(self):
        mutations = (
            lambda model: model._forward_hooks.update({1: 1}),
            lambda model: setattr(model, "unreviewed", "metadata"),
            lambda model: setattr(model, "_is_full_backward_hook", "unreviewed"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.save_module(mutate=mutate)
                with self.assertRaisesRegex(pickle.UnpicklingError, "module state or hooks"):
                    load_xrv_densenet_state(self.path)

    def test_tensor_and_parameter_backward_hooks_are_rejected(self):
        for attack in (_TensorWithHooks(), _ParameterWithHooks()):
            with self.subTest(attack=type(attack).__name__):
                self.save({"model": self.state, "attack": attack})
                with self.assertRaisesRegex(pickle.UnpicklingError, "tensor hooks are unsupported"):
                    load_xrv_densenet_state(self.path)

    def test_tensor_layout_cannot_resize_beyond_checkpoint_storage(self):
        for zipped in (False, True):
            with self.subTest(zipped=zipped):
                self.save({"model": self.state, "attack": _TensorOutOfBounds()}, zipped=zipped)
                with self.assertRaisesRegex(pickle.UnpicklingError, "layout exceeds its stored data"):
                    load_xrv_densenet_state(self.path)

    def test_cyclic_module_graph_is_rejected(self):
        self.save_module(mutate=lambda model: model.add_module("cycle", model))
        with self.assertRaisesRegex(pickle.UnpicklingError, "Cyclic or incomplete"):
            load_xrv_densenet_state(self.path)

    def test_embedded_module_source_is_discarded_without_execution(self):
        cls = b"ctorchxrayvision.models\nDenseNet\n"
        source = f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')"
        # BINPERSID carries class + saved source. The result is its inert token,
        # and the source string is never handed to torch's source checker.
        source_pickle = pickle.dumps(source, protocol=2)[2:-1]
        payload = b"\x80\x02(X\x06\x00\x00\x00module" + cls + b"N" + source_pickle + b"tQ."
        token = _XRVCheckpointUnpickler(io.BytesIO(payload)).load()
        self.assertIs(token, _MODULE_CLASSES["torchxrayvision.models.DenseNet"])
        self.assertFalse(self.marker.exists())

    def test_unknown_storage_namespace_cannot_bypass_find_class(self):
        payload = b"\x80\x02cattacker\nFloatStorage\n."
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global: attacker.FloatStorage"):
            _XRVCheckpointUnpickler(io.BytesIO(payload)).load()

    def test_extension_cache_cannot_bypass_global_allowlist(self):
        code = max([255, *copyreg._inverted_registry]) + 1
        copyreg.add_extension("builtins", "eval", code)
        try:
            copyreg._extension_cache[code] = eval
            self.save({"model": self.state, "attack": _ExecuteOnUnpickle(self.marker)})
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint object/extension opcode"):
                load_xrv_densenet_state(self.path)
        finally:
            copyreg.remove_extension("builtins", "eval", code)
        self.assertFalse(self.marker.exists())

    def test_nested_pickle_and_nonliteral_codec_are_rejected(self):
        payload = io.BytesIO()
        torch.save(_ExecuteOnUnpickle(self.marker), payload)
        for attack, message in (
                (_NestedStorageLoader(payload.getvalue()), "Unsupported checkpoint global:.*_load_from_bytes"),
                (_CustomCodec(), "Only literal latin1 bytes")):
            with self.subTest(attack=type(attack).__name__):
                self.save({"model": self.state, "attack": attack})
                with self.assertRaisesRegex(pickle.UnpicklingError, message):
                    load_xrv_densenet_state(self.path)
                self.assertFalse(self.marker.exists())

    def test_legacy_tar_is_rejected_before_torch_or_tar_extraction(self):
        with tarfile.open(self.path, "w") as archive:
            data = b"untrusted"
            entry = tarfile.TarInfo("storages")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        with mock.patch("torch.load", side_effect=AssertionError("legacy tar entered Torch")), \
                mock.patch("tarfile.TarFile.extract", side_effect=AssertionError("tar extracted")):
            with self.assertRaisesRegex(pickle.UnpicklingError, "legacy tar archives are unsupported"):
                load_xrv_densenet_state(self.path)

    def test_flat_tensor_format_never_enters_legacy_tar_loader(self):
        self.save(self.state)
        with mock.patch("tarfile.open", side_effect=AssertionError("tar parser entered")):
            self.assert_tensors(load_xrv_densenet_state(self.path), self.state)

    def test_flat_storage_footer_must_cover_every_allocated_tensor(self):
        self.save({"weight": torch.tensor([123.0, 456.0])})
        data = self.path.read_bytes()
        with self.path.open("rb") as file:
            for _ in range(3):
                _XRVCheckpointUnpickler(file).load()
            # Locate the genuine final table using Torch's restricted parser.
            from foundation_sft.models.cnn_checkpoint import _checkpoint_pickle_facade, _NoFileDescriptor
            positions = []
            facade = _checkpoint_pickle_facade()
            original_load = facade.load

            def track_framing(stream, **kwargs):
                start = stream.tell()
                result = original_load(stream, **kwargs)
                positions.append((start, stream.tell()))
                return result

            facade.load = track_framing
            file.seek(0)
            torch.load(_NoFileDescriptor(file), map_location="cpu", weights_only=False, pickle_module=facade)
        start, end = positions[3]
        for keys in ([], ["unregistered"], ["duplicate", "duplicate"]):
            with self.subTest(keys=keys):
                self.path.write_bytes(data[:start] + pickle.dumps(keys, protocol=2) + data[end:])
                with self.assertRaisesRegex(pickle.UnpicklingError, "storage table does not cover"):
                    load_xrv_densenet_state(self.path)

    def test_torchscript_zip_is_rejected_before_torch_jit_dispatch(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("archive/constants.pkl", b"untrusted")
        with mock.patch("torch.load", side_effect=AssertionError("TorchScript entered Torch")), \
                mock.patch("torch.jit.load", side_effect=AssertionError("TorchScript dispatched")):
            with self.assertRaisesRegex(pickle.UnpicklingError, "TorchScript archives are unsupported"):
                load_xrv_densenet_state(self.path)

    def test_missing_and_nontensor_payloads_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_xrv_densenet_state(self.path)
        self.save({"model": {"weight": "not a tensor"}})
        with self.assertRaisesRegex(ValueError, "nonempty tensor state dict"):
            load_xrv_densenet_state(self.path)

    @unittest.skipUnless(os.environ.get("XRV_DENSENET_CHECKPOINT"), "official artifact path not supplied")
    def test_verified_official_artifact_matches_native_densenet121(self):
        from torchvision.models import densenet121

        path = Path(os.environ["XRV_DENSENET_CHECKPOINT"])
        self.assertEqual(sha256_file(path), "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899")
        state = load_xrv_densenet_state(path)
        self.assertEqual(len(state), 727)
        self.assertEqual(tuple(state["features.conv0.weight"].shape), (64, 1, 7, 7))
        self.assertEqual(tuple(state["classifier.weight"].shape), (18, 1024))
        native = densenet121(weights=None, num_classes=18)
        native.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        native.load_state_dict(state, strict=True)
        self.assert_tensors(state, native.state_dict())


if __name__ == "__main__":
    unittest.main()
