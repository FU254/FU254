"""Real legacy NumPy metadata and restricted checkpoint deserialization."""

from collections import OrderedDict
import codecs
import copyreg
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from foundation_sft.models.common import load_checkpoint


class _ExecuteOnUnpickle:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",)


class _NestedStorageLoader:
    def __init__(self, payload):
        self.payload = payload

    def __reduce__(self):
        return torch.storage._load_from_bytes, (self.payload,)


class _CustomCodec:
    def __reduce__(self):
        return codecs.encode, ("payload", "rot_13")


class LegacyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "checkpoint.pt"
        self.state = OrderedDict([
            ("module.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
            ("module.bias", torch.tensor([1.0, 2.0, 3.0])),
        ])
        # nn.Module.state_dict() commonly has an OrderedDict BUILD for metadata.
        self.state._metadata = OrderedDict([("", {"version": 1})])

    def tearDown(self):
        self.directory.cleanup()

    def test_plain_tensor_checkpoint_uses_native_weights_only_loader(self):
        torch.save({"model": self.state}, self.path)
        with patch("foundation_sft.models.common._load_numeric_checkpoint", side_effect=AssertionError("fallback")):
            loaded = load_checkpoint(self.path)
        torch.testing.assert_close(loaded["weight"], self.state["module.weight"])

    def test_real_legacy_scalar_checkpoint_preserves_tensors_and_prefix_handling(self):
        torch.save({"model": self.state, "best_auc": np.float64(0.875),
                    "epoch": np.int64(12), "loss": np.float32(0.125),
                    "flag": np.bool_(True), "complex": np.complex64(1 + 2j)}, self.path)
        loaded = load_checkpoint(self.path)
        self.assertEqual(set(loaded), {"weight", "bias"})
        for key, tensor in loaded.items():
            torch.testing.assert_close(tensor, self.state["module." + key])
            self.assertEqual(tensor.device.type, "cpu")

    def test_explicit_nested_state_key_survives_numeric_metadata(self):
        torch.save({"teacher": {"backbone": self.state}, "best_auc": np.float64(0.9)}, self.path)
        loaded = load_checkpoint(self.path, "teacher.backbone")
        torch.testing.assert_close(loaded["weight"], self.state["module.weight"])

    def test_malicious_reduce_after_scalar_is_rejected_without_execution(self):
        marker = self.root / "executed.txt"
        torch.save({"best_auc": np.float64(0.9), "model": self.state,
                    "attack": _ExecuteOnUnpickle(marker)}, self.path)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*eval"):
            load_checkpoint(self.path)
        self.assertFalse(marker.exists())

    def test_unknown_class_does_not_activate_numeric_fallback(self):
        marker = self.root / "executed.txt"
        torch.save({"model": self.state, "attack": _ExecuteOnUnpickle(marker)}, self.path)
        with patch("foundation_sft.models.common._load_numeric_checkpoint", side_effect=AssertionError("fallback")):
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(self.path)
        self.assertFalse(marker.exists())

    def test_nested_pickle_storage_loader_is_rejected(self):
        marker = self.root / "executed.txt"
        payload = io.BytesIO()
        torch.save(_ExecuteOnUnpickle(marker), payload)
        torch.save({"best_auc": np.float64(0.9), "model": self.state,
                    "attack": _NestedStorageLoader(payload.getvalue())}, self.path)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*_load_from_bytes"):
            load_checkpoint(self.path)
        self.assertFalse(marker.exists())

    def test_numpy_array_metadata_is_not_implicitly_allowlisted(self):
        torch.save({"best_auc": np.float64(0.9), "model": self.state,
                    "array": np.array([1.0, 2.0])}, self.path)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint global:.*_reconstruct"):
            load_checkpoint(self.path)

    def test_object_and_structured_dtypes_are_rejected(self):
        for dtype in (np.dtype(object), np.dtype([("value", "f8")])):
            with self.subTest(dtype=str(dtype)):
                torch.save({"best_auc": np.float64(0.9), "model": self.state,
                            "dtype": dtype}, self.path)
                with self.assertRaisesRegex(pickle.UnpicklingError, "Only plain numeric NumPy dtypes"):
                    load_checkpoint(self.path)

    def test_extension_cache_cannot_bypass_global_allowlist(self):
        marker = self.root / "executed.txt"
        code = max([255, *copyreg._inverted_registry]) + 1
        copyreg.add_extension("builtins", "eval", code)
        try:
            # CPython ordinarily consults this process-wide cache before calling
            # find_class; the restricted loader must reject the opcode itself.
            copyreg._extension_cache[code] = eval
            torch.save({"best_auc": np.float64(0.9), "model": self.state,
                        "attack": _ExecuteOnUnpickle(marker)}, self.path)
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported checkpoint object/extension opcode"):
                load_checkpoint(self.path)
        finally:
            copyreg.remove_extension("builtins", "eval", code)
        self.assertFalse(marker.exists())

    def test_custom_codecs_are_not_executed(self):
        torch.save({"best_auc": np.float64(0.9), "model": self.state,
                    "attack": _CustomCodec()}, self.path)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Only literal latin1 bytes"):
            load_checkpoint(self.path)

    def test_compatibility_does_not_enter_obsolete_tar_loader(self):
        torch.save({"model": self.state, "best_auc": np.float64(0.9)}, self.path,
                   _use_new_zipfile_serialization=False)
        with self.assertRaisesRegex(pickle.UnpicklingError, "requires a Torch ZIP checkpoint"):
            load_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
