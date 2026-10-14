"""A scoped CNN update preserves data settings, checkpoints and old code."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import update_foundation_cnn as updater


class CNNUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cnn-update-")
        self.root = Path(self.temporary.name)
        self.project, self.source = self.root / "project", self.root / "source"
        self.old_registry = b"SUPPORTED_MODELS = ('eva_x',)\n"
        self.new_registry = b"SUPPORTED_MODELS = ('eva_x', 'densenet121_cxr')\n"
        self.payloads = {updater.REGISTRY: self.new_registry,
                         "foundation_sft/models/cnn.py": b"CNN_ONLY = True\n",
                         "foundation_sft/configs/common_cnn.json": b'{"base_config":"common.json"}\n'}
        self.registry = self.project / updater.REGISTRY
        self.registry.parent.mkdir(parents=True)
        self.registry.write_bytes(self.old_registry)
        for relative, data in self.payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.preserved = self.project / "foundation_sft/configs/common.json"
        self.preserved.parent.mkdir(parents=True)
        self.preserved.write_bytes(b'{"data":"user-custom","split_manifest":"original.csv"}\n')
        self.trainer = self.project / "foundation_sft/train.py"
        self.trainer.write_bytes(b"EXISTING_CACHE_AND_VLM = True\n")
        self.checkpoint = self.project / "runs/old/best.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"OLD_TRAINED_WEIGHTS")
        self.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in self.payloads.items()}
        self.maps = patch.object(updater, "PAYLOAD_SHA256", self.hashes)
        self.previous = patch.object(updater, "PREVIOUS_REGISTRIES", {updater._digest(self.old_registry)})
        self.maps.start()
        self.previous.start()

    def tearDown(self):
        self.previous.stop()
        self.maps.stop()
        self.temporary.cleanup()

    def test_dry_plan_does_not_change_or_create_project_files(self):
        before = {str(path): path.read_bytes() for path in self.project.rglob("*") if path.is_file()}
        result = updater.install(self.project, self.source)
        self.assertEqual(set(result["files"]), set(self.payloads))
        after = {str(path): path.read_bytes() for path in self.project.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_apply_preserves_previous_sources_and_user_experiment_inputs(self):
        result = updater.install(self.project, self.source, apply=True)
        self.assertEqual(result["files"][-1], updater.REGISTRY)
        self.assertEqual(self.registry.read_bytes(), self.new_registry)
        self.assertEqual((Path(result["backup"]) / updater.REGISTRY).read_bytes(), self.old_registry)
        self.assertIn(b"user-custom", self.preserved.read_bytes())
        self.assertEqual(self.trainer.read_bytes(), b"EXISTING_CACHE_AND_VLM = True\n")
        self.assertEqual(self.checkpoint.read_bytes(), b"OLD_TRAINED_WEIGHTS")
        self.assertFalse((self.project / ".foundation-cnn-update.lock").exists())
        again = updater.install(self.project, self.source, apply=True)
        self.assertEqual(again["files"], [])

    def test_tampered_payload_or_unexpected_local_registry_refuses_all_writes(self):
        path = self.source / "foundation_sft/models/cnn.py"
        path.write_bytes(b"CHANGED_SOURCE = True\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            updater.install(self.project, self.source, apply=True)
        self.assertEqual(self.registry.read_bytes(), self.old_registry)
        path.write_bytes(self.payloads["foundation_sft/models/cnn.py"])
        self.registry.write_bytes(self.old_registry + b"USER_EDIT = True\n")
        with self.assertRaisesRegex(ValueError, "Unexpected local"):
            updater.install(self.project, self.source, apply=True)
        self.assertFalse((self.project / "foundation_sft/models/cnn.py").exists())

    def test_failure_during_registry_activation_rolls_back_new_files(self):
        write = updater._atomic_write

        def fail_registry(path, data, mode):
            if path == self.registry and data == self.new_registry:
                raise OSError("Synthetic final activation failure")
            write(path, data, mode)

        with patch.object(updater, "_atomic_write", side_effect=fail_registry), self.assertRaises(OSError):
            updater.install(self.project, self.source, apply=True)
        self.assertEqual(self.registry.read_bytes(), self.old_registry)
        self.assertFalse((self.project / "foundation_sft/models/cnn.py").exists())
        self.assertFalse((self.project / ".foundation-cnn-update.lock").exists())


class CNNPublishedManifestTests(unittest.TestCase):
    def test_exact_addon_manifest_preserves_all_existing_training_assets(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(len(updater.PAYLOAD_SHA256), 11)
        for relative, checksum in updater.PAYLOAD_SHA256.items():
            self.assertEqual(updater._digest((root / relative).read_bytes()), checksum)
            self.assertNotIn(relative, {"foundation_sft/train.py", "foundation_sft/data.py",
                                       "foundation_sft/configs/common.json", "foundation_sft/configs/experiments.json"})
            self.assertFalse(relative.startswith(("runs/", "foundation_weights/")))


if __name__ == "__main__":
    unittest.main()
