"""Offline VLM installation is checked, reversible, and preserves experiments."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import update_foundation_vlm as updater


class VLMUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="vlm-installer-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project, self.source = root / "project", root / "uploaded"
        self.trainer = self.project / "foundation_sft/train.py"
        self.trainer.parent.mkdir(parents=True)
        self.old = b"VERSION = 'old'\n"
        self.new = b"VERSION = 'vlm'\n"
        self.trainer.write_bytes(self.old)
        self.trainer.chmod(0o640)
        self.payloads = {"foundation_sft/train.py": self.new,
                         "foundation_sft/models/new.py": b"VISUAL_ONLY = True\n",
                         "foundation_sft/configs/new.json": b'{"base_config":"common.json"}\n'}
        for relative, data in self.payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.manifest = {name: {"updated": updater._digest(data),
                               "previous": [updater._digest(self.old)] if name.endswith("train.py") else [None]}
                         for name, data in self.payloads.items()}
        hashes = patch.object(updater, "FILES", self.manifest)
        hashes.start()
        self.addCleanup(hashes.stop)
        self.preserved = self.project / "foundation_sft/configs/common.json"
        self.preserved.parent.mkdir(parents=True)
        self.preserved.write_bytes(b'{"data":"user-specific"}\n')

    def install(self, apply=False):
        with redirect_stdout(io.StringIO()):
            return updater.install(self.project, self.source, apply=apply)

    def test_preview_changes_no_sources_and_apply_backs_up_and_is_idempotent(self):
        self.assertIsNone(self.install())
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / "runs").exists())
        backup = self.install(True)
        self.assertEqual((backup / "foundation_sft/train.py").read_bytes(), self.old)
        self.assertEqual(self.trainer.read_bytes(), self.new)
        self.assertEqual(self.trainer.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.preserved.read_bytes(), b'{"data":"user-specific"}\n')
        record = json.loads((backup / "installation.json").read_text())
        self.assertEqual(set(record["new_files"]), set(self.payloads) - {"foundation_sft/train.py"})
        self.assertIsNone(self.install(True))
        self.assertFalse((self.project / ".foundation-vlm-update.lock").exists())

    def test_tampered_payload_and_unknown_local_edits_change_no_sources(self):
        source = self.source / "foundation_sft/configs/new.json"
        correct = source.read_bytes()
        source.write_bytes(correct + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install(True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / "runs").exists())
        source.write_bytes(correct)
        self.trainer.write_bytes(self.old + b"CUSTOM = True\n")
        with self.assertRaisesRegex(ValueError, "local changes"):
            self.install(True)
        self.assertFalse((self.project / "foundation_sft/models/new.py").exists())

    def test_nested_symbolic_link_destination_is_refused(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        (self.project / "foundation_sft/models").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse(list(outside.iterdir()))

    def test_failed_late_replace_restores_old_files_and_removes_new_files(self):
        original_replace = updater.os.replace

        def fail_config(source, destination):
            if str(destination).endswith("new.json"):
                raise OSError("Synthetic interrupted update")
            original_replace(source, destination)

        with patch.object(updater.os, "replace", side_effect=fail_config):
            with self.assertRaisesRegex(OSError, "interrupted update"):
                self.install(True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertEqual(self.trainer.stat().st_mode & 0o777, 0o640)
        self.assertFalse((self.project / "foundation_sft/models/new.py").exists())
        self.assertFalse((self.project / ".foundation-vlm-update.lock").exists())

    def test_existing_installation_lock_is_retained(self):
        lock = self.project / ".foundation-vlm-update.lock"
        lock.write_text("another installer")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "another installer")
        self.assertEqual(self.trainer.read_bytes(), self.old)

    def test_old_python_is_rejected_before_creating_lock_or_reading_payloads(self):
        with patch.object(updater.sys, "version_info", (3, 7, 17)), patch.object(
                updater, "_path", side_effect=AssertionError("Old interpreter must not read payloads")):
            with self.assertRaisesRegex(ValueError, "conda activate evax_lora"):
                self.install(True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / ".foundation-vlm-update.lock").exists())
        self.assertFalse((self.project / "runs").exists())

    def test_invalid_python_payload_leaves_no_lock_or_project_changes(self):
        relative = "foundation_sft/models/new.py"
        invalid = b"def broken(:\n"
        (self.source / relative).write_bytes(invalid)
        self.manifest[relative]["updated"] = updater._digest(invalid)
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / ".foundation-vlm-update.lock").exists())
        self.assertFalse((self.project / "runs").exists())

    def test_targeted_fix_skips_unrelated_local_edits_and_missing_payloads(self):
        selected = "foundation_sft/models/new.py"
        self.trainer.write_bytes(self.old + b"CUSTOM = True\n")
        before = self.trainer.read_bytes()
        (self.source / "foundation_sft/configs/new.json").unlink()
        with patch.object(updater, "XRAY_UPDATE_FILES", (selected,)), redirect_stdout(io.StringIO()):
            backup = updater.install(self.project, self.source, apply=True, xray_siglip_only=True)
        self.assertEqual(self.trainer.read_bytes(), before)
        self.assertEqual((self.project / selected).read_bytes(), self.payloads[selected])
        self.assertFalse((self.project / "foundation_sft/configs/new.json").exists())
        self.assertEqual(json.loads((backup / "installation.json").read_text())["files"], [selected])
        self.assertEqual(self.preserved.read_bytes(), b'{"data":"user-specific"}\n')

    def test_targeted_fix_does_not_bypass_selected_file_checks(self):
        selected = "foundation_sft/models/new.py"
        (self.source / selected).write_bytes(b"TAMPERED = True\n")
        with patch.object(updater, "XRAY_UPDATE_FILES", (selected,)), self.assertRaisesRegex(ValueError, "checksum"):
            updater.install(self.project, self.source, apply=True, xray_siglip_only=True)
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / selected).exists())
        self.assertFalse((self.project / ".foundation-vlm-update.lock").exists())


class VLMManifestTests(unittest.TestCase):
    def test_published_manifest_matches_payloads_and_excludes_existing_data_configs(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(updater.FILES)
        self.assertNotIn("foundation_sft/configs/common.json", updater.FILES)
        self.assertNotIn("foundation_sft/configs/experiments.json", updater.FILES)
        for relative, record in updater.FILES.items():
            with self.subTest(relative=relative):
                self.assertFalse(relative.startswith(("runs/", "foundation_weights/")))
                self.assertEqual(updater._digest((root / relative).read_bytes()), record["updated"])


if __name__ == "__main__":
    unittest.main()
