"""Repair the real published core offline while retaining every saved result."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import repair_distillation_roundtrip as updater


class DistillationRoundtripRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(cls.repo / "packages/FU254-distillation-abcd-update.zip") as archive:
            cls.original = archive.read("distillation_abcd_update/foundation_sft/distillation.py")
        if hashlib.sha256(cls.original).hexdigest() != updater.SOURCE_SHA256:
            raise AssertionError("The fixture must be the real published official core")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="roundtrip-repair-")
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.target = self.project / updater.CORE
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.original)
        self.target.chmod(0o640)
        self.saved = {}
        for index in range(17):
            relative = f"runs/distillation_ablation/suites/keep/jobs/group{index}/result.json"
            self.saved[relative] = json.dumps({"completed": True, "index": index}).encode()
        self.saved.update({
            "foundation_sft/configs/distillation_ablation.json": b'{"settings":"keep"}\n',
            "foundation_sft/models/vit_tiny_student.py": b"KEEP_STUDENT = True\n",
            "foundation_sft/distillation_losses.py": b"KEEP_LOSSES = True\n",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"keep cached pixels",
            "foundation_weights/vit_tiny/keep.pth": b"keep pretrained weights",
        })
        for relative, data in self.saved.items():
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def assert_saved(self):
        for relative, data in self.saved.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)

    def assert_original(self):
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        self.assert_saved()

    def test_real_official_fragment_patch_matches_the_reviewed_core_exactly(self):
        # This historical repair targets its published version, while today's
        # core may add new experiments. Use the release's independent checksum
        # as the oracle rather than requiring the working tree to remain old.
        with zipfile.ZipFile(self.repo / "packages/FU254-distillation-roundtrip-fix.zip") as archive:
            prefix = "distillation_roundtrip_fix/"
            published = archive.read(prefix + "repair_distillation_roundtrip.py")
            hashes = dict((relative, checksum) for checksum, relative in (
                line.split("  ", 1) for line in archive.read(prefix + "SHA256SUMS.txt").decode().splitlines()))
        self.assertEqual(hashlib.sha256(published).hexdigest(), hashes["repair_distillation_roundtrip.py"])
        targets = [node.value.value for node in ast.parse(published.decode()).body
                   if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                   and any(isinstance(name, ast.Name) and name.id == "TARGET_SHA256" for name in node.targets)]
        self.assertEqual(len(targets), 1)
        repaired = updater._build(self.original)
        self.assertEqual(updater.TARGET_SHA256, targets[0])
        self.assertEqual(hashlib.sha256(repaired).hexdigest(), targets[0])
        compile(repaired, updater.CORE, "exec")

    def test_preview_creates_no_backup_or_lock_and_preserves_every_result(self):
        result = updater.repair(self.project)
        self.assertEqual(result["status"], "would_repair")
        self.assertIsNone(result["backup"])
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assertFalse((self.project / updater.LOCK_NAME).exists())
        self.assert_original()

    def test_apply_backs_up_full_bytes_permissions_and_is_idempotent(self):
        result = updater.repair(self.project, True)
        self.assertEqual(result["status"], "repaired")
        self.assertEqual(updater._digest(self.target.read_bytes()), updater.TARGET_SHA256)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        backup = Path(result["backup"])
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        self.assertEqual((backup / "distillation.py").read_bytes(), self.original)
        self.assertEqual((backup / "distillation.py").stat().st_mode & 0o777, 0o640)
        journal = json.loads((backup / "repair.json").read_text())
        self.assertEqual(journal["source_sha256"], updater.SOURCE_SHA256)
        self.assertEqual(journal["target_sha256"], updater.TARGET_SHA256)
        self.assertEqual(journal["previous_mode"], 0o640)
        self.assertEqual(updater.repair(self.project, True)["status"], "already_repaired")
        self.assertEqual(len(list((self.project / updater.BACKUP_DIRECTORY).iterdir())), 1)
        self.assert_saved()

    def test_unknown_user_edit_or_truncated_core_is_refused_before_writing(self):
        edited = self.original + b"\nUSER_LOCAL_EDIT = True\n"
        self.target.write_bytes(edited)
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            updater.repair(self.project, True)
        self.assertEqual(self.target.read_bytes(), edited)
        self.target.write_bytes(self.original[:100])
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            updater.repair(self.project, True)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_saved()

    def test_patch_fragments_must_each_be_unique_and_produce_the_exact_target(self):
        for previous in (b"fragment-not-present", b" "):
            with patch.object(updater, "PATCHES", ((previous, b"replacement"),)):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    updater.repair(self.project, True)
        changed = tuple(updater.PATCHES[:-1]) + ((updater.PATCHES[-1][0], updater.PATCHES[-1][1] + b"# tamper\n"),)
        with patch.object(updater, "PATCHES", changed):
            with self.assertRaisesRegex(ValueError, "reviewed target"):
                updater.repair(self.project, True)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_original()

    def test_invalid_python_is_rejected_even_if_a_fake_target_checksum_matches(self):
        broken = b"def invalid(:\n"
        candidate = self.original.replace(updater.PATCHES[0][0], broken, 1)
        with patch.object(updater, "PATCHES", ((updater.PATCHES[0][0], broken),)), patch.object(
                updater, "TARGET_SHA256", updater._digest(candidate)):
            with self.assertRaises(SyntaxError):
                updater.repair(self.project, True)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_original()

    def test_symlink_target_and_backup_directory_are_refused(self):
        outside = self.project / "outside.py"
        outside.write_bytes(self.original)
        self.target.unlink()
        self.target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            updater.repair(self.project, True)
        self.target.unlink()
        self.target.write_bytes(self.original)
        self.target.chmod(0o640)
        outside_directory = self.project / "outside"
        outside_directory.mkdir()
        (self.project / updater.BACKUP_DIRECTORY).symlink_to(outside_directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            updater.repair(self.project, True)
        self.assertEqual(list(outside_directory.iterdir()), [])
        self.assert_original()

    def test_existing_lock_and_old_python_preserve_all_files(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another repair")
        with self.assertRaises(FileExistsError):
            updater.repair(self.project, True)
        self.assertEqual(lock.read_text(), "another repair")
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)), patch.object(
                updater, "_path", side_effect=AssertionError("Old Python must not read the core")):
            with self.assertRaisesRegex(ValueError, "conda activate evax_lora"):
                updater.repair(self.project, True)
        self.assert_original()

    def test_atomic_replace_failure_leaves_old_core_and_complete_backup(self):
        with patch.object(updater.os, "replace", side_effect=OSError("Synthetic write interruption")):
            with self.assertRaisesRegex(OSError, "interruption"):
                updater.repair(self.project, True)
        backups = list((self.project / updater.BACKUP_DIRECTORY).iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "distillation.py").read_bytes(), self.original)
        self.assertFalse((self.project / updater.LOCK_NAME).exists())
        self.assertEqual(list(self.target.parent.glob(".roundtrip-repair-*.tmp")), [])
        self.assert_original()

    def test_concurrent_user_edit_during_temp_write_is_not_replaced(self):
        original_fsync = updater.os.fsync

        def edit_during_write(descriptor):
            self.target.write_bytes(b"USER_CONCURRENT_EDIT = True\n")
            return original_fsync(descriptor)

        with patch.object(updater.os, "fsync", side_effect=edit_during_write):
            with self.assertRaisesRegex(ValueError, "later local changes preserved"):
                updater.repair(self.project, True)
        self.assertEqual(self.target.read_bytes(), b"USER_CONCURRENT_EDIT = True\n")
        self.assertFalse((self.project / updater.LOCK_NAME).exists())
        self.assert_saved()

    def test_edit_after_publication_is_reported_and_preserved(self):
        original_replace = updater.os.replace

        def replace_then_edit(source, destination):
            original_replace(source, destination)
            self.target.write_bytes(b"USER_AFTER_PUBLICATION = True\n")

        with patch.object(updater.os, "replace", side_effect=replace_then_edit):
            with self.assertRaisesRegex(RuntimeError, "after publication.*local changes preserved"):
                updater.repair(self.project, True)
        self.assertEqual(self.target.read_bytes(), b"USER_AFTER_PUBLICATION = True\n")
        self.assert_saved()

    def test_standalone_cli_works_without_site_packages_or_torch(self):
        command = [sys.executable, "-S", str(Path(updater.__file__).resolve()), "--project", str(self.project)]
        preview = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(preview["status"], "would_repair")
        self.assert_original()
        applied = json.loads(subprocess.check_output(command + ["--apply"], text=True))
        self.assertEqual(applied["status"], "repaired")
        self.assertEqual(updater._digest(self.target.read_bytes()), updater.TARGET_SHA256)
        self.assert_saved()


if __name__ == "__main__":
    unittest.main()
