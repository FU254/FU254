"""Loss-ablation installation only upgrades the reviewed coordinator and additions."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_distillation_ablation as builder
import update_distillation_ablation as updater


class DistillationAblationUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ablation-update-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project, self.source = root / "project", root / "distillation_ablation_update"
        self.coordinator = "foundation_sft/distillation_suite.py"
        self.previous = b"ORIGINAL_ABCD_COORDINATOR = True\n"
        self.preserved = {
            "foundation_sft/train.py": b"OLD_MAIN_TRAINER = True\n",
            "run_distillation_experiments.py": b"OLD_ABCD_ENTRY = True\n",
            "foundation_sft/distillation.py": b"KEEP_TRAINER = True\n",
            "foundation_sft/distillation_losses.py": b"KEEP_LOSSES = True\n",
            "foundation_sft/distillation_data.py": b"KEEP_DATASET = True\n",
            "foundation_sft/models/vit_tiny_student.py": b"KEEP_STUDENT = True\n",
            "foundation_sft/models/__init__.py": b"KEEP_REGISTRY = True\n",
            "foundation_sft/configs/distillation_abcd.json": b'{"abcd":"user-custom"}\n',
            "foundation_sft/configs/distillation_rad_dino_vit_tiny.json": b'{"full":"keep"}\n',
            "foundation_sft/configs/distillation_rad_dino_vit_tiny_low_lr.json": b'{"lr":"keep"}\n',
            "foundation_weights/vit_tiny/existing.pth": b"keep weight bytes",
            "runs/distillation_abcd/suites/old/report/comparison.csv": b"keep history bytes",
            "runs/foundation_sft/data_cache/pixels/old.npy": b"keep cached pixels",
            "runs/foundation_sft/shared_patient_split.csv": b"keep fixed split",
            "packages/FU254-distillation-abcd-update.zip": b"keep previous archive",
        }
        for relative, data in {**self.preserved, self.coordinator: self.previous}.items():
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.project / self.coordinator).chmod(0o640)
        self.payload = {}
        for relative in updater.PAYLOAD_FILES:
            data = (b"NEW_ABLATION_CODE = True\n" if relative.endswith(".py") else
                    b'{"experiment":"ablation"}\n' if relative.endswith(".json") else
                    "# 六组消融\n".encode("utf-8"))
            self.payload[relative] = data
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name, value in (
            ("PREVIOUS_SHA256", {self.coordinator: updater._digest(self.previous)}),
            ("PAYLOAD_SHA256", {name: updater._digest(data) for name, data in self.payload.items()}),
        ):
            patched = patch.object(updater, name, value)
            patched.start()
            self.addCleanup(patched.stop)

    def install(self, apply=False):
        return updater.install(self.project, self.source, apply)

    def assert_preserved(self):
        for relative, data in self.preserved.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)

    def assert_old(self):
        self.assert_preserved()
        self.assertEqual((self.project / self.coordinator).read_bytes(), self.previous)
        self.assertEqual((self.project / self.coordinator).stat().st_mode & 0o777, 0o640)
        for relative in set(self.payload) - {self.coordinator}:
            self.assertFalse((self.project / relative).exists(), relative)

    def assert_updated(self):
        self.assert_preserved()
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)
        self.assertEqual((self.project / self.coordinator).stat().st_mode & 0o777, 0o640)

    def test_preview_zero_changes_upgrade_exact_six_files_and_idempotence(self):
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_old()
        result = self.install(True)
        self.assertEqual(len(result["files"]), 6)
        backup = Path(result["backup"])
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        self.assertEqual((backup / "previous" / self.coordinator).read_bytes(), self.previous)
        journal = json.loads((backup / "installation.json").read_text())
        self.assertEqual(journal["files"][self.coordinator]["previous_mode"], 0o640)
        self.assert_updated()
        self.assertEqual(self.install(True)["files"], [])
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_unknown_coordinator_changes_or_missing_old_file_refuse_whole_upgrade(self):
        target = self.project / self.coordinator
        target.write_bytes(b"USER_COORDINATOR = True\n")
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            self.install(True)
        self.assertEqual(target.read_bytes(), b"USER_COORDINATOR = True\n")
        self.assertFalse((self.project / "run_distillation_ablation.py").exists())
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            self.install(True)
        self.assert_preserved()

    def test_existing_custom_new_config_is_never_overwritten(self):
        target = self.project / "foundation_sft/configs/distillation_ablation.json"
        target.write_bytes(b'{"user_conditions":"keep"}\n')
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(target.read_bytes(), b'{"user_conditions":"keep"}\n')
        self.assertEqual((self.project / self.coordinator).read_bytes(), self.previous)
        self.assert_preserved()

    def test_source_tampering_missing_file_and_invalid_python_prevent_all_writes(self):
        relative = updater.PAYLOAD_FILES[-1]
        path = self.source / relative
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install(True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        path.write_bytes(self.payload[relative])
        relative = "foundation_sft/distillation_ablation.py"
        (self.source / relative).write_bytes(b"def invalid(:\n")
        updater.PAYLOAD_SHA256[relative] = updater._digest(b"def invalid(:\n")
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_old()

    def test_unfrozen_and_expanded_manifest_refuse_other_project_files(self):
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        expanded = dict(updater.PAYLOAD_SHA256, **{"foundation_sft/distillation.py": "0" * 64})
        with patch.object(updater, "PAYLOAD_SHA256", expanded):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_old()

    def test_source_destination_and_backup_symbolic_links_are_rejected(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        relative = "run_distillation_ablation.py"
        target = self.project / relative
        target.symlink_to(outside / "entry.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        target.unlink()
        source = self.source / relative
        source.unlink()
        source.symlink_to(outside / "entry.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        source.unlink()
        source.write_bytes(self.payload[relative])
        (self.project / updater.BACKUP_DIRECTORY).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        self.assertEqual(list(outside.iterdir()), [])
        self.assert_old()

    def test_late_failure_restores_original_coordinator_and_removes_new_files(self):
        original_link = updater.os.link
        last = self.project / updater.PAYLOAD_FILES[-1]

        def fail_last(source, destination):
            if Path(destination) == last:
                raise OSError("Synthetic interrupted installation")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=fail_last):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.install(True)
        self.assert_old()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_automatic_rollback_preserves_and_reports_a_later_user_edit(self):
        original_link = updater.os.link
        last = self.project / updater.PAYLOAD_FILES[-1]
        target = self.project / self.coordinator

        def edit_then_fail(source, destination):
            if Path(destination) == last:
                target.write_bytes(b"LATER_USER_EDIT = True\n")
                raise OSError("Synthetic interruption")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=edit_then_fail):
            with self.assertRaisesRegex(RuntimeError, "local changes preserved.*distillation_suite.py"):
                self.install(True)
        self.assertEqual(target.read_bytes(), b"LATER_USER_EDIT = True\n")
        self.assertFalse((self.project / "run_distillation_ablation.py").exists())
        self.assert_preserved()

    def test_edit_during_atomic_temp_write_is_detected_before_replacement(self):
        original_fsync = updater.os.fsync
        target = self.project / self.coordinator
        called = False

        def edit_once(descriptor):
            nonlocal called
            if not called:
                target.write_bytes(b"USER_ATOMIC_RACE = True\n")
                called = True
            return original_fsync(descriptor)

        with patch.object(updater.os, "fsync", side_effect=edit_once):
            with self.assertRaisesRegex(ValueError, "File changed during update"):
                self.install(True)
        self.assertEqual(target.read_bytes(), b"USER_ATOMIC_RACE = True\n")
        self.assert_preserved()

    def test_existing_lock_and_unsupported_python_do_not_write(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("other installation")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "other installation")
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)), patch.object(
                updater, "_path", side_effect=AssertionError("Do not inspect payloads")):
            with self.assertRaisesRegex(ValueError, "conda activate evax_lora"):
                self.install(True)
        self.assert_old()

    def test_manual_rollback_restores_original_bytes_modes_and_is_idempotent(self):
        result = self.install(True)
        self.assertEqual(len(updater.rollback(self.project, result["backup"])["files"]), 6)
        self.assert_updated()
        updater.rollback(self.project, result["backup"], True)
        self.assert_old()
        self.assertEqual(updater.rollback(self.project, result["backup"], True)["files"], [])

    def test_manual_rollback_does_not_delete_identical_preexisting_new_code(self):
        relative = "run_distillation_ablation.py"
        (self.project / relative).write_bytes(self.payload[relative])
        result = self.install(True)
        self.assertNotIn(relative, result["files"])
        updater.rollback(self.project, result["backup"], True)
        self.assertEqual((self.project / relative).read_bytes(), self.payload[relative])
        self.assert_preserved()

    def test_rollback_refuses_local_edits_or_corrupt_backup_before_any_change(self):
        result = self.install(True)
        target = self.project / self.coordinator
        target.write_bytes(b"USER_AFTER_INSTALL = True\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, result["backup"], True)
        self.assertTrue((self.project / "run_distillation_ablation.py").is_file())
        target.write_bytes(self.payload[self.coordinator])
        saved = Path(result["backup"]) / "previous" / self.coordinator
        saved.write_bytes(b"CORRUPTED_BACKUP = True\n")
        with self.assertRaisesRegex(ValueError, "Previous backup checksum"):
            updater.rollback(self.project, result["backup"], True)
        self.assert_updated()

    def test_interrupted_rollback_restores_the_new_version(self):
        result = self.install(True)
        original_unlink = Path.unlink
        last = self.project / updater.PAYLOAD_FILES[-1]

        def fail_last(path, *args, **kwargs):
            if path == last:
                raise OSError("Synthetic interrupted rollback")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_last):
            with self.assertRaisesRegex(OSError, "interrupted rollback"):
                updater.rollback(self.project, result["backup"], True)
        self.assert_updated()

    def test_builder_preserves_old_metadata_and_packages_exact_eight_files(self):
        (self.source / "update_distillation_ablation.py").write_bytes(Path(updater.__file__).read_bytes())
        old_tool = self.source / "update_distillation_abcd.py"
        old_tool.write_text("PAYLOAD_SHA256 = " + repr(dict(updater.PREVIOUS_SHA256)) + "\n")
        original = old_tool.read_bytes()
        (self.source / "teacher_weights.pth").write_bytes(b"must not ship")
        builder.freeze_manifest(self.source)
        self.assertEqual(old_tool.read_bytes(), original)
        output = self.source.parent / "ablation.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"distillation_ablation_update/" + name for name in self.payload}
            expected.update({"distillation_ablation_update/update_distillation_ablation.py",
                             "distillation_ablation_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 8)
            self.assertIsNone(archive.testzip())
        self.assertEqual(old_tool.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        changed = self.source / self.coordinator
        changed.write_bytes(changed.read_bytes() + b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed Ablation manifest"):
            builder.package(self.source.parent / "tampered.zip", self.source)
        old_tool.write_text("PAYLOAD_SHA256 = {}\n")
        with self.assertRaisesRegex(ValueError, "Old official checksum metadata"):
            builder.freeze_manifest(self.source)


if __name__ == "__main__":
    unittest.main()
