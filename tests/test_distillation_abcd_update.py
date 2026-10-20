"""ABCD upgrades are transactional and never overwrite unrecognized user code."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_distillation_abcd as builder
import update_distillation_abcd as updater


class DistillationABCDUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="distillation-abcd-update-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project, self.source = root / "project", root / "distillation_abcd_update"
        self.previous = {
            "foundation_sft/distillation.py": b"OLD_FULL_TRAINER = True\n",
            "foundation_sft/models/vit_tiny_student.py": b"OLD_FULL_STUDENT = True\n",
        }
        self.preserved = {
            "foundation_sft/train.py": b"USER_MAIN_TRAINER = True\n",
            "run_single_teacher_distillation.py": b"OLD_ENTRY = True\n",
            "foundation_sft/models/__init__.py": b"USER_REGISTRY = True\n",
            "foundation_sft/configs/common.json": b'{"custom_data":"keep"}\n',
            "foundation_sft/configs/distillation_rad_dino_vit_tiny.json": b'{"full":"keep"}\n',
            "foundation_sft/configs/local_lower_learning_rate.json": b'{"lr":0.000001}\n',
            "foundation_weights/teacher/best.pt": b"teacher unchanged",
            "runs/single_teacher_distillation/old/test/metrics.json": b'{"history":"keep"}',
            "runs/foundation_sft/data_cache/pixels/existing.npy": b"cache unchanged",
            "runs/foundation_sft/shared_patient_split.csv": b"patient_id,split\nfixture,train\n",
        }
        for relative, data in {**self.preserved, **self.previous}.items():
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.modes = {}
        for index, relative in enumerate(self.previous):
            mode = (0o640, 0o600)[index]
            (self.project / relative).chmod(mode)
            self.modes[relative] = mode
        self.payload = {}
        for relative in updater.PAYLOAD_FILES:
            data = (b"NEW_ABCD = True\n" if relative.endswith(".py") else
                    b'{"experiment":"abcd"}\n' if relative.endswith(".json") else
                    "# 四组蒸馏\n".encode("utf-8"))
            self.payload[relative] = data
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name, value in (
            ("PREVIOUS_SHA256", {name: updater._digest(data) for name, data in self.previous.items()}),
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
        for relative, data in self.previous.items():
            path = self.project / relative
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(path.stat().st_mode & 0o777, self.modes[relative])
        for relative in set(self.payload) - set(self.previous):
            self.assertFalse((self.project / relative).exists(), relative)

    def assert_updated(self):
        self.assert_preserved()
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)
        for relative, mode in self.modes.items():
            self.assertEqual((self.project / relative).stat().st_mode & 0o777, mode)

    def test_preview_then_upgrade_backs_up_old_full_code_modes_and_is_idempotent(self):
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_old()
        result = self.install(True)
        backup = Path(result["backup"])
        self.assert_updated()
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        for relative, data in self.previous.items():
            self.assertEqual((backup / "previous" / relative).read_bytes(), data)
        journal = json.loads((backup / "installation.json").read_text())
        for relative in self.previous:
            self.assertEqual(journal["files"][relative]["previous_sha256"], updater._digest(self.previous[relative]))
            self.assertEqual(journal["files"][relative]["previous_mode"], self.modes[relative])
        self.assertEqual(self.install(True)["files"], [])
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_unknown_local_edits_prevent_every_write(self):
        edited = self.project / "foundation_sft/models/vit_tiny_student.py"
        edited.write_bytes(b"USER_CUSTOM_STUDENT = True\n")
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            self.install(True)
        self.assertEqual(edited.read_bytes(), b"USER_CUSTOM_STUDENT = True\n")
        self.assertEqual((self.project / "foundation_sft/distillation.py").read_bytes(), self.previous["foundation_sft/distillation.py"])
        self.assertFalse((self.project / "run_distillation_experiments.py").exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_unknown_new_configuration_is_not_overwritten(self):
        path = self.project / "foundation_sft/configs/distillation_rad_dino_vit_tiny_lora.json"
        path.write_bytes(b'{"custom_lora":true}\n')
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(path.read_bytes(), b'{"custom_lora":true}\n')
        self.assertEqual((self.project / "foundation_sft/distillation.py").read_bytes(), self.previous["foundation_sft/distillation.py"])
        self.assert_preserved()

    def test_missing_old_full_file_is_not_treated_as_a_fresh_install(self):
        (self.project / "foundation_sft/models/vit_tiny_student.py").unlink()
        with self.assertRaisesRegex(ValueError, "Unknown local changes"):
            self.install(True)
        self.assertFalse((self.project / "run_distillation_experiments.py").exists())
        self.assert_preserved()

    def test_source_tampering_and_missing_source_stop_before_project_changes(self):
        last = self.source / updater.PAYLOAD_FILES[-1]
        last.write_bytes(last.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install(True)
        last.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        self.assert_old()

    def test_invalid_python_and_nonobject_json_reject_the_whole_batch(self):
        relative = "foundation_sft/distillation_suite.py"
        invalid = b"def invalid(:\n"
        (self.source / relative).write_bytes(invalid)
        updater.PAYLOAD_SHA256[relative] = updater._digest(invalid)
        with self.assertRaises(SyntaxError):
            self.install(True)
        (self.source / relative).write_bytes(self.payload[relative])
        updater.PAYLOAD_SHA256[relative] = updater._digest(self.payload[relative])
        relative = "foundation_sft/configs/distillation_abcd.json"
        (self.source / relative).write_bytes(b"[]")
        updater.PAYLOAD_SHA256[relative] = updater._digest(b"[]")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self.install(True)
        self.assert_old()

    def test_unfrozen_or_expanded_manifest_cannot_touch_old_code(self):
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        expanded = dict(updater.PAYLOAD_SHA256, **{"foundation_sft/train.py": "0" * 64})
        with patch.object(updater, "PAYLOAD_SHA256", expanded):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_old()

    def test_symbolic_link_source_destination_and_backup_are_refused(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        relative = "run_distillation_experiments.py"
        target = self.project / relative
        target.symlink_to(outside / "new.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        target.unlink()
        source = self.source / relative
        source.unlink()
        source.symlink_to(outside / "new.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        source.unlink()
        source.write_bytes(self.payload[relative])
        (self.project / updater.BACKUP_DIRECTORY).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        self.assertEqual(list(outside.iterdir()), [])
        self.assert_old()

    def test_late_failure_restores_both_full_sources_and_removes_new_files(self):
        original_link = updater.os.link
        last = self.project / updater.PAYLOAD_FILES[-1]

        def fail_late(source, destination):
            if Path(destination) == last:
                raise OSError("Synthetic interrupted update")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=fail_late):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.install(True)
        self.assert_old()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_automatic_rollback_keeps_a_later_user_edit_and_reports_it(self):
        original_link = updater.os.link
        last = self.project / updater.PAYLOAD_FILES[-1]
        edited = self.project / "foundation_sft/distillation.py"

        def edit_then_fail(source, destination):
            if Path(destination) == last:
                edited.write_bytes(b"USER_RACE_EDIT = True\n")
                raise OSError("Synthetic late failure")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=edit_then_fail):
            with self.assertRaisesRegex(RuntimeError, "local changes preserved.*distillation.py"):
                self.install(True)
        self.assertEqual(edited.read_bytes(), b"USER_RACE_EDIT = True\n")
        self.assertEqual((self.project / "foundation_sft/models/vit_tiny_student.py").read_bytes(), self.previous["foundation_sft/models/vit_tiny_student.py"])
        for relative in set(self.payload) - set(self.previous):
            self.assertFalse((self.project / relative).exists())
        self.assert_preserved()

    def test_change_during_temp_write_is_detected_before_replacing_user_bytes(self):
        original_fsync = updater.os.fsync
        edited = self.project / "foundation_sft/distillation.py"
        called = False

        def race(descriptor):
            nonlocal called
            if not called:
                edited.write_bytes(b"USER_DURING_TEMP_WRITE = True\n")
                called = True
            return original_fsync(descriptor)

        with patch.object(updater.os, "fsync", side_effect=race):
            with self.assertRaisesRegex(ValueError, "File changed during update"):
                self.install(True)
        self.assertEqual(edited.read_bytes(), b"USER_DURING_TEMP_WRITE = True\n")
        self.assertEqual((self.project / "foundation_sft/models/vit_tiny_student.py").read_bytes(), self.previous["foundation_sft/models/vit_tiny_student.py"])
        self.assert_preserved()

    def test_existing_lock_and_old_python_are_rejected_without_writes(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another updater")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "another updater")
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)), patch.object(
                updater, "_path", side_effect=AssertionError("Do not inspect files")):
            with self.assertRaisesRegex(ValueError, "conda activate evax_lora"):
                self.install(True)
        self.assert_old()

    def test_manual_rollback_previews_and_restores_full_code_with_original_modes(self):
        installed = self.install(True)
        preview = updater.rollback(self.project, installed["backup"])
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assert_updated()
        updater.rollback(self.project, installed["backup"], True)
        self.assert_old()
        self.assertEqual(updater.rollback(self.project, installed["backup"], True)["files"], [])

    def test_rollback_does_not_delete_identical_preexisting_new_files(self):
        relative = "run_distillation_experiments.py"
        (self.project / relative).write_bytes(self.payload[relative])
        installed = self.install(True)
        self.assertNotIn(relative, installed["files"])
        updater.rollback(self.project, installed["backup"], True)
        self.assertEqual((self.project / relative).read_bytes(), self.payload[relative])
        self.assert_preserved()

    def test_rollback_rejects_user_edits_or_corrupt_backup_before_any_change(self):
        installed = self.install(True)
        edited = self.project / "foundation_sft/models/vit_tiny_student.py"
        edited.write_bytes(b"USER_AFTER_INSTALL = True\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, installed["backup"], True)
        self.assertEqual((self.project / "foundation_sft/distillation.py").read_bytes(), self.payload["foundation_sft/distillation.py"])
        edited.write_bytes(self.payload["foundation_sft/models/vit_tiny_student.py"])
        saved = Path(installed["backup"]) / "previous/foundation_sft/distillation.py"
        saved.write_bytes(b"CORRUPT = True\n")
        with self.assertRaisesRegex(ValueError, "Previous backup checksum"):
            updater.rollback(self.project, installed["backup"], True)
        self.assert_updated()

    def test_interrupted_rollback_restores_new_version_transactionally(self):
        installed = self.install(True)
        original_unlink = Path.unlink
        last = self.project / updater.PAYLOAD_FILES[-1]

        def fail_late(path, *args, **kwargs):
            if path == last:
                raise OSError("Synthetic interrupted rollback")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_late):
            with self.assertRaisesRegex(OSError, "interrupted rollback"):
                updater.rollback(self.project, installed["backup"], True)
        self.assert_updated()

    def test_builder_readonly_old_metadata_and_exact_nine_file_archive(self):
        (self.source / "update_distillation_abcd.py").write_bytes(Path(updater.__file__).read_bytes())
        old_tool = self.source / "update_single_teacher_distillation.py"
        old_tool.write_text("PAYLOAD_SHA256 = " + repr(dict(updater.PREVIOUS_SHA256)) + "\n")
        original_metadata = old_tool.read_bytes()
        (self.source / "unrelated_weight.pth").write_bytes(b"must not ship")
        builder.freeze_manifest(self.source)
        self.assertEqual(old_tool.read_bytes(), original_metadata)
        output = self.source.parent / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"distillation_abcd_update/" + name for name in updater.PAYLOAD_FILES}
            expected.update({"distillation_abcd_update/update_distillation_abcd.py", "distillation_abcd_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 9)
            self.assertIsNone(archive.testzip())
        self.assertEqual(old_tool.read_bytes(), original_metadata)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        changed = self.source / updater.PAYLOAD_FILES[0]
        changed.write_bytes(changed.read_bytes() + b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed ABCD manifest"):
            builder.package(self.source.parent / "tampered.zip", self.source)
        old_tool.write_text("PAYLOAD_SHA256 = {}\n")
        with self.assertRaisesRegex(ValueError, "Old official checksum metadata"):
            builder.freeze_manifest(self.source)


if __name__ == "__main__":
    unittest.main()
