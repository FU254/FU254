"""The KD add-on preserves all old code/data and can safely undo its additions."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_single_teacher_distillation as builder
import update_single_teacher_distillation as updater


class SingleTeacherKDUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="single-teacher-kd-update-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project, self.source = root / "project", root / "kd_update"
        self.old = {
            "foundation_sft/train.py": b"LOCAL_OLD_TRAINER = True\n",
            "foundation_sft/configs/common.json": b'{"data":"user-custom-path"}\n',
            "foundation_sft/models/__init__.py": b"USER_REGISTRY = True\n",
            "foundation_weights/teacher/best.pt": b"weight bytes untouched",
            "runs/old_experiment/metrics.json": b'{"history":"keep"}',
            "runs/foundation_sft/shared_patient_split.csv": b"patient_id,split\nfixture,train\n",
            "data/existing.dcm": b"image bytes untouched",
        }
        for relative, data in self.old.items():
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.payload = {}
        for relative in updater.PAYLOAD_FILES:
            if relative.endswith(".py"):
                data = b"ADDITION_ONLY = True\n"
            elif relative.endswith(".json"):
                data = b'{"teacher":"rad_dino","student":"vit_tiny"}\n'
            else:
                data = "# 独立蒸馏说明\n".encode("utf-8")
            self.payload[relative] = data
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        checksums = {relative: updater._digest(data) for relative, data in self.payload.items()}
        hashes = patch.object(updater, "PAYLOAD_SHA256", checksums)
        hashes.start()
        self.addCleanup(hashes.stop)

    def assert_old_unchanged(self):
        for relative, data in self.old.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)

    def assert_no_payload(self):
        for relative in self.payload:
            self.assertFalse((self.project / relative).exists(), relative)

    def install(self, apply=False):
        return updater.install(self.project, self.source, apply)

    def test_preview_writes_nothing_and_apply_only_adds_with_idempotence(self):
        result = self.install()
        self.assertEqual(set(result["files"]), set(self.payload))
        self.assertIsNone(result["backup"])
        self.assert_no_payload()
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_old_unchanged()
        result = self.install(True)
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data)
            self.assertEqual((Path(result["backup"]) / "payload" / relative).read_bytes(), data)
        journal = json.loads((Path(result["backup"]) / "installation.json").read_text())
        self.assertEqual(journal["previous"], {name: None for name in self.payload})
        self.assertEqual(self.install(True)["files"], [])
        self.assert_old_unchanged()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_tampered_or_missing_source_prevents_every_write(self):
        target = self.source / updater.PAYLOAD_FILES[-1]
        data = target.read_bytes()
        target.write_bytes(data + b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install(True)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        self.assert_no_payload()
        self.assert_old_unchanged()
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())

    def test_existing_different_config_is_preserved_and_blocks_whole_installation(self):
        relative = "foundation_sft/configs/distillation_rad_dino_vit_tiny.json"
        path = self.project / relative
        path.write_text('{"teacher":"local-custom-choice"}')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.project / updater.PAYLOAD_FILES[0]).exists())
        self.assert_old_unchanged()

    def test_identical_existing_addition_is_not_deleted_by_rollback(self):
        relative = updater.PAYLOAD_FILES[0]
        preexisting = self.project / relative
        preexisting.write_bytes(self.payload[relative])
        result = self.install(True)
        self.assertNotIn(relative, result["files"])
        updater.rollback(self.project, result["backup"], True)
        self.assertEqual(preexisting.read_bytes(), self.payload[relative])
        for other in set(self.payload) - {relative}:
            self.assertFalse((self.project / other).exists())
        self.assert_old_unchanged()

    def test_bad_python_and_unfrozen_or_expanded_manifest_are_rejected(self):
        relative = updater.PAYLOAD_FILES[0]
        invalid = b"def bad(:\n"
        (self.source / relative).write_bytes(invalid)
        updater.PAYLOAD_SHA256[relative] = updater._digest(invalid)
        with self.assertRaises(SyntaxError):
            self.install(True)
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        expanded = dict(updater.PAYLOAD_SHA256, **{"foundation_sft/train.py": "0" * 64})
        with patch.object(updater, "PAYLOAD_SHA256", expanded):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_no_payload()
        self.assert_old_unchanged()

    def test_old_python_is_rejected_before_payload_or_lock(self):
        with patch.object(updater.sys, "version_info", (3, 7, 17)), patch.object(
                updater, "_path", side_effect=AssertionError("Do not read payloads")):
            with self.assertRaisesRegex(ValueError, "conda activate evax_lora"):
                self.install(True)
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_symbolic_link_destination_source_and_backup_are_refused(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        relative = updater.PAYLOAD_FILES[0]
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
        self.assert_no_payload()
        self.assert_old_unchanged()

    def test_late_installation_failure_removes_only_its_new_files(self):
        original_link = updater.os.link
        failed_file = self.project / "foundation_sft/configs/distillation_rad_dino_vit_tiny.json"

        def fail_late(source, destination):
            if Path(destination) == failed_file:
                raise OSError("Synthetic interrupted installation")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=fail_late):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.install(True)
        self.assert_no_payload()
        self.assert_old_unchanged()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_raced_in_user_file_is_never_overwritten_or_erased(self):
        original_link = updater.os.link
        raced_file = self.project / updater.PAYLOAD_FILES[1]

        def race(source, destination):
            if Path(destination) == raced_file:
                raced_file.write_bytes(b"LOCAL_USER_FILE = True\n")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                self.install(True)
        self.assertEqual(raced_file.read_bytes(), b"LOCAL_USER_FILE = True\n")
        self.assertFalse((self.project / updater.PAYLOAD_FILES[0]).exists())
        self.assert_old_unchanged()

    def test_existing_installer_lock_is_retained(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another installation")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "another installation")
        self.assert_no_payload()

    def test_rollback_previews_then_removes_only_added_code(self):
        installed = self.install(True)
        preview = updater.rollback(self.project, installed["backup"])
        self.assertEqual(set(preview["files"]), set(self.payload))
        for relative in self.payload:
            self.assertTrue((self.project / relative).is_file())
        updater.rollback(self.project, installed["backup"], True)
        self.assert_no_payload()
        self.assert_old_unchanged()
        self.assertEqual(updater.rollback(self.project, installed["backup"], True)["files"], [])

    def test_rollback_refuses_local_edit_before_removing_any_file(self):
        installed = self.install(True)
        edited = self.project / updater.PAYLOAD_FILES[-1]
        edited.write_bytes(b"User revised their KD instructions\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, installed["backup"], True)
        for relative in self.payload:
            self.assertTrue((self.project / relative).is_file())
        self.assert_old_unchanged()

    def test_rollback_rejects_tampered_backup_and_outside_project_journal(self):
        installed = self.install(True)
        backup = Path(installed["backup"])
        saved = backup / "payload" / updater.PAYLOAD_FILES[0]
        saved.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Backup payload checksum"):
            updater.rollback(self.project, backup, True)
        with self.assertRaisesRegex(ValueError, "belong to this project"):
            updater.rollback(self.project, self.project.parent / "outside", True)
        for relative in self.payload:
            self.assertTrue((self.project / relative).is_file())

    def test_interrupted_rollback_restores_removed_payload(self):
        installed = self.install(True)
        original_unlink = Path.unlink
        failed_path = self.project / updater.PAYLOAD_FILES[-1]

        def fail_once(path, *args, **kwargs):
            if path == failed_path:
                raise OSError("Synthetic interrupted rollback")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_once):
            with self.assertRaisesRegex(OSError, "interrupted rollback"):
                updater.rollback(self.project, installed["backup"], True)
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data)
        self.assert_old_unchanged()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_builder_freezes_checksums_and_packages_only_new_reviewed_code(self):
        # Package a fixture source tree: no real weights or patient data are read.
        installer_source = Path(updater.__file__).read_bytes()
        (self.source / "update_single_teacher_distillation.py").write_bytes(installer_source)
        (self.source / "unrelated_weight.pth").write_bytes(b"must not ship")
        checksums = builder.freeze_manifest(self.source)
        self.assertEqual(checksums, {name: updater._digest(data) for name, data in self.payload.items()})
        frozen = (self.source / "update_single_teacher_distillation.py").read_text()
        compile(frozen, "frozen_installer.py", "exec")
        output = self.source.parent / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"kd_update/" + name for name in self.payload}
            expected.update({"kd_update/update_single_teacher_distillation.py", "kd_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertIsNone(archive.testzip())
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        path = self.source / updater.PAYLOAD_FILES[0]
        path.write_bytes(path.read_bytes() + b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed KD manifest"):
            builder.package(self.source.parent / "tampered.zip", self.source)


if __name__ == "__main__":
    unittest.main()
