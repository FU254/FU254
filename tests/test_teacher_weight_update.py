"""Incremental teacher-weight experiments preserve all existing project data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_teacher_weight_study as builder
import update_teacher_weight_study as updater


class TeacherWeightUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="teacher-weight-update-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / "project"
        self.source = self.base / "teacher_weight_update"
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES}
        self.preserved.update({
            "runs/random_student_weight_decay/studies/keep/study.json": b'{"keep":"selected WD"}',
            "runs/random_student_soft_only/studies/keep/study.json": b'{"keep":"soft-only results"}',
            "runs/teacher_weight_study/studies/keep/study.json": b'{"keep":"existing study"}',
            "foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth": b"KEEP IMAGENET",
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA",
        })
        for name, data in self.preserved.items():
            self.write(self.project / name, data)
        self.payload = {
            name: (b"TEACHER_WEIGHT_UPDATE = True\n" if name.endswith(".py") else
                   b'{"seeds":[42],"student_initializations":["random","imagenet"]}\n'
                   if name.endswith(".json") else "# 教师权重实验\n".encode())
            for name in updater.PAYLOAD_FILES
        }
        for name, data in self.payload.items():
            self.write(self.source / name, data)
        manifest = patch.object(updater, "PAYLOAD_SHA256", {
            name: updater._digest(data) for name, data in self.payload.items()})
        manifest.start()
        self.addCleanup(manifest.stop)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def install(self, apply=False):
        return updater.install(self.project, self.source, apply)

    def assert_preserved(self):
        for name, data in self.preserved.items():
            self.assertEqual((self.project / name).read_bytes(), data, name)

    def assert_not_installed(self):
        self.assert_preserved()
        for name in self.payload:
            self.assertFalse((self.project / name).exists(), name)

    def test_preview_apply_idempotence_and_exact_rollback(self):
        result = self.install()
        self.assertEqual(set(result["files"]), set(self.payload))
        self.assertIsNone(result["backup"])
        self.assert_not_installed()
        result = self.install(True)
        self.assertEqual(set(result["files"]), set(self.payload))
        backup = Path(result["backup"])
        journal = json.loads((backup / "installation.json").read_text())
        self.assertEqual(set(journal["files"]), set(self.payload))
        for name, data in self.payload.items():
            self.assertEqual((self.project / name).read_bytes(), data)
            self.assertEqual((self.project / name).stat().st_mode & 0o777, 0o644)
            self.assertIsNone(journal["files"][name]["previous_sha256"])
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        updater.rollback(self.project, backup, True)
        self.assert_not_installed()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_all_payloads_are_new_and_no_old_version_is_replaceable(self):
        self.assertEqual(updater.PREVIOUS_SHA256, {})
        self.assertEqual(len(updater.PAYLOAD_FILES), 8)
        self.assertTrue(all("teacher_weight" in name or name.endswith("TEACHER_WEIGHT_STUDY.md")
                            for name in updater.PAYLOAD_FILES))

    def test_existing_unknown_file_preserved_before_any_installation(self):
        name = updater.PAYLOAD_FILES[0]
        user = b"USER_LOCAL_MODULE = True\n"
        self.write(self.project / name, user)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual((self.project / name).read_bytes(), user)
        for other in self.payload:
            if other != name:
                self.assertFalse((self.project / other).exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_all_sources_validated_before_destination_changes(self):
        name = updater.PAYLOAD_FILES[0]
        path = self.source / name
        path.write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        path.write_bytes(b"def invalid(:\n")
        updater.PAYLOAD_SHA256[name] = updater._digest(path.read_bytes())
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_not_installed()

    def test_invalid_configuration_and_unfrozen_manifest_prevent_writes(self):
        name = "foundation_sft/configs/teacher_weight_study.json"
        for data in (b"[1, 2]", b"{bad JSON"):
            (self.source / name).write_bytes(data)
            updater.PAYLOAD_SHA256[name] = updater._digest(data)
            with self.assertRaises(ValueError):
                self.install(True)
            self.assert_not_installed()
        for value in ({}, {**updater.PAYLOAD_SHA256, "unexpected.py": "0"*64}):
            with patch.object(updater, "PAYLOAD_SHA256", value), self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_not_installed()

    def test_missing_original_dependency_prevents_install(self):
        name = "foundation_sft/equal_teacher_data.py"
        (self.project / name).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + name):
            self.install(True)
        for name in self.payload:
            self.assertFalse((self.project / name).exists())

    def test_late_publication_failure_restores_absent_new_files(self):
        original = updater.os.link
        target = self.project / updater.PAYLOAD_FILES[-1]
        def fail_guide(source, destination):
            if Path(destination) == target:
                raise OSError("interrupted installation")
            return original(source, destination)
        with patch.object(updater.os, "link", side_effect=fail_guide), self.assertRaisesRegex(OSError, "interrupted installation"):
            self.install(True)
        self.assert_not_installed()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_rollback_preserves_later_user_edit(self):
        result = self.install(True)
        target = self.project / updater.PAYLOAD_FILES[0]
        target.write_bytes(b"USER_AFTER_INSTALL = True\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, result["backup"], True)
        self.assertEqual(target.read_bytes(), b"USER_AFTER_INSTALL = True\n")
        self.assert_preserved()

    def test_tampered_backup_cannot_claim_a_previous_version(self):
        result = self.install(True)
        path = Path(result["backup"]) / "installation.json"
        journal = json.loads(path.read_text())
        journal["files"][updater.PAYLOAD_FILES[0]]["previous_sha256"] = "0"*64
        path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, "recognized previous"):
            updater.rollback(self.project, result["backup"], True)
        self.assert_preserved()

    def test_symlink_lock_and_old_python_are_refused(self):
        target = self.project / updater.PAYLOAD_FILES[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        outside = self.base / "outside.py"
        outside.write_bytes(b"KEEP OUTSIDE")
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        target.unlink()
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another installer")
        with self.assertRaises(FileExistsError):
            self.install(True)
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)), self.assertRaisesRegex(ValueError, "Python 3.10"):
            self.install(True)
        self.assertEqual(outside.read_bytes(), b"KEEP OUTSIDE")
        self.assert_not_installed()

    def test_package_contains_only_reviewed_increment_and_manifest(self):
        name = "update_teacher_weight_study.py"
        self.write(self.source / name, Path(updater.__file__).read_bytes())
        self.write(self.source / "patient_data.npy", b"DO NOT SHIP")
        self.write(self.source / "teacher.pth", b"DO NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.base / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"teacher_weight_update/" + relative for relative in self.payload}
            expected.update({"teacher_weight_update/" + name, "teacher_weight_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 10)
            self.assertIsNone(archive.testzip())
            for line in archive.read("teacher_weight_update/SHA256SUMS.txt").decode().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read("teacher_weight_update/" + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"CHANGED = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed teacher-weight-study manifest"):
            builder.package(self.base / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
