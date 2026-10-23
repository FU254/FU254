"""Check incremental delivery without touching historical code or run artifacts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_random_student as builder
import update_random_student as updater


class RandomStudentUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="random-student-install-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.project, self.source = base / "project", base / "random_student_update"
        self.preserved = {
            "foundation_sft/train.py": b"OLD_TRAIN = True\n",
            "foundation_sft/distillation.py": b"OLD_KD = True\n",
            "foundation_sft/distillation_data.py": b"OLD_VIEWS = True\n",
            "foundation_sft/distillation_losses.py": b"OLD_LOSSES = True\n",
            "foundation_sft/distillation_suite.py": b"OLD_COORDINATOR = True\n",
            "foundation_sft/training_history.py": b"OLD_HISTORY = True\n",
            "foundation_sft/models/vit_tiny_student.py": b"OLD_STUDENT = True\n",
            "foundation_sft/configs/distillation_rad_dino_vit_tiny.json": b'{"student":"old"}\n',
            "foundation_weights/chexzero/old.pt": b"EXISTING WEIGHTS",
            "runs/distillation_ablation/suites/old/report.csv": b"EXISTING RESULTS",
            "runs/foundation_sft/data_cache/pixels/old.npy": b"EXISTING CACHE",
        }
        for relative, data in self.preserved.items():
            self.write(self.project / relative, data)
        self.factory = "foundation_sft/models/vit_tiny_student.py"
        self.previous_factory = self.preserved.pop(self.factory)
        known = patch.object(updater, "PREVIOUS_SHA256", {
            self.factory: updater._digest(self.previous_factory)
        })
        known.start()
        self.addCleanup(known.stop)
        self.payload = {
            relative: (b"NEW_RANDOM_KD = True\n" if relative.endswith(".py") else
                       b'{"protocol":"random_student"}\n' if relative.endswith(".json") else
                       "# 随机学生蒸馏\n".encode())
            for relative in updater.PAYLOAD_FILES
        }
        for relative, data in self.payload.items():
            self.write(self.source / relative, data)
        patched = patch.object(updater, "PAYLOAD_SHA256", {
            relative: updater._digest(data) for relative, data in self.payload.items()
        })
        patched.start()
        self.addCleanup(patched.stop)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def install(self, apply=False):
        return updater.install(self.project, self.source, apply)

    def assert_preserved(self):
        for relative, data in self.preserved.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)

    def assert_absent(self):
        self.assert_preserved()
        self.assertEqual((self.project / self.factory).read_bytes(), self.previous_factory)
        for relative in self.payload:
            if relative != self.factory:
                self.assertFalse((self.project / relative).exists(), relative)

    def assert_installed(self):
        self.assert_preserved()
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data, relative)

    def test_preview_install_same_version_and_rollback_preserve_all_old_artifacts(self):
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_absent()
        result = self.install(True)
        self.assertEqual(len(result["files"]), 7)
        journal = json.loads((Path(result["backup"]) / "installation.json").read_text())
        self.assertEqual(set(journal["files"]), set(self.payload))
        self.assertEqual(journal["files"][self.factory]["previous_sha256"],
                         updater._digest(self.previous_factory))
        self.assertTrue(all(row["previous_sha256"] is None
                            for name, row in journal["files"].items() if name != self.factory))
        self.assert_installed()
        self.assertEqual(self.install(True)["files"], [])
        self.assertEqual(len(updater.rollback(self.project, result["backup"])["files"]), 7)
        self.assert_installed()
        updater.rollback(self.project, result["backup"], True)
        self.assert_absent()
        self.assertEqual(updater.rollback(self.project, result["backup"], True)["files"], [])

    def test_unknown_local_new_config_prevents_every_publication(self):
        target = self.project / "foundation_sft/configs/random_student_distillation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'{"teacher":"user-choice"}\n')
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(target.read_bytes(), b'{"teacher":"user-choice"}\n')
        self.assertFalse((self.project / "run_random_student_experiments.py").exists())
        self.assert_preserved()

    def test_unknown_model_factory_is_preserved_before_any_new_file_is_written(self):
        target = self.project / self.factory
        custom = b"USER_RANDOM_STUDENT_IMPLEMENTATION = True\n"
        target.write_bytes(custom)
        with self.assertRaisesRegex(ValueError, "Unknown local changes preserved"):
            self.install(True)
        self.assertEqual(target.read_bytes(), custom)
        self.assertFalse((self.project / "run_random_student_experiments.py").exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_model_factory_permissions_and_previous_bytes_are_restored_on_rollback(self):
        target = self.project / self.factory
        target.chmod(0o640)
        result = self.install(True)
        self.assertEqual(target.read_bytes(), self.payload[self.factory])
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        backup = Path(result["backup"])
        self.assertEqual((backup / "previous" / self.factory).read_bytes(), self.previous_factory)
        updater.rollback(self.project, backup, True)
        self.assertEqual(target.read_bytes(), self.previous_factory)
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assert_absent()

    def test_bad_checksum_missing_or_invalid_code_prevents_all_writes(self):
        name = updater.PAYLOAD_FILES[0]
        target = self.source / name
        target.write_bytes(b"TAMPER = True\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        target.write_bytes(b"def broken(:\n")
        updater.PAYLOAD_SHA256[name] = updater._digest(target.read_bytes())
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_absent()

    def test_unfrozen_or_expanded_manifest_and_missing_legacy_dependency_fail(self):
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        with patch.object(updater, "PAYLOAD_SHA256", dict(updater.PAYLOAD_SHA256, **{
                "foundation_sft/train.py": "0" * 64})):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        (self.project / "foundation_sft/distillation.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing: foundation_sft/distillation.py"):
            self.install(True)
        self.assertFalse((self.project / "run_random_student_experiments.py").exists())

    def test_source_and_target_symlinks_cannot_write_outside_project(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        name = "run_random_student_experiments.py"
        target = self.project / name
        target.symlink_to(outside / "entry.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        target.unlink()
        origin = self.source / name
        origin.unlink()
        origin.symlink_to(outside / "entry.py")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        self.assertEqual(list(outside.iterdir()), [])
        self.assert_absent()

    def test_late_publication_failure_removes_only_newly_installed_files(self):
        original_link = updater.os.link
        last = self.project / updater.PAYLOAD_FILES[-1]

        def fail_last(source, destination):
            if Path(destination) == last:
                raise OSError("interrupted installation")
            return original_link(source, destination)

        with patch.object(updater.os, "link", side_effect=fail_last):
            with self.assertRaisesRegex(OSError, "interrupted installation"):
                self.install(True)
        self.assert_absent()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_atomic_race_preserves_a_file_created_during_install(self):
        original_fsync = updater.os.fsync
        target = self.project / updater.PAYLOAD_FILES[0]
        called = False

        def write_once(descriptor):
            nonlocal called
            if not called:
                target.write_bytes(b"USER_CREATED_DURING_INSTALL = True\n")
                called = True
            return original_fsync(descriptor)

        with patch.object(updater.os, "fsync", side_effect=write_once):
            with self.assertRaisesRegex(ValueError, "File changed during update"):
                self.install(True)
        self.assertEqual(target.read_bytes(), b"USER_CREATED_DURING_INSTALL = True\n")
        self.assertFalse((self.project / "run_random_student_experiments.py").exists())
        self.assert_preserved()

    def test_rollback_preserves_later_edit_and_preexisting_identical_file(self):
        existing = "run_random_student_experiments.py"
        self.write(self.project / existing, self.payload[existing])
        result = self.install(True)
        self.assertNotIn(existing, result["files"])
        target = self.project / updater.PAYLOAD_FILES[0]
        target.write_bytes(b"USER_EDIT_AFTER_INSTALL = True\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, result["backup"], True)
        self.assertTrue((self.project / updater.PAYLOAD_FILES[-1]).is_file())
        target.write_bytes(self.payload[updater.PAYLOAD_FILES[0]])
        updater.rollback(self.project, result["backup"], True)
        self.assertEqual((self.project / existing).read_bytes(), self.payload[existing])
        self.assert_preserved()

    def test_lock_and_unsupported_python_do_not_start_install(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("other installer")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "other installer")
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)):
            with self.assertRaisesRegex(ValueError, "Python 3.10"):
                self.install(True)
        self.assert_absent()

    def test_package_contains_only_reviewed_code_and_checksums_not_weights(self):
        self.write(self.source / "update_random_student.py", Path(updater.__file__).read_bytes())
        self.write(self.source / "secret_weights.pt", b"MUST NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.source.parent / "random.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"random_student_update/" + name for name in self.payload}
            expected.update({"random_student_update/update_random_student.py",
                             "random_student_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 9)
            self.assertIsNone(archive.testzip())
            checksums = archive.read("random_student_update/SHA256SUMS.txt").decode().splitlines()
            self.assertEqual(len(checksums), 8)
            for line in checksums:
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read("random_student_update/" + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        name = updater.PAYLOAD_FILES[0]
        (self.source / name).write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed random-student manifest"):
            builder.package(self.source.parent / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
