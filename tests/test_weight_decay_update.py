"""Verify code-only weight-decay study installation without replacing old runs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_weight_decay_study as builder
import update_weight_decay_study as updater


class WeightDecayUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="weight-decay-install-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.project, self.source = base / "project", base / "weight_decay_update"
        self.preserved = {relative: b"EXISTING_TRAINING_CODE = True\n"
                          for relative in updater.PREREQUISITES}
        self.preserved.update({
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER WEIGHTS",
            "runs/random_student_distillation/suites/baseline/suite.json": b'{"keep":"baseline"}',
            "runs/foundation_sft/data_cache/pixels/example.npy": b"KEEP PIXELS",
        })
        for relative, data in self.preserved.items():
            self.write(self.project / relative, data)
        self.payload = {
            relative: (b"WEIGHT_DECAY_STUDY = True\n" if relative.endswith(".py") else
                       b'{"weight_decays":[0.001,0.01]}\n' if relative.endswith(".json") else
                       "# 权重衰减实验\n".encode())
            for relative in updater.PAYLOAD_FILES
        }
        for relative, data in self.payload.items():
            self.write(self.source / relative, data)
        manifest = patch.object(updater, "PAYLOAD_SHA256", {
            relative: updater._digest(data) for relative, data in self.payload.items()
        })
        manifest.start()
        self.addCleanup(manifest.stop)

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
        for relative in self.payload:
            self.assertFalse((self.project / relative).exists(), relative)

    def test_preview_new_only_install_idempotence_and_rollback(self):
        self.assertEqual(updater.PREVIOUS_SHA256, {})
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assert_absent()
        result = self.install(True)
        self.assertEqual(len(result["files"]), 6)
        backup = Path(result["backup"])
        journal = json.loads((backup / "installation.json").read_text())
        self.assertEqual(set(journal["files"]), set(self.payload))
        self.assertTrue(all(row["previous_sha256"] is None for row in journal["files"].values()))
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data)
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        self.assertEqual(len(updater.rollback(self.project, backup)["files"]), 6)
        self.assertTrue((self.project / updater.PAYLOAD_FILES[0]).exists())
        updater.rollback(self.project, backup, True)
        self.assert_absent()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_unknown_local_config_prevents_all_publication(self):
        target = self.project / "foundation_sft/configs/random_student_weight_decay.json"
        custom = b'{"weight_decays":[0.2]}\n'
        self.write(target, custom)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(target.read_bytes(), custom)
        self.assertFalse((self.project / "run_weight_decay_experiments.py").exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_bad_checksum_missing_and_invalid_python_prevent_all_writes(self):
        name = updater.PAYLOAD_FILES[0]
        origin = self.source / name
        origin.write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        origin.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        origin.write_bytes(b"def broken(:\n")
        updater.PAYLOAD_SHA256[name] = updater._digest(origin.read_bytes())
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_absent()

    def test_invalid_config_and_unfrozen_or_expanded_manifest_fail(self):
        name = "foundation_sft/configs/random_student_weight_decay.json"
        origin = self.source / name
        origin.write_bytes(b"[0.001, 0.01]\n")
        updater.PAYLOAD_SHA256[name] = updater._digest(origin.read_bytes())
        with self.assertRaisesRegex(ValueError, "must be an object"):
            self.install(True)
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        expanded = dict(updater.PAYLOAD_SHA256, **{"foundation_sft/train.py": "0" * 64})
        with patch.object(updater, "PAYLOAD_SHA256", expanded):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_absent()

    def test_missing_random_student_dependency_fails_before_publication(self):
        missing = "foundation_sft/random_student_distillation.py"
        (self.project / missing).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + missing):
            self.install(True)
        self.assertFalse((self.project / "run_weight_decay_experiments.py").exists())

    def test_source_and_target_symlinks_cannot_write_outside_project(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        name = "run_weight_decay_experiments.py"
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

    def test_late_failure_removes_only_published_new_files(self):
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

    def test_publication_race_preserves_user_created_file(self):
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
        self.assertFalse((self.project / "run_weight_decay_experiments.py").exists())
        self.assert_preserved()

    def test_rollback_preserves_later_edit_and_preexisting_identical_file(self):
        existing = "run_weight_decay_experiments.py"
        self.write(self.project / existing, self.payload[existing])
        result = self.install(True)
        self.assertNotIn(existing, result["files"])
        target = self.project / updater.PAYLOAD_FILES[0]
        target.write_bytes(b"USER_EDIT_AFTER_INSTALL = True\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, result["backup"], True)
        self.assertTrue((self.project / updater.PAYLOAD_FILES[-1]).exists())
        target.write_bytes(self.payload[updater.PAYLOAD_FILES[0]])
        updater.rollback(self.project, result["backup"], True)
        self.assertEqual((self.project / existing).read_bytes(), self.payload[existing])
        self.assert_preserved()

    def test_lock_and_unsupported_python_fail_without_installing(self):
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another installation")
        with self.assertRaises(FileExistsError):
            self.install(True)
        self.assertEqual(lock.read_text(), "another installation")
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)):
            with self.assertRaisesRegex(ValueError, "Python 3.10"):
                self.install(True)
        self.assert_absent()

    def test_package_contains_only_reviewed_code_and_checksum_manifest(self):
        installer = "update_weight_decay_study.py"
        self.write(self.source / installer, Path(updater.__file__).read_bytes())
        self.write(self.source / "secret_weights.pt", b"MUST NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.source.parent / "study.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"weight_decay_update/" + name for name in self.payload}
            expected.update({"weight_decay_update/" + installer,
                             "weight_decay_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 8)
            self.assertIsNone(archive.testzip())
            checksums = archive.read("weight_decay_update/SHA256SUMS.txt").decode().splitlines()
            self.assertEqual(len(checksums), 7)
            for line in checksums:
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read("weight_decay_update/" + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed weight-decay-study manifest"):
            builder.package(self.source.parent / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
