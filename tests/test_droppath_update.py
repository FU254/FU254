"""Verify reviewed DropPath code installation while preserving existing research data."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_droppath_study as builder
import update_droppath_study as updater


class DropPathUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="droppath-install-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.project, self.source = base / "project", base / "droppath_update"
        self.reviewed_previous = dict(updater.PREVIOUS_SHA256)
        self.previous = {relative: ("LEGACY_" + str(index) + " = True\n").encode()
                         for index, relative in enumerate(self.reviewed_previous)}
        previous_manifest = patch.object(updater, "PREVIOUS_SHA256", {
            relative: updater._digest(data) for relative, data in self.previous.items()
        })
        previous_manifest.start()
        self.addCleanup(previous_manifest.stop)
        self.preserved = {relative: b"EXISTING_TRAINING_CODE = True\n"
                          for relative in updater.PREREQUISITES
                          if relative not in self.previous}
        self.preserved.update({
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER WEIGHTS",
            "runs/random_student_distillation/suites/baseline/suite.json": b'{"keep":"baseline"}',
            "runs/foundation_sft/data_cache/pixels/example.npy": b"KEEP PIXELS",
        })
        for relative, data in {**self.preserved, **self.previous}.items():
            self.write(self.project / relative, data)
        self.payload = {
            relative: (b"DROPPATH_STUDY = True\n" if relative.endswith(".py") else
                       b'{"drop_path_rate":0.1}\n' if relative.endswith(".json") else
                       "# DropPath实验\n".encode())
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
            if relative in self.previous:
                self.assertEqual((self.project / relative).read_bytes(), self.previous[relative], relative)
            else:
                self.assertFalse((self.project / relative).exists(), relative)

    def test_preview_reviewed_replacement_install_idempotence_and_rollback(self):
        self.assertEqual(len(updater.PREVIOUS_SHA256), 3)
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assert_absent()
        result = self.install(True)
        self.assertEqual(len(result["files"]), 9)
        backup = Path(result["backup"])
        journal = json.loads((backup / "installation.json").read_text())
        self.assertEqual(set(journal["files"]), set(self.payload))
        for relative, row in journal["files"].items():
            expected = updater._digest(self.previous[relative]) if relative in self.previous else None
            self.assertEqual(row["previous_sha256"], expected)
        for relative, data in self.payload.items():
            self.assertEqual((self.project / relative).read_bytes(), data)
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        self.assertEqual(len(updater.rollback(self.project, backup)["files"]), 9)
        self.assertTrue((self.project / updater.PAYLOAD_FILES[0]).exists())
        updater.rollback(self.project, backup, True)
        self.assert_absent()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_unknown_local_config_prevents_all_publication(self):
        target = self.project / "foundation_sft/configs/random_student_droppath.json"
        custom = b'{"drop_path_rate":0.2}\n'
        self.write(target, custom)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual(target.read_bytes(), custom)
        self.assertFalse((self.project / "run_droppath_experiments.py").exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_only_the_three_reviewed_legacy_versions_are_replaceable(self):
        self.assertEqual(self.reviewed_previous, {
            "foundation_sft/models/vit_tiny_student.py": "938797bd5be22349119905e9a9121d115e37db62a505281a4655d84f61344217",
            "foundation_sft/distillation.py": "dceb270c7af766d2316968327c46c1c67d58223f8b023ceee1e853f5250879c8",
            "foundation_sft/random_student_distillation.py": "b7d9279addd86a562685a90e9a094c34ae0391d8d8334e37733442b3ce8be6a0",
        })

    def test_known_broken_release_hashes_are_limited_to_the_two_fixed_files(self):
        self.assertEqual(updater.KNOWN_PREVIOUS_SHA256, {
            "foundation_sft/models/vit_tiny_student.py": (
                "78cf4cfc978a6bd88ad2ce554fc241d2dec994d64592667ffd5e5515fa8612c2",),
            "foundation_sft/distillation.py": (
                "d4cddbbd5255f87eeeba4eba5b9a1ac5cd3c9d91cc6a8889ef3dba3a1077ca17",),
        })

    def test_known_broken_release_upgrade_and_rollback_restore_exact_source_bytes(self):
        affected = ("foundation_sft/models/vit_tiny_student.py", "foundation_sft/distillation.py")
        broken = {name: ("BROKEN_DROPPATH_" + str(index) + " = True\n").encode()
                  for index, name in enumerate(affected)}
        for name, data in self.payload.items():
            self.write(self.project / name, data)
        for name, data in broken.items():
            self.write(self.project / name, data)
            (self.project / name).chmod(0o640)
        allowed = {name: (updater._digest(data),) for name, data in broken.items()}
        with patch.object(updater, "KNOWN_PREVIOUS_SHA256", allowed):
            result = self.install(True)
            self.assertEqual(set(result["files"]), set(affected))
            self.assert_preserved()
            for name in affected:
                self.assertEqual((self.project / name).read_bytes(), self.payload[name])
                self.assertEqual((self.project / name).stat().st_mode & 0o777, 0o640)
            journal = json.loads((Path(result["backup"]) / "installation.json").read_text())
            for name in affected:
                self.assertEqual(journal["files"][name]["previous_sha256"], updater._digest(broken[name]))
            self.assertEqual(self.install(True)["files"], [])
            updater.rollback(self.project, result["backup"], True)
            for name, data in broken.items():
                self.assertEqual((self.project / name).read_bytes(), data)
                self.assertEqual((self.project / name).stat().st_mode & 0o777, 0o640)
            for name, data in self.payload.items():
                if name not in affected:
                    self.assertEqual((self.project / name).read_bytes(), data)
            self.assertEqual(updater.rollback(self.project, result["backup"], True)["files"], [])

    def test_mixed_reviewed_legacy_and_broken_versions_can_upgrade_and_roll_back(self):
        name = "foundation_sft/models/vit_tiny_student.py"
        broken = b"KNOWN_BROKEN_DROPPATH = True\n"
        self.write(self.project / name, broken)
        with patch.object(updater, "KNOWN_PREVIOUS_SHA256", {name: (updater._digest(broken),)}):
            result = self.install(True)
            self.assertEqual(len(result["files"]), 9)
            updater.rollback(self.project, result["backup"], True)
        self.assertEqual((self.project / name).read_bytes(), broken)
        for relative, data in self.previous.items():
            if relative != name:
                self.assertEqual((self.project / relative).read_bytes(), data)
        self.assert_preserved()

    def test_rollback_rejects_an_unrecognized_recorded_previous_hash(self):
        result = self.install(True)
        journal_path = Path(result["backup"]) / "installation.json"
        journal = json.loads(journal_path.read_text())
        journal["files"]["foundation_sft/models/vit_tiny_student.py"]["previous_sha256"] = "f" * 64
        journal_path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, "recognized previous code version"):
            updater.rollback(self.project, result["backup"], True)
        for name, data in self.payload.items():
            self.assertEqual((self.project / name).read_bytes(), data)
        self.assert_preserved()

    def test_unknown_legacy_edit_preserves_all_files(self):
        relative = "foundation_sft/models/vit_tiny_student.py"
        changed = b"USER_CHANGED_STUDENT = True\n"
        self.write(self.project / relative, changed)
        with self.assertRaisesRegex(ValueError, "Unknown local changes preserved"):
            self.install(True)
        self.assertEqual((self.project / relative).read_bytes(), changed)
        for name, data in self.previous.items():
            if name != relative:
                self.assertEqual((self.project / name).read_bytes(), data)
        self.assertFalse((self.project / "run_droppath_experiments.py").exists())
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
        name = "foundation_sft/configs/random_student_droppath.json"
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
        self.assertFalse((self.project / "run_droppath_experiments.py").exists())

    def test_source_and_target_symlinks_cannot_write_outside_project(self):
        outside = self.project.parent / "outside"
        outside.mkdir()
        name = "run_droppath_experiments.py"
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
        self.assertFalse((self.project / "run_droppath_experiments.py").exists())
        self.assert_preserved()

    def test_rollback_preserves_later_edit_and_preexisting_identical_file(self):
        existing = "run_droppath_experiments.py"
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
        installer = "update_droppath_study.py"
        self.write(self.source / installer, Path(updater.__file__).read_bytes())
        self.write(self.source / "secret_weights.pt", b"MUST NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.source.parent / "study.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {"droppath_update/" + name for name in self.payload}
            expected.update({"droppath_update/" + installer,
                             "droppath_update/SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 11)
            self.assertIsNone(archive.testzip())
            checksums = archive.read("droppath_update/SHA256SUMS.txt").decode().splitlines()
            self.assertEqual(len(checksums), 10)
            for line in checksums:
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read("droppath_update/" + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed droppath-study manifest"):
            builder.package(self.source.parent / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
