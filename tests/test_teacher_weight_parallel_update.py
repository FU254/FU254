"""Live-parallel delivery changes one known shim and preserves running studies."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_teacher_weight_parallel as builder
import update_teacher_weight_parallel as updater


OLD_ENTRY = b'''#!/usr/bin/env python3
"""Run one checked worker job from the teacher-weight study."""

from foundation_sft.teacher_weight_training import main

if __name__ == "__main__":
    main()
'''
ENTRY = "run_teacher_weight_training.py"


class TeacherWeightParallelUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="teacher-weight-parallel-update-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / "project"
        self.source = self.base / "teacher_weight_parallel_update"
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES}
        self.preserved.update({
            "runs/teacher_weight_study/studies/active/study.json": b'{"status":"running"}',
            "runs/teacher_weight_study/studies/active/jobs/P-C/train_1/train.log": b"epoch 11 started\n",
            "runs/teacher_weight_study/studies/active/jobs/P-C/train_1/outputs/last.pt": b"KEEP CURRENT MODEL",
            "foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth": b"KEEP IMAGENET",
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA",
            "run_single_teacher_distillation.py": b"KEEP OLD ENTRY\n",
        })
        for name, data in self.preserved.items():
            self.write(self.project / name, data)
        self.write(self.project / ENTRY, OLD_ENTRY)
        (self.project / ENTRY).chmod(0o750)
        self.payload = {
            name: (b"PARALLEL_UPDATE = True\n" if name.endswith(".py") else
                   "# 独立并行教师权重任务\n".encode())
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

    def assert_before(self):
        self.assert_preserved()
        self.assertEqual((self.project / ENTRY).read_bytes(), OLD_ENTRY)
        self.assertEqual((self.project / ENTRY).stat().st_mode & 0o777, 0o750)
        for name in self.payload:
            if name != ENTRY:
                self.assertFalse((self.project / name).exists(), name)

    def test_known_shim_is_the_only_replaceable_file_and_published_last(self):
        self.assertEqual(updater.PREVIOUS_SHA256, {
            ENTRY: "e40b97b736fb2641ae744e4ceea8f22a3087e1fc098fc4876da177f834f09ace"})
        self.assertEqual(updater._digest(OLD_ENTRY), updater.PREVIOUS_SHA256[ENTRY])
        self.assertEqual(updater.PAYLOAD_FILES[-1], ENTRY)
        self.assertEqual(set(updater.PAYLOAD_FILES), {
            "foundation_sft/teacher_weight_parallel.py", "run_teacher_weight_parallel.py",
            "foundation_sft/TEACHER_WEIGHT_PARALLEL.md", ENTRY})

    def test_preview_install_idempotence_and_exact_known_shim_rollback(self):
        preview = self.install()
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assertIsNone(preview["backup"])
        self.assert_before()
        installed = self.install(True)
        backup = Path(installed["backup"])
        journal = json.loads((backup / "installation.json").read_text())
        self.assertEqual(set(journal["files"]), set(self.payload))
        for name, data in self.payload.items():
            self.assertEqual((self.project / name).read_bytes(), data)
            expected_mode = 0o750 if name == ENTRY else 0o644
            self.assertEqual((self.project / name).stat().st_mode & 0o777, expected_mode)
            self.assertEqual(journal["files"][name]["previous_sha256"],
                             updater.PREVIOUS_SHA256.get(name))
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        updater.rollback(self.project, backup, True)
        self.assert_before()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_unknown_shim_preserved_before_any_destination_changes(self):
        local = b"USER_MODIFIED_WORKER_ENTRY = True\n"
        (self.project / ENTRY).write_bytes(local)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual((self.project / ENTRY).read_bytes(), local)
        for name in self.payload:
            if name != ENTRY:
                self.assertFalse((self.project / name).exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_missing_required_old_shim_is_not_silently_added(self):
        (self.project / ENTRY).unlink()
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertFalse((self.project / ENTRY).exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_unknown_new_module_preserved_without_changing_shim(self):
        name = "foundation_sft/teacher_weight_parallel.py"
        local = b"USER_EXISTING_MODULE = True\n"
        self.write(self.project / name, local)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual((self.project / name).read_bytes(), local)
        self.assertEqual((self.project / ENTRY).read_bytes(), OLD_ENTRY)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_all_uploaded_sources_validated_before_any_change(self):
        target = self.source / ENTRY
        target.write_bytes(b"TAMPERED = True\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.install(True)
        target.write_bytes(b"def invalid(:\n")
        updater.PAYLOAD_SHA256[ENTRY] = updater._digest(target.read_bytes())
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_before()

    def test_unfrozen_manifest_and_invalid_text_prevent_writes(self):
        for value in ({}, {**updater.PAYLOAD_SHA256, "unexpected.py": "0" * 64}):
            with patch.object(updater, "PAYLOAD_SHA256", value):
                with self.assertRaisesRegex(ValueError, "not been frozen"):
                    self.install(True)
        name = "foundation_sft/TEACHER_WEIGHT_PARALLEL.md"
        (self.source / name).write_bytes(b"\xff\xfe")
        updater.PAYLOAD_SHA256[name] = updater._digest(b"\xff\xfe")
        with self.assertRaises(UnicodeError):
            self.install(True)
        self.assert_before()

    def test_missing_old_study_dependency_prevents_install(self):
        name = "foundation_sft/teacher_weight_study.py"
        (self.project / name).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + name):
            self.install(True)
        self.assertEqual((self.project / ENTRY).read_bytes(), OLD_ENTRY)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())

    def test_install_compiles_without_importing_payload_or_signaling_processes(self):
        name = "foundation_sft/teacher_weight_parallel.py"
        data = b"raise RuntimeError('installer must never execute this module')\n"
        (self.source / name).write_bytes(data)
        updater.PAYLOAD_SHA256[name] = updater._digest(data)
        with patch("os.kill", side_effect=AssertionError("must not signal")):
            with patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
                self.install(True)
        self.assertEqual((self.project / name).read_bytes(), data)
        self.assert_preserved()

    def test_shim_publication_failure_restores_new_files_and_old_shim(self):
        original = updater.os.replace
        shim = self.project / ENTRY
        def fail_entry(source, destination):
            if Path(destination) == shim:
                raise OSError("interrupted shim publication")
            return original(source, destination)
        with patch.object(updater.os, "replace", side_effect=fail_entry):
            with self.assertRaisesRegex(OSError, "interrupted shim publication"):
                self.install(True)
        self.assert_before()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_rollback_preserves_later_local_shim_edit(self):
        installed = self.install(True)
        local = b"USER_AFTER_INSTALL = True\n"
        (self.project / ENTRY).write_bytes(local)
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, installed["backup"], True)
        self.assertEqual((self.project / ENTRY).read_bytes(), local)
        self.assert_preserved()

    def test_tampered_old_version_backup_refused(self):
        installed = self.install(True)
        backup = Path(installed["backup"])
        journal_path = backup / "installation.json"
        journal = json.loads(journal_path.read_text())
        journal["files"][ENTRY]["previous_sha256"] = "0" * 64
        journal_path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, "recognized previous"):
            updater.rollback(self.project, backup, True)
        self.assertEqual((self.project / ENTRY).read_bytes(), self.payload[ENTRY])
        self.assert_preserved()

    def test_symlink_install_lock_and_old_python_refused(self):
        target = self.project / updater.PAYLOAD_FILES[0]
        outside = self.base / "outside.py"
        outside.write_bytes(b"KEEP OUTSIDE")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        target.unlink()
        lock = self.project / updater.LOCK_NAME
        lock.write_text("another installer")
        with self.assertRaises(FileExistsError):
            self.install(True)
        lock.unlink()
        with patch.object(updater.sys, "version_info", (3, 7, 17)):
            with self.assertRaisesRegex(ValueError, "Python 3.10"):
                self.install(True)
        self.assertEqual(outside.read_bytes(), b"KEEP OUTSIDE")
        self.assert_before()

    def test_package_contains_only_reviewed_parallel_increment_and_manifest(self):
        installer_name = "update_teacher_weight_parallel.py"
        self.write(self.source / installer_name, Path(updater.__file__).read_bytes())
        self.write(self.source / "patient_data.npy", b"DO NOT SHIP")
        self.write(self.source / "teacher.pth", b"DO NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.base / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            prefix = "teacher_weight_parallel_update/"
            expected = {prefix + name for name in self.payload}
            expected.update({prefix + installer_name, prefix + "SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 6)
            self.assertIsNone(archive.testzip())
            for line in archive.read(prefix + "SHA256SUMS.txt").decode().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read(prefix + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"CHANGED = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed teacher-weight-parallel manifest"):
            builder.package(self.base / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
