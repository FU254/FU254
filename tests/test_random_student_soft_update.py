"""Soft-label-only increment installs reviewed code and preserves existing results."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_random_student_soft_only as builder
import update_random_student_soft_only as updater


class RandomStudentSoftOnlyUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="random-soft-update-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.project, self.source = base / "project", base / "random_student_soft_only_update"
        self.reviewed_previous = dict(updater.PREVIOUS_SHA256)
        self.previous = {name: f"PREVIOUS_{i} = True\n".encode()
                         for i, name in enumerate(self.reviewed_previous)}
        previous = patch.object(updater, "PREVIOUS_SHA256", {
            name: updater._digest(data) for name, data in self.previous.items()})
        previous.start()
        self.addCleanup(previous.stop)
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES
                          if name not in self.previous}
        self.preserved.update({
            "runs/random_student_droppath/studies/keep/study.json": b'{"keep":"failed study"}',
            "runs/random_student_weight_decay/studies/keep/report/comparison.csv": b"keep baseline results",
            "runs/random_student_soft_only/studies/keep/study.json": b'{"keep":"original soft study"}',
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA",
        })
        for name, data in {**self.preserved, **self.previous}.items():
            self.write(self.project / name, data)
        for name in self.previous:
            (self.project / name).chmod(0o640)
        self.payload = {name: (b"RANDOM_SOFT_UPDATE = True\n" if name.endswith('.py') else
                        b'{"distillation_profile":"soft_only"}\n' if name.endswith('.json') else
                        "# 随机学生仅软标签蒸馏\n".encode()) for name in updater.PAYLOAD_FILES}
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

    def assert_previous(self):
        self.assert_preserved()
        for name, data in self.previous.items():
            self.assertEqual((self.project / name).read_bytes(), data)
            self.assertEqual((self.project / name).stat().st_mode & 0o777, 0o640)
        for name in self.payload:
            if name not in self.previous:
                self.assertFalse((self.project / name).exists())

    def test_preview_apply_idempotence_and_full_rollback(self):
        result = self.install()
        self.assertEqual(set(result['files']), set(self.payload))
        self.assertIsNone(result['backup'])
        self.assert_previous()
        result = self.install(True)
        self.assertEqual(len(result['files']), 8)
        backup = Path(result['backup'])
        journal = json.loads((backup / 'installation.json').read_text())
        self.assertEqual(set(journal['files']), set(self.payload))
        for name, data in self.payload.items():
            self.assertEqual((self.project / name).read_bytes(), data)
        self.assert_preserved()
        self.assertEqual(self.install(True)['files'], [])
        updater.rollback(self.project, backup, True)
        self.assert_previous()
        self.assertEqual(updater.rollback(self.project, backup, True)['files'], [])

    def test_only_the_two_published_previous_versions_are_replaceable(self):
        self.assertEqual(self.reviewed_previous, {
            "foundation_sft/random_student_distillation.py": "6cc060695bd0896631660512dd8d839944e714c6d83f754cb621b1f0c87618ab",
            "foundation_sft/random_student_suite.py": "57b28038c961ad7f63903999b20937afb2e6388e3b92f959db668fce300b4c22",
        })

    def test_unknown_model_edit_preserves_all_files(self):
        name = updater.PAYLOAD_FILES[0]
        user = b"USER_LOCAL_CHANGE = True\n"
        self.write(self.project / name, user)
        with self.assertRaisesRegex(ValueError, 'Unknown local changes preserved'):
            self.install(True)
        self.assertEqual((self.project / name).read_bytes(), user)
        self.assertEqual((self.project / updater.PAYLOAD_FILES[1]).read_bytes(), self.previous[updater.PAYLOAD_FILES[1]])
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_existing_unknown_guide_prevents_any_code_update(self):
        guide = self.project / updater.PAYLOAD_FILES[-1]
        self.write(guide, b"USER GUIDE")
        with self.assertRaisesRegex(ValueError, 'local changes preserved'):
            self.install(True)
        for name, data in self.previous.items():
            self.assertEqual((self.project / name).read_bytes(), data)
        self.assertEqual(guide.read_bytes(), b"USER GUIDE")

    def test_bad_checksum_missing_and_invalid_python_prevent_writes(self):
        name = updater.PAYLOAD_FILES[0]
        path = self.source / name
        path.write_bytes(b"AFTER_REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.install(True)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            self.install(True)
        path.write_bytes(b"def invalid(:\n")
        updater.PAYLOAD_SHA256[name] = updater._digest(path.read_bytes())
        with self.assertRaises(SyntaxError):
            self.install(True)
        self.assert_previous()

    def test_unfrozen_and_expanded_manifests_are_refused(self):
        for value in ({}, {**updater.PAYLOAD_SHA256, 'unexpected.py': '0'*64}):
            with patch.object(updater, 'PAYLOAD_SHA256', value), self.assertRaisesRegex(ValueError, 'not been frozen'):
                self.install(True)
        self.assert_previous()

    def test_invalid_configuration_json_prevents_any_update(self):
        name = 'foundation_sft/configs/random_student_soft_only.json'
        for data in (b'[1, 2]', b'{invalid JSON'):
            with self.subTest(data=data):
                (self.source / name).write_bytes(data)
                updater.PAYLOAD_SHA256[name] = updater._digest(data)
                with self.assertRaises(ValueError):
                    self.install(True)
                self.assert_previous()

    def test_missing_study_dependency_prevents_update(self):
        name = 'foundation_sft/droppath_baseline.py'
        (self.project / name).unlink()
        with self.assertRaisesRegex(ValueError, 'missing: ' + name):
            self.install(True)
        for name, data in self.previous.items():
            self.assertEqual((self.project / name).read_bytes(), data)

    def test_atomic_late_failure_rolls_back_reviewed_code(self):
        original = updater.os.link
        target = self.project / updater.PAYLOAD_FILES[-1]

        def fail_guide(source, destination):
            if Path(destination) == target:
                raise OSError('interrupted update')
            return original(source, destination)

        with patch.object(updater.os, 'link', side_effect=fail_guide), self.assertRaisesRegex(OSError, 'interrupted update'):
            self.install(True)
        self.assert_previous()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_rollback_preserves_later_user_edits(self):
        result = self.install(True)
        target = self.project / updater.PAYLOAD_FILES[0]
        target.write_bytes(b"USER_AFTER_REPAIR = True\n")
        with self.assertRaisesRegex(ValueError, 'Local changes preserved'):
            updater.rollback(self.project, result['backup'], True)
        self.assertEqual(target.read_bytes(), b"USER_AFTER_REPAIR = True\n")
        self.assert_preserved()

    def test_symlinks_lock_and_old_python_are_refused(self):
        target = self.project / updater.PAYLOAD_FILES[0]
        outside = self.project.parent / 'outside.py'
        outside.write_bytes(b"KEEP OUTSIDE")
        target.unlink()
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symbolic-link'):
            self.install(True)
        target.unlink()
        target.write_bytes(self.previous[updater.PAYLOAD_FILES[0]])
        target.chmod(0o640)
        lock = self.project / updater.LOCK_NAME
        lock.write_text('existing update')
        with self.assertRaises(FileExistsError):
            self.install(True)
        lock.unlink()
        with patch.object(updater.sys, 'version_info', (3,7,17)), self.assertRaisesRegex(ValueError, 'Python 3.10'):
            self.install(True)
        self.assertEqual(outside.read_bytes(), b"KEEP OUTSIDE")
        self.assert_previous()

    def test_package_has_only_fix_files_installer_and_manifest(self):
        installer = 'update_random_student_soft_only.py'
        self.write(self.source / installer, Path(updater.__file__).read_bytes())
        self.write(self.source / 'secret.pt', b"DO NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.source.parent / 'fix.zip'
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            expected = {'random_student_soft_only_update/' + name for name in self.payload}
            expected.update({'random_student_soft_only_update/' + installer, 'random_student_soft_only_update/SHA256SUMS.txt'})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 10)
            self.assertIsNone(archive.testzip())
            for line in archive.read('random_student_soft_only_update/SHA256SUMS.txt').decode().splitlines():
                expected, name = line.split('  ',1)
                self.assertEqual(updater._digest(archive.read('random_student_soft_only_update/' + name)), expected)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"CHANGED = True\n")
        with self.assertRaisesRegex(ValueError, 'reviewed random-student-soft-only manifest'):
            builder.package(self.source.parent/'bad.zip', self.source)


if __name__ == '__main__':
    unittest.main()
