"""Verify the incremental overfit ZIP, preserved experiments, and exact rollback."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_single_teacher_overfit as builder
import update_partial_finetune_study as original_updater
import update_single_teacher_overfit as updater


REPO = Path(__file__).resolve().parents[1]
MODIFIED = tuple(updater.PREVIOUS_SHA256)
OLD_BYTES = {name: subprocess.check_output(
    ["git", "show", updater.BASE_REVISION + ":" + name], cwd=REPO) for name in MODIFIED}
CONFIG = "foundation_sft/configs/single_teacher_overfit.json"
GUIDE = "foundation_sft/SINGLE_TEACHER_OVERFIT.md"
ENTRY = "run_single_teacher_overfit.py"


class SingleTeacherOverfitUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="single-teacher-overfit-update-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / "project"
        self.source = self.base / "single_teacher_overfit_update"
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES}
        self.preserved.update({
            "runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96/suite.json":
                b'{"status":"completed","historical":true}',
            "runs/single_teacher_distillation/suites/original/distillation/best.pt": b"KEEP OLD STUDENT",
            "runs/single_teacher_overfit/studies/existing/jobs/A/best.pt": b"KEEP COMPLETED NEW A",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA CACHE",
            "foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth": b"KEEP IMAGENET",
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "foundation_sft/configs/partial_finetune_augmentation.json": b'{"KEEP":"OTHER STUDY"}',
            ".partial-finetune-study-backups/original/installation.json": b"KEEP OLDER BACKUP",
            updater.ENGINE_FILE: (REPO / updater.ENGINE_FILE).read_bytes(),
        })
        for name, data in {**self.preserved, **OLD_BYTES}.items():
            self.write(self.project / name, data)
        for name in MODIFIED:
            (self.project / name).chmod(0o640)
        self.previous_modes = {name: (self.project / name).stat().st_mode & 0o777 for name in MODIFIED}
        self.payload = {name: b"OVERFIT_UPDATE = True\n" for name in updater.PAYLOAD_FILES}
        self.payload[CONFIG] = b'{"study_type":"single_teacher_overfit","seed":42}\n'
        self.payload[GUIDE] = "# 单教师过拟合诊断\n".encode()
        for name, data in self.payload.items():
            self.write(self.source / name, data)
        self.manifest = {name: hashlib.sha256(data).hexdigest() for name, data in self.payload.items()}
        manifest = patch.object(updater, "PAYLOAD_SHA256", self.manifest)
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
        for name, data in OLD_BYTES.items():
            self.assertEqual((self.project / name).read_bytes(), data, name)
            self.assertEqual((self.project / name).stat().st_mode & 0o777, self.previous_modes[name])
        for name in set(updater.PAYLOAD_FILES) - set(MODIFIED):
            self.assertFalse((self.project / name).exists(), name)

    def prepare_packaging(self):
        for name in ("update_single_teacher_overfit.py", updater.ENGINE_FILE):
            data = (REPO / name).read_bytes()
            if name == "update_single_teacher_overfit.py":
                data = re.sub(rb"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", b"PAYLOAD_SHA256 = {}", data, count=1)
            self.write(self.source / name, data)

    def test_only_one_baseline_file_replaced_and_shared_engine_stays_unchanged(self):
        names = ("PAYLOAD_FILES", "PREVIOUS_SHA256", "PAYLOAD_SHA256", "PREREQUISITES", "LOCK_NAME")
        before = copy.deepcopy({name: getattr(original_updater, name) for name in names})
        self.assertEqual(len(updater.PAYLOAD_FILES), 7)
        self.assertEqual(MODIFIED, ("foundation_sft/distillation.py",))
        for name, data in OLD_BYTES.items():
            self.assertEqual(hashlib.sha256(data).hexdigest(), updater.PREVIOUS_SHA256[name])
        self.install()
        self.assertEqual(before, {name: getattr(original_updater, name) for name in names})
        self.assertEqual(hashlib.sha256((REPO / updater.ENGINE_FILE).read_bytes()).hexdigest(), updater.ENGINE_SHA256)
        self.assertNotEqual(updater.BACKUP_DIRECTORY, original_updater.BACKUP_DIRECTORY)
        self.assertNotEqual(updater.LOCK_NAME, original_updater.LOCK_NAME)

    def test_preview_install_idempotence_and_exact_rollback(self):
        self.assertEqual(set(self.install()["files"]), set(self.payload))
        self.assert_before()
        installed = self.install(True)
        backup = Path(installed["backup"])
        self.assertEqual(backup.parent.name, updater.BACKUP_DIRECTORY)
        journal = json.loads((backup / "installation.json").read_text())
        for name, data in self.payload.items():
            self.assertEqual((self.project / name).read_bytes(), data)
            self.assertEqual(journal["files"][name]["previous_sha256"], updater.PREVIOUS_SHA256.get(name))
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        self.assertEqual(set(updater.rollback(self.project, backup)["files"]), set(self.payload))
        updater.rollback(self.project, backup, True)
        self.assert_before()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_unknown_edits_to_old_or_new_files_refuse_all_writes(self):
        for name in (*MODIFIED, CONFIG, ENTRY):
            with self.subTest(name=name):
                self.write(self.project / name, b"USER_LOCAL_EDIT = True\n")
                with self.assertRaisesRegex(ValueError, "local changes preserved"):
                    self.install(True)
                self.assertEqual((self.project / name).read_bytes(), b"USER_LOCAL_EDIT = True\n")
                self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
                self.assert_preserved()
                if name in OLD_BYTES:
                    (self.project / name).write_bytes(OLD_BYTES[name])
                else:
                    (self.project / name).unlink()
        self.assert_before()

    def test_missing_dependencies_tampered_sources_and_unfrozen_manifest_refused(self):
        (self.source / GUIDE).write_bytes(b"SOURCE TAMPERING")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        (self.source / GUIDE).write_bytes(self.payload[GUIDE])
        name = updater.PREREQUISITES[0]
        (self.project / name).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + name):
            self.install(True)
        self.write(self.project / name, self.preserved[name])
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        self.assert_before()

    def test_invalid_python_json_or_symlink_source_is_refused_before_writes(self):
        for name, data, error in ((ENTRY, b"def invalid(:\n", SyntaxError),
                                   (CONFIG, b"[]", ValueError)):
            with self.subTest(name=name):
                (self.source / name).write_bytes(data)
                with patch.dict(updater.PAYLOAD_SHA256, {name: hashlib.sha256(data).hexdigest()}):
                    with self.assertRaises(error):
                        self.install(True)
                (self.source / name).write_bytes(self.payload[name])
                self.assert_before()
        external = self.base / "external.py"
        external.write_bytes(self.payload[ENTRY])
        (self.source / ENTRY).unlink()
        (self.source / ENTRY).symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.install(True)
        self.assert_before()

    def test_failure_publishing_last_entry_restores_baseline_and_removes_new_files(self):
        original = updater._engine.os.link

        def fail_entry(source, destination):
            if Path(destination) == self.project / ENTRY:
                raise OSError("interrupted entry publication")
            return original(source, destination)

        with patch.object(updater._engine.os, "link", side_effect=fail_entry), \
                self.assertRaisesRegex(OSError, "interrupted entry"):
            self.install(True)
        self.assert_before()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_rollback_preserves_later_edits_and_refuses_tampered_backup(self):
        installed = self.install(True)
        name = MODIFIED[0]
        (self.project / name).write_bytes(b"USER AFTER INSTALL\n")
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, installed["backup"], True)
        self.assertEqual((self.project / name).read_bytes(), b"USER AFTER INSTALL\n")
        (self.project / name).write_bytes(self.payload[name])
        (Path(installed["backup"]) / "previous" / name).write_bytes(b"TAMPERED BACKUP")
        with self.assertRaisesRegex(ValueError, "backup checksum mismatch"):
            updater.rollback(self.project, installed["backup"], True)
        self.assert_preserved()

    def test_install_never_launches_training_signals_processes_or_mutates_upload(self):
        with patch("subprocess.Popen", side_effect=AssertionError("must not launch training")), \
                patch("os.kill", side_effect=AssertionError("must not signal processes")):
            self.install(True)
        for name, data in self.payload.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        self.assert_preserved()

    def test_real_zip_contains_only_payload_helper_installer_and_checksums(self):
        self.prepare_packaging()
        self.write(self.source / "patient_data.csv", b"DO NOT SHIP DATA")
        self.write(self.source / "teacher.pth", b"DO NOT SHIP WEIGHTS")
        with self.assertRaisesRegex(ValueError, "Freeze the reviewed"):
            builder.package(self.base / "unfrozen.zip", self.source)
        builder.freeze_manifest(self.source)
        output = self.base / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        unpacked = self.base / "unpacked"
        with zipfile.ZipFile(output) as archive:
            expected = {builder.PREFIX + name for name in (*updater.PAYLOAD_FILES,
                "update_single_teacher_overfit.py", updater.ENGINE_FILE, "SHA256SUMS.txt")}
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 10)
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read(builder.PREFIX + updater.ENGINE_FILE), (REPO / updater.ENGINE_FILE).read_bytes())
            for line in archive.read(builder.PREFIX + "SHA256SUMS.txt").decode().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(builder.PREFIX + relative)).hexdigest(), digest)
            archive.extractall(unpacked)
        uploaded = unpacked / builder.PREFIX.rstrip("/")
        command = [sys.executable, str(uploaded / "update_single_teacher_overfit.py"),
                   "--project", str(self.project)]
        preview = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(set(preview["files"]), set(self.payload))
        self.assert_before()
        installed = json.loads(subprocess.check_output([*command, "--apply"], text=True))
        repeated = json.loads(subprocess.check_output([*command, "--apply"], text=True))
        self.assertEqual(repeated["files"], [])
        self.assert_preserved()
        subprocess.check_output([*command, "--rollback", installed["backup"], "--apply"], text=True)
        self.assert_before()
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / MODIFIED[0]).write_bytes(b"CHANGED AFTER REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed single-teacher overfit manifest"):
            builder.package(self.base / "changed.zip", self.source)


if __name__ == "__main__":
    unittest.main()
