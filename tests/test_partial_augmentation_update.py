"""Exercise the incremental installer against the published project and preserved runs."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_partial_augmentation as builder
import update_partial_augmentation as updater
import update_partial_finetune_study as original_updater


REPO = Path(__file__).resolve().parents[1]
MODIFIED = tuple(updater.PREVIOUS_SHA256)
OLD_BYTES = {name: subprocess.check_output(
    ["git", "show", updater.BASE_REVISION + ":" + name], cwd=REPO) for name in MODIFIED}
ENTRY = "run_partial_finetune_experiments.py"
CONFIG = "foundation_sft/configs/partial_finetune_augmentation.json"
GUIDE = "foundation_sft/PARTIAL_FINETUNE_AUGMENTATION.md"


class PartialAugmentationUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="partial-augmentation-update-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / "project"
        self.source = self.base / "partial_augmentation_update"
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES
                          if name not in MODIFIED}
        for name in (ENTRY, "foundation_sft/models/vit_tiny_student.py",
                     "foundation_sft/configs/partial_finetune_study.json",
                     "foundation_sft/PARTIAL_FINETUNE_STUDY.md", updater.ENGINE_FILE):
            self.preserved[name] = subprocess.check_output(
                ["git", "show", updater.BASE_REVISION + ":" + name], cwd=REPO)
        self.preserved.update({
            "runs/partial_finetune_study/studies/study_20261027T150707Z_1ad30256/study.json": b'{"status":"completed"}',
            "runs/teacher_weight_study/studies/old/jobs/P-E/best.pt": b"KEEP FULL12 MODEL",
            "runs/teacher_weight_study/studies/old/jobs/P-R/parallel_registry.json": b'{"recovered":true}',
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA CACHE",
            "foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth": b"KEEP IMAGENET",
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "foundation_sft/teacher_weight_parallel.py": b"KEEP P-R SHIM\n",
            "run_teacher_weight_training.py": b"KEEP ORIGINAL WORKER\n",
            "recover_teacher_weight_result.py": b"KEEP RECOVERY TOOL\n",
            ".partial-finetune-study-backups/original/installation.json": b"KEEP ORIGINAL BACKUP",
        })
        for arm in ("last8", "last4", "last2"):
            self.preserved[f"runs/partial_finetune_study/studies/old/jobs/{arm}/best.pt"] = ("KEEP " + arm).encode()
        for name, data in {**self.preserved, **OLD_BYTES}.items():
            self.write(self.project / name, data)
        for index, name in enumerate(MODIFIED):
            (self.project / name).chmod(0o640 + index)
        self.previous_modes = {name: (self.project / name).stat().st_mode & 0o777 for name in MODIFIED}
        self.payload = {name: b"AUGMENTATION_UPDATE = True\n" for name in MODIFIED}
        self.payload.update({CONFIG: b'{"study_type":"intensity_augmentation"}\n',
                             GUIDE: "# 强度增强筛选\n".encode()})
        for name, data in self.payload.items():
            self.write(self.source / name, data)
        manifest = patch.object(updater, "PAYLOAD_SHA256", {
            name: hashlib.sha256(data).hexdigest() for name, data in self.payload.items()})
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
        for name in (CONFIG, GUIDE):
            self.assertFalse((self.project / name).exists(), name)

    def test_only_five_published_files_replaced_and_original_engine_is_isolated(self):
        before = copy.deepcopy({name: getattr(original_updater, name) for name in (
            "PAYLOAD_FILES", "PREVIOUS_SHA256", "PAYLOAD_SHA256", "PREREQUISITES", "LOCK_NAME")})
        self.assertEqual(len(updater.PAYLOAD_FILES), 7)
        self.assertEqual(set(updater.PAYLOAD_FILES) - set(MODIFIED), {CONFIG, GUIDE})
        for name, data in OLD_BYTES.items():
            self.assertEqual(hashlib.sha256(data).hexdigest(), updater.PREVIOUS_SHA256[name])
        self.assertTrue({ENTRY, "foundation_sft/data.py", "foundation_sft/partial_finetune_study.py",
                         "foundation_sft/models/vit_tiny_student.py"}.issubset(updater.PREREQUISITES))
        self.install()
        self.assertEqual(before, {name: getattr(original_updater, name) for name in before})
        self.assertNotEqual(updater.BACKUP_DIRECTORY, original_updater.BACKUP_DIRECTORY)
        self.assertNotEqual(updater.LOCK_NAME, original_updater.LOCK_NAME)
        self.assertEqual(hashlib.sha256((REPO / updater.ENGINE_FILE).read_bytes()).hexdigest(), updater.ENGINE_SHA256)

    def test_preview_apply_idempotence_and_exact_rollback(self):
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

    def test_unknown_local_source_or_new_config_prevents_all_writes(self):
        for name in (*MODIFIED, CONFIG):
            with self.subTest(name=name):
                local = b"USER_LOCAL_EDIT = True\n"
                self.write(self.project / name, local)
                with self.assertRaisesRegex(ValueError, "local changes preserved"):
                    self.install(True)
                self.assertEqual((self.project / name).read_bytes(), local)
                self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
                self.assert_preserved()
                if name in OLD_BYTES:
                    (self.project / name).write_bytes(OLD_BYTES[name])
                else:
                    (self.project / name).unlink()
        self.assert_before()

    def test_all_payload_checked_before_install_and_missing_dependency_refused(self):
        name = GUIDE
        (self.source / name).write_bytes(b"SOURCE TAMPERING")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.install(True)
        (self.source / name).write_bytes(self.payload[name])
        (self.project / ENTRY).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + ENTRY):
            self.install(True)
        self.write(self.project / ENTRY, self.preserved[ENTRY])
        self.assert_before()

    def test_unfrozen_manifest_invalid_python_and_json_prevent_writes(self):
        with patch.object(updater, "PAYLOAD_SHA256", {}):
            with self.assertRaisesRegex(ValueError, "not been frozen"):
                self.install(True)
        for name, data, error in ((MODIFIED[-1], b"def invalid(:\n", SyntaxError),
                                   (CONFIG, b"[]", ValueError)):
            with self.subTest(name=name):
                (self.source / name).write_bytes(data)
                with patch.dict(updater.PAYLOAD_SHA256, {name: hashlib.sha256(data).hexdigest()}):
                    with self.assertRaises(error):
                        self.install(True)
                (self.source / name).write_bytes(self.payload[name])
                self.assert_before()

    def test_late_publication_failure_restores_five_files_and_removes_new_config(self):
        original = updater._engine.os.link
        def fail_guide(source, destination):
            if Path(destination) == self.project / GUIDE:
                raise OSError("interrupted guide publication")
            return original(source, destination)
        with patch.object(updater._engine.os, "link", side_effect=fail_guide), \
                self.assertRaisesRegex(OSError, "interrupted guide"):
            self.install(True)
        self.assert_before()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_rollback_preserves_later_user_edits_and_rejects_tampered_backup(self):
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

    def test_new_installer_does_not_execute_training_or_modify_source(self):
        with patch("subprocess.Popen", side_effect=AssertionError("must not launch training")), \
                patch("os.kill", side_effect=AssertionError("must not signal processes")):
            self.install(True)
        for name, data in self.payload.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        self.assert_preserved()

    def test_package_contains_only_seven_payloads_both_installers_and_manifest(self):
        for name in ("update_partial_augmentation.py", updater.ENGINE_FILE):
            data = (REPO / name).read_bytes()
            if name == "update_partial_augmentation.py":
                data = re.sub(rb"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", b"PAYLOAD_SHA256 = {}", data, count=1)
            self.write(self.source / name, data)
        self.write(self.source / "patient_data.csv", b"DO NOT SHIP DATA")
        self.write(self.source / "teacher.pth", b"DO NOT SHIP WEIGHTS")
        with self.assertRaisesRegex(ValueError, "Freeze the reviewed"):
            builder.package(self.base / "unfrozen.zip", self.source)
        builder.freeze_manifest(self.source)
        output = self.base / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            prefix = "partial_augmentation_update/"
            expected = {prefix + name for name in (*updater.PAYLOAD_FILES,
                "update_partial_augmentation.py", updater.ENGINE_FILE, "SHA256SUMS.txt")}
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 10)
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read(prefix + updater.ENGINE_FILE), (REPO / updater.ENGINE_FILE).read_bytes())
            for line in archive.read(prefix + "SHA256SUMS.txt").decode().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(prefix + relative)).hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / MODIFIED[0]).write_bytes(b"CHANGED AFTER REVIEW = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed augmentation manifest"):
            builder.package(self.base / "changed.zip", self.source)

    def test_configuration_names_exact_completed_last8_study(self):
        config = json.loads((REPO / CONFIG).read_text())
        self.assertEqual(config, {
            "study_type": "intensity_augmentation",
            "source_study": "runs/partial_finetune_study/studies/study_20261027T150707Z_1ad30256",
            "seeds": [42], "brightness_jitter": 0.1, "contrast_jitter": 0.1,
            "devices": ["cuda:0"], "max_parallel": 1,
            "cache_dir": "runs/foundation_sft/data_cache", "output_root": "runs/partial_finetune_augmentation",
        })


if __name__ == "__main__":
    unittest.main()
