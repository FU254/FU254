"""Partial-finetune delivery updates only three recognized training sources."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_partial_finetune_study as builder
import update_partial_finetune_study as updater


REPO = Path(__file__).resolve().parents[1]
BASELINE = "f374fbeb79abc2bd2702c2cf30ab31f31f242172"
MODIFIED = (
    "foundation_sft/models/vit_tiny_student.py",
    "foundation_sft/distillation.py",
    "foundation_sft/teacher_weight_training.py",
)
OLD_BYTES = {name: subprocess.check_output(["git", "show", BASELINE + ":" + name], cwd=REPO)
             for name in MODIFIED}
ENTRY = "run_partial_finetune_experiments.py"


class PartialFinetuneUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="partial-finetune-update-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.project = self.base / "project"
        self.source = self.base / "partial_finetune_update"
        self.preserved = {name: b"EXISTING = True\n" for name in updater.PREREQUISITES
                          if name not in MODIFIED}
        self.preserved.update({
            "runs/teacher_weight_study/studies/source/study.json": b'{"status":"completed"}',
            "runs/teacher_weight_study/studies/source/jobs/P-E/last.pt": b"KEEP BASELINE MODEL",
            "runs/teacher_weight_study/studies/source/jobs/P-R/parallel_registry.json": b'{"status":"completed"}',
            "foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth": b"KEEP IMAGENET",
            "foundation_weights/rad_dino/model.safetensors": b"KEEP TEACHER",
            "runs/foundation_sft/data_cache/pixels/keep.npy": b"KEEP DATA",
            "foundation_sft/teacher_weight_parallel.py": b"KEEP INSTALLED PARALLEL MODULE\n",
            "run_teacher_weight_training.py": b"KEEP INSTALLED WORKER ENTRY\n",
            "recover_teacher_weight_result.py": b"KEEP RECOVERY TOOL\n",
        })
        for name, data in self.preserved.items():
            self.write(self.project / name, data)
        for index, (name, data) in enumerate(OLD_BYTES.items()):
            self.write(self.project / name, data)
            (self.project / name).chmod(0o640 + index)
        self.previous_modes = {name: (self.project / name).stat().st_mode & 0o777 for name in MODIFIED}
        self.payload = {
            name: (b"PARTIAL_FINETUNE_UPDATE = True\n" if name.endswith(".py") else
                   b'{"source_study":"runs/teacher_weight_study/studies/source","seed":42}\n'
                   if name.endswith(".json") else "# 局部解冻实验\n".encode())
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
        for name, data in OLD_BYTES.items():
            self.assertEqual((self.project / name).read_bytes(), data, name)
            self.assertEqual((self.project / name).stat().st_mode & 0o777, self.previous_modes[name])
        for name in self.payload:
            if name not in MODIFIED:
                self.assertFalse((self.project / name).exists(), name)

    def test_only_three_known_sources_replaceable_and_new_entry_published_last(self):
        self.assertEqual(set(updater.PREVIOUS_SHA256), set(MODIFIED))
        for name, data in OLD_BYTES.items():
            self.assertEqual(updater._digest(data), updater.PREVIOUS_SHA256[name])
        self.assertEqual(updater.PAYLOAD_FILES[-1], ENTRY)
        self.assertEqual(len(updater.PAYLOAD_FILES), 8)
        self.assertTrue(set(MODIFIED).issubset(updater.PAYLOAD_FILES))
        self.assertNotIn("foundation_sft/teacher_weight_parallel.py", updater.PAYLOAD_FILES)
        self.assertNotIn("run_teacher_weight_training.py", updater.PAYLOAD_FILES)

    def test_preview_apply_idempotence_and_exact_rollback(self):
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
            self.assertEqual((self.project / name).stat().st_mode & 0o777,
                             self.previous_modes.get(name, 0o644))
            self.assertEqual(journal["files"][name]["previous_sha256"],
                             updater.PREVIOUS_SHA256.get(name))
        self.assert_preserved()
        self.assertEqual(self.install(True)["files"], [])
        updater.rollback(self.project, backup, True)
        self.assert_before()
        self.assertEqual(updater.rollback(self.project, backup, True)["files"], [])

    def test_unknown_existing_training_source_preserved_before_any_write(self):
        for name in MODIFIED:
            with self.subTest(name=name):
                local = b"USER_MODIFIED_TRAINING_SOURCE = True\n"
                (self.project / name).write_bytes(local)
                with self.assertRaisesRegex(ValueError, "local changes preserved"):
                    self.install(True)
                self.assertEqual((self.project / name).read_bytes(), local)
                self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
                self.assert_preserved()
                (self.project / name).write_bytes(OLD_BYTES[name])
        self.assert_before()

    def test_missing_required_old_source_is_not_added(self):
        name = MODIFIED[0]
        (self.project / name).unlink()
        with self.assertRaises(ValueError):
            self.install(True)
        self.assertFalse((self.project / name).exists())
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_unknown_new_module_preserved_without_training_changes(self):
        name = "foundation_sft/partial_finetune_study.py"
        local = b"USER_EXISTING_STUDY = True\n"
        self.write(self.project / name, local)
        with self.assertRaisesRegex(ValueError, "local changes preserved"):
            self.install(True)
        self.assertEqual((self.project / name).read_bytes(), local)
        for relative, data in OLD_BYTES.items():
            self.assertEqual((self.project / relative).read_bytes(), data)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        self.assert_preserved()

    def test_all_sources_checked_before_any_destination_change(self):
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

    def test_unfrozen_manifest_invalid_configuration_or_text_prevent_writes(self):
        for manifest in ({}, {**updater.PAYLOAD_SHA256, "unexpected.py": "0" * 64}):
            with patch.object(updater, "PAYLOAD_SHA256", manifest):
                with self.assertRaisesRegex(ValueError, "not been frozen"):
                    self.install(True)
        name = "foundation_sft/configs/partial_finetune_study.json"
        for data in (b"[1, 2]", b"{bad JSON"):
            (self.source / name).write_bytes(data)
            updater.PAYLOAD_SHA256[name] = updater._digest(data)
            with self.assertRaises(ValueError):
                self.install(True)
            self.assert_before()
        (self.source / name).write_bytes(self.payload[name])
        updater.PAYLOAD_SHA256[name] = updater._digest(self.payload[name])
        name = "foundation_sft/PARTIAL_FINETUNE_STUDY.md"
        (self.source / name).write_bytes(b"\xff\xfe")
        updater.PAYLOAD_SHA256[name] = updater._digest(b"\xff\xfe")
        with self.assertRaises(UnicodeError):
            self.install(True)
        self.assert_before()

    def test_missing_original_dependency_prevents_install(self):
        name = next(name for name in updater.PREREQUISITES if name not in MODIFIED)
        (self.project / name).unlink()
        with self.assertRaisesRegex(ValueError, "missing: " + name):
            self.install(True)
        self.assertFalse((self.project / updater.BACKUP_DIRECTORY).exists())
        for relative, data in OLD_BYTES.items():
            self.assertEqual((self.project / relative).read_bytes(), data)

    def test_install_compiles_without_importing_code_or_signaling_processes(self):
        name = "foundation_sft/partial_finetune_study.py"
        data = b"raise RuntimeError('installer must not execute code')\n"
        (self.source / name).write_bytes(data)
        updater.PAYLOAD_SHA256[name] = updater._digest(data)
        with patch("os.kill", side_effect=AssertionError("must not signal")), \
                patch("subprocess.Popen", side_effect=AssertionError("must not launch")):
            self.install(True)
        self.assertEqual((self.project / name).read_bytes(), data)
        self.assert_preserved()

    def test_last_entry_failure_restores_all_three_sources_and_new_files(self):
        original = updater.os.link
        entry = self.project / ENTRY
        def fail_entry(source, destination):
            if Path(destination) == entry:
                raise OSError("interrupted entry publication")
            return original(source, destination)
        with patch.object(updater.os, "link", side_effect=fail_entry), \
                self.assertRaisesRegex(OSError, "interrupted entry"):
            self.install(True)
        self.assert_before()
        self.assertFalse((self.project / updater.LOCK_NAME).exists())

    def test_late_existing_source_edit_is_preserved_on_rollback(self):
        installed = self.install(True)
        name = MODIFIED[1]
        local = b"USER_AFTER_INSTALL = True\n"
        (self.project / name).write_bytes(local)
        with self.assertRaisesRegex(ValueError, "Local changes preserved"):
            updater.rollback(self.project, installed["backup"], True)
        self.assertEqual((self.project / name).read_bytes(), local)
        self.assert_preserved()

    def test_tampered_old_version_backup_refused(self):
        installed = self.install(True)
        backup = Path(installed["backup"])
        journal_path = backup / "installation.json"
        journal = json.loads(journal_path.read_text())
        journal["files"][MODIFIED[0]]["previous_sha256"] = "0" * 64
        journal_path.write_text(json.dumps(journal))
        with self.assertRaisesRegex(ValueError, "recognized previous"):
            updater.rollback(self.project, backup, True)
        self.assertEqual((self.project / MODIFIED[0]).read_bytes(), self.payload[MODIFIED[0]])
        self.assert_preserved()

    def test_symlink_lock_and_old_python_refused(self):
        target = self.project / ENTRY
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
        with patch.object(updater.sys, "version_info", (3, 7, 17)), \
                self.assertRaisesRegex(ValueError, "Python 3.10"):
            self.install(True)
        self.assertEqual(outside.read_bytes(), b"KEEP OUTSIDE")
        self.assert_before()

    def test_package_contains_only_reviewed_code_configuration_and_manifest(self):
        installer_name = "update_partial_finetune_study.py"
        self.write(self.source / installer_name, Path(updater.__file__).read_bytes())
        self.write(self.source / "patients.csv", b"DO NOT SHIP")
        self.write(self.source / "teacher.pth", b"DO NOT SHIP")
        builder.freeze_manifest(self.source)
        output = self.base / "update.zip"
        with redirect_stdout(io.StringIO()):
            builder.package(output, self.source)
        with zipfile.ZipFile(output) as archive:
            prefix = "partial_finetune_update/"
            expected = {prefix + name for name in self.payload}
            expected.update({prefix + installer_name, prefix + "SHA256SUMS.txt"})
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(len(archive.namelist()), 10)
            self.assertIsNone(archive.testzip())
            for line in archive.read(prefix + "SHA256SUMS.txt").decode().splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(updater._digest(archive.read(prefix + relative)), digest)
        with self.assertRaises(FileExistsError):
            builder.package(output, self.source)
        (self.source / updater.PAYLOAD_FILES[0]).write_bytes(b"CHANGED = True\n")
        with self.assertRaisesRegex(ValueError, "reviewed partial-finetune-study manifest"):
            builder.package(self.base / "bad.zip", self.source)


if __name__ == "__main__":
    unittest.main()
