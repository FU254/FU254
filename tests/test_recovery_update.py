"""Verify that the two-source updater protects existing experiments and edits."""

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import update_foundation_recovery as updater


class RecoveryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recovery-update-test-")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        (self.project / "foundation_sft").mkdir()
        self.old = b"VERSION = 'original'\n"
        self.new = b"VERSION = 'fp32-recovery'\n"
        self.helper = b"RECOVERY = True\n"
        self.trainer = self.project / "foundation_sft" / "train.py"
        self.trainer.write_bytes(self.old)
        self.trainer.chmod(0o640)
        self.config = self.project / "foundation_sft" / "config.json"
        self.config.write_bytes(b'{"preserve": true}\n')
        self.fingerprints = {
            "foundation_sft/train.py": {
                "previous": hashlib.sha256(self.old).hexdigest(),
                "updated": hashlib.sha256(self.new).hexdigest(),
            },
            "foundation_sft/recover_experiment.py": {
                "previous": None,
                "updated": hashlib.sha256(self.helper).hexdigest(),
            },
        }
        self.payloads = {"train.py": self.new, "recover_experiment.py": self.helper}
        self.hash_patch = patch.object(updater, "FILES", self.fingerprints)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def install(self, download=None):
        with redirect_stdout(io.StringIO()):
            return updater.install(
                self.project, "a" * 40,
                download=download or (lambda url: self.payloads[url.rsplit("/", 1)[-1]]),
            )

    def test_update_backs_up_original_and_is_idempotent(self):
        backup = self.install()
        self.assertEqual((backup / "train.py").read_bytes(), self.old)
        self.assertEqual(self.trainer.read_bytes(), self.new)
        self.assertEqual(self.trainer.stat().st_mode & 0o777, 0o640)
        self.assertEqual((self.project / "foundation_sft" / "recover_experiment.py").read_bytes(), self.helper)
        self.assertEqual(self.config.read_bytes(), b'{"preserve": true}\n')
        self.assertIsNone(self.install(lambda url: self.fail("Idempotent update must not download")))
        self.assertEqual(len(list(backup.parent.iterdir())), 1)
        self.assertFalse((self.project / ".foundation-recovery-update.lock").exists())

    def test_unknown_local_edits_are_refused_before_download(self):
        customized = self.old + b"CUSTOM = True\n"
        self.trainer.write_bytes(customized)
        with self.assertRaisesRegex(ValueError, "local changes"):
            self.install(lambda url: self.fail("Unknown edits must not trigger downloads"))
        self.assertEqual(self.trainer.read_bytes(), customized)
        self.assertFalse((self.project / "foundation_sft" / "recover_experiment.py").exists())
        self.assertFalse((self.project / ".foundation-recovery-update.lock").exists())

    def test_bad_second_payload_changes_neither_source(self):
        self.payloads["recover_experiment.py"] += b"TAMPERED = True\n"
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.install()
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertFalse((self.project / "foundation_sft" / "recover_experiment.py").exists())
        self.assertFalse((self.project / "runs").exists())

    def test_second_replace_failure_rolls_back_first_source(self):
        original_replace = updater.os.replace

        def fail_helper_replace(source, target):
            if Path(target).name == "recover_experiment.py":
                raise OSError("Synthetic publication failure")
            return original_replace(source, target)

        with patch.object(updater.os, "replace", side_effect=fail_helper_replace):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.install()
        self.assertEqual(self.trainer.read_bytes(), self.old)
        self.assertEqual(self.trainer.stat().st_mode & 0o777, 0o640)
        self.assertFalse((self.project / "foundation_sft" / "recover_experiment.py").exists())
        self.assertFalse((self.project / ".foundation-recovery-update.lock").exists())
        self.assertFalse(list((self.project / "foundation_sft").glob(".recovery-update-*.tmp")))

    def test_symbolic_link_destination_is_refused(self):
        outside = self.project / "outside.py"
        outside.write_bytes(self.old)
        self.trainer.unlink()
        self.trainer.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.install(lambda url: self.fail("Links must be rejected before downloads"))
        self.assertEqual(outside.read_bytes(), self.old)


if __name__ == "__main__":
    unittest.main()
