"""Incremental installation and weight downloads fail without partial updates."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import download_extra_vlm_weights as downloader
import update_foundation_vlm_extra as updater


class ExtraVLMInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project, self.source = self.root/"project", self.root/"source"
        self.old = b"SUPPORTED_MODELS = ('eva_x',)\n"
        self.payloads = {updater.REGISTRY: b"SUPPORTED_MODELS = ('eva_x', 'mgca')\n",
                         "foundation_sft/models/mgca.py": b"MGCA = True\n",
                         "foundation_sft/configs/common_vlm.json": b'{"base_config":"common.json"}\n'}
        for relative, data in self.payloads.items():
            path = self.source/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        target = self.project/updater.REGISTRY
        target.parent.mkdir(parents=True)
        target.write_bytes(self.old)
        (self.project/"foundation_sft/train.py").write_bytes(b"EXISTING_TRAINER = True\n")
        common = self.project/"foundation_sft/configs"
        common.mkdir()
        (common/"common.json").write_bytes(b'{"data":"custom-cohort","split_manifest":"frozen.csv"}\n')
        (common/"common_vlm.json").write_bytes(b'{"base_config":"common.json","training":{"batch_size":2}}\n')
        self.originals = {str(path.relative_to(self.project)):path.read_bytes()
                          for path in self.project.rglob("*") if path.is_file()}
        self.hashes = patch.object(updater,"PAYLOAD_SHA256", {key:updater._digest(value) for key,value in self.payloads.items()})
        self.previous = patch.object(updater,"PREVIOUS_SHA256", {updater.REGISTRY:[updater._digest(self.old)]})
        self.hashes.start(); self.previous.start()
        self.addCleanup(self.hashes.stop);self.addCleanup(self.previous.stop)

    def test_dry_plan_apply_and_repeat_preserve_custom_data_and_base_settings(self):
        dry = updater.install(self.project,self.source)
        self.assertEqual(set(dry["files"]), {updater.REGISTRY,"foundation_sft/models/mgca.py"})
        self.assertFalse((self.project/"runs").exists())
        applied = updater.install(self.project,self.source,True)
        self.assertEqual(applied["files"][-1],updater.REGISTRY)
        for path in ("foundation_sft/train.py","foundation_sft/configs/common.json","foundation_sft/configs/common_vlm.json"):
            self.assertEqual((self.project/path).read_bytes(),self.originals[path])
        self.assertEqual((Path(applied["backup"])/updater.REGISTRY).read_bytes(),self.old)
        self.assertEqual(updater.install(self.project,self.source,True)["files"],[])

    def test_invalid_payload_and_unknown_local_edits_refuse_every_write(self):
        path = self.source/"foundation_sft/models/mgca.py"
        path.write_bytes(b"INVALID = True\n")
        with self.assertRaisesRegex(ValueError,"checksum"):
            updater.install(self.project,self.source,True)
        self.assertEqual((self.project/updater.REGISTRY).read_bytes(),self.old)
        self.assertFalse((self.project/"runs").exists())
        path.write_bytes(self.payloads["foundation_sft/models/mgca.py"])
        (self.project/updater.REGISTRY).write_bytes(self.old+b"LOCAL = True\n")
        with self.assertRaisesRegex(ValueError,"Unknown local"):
            updater.install(self.project,self.source,True)

    def test_registry_failure_rolls_back_and_symlink_parent_cannot_escape_project(self):
        write = updater._atomic_write
        def fail(path,data,mode):
            if path==self.project/updater.REGISTRY and data!=self.old:
                raise OSError("simulated disk failure")
            write(path,data,mode)
        with patch.object(updater,"_atomic_write",side_effect=fail), self.assertRaises(OSError):
            updater.install(self.project,self.source,True)
        self.assertFalse((self.project/"foundation_sft/models/mgca.py").exists())
        self.assertEqual((self.project/updater.REGISTRY).read_bytes(),self.old)
        self.assertFalse((self.project/".foundation-vlm-extra-update.lock").exists())
        outside = self.root/"outside"
        outside.mkdir()
        target = self.project/"foundation_sft/models"
        (target/"__init__.py").unlink()
        target.rmdir()
        target.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"symbolic"):
            updater.install(self.project,self.source,True)
        self.assertEqual(list(outside.iterdir()),[])


class ExtraVLMDownloadTests(unittest.TestCase):
    def test_plan_has_no_network_or_torch_dependency_and_manual_models_remain_missing(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), patch.object(
                downloader.urllib.request,"urlopen",side_effect=AssertionError("No network")):
            root=Path(folder)/"not-created"
            self.assertEqual(downloader.main(["--output",str(root),"--plan"]),0)
            self.assertFalse(root.exists())
            self.assertEqual(downloader.main(["--models","mgca","--output",str(root)]),2)
            self.assertFalse(root.exists())

    def test_interrupted_tls_or_wrong_length_and_digest_leave_no_installed_file(self):
        class Response(io.BytesIO):
            headers={"Content-Length":"100"}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            asset={"filename":"model.pt","url":"https://example.invalid/model.pt"}
            errors=[urllib.error.URLError("TLS EOF"), Response(b"truncated")]
            for response in errors:
                options={"side_effect":response} if isinstance(response,Exception) else {"return_value":response}
                with redirect_stdout(io.StringIO()), patch.object(downloader.urllib.request,"urlopen",**options):
                    with self.assertRaises((OSError,ValueError)):
                        downloader.download(asset,root)
                self.assertFalse((root/"model.pt").exists())
                self.assertEqual(list(root.glob("*.part")),[])
            (root/"model.pt").write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError,"checksum"):
                downloader.download({**asset,"sha256":"0"*64},root)
            self.assertEqual((root/"model.pt").read_bytes(),b"existing")


class ExtraVLMPublishedManifestTests(unittest.TestCase):
    def test_all_payload_hashes_match_and_original_patient_data_settings_are_not_payloads(self):
        root=Path(__file__).resolve().parents[1]
        self.assertNotIn("foundation_sft/configs/common.json",updater.PAYLOAD_SHA256)
        self.assertNotIn("foundation_sft/train.py",updater.PAYLOAD_SHA256)
        self.assertNotIn("foundation_sft/configs/experiments.json",updater.PAYLOAD_SHA256)
        self.assertNotIn("foundation_sft/configs/experiments_vlm.json",updater.PAYLOAD_SHA256)
        for relative,digest in updater.PAYLOAD_SHA256.items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),digest,relative)
            self.assertFalse(relative.startswith(("runs/","foundation_weights/")))


if __name__=="__main__":
    unittest.main()
