"""Real truncated DICOMs are restored only from unique matching originals."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from foundation_sft.audit_data import audit_split
from foundation_sft.data import _dicom_pixels
from foundation_sft.recover_data import recover_from_backup


class DicomRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dicom-recovery-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "PRIVATE_ORIGINALS"
        self.source.mkdir()
        self.target = self.root / "PRIVATE_COHORT" / "PRIVATE_IMAGE.dcm"
        self.target.parent.mkdir()
        self.original = self.source / self.target.name
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        self.dataset = FileDataset(str(self.original), {}, file_meta=meta, preamble=b"\0" * 128)
        self.dataset.is_little_endian = True
        self.dataset.is_implicit_VR = False
        self.dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        self.dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        self.dataset.PatientID = "PRIVATE_PATIENT"
        self.dataset.PatientName = "Private^Name"
        self.dataset.AccessionNumber = "PRIVATE_ACC"
        self.dataset.Rows = self.dataset.Columns = 8
        self.dataset.SamplesPerPixel = 1
        self.dataset.PhotometricInterpretation = "MONOCHROME2"
        self.dataset.BitsAllocated = self.dataset.BitsStored = 16
        self.dataset.HighBit = 15
        self.dataset.PixelRepresentation = 0
        self.dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes()
        self.write(self.original, self.dataset)
        self.complete = self.original.read_bytes()
        self.damaged = self.complete[:-64]
        self.target.write_bytes(self.damaged)
        self.target.chmod(0o640)
        self.manifest = self.root / "frozen.csv"
        self.split = pd.DataFrame([{"path": str(self.target), "split": "train", "label": 1}])
        self.split.to_csv(self.manifest, index=False)
        self.frozen = self.manifest.read_bytes()
        self.audit = self.root / "audit"
        with contextlib.redirect_stdout(io.StringIO()):
            audit_split(self.split, self.audit)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, dataset):
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dataset, **options)

    def recover(self, apply=False):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = recover_from_backup(self.audit, self.source, apply=apply)
        technical = json.dumps(result) + stdout.getvalue()
        for secret in ("PRIVATE_ORIGINALS", "PRIVATE_COHORT", "PRIVATE_IMAGE", "PRIVATE_PATIENT",
                       "Private^Name", "PRIVATE_ACCESSION", str(self.dataset.SOPInstanceUID)):
            self.assertNotIn(secret, technical)
        self.assertEqual(self.manifest.read_bytes(), self.frozen)
        return result

    def test_dry_run_verifies_source_but_preserves_current_bytes_and_manifest(self):
        result = self.recover()
        self.assertEqual(result["counts"]["ready"], 1)
        self.assertEqual(result["counts"]["restored"], 0)
        self.assertEqual(self.target.read_bytes(), self.damaged)
        self.assertFalse((Path(result["output"]) / "backup").exists())
        self.assertEqual(self.original.read_bytes(), self.complete)

    def test_apply_restores_original_atomically_and_preserves_previous_bytes(self):
        result = self.recover(apply=True)
        self.assertEqual(result["counts"]["restored"], 1)
        self.assertEqual(self.target.read_bytes(), self.complete)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        backup = Path(result["output"]) / "backup" / "000000.dcm"
        self.assertEqual(backup.read_bytes(), self.damaged)
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        self.assertEqual((Path(result["output"]) / "recovery_private.csv").stat().st_mode & 0o777, 0o600)
        self.assertEqual(_dicom_pixels(pydicom.dcmread(self.target)).shape, (8, 8))
        self.assertEqual(self.original.read_bytes(), self.complete)
        self.assertFalse(list(self.target.parent.glob(".dicom_recovery_*")))
        second = self.recover(apply=True)
        self.assertEqual(second["counts"]["already_valid"], 1)
        self.assertFalse((Path(second["output"]) / "backup").exists())

    def test_renamed_original_is_found_even_when_same_basename_file_is_wrong(self):
        self.original.rename(self.source / "renamed_original.dcm")
        self.dataset.SOPInstanceUID = generate_uid()
        self.dataset.file_meta.MediaStorageSOPInstanceUID = self.dataset.SOPInstanceUID
        self.write(self.original, self.dataset)
        self.assertEqual(self.recover()["counts"]["ready"], 1)

    def test_same_sop_with_changed_identity_or_pixel_metadata_is_refused(self):
        for field, value in (("PatientID", "OTHER_PATIENT"), ("Rows", 4), ("WindowCenter", 30)):
            with self.subTest(field=field):
                changed = pydicom.dcmread(self.original)
                setattr(changed, field, value)
                self.write(self.original, changed)
                self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
                self.assertEqual(self.target.read_bytes(), self.damaged)
                self.original.write_bytes(self.complete)

    def test_inconsistent_file_meta_sop_identity_is_refused(self):
        self.dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        # write_like_original preserves deliberately contradictory file meta.
        pydicom.dcmwrite(self.original, self.dataset, write_like_original=True)
        self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)

    def test_corrupt_original_is_not_copied(self):
        self.original.write_bytes(self.complete[:-32])
        self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)

    def test_decodable_original_with_short_declared_payload_is_refused(self):
        # Two trailing bytes are padding beyond the declared pixel layout.
        self.dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes() + b"\0\0"
        self.write(self.original, self.dataset)
        self.original.write_bytes(self.original.read_bytes()[:-2])
        self.assertEqual(pydicom.dcmread(self.original).pixel_array.shape, (8, 8))
        self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)

    def test_different_complete_originals_with_same_sop_are_ambiguous(self):
        pixels = np.arange(64, dtype=np.uint16)
        pixels[32:] += 7
        self.dataset.PixelData = pixels.tobytes()
        self.write(self.source / "different_pixels.dcm", self.dataset)
        self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)

    def test_original_must_match_surviving_pixels_of_truncated_file(self):
        self.dataset.PixelData = (np.arange(64, dtype=np.uint16) + 7).tobytes()
        self.write(self.original, self.dataset)
        self.assertEqual(self.recover(apply=True)["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)

    def test_identical_duplicate_originals_are_one_verified_choice(self):
        shutil.copy2(self.original, self.source / "identical_copy.dcm")
        self.assertEqual(self.recover(apply=True)["counts"]["restored"], 1)
        self.assertEqual(self.target.read_bytes(), self.complete)

    def test_changed_source_during_copy_keeps_target_and_backup(self):
        copy = shutil.copy2

        def unstable_copy(source, destination):
            result = copy(source, destination)
            if Path(source) == self.original:
                Path(destination).write_bytes(self.complete[:-32])
            return result

        with mock.patch("foundation_sft.recover_data.shutil.copy2", side_effect=unstable_copy):
            result = self.recover(apply=True)
        self.assertEqual(result["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), self.damaged)
        self.assertEqual((Path(result["output"]) / "backup" / "000000.dcm").read_bytes(), self.damaged)
        self.assertFalse(list(self.target.parent.glob(".dicom_recovery_*")))

    def test_changed_target_before_replacement_is_not_overwritten(self):
        copy = shutil.copy2

        def concurrent_copy(source, destination):
            result = copy(source, destination)
            if Path(source) == self.original:
                self.target.write_bytes(b"CONCURRENT_UPDATE")
            return result

        with mock.patch("foundation_sft.recover_data.shutil.copy2", side_effect=concurrent_copy):
            result = self.recover(apply=True)
        self.assertEqual(result["counts"]["unresolved"], 1)
        self.assertEqual(self.target.read_bytes(), b"CONCURRENT_UPDATE")
        self.assertEqual((Path(result["output"]) / "backup" / "000000.dcm").read_bytes(), self.damaged)
        self.assertFalse(list(self.target.parent.glob(".dicom_recovery_*")))

    def test_audit_private_list_must_match_technical_failures(self):
        report = json.loads((self.audit / "report.json").read_text())
        report["failures"][0]["row"] = 99
        (self.audit / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.recover(apply=True)
        self.assertEqual(self.target.read_bytes(), self.damaged)


if __name__ == "__main__":
    unittest.main()
