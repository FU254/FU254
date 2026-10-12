"""Full-cohort pixel audits fail early without revealing medical identities."""

import contextlib
import copy
import csv
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    ExplicitVRLittleEndian, JPEG2000Lossless, SecondaryCaptureImageStorage, generate_uid,
)

from foundation_sft.audit_data import audit_split
from foundation_sft.data import _dicom_pixels


class DicomAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dicom-audit-")
        self.root = Path(self.temporary.name)
        self.directory = self.root / "PRIVATE_DATA_DIRECTORY"
        self.directory.mkdir()
        self.output = self.root / "audit"
        self.rows = []

    def tearDown(self):
        self.temporary.cleanup()

    def add_dicom(self, partition, *, truncated=False, compressed=False, index=None):
        number = len(self.rows)
        path = self.directory / f"PRIVATE_FILENAME_{number}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = JPEG2000Lossless if compressed else ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.PatientName = "Private^Name"
        dataset.PatientID = "PRIVATE_DICOM_ID"
        dataset.AccessionNumber = "PRIVATE_ACCESSION"
        dataset.Rows = 8
        dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = 2
        dataset.RescaleIntercept = -7
        dataset.WindowCenter = 20
        dataset.WindowWidth = 80
        if compressed:
            dataset.PixelData = encapsulate([b"\x00\x01\x02\x03"])
            dataset["PixelData"].is_undefined_length = True
        else:
            pixels = np.arange(64, dtype=np.uint16)
            dataset.PixelData = pixels[:2].tobytes() if truncated else pixels.tobytes()
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dataset, **options)
        self.rows.append({
            "split": partition, "path": str(path), "patient_id": "PRIVATE_COHORT_ID",
            "sample_id": f"PRIVATE_SAMPLE_{number}", "accession": "PRIVATE_COHORT_ACCESSION",
            "label": number % 2,
        })
        return path

    def frame(self):
        result = pd.DataFrame(self.rows)
        result.index = np.arange(len(result)) * 7 + 100
        result.attrs["cohort_hash"] = "frozen-test-cohort"
        result.attrs["nested"] = {"unaltered": True}
        return result

    def assert_safe(self, report, stdout="", stderr=""):
        technical = json.dumps(report) + (self.output / "report.json").read_text() + stdout + stderr
        for secret in (
            str(self.directory), "PRIVATE_FILENAME", "Private^Name", "PRIVATE_DICOM_ID",
            "PRIVATE_ACCESSION", "PRIVATE_COHORT_ID", "PRIVATE_SAMPLE", "PRIVATE_COHORT_ACCESSION",
        ):
            self.assertNotIn(secret, technical)
        for failure in report["failures"]:
            self.assertNotIn("path", failure)
            self.assertNotIn("patient_id", failure)
            self.assertNotIn("sample_id", failure)

    def test_valid_images_all_partitions_use_real_shared_pixel_preprocessing(self):
        for partition in ("test", "train", "val", "train"):
            self.add_dicom(partition)
        split = self.frame()
        original = copy.deepcopy(split)
        manifest = self.root / "frozen_manifest.csv"
        split.to_csv(manifest, index=False)
        frozen_bytes = manifest.read_bytes()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch(
                "foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            report = audit_split(split, self.output)
        self.assertEqual(decode.call_count, 4)
        self.assertTrue(report["audit_passed"])
        self.assertEqual(report["checked_images"], 4)
        self.assertEqual(report["passed_images"], 4)
        self.assertEqual(report["failed_images"], 0)
        self.assertEqual(report["split_counts"]["train"], {"checked": 2, "passed": 2, "failed": 0})
        self.assertEqual(report["transfer_syntax_counts"][str(ExplicitVRLittleEndian)]["images"], 4)
        self.assertEqual(json.loads((self.output / "report.json").read_text()), report)
        pd.testing.assert_frame_equal(split, original)
        self.assertEqual(split.attrs, original.attrs)
        self.assertEqual(manifest.read_bytes(), frozen_bytes)
        self.assertIn("Checked 4/4; failed=0", stdout.getvalue())
        self.assertEqual((self.output / "failures_private.csv").stat().st_mode & 0o777, 0o600)
        with (self.output / "failures_private.csv").open(newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        self.assert_safe(report, stdout.getvalue())

    def test_truncated_train_val_test_pixels_are_all_reported_without_changing_split(self):
        self.add_dicom("test", truncated=True)
        self.add_dicom("train")
        self.add_dicom("val", truncated=True)
        self.add_dicom("train", truncated=True)
        split = self.frame()
        original = copy.deepcopy(split)
        report = audit_split(split, self.output)
        self.assertFalse(report["audit_passed"])
        self.assertEqual(report["checked_images"], 4)
        self.assertEqual(report["failed_images"], 3)
        self.assertEqual([(item["split"], item["row"]) for item in report["failures"]],
                         [("train", 1), ("val", 0), ("test", 0)])
        self.assertTrue(all(item["error_category"] == "invalid_pixel_data" for item in report["failures"]))
        pixel_metadata = report["failures"][0]["pixel_metadata"]
        self.assertEqual(pixel_metadata["Rows"], 8)
        self.assertEqual(pixel_metadata["Columns"], 8)
        self.assertEqual(pixel_metadata["BitsAllocated"], 16)
        self.assertEqual(pixel_metadata["PixelData_bytes"], 4)
        self.assertEqual(pixel_metadata["expected_uncompressed_bytes"], 128)
        pd.testing.assert_frame_equal(split, original)
        self.assertEqual(split.attrs, original.attrs)
        with (self.output / "failures_private.csv").open(newline="") as stream:
            private_rows = list(csv.DictReader(stream))
        self.assertEqual(len(private_rows), 3)
        self.assertEqual(private_rows[0]["path"], self.rows[3]["path"])
        private_text = (self.output / "failures_private.csv").read_text()
        self.assertNotIn("Private^Name", private_text)
        self.assertNotIn("PRIVATE_DICOM_ID", private_text)
        self.assert_safe(report)

    def test_missing_decoder_reports_only_allowlisted_packages_and_suppresses_phi_logs(self):
        private_path = self.add_dicom("train", compressed=True)
        self.add_dicom("val")
        self.add_dicom("test")
        stdout, stderr = io.StringIO(), io.StringIO()
        previous_disable = logging.root.manager.disable

        def decode(dataset):
            if dataset.file_meta.TransferSyntaxUID == JPEG2000Lossless:
                message = (
                    "Unable to decompress because all plugins are missing dependencies: "
                    "pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-openjpeg>=2.0. "
                    f"{private_path} {dataset.PatientName} {dataset.PatientID} "
                    f"{dataset.AccessionNumber} PRIVATE_COHORT_ID PRIVATE_SAMPLE"
                )
                warnings.warn(message)
                logging.getLogger("pydicom.pixels.decoders.base").error(message)
                raise RuntimeError(message)
            return _dicom_pixels(dataset)

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), mock.patch(
                "foundation_sft.audit_data._dicom_pixels", side_effect=decode):
            report = audit_split(self.frame(), self.output)
        self.assertEqual(logging.root.manager.disable, previous_disable)
        self.assertEqual(report["checked_images"], 3)
        self.assertEqual(report["failed_images"], 1)
        failure = report["failures"][0]
        self.assertEqual(failure["transfer_syntax_uid"], str(JPEG2000Lossless))
        self.assertEqual(failure["transfer_syntax_name"], "JPEG 2000 Image Compression (Lossless Only)")
        self.assertEqual(failure["error_category"], "missing_pixel_decoder")
        self.assertIn("pylibjpeg>=2.0", failure["decoder_dependencies"])
        self.assertIn("pylibjpeg-openjpeg>=2.0", failure["decoder_dependencies"])
        self.assert_safe(report, stdout.getvalue(), stderr.getvalue())

    def test_physically_truncated_uncompressed_file_reports_declared_and_actual_bytes(self):
        path = self.add_dicom("train")
        path.write_bytes(path.read_bytes()[:-64])
        report = audit_split(self.frame(), self.output)
        failure = report["failures"][0]
        self.assertEqual(failure["error_category"], "truncated_pixel_data")
        self.assertEqual(failure["pixel_metadata"]["PixelData_declared_bytes"], 128)
        self.assertEqual(failure["pixel_metadata"]["PixelData_bytes"], 64)
        self.assertFalse(failure["pixel_metadata"]["PixelData_encapsulated"])
        self.assertIn("Restore the complete original DICOM", failure["error_detail"])
        self.assert_safe(report)

    def test_missing_declared_padding_is_rejected_even_if_decoder_succeeds(self):
        path = self.add_dicom("train")
        dataset = pydicom.dcmread(path)
        dataset.Rows = dataset.Columns = 3
        dataset.BitsAllocated = dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelData = np.arange(9, dtype=np.uint8).tobytes() + b"\0"
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dataset, **options)
        path.write_bytes(path.read_bytes()[:-1])
        self.assertEqual(pydicom.dcmread(path).pixel_array.shape, (3, 3))
        report = audit_split(self.frame(), self.output)
        self.assertFalse(report["audit_passed"])
        self.assertEqual(report["failures"][0]["error_category"], "truncated_pixel_data")
        self.assertEqual(report["failures"][0]["pixel_metadata"]["PixelData_declared_bytes"], 10)
        self.assertEqual(report["failures"][0]["pixel_metadata"]["PixelData_bytes"], 9)

    def test_arbitrary_error_text_and_exception_class_names_are_not_returned(self):
        self.add_dicom("train")
        secret_type = type("PRIVATE_DICOM_ID", (RuntimeError,), {})
        with mock.patch("foundation_sft.audit_data._dicom_pixels", side_effect=secret_type(
                "PRIVATE_UNKNOWN_PATIENT_FIELD /home/PRIVATE_FILENAME")):
            report = audit_split(self.frame(), self.output)
        failure = report["failures"][0]
        self.assertEqual(failure["error_type"], "DecoderError")
        self.assertEqual(failure["error_category"], "pixel_decode_failed")
        self.assertNotIn("PRIVATE_UNKNOWN_PATIENT_FIELD", json.dumps(report))
        self.assert_safe(report)

    def test_unreadable_file_failure_is_sanitized_and_later_images_are_checked(self):
        path = self.add_dicom("train")
        path.unlink()
        self.add_dicom("test")
        report = audit_split(self.frame(), self.output)
        self.assertEqual(report["checked_images"], 2)
        self.assertEqual(report["failed_images"], 1)
        self.assertEqual(report["failures"][0]["error_type"], "FileNotFoundError")
        self.assertEqual(report["failures"][0]["error_category"], "dicom_read_failed")
        self.assertIsNone(report["failures"][0]["transfer_syntax_uid"])
        self.assert_safe(report)

    def test_nonfinite_preprocessing_metadata_is_detected_using_shared_contract(self):
        path = self.add_dicom("train")
        dataset = pydicom.dcmread(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset.RescaleSlope = float("nan")
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dataset, **options)
        report = audit_split(self.frame(), self.output)
        failure = report["failures"][0]
        self.assertEqual(failure["error_category"], "pixel_preprocessing_failed")
        self.assertEqual(failure["error_detail"], "DICOM rescaling or window metadata is nonfinite.")
        self.assert_safe(report)

    def test_existing_evidence_is_not_overwritten(self):
        self.add_dicom("train")
        frame = self.frame()
        audit_split(frame, self.output)
        report_bytes = (self.output / "report.json").read_bytes()
        private_bytes = (self.output / "failures_private.csv").read_bytes()
        with mock.patch("foundation_sft.audit_data.pydicom.dcmread") as read:
            with self.assertRaisesRegex(FileExistsError, "fresh audit directory"):
                audit_split(frame, self.output)
        read.assert_not_called()
        self.assertEqual((self.output / "report.json").read_bytes(), report_bytes)
        self.assertEqual((self.output / "failures_private.csv").read_bytes(), private_bytes)

    def test_empty_or_unknown_split_is_rejected_without_reading_images(self):
        for frame in (
                pd.DataFrame(columns=["split", "path"]),
                pd.DataFrame({"split": ["PRIVATE_DICOM_ID"], "path": ["PRIVATE_FILENAME"]}),
                pd.DataFrame({"path": ["PRIVATE_FILENAME"]})):
            with self.subTest(columns=list(frame.columns)), mock.patch(
                    "foundation_sft.audit_data.pydicom.dcmread") as read:
                with self.assertRaises(ValueError) as caught:
                    audit_split(frame, self.output)
                self.assertNotIn("PRIVATE", str(caught.exception))
                read.assert_not_called()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
