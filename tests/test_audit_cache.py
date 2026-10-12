"""Reusable full pixel audits stay bound to files, cohort and decoder runtime."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from foundation_sft import audit_cache
from foundation_sft.audit_data import audit_split
from foundation_sft.cache import load_json, save_json
from foundation_sft.data import _dicom_pixels


class AuditCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="audit-cache-")
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.sources = self.root / "PRIVATE_SOURCE_DIRECTORY"
        self.sources.mkdir()
        rows = []
        for index, partition in enumerate(("train", "val", "test")):
            path = self.sources / f"PRIVATE_SAMPLE_{index}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            dataset.is_little_endian = True
            dataset.is_implicit_VR = False
            dataset.SOPClassUID = meta.MediaStorageSOPClassUID
            dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            dataset.PatientName = "PRIVATE_PATIENT_NAME"
            dataset.PatientID = f"PRIVATE_PATIENT_{index}"
            dataset.Rows = dataset.Columns = 8
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelData = np.arange(64, dtype=np.uint16).tobytes()
            options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                       else {"write_like_original": False})
            pydicom.dcmwrite(path, dataset, **options)
            rows.append({"sample_id": f"PRIVATE_SAMPLE_{index}",
                         "patient_id": f"PRIVATE_PATIENT_{index}",
                         "path": str(path), "split": partition, "label": index % 2,
                         "label_source_signature": "original-workbook-digest"})
        self.split = pd.DataFrame(rows)
        self.split.attrs["split_metadata"] = {"_seed": 42, "_ratios": [0.7, 0.1, 0.2]}
        self.split.attrs["cohort_hash"] = "caller-supplied-hash-is-not-trusted-alone"
        self.output_number = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_audit(self, split=None, **options):
        self.output_number += 1
        output = self.root / f"audit-{self.output_number}"
        report = audit_split(self.split if split is None else split, output,
                             cache_dir=self.cache, **options)
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        return report, output

    def entry(self):
        entries = list((self.cache / "audits").glob("*.json"))
        self.assertEqual(len(entries), 1)
        return entries[0]

    def test_success_hit_does_not_read_or_decode_images_and_writes_fresh_reports(self):
        before = self.split.copy(deep=True)
        first, first_output = self.run_audit()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), mock.patch(
                "foundation_sft.audit_data.pydicom.dcmread", side_effect=AssertionError("unexpected image read")), mock.patch(
                "foundation_sft.audit_data._dicom_pixels", side_effect=AssertionError("unexpected pixel decode")):
            second, second_output = self.run_audit()
        self.assertFalse(first["audit_cached"])
        self.assertTrue(second["audit_cached"])
        self.assertTrue(second["audit_passed"])
        self.assertEqual(second["checked_images"], 3)
        self.assertNotEqual(first_output, second_output)
        pd.testing.assert_frame_equal(before, self.split)
        self.assertIn("Reused successful pixel audit for 3", stdout.getvalue())
        for output in (first_output, second_output):
            self.assertEqual((output / "failures_private.csv").stat().st_mode & 0o777, 0o600)
            with (output / "failures_private.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
        self.assertEqual((self.cache / "audits").stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.entry().stat().st_mode & 0o777, 0o600)
        safe_text = stdout.getvalue() + json.dumps(second) + self.entry().read_text()
        for secret in ("PRIVATE_SOURCE_DIRECTORY", "PRIVATE_SAMPLE", "PRIVATE_PATIENT"):
            self.assertNotIn(secret, safe_text)

    def test_force_repeats_full_pixel_checks(self):
        self.run_audit()
        with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            report, _ = self.run_audit(force=True)
        self.assertEqual(decode.call_count, 3)
        self.assertFalse(report["audit_cached"])

    def test_cohort_cache_hit_status_does_not_invalidate_scientific_inputs(self):
        self.split.attrs["cohort_cache_hit"] = False
        self.run_audit()
        self.split.attrs["cohort_cache_hit"] = True
        with mock.patch("foundation_sft.audit_data._dicom_pixels", side_effect=AssertionError("unexpected decode")):
            report, _ = self.run_audit()
        self.assertTrue(report["audit_cached"])

    def test_same_size_content_change_with_restored_mtime_invalidates(self):
        self.run_audit()
        path = Path(self.split.iloc[0]["path"])
        previous = path.stat()
        data = path.read_bytes()
        path.write_bytes(data[:-2] + b"\x01\x00")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertEqual(path.stat().st_size, previous.st_size)
        self.assertEqual(path.stat().st_mtime_ns, previous.st_mtime_ns)
        with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            report, _ = self.run_audit()
        self.assertFalse(report["audit_cached"])
        self.assertEqual(decode.call_count, 3)

    def test_frozen_patient_label_partition_provenance_and_attrs_all_invalidate(self):
        self.run_audit()
        variants = []
        for column, value in (("patient_id", "changed-patient"), ("sample_id", "changed-sample"),
                              ("label", 1), ("split", "test"),
                              ("label_source_signature", "new-workbook-digest")):
            changed = self.split.copy(deep=True)
            changed.loc[0, column] = value
            variants.append(changed)
        changed = self.split.copy(deep=True)
        changed.attrs["new_label_definition"] = "different-definition"
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed.columns.tolist()), mock.patch(
                    "foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
                report, _ = self.run_audit(changed)
            self.assertFalse(report["audit_cached"])
            self.assertEqual(decode.call_count, 3)

    def test_decoder_environment_signature_change_invalidates(self):
        with mock.patch("foundation_sft.audit_cache.runtime_signature", return_value={"decoder": "first-env"}):
            self.run_audit()
        with mock.patch("foundation_sft.audit_cache.runtime_signature", return_value={"decoder": "second-env"}), mock.patch(
                "foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            report, _ = self.run_audit()
        self.assertFalse(report["audit_cached"])
        self.assertEqual(decode.call_count, 3)

    def test_corrupt_cache_and_legacy_unversioned_reports_repeat_audit(self):
        self.run_audit()
        entry = self.entry()
        original = load_json(entry)
        for payload in ("corrupt-json", {"report": original["report"]},
                        {**original, "version": "legacy-full-pixel-audit"}):
            if isinstance(payload, str):
                entry.write_text(payload)
            else:
                save_json(entry, payload)
            with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
                report, _ = self.run_audit()
            self.assertFalse(report["audit_cached"])
            self.assertEqual(decode.call_count, 3)

    def test_cache_count_mismatch_and_failed_report_are_not_reused(self):
        self.run_audit()
        entry = self.entry()
        for field, value in (("checked_images", 2), ("passed_images", 2),
                             ("failed_images", 1), ("audit_passed", False),
                             ("failures", [{"error_category": "invalid_pixel_data"}])):
            payload = load_json(entry)
            payload["report"][field] = value
            save_json(entry, payload)
            with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
                report, _ = self.run_audit()
            self.assertFalse(report["audit_cached"])
            self.assertEqual(decode.call_count, 3)

    def test_failed_audit_is_never_saved_or_reused(self):
        path = Path(self.split.iloc[0]["path"])
        path.write_bytes(path.read_bytes()[:-64])
        with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            for _ in range(2):
                report, _ = self.run_audit()
                self.assertFalse(report["audit_passed"])
                self.assertFalse(report["audit_cached"])
                self.assertEqual(report["failed_images"], 1)
        self.assertEqual(decode.call_count, 4)
        self.assertEqual(list((self.cache / "audits").glob("*.json")), [])

    def test_missing_file_after_success_does_not_reuse_success(self):
        self.run_audit()
        Path(self.split.iloc[0]["path"]).unlink()
        report, _ = self.run_audit()
        self.assertFalse(report["audit_passed"])
        self.assertFalse(report["audit_cached"])
        self.assertEqual(report["checked_images"], 3)
        self.assertEqual(report["failed_images"], 1)
        self.assertEqual(report["failures"][0]["error_category"], "dicom_read_failed")

    def test_symlink_loop_reports_sanitized_failure_instead_of_snapshot_exception(self):
        path = Path(self.split.iloc[0]["path"])
        path.unlink()
        path.symlink_to(path.name)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            report, _ = self.run_audit()
        self.assertFalse(report["audit_passed"])
        self.assertEqual(report["failed_images"], 1)
        self.assertEqual(report["failures"][0]["error_category"], "dicom_read_failed")
        self.assertNotIn("PRIVATE_SAMPLE", stdout.getvalue() + json.dumps(report))

    def test_existing_output_evidence_is_preserved_even_on_cache_hit(self):
        _, output = self.run_audit()
        original = (output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError), mock.patch(
                "foundation_sft.audit_data._dicom_pixels", side_effect=AssertionError("unexpected decode")):
            audit_split(self.split, output, cache_dir=self.cache)
        self.assertEqual((output / "report.json").read_bytes(), original)

    def test_file_mutation_during_scan_blocks_successful_cache_publication(self):
        changed = False

        def mutate_after_decode(dataset):
            nonlocal changed
            result = _dicom_pixels(dataset)
            if not changed:
                path = Path(self.split.iloc[0]["path"])
                path.write_bytes(path.read_bytes() + b"\0\0")
                changed = True
            return result

        with mock.patch("foundation_sft.audit_data._dicom_pixels", side_effect=mutate_after_decode), self.assertRaisesRegex(
                RuntimeError, "inputs changed"):
            self.run_audit()
        self.assertEqual(list((self.cache / "audits").glob("*.json")), [])

    def test_source_mutation_during_cache_read_blocks_hit_before_report_write(self):
        self.run_audit()

        def mutate_after_cache_read(entry):
            payload = load_json(entry)
            path = Path(self.split.iloc[0]["path"])
            path.write_bytes(path.read_bytes() + b"\0\0")
            return payload

        with mock.patch("foundation_sft.audit_cache.load_json", side_effect=mutate_after_cache_read), self.assertRaisesRegex(
                RuntimeError, "inputs changed"):
            self.run_audit()
        self.assertFalse((self.root / "audit-2" / "report.json").exists())

    def test_uncached_default_always_decodes_all_images(self):
        with mock.patch("foundation_sft.audit_data._dicom_pixels", wraps=_dicom_pixels) as decode:
            first = audit_split(self.split, self.root / "uncached-1")
            second = audit_split(self.split, self.root / "uncached-2")
        self.assertEqual(decode.call_count, 6)
        self.assertFalse(first["audit_cached"])
        self.assertFalse(second["audit_cached"])


if __name__ == "__main__":
    unittest.main()
