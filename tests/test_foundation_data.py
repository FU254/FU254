"""Synthetic Excel/DICOM tests for cohort integrity and patient isolation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch

from foundation_sft.data import build_cohort, CXRDataset
from foundation_sft.split import load_or_create_split


class FoundationDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="foundation-data-test-")
        self.root = Path(self.temporary.name)
        self.dicom_dir = self.root / "dicoms"
        self.dicom_dir.mkdir()
        self.excel = self.root / "labels.xlsx"
        self.config = {
            "excel_path": str(self.excel),
            "dicom_dir": str(self.dicom_dir),
            "accession_column": "放射编号",
            "patient_column": "PatientID",
            "label_column": "阻塞性通气障碍",
            "label_definition": "synthetic pre-existing binary labels; LLN not recomputed",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def workbook(self, rows):
        pd.DataFrame(rows).to_excel(self.excel, index=False)

    def dicom(self, filename, accession, patient="synthetic-patient", photometric="MONOCHROME2"):
        path = self.dicom_dir / filename
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.AccessionNumber = accession
        dataset.PatientName = "Synthetic^Only"
        if patient is not None:
            dataset.PatientID = patient
        dataset.Rows = 8
        dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = photometric
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = 2
        dataset.RescaleIntercept = -7
        dataset.PixelData = np.arange(64, dtype=np.uint16).reshape(8, 8).tobytes()
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dataset, **options)
        return path

    def test_unmatched_dicom_is_skipped_without_reusing_previous_label(self):
        self.workbook([
            {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
            {"放射编号": "B", "PatientID": "P1", "阻塞性通气障碍": 1},
        ])
        self.dicom("00-matched.dcm", "A")
        self.dicom("01-unmatched.dcm", "UNKNOWN")
        self.dicom("02-matched.dcm", "B")
        cohort = build_cohort(self.config)
        self.assertEqual(dict(zip(cohort.accession, cohort.label)), {"A": 0, "B": 1})
        self.assertEqual(len(cohort), 2)
        self.assertEqual(cohort.attrs["cohort_diagnostics"]["unmatched_dicoms"], 1)
        self.assertTrue(cohort.sample_id.is_unique)
        self.assertIn("LLN", cohort.attrs["label_definition"])

    def test_auto_patient_column_uses_workbook_id_then_dicom_id(self):
        self.config["patient_column"] = "auto"
        self.workbook([{"放射编号": "A", "住院号": "EXCEL-ID", "阻塞性通气障碍": 0}])
        self.dicom("a.dcm", "A", patient="DICOM-ID")
        cohort = build_cohort(self.config)
        self.assertEqual(cohort.patient_id.iloc[0], "EXCEL-ID")
        self.assertEqual(cohort.attrs["cohort_diagnostics"]["selected_patient_column"], "住院号")
        self.workbook([{"放射编号": "A", "阻塞性通气障碍": 0}])
        cohort = build_cohort(self.config)
        self.assertEqual(cohort.patient_id.iloc[0], "DICOM-ID")
        self.assertEqual(cohort.patient_id_source.iloc[0], "dicom:PatientID")

    def test_fractional_labels_and_conflicting_duplicates_are_rejected(self):
        for rows in (
            [{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0.7}],
            [{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
             {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 1}],
            [{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
             {"放射编号": "A", "PatientID": "P1", "阻塞性通气障碍": 0}],
        ):
            with self.subTest(rows=len(rows)):
                self.workbook(rows)
                with self.assertRaises(ValueError):
                    build_cohort(self.config)

    def test_reliable_patient_ids_are_required_and_origins_cannot_mix(self):
        self.workbook([
            {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
            {"放射编号": "B", "PatientID": None, "阻塞性通气障碍": 1},
        ])
        self.dicom("a.dcm", "A", patient="D0")
        second = self.dicom("b.dcm", "B", patient="D1")
        with self.assertRaisesRegex(ValueError, "Mixed patient identifier origins"):
            build_cohort(self.config)
        cohort = build_cohort({**self.config, "patient_id_source": "dicom"})
        self.assertEqual(set(cohort.patient_id), {"D0", "D1"})
        self.assertEqual(set(cohort.patient_id_source), {"dicom:PatientID"})
        second.unlink()
        self.dicom("b.dcm", "B", patient=None)
        with self.assertRaisesRegex(ValueError, "Reliable patient IDs are missing"):
            build_cohort(self.config)

    def test_integer_excel_identifiers_preserve_leading_zeros(self):
        self.workbook([{"放射编号": "001.0", "PatientID": "0007.0", "阻塞性通气障碍": 1}])
        self.dicom("a.dcm", "001")
        cohort = build_cohort({"data": self.config})
        self.assertEqual(cohort.iloc[0].accession, "001")
        self.assertEqual(cohort.iloc[0].patient_id, "0007")

    def test_patient_names_require_explicit_opt_in(self):
        self.workbook([{"放射编号": "A", "阻塞性通气障碍": 0}])
        self.dicom("a.dcm", "A", patient=None)
        with self.assertRaisesRegex(ValueError, "Reliable patient IDs are missing"):
            build_cohort(self.config)
        cohort = build_cohort({**self.config, "allow_name_patient_id": True})
        self.assertEqual(set(cohort.patient_id_source), {"dicom_name:PatientName"})

    def test_conflicting_labels_within_a_patient_require_paired_exam_protocol(self):
        self.workbook([
            {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
            {"放射编号": "B", "PatientID": "P0", "阻塞性通气障碍": 1},
        ])
        self.dicom("a.dcm", "A")
        self.dicom("b.dcm", "B")
        with self.assertRaisesRegex(ValueError, "one paired CXR/PFT"):
            build_cohort(self.config)

    def test_dataset_decodes_monochrome_and_returns_normalized_three_channels(self):
        self.workbook([
            {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
            {"放射编号": "B", "PatientID": "P1", "阻塞性通气障碍": 1},
        ])
        self.dicom("a.dcm", "A", photometric="MONOCHROME2")
        self.dicom("b.dcm", "B", photometric="MONOCHROME1")
        cohort = build_cohort(self.config).sort_values("accession")
        spec = SimpleNamespace(image_size=8, mean=(0.0,) * 3, std=(1.0,) * 3, resize_mode="resize")
        dataset = CXRDataset(cohort, spec)
        first, second = dataset[0], dataset[1]
        self.assertEqual(first["image"].shape, (3, 8, 8))
        self.assertEqual(first["label"].dtype, torch.float32)
        self.assertEqual(first["label"].item(), 0)
        self.assertEqual(first["sample_id"], cohort.iloc[0].sample_id)
        torch.testing.assert_close(first["image"][0], first["image"][1])
        torch.testing.assert_close(second["image"], 1 - first["image"], rtol=1e-5, atol=1e-6)
        pixels = np.arange(64, dtype=np.float32).reshape(8, 8) * 2 - 7
        lower, upper = np.percentile(pixels, [0.5, 99.5])
        expected = torch.from_numpy(((np.clip(pixels, lower, upper) - lower) / (upper - lower)).astype(np.float32))
        torch.testing.assert_close(first["image"][0], expected)

        spec.mean, spec.std = (0.5,) * 3, (0.25,) * 3
        normalized = CXRDataset(cohort, spec)[0]["image"]
        torch.testing.assert_close(normalized, (first["image"] - 0.5) / 0.25)

    def test_monochrome1_inverts_after_asymmetric_modality_window(self):
        self.workbook([
            {"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0},
            {"放射编号": "B", "PatientID": "P1", "阻塞性通气障碍": 1},
        ])
        for filename, accession, photometric in (
                ("a.dcm", "A", "MONOCHROME2"), ("b.dcm", "B", "MONOCHROME1")):
            path = self.dicom(filename, accession, photometric=photometric)
            dicom = pydicom.dcmread(path)
            dicom.WindowCenter, dicom.WindowWidth = 20, 30
            pydicom.dcmwrite(path, dicom)
        cohort = build_cohort(self.config).sort_values("accession")
        spec = {"image_size": 8, "mean": (0,) * 3, "std": (1,) * 3}
        dataset = CXRDataset(cohort, spec)
        torch.testing.assert_close(dataset[1]["image"], 1 - dataset[0]["image"], rtol=0, atol=1e-7)
        # The original modality window remains [5,35] after slope/intercept.
        self.assertEqual(dataset[0]["image"][0, 0, 0].item(), 0)
        self.assertEqual(dataset[1]["image"][0, 0, 0].item(), 1)

    def test_only_opt_in_horizontal_flip_changes_training_geometry(self):
        self.workbook([{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0}])
        self.dicom("a.dcm", "A")
        cohort = build_cohort(self.config)
        spec = {"image_size": 8, "mean": (0,) * 3, "std": (1,) * 3, "resize_mode": "resize"}
        original = CXRDataset(cohort, spec)[0]["image"]
        training = CXRDataset(cohort, spec, training=True, horizontal_flip_probability=1)[0]["image"]
        evaluation = CXRDataset(cohort, spec, training=False, horizontal_flip_probability=1)[0]["image"]
        torch.testing.assert_close(training, original.flip(-1))
        torch.testing.assert_close(evaluation, original)

    def test_rescaling_and_multivalued_window_are_applied_before_normalization(self):
        self.workbook([{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0}])
        path = self.dicom("a.dcm", "A")
        dicom = pydicom.dcmread(path)
        dicom.WindowCenter, dicom.WindowWidth = [20, 40], [20, 80]
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dicom, **options)
        cohort = build_cohort(self.config)
        spec = {"image_size": 8, "mean": (0,) * 3, "std": (1,) * 3}
        actual = CXRDataset(cohort, spec)[0]["image"][0]
        pixels = np.clip(np.arange(64, dtype=np.float32).reshape(8, 8) * 2 - 7, 10, 30)
        lower, upper = np.percentile(pixels, [0.5, 99.5])
        expected = torch.from_numpy(((pixels - lower) / (upper - lower)).astype(np.float32))
        torch.testing.assert_close(actual, expected)

    def test_rotation_is_training_only_and_reproducible_with_torch_seed(self):
        self.workbook([{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0}])
        self.dicom("a.dcm", "A")
        cohort = build_cohort(self.config)
        spec = {"image_size": 16, "mean": (0,) * 3, "std": (1,) * 3}
        original = CXRDataset(cohort, spec)[0]["image"]
        augmented = CXRDataset(cohort, spec, training=True, rotation_degrees=30)
        torch.manual_seed(10)
        first = augmented[0]["image"]
        torch.manual_seed(10)
        repeated = augmented[0]["image"]
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.allclose(first, original))
        evaluation = CXRDataset(cohort, spec, training=False, rotation_degrees=30)[0]["image"]
        torch.testing.assert_close(evaluation, original)

    def test_nonfinite_float_dicom_pixels_fail_without_patient_values_in_error(self):
        self.workbook([{"放射编号": "A", "PatientID": "P0", "阻塞性通气障碍": 0}])
        path = self.dicom("a.dcm", "A")
        dicom = pydicom.dcmread(path)
        del dicom.PixelData
        del dicom.BitsStored
        del dicom.HighBit
        del dicom.PixelRepresentation
        dicom.BitsAllocated = 32
        dicom.FloatPixelData = np.full((8, 8), np.nan, dtype=np.float32).tobytes()
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dicom, **options)
        cohort = build_cohort(self.config)
        spec = {"image_size": 8, "mean": (0,) * 3, "std": (1,) * 3}
        with self.assertRaisesRegex(RuntimeError, "no finite pixels") as failure:
            CXRDataset(cohort, spec)[0]
        self.assertNotIn("P0", str(failure.exception))
        self.assertNotIn(str(path), str(failure.exception))
        pixels = np.ones((8, 8), dtype=np.float32)
        pixels[0, 0] = np.inf
        dicom.FloatPixelData = pixels.tobytes()
        pydicom.dcmwrite(path, dicom, **options)
        cohort = build_cohort(self.config)
        with self.assertRaisesRegex(RuntimeError, "contains nonfinite pixels"):
            CXRDataset(cohort, spec)[0]

    @staticmethod
    def synthetic_cohort():
        rows = []
        for patient in range(20):
            for image in range(2):
                sample = f"sample-{patient:02d}-{image}"
                rows.append({
                    "sample_id": sample,
                    "patient_id": f"patient-{patient:02d}",
                    "accession": sample,
                    "path": f"/synthetic/{sample}.dcm",
                    "label": patient % 2,
                    "dicom_size": 1234,
                    "dicom_mtime_ns": 1000000000000000000,
                    "patient_id_source": "excel:PatientID",
                })
        cohort = pd.DataFrame(rows)
        cohort.attrs["label_source_signature"] = "synthetic-source-v1"
        cohort.attrs["label_definition"] = "synthetic strict binary label"
        return cohort

    def test_patient_split_is_fixed_and_independent_of_input_order(self):
        cohort = self.synthetic_cohort()
        manifest = self.root / "split.csv"
        result = load_or_create_split(cohort, manifest)
        original_bytes = manifest.read_bytes()
        self.assertTrue(result.groupby("patient_id").split.nunique().eq(1).all())
        patient_counts = result.groupby("split").patient_id.nunique().to_dict()
        self.assertEqual(patient_counts, {"train": 14, "val": 2, "test": 4})
        for _, partition in result.groupby("split"):
            self.assertEqual(set(partition.label), {0, 1})
        shuffled = cohort.sample(frac=1, random_state=1)
        shuffled.attrs.update(cohort.attrs)
        reloaded = load_or_create_split(shuffled, manifest)
        self.assertEqual(result.set_index("sample_id").split.to_dict(), reloaded.set_index("sample_id").split.to_dict())
        self.assertEqual(manifest.read_bytes(), original_bytes)

    def test_real_excel_dicom_prepare_split_and_dataset_end_to_end(self):
        rows = []
        for patient in range(20):
            for image in range(2):
                accession = f"A{patient:02d}-{image}"
                rows.append({"放射编号": accession, "PatientID": f"P{patient:02d}",
                             "阻塞性通气障碍": patient % 2})
                self.dicom(f"{accession}.dcm", accession)
        self.workbook(rows)
        cohort = build_cohort(self.config)
        prepared = load_or_create_split(cohort, self.root / "prepared.csv")
        self.assertEqual(len(prepared), 40)
        self.assertTrue(prepared.groupby("patient_id").split.nunique().eq(1).all())
        self.assertEqual(prepared.attrs["label_source_signature"], cohort.attrs["label_source_signature"])
        spec = SimpleNamespace(image_size=12, mean=(0.5,) * 3, std=(0.5,) * 3,
                               resize_mode="shortest_edge_center_crop", resize_shortest_edge=14)
        dataset = CXRDataset(prepared.loc[prepared.split.eq("val")], spec)
        self.assertEqual(dataset[0]["image"].shape, (3, 12, 12))
        self.assertTrue(torch.isfinite(dataset[0]["image"]).all())
        self.assertEqual(set(prepared.loc[prepared.split.eq("val"), "label"]), {0, 1})

    def test_split_rejects_changed_labels_patients_source_or_file_metadata(self):
        cohort = self.synthetic_cohort()
        manifest = self.root / "split.csv"
        load_or_create_split(cohort, manifest)
        for column, replacement in (("label", 1), ("patient_id", "changed-patient"),
                                    ("dicom_mtime_ns", 1000000000000000001)):
            with self.subTest(column=column):
                changed = cohort.copy()
                changed.loc[0, column] = replacement
                with self.assertRaisesRegex(ValueError, "Frozen split manifest"):
                    load_or_create_split(changed, manifest)
        changed = cohort.copy()
        changed.attrs["label_source_signature"] = "synthetic-source-v2"
        with self.assertRaisesRegex(ValueError, "Frozen split manifest"):
            load_or_create_split(changed, manifest)
        changed = cohort.copy()
        changed.attrs["preprocess_version"] = "a different pixel preprocessing pipeline"
        with self.assertRaisesRegex(ValueError, "Frozen split manifest"):
            load_or_create_split(changed, manifest)
        with self.assertRaises(ValueError):
            load_or_create_split(cohort, manifest, seed=43)

    def test_manually_modified_patient_partition_is_rejected(self):
        cohort = self.synthetic_cohort()
        manifest = self.root / "split.csv"
        load_or_create_split(cohort, manifest)
        saved = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        saved.loc[0, "split"] = "val" if saved.loc[0, "split"] != "val" else "train"
        saved.to_csv(manifest, index=False)
        with self.assertRaises(ValueError):
            load_or_create_split(cohort, manifest)


if __name__ == "__main__":
    unittest.main()
