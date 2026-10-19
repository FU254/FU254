"""Shared teacher/student geometry, metadata, and existing cache contracts."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch

from foundation_sft.data import CXRDataset
from foundation_sft.distillation_data import SharedViewDataset
from foundation_sft.models.common import PreprocessSpec


class FakeBase:
    def __init__(self):
        self.calls = []
        self.mean = torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1)
        self.std = torch.tensor([0.2, 0.3, 0.4]).view(3, 1, 1)
        # Increasing rows reveal an accidental vertical flip.
        self.gray = torch.linspace(0.1, 0.9, 8).view(1, 8, 1).expand(1, 8, 12).clone()
        self.image = (self.gray.repeat(3, 1, 1) - self.mean) / self.std
        self.label = torch.tensor(1.0)
        self.preprocess = SimpleNamespace(
            image_size=12, mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4),
            resize_mode="shortest_edge_center_crop", interpolation="bicubic",
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.calls.append(index)
        return {"image": self.image, "label": self.label, "sample_id": "sample",
                "patient_id": "patient", "extra": "preserved"}


class SharedViewDatasetTests(unittest.TestCase):
    def test_calls_base_once_preserves_teacher_tensor_and_identity_objects(self):
        base = FakeBase()
        original = base.image.clone()
        dataset = SharedViewDataset(base)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(base.calls, [])
        result = dataset[2]
        self.assertEqual(base.calls, [2])
        self.assertIs(result["teacher_image"], base.image)
        torch.testing.assert_close(base.image, original, rtol=0, atol=0)
        self.assertIs(result["label"], base.label)
        self.assertEqual(result["sample_id"], "sample")
        self.assertEqual(result["patient_id"], "patient")
        self.assertEqual(result["extra"], "preserved")
        self.assertEqual(result["image"].shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(result["image"]).all())

    def test_supervised_and_distillation_views_identical_without_extra_rng_draws(self):
        base = FakeBase()
        before = torch.get_rng_state().clone()
        kd = SharedViewDataset(base, include_teacher=True)[0]
        supervised = SharedViewDataset(base, include_teacher=False)[0]
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(kd["image"], supervised["image"], rtol=0, atol=0)
        self.assertNotIn("teacher_image", supervised)
        self.assertEqual(base.calls, [0, 0])

    def test_student_geometry_preserves_vertical_orientation_and_grayscale_channels(self):
        result = SharedViewDataset(FakeBase(), student_size=32)[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display = result["image"] * std + mean
        self.assertLess(display[:, 0].mean().item(), display[:, -1].mean().item())
        torch.testing.assert_close(display[0], display[1], rtol=0, atol=1e-7)
        torch.testing.assert_close(display[0], display[2], rtol=0, atol=1e-7)
        self.assertGreaterEqual(display.min().item(), -1e-7)
        self.assertLessEqual(display.max().item(), 1 + 1e-7)

    def test_inverse_normalization_uses_teacher_channel_zero_and_student_profile(self):
        base = FakeBase()
        base.gray.fill_(0.25)
        base.image = (base.gray.repeat(3, 1, 1) - base.mean) / base.std
        result = SharedViewDataset(base, student_size=16, student_mean=(0.1, 0.2, 0.3),
                                   student_std=(0.5, 0.25, 1.0))[0]
        expected = torch.tensor([0.3, 0.2, -0.05]).view(3, 1, 1).expand(3, 16, 16)
        torch.testing.assert_close(result["image"], expected, rtol=1e-5, atol=2e-7)

    def test_metadata_records_second_resize_existing_cache_and_is_independent(self):
        dataset = SharedViewDataset(FakeBase())
        metadata = dataset.metadata()
        json.dumps(metadata, allow_nan=False)
        self.assertTrue(metadata["shared_field_of_view"])
        self.assertEqual(metadata["teacher_preprocessing"]["resize_mode"], "shortest_edge_center_crop")
        self.assertEqual(metadata["student_preprocessing"]["resize_mode"], "resize_shared_teacher_view")
        self.assertTrue(metadata["student_preprocessing"]["antialias"])
        self.assertIn("base dataset PixelCache", metadata["cache_policy"])
        metadata["student_preprocessing"]["mean"][0] = 100
        self.assertNotEqual(dataset.metadata()["student_preprocessing"]["mean"][0], 100)

    def test_metadata_preserves_every_dataclass_mapping_and_namespace_field(self):
        @dataclass(frozen=True)
        class ExtendedSpec(PreprocessSpec):
            future_setting: str = "preserve-me"

        dataclass_spec = ExtendedSpec(image_size=12)
        spec_dict = asdict(dataclass_spec)
        spec_dict["future_nested"] = {"other_setting": [1, 2, 3]}
        for spec in (dataclass_spec, spec_dict, SimpleNamespace(**spec_dict)):
            base = FakeBase()
            base.preprocess = spec
            dataset = SharedViewDataset(base)
            metadata = dataset.metadata()["teacher_preprocessing"]
            self.assertEqual(metadata["future_setting"], "preserve-me")
            if spec is not dataclass_spec:
                self.assertEqual(metadata["future_nested"], {"other_setting": [1, 2, 3]})
            self.assertEqual(metadata["do_hist_equalization"], False)
        spec_dict["future_nested"]["other_setting"].append(4)
        self.assertEqual(metadata["future_nested"], {"other_setting": [1, 2, 3]})

    def test_invalid_sizes_profiles_teacher_normalization_and_metadata_are_rejected(self):
        for size in (0, -1, True, 224.0, "224"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                SharedViewDataset(FakeBase(), student_size=size)
        for kwargs in (
                {"student_std": (1, 0, 1)}, {"student_std": (1, float("nan"), 1)},
                {"student_mean": (0, 0)}, {"include_teacher": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SharedViewDataset(FakeBase(), **kwargs)
        for attribute, value in (("mean", torch.zeros(3)), ("std", torch.zeros(3, 1, 1)),
                                 ("preprocess", {"not_finite": float("inf")})):
            base = FakeBase()
            setattr(base, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                SharedViewDataset(base)

    def test_malformed_or_nonfinite_base_images_are_rejected(self):
        for image in (None, torch.zeros(3, 8, 8, dtype=torch.int64), torch.zeros(1, 8, 8),
                      torch.zeros(3, 0, 8), torch.full((3, 8, 8), float("nan")),
                      torch.full((3, 8, 8), float("inf"))):
            base = FakeBase()
            base.image = image
            with self.subTest(image_type=type(image).__name__), self.assertRaises(ValueError):
                SharedViewDataset(base)[0]
            self.assertEqual(base.calls, [0])

    def test_base_cache_or_decode_failure_is_not_retried(self):
        class BrokenBase(FakeBase):
            def __getitem__(self, index):
                self.calls.append(index)
                raise RuntimeError("synthetic base decode failure")

        base = BrokenBase()
        with self.assertRaisesRegex(RuntimeError, "synthetic base decode failure"):
            SharedViewDataset(base)[1]
        self.assertEqual(base.calls, [1])

    def test_existing_real_dicom_cache_and_augmentation_are_reused_exactly(self):
        with tempfile.TemporaryDirectory(prefix="shared-view-cache-") as temporary:
            root = Path(temporary)
            path = root / "synthetic.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            dicom = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            dicom.SOPClassUID = meta.MediaStorageSOPClassUID
            dicom.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            dicom.Rows, dicom.Columns = 13, 19
            dicom.SamplesPerPixel = 1
            dicom.PhotometricInterpretation = "MONOCHROME1"
            dicom.BitsAllocated = dicom.BitsStored = 16
            dicom.HighBit, dicom.PixelRepresentation = 15, 0
            dicom.RescaleSlope, dicom.RescaleIntercept = 1.7, -28
            dicom.WindowCenter, dicom.WindowWidth = 1450, 2900
            dicom.PixelData = ((np.arange(247, dtype=np.uint16) ** 2) % 4095).tobytes()
            options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                       else {"write_like_original": False})
            pydicom.dcmwrite(path, dicom, **options)
            stat = path.stat()
            cohort = pd.DataFrame([{
                "path": str(path), "label": 1, "sample_id": "image", "patient_id": "patient",
                "dicom_size": stat.st_size, "dicom_mtime_ns": stat.st_mtime_ns,
            }])
            spec = PreprocessSpec(32, resize_mode="shortest_edge_center_crop", resize_shortest_edge=37)
            base = CXRDataset(cohort, spec, training=True, rotation_degrees=15,
                              horizontal_flip_probability=0.5, cache_dir=root / "cache")
            wrapper = SharedViewDataset(base, student_size=16)
            torch.manual_seed(91)
            direct_teacher = base[0]["image"]
            direct_rng_state = torch.get_rng_state().clone()
            # A hit must avoid decoding while retaining the original base's
            # rotation/flip draw count and exact normalized teacher tensor.
            with patch("foundation_sft.data.pydicom.dcmread", side_effect=AssertionError("decoded cache hit")):
                torch.manual_seed(91)
                wrapped = wrapper[0]
                torch.testing.assert_close(wrapped["teacher_image"], direct_teacher, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), direct_rng_state, rtol=0, atol=0)
                torch.manual_seed(91)
                supervised = SharedViewDataset(base, student_size=16, include_teacher=False)[0]
            torch.testing.assert_close(wrapped["image"], supervised["image"], rtol=0, atol=0)
            self.assertEqual(len(list((root / "cache" / "pixels").glob("*.npy"))), 1)
            self.assertEqual(wrapped["image"].shape, (3, 16, 16))
            self.assertEqual(wrapper.metadata()["teacher_preprocessing"], json.loads(json.dumps(asdict(spec))))


if __name__ == "__main__":
    unittest.main()
