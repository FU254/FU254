"""Released CXformer processor compatibility without executing vendor code."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from foundation_sft.data import CXRDataset, _dicom_pixels, _equalize_histogram_uint8, _resize_image
from foundation_sft.models.common import PreprocessSpec
from foundation_sft.models.cxformer import _cxformer_processor_spec, build_cxformer


# Actual saved metadata supplied from m42-health/CXformer-base by the user.
PROCESSOR = {
    "auto_map": {"AutoImageProcessor": "custom_processor.CustomBitImageProcessor"},
    "crop_size": {"height": 518, "width": 518},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_hist_equalization": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "CustomBitImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"shortest_edge": 518},
}

try:
    import cv2
    from transformers import BitImageProcessor
except ImportError:
    cv2 = BitImageProcessor = None


class CXformerPreprocessTests(unittest.TestCase):
    def test_actual_metadata_preserves_equalization_and_saved_fingerprint(self):
        spec = _cxformer_processor_spec(PROCESSOR)
        self.assertEqual(spec.image_size, 518)
        self.assertEqual(spec.resize_shortest_edge, 518)
        self.assertEqual(spec.resize_mode, "shortest_edge_center_crop")
        self.assertEqual(spec.interpolation, "bicubic")
        self.assertTrue(spec.do_hist_equalization)
        self.assertTrue(asdict(spec)["do_hist_equalization"])
        self.assertFalse(PreprocessSpec(518).do_hist_equalization)

    def test_unknown_or_incomplete_custom_processor_cannot_silently_skip_rule(self):
        bad = [
            {**PROCESSOR, "auto_map": {"AutoImageProcessor": "external.UnreviewedProcessor"}},
            {**PROCESSOR, "image_processor_type": "OtherImageProcessor"},
            {**PROCESSOR, "do_hist_equalization": "true"},
            {key: value for key, value in PROCESSOR.items() if key != "do_hist_equalization"},
            {key: value for key, value in PROCESSOR.items() if key != "image_mean"},
        ]
        for metadata in bad:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    _cxformer_processor_spec(metadata)

    def test_equalization_preserves_constants_and_maps_nonuniform_gray_or_rgb(self):
        for value in (0, 7, 127, 255):
            source = np.full((8, 11), value, np.uint8)
            np.testing.assert_array_equal(_equalize_histogram_uint8(source), source)
        # First-bin mass is removed. Remaining cumulative fractions are
        # 1/4, 3/4 and 1, which OpenCV rounds to 64, 191 and 255.
        source = np.array([[10, 10, 20], [40, 40, 90]], dtype=np.uint8)
        expected = np.array([[0, 0, 64], [191, 191, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(_equalize_histogram_uint8(source), expected)
        rgb = np.stack([source, np.full_like(source, 61), source[:, ::-1]], axis=-1)
        expected_rgb = np.stack([expected, np.full_like(source, 61), expected[:, ::-1]], axis=-1)
        np.testing.assert_array_equal(_equalize_histogram_uint8(rgb), expected_rgb)

    @unittest.skipIf(cv2 is None, "OpenCV is an optional reference dependency")
    def test_numpy_equalization_matches_opencv_on_varied_histograms(self):
        generator = np.random.default_rng(781)
        for shape in ((8, 13), (37, 59), (512, 721)):
            samples = [
                generator.integers(0, 256, size=shape, dtype=np.uint8),
                np.clip(generator.gamma(1.8, 19.0, size=shape), 0, 255).astype(np.uint8),
                generator.choice(np.array([3, 14, 19, 70, 180], dtype=np.uint8), size=shape,
                                 p=[0.61, 0.17, 0.13, 0.06, 0.03]),
            ]
            for source in samples:
                np.testing.assert_array_equal(_equalize_histogram_uint8(source), cv2.equalizeHist(source))

    @unittest.skipIf(BitImageProcessor is None, "Transformers/OpenCV reference not installed")
    def test_dataset_matches_reviewed_cv2_then_bit_pipeline(self):
        generator = np.random.default_rng(16)
        processor = BitImageProcessor(**{
            key: value for key, value in PROCESSOR.items()
            if key not in {"auto_map", "image_processor_type", "do_hist_equalization"}
        })
        spec = _cxformer_processor_spec(PROCESSOR)
        for shape in ((59, 113), (113, 59), (67, 67)):
            with self.subTest(shape=shape):
                dicom = SimpleNamespace(
                    pixel_array=np.clip(generator.gamma(1.4, 240, size=shape), 0, 4095).astype(np.uint16),
                    PhotometricInterpretation="MONOCHROME2",
                )
                cohort = pd.DataFrame([{
                    "sample_id": "synthetic", "patient_id": "synthetic", "path": "unused", "label": 0,
                }])
                with patch("foundation_sft.data.pydicom.dcmread", return_value=dicom):
                    actual = CXRDataset(cohort, spec)[0]["image"]
                uint8 = (np.clip(_dicom_pixels(dicom), 0, 1) * 255).astype(np.uint8)
                rgb = np.repeat(uint8[..., None], 3, axis=-1)
                equalized = np.stack([cv2.equalizeHist(rgb[..., channel]) for channel in range(3)], axis=-1)
                expected = processor(Image.fromarray(equalized), return_tensors="pt")["pixel_values"][0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
                # The unequalized float pathway is measurably different and
                # cannot be substituted just because image size/stats match.
                uncorrected = _resize_image(_dicom_pixels(dicom), PreprocessSpec(
                    518, resize_mode="shortest_edge_center_crop", resize_shortest_edge=518,
                ))
                self.assertGreater(np.abs(uncorrected - _resize_image(_dicom_pixels(dicom), spec)).mean(), 0.01)

    def test_other_models_keep_float_resize_without_uint8_quantization(self):
        pixels = np.array([[0.011, 0.135], [0.468, 0.983]], dtype=np.float32)
        expected = np.clip(np.asarray(Image.fromarray(pixels).resize(
            (5, 5), resample=Image.Resampling.BICUBIC,
        ), dtype=np.float32), 0, 1)
        np.testing.assert_array_equal(_resize_image(pixels, PreprocessSpec(5)), expected)

    @unittest.skipIf(BitImageProcessor is None, "Transformers reference not installed")
    def test_loader_uses_local_metadata_and_never_imports_custom_processor(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    model_type="dinov2_with_registers", hidden_size=768,
                    num_register_tokens=4, num_channels=3, patch_size=14,
                )

        architecture = {
            "model_type": "dinov2_with_registers", "patch_size": 14,
            "num_register_tokens": 4, "num_hidden_layers": 12,
            "hidden_size": 768, "num_attention_heads": 12,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(architecture))
            (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))
            (root / "custom_processor.py").write_text("raise RuntimeError('Vendor code was executed')")
            with patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(**architecture)) as config_loader, \
                 patch("transformers.AutoImageProcessor.from_pretrained", side_effect=AssertionError("AutoImageProcessor is forbidden")), \
                 patch("transformers.Dinov2WithRegistersModel.from_pretrained", return_value=(FakeEncoder(), {})) as encoder_loader:
                adapter = build_cxformer({"model_dir": directory})
            self.assertTrue(adapter.preprocess.do_hist_equalization)
            self.assertEqual(adapter.provenance["preprocessing_implementation"], "cxformer_uint8_hist_equalization_bit_v1")
            self.assertTrue(config_loader.call_args.kwargs["local_files_only"])
            self.assertFalse(encoder_loader.call_args.kwargs["trust_remote_code"])
            self.assertTrue(encoder_loader.call_args.kwargs["weights_only"])
            self.assertFalse((root / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
