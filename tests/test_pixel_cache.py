"""Lossless deterministic image caching and source/augmentation integrity."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as torchvision_functional

from foundation_sft.data import CXRDataset, _dicom_pixels, _resize_image
from foundation_sft.models.common import PreprocessSpec
from foundation_sft.pixel_cache import PixelCache, PixelCacheError


def _chexzero_geometry_reference(display_pixels):
    """Independent PIL/torchvision rendition of the released image geometry.

    The input display pixels intentionally use this study's shared DICOM
    rendering. This verifies the released geometry, not original PNG parity.
    """
    source = Image.fromarray((np.clip(display_pixels, 0, 1) * 255).astype(np.uint8))
    width, height = source.size
    # The official release truncates floating-point products. Preserve its
    # 77-pixel edge case (319, not idealized 320) rather than integer division.
    # A one-pixel minimum is the shared defensive extension for extreme aspect
    # ratios where the original release's PIL resize would reject a zero side.
    release_ratio = float(320) / max(source.size)
    dimensions = tuple(max(1, int(edge * release_ratio)) for edge in source.size)
    resized = source.resize(dimensions, Image.Resampling.LANCZOS)
    left, top = (320 - dimensions[0]) // 2, (320 - dimensions[1]) // 2
    padded = ImageOps.expand(resized, border=(left, top, 320 - dimensions[0] - left,
                                            320 - dimensions[1] - top), fill=0)
    tensor = torchvision_functional.to_tensor(padded)
    output = torchvision_functional.resize(tensor, [224, 224],
                                           interpolation=InterpolationMode.BICUBIC, antialias=False)
    return output.clamp(0, 1).squeeze(0).numpy()


class PixelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="foundation-pixel-cache-")
        self.root = Path(self.temporary.name)
        self.path = self.root / "sensitive-patient-source.dcm"
        self.cache_dir = self.root / "cache"
        # Rewrites alter pixels, not identity/header length. This keeps
        # source-replacement tests focused on ctime/inode invalidation.
        self.instance_uid = generate_uid()
        self.write_dicom()

    def tearDown(self):
        self.temporary.cleanup()

    def write_dicom(self, reverse=False):
        metadata = FileMetaDataset()
        metadata.TransferSyntaxUID = ExplicitVRLittleEndian
        metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        metadata.MediaStorageSOPInstanceUID = self.instance_uid
        dicom = FileDataset(str(self.path), {}, file_meta=metadata, preamble=b"\0" * 128)
        dicom.SOPClassUID = metadata.MediaStorageSOPClassUID
        dicom.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
        dicom.PatientID = "sensitive-patient-identity"
        dicom.Rows, dicom.Columns = 13, 19
        dicom.SamplesPerPixel = 1
        dicom.PhotometricInterpretation = "MONOCHROME2"
        dicom.BitsAllocated = dicom.BitsStored = 16
        dicom.HighBit, dicom.PixelRepresentation = 15, 0
        pixels = (np.arange(247, dtype=np.uint16).reshape(13, 19) ** 2) % 4095
        if reverse:
            pixels = pixels[:, ::-1]
        dicom.RescaleSlope, dicom.RescaleIntercept = 1.7, -28
        dicom.WindowCenter, dicom.WindowWidth = 1450, 2900
        dicom.PixelData = pixels.tobytes()
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(self.path, dicom, **options)

    def decode(self, spec):
        return _resize_image(_dicom_pixels(pydicom.dcmread(self.path)), spec)

    def test_real_dicom_roundtrips_exactly_for_all_resize_and_equalization_paths(self):
        cache = PixelCache(self.cache_dir)
        for mode in ("resize", "letterbox", "shortest_edge_center_crop"):
            for equalize in (False, True):
                for interpolation in ("nearest", "bilinear", "bicubic"):
                    with self.subTest(mode=mode, equalize=equalize, interpolation=interpolation):
                        spec = PreprocessSpec(11, resize_mode=mode, resize_shortest_edge=15,
                                              interpolation=interpolation, do_hist_equalization=equalize)
                        direct = self.decode(spec)
                        first = cache.get(self.path, spec, lambda: self.decode(spec))
                        second = cache.get(self.path, spec, lambda: self.fail("Cache hit decoded DICOM"))
                        np.testing.assert_array_equal(direct, first)
                        np.testing.assert_array_equal(direct, second)
                        self.assertEqual(second.dtype, np.float32)

    def test_real_dicom_chexzero_cache_exactly_matches_independent_geometry_reference(self):
        spec = PreprocessSpec(224, resize_mode="chexzero_pad320")
        display = _dicom_pixels(pydicom.dcmread(self.path))
        reference = _chexzero_geometry_reference(display)
        direct = self.decode(spec)
        cache = PixelCache(self.cache_dir)
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            first = cache.get(self.path, spec, lambda: self.decode(spec))
            hit = cache.get(self.path, spec, lambda: self.fail("CheXzero cache hit decoded DICOM"))
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(direct.shape, (224, 224))
        self.assertEqual(direct.dtype, np.float32)
        self.assertTrue(np.isfinite(direct).all())
        self.assertGreaterEqual(float(direct.min()), 0)
        self.assertLessEqual(float(direct.max()), 1)
        np.testing.assert_array_equal(direct, reference)
        np.testing.assert_array_equal(first, direct)
        np.testing.assert_array_equal(hit, direct)

    def test_chexzero_geometry_matches_release_float_floor_and_odd_black_padding(self):
        spec = PreprocessSpec(224, resize_mode="chexzero_pad320")
        for shape in ((13, 19), (19, 13), (37, 77), (77, 37), (71, 71), (1, 513)):
            with self.subTest(shape=shape):
                display = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 83) / 82
                actual = _resize_image(display, spec)
                expected = _chexzero_geometry_reference(display)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (224, 224))
                self.assertEqual(actual.dtype, np.float32)
                self.assertGreaterEqual(float(actual.min()), 0)
                self.assertLessEqual(float(actual.max()), 1)
                if shape[1] > shape[0] * 1.5:
                    np.testing.assert_array_equal(actual[0], np.zeros(224, dtype=np.float32))

    def test_chexzero_and_ordinary_224_resize_do_not_share_cache_entries(self):
        cache = PixelCache(self.cache_dir)
        chexzero = PreprocessSpec(224, resize_mode="chexzero_pad320")
        ordinary = PreprocessSpec(224, resize_mode="resize")
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            chexzero_pixels = cache.get(self.path, chexzero, lambda: self.decode(chexzero))
            ordinary_pixels = cache.get(self.path, ordinary, lambda: self.decode(ordinary))
        self.assertEqual(decoder.call_count, 2)
        self.assertFalse(np.array_equal(chexzero_pixels, ordinary_pixels))
        for spec, pixels in ((chexzero, chexzero_pixels), (ordinary, ordinary_pixels)):
            np.testing.assert_array_equal(cache.get(self.path, spec, lambda: self.fail("Expected cached mode")), pixels)
        self.assertEqual(len(list(cache.directory.glob("*.npy"))), 2)

    def test_chexzero_rejects_non_224_non_bicubic_or_equalized_specs_before_cache_build(self):
        valid = PreprocessSpec(224, resize_mode="chexzero_pad320")
        cache = PixelCache(self.cache_dir)
        for spec in (replace(valid, image_size=223), replace(valid, image_size=320),
                     replace(valid, interpolation="bilinear"), replace(valid, interpolation="lanczos"),
                     replace(valid, do_hist_equalization=True)):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(PixelCacheError, "CheXzero preprocessing requires"):
                    cache.get(self.path, spec, lambda: self.fail("Invalid CheXzero spec built cache"))
                with self.assertRaisesRegex(ValueError, "CheXzero requires"):
                    self.decode(spec)
        self.assertEqual(list(cache.directory.glob("*.npy")), [])

    def test_one_decode_per_source_and_preprocess_across_instances_and_modes(self):
        spec = PreprocessSpec(11)
        probe_cache = PixelCache(self.cache_dir)
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            first = probe_cache.get(self.path, spec, lambda: self.decode(spec))
            lora_spec = replace(spec, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
            second = PixelCache(self.cache_dir).get(self.path, lora_spec, lambda: self.decode(lora_spec))
        self.assertEqual(decoder.call_count, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(list(probe_cache.directory.glob("*.npy"))), 1)

    def test_optional_irrelevant_shortest_edge_does_not_prevent_sharing(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        cache.get(self.path, spec, lambda: self.decode(spec))
        cache.get(self.path, replace(spec, resize_shortest_edge=45), lambda: self.fail("Unused edge made a new entry"))
        mapping = {"image_size": 11, "resize_mode": "resize", "interpolation": "bicubic"}
        cache.get(self.path, mapping, lambda: self.fail("Dict spec made a new entry"))

    def test_modified_source_invalidates_cached_pixels(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        original = cache.get(self.path, spec, lambda: self.decode(spec))
        self.write_dicom(reverse=True)
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            updated = cache.get(self.path, spec, lambda: self.decode(spec))
        self.assertEqual(decoder.call_count, 1)
        self.assertFalse(np.array_equal(original, updated))
        np.testing.assert_array_equal(updated, self.decode(spec))

    def test_restored_mtime_and_size_still_invalidates_using_ctime(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        before = self.path.stat()
        original = cache.get(self.path, spec, lambda: self.decode(spec))
        self.write_dicom(reverse=True)
        self.assertEqual(self.path.stat().st_size, before.st_size)
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        updated = cache.get(self.path, spec, lambda: self.decode(spec),
                            expected_size=before.st_size, expected_mtime_ns=before.st_mtime_ns)
        self.assertFalse(np.array_equal(original, updated))

    def test_source_replacement_changes_inode_and_invalidates(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        cache.get(self.path, spec, lambda: self.decode(spec))
        old_stat = self.path.stat()
        old_source = self.path.with_suffix(".old")
        self.path.rename(old_source)
        self.write_dicom(reverse=True)
        os.utime(self.path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.assertNotEqual(self.path.stat().st_ino, old_stat.st_ino)
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            actual = cache.get(self.path, spec, lambda: self.decode(spec),
                               expected_size=old_stat.st_size, expected_mtime_ns=old_stat.st_mtime_ns)
        self.assertEqual(decoder.call_count, 1)
        np.testing.assert_array_equal(actual, self.decode(spec))

    def test_missing_and_changed_frozen_sources_never_fall_back_to_cached_pixels(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        expected = self.path.stat()
        cache.get(self.path, spec, lambda: self.decode(spec))
        os.utime(self.path, ns=(expected.st_atime_ns, expected.st_mtime_ns + 1))
        with self.assertRaisesRegex(PixelCacheError, "changed since the frozen cohort") as caught:
            cache.get(self.path, spec, lambda: self.fail("Changed source decoded"),
                      expected_size=expected.st_size, expected_mtime_ns=expected.st_mtime_ns)
        self.assertNotIn(str(self.path), str(caught.exception))
        self.assertNotIn("sensitive-patient", str(caught.exception))
        self.path.unlink()
        with self.assertRaisesRegex(PixelCacheError, "source image is unavailable"):
            cache.get(self.path, spec, lambda: self.fail("Missing source decoded"))

    def test_source_changed_during_build_is_rejected(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)

        def unstable():
            pixels = self.decode(spec)
            self.write_dicom(reverse=True)
            return pixels

        with self.assertRaisesRegex(PixelCacheError, "changed while reading"):
            cache.get(self.path, spec, unstable)
        self.assertEqual(list(cache.directory.glob("*.npy")), [])

    def test_source_changed_during_cache_hit_is_rejected(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        cache.get(self.path, spec, lambda: self.decode(spec))
        original_read = cache._read

        def unstable(*args):
            image = original_read(*args)
            self.write_dicom(reverse=True)
            return image

        with patch.object(cache, "_read", side_effect=unstable):
            with self.assertRaisesRegex(PixelCacheError, "changed while reading"):
                cache.get(self.path, spec, lambda: self.fail("Hit should not decode"))

    def test_corrupt_array_or_metadata_is_rebuilt(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        direct = cache.get(self.path, spec, lambda: self.decode(spec))
        array_path = next(cache.directory.glob("*.npy"))
        metadata_path = next(cache.directory.glob("*.json"))
        for corrupt in (
                lambda: array_path.write_bytes(b"broken NPY payload"),
                lambda: metadata_path.write_text("{malformed"),
                lambda: metadata_path.write_text(json.dumps({"dtype": "float16"})),
                lambda: array_path.unlink()):
            corrupt()
            with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
                image = cache.get(self.path, spec, lambda: self.decode(spec))
            self.assertEqual(decoder.call_count, 1)
            np.testing.assert_array_equal(image, direct)

    def test_runtime_changes_invalidate_entries(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        cache.get(self.path, spec, lambda: self.decode(spec))
        with patch("foundation_sft.pixel_cache.runtime_signature", return_value={"decoder": "new-version"}), \
                patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            cache.get(self.path, spec, lambda: self.decode(spec))
        self.assertEqual(decoder.call_count, 1)

    def test_live_pydicom_decoder_configuration_changes_invalidate_same_instance(self):
        spec = PreprocessSpec(11)
        cache = PixelCache(self.cache_dir)
        cache.get(self.path, spec, lambda: self.decode(spec))
        current = pydicom.config.APPLY_J2K_CORRECTIONS
        with patch.object(pydicom.config, "APPLY_J2K_CORRECTIONS", not current), \
                patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            cache.get(self.path, spec, lambda: self.decode(spec))
        self.assertEqual(decoder.call_count, 1)

    def test_concurrent_workers_publish_complete_arrays_and_private_files(self):
        spec = PreprocessSpec(11)
        barrier = Barrier(4)

        def worker(_):
            cache = PixelCache(self.cache_dir)

            def build():
                image = self.decode(spec)
                barrier.wait(timeout=10)
                return image

            return cache.get(self.path, spec, build)

        with ThreadPoolExecutor(max_workers=4) as pool:
            images = list(pool.map(worker, range(4)))
        direct = self.decode(spec)
        for image in images:
            np.testing.assert_array_equal(image, direct)
        cache = PixelCache(self.cache_dir)
        np.testing.assert_array_equal(cache.get(self.path, spec, lambda: self.fail("Publication incomplete")), direct)
        self.assertEqual(cache.directory.stat().st_mode & 0o777, 0o700)
        for file in cache.directory.iterdir():
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("sensitive-patient", file.name)
        self.assertEqual(len(list(cache.directory.glob("*.npy"))), 1)

    def test_invalid_build_pixels_are_not_published(self):
        spec = PreprocessSpec(11)
        for pixels in (np.zeros((11, 11), np.float16), np.zeros((11, 11, 3), np.float32),
                       np.full((11, 11), np.nan, np.float32), np.full((11, 11), 2, np.float32)):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                cache = PixelCache(self.cache_dir)
                with self.assertRaisesRegex(PixelCacheError, "invalid pixels"):
                    cache.get(self.path, spec, lambda: pixels)
        self.assertEqual(list(cache.directory.glob("*.npy")), [])

    def test_dataset_random_geometry_and_normalization_stay_after_pixel_cache(self):
        stat = self.path.stat()
        cohort = pd.DataFrame([{
            "path": str(self.path), "label": 1, "sample_id": "synthetic-image",
            "patient_id": "synthetic-patient", "dicom_size": stat.st_size,
            "dicom_mtime_ns": stat.st_mtime_ns,
        }])
        spec = PreprocessSpec(11, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
        plain = CXRDataset(cohort, spec)
        cached = CXRDataset(cohort, spec, cache_dir=self.cache_dir)
        torch.testing.assert_close(plain[0]["image"], cached[0]["image"], rtol=0, atol=0)
        augmented = CXRDataset(cohort, spec, training=True, rotation_degrees=15,
                               horizontal_flip_probability=0.5, cache_dir=self.cache_dir)
        direct_augmented = CXRDataset(cohort, spec, training=True, rotation_degrees=15,
                                      horizontal_flip_probability=0.5)
        with patch("pydicom.dcmread", wraps=pydicom.dcmread) as decoder:
            torch.manual_seed(8)
            first = augmented[0]["image"]
            torch.manual_seed(91)
            second = augmented[0]["image"]
        self.assertEqual(decoder.call_count, 0)
        self.assertFalse(torch.equal(first, second))
        torch.manual_seed(8)
        torch.testing.assert_close(first, direct_augmented[0]["image"], rtol=0, atol=0)
        torch.manual_seed(91)
        torch.testing.assert_close(second, direct_augmented[0]["image"], rtol=0, atol=0)

    def test_chexzero_cache_reuses_only_pixels_and_keeps_normalization_and_augmentation_fresh(self):
        stat = self.path.stat()
        cohort = pd.DataFrame([{
            "path": str(self.path), "label": 1, "sample_id": "synthetic-image",
            "patient_id": "synthetic-patient", "dicom_size": stat.st_size,
            "dicom_mtime_ns": stat.st_mtime_ns,
        }])
        spec = PreprocessSpec(224, mean=(101.48761 / 255,) * 3, std=(83.43944 / 255,) * 3,
                              resize_mode="chexzero_pad320")
        plain = CXRDataset(cohort, spec)
        cached = CXRDataset(cohort, spec, cache_dir=self.cache_dir)
        direct = plain[0]["image"]
        torch.testing.assert_close(cached[0]["image"], direct, rtol=0, atol=0)
        alternate = replace(spec, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
        with patch("pydicom.dcmread", side_effect=AssertionError("Normalization change decoded DICOM")):
            alternate_image = CXRDataset(cohort, alternate, cache_dir=self.cache_dir)[0]["image"]
        shared_pixels = torch.from_numpy(self.decode(spec)).expand(3, -1, -1)
        expected = ((shared_pixels - torch.tensor(alternate.mean).view(3, 1, 1))
                    / torch.tensor(alternate.std).view(3, 1, 1))
        torch.testing.assert_close(alternate_image, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(alternate_image, direct))
        augmented = CXRDataset(cohort, spec, training=True, rotation_degrees=15,
                               horizontal_flip_probability=0.5, cache_dir=self.cache_dir)
        direct_augmented = CXRDataset(cohort, spec, training=True, rotation_degrees=15,
                                      horizontal_flip_probability=0.5)
        with patch("pydicom.dcmread", side_effect=AssertionError("CheXzero augmentation decoded DICOM")):
            torch.manual_seed(8)
            first = augmented[0]["image"]
            torch.manual_seed(91)
            second = augmented[0]["image"]
        self.assertFalse(torch.equal(first, second))
        torch.manual_seed(8)
        torch.testing.assert_close(first, direct_augmented[0]["image"], rtol=0, atol=0)
        torch.manual_seed(91)
        torch.testing.assert_close(second, direct_augmented[0]["image"], rtol=0, atol=0)
        self.assertEqual(len(list((self.cache_dir / "pixels").glob("*.npy"))), 1)


if __name__ == "__main__":
    unittest.main()
