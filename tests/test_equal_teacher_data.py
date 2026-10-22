"""One augmented field, preserved control views, explicit teacher geometry."""

import copy
import json
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from foundation_sft.distillation_data import SharedViewDataset
from foundation_sft.equal_teacher_data import MultiTeacherViewDataset
from foundation_sft.equal_teacher_losses import TEACHER_NAMES


class AugmentedBase:
    def __init__(self, random_augmentation=False):
        self.calls = []
        self.random_augmentation = random_augmentation
        self.mean = torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1)
        self.std = torch.tensor([0.2, 0.3, 0.4]).view(3, 1, 1)
        rows = torch.linspace(0.0, 0.7, 32).view(1, 32, 1)
        columns = torch.linspace(0.0, 0.3, 32).view(1, 1, 32)
        self.gray = rows + columns
        self.preprocess = SimpleNamespace(
            image_size=32, mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4),
            resize_mode="shortest_edge_center_crop", resize_shortest_edge=32,
            interpolation="bicubic", do_hist_equalization=False,
        )
        self.last_image = None
        self.label = torch.tensor(1.0)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.calls.append(index)
        gray = self.gray
        if self.random_augmentation:
            if torch.rand(()) < 0.5:
                gray = gray.flip(-1)
            gray = gray.roll(int(torch.randint(0, 3, ()).item()), dims=-2)
        self.last_image = (gray.repeat(3, 1, 1) - self.mean) / self.std
        return {"image": self.last_image, "label": self.label, "sample_id": "synthetic",
                "patient_id": "synthetic-patient", "extra": "preserved"}


def teacher_specs(base):
    return {
        "rad_dino": copy.deepcopy(vars(base.preprocess)),
        "chexfound": {
            "image_size": 24, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "resize_mode": "shortest_edge_center_crop", "resize_shortest_edge": 24,
            "interpolation": "bicubic", "do_hist_equalization": False,
        },
        "chexzero": {
            "image_size": 16, "mean": [101.48761/255] * 3, "std": [83.43944/255] * 3,
            "resize_mode": "chexzero_pad320", "resize_shortest_edge": None,
            "interpolation": "bicubic", "do_hist_equalization": False,
        },
    }


class EqualTeacherDataTests(unittest.TestCase):
    def dataset(self, base=None, specs=None):
        base = AugmentedBase() if base is None else base
        shared = SharedViewDataset(base, student_size=12, include_teacher=True)
        return MultiTeacherViewDataset(shared, teacher_specs(base) if specs is None else specs)

    def test_fetches_base_once_preserves_anchor_student_and_identity(self):
        base = AugmentedBase()
        dataset = self.dataset(base)
        result = dataset[1]
        self.assertEqual(len(dataset), 3)
        self.assertEqual(base.calls, [1])
        self.assertIs(result["teacher_image"], base.last_image)
        self.assertIs(result["teacher_images"]["rad_dino"], result["teacher_image"])
        self.assertIs(result["label"], base.label)
        self.assertEqual(result["extra"], "preserved")
        self.assertEqual(result["image"].shape, (3, 12, 12))
        self.assertEqual(tuple(result["teacher_images"]), TEACHER_NAMES)
        for name, size in (("rad_dino", 32), ("chexfound", 24), ("chexzero", 16)):
            self.assertEqual(result["teacher_images"][name].shape, (3, size, size))

    def test_reuses_exact_augmentation_with_no_extra_rng_draws_or_student_change(self):
        base = AugmentedBase(random_augmentation=True)
        dataset = self.dataset(base)
        torch.manual_seed(192)
        legacy = dataset.shared[0]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(192)
        actual = dataset[0]
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(actual["image"], legacy["image"], rtol=0, atol=0)
        torch.testing.assert_close(actual["teacher_image"], legacy["teacher_image"], rtol=0, atol=0)
        self.assertEqual(base.calls, [0, 0])

    def test_extra_teachers_use_the_same_grayscale_field_and_own_normalization(self):
        base = AugmentedBase()
        dataset = self.dataset(base)
        result = dataset[0]
        original = result["teacher_image"].clone()
        gray = result["teacher_image"][0:1] * base.std[0:1] + base.mean[0:1]
        for name in ("chexfound", "chexzero"):
            image = result["teacher_images"][name]
            spec = teacher_specs(base)[name]
            mean = torch.tensor(spec["mean"]).view(3, 1, 1)
            std = torch.tensor(spec["std"]).view(3, 1, 1)
            display = image * std + mean
            expected = F.interpolate(gray.unsqueeze(0), size=(spec["image_size"],) * 2,
                                     mode="bicubic", align_corners=False,
                                     antialias=True).squeeze(0).clamp(0, 1)
            torch.testing.assert_close(display, expected.repeat(3, 1, 1), atol=1e-7, rtol=1e-6)
            self.assertGreater(display[0, -1].mean().item(), display[0, 0].mean().item())
        torch.testing.assert_close(result["teacher_image"], original, rtol=0, atol=0)

    def test_metadata_preserves_legacy_contract_and_records_native_geometry_override(self):
        dataset = self.dataset()
        expected = dataset.shared.metadata()
        self.assertEqual(dataset.metadata(), expected)
        views = dataset.teacher_view_metadata()
        json.dumps(views, allow_nan=False)
        self.assertEqual(views["reference_preprocessing"], expected)
        self.assertTrue(views["shared_field_of_view"])
        chexzero = views["teacher_views"]["chexzero"]
        self.assertEqual(chexzero["native_preprocessing"]["resize_mode"], "chexzero_pad320")
        self.assertEqual(chexzero["effective_preprocessing"]["resize_mode"], "resize_shared_anchor_view")
        self.assertFalse(chexzero["native_geometry_applied"])
        self.assertTrue(views["teacher_views"]["rad_dino"]["native_geometry_applied"])
        self.assertEqual(views["teacher_views"]["rad_dino"]["effective_preprocessing"]["resize_mode"],
                         "preserve_anchor_tensor")
        before = copy.deepcopy(dataset.teacher_view_metadata())
        views["teacher_views"]["chexzero"]["native_preprocessing"]["mean"][0] = 99
        dataset.metadata()["student_preprocessing"]["mean"][0] = 99
        dataset[0]
        self.assertEqual(dataset.teacher_view_metadata(), before)
        self.assertEqual(dataset.metadata(), expected)

    def test_batch_collation_preserves_canonical_per_teacher_images(self):
        dataset = self.dataset()
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(batch["image"].shape, (2, 3, 12, 12))
        self.assertEqual(tuple(batch["teacher_images"]), TEACHER_NAMES)
        torch.testing.assert_close(batch["teacher_images"]["rad_dino"], batch["teacher_image"],
                                   rtol=0, atol=0)

    def test_rejects_missing_teacher_anchor_and_unsupported_preprocessing(self):
        base = AugmentedBase()
        shared = SharedViewDataset(base, student_size=12)
        specs = teacher_specs(base)
        for mapping in ({}, {"rad_dino": specs["rad_dino"]}, {**specs, "eva_x": specs["rad_dino"]}):
            with self.subTest(names=tuple(mapping)), self.assertRaises(ValueError):
                MultiTeacherViewDataset(shared, mapping)
        with self.assertRaises(ValueError):
            MultiTeacherViewDataset(shared, specs, anchor="chexfound")
        with self.assertRaises(ValueError):
            MultiTeacherViewDataset(SharedViewDataset(base, include_teacher=False), specs)
        for change in (
            {"image_size": 0}, {"image_size": True}, {"image_size": 224.0},
            {"std": [1, 0, 1]}, {"mean": [1, float("nan"), 1]}, {"mean": [1, 1]},
            {"resize_mode": "random_resized_crop"}, {"interpolation": "bilinear"},
            {"do_hist_equalization": True}, {"resize_shortest_edge": 0},
        ):
            bad = copy.deepcopy(specs)
            bad["chexzero"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                MultiTeacherViewDataset(shared, bad)
        bad = copy.deepcopy(specs)
        bad["rad_dino"]["mean"] = [0.1, 0.2, 0.3]
        with self.assertRaisesRegex(ValueError, "anchor profile"):
            MultiTeacherViewDataset(shared, bad)

    def test_anchor_size_mismatch_is_rejected(self):
        base = AugmentedBase()
        base.gray = torch.ones(1, 31, 32)
        with self.assertRaisesRegex(ValueError, "native-size"):
            self.dataset(base)[0]

    def test_base_failure_propagates_without_retry(self):
        class BrokenBase(AugmentedBase):
            def __getitem__(self, index):
                self.calls.append(index)
                raise RuntimeError("synthetic decode failure")

        base = BrokenBase()
        with self.assertRaisesRegex(RuntimeError, "synthetic decode failure"):
            self.dataset(base)[2]
        self.assertEqual(base.calls, [2])


if __name__ == "__main__":
    unittest.main()
