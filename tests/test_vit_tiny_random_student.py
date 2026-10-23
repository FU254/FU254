"""Native random students remain explicit, reproducible and source independent."""

from copy import deepcopy
import importlib.util
from importlib.metadata import version
import io
import unittest
from unittest import mock

import torch

from foundation_sft.models import vit_tiny_student as student


HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    HAS_TIMM = tuple(int(part) for part in version("timm").split(".")[:2]) >= (0, 9)


@unittest.skipUnless(HAS_TIMM, "Native random student requires timm>=0.9")
class RandomViTTinyStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0, msg=name)

    def test_random_factory_reads_no_checkpoint_download_or_cuda_rng(self):
        with mock.patch.object(student, "sha256_file", side_effect=AssertionError("No source file")), mock.patch.object(
            torch, "load", side_effect=AssertionError("No source deserialization")
        ), mock.patch.object(
            torch.hub, "download_url_to_file", side_effect=AssertionError("No download")
        ), mock.patch.object(
            torch.cuda, "get_rng_state", side_effect=AssertionError("Do not read CUDA RNG")
        ), mock.patch.object(
            torch.cuda, "set_rng_state", side_effect=AssertionError("Do not change CUDA RNG")
        ):
            model = student.build_vit_tiny_student(random_seed=42)
        metadata = model.training_metadata()
        self.assertEqual(metadata["initialization"], {
            "architecture": student.ARCHITECTURE,
            "initialization_mode": "random", "pretraining": "none", "seed": 42,
            "source_num_classes": 1, "variant": "single_cls",
        })
        self.assertEqual(metadata["total_parameters"], 5524609)
        self.assertEqual(metadata["trainable_parameters"], 5524609)
        self.assertGreater(model.encoder.head.weight.abs().sum().item(), 0)
        self.assertEqual(tuple(model.encoder.head.weight.shape), (1, 192))

    def test_same_seed_repeats_all_weights_and_different_seed_changes_encoder_and_head(self):
        first = student.build_vit_tiny_student(random_seed=42)
        torch.manual_seed(913)
        second = student.build_vit_tiny_student(random_seed=42)
        self.assert_state_equal(first.state_dict(), second.state_dict())
        different = student.build_vit_tiny_student(random_seed=43)
        self.assertFalse(torch.equal(first.encoder.patch_embed.proj.weight,
                                     different.encoder.patch_embed.proj.weight))
        self.assertFalse(torch.equal(first.encoder.head.weight, different.encoder.head.weight))
        self.assertNotEqual(first.training_metadata()["initialization"],
                            different.training_metadata()["initialization"])

    def test_factory_preserves_callers_cpu_rng_in_full_and_lora_modes(self):
        for mode in ("full_finetune", "lora"):
            with self.subTest(mode=mode):
                torch.manual_seed(700)
                before = torch.get_rng_state().clone()
                first = student.build_vit_tiny_student(random_seed=0, mode=mode)
                torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
                torch.rand(17)
                second_before = torch.get_rng_state().clone()
                second = student.build_vit_tiny_student(random_seed=0, mode=mode)
                torch.testing.assert_close(torch.get_rng_state(), second_before, rtol=0, atol=0)
                self.assert_state_equal(first.state_dict(), second.state_dict())

    def test_random_full_student_has_native_features_and_updates_encoder_and_head(self):
        model = student.build_vit_tiny_student(random_seed=42).train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        images = torch.randn(2, 3, 224, 224)
        features = model(images, return_features=True)
        self.assertEqual(tuple(features["global_features"].shape), (2, 192))
        self.assertEqual(tuple(features["local_features"].shape), (2, 196, 192))
        self.assertEqual(features["grid_shape"], (14, 14))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(features["logits"], torch.tensor([0., 1.]))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.encoder.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.head.weight.grad.abs().sum().item(), 0)

    def test_random_full_and_lora_students_save_and_restore_without_original_sources(self):
        images = torch.randn(1, 3, 224, 224)
        for mode in ("full_finetune", "lora"):
            with self.subTest(mode=mode):
                original = student.build_vit_tiny_student(random_seed=42, mode=mode).eval()
                with torch.no_grad():
                    original.encoder.head.weight.add_(0.02)
                    expected = original(images)
                stream = io.BytesIO()
                torch.save({"model": original.trainable_state_dict(),
                            "metadata": original.training_metadata()}, stream)
                stream.seek(0)
                saved = torch.load(stream, weights_only=True)
                initialization = saved["metadata"]["initialization"]
                with mock.patch.object(student, "sha256_file", side_effect=AssertionError("No ImageNet source")):
                    restored = student.build_vit_tiny_student(initialization=initialization, mode=mode).eval()
                    with self.assertRaisesRegex(RuntimeError, "complete student state"):
                        restored(images)
                    restored.load_trainable_state_dict(saved["model"])
                self.assertEqual(restored.training_metadata(), original.training_metadata())
                self.assert_state_equal(restored.state_dict(), original.state_dict())
                with torch.no_grad():
                    torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_random_lora_freezes_random_base_and_only_updates_lora_and_head(self):
        model = student.build_vit_tiny_student(random_seed=42, mode="lora")
        metadata = model.training_metadata()
        self.assertEqual(metadata["trainable_parameters"], 73921)
        self.assertEqual(len(model.lora["targets"]), 12)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) or name in {
            "encoder.head.weight", "encoder.head.bias"
        } for name, parameter in model.named_parameters() if parameter.requires_grad))
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                  if not value.requires_grad}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        loss = model(torch.randn(2, 3, 224, 224)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for name, p in model.named_parameters() if name.endswith("lora_B")))
        for name, value in model.named_parameters():
            if name in frozen:
                self.assertIsNone(value.grad)
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)

    def test_init_arguments_and_random_seeds_are_explicit_and_exclusive(self):
        for seed in (True, False, -1, 2**32, 1.5, "42"):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "random_seed"):
                student.build_vit_tiny_student(random_seed=seed)
        for kwargs in ({}, {"checkpoint": "missing.pth", "random_seed": 42},
                       {"initialization": {}, "random_seed": 42},
                       {"checkpoint": "missing.pth", "initialization": {}}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "exactly one"):
                student.build_vit_tiny_student(**kwargs)

    def test_random_metadata_rejects_false_source_claims_and_architecture_changes(self):
        baseline = student.build_vit_tiny_student(random_seed=42).initialization
        mutations = [
            {"architecture": "deit_small_patch16_224"}, {"variant": "distilled"},
            {"source_num_classes": 1000}, {"source_num_classes": True},
            {"source_num_classes": 1.0}, {"pretraining": "ImageNet-1k"},
            {"initialization_mode": "imagenet"}, {"seed": True}, {"seed": -1},
            {"checkpoint": "/must/not/be/loaded.pth"}, {"checkpoint_sha256": "a1311bcf" + "0" * 56},
            {"official_checkpoint_url": student.CHECKPOINT_URL}, {"unknown_field": 1},
        ]
        for mutation in mutations:
            candidate = deepcopy(baseline)
            candidate.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                student.build_vit_tiny_student(initialization=candidate)
        for key in baseline:
            candidate = deepcopy(baseline)
            del candidate[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                student.build_vit_tiny_student(initialization=candidate)


if __name__ == "__main__":
    unittest.main()
