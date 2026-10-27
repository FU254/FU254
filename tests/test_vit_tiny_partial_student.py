"""Native partial DeiT-Tiny tuning keeps the ImageNet prefix immutable."""
from copy import deepcopy
import importlib.util
from importlib.metadata import version
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from foundation_sft.models import vit_tiny_student as student
from foundation_sft.distillation_losses import SingleTeacherDistillationLoss


HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    HAS_TIMM = tuple(int(part) for part in version("timm").split(".")[:2]) >= (0, 9)
SYNTHETIC_SHA256 = student.CHECKPOINT_SHA256_PREFIX + "0" * 56


@unittest.skipUnless(HAS_TIMM, "Native partial student requires timm>=0.9")
class PartialViTTinyStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="partial-vit-tiny-")
        cls.checkpoint = Path(cls.temporary.name) / "synthetic_imagenet.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(317)
            reference = timm.create_model(student.ARCHITECTURE, pretrained=False, num_classes=1000)
            torch.save({"model": reference.state_dict()}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def build(self, last=4, **kwargs):
        options = {"mode": "partial_finetune", "train_last_blocks": last, **kwargs}
        with mock.patch.object(student, "sha256_file", return_value=SYNTHETIC_SHA256):
            return student.build_vit_tiny_student(self.checkpoint, **options)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0, msg=name)

    def test_exact_partial_parameter_sets_and_metadata_for_all_three_scopes(self):
        counts = {}
        for last in (8, 4, 2):
            with self.subTest(last=last):
                model = self.build(last)
                first = 12 - last
                expected = {name for name, _ in model.named_parameters()
                    if name.startswith(("encoder.norm.", "encoder.head.")) or any(
                        name.startswith(f"encoder.blocks.{index}.") for index in range(first, 12))}
                actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(actual, expected)
                self.assertFalse(model.encoder.cls_token.requires_grad)
                self.assertFalse(model.encoder.pos_embed.requires_grad)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.patch_embed.parameters()))
                metadata = model.training_metadata()
                self.assertEqual(metadata["mode"], "partial_finetune")
                self.assertNotIn("lora", metadata)
                self.assertNotIn("regularization", metadata)
                self.assertEqual(metadata["partial_finetune"], {
                    "train_last_blocks": last,
                    "total_blocks": 12,
                    "trainable_block_indices": list(range(first, 12)),
                    "frozen_block_indices": list(range(first)),
                    "frozen_embeddings": ["patch_embed", "cls_token", "pos_embed"],
                    "trainable_modules": [f"encoder.blocks.{index}" for index in range(first, 12)]
                                         + ["encoder.norm", "encoder.head"],
                })
                self.assertEqual(metadata["total_parameters"], 5524609)
                counts[last] = metadata["trainable_parameters"]
                self.assertEqual(counts[last], sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertLess(counts[2], counts[4])
        self.assertLess(counts[4], counts[8])
        self.assertLess(counts[8], 5524609)
        self.assertEqual(counts, {8: 3559489, 4: 1780033, 2: 890305})

    def test_freezing_does_not_change_any_initial_tensor_or_classifier(self):
        torch.manual_seed(42)
        with mock.patch.object(student, "sha256_file", return_value=SYNTHETIC_SHA256):
            full = student.build_vit_tiny_student(self.checkpoint)
        for last in (8, 4, 2):
            torch.manual_seed(42)
            partial = self.build(last)
            self.assert_state_equal(full.state_dict(), partial.state_dict())
            self.assertEqual(full.initialization, partial.initialization)
        metadata = full.training_metadata()
        self.assertEqual(metadata["mode"], "full_finetune")
        self.assertNotIn("partial_finetune", metadata)

    def test_native_optimizer_updates_last_blocks_but_preserves_entire_frozen_prefix(self):
        for last in (8, 4, 2):
            model = self.build(last).train()
            frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                      if not parameter.requires_grad}
            trainable = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            images = torch.randn(2, 3, 224, 224)
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name in frozen:
                    self.assertIsNone(parameter.grad, name)
                else:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            actual = dict(model.named_parameters())
            for name, before in frozen.items():
                torch.testing.assert_close(actual[name], before, rtol=0, atol=0, msg=name)
            for index in range(12 - last, 12):
                name = f"encoder.blocks.{index}.attn.qkv.weight"
                self.assertFalse(torch.equal(actual[name].detach(), trainable[name]), name)
            self.assertFalse(torch.equal(actual["encoder.head.weight"].detach(), trainable["encoder.head.weight"]))
            self.assertFalse(torch.equal(actual["encoder.norm.weight"].detach(), trainable["encoder.norm.weight"]))

    def test_each_kd_term_reaches_trainable_blocks_without_updating_frozen_layers(self):
        images = torch.randn(1, 3, 224, 224)
        for last in (8, 4, 2):
            model = self.build(last).train()
            first = 12 - last
            for term in ("soft", "global", "local"):
                with self.subTest(last=last, term=term):
                    model.zero_grad(set_to_none=True)
                    features = model(images, return_features=True)
                    teacher_logits = torch.tensor([1.25], requires_grad=True)
                    teacher_global = torch.randn(1, 384, requires_grad=True)
                    teacher_local = torch.randn(1, 196, 384, requires_grad=True)
                    kd = SingleTeacherDistillationLoss(student_dim=192, teacher_dim=384,
                        soft_weight=float(term == "soft"), global_weight=float(term == "global"),
                        local_weight=float(term == "local"), pool_grid=(7, 7))
                    losses = kd(features["logits"], teacher_logits,
                        student_global=features["global_features"], teacher_global=teacher_global,
                        student_patches=features["local_features"], teacher_patches=teacher_local,
                        student_grid=features["grid_shape"], teacher_grid=(14, 14))
                    losses["total"].backward()
                    for index in range(first, 12):
                        parameter = model.encoder.blocks[index].attn.qkv.weight
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    self.assertTrue(all(parameter.grad is None for parameter in model.parameters()
                                        if not parameter.requires_grad))
                    self.assertTrue(all(value.grad is None for value in (teacher_logits, teacher_global, teacher_local)))
                    if term != "soft":
                        self.assertIsNone(model.encoder.head.weight.grad)

    def test_complete_partial_checkpoint_reconstructs_exactly_without_source_weights(self):
        images = torch.randn(1, 3, 224, 224)
        for last in (8, 4, 2):
            model = self.build(last).eval()
            with torch.no_grad():
                model.encoder.blocks[-1].attn.qkv.weight.add_(.01)
                expected = model(images, return_features=True)
            state = model.trainable_state_dict()
            metadata = deepcopy(model.training_metadata())
            self.assertEqual(set(state), set(model.state_dict()))
            self.assertIn("encoder.patch_embed.proj.weight", state)
            self.assertIn("encoder.blocks.0.attn.qkv.weight", state)
            metadata["initialization"]["checkpoint"] = "/missing/imagenet.pth"
            with mock.patch.object(torch, "load", side_effect=AssertionError("No ImageNet source needed")):
                restored = student.build_vit_tiny_student(initialization=metadata["initialization"],
                    mode=metadata["mode"], train_last_blocks=metadata["partial_finetune"]["train_last_blocks"]).eval()
                self.assertEqual(restored.training_metadata(), metadata)
                with self.assertRaisesRegex(RuntimeError, "complete student state"):
                    restored(images)
                restored.load_trainable_state_dict(state)
                with torch.no_grad():
                    actual = restored(images, return_features=True)
            self.assert_state_equal(state, restored.state_dict())
            for key in ("logits", "global_features", "local_features"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_partial_scope_is_explicit_imagenet_only_and_droppath_unchanged(self):
        for value in (None, True, False, 0, 1, 3, 6, 12, 2.0, "4"):
            with self.subTest(last=value), self.assertRaisesRegex(ValueError, "train_last_blocks"):
                student.build_vit_tiny_student("/missing.pth", mode="partial_finetune", train_last_blocks=value)
        for mode in ("full_finetune", "lora"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "only valid"):
                student.build_vit_tiny_student("/missing.pth", mode=mode, train_last_blocks=4)
        with self.assertRaisesRegex(ValueError, "ImageNet"):
            student.build_vit_tiny_student(random_seed=42, mode="partial_finetune", train_last_blocks=4)
        random_initialization = student.build_vit_tiny_student(random_seed=42).initialization
        with self.assertRaisesRegex(ValueError, "ImageNet"):
            student.build_vit_tiny_student(initialization=random_initialization,
                mode="partial_finetune", train_last_blocks=2)
        with self.assertRaisesRegex(ValueError, "DropPath"):
            self.build(4, drop_path_rate=.1)


if __name__ == "__main__":
    unittest.main()
