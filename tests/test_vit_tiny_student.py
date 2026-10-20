"""Native DeiT-Tiny contracts using synthetic local checkpoints, no downloads."""

import importlib.util
from importlib.metadata import version
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import torch

from foundation_sft.models import vit_tiny_student as student
from foundation_sft.distillation_losses import SingleTeacherDistillationLoss


HAS_TIMM = importlib.util.find_spec("timm") is not None
if HAS_TIMM:
    # EVA-X intentionally retains timm 0.4.12; this optional student's native
    # token-returning API is exercised in its separate timm 0.9 environment.
    HAS_TIMM = tuple(int(part) for part in version("timm").split(".")[:2]) >= (0, 9)
SYNTHETIC_SHA256 = student.CHECKPOINT_SHA256_PREFIX + "0" * 56
UNSAFE_CALLED = False


def _untrusted_callable():
    global UNSAFE_CALLED
    UNSAFE_CALLED = True
    return {}


class UntrustedPayload:
    def __reduce__(self):
        return _untrusted_callable, ()


@unittest.skipUnless(HAS_TIMM, "Student native architecture needs timm>=0.9 (validated with 0.9.0)")
class ViTTinyStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm

        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temp.name) / "synthetic_deit.pth"
        torch.manual_seed(17)
        reference = timm.create_model(student.ARCHITECTURE, pretrained=False, num_classes=1000)
        cls.reference_state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
        torch.save({"model": cls.reference_state}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def build(self, **kwargs):
        # Fixture architecture tests do not pretend synthetic tensors have the
        # official file's identity; bypass only its external source checksum.
        with mock.patch.object(student, "sha256_file", return_value=SYNTHETIC_SHA256):
            return student.build_vit_tiny_student(self.checkpoint, **kwargs)

    def test_full_finetune_legacy_metadata_and_state_schema_are_unchanged(self):
        model = self.build()
        expected = {
            "model": "vit_tiny_student", "architecture": student.ARCHITECTURE,
            "mode": "full_finetune", "feature_dim": 192,
            "preprocessing": {
                "image_size": 224, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
                "resize_mode": "resize", "resize_shortest_edge": None,
                "interpolation": "bicubic", "do_hist_equalization": False,
            },
            "initialization": {
                "architecture": student.ARCHITECTURE, "official_checkpoint_url": student.CHECKPOINT_URL,
                "checkpoint": str(self.checkpoint.resolve()), "checkpoint_sha256": SYNTHETIC_SHA256,
                "pretraining": "ImageNet-1k", "source_num_classes": 1000, "variant": "single_cls",
            },
            "total_parameters": 5524609, "trainable_parameters": 5524609,
        }
        self.assertEqual(model.training_metadata(), expected)
        self.assertEqual(model.training_mode, "full_finetune")
        self.assertIsNone(model.lora)
        self.assertFalse(any("lora_" in name or ".base." in name for name in model.state_dict()))

    def test_native_cls_patch_layout_single_forward_and_full_gradients(self):
        model = self.build().eval()
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            expected_tokens = model.encoder.forward_features(images)
            expected_logits = model.encoder.forward_head(expected_tokens).squeeze(-1)
        with mock.patch.object(model.encoder, "forward_features", wraps=model.encoder.forward_features) as forward:
            result = model(images, return_features=True)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(set(result), {"logits", "global_features", "local_features", "grid_shape"})
        self.assertEqual(tuple(result["logits"].shape), (2,))
        self.assertEqual(tuple(result["global_features"].shape), (2, 192))
        self.assertEqual(tuple(result["local_features"].shape), (2, 196, 192))
        self.assertEqual(result["grid_shape"], (14, 14))
        torch.testing.assert_close(result["global_features"], expected_tokens[:, 0])
        torch.testing.assert_close(result["local_features"], expected_tokens[:, 1:])
        torch.testing.assert_close(result["logits"], expected_logits)
        self.assertEqual(tuple(model(images[:1]).shape), (1,))

        model.train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        torch.nn.functional.binary_cross_entropy_with_logits(
            model(images), torch.tensor([0.0, 1.0])
        ).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.encoder.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.head.weight.grad.abs().sum().item(), 0)

    def test_official_encoder_is_loaded_before_replacing_1000_class_head(self):
        model = self.build()
        for name, value in model.encoder.state_dict().items():
            if not name.startswith("head."):
                torch.testing.assert_close(value, self.reference_state[name], rtol=0, atol=0)
        self.assertEqual(tuple(model.encoder.head.weight.shape), (1, 192))
        metadata = model.training_metadata()
        self.assertEqual(metadata["total_parameters"], metadata["trainable_parameters"])
        self.assertEqual(metadata["initialization"]["checkpoint_sha256"], SYNTHETIC_SHA256)
        self.assertEqual(metadata["initialization"]["official_checkpoint_url"], student.CHECKPOINT_URL)

    def test_full_student_restore_needs_no_original_pretrained_file(self):
        model = self.build().eval()
        with torch.no_grad():
            model.encoder.blocks[0].attn.qkv.weight.add_(0.01)
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(images, return_features=True)
        metadata, state = model.training_metadata(), model.trainable_state_dict()
        self.assertEqual(set(state), set(model.state_dict()))
        # Prove that construction/evaluation need neither a torch.load call
        # nor the original source path, including paths from another machine.
        metadata["initialization"]["checkpoint"] = "/missing/original-imagenet.pth"
        with mock.patch.object(student.torch, "load", side_effect=AssertionError("No ImageNet file needed")):
            restored = student.build_vit_tiny_student(initialization=metadata["initialization"]).eval()
            with self.assertRaisesRegex(RuntimeError, "complete student state"):
                restored(images)
            restored.load_trainable_state_dict(state)
            with torch.no_grad():
                actual = restored(images, return_features=True)
        for key in ("logits", "global_features", "local_features"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_snapshot_is_independent_and_restore_rejects_partial_or_invalid_state(self):
        model = self.build()
        state = model.trainable_state_dict()
        before = state["encoder.head.weight"].clone()
        with torch.no_grad():
            model.encoder.head.weight.add_(1)
        torch.testing.assert_close(state["encoder.head.weight"], before, rtol=0, atol=0)
        incomplete = dict(state)
        del incomplete["encoder.blocks.0.attn.qkv.weight"]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            model.load_trainable_state_dict(incomplete)
        bad_shape = dict(state)
        bad_shape["encoder.head.weight"] = torch.zeros(2, 192)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            model.load_trainable_state_dict(bad_shape)
        bad_dtype = dict(state)
        bad_dtype["encoder.head.weight"] = state["encoder.head.weight"].double()
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            model.load_trainable_state_dict(bad_dtype)
        bad_nan = dict(state)
        bad_nan["encoder.head.weight"] = torch.full_like(state["encoder.head.weight"], float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            model.load_trainable_state_dict(bad_nan)

    def test_source_hash_and_encoder_head_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            student.build_vit_tiny_student(self.checkpoint)
        for defect in ("missing_encoder", "wrong_source_head", "extra_distillation_head"):
            state = dict(self.reference_state)
            if defect == "missing_encoder":
                del state["blocks.0.attn.qkv.weight"]
            elif defect == "wrong_source_head":
                state["head.weight"] = torch.zeros(1, 192)
            else:
                state["head_dist.weight"] = torch.zeros(1000, 192)
            checkpoint = Path(self.temp.name) / (defect + ".pth")
            torch.save({"model": state}, checkpoint)
            with self.subTest(defect=defect), mock.patch.object(
                student, "sha256_file", return_value=SYNTHETIC_SHA256
            ), self.assertRaisesRegex(ValueError, "mismatch"):
                student.build_vit_tiny_student(checkpoint)

    def test_only_safe_tensor_deserialization_is_used(self):
        global UNSAFE_CALLED
        UNSAFE_CALLED = False
        checkpoint = Path(self.temp.name) / "untrusted.pth"
        torch.save({"model": {}, "untrusted": UntrustedPayload()}, checkpoint)
        with mock.patch.object(student, "sha256_file", return_value=SYNTHETIC_SHA256), mock.patch.object(
            student.torch, "load", wraps=torch.load
        ) as load:
            with self.assertRaises(pickle.UnpicklingError):
                student.build_vit_tiny_student(checkpoint)
            self.assertEqual(load.call_count, 1)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertFalse(UNSAFE_CALLED)

    def test_missing_initialization_and_wrong_image_geometry_fail(self):
        with self.assertRaisesRegex(ValueError, "either"):
            student.build_vit_tiny_student()
        with self.assertRaisesRegex(ValueError, "initialization"):
            student.build_vit_tiny_student(initialization={})
        model = self.build()
        for shape in ((1, 3, 224, 240), (1, 1, 224, 224), (3, 224, 224)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "224"):
                model(torch.randn(shape))

    def test_lora_trains_only_twelve_qkv_adapters_and_linear_head(self):
        model = self.build(mode="lora", rank=8, alpha=16.0).train()
        expected_targets = [f"encoder.blocks.{index}.attn.qkv" for index in range(12)]
        self.assertEqual(model.training_mode, "lora")
        self.assertEqual(model.lora, {"rank": 8, "alpha": 16.0, "targets": expected_targets})
        metadata = model.training_metadata()
        self.assertEqual(metadata["mode"], "lora")
        self.assertEqual(metadata["lora"], model.lora)
        self.assertEqual(metadata["trainable_parameters"], 73921)
        self.assertEqual(metadata["total_parameters"], 5598337)
        self.assertIsInstance(model.encoder.head, torch.nn.Linear)
        self.assertEqual(tuple(model.encoder.head.weight.shape), (1, 192))
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        trainable = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        self.assertEqual(len(trainable), 26)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) or name in {
            "encoder.head.weight", "encoder.head.bias"
        } for name in trainable))
        for name, value in frozen.items():
            source_name = name.removeprefix("encoder.").replace(".qkv.base.", ".qkv.")
            torch.testing.assert_close(value, self.reference_state[source_name], rtol=0, atol=0)

        images = torch.randn(2, 3, 224, 224)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(images, return_features=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], torch.tensor([0.0, 1.0]))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name in frozen:
                    self.assertIsNone(parameter.grad, name)
                else:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        parameters = dict(model.named_parameters())
        for name, value in frozen.items():
            torch.testing.assert_close(parameters[name], value, rtol=0, atol=0, msg=name)
        for name in trainable:
            self.assertFalse(torch.equal(trainable[name], parameters[name].detach()), name)
        self.assertTrue(all(parameters[f"{target}.lora_B"].detach().abs().sum().item() > 0
                            for target in expected_targets))

    def test_lora_all_three_kd_terms_backpropagate_to_adapters(self):
        model = self.build(mode="lora")
        images = torch.randn(1, 3, 224, 224)
        teacher_logits = torch.tensor([2.0], requires_grad=True)
        teacher_global = torch.randn(1, 768, requires_grad=True)
        teacher_patches = torch.randn(1, 196, 768, requires_grad=True)
        for term in ("soft", "global", "local"):
            with self.subTest(term=term):
                model.zero_grad(set_to_none=True)
                output = model(images, return_features=True)
                kd = SingleTeacherDistillationLoss(
                    student_dim=192, teacher_dim=768,
                    soft_weight=float(term == "soft"), global_weight=float(term == "global"),
                    local_weight=float(term == "local"), pool_grid=(7, 7),
                )
                values = kd(output["logits"], teacher_logits,
                            student_global=output["global_features"], teacher_global=teacher_global,
                            student_patches=output["local_features"], teacher_patches=teacher_patches,
                            student_grid=output["grid_shape"], teacher_grid=(14, 14))
                values["total"].backward()
                for target in model.lora["targets"]:
                    gradient = model.get_parameter(f"{target}.lora_B").grad
                    self.assertIsNotNone(gradient, target)
                    self.assertTrue(torch.isfinite(gradient).all(), target)
                    self.assertGreater(gradient.abs().sum().item(), 0, target)
                if term == "soft":
                    self.assertIsNotNone(model.encoder.head.weight.grad)
                else:
                    self.assertIsNone(model.encoder.head.weight.grad)
                if term == "global":
                    self.assertEqual(sum(parameter.numel() for parameter in kd.parameters()), 147456)
                    self.assertIsNotNone(kd.projector.weight.grad)
                    self.assertGreater(kd.projector.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()
                                    if not parameter.requires_grad))
                self.assertTrue(all(value.grad is None for value in (teacher_logits, teacher_global, teacher_patches)))

    def test_lora_complete_restore_preserves_metadata_and_rejects_other_modes_or_ranks(self):
        model = self.build(mode="lora").eval()
        with torch.no_grad():
            model.encoder.blocks[0].attn.qkv.lora_B.add_(0.01)
            model.encoder.head.bias.add_(0.2)
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            expected = model(images, return_features=True)
        state, metadata = model.trainable_state_dict(), model.training_metadata()
        self.assertEqual(set(state), set(model.state_dict()))
        self.assertIn("encoder.patch_embed.proj.weight", state)
        self.assertIn("encoder.blocks.0.attn.qkv.base.weight", state)
        metadata["initialization"]["checkpoint"] = "/missing/original-imagenet.pth"
        with mock.patch.object(student.torch, "load", side_effect=AssertionError("No source file needed")):
            restored = student.build_vit_tiny_student(
                initialization=metadata["initialization"], mode=metadata["mode"],
                rank=metadata["lora"]["rank"], alpha=metadata["lora"]["alpha"],
            ).eval()
            self.assertEqual(restored.training_metadata(), metadata)
            with self.assertRaisesRegex(RuntimeError, "complete student state"):
                restored(images)
            restored.load_trainable_state_dict(state)
            with torch.no_grad():
                actual = restored(images, return_features=True)
        for name in ("logits", "global_features", "local_features"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        for name, value in state.items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(restored.training_metadata(), metadata)

        full = self.build()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            full.load_trainable_state_dict(state)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            model.load_trainable_state_dict(full.trainable_state_dict())
        wrong_rank = student.build_vit_tiny_student(initialization=metadata["initialization"], mode="lora", rank=4)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            wrong_rank.load_trainable_state_dict(state)
        for defect in ("missing_frozen_base", "nonfinite_frozen_base", "adapter_dtype"):
            damaged = dict(state)
            if defect == "missing_frozen_base":
                del damaged["encoder.patch_embed.proj.weight"]
            elif defect == "nonfinite_frozen_base":
                key = "encoder.patch_embed.proj.weight"
                damaged[key] = torch.full_like(state[key], float("nan"))
            else:
                key = "encoder.blocks.0.attn.qkv.lora_B"
                damaged[key] = state[key].double()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                restored.load_trainable_state_dict(damaged)

    def test_mode_rank_and_alpha_validation_precedes_checkpoint_io(self):
        invalid = [
            {"mode": "linear_probe"}, {"mode": True},
            {"rank": True}, {"rank": 0}, {"rank": -1}, {"rank": 8.0},
            {"alpha": True}, {"alpha": 0}, {"alpha": -1},
            {"alpha": float("nan")}, {"alpha": float("inf")}, {"alpha": "16"},
        ]
        with mock.patch.object(student, "sha256_file", side_effect=AssertionError("Validation must precede IO")):
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    student.build_vit_tiny_student(self.checkpoint, **kwargs)


if __name__ == "__main__":
    unittest.main()
