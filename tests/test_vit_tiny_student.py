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

    def build(self):
        # Fixture architecture tests do not pretend synthetic tensors have the
        # official file's identity; bypass only its external source checksum.
        with mock.patch.object(student, "sha256_file", return_value=SYNTHETIC_SHA256):
            return student.build_vit_tiny_student(self.checkpoint)

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


if __name__ == "__main__":
    unittest.main()
