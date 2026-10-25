"""Native timm DropPath preserves weights, data RNG, and reloadable mask streams."""

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


@unittest.skipUnless(HAS_TIMM, "Native DropPath requires timm>=0.9")
class ViTTinyDropPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def build(self, rate=.1, **kwargs):
        return student.build_vit_tiny_student(random_seed=42, drop_path_rate=rate, **kwargs)

    def assert_tensors_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0, msg=key)

    def assert_stream_equal(self, first, second):
        self.assertEqual(first["format_version"], second["format_version"])
        self.assertEqual(first["seed"], second["seed"])
        for key in ("cpu", "cuda"):
            if first[key] is None:
                self.assertIsNone(second[key])
            else:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_same_random_weights_and_legacy_zero_metadata_without_global_rng_or_sources(self):
        torch.manual_seed(510)
        before = torch.get_rng_state().clone()
        with mock.patch.object(torch, "load", side_effect=AssertionError("No source weights")), mock.patch.object(
            torch.hub, "download_url_to_file", side_effect=AssertionError("No network")):
            original = self.build(0)
            candidate = self.build()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assert_tensors_equal(original.state_dict(), candidate.state_dict())
        self.assertEqual(original.training_metadata()["initialization"], candidate.training_metadata()["initialization"])
        self.assertNotIn("regularization", original.training_metadata())
        self.assertIsNone(original.stochastic_state_dict())
        original.load_stochastic_state_dict(None)
        self.assertEqual(candidate.training_metadata()["regularization"], {
            "drop_path_rate": .1, "schedule": "linear_12_blocks", "rng": "isolated_torch",
            "rng_seed": 42 ^ student.DROP_PATH_SEED_XOR})
        self.assertEqual(candidate.training_metadata()["trainable_parameters"], 5524609)

    def test_native_probability_is_linear_maximum_not_point_one_in_every_block(self):
        model = self.build()
        expected = torch.linspace(0, .1, 12).tolist()
        for block, rate in zip(model.encoder.blocks, expected):
            for key in ("drop_path1", "drop_path2"):
                layer = getattr(block, key)
                actual = 0 if isinstance(layer, torch.nn.Identity) else layer.drop_prob
                self.assertEqual(actual, rate)
        self.assertEqual(expected[0], 0)
        self.assertAlmostEqual(expected[-1], .1)

    def test_invalid_or_unrequested_initialization_modes_reject_before_loading(self):
        for rate in (True, False, -1, 1., float("inf"), float("nan"), "0.1", None):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "drop_path_rate"):
                self.build(rate)
        with self.assertRaisesRegex(ValueError, "random, fully trained"):
            self.build(mode="lora")
        with self.assertRaisesRegex(ValueError, "random, fully trained"):
            student.build_vit_tiny_student("missing.pth", drop_path_rate=.1)
        pretrained = {"architecture": student.ARCHITECTURE, "official_checkpoint_url": student.CHECKPOINT_URL,
            "checkpoint_sha256": student.CHECKPOINT_SHA256_PREFIX + "0"*56,
            "pretraining": "ImageNet-1k", "source_num_classes": 1000, "variant": "single_cls"}
        with self.assertRaisesRegex(ValueError, "random, fully trained"):
            student.build_vit_tiny_student(initialization=pretrained, drop_path_rate=.1)

    def test_training_advances_own_stream_and_has_finite_gradients_while_preserving_outer_rng(self):
        model = self.build().train()
        images = torch.randn(4, 3, 224, 224)
        before = torch.get_rng_state().clone()
        internal = model.stochastic_state_dict()
        first = model(images, return_features=True)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertFalse(torch.equal(internal["cpu"], model.stochastic_state_dict()["cpu"]))
        second = model(images)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertFalse(torch.equal(first["logits"], second))
        self.assertEqual(tuple(first["local_features"].shape), (4, 196, 192))
        first["logits"].square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.encoder.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.head.weight.grad.abs().sum().item(), 0)

    def test_external_augmentation_draws_match_the_zero_droppath_control(self):
        control, candidate = self.build(0).train(), self.build().train()
        image = torch.randn(2, 3, 224, 224)
        draws = []
        for model in (control, candidate):
            torch.manual_seed(149)
            with torch.no_grad():
                model(image)
            draws.append(torch.rand(20))
        torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)

    def test_eval_equals_zero_rate_and_does_not_advance_saved_stream(self):
        control, candidate = self.build(0).eval(), self.build().eval()
        image = torch.randn(2, 3, 224, 224)
        before = candidate.stochastic_state_dict()
        outer = torch.get_rng_state().clone()
        with torch.no_grad():
            expected, actual = control(image), candidate(image)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), outer, rtol=0, atol=0)
        self.assert_stream_equal(before, candidate.stochastic_state_dict())

    def test_saved_weights_and_stream_reproduce_the_next_training_forward_without_teacher(self):
        original = self.build().train()
        image = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            original(image)
        stream = io.BytesIO()
        torch.save({"model": original.trainable_state_dict(), "metadata": original.training_metadata(),
                    "stochastic": original.stochastic_state_dict()}, stream)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        restored = student.build_vit_tiny_student(initialization=payload["metadata"]["initialization"],
            drop_path_rate=payload["metadata"]["regularization"]["drop_path_rate"]).train()
        with self.assertRaisesRegex(RuntimeError, "complete student state"):
            restored(image)
        restored.load_trainable_state_dict(payload["model"])
        before = torch.get_rng_state().clone()
        restored.load_stochastic_state_dict(payload["stochastic"])
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(restored.training_metadata(), original.training_metadata())
        with torch.no_grad():
            expected = original(image)
            torch.manual_seed(991)
            actual = restored(image)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_stream_equal(original.stochastic_state_dict(), restored.stochastic_state_dict())

    def test_stochastic_state_rejects_invalid_payloads_and_copies_valid_state(self):
        model = self.build()
        original = model.stochastic_state_dict()
        mutations = [None, {}, {**original, "extra": 1}, {**original, "format_version": True},
            {**original, "format_version": 2}, {**original, "seed": 17},
            {**original, "cpu": original["cpu"].float()},
            {**original, "cpu": torch.zeros(3, dtype=torch.uint8)},
            {**original, "cpu": torch.Generator().manual_seed(123).get_state()},
            {**original, "cuda": torch.zeros(0, dtype=torch.uint8)},
            {**original, "cuda": torch.zeros(15, dtype=torch.uint8)},
            {**original, "cuda": torch.zeros(16, dtype=torch.uint8)}]
        before = torch.get_rng_state().clone()
        for value in mutations:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                model.load_stochastic_state_dict(value)
            self.assert_stream_equal(original, model.stochastic_state_dict())
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        valid = deepcopy(original)
        model.load_stochastic_state_dict(valid)
        valid["cpu"].zero_()
        self.assert_stream_equal(original, model.stochastic_state_dict())
        with self.assertRaises(ValueError):
            self.build(0).load_stochastic_state_dict(original)

    def test_unexpected_native_schedule_is_rejected(self):
        import timm
        create = timm.create_model

        def incorrect(*args, **kwargs):
            model = create(*args, **kwargs)
            model.blocks[-1].drop_path1.drop_prob = .2
            return model

        with mock.patch.object(timm, "create_model", side_effect=incorrect), self.assertRaisesRegex(
            ValueError, "linear DropPath schedule"):
            self.build()


if __name__ == "__main__":
    unittest.main()
