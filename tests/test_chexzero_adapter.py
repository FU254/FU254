"""CheXzero image-only adapter checks using independent tiny CLIP vision towers."""

from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundation_sft.models.chexzero import (
    CheXzeroAdapter, CheXzeroVisionTransformer, ProjectedSelfAttention,
    _load_visual_weights, _validate_release, _visual_state, build_chexzero,
)
from foundation_sft.models.common import FoundationClassifier, sha256_file


class ReferenceQuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


class ReferenceBlock(nn.Module):
    """Reviewed OpenAI residual block, using actual nn.MultiheadAttention."""

    def __init__(self, width=32, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)), ("gelu", ReferenceQuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, value):
        normalized = self.ln_1(value)
        value = value + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return value + self.mlp(self.ln_2(value))


class ReferenceVision(nn.Module):
    """Tiny original CLIP image path and its original checkpoint key layout."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=8, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(32) / 32 ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn(5, 32) / 32 ** 0.5)
        self.ln_pre = nn.LayerNorm(32)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(ReferenceBlock(), ReferenceBlock())
        self.ln_post = nn.LayerNorm(32)
        self.proj = nn.Parameter(torch.randn(32, 12) / 32 ** 0.5)

    def tokens(self, images):
        value = self.conv1(images).reshape(images.shape[0], 32, -1).permute(0, 2, 1)
        cls = self.class_embedding.to(value.dtype) + torch.zeros(
            value.shape[0], 1, value.shape[-1], dtype=value.dtype, device=value.device)
        value = torch.cat((cls, value), dim=1)
        value = self.ln_pre(value + self.positional_embedding.to(value.dtype))
        return self.transformer.resblocks(value.permute(1, 0, 2)).permute(1, 0, 2)

    def forward(self, images):
        return self.ln_post(self.tokens(images)[:, 0, :]) @ self.proj


def official_state(vision):
    """Use original full-CLIP visual keys, retaining unused text tensors."""
    return {**{"visual." + key: value.detach().clone() for key, value in vision.state_dict().items()},
            "token_embedding.weight": torch.randn(4, 8), "logit_scale": torch.tensor(1.0)}


def tiny_vision():
    return CheXzeroVisionTransformer(input_resolution=16, patch_size=8, width=32,
                                    layers=2, heads=4, output_dim=12)


class CheXzeroAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(91)
        self.reference = ReferenceVision()
        self.state = official_state(self.reference)

    def loaded_adapter(self):
        vision = tiny_vision()
        _load_visual_weights(vision, self.state)
        return CheXzeroAdapter(vision, {"source": "synthetic original-layout CLIP"})

    def test_attention_matches_real_mha_forward_and_backward(self):
        for training in (True, False):
            with self.subTest(training=training):
                original = nn.MultiheadAttention(32, 4).train(training)
                converted = ProjectedSelfAttention(32, 4).train(training)
                with torch.no_grad():
                    converted.qkv.weight.copy_(original.in_proj_weight)
                    converted.qkv.bias.copy_(original.in_proj_bias)
                    converted.out_proj.load_state_dict(original.out_proj.state_dict())
                left = torch.randn(5, 2, 32, requires_grad=True)
                right = left.detach().clone().requires_grad_()
                expected = original(left, left, left, need_weights=False)[0]
                actual = converted(right)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().sum().backward()
                actual.square().sum().backward()
                torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
                torch.testing.assert_close(converted.qkv.weight.grad, original.in_proj_weight.grad, rtol=0, atol=0)
                torch.testing.assert_close(converted.qkv.bias.grad, original.in_proj_bias.grad, rtol=0, atol=0)

    def test_original_tiny_visual_global_path_and_input_gradients_match(self):
        adapter = self.loaded_adapter()
        left = torch.randn(2, 3, 16, 16, requires_grad=True)
        right = left.detach().clone().requires_grad_()
        expected = self.reference(left)
        features = adapter.forward_features(right)
        torch.testing.assert_close(features.global_features, expected, rtol=0, atol=0)
        with torch.no_grad():
            local = self.reference.ln_post(self.reference.tokens(left.detach())[:, 1:, :]) @ self.reference.proj
        torch.testing.assert_close(features.local_features, local, rtol=0, atol=0)
        self.assertEqual(features.grid_shape, (2, 2))
        self.assertEqual(features.local_features.shape, (2, 4, 12))
        expected.square().sum().backward()
        features.global_features.square().sum().backward()
        torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)

    def test_zero_lora_preserves_features_and_trains_only_adapter_parameters(self):
        adapter = self.loaded_adapter()
        images = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            before = adapter.forward_features(images).global_features.clone()
        model = FoundationClassifier(adapter, mode="lora", rank=2, alpha=4, hidden_dims=())
        with torch.no_grad():
            after = model.adapter.forward_features(images).global_features
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        model(images).square().sum().backward()
        for name, parameter in model.adapter.named_parameters():
            if name.endswith("lora_B"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            elif not name.endswith("lora_A"):
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        self.assertEqual(model.adapter.lora_target_names(),
                         ["backbone.blocks.0.attn.qkv", "backbone.blocks.1.attn.qkv"])
        self.assertEqual(list(model.adapter.named_buffers()), [])

    def test_updated_lora_and_head_reload_exactly(self):
        model = FoundationClassifier(self.loaded_adapter(), mode="lora", rank=2, alpha=4, hidden_dims=())
        images = torch.randn(2, 3, 16, 16)
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        model(images).square().sum().backward()
        optimizer.step()
        state = {name: value.clone() for name, value in model.trainable_state_dict().items()}
        reloaded = FoundationClassifier(self.loaded_adapter(), mode="lora", rank=2, alpha=4, hidden_dims=())
        reloaded.load_trainable_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(reloaded(images), model(images), rtol=0, atol=0)
        self.assertEqual(model.training_metadata(), reloaded.training_metadata())

    def test_linear_probe_freezes_every_visual_parameter(self):
        model = FoundationClassifier(self.loaded_adapter(), mode="linear_probe")
        model.train()
        self.assertFalse(model.adapter.training)
        model(torch.randn(2, 3, 16, 16)).sum().backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.adapter.parameters()))
        self.assertIsNotNone(model.head.weight.grad)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 13)

    def test_strict_weights_reject_missing_extra_shape_and_nonfinite_before_copying(self):
        cases = []
        missing = dict(self.state)
        missing.pop("visual.ln_post.bias")
        cases.append(missing)
        extra = dict(self.state)
        extra["visual.unused_head.weight"] = torch.zeros(2, 2)
        cases.append(extra)
        shape = dict(self.state)
        shape["visual.proj"] = torch.zeros(32, 13)
        cases.append(shape)
        nonfinite = dict(self.state)
        nonfinite["visual.conv1.weight"] = self.state["visual.conv1.weight"].clone()
        nonfinite["visual.conv1.weight"].flatten()[0] = float("nan")
        cases.append(nonfinite)
        for index, state in enumerate(cases):
            with self.subTest(case=index):
                vision = tiny_vision()
                original = {name: value.clone() for name, value in vision.state_dict().items()}
                with self.assertRaises(ValueError):
                    _load_visual_weights(vision, state)
                for name, value in vision.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_checkpoint_extraction_requires_visual_prefix_and_rejects_aliases(self):
        with self.assertRaisesRegex(ValueError, "visual"):
            _visual_state(self.reference.state_dict())
        ambiguous = dict(self.state)
        ambiguous["visual.blocks.0.attn.qkv.weight"] = self.state["visual.transformer.resblocks.0.attn.in_proj_weight"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _visual_state(ambiguous)
        self.assertNotIn("token_embedding.weight", _visual_state(self.state))

    def test_release_architecture_and_preprocessing_without_large_allocations(self):
        with torch.device("meta"):
            vision = CheXzeroVisionTransformer()
            adapter = CheXzeroAdapter(vision, {})
        self.assertEqual(sum(p.numel() for p in vision.parameters()), 87849216)
        self.assertEqual(vision.conv1.weight.shape, (768, 3, 32, 32))
        self.assertEqual(vision.positional_embedding.shape, (50, 768))
        self.assertEqual(vision.proj.shape, (768, 512))
        self.assertEqual(len(adapter.lora_target_names()), 12)
        self.assertEqual(adapter.preprocess.image_size, 224)
        self.assertEqual(adapter.preprocess.resize_mode, "chexzero_pad320")
        self.assertEqual(adapter.preprocess.interpolation, "bicubic")
        self.assertEqual(adapter.preprocess.mean, (101.48761 / 255,) * 3)
        self.assertEqual(adapter.preprocess.std, (83.43944 / 255,) * 3)
        self.assertIn("shared DICOM", adapter.provenance["preprocessing_scope"])
        self.assertFalse(any("text" in name for name, _ in vision.named_parameters()))

    def test_builder_refuses_wrong_identity_and_hash_before_loading(self):
        with self.assertRaisesRegex(ValueError, "224"):
            build_chexzero({"image_size": 320})
        with self.assertRaisesRegex(ValueError, "ViT-B/32"):
            build_chexzero({"architecture": "RN50"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "medical.pt"
            path.write_bytes(b"synthetic checkpoint bytes")
            with patch("foundation_sft.models.chexzero.load_checkpoint") as loader:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    build_chexzero({"checkpoint": str(path), "checkpoint_sha256": "a" * 64})
                loader.assert_not_called()
            # Exercise real tiny loading and provenance without allocating an
            # 88M model. Fixed full-release dimensions are checked above.
            with patch("foundation_sft.models.chexzero.load_checkpoint", return_value=self.state), \
                    patch("foundation_sft.models.chexzero._validate_release"), \
                    patch("foundation_sft.models.chexzero.CheXzeroVisionTransformer", side_effect=tiny_vision):
                adapter = build_chexzero({"checkpoint": str(path), "checkpoint_sha256": sha256_file(path)})
            self.assertEqual(adapter.provenance["sha256"], sha256_file(path))
            self.assertTrue(adapter.provenance["checkpoint_sha256_verified"])
            self.assertEqual(adapter.provenance["checkpoint"], str(path.resolve()))
            self.assertIn("token_embedding.weight", adapter.provenance["ignored_checkpoint_keys"])
            torch.testing.assert_close(adapter.backbone.conv1.weight, self.reference.conv1.weight, rtol=0, atol=0)

    def test_checkpoint_header_requires_released_patch_grid_and_twelve_blocks(self):
        # Only inspect shapes on meta tensors; no large weights are allocated.
        state = {
            "visual.conv1.weight": torch.empty(768, 3, 32, 32, device="meta"),
            "visual.class_embedding": torch.empty(768, device="meta"),
            "visual.positional_embedding": torch.empty(50, 768, device="meta"),
            "visual.proj": torch.empty(768, 512, device="meta"),
            **{f"visual.transformer.resblocks.{index}.ln_1.weight": torch.empty(768, device="meta")
               for index in range(12)},
        }
        _validate_release(state)
        wrong_patch = dict(state)
        wrong_patch["visual.conv1.weight"] = torch.empty(768, 3, 16, 16, device="meta")
        with self.assertRaisesRegex(ValueError, "ViT-B/32"):
            _validate_release(wrong_patch)
        missing_block = dict(state)
        missing_block.pop("visual.transformer.resblocks.11.ln_1.weight")
        with self.assertRaisesRegex(ValueError, "12"):
            _validate_release(missing_block)


if __name__ == "__main__":
    unittest.main()
