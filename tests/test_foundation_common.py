"""LoRA gradient, frozen probe, feature-grid and adapter checkpoint contracts."""

import copy
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.models.common import (
    BackboneAdapter,
    BackboneFeatures,
    FoundationClassifier,
    LoRALinear,
    PreprocessSpec,
    inject_lora,
)


class SmallBlock(nn.Module):
    def __init__(self, direct_linear=False):
        super().__init__()
        self.qkv = nn.Linear(6, 18)
        self.dropout = nn.Dropout(0.4)
        self.direct_linear = direct_linear

    def forward(self, inputs):
        # EVA-style encoders can bypass the qkv module's forward method.
        projected = F.linear(inputs, self.qkv.weight, self.qkv.bias) if self.direct_linear else self.qkv(inputs)
        query, key, value = projected.chunk(3, dim=-1)
        return inputs + self.dropout(torch.tanh(query + key + value))


class SmallBackbone(BackboneAdapter):
    """A four-patch encoder requiring no external weight or architecture files."""
    def __init__(self, direct_linear=False, targets=None):
        super().__init__()
        self.feature_dim = 6
        self.preprocess = PreprocessSpec(image_size=4)
        self.provenance = {"source": "deterministic unit-test fixture"}
        self.patch_embed = nn.Conv2d(3, 6, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList([SmallBlock(direct_linear), SmallBlock(direct_linear)])
        self.norm = nn.LayerNorm(6)
        self.targets = ["blocks.0.qkv", "blocks.1.qkv"] if targets is None else targets

    def lora_target_names(self):
        return self.targets

    def forward_features(self, images):
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return BackboneFeatures(tokens.mean(dim=1), tokens, (2, 2))


class FoundationCommonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314)

    def test_lora_preserves_initial_projection_and_receives_first_step_gradients(self):
        for direct_linear in (False, True):
            for use_bias in (False, True):
                with self.subTest(direct_linear=direct_linear, bias=use_bias):
                    base = nn.Linear(6, 18, bias=use_bias)
                    original = copy.deepcopy(base)
                    adapted = LoRALinear(base, rank=2, alpha=4)
                    inputs = torch.randn(2, 4, 6, requires_grad=True)
                    output = F.linear(inputs, adapted.weight, adapted.bias) if direct_linear else adapted(inputs)
                    torch.testing.assert_close(output, original(inputs), rtol=0, atol=0)
                    (output * torch.randn_like(output)).sum().backward()
                    self.assertIsNone(adapted.base.weight.grad)
                    self.assertFalse(adapted.base.weight.requires_grad)
                    if use_bias:
                        self.assertIsNone(adapted.base.bias.grad)
                        self.assertFalse(adapted.base.bias.requires_grad)
                    self.assertGreater(torch.count_nonzero(adapted.lora_B.grad).item(), 0)
                    # Zero-initialized B makes the initial A gradient exactly zero.
                    self.assertEqual(torch.count_nonzero(adapted.lora_A.grad).item(), 0)
                    self.assertTrue(torch.isfinite(inputs.grad).all())
                    torch.optim.SGD([adapted.lora_A, adapted.lora_B], lr=0.05).step()
                    changed = F.linear(inputs, adapted.weight, adapted.bias) if direct_linear else adapted(inputs)
                    self.assertFalse(torch.equal(changed, original(inputs)))
                    torch.testing.assert_close(adapted(inputs), F.linear(inputs, adapted.weight, adapted.bias),
                                               rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(adapted.base.weight, original.weight, rtol=0, atol=0)

    def test_linear_probe_is_single_head_and_encoder_stays_frozen_in_eval(self):
        model = FoundationClassifier(SmallBackbone(), mode="linear_probe")
        model.train()
        self.assertIsInstance(model.head, nn.Linear)
        self.assertIsInstance(model.output_norm, nn.Identity)
        self.assertTrue(model.head.training)
        self.assertTrue(all(not module.training for module in model.adapter.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.adapter.parameters()))
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {"head.weight", "head.bias"})
        images = torch.randn(2, 3, 4, 4, requires_grad=True)
        first = model(images, return_features=True)
        second = model(images, return_features=True)
        torch.testing.assert_close(first["global_features"], second["global_features"], rtol=0, atol=0)
        self.assertEqual(first["grid_shape"], (2, 2))
        self.assertEqual(tuple(first["local_features"].shape), (2, 4, 6))
        F.binary_cross_entropy_with_logits(first["logits"], torch.tensor([0.0, 1.0])).backward()
        self.assertIsNone(images.grad)
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.adapter.parameters()))

    def test_trainable_only_checkpoint_roundtrip_restores_logits_and_features(self):
        for direct_linear in (False, True):
            with self.subTest(direct_linear=direct_linear):
                original_backbone = SmallBackbone(direct_linear)
                restore_backbone = copy.deepcopy(original_backbone)
                original_frozen = {key: value.clone() for key, value in original_backbone.state_dict().items()}
                model = FoundationClassifier(original_backbone, mode="lora", rank=2, alpha=4, hidden_dims=(5,))
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
                images = torch.randn(2, 3, 4, 4)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(model(images), torch.tensor([0.0, 1.0]))
                    loss.backward()
                    optimizer.step()
                # Every pretrained weight remains unchanged, including the wrapped QKV.
                for key, original in original_frozen.items():
                    adapted_key = key.replace(".qkv.", ".qkv.base.")
                    torch.testing.assert_close(model.adapter.state_dict()[adapted_key], original, rtol=0, atol=0)
                state = model.trainable_state_dict()
                self.assertTrue(any(name.endswith("lora_B") for name in state))
                self.assertTrue(all(".base." not in name and "patch_embed" not in name for name in state))
                expected_names = {name for name, p in model.named_parameters() if p.requires_grad}
                self.assertEqual(set(state), expected_names)
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                restored = FoundationClassifier(restore_backbone, mode="lora", rank=2, alpha=4, hidden_dims=(5,))
                restored.load_trainable_state_dict(torch.load(buffer, weights_only=True))
                model.eval()
                restored.eval()
                actual, expected = restored(images, return_features=True), model(images, return_features=True)
                for key in ("logits", "global_features", "local_features"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                missing = dict(state)
                missing.pop(next(iter(missing)))
                with self.assertRaisesRegex(ValueError, "missing="):
                    restored.load_trainable_state_dict(missing)
                unexpected = dict(state, unrelated_weight=torch.ones(1))
                with self.assertRaisesRegex(ValueError, "unexpected="):
                    restored.load_trainable_state_dict(unexpected)

    def test_target_validation_prevents_partial_or_wrong_projection_injection(self):
        for targets in ([], ["blocks.0.qkv", "blocks.0.qkv"], ["blocks.0.qkv", "blocks.1.typo"], ["blocks.0.dropout"]):
            with self.subTest(targets=targets):
                adapter = SmallBackbone(targets=targets)
                with self.assertRaises(ValueError):
                    inject_lora(adapter)
                self.assertTrue(all(isinstance(block.qkv, nn.Linear) for block in adapter.blocks))
        adapter = SmallBackbone()
        targets = inject_lora(adapter, rank=2, alpha=4)
        self.assertEqual(targets, ["blocks.0.qkv", "blocks.1.qkv"])
        self.assertTrue(all(isinstance(block.qkv, LoRALinear) for block in adapter.blocks))
        # Non-attention linear layers and normalization are untouched.
        self.assertIsInstance(adapter.patch_embed, nn.Conv2d)
        self.assertIsInstance(adapter.norm, nn.LayerNorm)
        with self.assertRaisesRegex(ValueError, "mode"):
            FoundationClassifier(SmallBackbone(), mode="unverified_full_finetune")

    def test_feature_contract_rejects_prefix_tokens_and_incompatible_shapes(self):
        valid = BackboneFeatures(torch.ones(2, 6), torch.ones(2, 4, 6), (2, 2))
        self.assertIs(valid.validate(), valid)
        invalid = (
            BackboneFeatures(torch.ones(2, 6), torch.ones(2, 5, 6), (2, 2)),  # CLS must be removed.
            BackboneFeatures(torch.ones(2, 6), torch.ones(1, 4, 6), (2, 2)),
            BackboneFeatures(torch.ones(2, 6), torch.ones(2, 4, 3), (2, 2)),
            BackboneFeatures(torch.ones(2, 6), torch.ones(2, 4, 6), (0, 4)),
            BackboneFeatures(torch.ones(2, 6), torch.ones(2, 3, 6), (1.5, 2)),
            BackboneFeatures(torch.ones(2, 6), torch.ones(2, 4, 6), (True, 4)),
            BackboneFeatures(torch.ones(2, 1, 6), torch.ones(2, 4, 6), (2, 2)),
        )
        for features in invalid:
            with self.subTest(shapes=(features.global_features.shape, features.local_features.shape), grid=features.grid_shape), self.assertRaises(ValueError):
                features.validate()

    def test_lora_rank_and_scaling_require_valid_numeric_configuration(self):
        for rank in (0, -1, 0.5, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                LoRALinear(nn.Linear(3, 6), rank=rank)
        for alpha in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                LoRALinear(nn.Linear(3, 6), alpha=alpha)


if __name__ == "__main__":
    unittest.main()
