"""EVA-X MIM checkpoints retain their trained normalization and feature meaning."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from foundation_sft.models.common import FoundationClassifier
from foundation_sft.models.eva_x import build_eva_x


FACTORY = "classification.models.models_eva.eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE"


def small_native_eva(use_mean_pooling):
    from classification.models.models_eva import VisionTransformer
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=24, depth=2,
                             num_heads=3, num_classes=0, qkv_bias=True, subln=True,
                             xattn=False, rope=False, use_checkpoint=False,
                             use_mean_pooling=use_mean_pooling)


class EVACheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117)
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "eva_x_mim.pt"
        self.images = torch.randn(3, 3, 32, 32)

    def tearDown(self):
        self.directory.cleanup()

    def _state(self, pooled=False):
        backbone = small_native_eva(pooled)
        normalizer = backbone.fc_norm if pooled else backbone.norm
        # Nontrivial learned affine values catch discarded or freshly initialized
        # normalization, which an all-ones/all-zeros fixture cannot detect.
        with torch.no_grad():
            normalizer.weight.copy_(torch.linspace(0.45, 1.7, 24))
            normalizer.bias.copy_(torch.linspace(-0.2, 0.35, 24))
        return {key: value.detach().clone() for key, value in backbone.state_dict().items()}

    def _save(self, state, nested=False):
        metadata = {"best_auc": np.float64(0.875),
                    "ignored_keys": {"mask_token", "lm_head.weight", "lm_head.bias"}}
        envelope = {"teacher": {"encoder": state}, **metadata} if nested else {"model": state, **metadata}
        torch.save(envelope, self.path)

    def _build(self, state, pooling="patch_mean", nested=False):
        self._save(state, nested=nested)

        def factory(pretrained=False, **kwargs):
            self.assertFalse(pretrained)
            self.assertEqual(kwargs.get("num_classes"), 0)
            self.assertIsInstance(kwargs.get("use_mean_pooling"), bool)
            return small_native_eva(kwargs["use_mean_pooling"])

        config = {"checkpoint": str(self.path), "pooling": pooling}
        if nested:
            config["state_key"] = "teacher.encoder"
        with patch(FACTORY, side_effect=factory) as mocked:
            adapter = build_eva_x(config)
            mocked.assert_called_once()
        return adapter

    @staticmethod
    def _raw_tokens(backbone, images):
        patches = backbone.patch_embed(images)
        tokens = torch.cat((backbone.cls_token.expand(len(images), -1, -1), patches), dim=1)
        if backbone.pos_embed is not None:
            tokens = tokens + backbone.pos_embed
        tokens = backbone.pos_drop(tokens)
        bias = backbone.rel_pos_bias() if backbone.rel_pos_bias is not None else None
        for block in backbone.blocks:
            tokens = block(tokens, bias)
        return tokens

    def test_mim_checkpoint_preserves_token_norm_and_discards_only_known_heads(self):
        state = self._state()
        encoder_keys = set(state)
        extras = {"mask_token": torch.randn(1, 1, 24),
                  "lm_head.weight": torch.randn(13, 24), "lm_head.bias": torch.randn(13),
                  "head.weight": torch.randn(9, 24), "head.bias": torch.randn(9)}
        state.update(extras)
        adapter = self._build(state)
        self.assertIsInstance(adapter.backbone.norm, nn.LayerNorm)
        self.assertIsNone(adapter.backbone.fc_norm)
        self.assertIsInstance(adapter.backbone.head, nn.Identity)
        self.assertEqual(adapter.provenance["normalization_layout"], "token_norm")
        self.assertEqual(adapter.provenance["ignored_checkpoint_keys"], sorted(extras))
        self.assertEqual(set(adapter.backbone.state_dict()), encoder_keys)
        for key, loaded in adapter.backbone.state_dict().items():
            torch.testing.assert_close(loaded, state[key], rtol=0, atol=0)

    def test_pooled_norm_checkpoint_preserves_exact_encoder_parameters(self):
        state = self._state(pooled=True)
        adapter = self._build(state)
        self.assertIsInstance(adapter.backbone.norm, nn.Identity)
        self.assertIsInstance(adapter.backbone.fc_norm, nn.LayerNorm)
        self.assertEqual(adapter.provenance["normalization_layout"], "pooled_norm")
        self.assertEqual(adapter.provenance["ignored_checkpoint_keys"], [])
        for key, loaded in adapter.backbone.state_dict().items():
            torch.testing.assert_close(loaded, state[key], rtol=0, atol=0)

    def test_token_norm_features_match_native_patches_and_pool_after_normalization(self):
        adapter = self._build(self._state()).eval()
        with torch.no_grad():
            result = adapter.forward_features(self.images)
            expected_patches = adapter.backbone.forward_features(self.images, return_patch_tokens=True)
            raw_patches = self._raw_tokens(adapter.backbone, self.images)[:, 1:]
            incorrectly_pool_then_normalize = adapter.backbone.norm(raw_patches.mean(1))
        torch.testing.assert_close(result.local_features, expected_patches, rtol=0, atol=0)
        torch.testing.assert_close(result.global_features, expected_patches.mean(1), rtol=0, atol=0)
        self.assertGreater((result.global_features - incorrectly_pool_then_normalize).abs().max().item(), 0.01)
        self.assertEqual(result.grid_shape, (2, 2))

    def test_pooled_norm_features_match_native_global_and_local_features(self):
        adapter = self._build(self._state(pooled=True)).eval()
        with torch.no_grad():
            result = adapter.forward_features(self.images)
            expected_global = adapter.backbone.forward_features(self.images)
            expected_patches = adapter.backbone.forward_features(self.images, return_patch_tokens=True)
        torch.testing.assert_close(result.global_features, expected_global, rtol=0, atol=0)
        torch.testing.assert_close(result.local_features, expected_patches, rtol=0, atol=0)

    def test_explicit_legacy_pooling_respects_each_checkpoint_normalization_layout(self):
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                adapter = self._build(self._state(pooled), pooling="legacy_all_tokens_mean").eval()
                with torch.no_grad():
                    tokens = self._raw_tokens(adapter.backbone, self.images)
                    if pooled:
                        expected = adapter.backbone.fc_norm(tokens.mean(1))
                    else:
                        expected = adapter.backbone.norm(tokens).mean(1)
                    result = adapter.forward_features(self.images)
                torch.testing.assert_close(result.global_features, expected, rtol=0, atol=0)
                self.assertEqual(adapter.provenance["pooling"], "legacy_all_tokens_mean")

    def test_linear_probe_and_lora_freeze_pretrained_norm_and_reload_trained_parameters(self):
        labels = torch.tensor([0.0, 1.0, 0.0])
        for pooled in (False, True):
            state = self._state(pooled)
            for mode in ("linear_probe", "lora"):
                with self.subTest(pooled=pooled, mode=mode):
                    model = FoundationClassifier(self._build(state), mode=mode, rank=2,
                                                 alpha=4, hidden_dims=[12]).train()
                    normalizer = model.adapter.backbone.fc_norm if pooled else model.adapter.backbone.norm
                    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(model(self.images), labels)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertFalse(normalizer.weight.requires_grad)
                    self.assertIsNone(normalizer.weight.grad)
                    self.assertIsNone(normalizer.bias.grad)
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            self.assertIsNone(parameter.grad, name)
                    if mode == "lora":
                        lora_grads = [p.grad for name, p in model.named_parameters() if name.endswith("lora_B")]
                        self.assertEqual(len(lora_grads), 6)
                        self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in lora_grads))
                        self.assertTrue(any(torch.count_nonzero(grad).item() for grad in lora_grads))
                    else:
                        self.assertFalse(model.adapter.training)
                    optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        expected = model(self.images)
                    trained_file = self.path.parent / "trainable.pt"
                    torch.save(model.trainable_state_dict(), trained_file)
                    trained_state = torch.load(trained_file, map_location="cpu", weights_only=True)
                    reloaded = FoundationClassifier(self._build(state), mode=mode, rank=2,
                                                    alpha=4, hidden_dims=[12]).eval()
                    reloaded.load_trainable_state_dict(trained_state)
                    for name, tensor in reloaded.trainable_state_dict().items():
                        torch.testing.assert_close(tensor, trained_state[name], rtol=0, atol=0)
                    with torch.no_grad():
                        torch.testing.assert_close(reloaded(self.images), expected, rtol=0, atol=0)

    def test_missing_or_unexpected_encoder_and_unknown_head_keys_are_rejected(self):
        state = self._state()
        broken = dict(state)
        del broken["blocks.0.attn.q_proj.weight"]
        with self.assertRaisesRegex(ValueError, "mismatch.*q_proj"):
            self._build(broken)
        for key in ("extra_encoder.weight", "lm_head.extra", "head.extra", "mask_token.extra"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self._build({**state, key: torch.ones(1)})

    def test_incorrect_encoder_and_normalization_shapes_are_rejected(self):
        state = self._state()
        for key in ("norm.weight", "blocks.0.attn.q_proj.weight"):
            with self.subTest(key=key):
                broken = {**state, key: torch.ones(3)}
                with self.assertRaises((RuntimeError, ValueError)):
                    self._build(broken)

    def test_partial_ambiguous_or_missing_normalization_is_rejected_before_model_build(self):
        state = self._state()
        variants = []
        for missing in ("norm.weight", "norm.bias"):
            variants.append({key: value for key, value in state.items() if key != missing})
        no_norm = {key: value for key, value in state.items() if not key.startswith("norm.")}
        variants.append(no_norm)
        variants.append({**no_norm, "fc_norm.weight": torch.ones(24)})
        variants.append({**no_norm, "fc_norm.bias": torch.zeros(24)})
        variants.append({**state, "fc_norm.weight": torch.ones(24), "fc_norm.bias": torch.zeros(24)})
        for index, broken in enumerate(variants):
            with self.subTest(variant=index):
                self._save(broken)
                with patch(FACTORY) as mocked:
                    with self.assertRaisesRegex(ValueError, "norm"):
                        build_eva_x({"checkpoint": str(self.path)})
                    mocked.assert_not_called()

    def test_explicit_nested_state_key_loads_trained_token_normalization(self):
        state = self._state()
        adapter = self._build(state, nested=True)
        self.assertEqual(adapter.provenance["normalization_layout"], "token_norm")
        torch.testing.assert_close(adapter.backbone.norm.weight, state["norm.weight"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
