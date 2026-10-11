"""Native EVA-X and chunked RayDINO feature contracts without external assets."""

import unittest

import torch
from torch import nn

from foundation_sft.models.common import (FoundationClassifier, PreprocessSpec,
                                          load_backbone_weights)
from foundation_sft.models.eva_x import EVAXAdapter
from foundation_sft.models.ray_dino import RayDINOAdapter, build_ray_dino


class NativeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(6, 18)

    def forward(self, tokens):
        return tokens + self.attn.qkv(tokens).reshape(*tokens.shape, 3).sum(-1)


class SmallChunkedDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim, self.n_blocks, self.patch_size = 6, 4, 2
        self.embedding = nn.Conv2d(3, 6, 2, stride=2)
        self.blocks = nn.ModuleList([nn.ModuleList([NativeBlock(), NativeBlock()]) for _ in range(2)])

    def forward_features(self, images):
        spatial = self.embedding(images).flatten(2).transpose(1, 2)
        cls = spatial.mean(1, keepdim=True)
        tokens = torch.cat((cls, spatial), dim=1)
        for chunk in self.blocks:
            for block in chunk:
                tokens = block(tokens)
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens[:, 1:]}


class FoundationNativeTests(unittest.TestCase):
    def test_ray_dino_chunked_qkv_and_spatial_grid(self):
        torch.manual_seed(9)
        backbone = SmallChunkedDINO()
        images = torch.randn(2, 3, 4, 6)
        expected = backbone.forward_features(images)
        adapter = RayDINOAdapter(backbone, PreprocessSpec(4), {"source": "synthetic chunked test"})
        features = adapter.forward_features(images)
        torch.testing.assert_close(features.global_features, expected["x_norm_clstoken"])
        torch.testing.assert_close(features.local_features, expected["x_norm_patchtokens"])
        self.assertEqual(features.grid_shape, (2, 3))
        self.assertEqual(len(adapter.lora_target_names()), 4)
        model = FoundationClassifier(adapter, rank=2, alpha=4, hidden_dims=[])
        model(images).square().mean().backward()
        for name, param in model.named_parameters():
            if name.endswith("lora_B"):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
            if not param.requires_grad:
                self.assertIsNone(param.grad)
        with self.assertRaisesRegex(ValueError, "divisible"):
            adapter.forward_features(torch.randn(1, 3, 3, 4))

    def test_eva_patch_features_match_native_encoder(self):
        from classification.models.models_eva import VisionTransformer
        backbone = VisionTransformer(img_size=32, patch_size=16, embed_dim=24, depth=2,
                                     num_heads=3, num_classes=0, subln=True, xattn=False, rope=False)
        images = torch.randn(2, 3, 32, 32)
        adapter = EVAXAdapter(backbone, {"source": "synthetic EVA architecture"})
        adapter.eval()
        expected = backbone.forward_features(images)
        features = adapter.forward_features(images)
        torch.testing.assert_close(features.global_features, expected)
        torch.testing.assert_close(features.local_features,
                                   backbone.forward_features(images, return_patch_tokens=True))
        self.assertEqual(features.grid_shape, (2, 2))
        self.assertEqual(len(adapter.lora_target_names()), 6)
        model = FoundationClassifier(adapter, rank=2, alpha=4, hidden_dims=[])
        model(images).square().mean().backward()
        self.assertTrue(all(p.grad is not None for name, p in model.named_parameters()
                            if name.endswith("lora_B")))

    def test_unused_head_shape_is_ignored_but_backbone_mismatch_is_not(self):
        module = nn.Module()
        module.backbone = nn.Linear(3, 4)
        module.head = nn.Linear(4, 2)
        state = module.state_dict()
        state["head.weight"] = torch.randn(11, 4)
        state["head.bias"] = torch.randn(11)
        load_backbone_weights(module, state)
        del state["backbone.weight"]
        with self.assertRaisesRegex(ValueError, "backbone mismatch"):
            load_backbone_weights(module, state)

    def test_ray_dino_requires_real_local_source_and_weights(self):
        with self.assertRaisesRegex(ValueError, "local official"):
            build_ray_dino({})


if __name__ == "__main__":
    unittest.main()
