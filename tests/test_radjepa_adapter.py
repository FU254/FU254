"""RadJEPA feature and local-checkpoint guardrails, without author weights."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from foundation_sft.models.common import PreprocessSpec
from foundation_sft.models.radjepa import (RadJEPAAdapter, _hf_weights,
                                           _preprocessing, build_radjepa)


class TokenEncoder(nn.Module):
    """Known tokens with the structural contract of the published ViT-B/14."""

    def __init__(self, prefix=0, pooled=False):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 768, 14, stride=14, device="meta")
        self.blocks = nn.ModuleList()
        for _ in range(12):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(768, 2304, device="meta")
            self.blocks.append(block)
        self.num_prefix_tokens = prefix
        self.tokens = nn.Parameter(torch.arange((4 + prefix) * 768, dtype=torch.float32)
                                   .reshape(1, 4 + prefix, 768))
        self.pooled = pooled

    def forward_features(self, images):
        tokens = self.tokens.expand(images.shape[0], -1, -1)
        return tokens.mean(dim=1) if self.pooled else tokens


class RadJEPAAdapterTests(unittest.TestCase):
    def test_local_tokens_exclude_prefix_and_preserve_spatial_order(self):
        for prefix in (0, 1, 5):
            with self.subTest(prefix=prefix):
                encoder = TokenEncoder(prefix=prefix)
                adapter = RadJEPAAdapter(encoder, PreprocessSpec(28))
                features = adapter.forward_features(torch.zeros(2, 3, 28, 28))
                expected = encoder.tokens[:, prefix:].expand(2, -1, -1)
                torch.testing.assert_close(features.local_features, expected)
                torch.testing.assert_close(features.global_features, expected.mean(dim=1))
                self.assertEqual(features.grid_shape, (2, 2))
                self.assertEqual(adapter.lora_target_names(),
                                 [f"backbone.blocks.{index}.attn.qkv" for index in range(12)])
                features.global_features.sum().backward()
                self.assertEqual(torch.count_nonzero(encoder.tokens.grad[:, :prefix]).item(), 0)
                self.assertGreater(torch.count_nonzero(encoder.tokens.grad[:, prefix:]).item(), 0)

    def test_pooled_output_and_incorrect_image_size_are_rejected(self):
        adapter = RadJEPAAdapter(TokenEncoder(pooled=True), PreprocessSpec(28))
        with self.assertRaisesRegex(ValueError, "patch-token"):
            adapter.forward_features(torch.zeros(1, 3, 28, 28))
        with self.assertRaisesRegex(ValueError, "image_size"):
            adapter.forward_features(torch.zeros(1, 3, 42, 42))

    def test_preprocessing_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "verified preprocessing"):
            _preprocessing({})
        values = {"preprocessing": {"image_size": 28, "mean": [0.5] * 3, "std": [0.25] * 3}}
        self.assertEqual(_preprocessing(values).mean, (0.5, 0.5, 0.5))
        values["preprocessing"]["image_size"] = 29
        with self.assertRaisesRegex(ValueError, "divisible"):
            _preprocessing(values)

    def test_builder_requires_weights_and_explicit_source_review(self):
        with self.assertRaisesRegex(ValueError, "random initialization"):
            build_radjepa({})
        with self.assertRaisesRegex(ValueError, "Review the local source"):
            build_radjepa({"checkpoint": "/not/a/checkpoint"})

    def test_local_processor_metadata_supplies_only_verified_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            processor = {"do_resize": True, "size": {"shortest_edge": 42},
                         "do_center_crop": True, "crop_size": {"height": 28, "width": 28},
                         "do_normalize": True, "image_mean": [0.5, 0.4, 0.3],
                         "image_std": [0.25, 0.2, 0.15], "do_rescale": True,
                         "rescale_factor": 1 / 255, "resample": 3,
                         "image_processor_type": "UnreviewedCustomClassNotExecuted"}
            file = directory / "preprocessor_config.json"
            file.write_text(json.dumps(processor))
            config = {"checkpoint": str(directory), "preprocessing": {"image_size": None}}
            spec = _preprocessing(config)
            self.assertEqual(spec.image_size, 28)
            self.assertEqual(spec.mean, (0.5, 0.4, 0.3))
            self.assertEqual(spec.resize_mode, "shortest_edge_center_crop")
            self.assertEqual(spec.resize_shortest_edge, 42)
            self.assertEqual(spec.interpolation, "bicubic")
            config["preprocessing"] = {"image_size": 28, "mean": [0.1] * 3, "std": [0.2] * 3}
            self.assertEqual(_preprocessing(config).mean, (0.1,) * 3)
            config["preprocessing"] = {}
            del processor["image_std"]
            file.write_text(json.dumps(processor))
            with self.assertRaisesRegex(ValueError, "metadata is incomplete"):
                _preprocessing(config)

    def test_missing_shard_and_external_code_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "modeling_radjepa.py").write_text("# reviewed fixture source\n")
            config = {"auto_map": {"AutoModel": "modeling_radjepa.RadJEPA"}}
            (directory / "config.json").write_text(json.dumps(config))
            index = {"weight_map": {"encoder.weight": "model-00001.safetensors"}}
            (directory / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "checkpoint shard"):
                _hf_weights(directory)
            config["auto_map"]["AutoModel"] = "another/repository--modeling_radjepa.RadJEPA"
            (directory / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "reviewed local snapshot"):
                _hf_weights(directory)


if __name__ == "__main__":
    unittest.main()
