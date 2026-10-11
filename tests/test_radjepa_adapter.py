"""RadJEPA feature and local-checkpoint guardrails, without author weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from foundation_sft.models.common import PreprocessSpec
from foundation_sft.models.radjepa import (OFFICIAL_CHECKPOINT_NAME, RadJEPAAdapter, _hf_weights,
                                           _preprocessing, build_radjepa)


REVIEWED_FIXTURE_SOURCE = '''
import torch
from torch import nn
from huggingface_hub import hf_hub_download

class RadJEPAConfig:
    def __init__(self, **values):
        self.__dict__.update(values)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("The vendor config loader must not be called")

class FixtureEncoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 768, 14, stride=14, device="meta")
        # Structural stand-ins avoid allocating 12 full ViT projections. The
        # real numerical checkpoint below still includes tokens and pos_embed.
        self.patch_embed.proj.register_parameter("weight", None)
        self.patch_embed.proj.register_parameter("bias", None)
        self.blocks = nn.ModuleList()
        for _ in range(12):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(768, 2304, device="meta")
            block.attn.qkv.register_parameter("weight", None)
            block.attn.qkv.register_parameter("bias", None)
            self.blocks.append(block)
        count = (image_size // 14) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, count, 768))
        self.tokens = nn.Parameter(torch.zeros(1, count, 768))
        self.num_prefix_tokens = 0

    def forward_features(self, images):
        return (self.tokens + self.pos_embed).expand(images.shape[0], -1, -1)

class RadJEPAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = FixtureEncoder(config.image_size)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        hf_hub_download(repo_id="vendor-must-not-be-called", filename="invalid")
        raise AssertionError("The vendor model loader must not be called")
'''


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
    def _official_fixture(self, directory, *, nested=False, prefixed=False):
        metadata = {"model_type": "radjepa", "image_size": 28, "patch_size": 14,
                    "embed_dim": 768,
                    "auto_map": {"AutoConfig": "modeling_radjepa.RadJEPAConfig",
                                 "AutoModel": "modeling_radjepa.RadJEPAEncoder"}}
        (directory / "config.json").write_text(json.dumps(metadata))
        (directory / "modeling_radjepa.py").write_text(REVIEWED_FIXTURE_SOURCE)
        state = {"tokens": torch.arange(4 * 768, dtype=torch.float32).reshape(1, 4, 768),
                 "pos_embed": torch.full((1, 4, 768), 0.25)}
        if prefixed:
            state = {"module.encoder." + key: value for key, value in state.items()}
        payload = {"encoder": state}
        if nested:
            payload = {"state_dict": payload}
        # Unused predictor contents must never become the image encoder.
        payload["predictor"] = {"unrelated": torch.ones(2)}
        torch.save(payload, directory / OFFICIAL_CHECKPOINT_NAME)
        config = {"checkpoint": str(directory), "allow_remote_code": True,
                  "preprocessing": {"image_size": None, "mean": [0.5] * 3, "std": [0.25] * 3}}
        return config, state

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

    def test_official_tar_loads_offline_without_vendor_factories(self):
        for nested, prefixed in ((False, False), (True, True)):
            with self.subTest(nested=nested, prefixed=prefixed), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, state = self._official_fixture(directory, nested=nested, prefixed=prefixed)
                before = dict(config["preprocessing"])
                with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Network is forbidden")), \
                     patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("HF factory is forbidden")), \
                     patch("socket.create_connection", side_effect=AssertionError("Network is forbidden")):
                    adapter = build_radjepa(config)
                self.assertEqual(config["preprocessing"], before)
                expected = state[("module.encoder." if prefixed else "") + "tokens"] + 0.25
                features = adapter.forward_features(torch.zeros(2, 3, 28, 28))
                torch.testing.assert_close(features.local_features, expected.expand(2, -1, -1))
                torch.testing.assert_close(features.global_features, expected.mean(dim=1).expand(2, -1))
                self.assertEqual(adapter.num_prefix_tokens, 0)
                self.assertEqual(adapter.provenance["weight_format"], "official_encoder_tar")
                self.assertEqual(adapter.provenance["state_key"], "state_dict.encoder" if nested else "encoder")
                self.assertEqual(adapter.provenance["encoder_path"], "model")
                self.assertIn(OFFICIAL_CHECKPOINT_NAME, adapter.provenance["weights_sha256"])
                self.assertIn("modeling_radjepa.py", adapter.provenance["source_sha256"])
                self.assertEqual(len(adapter.provenance["config_sha256"]), 64)

    def test_official_config_supplies_size_without_guessing_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, _ = self._official_fixture(directory)
            config["preprocessing"] = {"image_size": None, "mean": None, "std": None}
            with self.assertRaisesRegex(ValueError, "verified preprocessing settings: mean, std"):
                _preprocessing(config)
            self.assertIsNone(config["preprocessing"]["image_size"])

    def test_official_config_and_preprocessing_size_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config, _ = self._official_fixture(directory)
            config["preprocessing"]["image_size"] = 42
            with self.assertRaisesRegex(ValueError, "fixed positional-embedding image_size=28"):
                build_radjepa(config)

    def test_official_tar_requires_all_encoder_weights_and_exact_shapes(self):
        for variation in ("missing", "unexpected", "shape", "interior_prefix"):
            with self.subTest(variation=variation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, state = self._official_fixture(directory)
                if variation == "missing":
                    state.pop("pos_embed")
                elif variation == "unexpected":
                    state["unexpected"] = torch.ones(1)
                elif variation == "shape":
                    state["tokens"] = torch.ones(1, 5, 768)
                else:
                    state["some.encoder.tokens"] = state.pop("tokens")
                torch.save({"encoder": state}, directory / OFFICIAL_CHECKPOINT_NAME)
                with self.assertRaisesRegex(ValueError, "pretrained backbone mismatch"):
                    build_radjepa(config)

    def test_official_prefix_collisions_and_ambiguous_containers_are_rejected(self):
        for variation in ("duplicate", "two_encoders", "only_target", "non_tensor"):
            with self.subTest(variation=variation), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                config, state = self._official_fixture(directory)
                payload = {"encoder": state}
                if variation == "duplicate":
                    state["module.encoder.tokens"] = state["tokens"]
                    message = "Duplicate or empty"
                elif variation == "two_encoders":
                    payload["state_dict"] = {"encoder": state}
                    message = "exactly one encoder"
                elif variation == "only_target":
                    payload = {"target_encoder": state}
                    message = "exactly one encoder"
                else:
                    state["non_tensor"] = "metadata is not a weight"
                    message = "tensor state dict"
                torch.save(payload, directory / OFFICIAL_CHECKPOINT_NAME)
                with self.assertRaisesRegex(ValueError, message):
                    build_radjepa(config)

    def test_official_and_standard_weight_formats_are_never_guessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._official_fixture(directory)
            self.assertEqual(_hf_weights(directory), [directory / OFFICIAL_CHECKPOINT_NAME])
            (directory / "model.safetensors").write_bytes(b"second format")
            with self.assertRaisesRegex(ValueError, "multiple weight formats"):
                _hf_weights(directory)
            (directory / OFFICIAL_CHECKPOINT_NAME).unlink()
            (directory / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"x": "x.bin"}}))
            with self.assertRaisesRegex(ValueError, "multiple weight formats"):
                _hf_weights(directory)

    def test_standard_hf_shards_remain_supported_and_path_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._official_fixture(directory)
            (directory / OFFICIAL_CHECKPOINT_NAME).unlink()
            index = directory / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "shard-a.safetensors", "b": "shard-b.safetensors"}}))
            for name in ("shard-a.safetensors", "shard-b.safetensors"):
                (directory / name).write_bytes(b"fixture")
            self.assertEqual([file.name for file in _hf_weights(directory)],
                             ["shard-a.safetensors", "shard-b.safetensors"])
            index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "checkpoint shard"):
                _hf_weights(directory)


if __name__ == "__main__":
    unittest.main()
