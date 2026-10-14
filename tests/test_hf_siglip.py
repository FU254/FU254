"""Native SigLIP software integration using synthetic local weights only."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import SiglipConfig, SiglipModel, SiglipVisionConfig, SiglipVisionModel

from foundation_sft.models import build_adapter
from foundation_sft.models.common import FoundationClassifier, PreprocessSpec, sha256_file
from foundation_sft.models import hf_siglip as siglip


def _tiny_vision():
    config = SiglipVisionConfig(hidden_size=12, intermediate_size=24, num_hidden_layers=2,
                               num_attention_heads=3, image_size=8, patch_size=4)
    config._attn_implementation = "eager"
    return config


def _processor(size=512):
    # Actual processor fields supplied by the user's official snapshot; no
    # medical weights or invented preprocessing statistics are in this fixture.
    return {"do_normalize": True, "do_rescale": True, "do_resize": True,
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
            "image_processor_type": "SiglipImageProcessor", "processor_class": "SiglipProcessor",
            "resample": 3, "rescale_factor": 1 / 255,
            "size": {"height": size, "width": size}}


class SiglipConfigTests(unittest.TestCase):
    def test_native_config_inherits_only_pinned_library_defaults(self):
        payload = {"model_type": "siglip", "vision_config": {"image_size": 512}}
        vision, raw, inherited = siglip._vision_config(payload)
        self.assertEqual(vision.image_size, 512)
        self.assertEqual(vision.hidden_size, 768)
        self.assertEqual(vision._attn_implementation, "eager")
        self.assertNotIn("image_size", inherited)
        self.assertEqual(inherited["layer_norm_eps"], SiglipVisionConfig().layer_norm_eps)
        self.assertEqual(raw, payload["vision_config"])
        self.assertEqual(payload["vision_config"], {"image_size": 512})

    def test_full_or_vision_only_official_config_is_supported(self):
        raw = SiglipVisionConfig(image_size=512).to_dict()
        a = siglip._vision_config({"model_type": "siglip", "vision_config": raw})[0]
        b = siglip._vision_config(raw)[0]
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_wrong_architecture_missing_vision_and_pool_removal_fail(self):
        cases = [{"model_type": "clip"}, {"model_type": "siglip"}]
        for fields in ({"image_size": 224}, {"hidden_size": 512}, {"patch_size": 32},
                       {"num_hidden_layers": 6}, {"num_attention_heads": 6},
                       {"vision_use_head": False}, {"layer_norm_eps": 0.0},
                       {"attention_dropout": float("nan")}, {"model_type": "dinov2"}):
            raw = SiglipVisionConfig(image_size=512).to_dict()
            raw.update(fields)
            cases.append({"model_type": "siglip", "vision_config": raw})
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                siglip._vision_config(payload)

    def test_actual_supplied_processor_maps_to_shared_pixels_without_double_rescale(self):
        payload = _processor()
        spec, record = siglip._preprocess(payload, SiglipVisionConfig(image_size=512))
        self.assertEqual(spec, PreprocessSpec(512, mean=(0.5,) * 3, std=(0.5,) * 3,
                                             resize_mode="resize", interpolation="bicubic"))
        self.assertEqual(record["published"], payload)
        self.assertEqual(record["shared_preprocess"], asdict(spec))
        self.assertEqual(record["defaults_library_version"], "4.48.3")
        # Native uint8 normalization and shared [0,1] normalization agree.
        from transformers import SiglipImageProcessor
        image = np.full((512, 512, 3), 153, dtype=np.uint8)
        official = SiglipImageProcessor.from_dict(payload)(image, return_tensors="pt")["pixel_values"]
        torch.testing.assert_close(official, torch.full_like(official, (153 / 255 - 0.5) / 0.5))

    def test_unknown_or_unsupported_preprocessing_fails(self):
        cases = ({"size": {"height": 224, "width": 224}}, {"rescale_factor": 1.0},
                 {"image_std": [0.0] * 3}, {"do_resize": False}, {"do_center_crop": True},
                 {"do_pad": True}, {"do_convert_rgb": "true"},
                 {"image_processor_type": "CLIPImageProcessor"})
        for fields in cases:
            payload = _processor()
            payload.update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                siglip._preprocess(payload, SiglipVisionConfig(image_size=512))

    def test_missing_config_is_actionable_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as root, patch("urllib.request.urlopen",
                side_effect=AssertionError("No downloads allowed")):
            with self.assertRaisesRegex(FileNotFoundError, "config.json.*small JSON"):
                siglip.build_hf_siglip({"model": "xray_siglip", "model_dir": root})

    def test_missing_processor_is_required_before_constructing_model(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text('{"model_type":"siglip"}')
            with patch("transformers.SiglipVisionModel", side_effect=AssertionError("No model construction")):
                with self.assertRaisesRegex(FileNotFoundError, "preprocessor_config.json"):
                    siglip.build_hf_siglip({"model": "xray_siglip", "model_dir": root})

    def test_registry_routes_xray_to_native_hf(self):
        config = {"model": "xray_siglip", "model_dir": "unused"}
        sentinel = object()
        with patch.object(siglip, "build_hf_siglip", return_value=sentinel) as builder:
            self.assertIs(build_adapter(config), sentinel)
            builder.assert_called_once_with(config)


class SiglipNativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="siglip-local-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.manual_seed(610)
        self.encoder = SiglipVisionModel(_tiny_vision()).eval()
        self.spec = PreprocessSpec(8, mean=(0.5,) * 3, std=(0.5,) * 3)

    def _adapter(self, normalize=False):
        return siglip.HFSiglipAdapter(copy.deepcopy(self.encoder), self.spec,
                                     {"fixture": "synthetic native SigLIP"}, normalize=normalize)

    def _weights(self, extra=None, remove=None, change=None):
        state = {name: tensor.clone().contiguous() for name, tensor in self.encoder.state_dict().items()}
        state.update(extra or {})
        if remove:
            del state[remove]
        if change:
            state[change[0]] = change[1]
        path = self.root / "model.safetensors"
        save_file(state, str(path))
        return path

    def test_pool_tokens_and_input_gradient_match_native_vision_model(self):
        adapter = self._adapter().eval()
        images = torch.randn(2, 3, 8, 8, requires_grad=True)
        native = self.encoder(pixel_values=images, return_dict=True)
        features = adapter.forward_features(images).validate()
        torch.testing.assert_close(features.global_features, native.pooler_output, rtol=0, atol=0)
        torch.testing.assert_close(features.local_features, native.last_hidden_state, rtol=0, atol=0)
        self.assertEqual(features.grid_shape, (2, 2))
        native_grad = torch.autograd.grad(native.pooler_output.square().sum(), images)[0]
        actual_grad = torch.autograd.grad(features.global_features.square().sum(), images)[0]
        torch.testing.assert_close(actual_grad, native_grad, rtol=0, atol=0)

    def test_optional_normalization_uses_native_features(self):
        adapter = self._adapter(normalize=True).eval()
        images = torch.randn(1, 3, 8, 8)
        native = self.encoder(pixel_values=images)
        features = adapter.forward_features(images)
        torch.testing.assert_close(features.global_features, F.normalize(native.pooler_output, dim=-1))
        torch.testing.assert_close(features.local_features, F.normalize(native.last_hidden_state, dim=-1))

    def test_qkv_lora_excludes_attention_pool_and_keeps_original_features(self):
        adapter = self._adapter()
        targets = adapter.lora_target_names()
        self.assertEqual(len(targets), 6)
        self.assertTrue(all(".self_attn." in name for name in targets))
        self.assertTrue(all(".head." not in name for name in targets))
        images = torch.randn(2, 3, 8, 8)
        before = adapter.eval().forward_features(images).global_features.detach()
        model = FoundationClassifier(adapter, mode="lora", rank=2, alpha=4, hidden_dims=(8,))
        # Wrapping/freezing Linear can select a different CPU matmul kernel;
        # the initial zero LoRA update preserves features to float32 precision.
        torch.testing.assert_close(adapter.forward_features(images).global_features, before,
                                   rtol=1e-5, atol=1e-6)
        loss = F.binary_cross_entropy_with_logits(model(images), torch.tensor([0.0, 1.0]))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for name, parameter in model.named_parameters() if name.endswith("lora_B")))
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad
                            for parameter in model.adapter.encoder.vision_model.head.parameters()))

    def test_linear_probe_freezes_complete_visual_encoder(self):
        model = FoundationClassifier(self._adapter(), mode="linear_probe").train()
        F.binary_cross_entropy_with_logits(model(torch.randn(2, 3, 8, 8)), torch.tensor([0.0, 1.0])).backward()
        self.assertFalse(model.adapter.training)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {"head.weight", "head.bias"})
        self.assertTrue(all(parameter.grad is None for parameter in model.adapter.parameters()))

    def test_strict_load_reads_only_visual_tensors(self):
        path = self._weights({"text_model.embeddings.token_embedding.weight": torch.full((3, 12), float("nan")),
                              "logit_scale": torch.zeros(1), "logit_bias": torch.ones(1)})
        from safetensors import safe_open
        opened = safe_open(str(path), framework="pt", device="cpu")

        class VisualOnlyReader:
            def __enter__(reader):
                opened.__enter__()
                return reader

            def __exit__(reader, *args):
                return opened.__exit__(*args)

            def keys(reader):
                return opened.keys()

            def get_slice(reader, name):
                return opened.get_slice(name)

            def get_tensor(reader, name):
                self.assertTrue(name.startswith("vision_model."), "Text/scalar tensors must not be read")
                return opened.get_tensor(name)

        target = SiglipVisionModel(_tiny_vision())
        with patch("safetensors.safe_open", return_value=VisualOnlyReader()):
            loading = siglip._load_visual_weights(target, path)
        self.assertEqual(loading["ignored_text_key_count"], 1)
        self.assertEqual(loading["ignored_contrastive_scalars"], ["logit_bias", "logit_scale"])
        for name, tensor in target.state_dict().items():
            torch.testing.assert_close(tensor, self.encoder.state_dict()[name], rtol=0, atol=0)

    def test_missing_extra_wrong_shape_and_nonfinite_visual_weights_fail(self):
        name = "vision_model.encoder.layers.0.self_attn.q_proj.weight"
        cases = ({"remove": name}, {"extra": {"vision_model.extra.weight": torch.ones(2)}},
                 {"change": (name, torch.ones(11, 12))},
                 {"change": (name, torch.full((12, 12), float("inf")))},
                 {"change": (name, torch.ones(12, 12, dtype=torch.int64))},
                 {"extra": {"unknown.weight": torch.ones(1)}},
                 {"extra": {"logit_scale": torch.ones(2)}})
        for fields in cases:
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                siglip._load_visual_weights(self.encoder, self._weights(**fields))

    def test_invalid_input_resolution_and_removed_pool_fail(self):
        adapter = self._adapter()
        for shape in ((1, 1, 8, 8), (1, 3, 12, 12), (3, 8, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                adapter.forward_features(torch.randn(shape))
        self.encoder.vision_model.use_head = False
        with self.assertRaisesRegex(ValueError, "attention pool"):
            siglip.HFSiglipAdapter(self.encoder, self.spec, {})

    def test_full_dual_tower_fixture_builds_vision_only_and_records_identity(self):
        raw = _tiny_vision().to_dict()
        config = SiglipConfig(vision_config=raw, text_config={"hidden_size": 12,
            "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 3,
            "vocab_size": 32, "max_position_embeddings": 8})
        config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        full = SiglipModel(config).eval()
        path = self.root / "model.safetensors"
        save_file(full.state_dict(), str(path))
        (self.root / "config.json").write_text(json.dumps(config.to_dict()))
        (self.root / "preprocessor_config.json").write_text(json.dumps(_processor(8)))
        release_shape = {name: getattr(_tiny_vision(), name) for name in siglip._RELEASE_SHAPE}
        with patch.object(siglip, "_RELEASE_SHAPE", release_shape), patch(
                "transformers.SiglipModel", side_effect=AssertionError("Runtime must not construct text tower")), patch(
                "urllib.request.urlopen", side_effect=AssertionError("Offline loader")):
            adapter = build_adapter({"model": "xray_siglip", "model_dir": str(self.root),
                "checkpoint": "model.safetensors", "expected_checkpoint_sha256": sha256_file(path)})
        images = torch.randn(2, 3, 8, 8)
        expected = full.vision_model(images).pooler_output
        torch.testing.assert_close(adapter.eval().forward_features(images).global_features, expected, rtol=0, atol=0)
        record = adapter.provenance
        self.assertFalse(record["text_tower_constructed"])
        self.assertFalse(record["text_weights_loaded"])
        self.assertEqual(record["checkpoint_sha256"], sha256_file(path))
        self.assertEqual(record["config_sha256"], sha256_file(self.root / "config.json"))
        self.assertEqual(record["preprocessor_config_sha256"], sha256_file(self.root / "preprocessor_config.json"))
        self.assertEqual(record["ignored_text_key_count"], len(full.text_model.state_dict()))


if __name__ == "__main__":
    unittest.main()
