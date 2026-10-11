"""Real tiny HF encoders verify loading, token layout, and LoRA gradients."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from foundation_sft.models.common import FoundationClassifier
from foundation_sft.models.hf_dino import build_hf_dino


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TRANSFORMERS, "Pinned Transformers dependency is optional for EVA-X-only use")
class HFDinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import (
            BitImageProcessor,
            Dinov2Config,
            Dinov2Model,
            Dinov2WithRegistersConfig,
            Dinov2WithRegistersModel,
        )

        cls.temp = tempfile.TemporaryDirectory()
        cls.model_dirs = {}
        cls.references = {}
        torch.manual_seed(71)
        for name, config_type, model_type in (
            ("rad_dino", Dinov2Config, Dinov2Model),
            ("ray_dino", Dinov2WithRegistersConfig, Dinov2WithRegistersModel),
        ):
            options = dict(
                image_size=28,
                patch_size=7,
                hidden_size=24,
                num_hidden_layers=2,
                num_attention_heads=3,
                mlp_ratio=2,
            )
            if name == "ray_dino":
                options["num_register_tokens"] = 4
            config = config_type(**options)
            config._attn_implementation = "eager"
            reference = model_type(config).eval()
            path = Path(cls.temp.name) / name
            reference.save_pretrained(path, safe_serialization=True)
            BitImageProcessor(
                size={"shortest_edge": 32},
                crop_size={"height": 28, "width": 28},
                image_mean=[0.1, 0.2, 0.3],
                image_std=[0.4, 0.5, 0.6],
            ).save_pretrained(path)
            cls.model_dirs[name] = path
            cls.references[name] = reference

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, name):
        return build_hf_dino({
            "model": name,
            "model_dir": str(self.model_dirs[name]),
            # Fixtures test architecture, not medical pretrained weight identity.
            "checkpoint_source": f"test-fixture:{name}",
        })

    def test_exact_checkpoint_features_exclude_cls_and_registers(self):
        # Rectangular inputs verify that the local grid is derived from pixels,
        # rather than assuming sqrt(token_count) or a fixed 224-pixel input.
        images = torch.randn(2, 3, 28, 42)
        for name in self.model_dirs:
            with self.subTest(name=name), mock.patch(
                "huggingface_hub.file_download.http_get",
                side_effect=AssertionError("Local snapshots must not access the network"),
            ):
                adapter = self.build(name).eval()
                with torch.no_grad():
                    features = adapter.forward_features(images).validate()
                    tokens = self.references[name](pixel_values=images).last_hidden_state
                prefix = 5 if name == "ray_dino" else 1
                self.assertEqual(features.grid_shape, (4, 6))
                self.assertEqual(tuple(features.local_features.shape), (2, 24, 24))
                torch.testing.assert_close(features.global_features, tokens[:, 0], rtol=0, atol=0)
                torch.testing.assert_close(features.local_features, tokens[:, prefix:], rtol=0, atol=0)
                self.assertEqual(adapter.preprocess.image_size, 28)
                self.assertEqual(adapter.preprocess.resize_shortest_edge, 32)
                self.assertEqual(adapter.preprocess.resize_mode, "shortest_edge_center_crop")
                self.assertEqual(adapter.preprocess.interpolation, "bicubic")
                self.assertEqual(adapter.preprocess.mean, (0.1, 0.2, 0.3))
                self.assertEqual(adapter.preprocess.std, (0.4, 0.5, 0.6))
                self.assertFalse(adapter.provenance["official_source_verified"])
                self.assertIn("model.safetensors", adapter.provenance["weights_sha256"])

    def test_real_encoder_lora_backward_preserves_frozen_weights(self):
        for name in self.model_dirs:
            with self.subTest(name=name):
                adapter = self.build(name)
                targets = adapter.lora_target_names()
                self.assertEqual(len(targets), 6)
                classifier = FoundationClassifier(
                    adapter, mode="lora", rank=2, alpha=4, hidden_dims=(16, 8)
                )
                images = torch.randn(2, 3, 28, 28)
                logits = classifier(images)
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.tensor([0.0, 1.0])
                ).backward()
                b_gradients = [
                    parameter.grad
                    for parameter_name, parameter in classifier.named_parameters()
                    if parameter_name.endswith("lora_B")
                ]
                self.assertEqual(len(b_gradients), 6)
                self.assertTrue(all(gradient is not None for gradient in b_gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in b_gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in b_gradients), 0)
                self.assertTrue(all(
                    parameter.grad is None
                    for parameter in classifier.adapter.parameters()
                    if not parameter.requires_grad
                ))

    def test_missing_encoder_weight_is_rejected(self):
        from safetensors.torch import load_file, save_file
        import shutil

        path = Path(self.temp.name) / "incomplete"
        shutil.copytree(self.model_dirs["rad_dino"], path)
        weights_file = path / "model.safetensors"
        state = load_file(str(weights_file))
        del state["encoder.layer.0.attention.attention.query.weight"]
        save_file(state, str(weights_file), metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "refusing partially/randomly initialized"):
            build_hf_dino({"model": "rad_dino", "model_dir": str(path)})

    def test_generic_weights_and_missing_ray_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not RAD-DINO or RayDINO"):
            build_hf_dino({
                "model": "rad_dino",
                "model_dir": str(self.model_dirs["rad_dino"]),
                "checkpoint_source": "facebook/dinov2-base",
            })
        with self.assertRaisesRegex(ValueError, "RayDINO requires checkpoint_source"):
            build_hf_dino({"model": "ray_dino", "model_dir": str(self.model_dirs["ray_dino"])})


if __name__ == "__main__":
    unittest.main()
