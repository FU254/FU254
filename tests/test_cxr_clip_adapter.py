"""Actual tiny HF Swin encoders validate CXR-CLIP features and Q/K/V LoRA."""

import copy
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from foundation_sft.models.common import FoundationClassifier, PreprocessSpec, sha256_file
from foundation_sft.models.cxr_clip import (
    CXRClipAdapter, OFFICIAL_ARCHITECTURE, OFFICIAL_CHECKPOINT_URL, OFFICIAL_IMAGE_MODEL_ID,
    VERIFIED_SOURCE_REVISION, _preprocess, _training_config, _visual_state, build_cxr_clip,
)


def training_config():
    return {
        "model": {"name": "clip_custom", "image_encoder": {
            "source": "huggingface", "name": OFFICIAL_IMAGE_MODEL_ID,
            "model_type": "swin", "pretrained": True,
        }},
        "base": {"image_size": 224},
        "data_train": {"mimic_cxr": {}, "chexpert": {}, "chest14": {}},
        "transform": {
            # CLAHE is a training augmentation, not an evaluation operation.
            "train": {"CLAHE": {"clip_limit": 4.0}},
            "valid": {"Resize": {"size": 224}, "CenterCrop": {"size": 224}},
            "test": {"Resize": {"size": 224}, "CenterCrop": {"size": 224}},
        },
    }


class CXRClipMetadataTests(unittest.TestCase):
    def test_eval_protocol_has_correct_geometry_normalization_and_no_equalization(self):
        config = training_config()
        spec, source = _preprocess(config)
        self.assertEqual(asdict(spec), {
            "image_size": 224, "mean": (0.5,) * 3, "std": (0.5,) * 3,
            "resize_mode": "shortest_edge_center_crop", "resize_shortest_edge": 224,
            "interpolation": "bilinear", "do_hist_equalization": False,
        })
        self.assertIn("checkpoint_eval_transforms", source)
        del config["transform"]
        fallback, source = _preprocess(config)
        self.assertEqual(fallback, spec)
        self.assertEqual(source, "verified_official_swin_eval_protocol")

    def test_incompatible_eval_settings_fail_instead_of_guessing(self):
        changes = (
            lambda cfg: cfg["base"].update(image_size=448),
            lambda cfg: cfg["transform"]["valid"].update(CLAHE={"clip_limit": 4}),
            lambda cfg: cfg["transform"]["test"]["Resize"].update(size=256),
            lambda cfg: cfg["transform"]["test"]["Resize"].update(interpolation="bicubic"),
            lambda cfg: cfg["transform"]["test"]["Resize"].update(antialias=False),
            lambda cfg: cfg["data_train"]["mimic_cxr"].update(normalize="imagenet"),
        )
        for change in changes:
            config = training_config()
            change(config)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _preprocess(config)

    def test_wrong_image_architecture_initialization_dataset_or_missing_config_rejected(self):
        changes = (
            lambda cfg: cfg["model"]["image_encoder"].update(name="facebook/dinov2-base"),
            lambda cfg: cfg["model"]["image_encoder"].update(source="torchvision"),
            lambda cfg: cfg["model"]["image_encoder"].update(model_type="vit"),
            lambda cfg: cfg["model"]["image_encoder"].update(pretrained=False),
            lambda cfg: cfg["model"].update(name="finetune_classification"),
            lambda cfg: cfg["data_train"].pop("chest14"),
        )
        for change in changes:
            config = training_config()
            change(config)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _training_config({"config": config})
        with self.assertRaises(ValueError):
            _training_config({})

    def test_extracts_only_the_official_visual_wrapper_and_rejects_bad_layout(self):
        visual = torch.ones(3, 3)
        self.assertEqual(_visual_state({"model": {
            "image_encoder.image_encoder.encoder.weight": visual,
            "text_encoder.weight": torch.zeros(1), "image_projection.weight": torch.zeros(1),
        }}), {"encoder.weight": visual})
        for model in ({}, {"image_encoder.bad.weight": visual},
                      {"module.image_encoder.image_encoder.encoder.weight": visual},
                      {"image_encoder.image_encoder.encoder.weight": "not a tensor"}):
            with self.subTest(model=list(model)), self.assertRaises(ValueError):
                _visual_state({"model": model})

    def test_explicit_generic_checkpoint_source_and_missing_assets_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires the local official"):
            build_cxr_clip({})
        with self.assertRaises(FileNotFoundError):
            build_cxr_clip({"checkpoint": "/nonexistent/cxr-clip/swint_mcc.tar"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tar"
            path.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "checkpoint_source"):
                build_cxr_clip({"checkpoint": str(path), "checkpoint_source": "microsoft/swin-tiny-patch4-window7-224"})


@unittest.skipIf(importlib.util.find_spec("transformers") is None, "Transformers not installed")
class CXRClipSwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import SwinConfig, SwinModel
        cls.SwinConfig, cls.SwinModel = SwinConfig, SwinModel
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tiny_architecture = {
            "image_size": 32, "patch_size": 2, "embed_dim": 8,
            "depths": [1, 1, 1, 1], "num_heads": [1, 2, 4, 8], "window_size": 2,
            "drop_path_rate": 0.0,
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cxr-clip-adapter-")
        self.root = Path(self.temporary.name)
        torch.manual_seed(107)
        self.reference = self.SwinModel(self.SwinConfig(**self.tiny_architecture)).eval()
        self.payload = {
            "config": training_config(),
            "model": {"image_encoder.image_encoder." + name: tensor for name, tensor in self.reference.state_dict().items()},
        }
        # Neither the unused text model nor its projection is constructed.
        self.payload["model"]["text_encoder.unused_weight"] = torch.zeros(1)
        self.payload["model"]["image_projection.unused_weight"] = torch.zeros(1)
        self.checkpoint = self.root / "swint_mcc.tar"
        self.save_payload()
        self.preprocess = PreprocessSpec(32, mean=(0.5,) * 3, std=(0.5,) * 3,
                                         resize_mode="shortest_edge_center_crop", resize_shortest_edge=32,
                                         interpolation="bilinear")

    def tearDown(self):
        self.temporary.cleanup()

    def save_payload(self):
        torch.save(self.payload, self.checkpoint)

    def adapter(self):
        return CXRClipAdapter(copy.deepcopy(self.reference), self.preprocess, {"fixture": True})

    def build_fixture(self):
        with mock.patch("foundation_sft.models.cxr_clip.OFFICIAL_ARCHITECTURE", self.tiny_architecture):
            return build_cxr_clip({"checkpoint": str(self.checkpoint)})

    def test_real_features_use_first_spatial_token_and_preserve_all_patch_tokens(self):
        for shape in ((32, 32), (48, 32), (34, 42)):
            with self.subTest(shape=shape):
                images = torch.randn(2, 3, *shape)
                adapter = self.adapter().eval()
                with torch.no_grad():
                    actual = adapter.forward_features(images).validate()
                    expected = self.reference(pixel_values=images).last_hidden_state
                torch.testing.assert_close(actual.global_features, expected[:, 0], rtol=0, atol=0)
                torch.testing.assert_close(actual.local_features, expected, rtol=0, atol=0)
                self.assertEqual(actual.local_features.shape[1], actual.grid_shape[0] * actual.grid_shape[1])
                self.assertGreater((actual.global_features - expected.mean(dim=1)).abs().sum().item(), 0)
        self.assertEqual(self.adapter().feature_dim, 64)

    def test_real_linear_probe_freezes_entire_encoder_and_only_trains_linear_head(self):
        classifier = FoundationClassifier(self.adapter(), mode="linear_probe").train()
        self.assertIsInstance(classifier.head, nn.Linear)
        self.assertFalse(classifier.adapter.training)
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        torch.nn.functional.binary_cross_entropy_with_logits(classifier(images), torch.tensor([0.0, 1.0])).backward()
        self.assertIsNone(images.grad)
        self.assertTrue(all(parameter.grad is None for parameter in classifier.adapter.parameters()))
        self.assertIsNotNone(classifier.head.weight.grad)
        self.assertEqual({name for name, parameter in classifier.named_parameters() if parameter.requires_grad},
                         {"head.weight", "head.bias"})

    def test_real_lora_gradients_update_save_reload_and_preserve_pretrained_weights(self):
        original_adapter = self.adapter()
        targets = original_adapter.lora_target_names()
        self.assertEqual(len(targets), 12)
        frozen = {name: value.clone() for name, value in original_adapter.state_dict().items()}
        classifier = FoundationClassifier(original_adapter, mode="lora", rank=2, alpha=4, hidden_dims=(16,)).train()
        optimizer = torch.optim.SGD([parameter for parameter in classifier.parameters() if parameter.requires_grad], lr=0.2)
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0.0, 1.0])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(classifier(images), labels).backward()
            gradients = [parameter.grad for name, parameter in classifier.named_parameters() if name.endswith("lora_B")]
            self.assertEqual(len(gradients), 12)
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
            self.assertTrue(all(parameter.grad is None for parameter in classifier.adapter.parameters() if not parameter.requires_grad))
            optimizer.step()
        for name, original in frozen.items():
            transformed = name
            for target in targets:
                if name.startswith(target + "."):
                    transformed = target + ".base." + name[len(target) + 1:]
                    break
            torch.testing.assert_close(classifier.adapter.state_dict()[transformed], original, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(classifier.trainable_state_dict(), stream)
        stream.seek(0)
        restored = FoundationClassifier(self.adapter(), mode="lora", rank=2, alpha=4, hidden_dims=(16,))
        restored.load_trainable_state_dict(torch.load(stream, weights_only=True))
        classifier.eval()
        restored.eval()
        torch.testing.assert_close(restored(images), classifier(images), rtol=0, atol=0)

    def test_real_local_checkpoint_loading_is_strict_and_never_uses_network(self):
        with mock.patch("huggingface_hub.file_download.http_get", side_effect=AssertionError("Unexpected network access")), \
             mock.patch.object(self.SwinModel, "from_pretrained", side_effect=AssertionError("Hub loaders are forbidden")):
            adapter = self.build_fixture().eval()
        for name, value in self.reference.state_dict().items():
            torch.testing.assert_close(adapter.encoder.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(adapter.provenance["checkpoint_sha256"], sha256_file(self.checkpoint))
        self.assertEqual(adapter.provenance["checkpoint_source"], OFFICIAL_CHECKPOINT_URL)
        self.assertEqual(adapter.provenance["verified_source_revision"], VERIFIED_SOURCE_REVISION)
        self.assertEqual(adapter.provenance["image_encoder_source"], OFFICIAL_IMAGE_MODEL_ID)
        self.assertEqual(adapter.provenance["global_pooling"], "official_first_final_spatial_token")
        self.assertFalse(adapter.provenance["has_cls_token"])
        self.assertFalse(adapter.preprocess.do_hist_equalization)

    def test_missing_unexpected_and_wrong_shape_visual_weights_are_rejected(self):
        original = copy.deepcopy(self.payload)
        name = "image_encoder.image_encoder.encoder.layers.0.blocks.0.attention.self.query.weight"
        modifications = (
            lambda payload: payload["model"].pop(name),
            lambda payload: payload["model"].update({"image_encoder.image_encoder.unexpected.weight": torch.zeros(1)}),
            lambda payload: payload["model"].update({name: torch.zeros(1, 1)}),
        )
        for modification in modifications:
            self.payload = copy.deepcopy(original)
            modification(self.payload)
            self.save_payload()
            with self.subTest(modification=modification), self.assertRaisesRegex(ValueError, "incomplete or incompatible"):
                self.build_fixture()

    def test_actual_attention_structure_and_projection_shapes_are_validated(self):
        encoder = copy.deepcopy(self.reference)
        encoder.encoder.layers[0].blocks[0].attention.self.query = nn.Linear(8, 7)
        with self.assertRaisesRegex(ValueError, "Q/K/V"):
            CXRClipAdapter(encoder, self.preprocess)
        encoder = copy.deepcopy(self.reference)
        encoder.encoder.layers[0].blocks = nn.ModuleList([])
        with self.assertRaisesRegex(ValueError, "block count"):
            CXRClipAdapter(encoder, self.preprocess)

    def test_nonfinite_and_integer_floating_parameters_cannot_be_silently_cast(self):
        original = copy.deepcopy(self.payload)
        name = "image_encoder.image_encoder.encoder.layers.0.blocks.0.attention.self.query.weight"
        for replacement in (torch.full_like(original["model"][name], float("nan")),
                            torch.full_like(original["model"][name], float("inf")),
                            original["model"][name].to(torch.int64)):
            self.payload = copy.deepcopy(original)
            self.payload["model"][name] = replacement
            self.save_payload()
            with self.subTest(dtype=replacement.dtype), self.assertRaisesRegex(ValueError, "finite floating-point"):
                self.build_fixture()

    def test_checkpoint_content_cannot_change_between_load_and_provenance_hash(self):
        from foundation_sft.models.common import load_checkpoint_payload

        def change_after_read(path, **kwargs):
            payload = load_checkpoint_payload(path, **kwargs)
            Path(path).write_bytes(b"changed after reading valid tensors")
            return payload

        with mock.patch("foundation_sft.models.common.load_checkpoint_payload", side_effect=change_after_read):
            with self.assertRaisesRegex(ValueError, "checkpoint changed during loading"):
                self.build_fixture()

    def test_canonical_swin_tiny_state_loads_36_actual_qkv_targets_and_7_by_7_features(self):
        encoder = self.SwinModel(self.SwinConfig(**OFFICIAL_ARCHITECTURE)).eval()
        payload = {"config": training_config(), "model": {
            "image_encoder.image_encoder." + name: tensor for name, tensor in encoder.state_dict().items()
        }}
        with mock.patch("foundation_sft.models.common.load_checkpoint_payload", return_value=payload), \
             mock.patch.object(self.SwinModel, "from_pretrained", side_effect=AssertionError("No Hub access")):
            adapter = build_cxr_clip({"checkpoint": str(self.checkpoint)}).eval()
        self.assertEqual(adapter.feature_dim, 768)
        targets = adapter.lora_target_names()
        self.assertEqual(len(targets), 36)
        self.assertEqual(len(set(targets)), 36)
        self.assertTrue(all(isinstance(adapter.get_submodule(name), nn.Linear) for name in targets))
        self.assertEqual([len(stage.blocks) for stage in adapter.encoder.encoder.layers], [2, 2, 6, 2])
        images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            actual = adapter.forward_features(images).validate()
            expected = encoder(pixel_values=images).last_hidden_state
        self.assertEqual(tuple(actual.local_features.shape), (1, 49, 768))
        self.assertEqual(actual.grid_shape, (7, 7))
        torch.testing.assert_close(actual.local_features, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.global_features, expected[:, 0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
