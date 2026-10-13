"""Release-config checks and real tiny timm visual-only integration tests.

Fixtures are local, randomly initialized software tests, not medical weights.
No complete text encoder, public model weights, or network is needed.
"""

from collections import OrderedDict
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.models.common import FoundationClassifier, PreprocessSpec
from foundation_sft.models import openclip_vlm as vlm

HAS_OPENCLIP = importlib.util.find_spec("open_clip") is not None


def _config_payload(model_name):
    release = vlm.RELEASES[model_name]
    return {
        "model_cfg": {
            "embed_dim": release["embed_dim"],
            "vision_cfg": {key: release[key] for key in
                           ("image_size", "timm_model_name", "timm_pool", "timm_proj")},
            "text_cfg": {"hf_model_name": "UNUSED_TEXT_ENCODER_MUST_NOT_LOAD"},
        },
        "preprocess_cfg": {
            "mean": [0.2, 0.3, 0.4], "std": [0.5, 0.6, 0.7],
            "resize_mode": "squash" if model_name == "xray_siglip" else "shortest",
            "interpolation": "bicubic",
        },
    }


class OpenCLIPReleaseConfigTests(unittest.TestCase):
    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP preprocessing defaults required")
    def test_both_release_architectures_and_declared_preprocessing(self):
        for name in vlm.RELEASES:
            with self.subTest(name=name):
                payload = _config_payload(name)
                release, vision = vlm._release_config({"model": name}, payload)
                spec, record = vlm._preprocess(payload, release, {})
                self.assertEqual(spec.image_size, release["image_size"])
                self.assertEqual(spec.mean, (0.2, 0.3, 0.4))
                self.assertEqual(spec.std, (0.5, 0.6, 0.7))
                self.assertEqual(record["shared_preprocess"], asdict(spec))
                self.assertEqual(vision["timm_model_name"], release["timm_model_name"])
                self.assertFalse(spec.do_hist_equalization)
                self.assertEqual(spec.resize_mode, "resize" if name == "xray_siglip"
                                 else "shortest_edge_center_crop")

    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP visual config defaults required")
    def test_generic_source_or_changed_visual_architecture_is_rejected(self):
        for name in vlm.RELEASES:
            with self.subTest(name=name):
                payload = _config_payload(name)
                with self.assertRaises(ValueError):
                    vlm._release_config({"model": name, "checkpoint_source": "openai/clip"}, payload)
                for key, value in (("image_size", 384), ("timm_model_name", "vit_base_patch16_384"),
                                   ("timm_pool", "avg"), ("timm_proj", "mlp"),
                                   ("timm_model_pretrained", True)):
                    altered = copy.deepcopy(payload)
                    altered["model_cfg"]["vision_cfg"][key] = value
                    with self.subTest(field=key), self.assertRaises(ValueError):
                        vlm._release_config({"model": name}, altered)

    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP visual config defaults required")
    def test_biomedclip_empty_pool_and_omitted_size_use_official_defaults(self):
        from open_clip.model import CLIPVisionCfg

        payload = _config_payload("biomedclip")
        raw = payload["model_cfg"]["vision_cfg"]
        raw.pop("image_size")
        raw.pop("timm_proj")
        raw["timm_pool"] = ""
        release, canonical = vlm._release_config({"model": "biomedclip"}, payload)
        self.assertEqual(canonical, asdict(CLIPVisionCfg(**raw)))
        self.assertEqual(canonical["image_size"], 224)
        self.assertEqual(canonical["timm_proj"], "linear")
        self.assertEqual(canonical["timm_pool"], "")
        self.assertEqual(release["embed_dim"], 512)
        self.assertNotIn("image_size", raw)
        raw.pop("timm_pool")
        with self.assertRaises(ValueError):
            vlm._release_config({"model": "biomedclip"}, payload)
        raw["timm_pool"] = ""
        raw.pop("timm_model_name")
        with self.assertRaises(ValueError):
            vlm._release_config({"model": "biomedclip"}, payload)

    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP preprocessing defaults required")
    def test_missing_fields_inherit_the_same_official_factory_defaults(self):
        from open_clip.transform import PreprocessCfg, merge_preprocess_dict

        payload = _config_payload("biomedclip")
        release = vlm.RELEASES["biomedclip"]
        del payload["preprocess_cfg"]["std"]
        del payload["preprocess_cfg"]["resize_mode"]
        del payload["preprocess_cfg"]["interpolation"]
        spec, record = vlm._preprocess(payload, release, {})
        expected = merge_preprocess_dict(asdict(PreprocessCfg()), payload["preprocess_cfg"])
        expected["size"] = release["image_size"]
        self.assertEqual(record["effective_openclip_preprocess"], expected)
        self.assertEqual(spec.std, expected["std"])
        self.assertEqual(record["inherited_library_defaults"]["std"], expected["std"])
        self.assertEqual(record["defaults_library_version"], "2.23.0")
        self.assertEqual(spec.resize_mode, "shortest_edge_center_crop")
        self.assertNotIn("mean", record["inherited_library_defaults"])
        override = {"preprocess_override": {"std": [0.8, 0.9, 1.0]}}
        with self.assertRaisesRegex(ValueError, "research choice"):
            vlm._preprocess(payload, release, override)
        override["preprocess_override_note"] = "Synthetic fixture; explicit test statistics"
        spec, record = vlm._preprocess(payload, release, override)
        self.assertEqual(spec.std, (0.8, 0.9, 1.0))
        self.assertEqual(record["research_choice_note"], override["preprocess_override_note"])
        override["preprocess_override"]["mean"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "Do not override published"):
            vlm._preprocess(payload, release, override)

    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP preprocessing defaults required")
    def test_invalid_preprocessing_is_rejected(self):
        cases = ({"std": [0.0, 1.0, 1.0]}, {"mean": [float("nan"), 0.0, 0.0]},
                 {"resize_mode": "longest"}, {"interpolation": "random"},
                 {"size": 128}, {"mean": [0.0]}, {"std": [True, 1.0, 1.0]})
        for settings in cases:
            payload = _config_payload("xray_siglip")
            payload["preprocess_cfg"].update(settings)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                vlm._preprocess(payload, vlm.RELEASES["xray_siglip"], {})

    def test_missing_complete_published_preprocessor_is_rejected(self):
        for value in (None, [], "unknown"):
            payload = _config_payload("xray_siglip")
            payload["preprocess_cfg"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                vlm._preprocess(payload, vlm.RELEASES["xray_siglip"], {})

    @unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP visual config defaults required")
    def test_patch_dropout_and_unknown_model_options_are_rejected(self):
        for options in ({"patch_dropout": 0.1}, {"unsupported_option": True}):
            payload = _config_payload("biomedclip")
            payload["model_cfg"]["vision_cfg"].update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                vlm._release_config({"model": "biomedclip"}, payload)

    def test_missing_snapshot_does_not_attempt_a_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
                "urllib.request.urlopen", side_effect=AssertionError("Network prohibited")):
            with self.assertRaises(FileNotFoundError):
                vlm.build_openclip_vlm({"model": "biomedclip", "model_dir": temporary})


@unittest.skipUnless(HAS_OPENCLIP, "Pinned OpenCLIP/timm environment required")
class OpenCLIPVisualIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="openclip-vlm-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _visual(self, model_name):
        from timm.models.vision_transformer import VisionTransformer

        class TinyVisual(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = VisionTransformer(
                    img_size=8, patch_size=4, embed_dim=12, depth=2,
                    num_heads=3, num_classes=0,
                    class_token=model_name == "biomedclip",
                    global_pool="token" if model_name == "biomedclip" else "map",
                )
                self.trunk.default_cfg = {"pool_size": None}
                self.head = nn.Sequential(OrderedDict([
                    ("drop", nn.Dropout(0.0)), ("proj", nn.Linear(12, 8, bias=False)),
                ])) if model_name == "biomedclip" else nn.Sequential()

            def forward(self, images):
                return self.head(self.trunk(images))

        torch.manual_seed(391)
        return TinyVisual()

    def _adapter(self, name, visual=None, normalize=False):
        return vlm.OpenCLIPVisionAdapter(
            self._visual(name) if visual is None else visual,
            PreprocessSpec(8, mean=(0.2, 0.3, 0.4), std=(0.5, 0.6, 0.7)),
            {"source": "synthetic software fixture"},
            embed_dim=8 if name == "biomedclip" else 12,
            prefix_tokens=1 if name == "biomedclip" else 0, normalize=normalize,
        )

    def _checkpoint(self, visual, suffix=".bin", mutate=None):
        state = {"visual." + name: value.detach().clone() for name, value in visual.state_dict().items()}
        state["text.unused.weight"] = torch.zeros(1)
        state["logit_scale"] = torch.tensor(1.0)
        if mutate:
            mutate(state)
        path = self.root / ("open_clip_pytorch_model" + suffix)
        if suffix == ".safetensors":
            from safetensors.torch import save_file
            save_file(state, str(path))
        else:
            torch.save(state, path)
        return path

    def test_native_pooling_projection_and_spatial_tokens_are_preserved(self):
        images = torch.randn(2, 3, 8, 8)
        for name in vlm.RELEASES:
            with self.subTest(name=name):
                visual = self._visual(name).eval()
                adapter = self._adapter(name, visual).eval()
                with torch.no_grad():
                    expected_global = visual(images)
                    tokens = visual.trunk.forward_features(images)
                    local = tokens[:, 1:] if name == "biomedclip" else tokens
                    expected_local = visual.head.proj(local) if name == "biomedclip" else local
                    features = adapter.forward_features(images)
                torch.testing.assert_close(features.global_features, expected_global, rtol=0, atol=0)
                torch.testing.assert_close(features.local_features, expected_local, rtol=0, atol=0)
                self.assertEqual(features.grid_shape, (2, 2))
                self.assertEqual(features.local_features.shape, (2, 4, adapter.feature_dim))
                self.assertEqual(adapter.lora_target_names(), [
                    "visual.trunk.blocks.0.attn.qkv", "visual.trunk.blocks.1.attn.qkv"])

    def test_optional_embedding_normalization_is_explicit(self):
        for name in vlm.RELEASES:
            with self.subTest(name=name), torch.no_grad():
                features = self._adapter(name, normalize=True).eval().forward_features(torch.randn(2, 3, 8, 8))
                torch.testing.assert_close(features.global_features.norm(dim=-1), torch.ones(2))
                torch.testing.assert_close(features.local_features.norm(dim=-1), torch.ones(2, 4))

    def test_lp_and_lora_update_only_selected_parameters_and_reload_exactly(self):
        images = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0.0, 1.0])
        for name in vlm.RELEASES:
            for mode in ("linear_probe", "lora"):
                with self.subTest(name=name, mode=mode):
                    adapter = self._adapter(name)
                    clean_visual = copy.deepcopy(adapter.visual)
                    classifier = FoundationClassifier(adapter, mode=mode, rank=2, alpha=4.0, hidden_dims=(5,))
                    frozen = [(parameter, parameter.detach().clone()) for parameter in classifier.parameters()
                              if not parameter.requires_grad]
                    optimizer = torch.optim.SGD([p for p in classifier.parameters() if p.requires_grad], lr=0.2)
                    loss = nn.functional.binary_cross_entropy_with_logits(classifier(images), labels)
                    loss.backward()
                    self.assertTrue(all(parameter.grad is None for parameter, _ in frozen))
                    gradients = [p.grad for p in classifier.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all().item() for g in gradients))
                    if mode == "lora":
                        self.assertTrue(any(torch.count_nonzero(p.grad).item() for key, p in classifier.named_parameters()
                                            if key.endswith("lora_B")))
                    optimizer.step()
                    self.assertTrue(all(torch.equal(p, before) for p, before in frozen))
                    classifier.eval()
                    state = {key: tensor.clone() for key, tensor in classifier.trainable_state_dict().items()}
                    path = self.root / "trainable.pt"
                    torch.save(state, path)
                    fresh = FoundationClassifier(self._adapter(name, clean_visual), mode=mode,
                                                 rank=2, alpha=4.0, hidden_dims=(5,)).eval()
                    fresh.load_trainable_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    with torch.no_grad():
                        torch.testing.assert_close(classifier(images), fresh(images), rtol=0, atol=0)

    def test_torch_and_safetensors_visual_weights_are_loaded_strictly(self):
        for name in vlm.RELEASES:
            for suffix in (".bin", ".safetensors"):
                with self.subTest(name=name, suffix=suffix):
                    visual = self._visual(name)
                    expected = copy.deepcopy(visual.state_dict())
                    path = self._checkpoint(visual, suffix)
                    with torch.no_grad():
                        for parameter in visual.parameters():
                            parameter.zero_()
                    record = vlm._load_visual_weights(visual, path)
                    self.assertEqual(record["ignored_text_key_count"], 1)
                    self.assertEqual(record["ignored_contrastive_scalars"], ["logit_scale"])
                    for key, value in visual.state_dict().items():
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_missing_projection_unknown_keys_and_invalid_weights_are_rejected(self):
        visual = self._visual("biomedclip")
        cases = {
            "projection": lambda state: state.pop("visual.head.proj.weight"),
            "unknown": lambda state: state.update({"unknown.weight": torch.zeros(1)}),
            "shape": lambda state: state.update({"visual.head.proj.weight": torch.zeros(7, 12)}),
            "nonfinite": lambda state: state.update({"visual.head.proj.weight": torch.full((8, 12), float("nan"))}),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                path = self._checkpoint(visual, mutate=mutate)
                with self.assertRaises(ValueError):
                    vlm._load_visual_weights(visual, path)

    def test_builder_uses_only_local_visual_weights_and_records_provenance(self):
        payload = _config_payload("biomedclip")
        config_path = self.root / "open_clip_config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = self._visual("biomedclip")
        path = self._checkpoint(expected)
        config = {"model": "biomedclip", "model_dir": str(self.root)}
        with patch.object(vlm, "_create_visual_tower", return_value=copy.deepcopy(expected)), patch(
                "urllib.request.urlopen", side_effect=AssertionError("Network prohibited")), patch(
                "transformers.AutoModel.from_pretrained", side_effect=AssertionError("Text model prohibited")):
            adapter = vlm.build_openclip_vlm(config)
        self.assertFalse(adapter.provenance["text_tower_constructed"])
        self.assertEqual(adapter.provenance["checkpoint_source"], vlm.RELEASES["biomedclip"]["source"])
        self.assertEqual(adapter.provenance["checkpoint_sha256"], vlm.sha256_file(path))
        self.assertEqual(adapter.provenance["config_sha256"], vlm.sha256_file(config_path))
        for key, value in adapter.visual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            vlm.build_openclip_vlm(dict(config, expected_checkpoint_sha256="0" * 64))

    def test_biomedclip_official_style_config_keeps_native_token_pooling(self):
        payload = _config_payload("biomedclip")
        raw = payload["model_cfg"]["vision_cfg"]
        raw.pop("image_size")
        raw.pop("timm_proj")
        raw["timm_pool"] = ""
        (self.root / "open_clip_config.json").write_text(json.dumps(payload), encoding="utf-8")
        visual = self._visual("biomedclip")
        path = self._checkpoint(visual)

        # The builder test below uses an explicit replacement which delegates
        # to the real visual-only factory instead of the complete text model.
        original_factory = vlm._create_visual_tower

        def build_tiny(vision, embed_dim):
            with patch("timm.create_model", return_value=copy.deepcopy(visual.trunk)):
                return original_factory(vision, 8)

        with patch.object(vlm, "_create_visual_tower", side_effect=build_tiny):
            adapter = vlm.build_openclip_vlm({"model": "biomedclip", "model_dir": str(self.root)})
        self.assertEqual(adapter.visual.trunk.global_pool, "token")
        self.assertEqual(adapter.provenance["raw_vision_cfg"]["timm_pool"], "")
        self.assertNotIn("image_size", adapter.provenance["raw_vision_cfg"])
        self.assertEqual(adapter.provenance["inherited_vision_defaults"]["image_size"], 224)
        self.assertEqual(adapter.provenance["checkpoint_sha256"], vlm.sha256_file(path))

    def test_visual_tower_creation_does_not_construct_a_text_model(self):
        from open_clip.timm_model import TimmModel
        # Use the actual OpenCLIP visual wrapper and actual tiny timm transformer
        # while replacing only the large factory-selected architecture.
        tiny = self._visual("biomedclip").trunk
        vision = {"timm_model_name": "synthetic-local-fixture", "image_size": 8,
                  "timm_pool": "token", "timm_proj": "linear", "timm_model_pretrained": False}
        with patch("timm.create_model", return_value=tiny) as factory, patch(
                "transformers.AutoModel.from_pretrained", side_effect=AssertionError("Text model prohibited")):
            visual = vlm._create_visual_tower(vision, 8)
        self.assertIsInstance(visual, TimmModel)
        self.assertFalse(factory.call_args.kwargs["pretrained"])
        with torch.no_grad():
            self.assertEqual(visual(torch.randn(2, 3, 8, 8)).shape, (2, 8))

    def test_full_canonical_visual_towers_construct_without_text_or_weight_downloads(self):
        for name in vlm.RELEASES:
            payload = _config_payload(name)
            if name == "biomedclip":
                payload["model_cfg"]["vision_cfg"].pop("image_size")
                payload["model_cfg"]["vision_cfg"]["timm_pool"] = ""
            release, vision = vlm._release_config({"model": name}, payload)
            with self.subTest(name=name), patch(
                    "transformers.AutoModel.from_pretrained", side_effect=AssertionError("Text model prohibited")), patch(
                    "huggingface_hub.hf_hub_download", side_effect=AssertionError("Download prohibited")):
                visual = vlm._create_visual_tower(vision, release["embed_dim"])
                self.assertEqual(visual.trunk.global_pool, release["timm_pool"])
                self.assertEqual(visual.trunk.num_prefix_tokens, release["prefix_tokens"])
                self.assertEqual(len(visual.trunk.blocks), 12)
                self.assertEqual(visual.trunk.patch_embed.img_size, (release["image_size"],) * 2)
                self.assertEqual(visual.trunk.blocks[0].attn.qkv.weight.shape, (2304, 768))
                keys = set(visual.state_dict())
                if name == "biomedclip":
                    self.assertEqual(visual.head.proj.weight.shape, (512, 768))
                    self.assertIn("head.proj.weight", keys)
                    self.assertIn("trunk.cls_token", keys)
                else:
                    self.assertIn("trunk.attn_pool.q.weight", keys)
                    self.assertNotIn("trunk.cls_token", keys)
                    self.assertNotIn("head.proj.weight", keys)
                del visual


if __name__ == "__main__":
    unittest.main()
