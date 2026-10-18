"""Additional VLMs: real architecture paths with synthetic local weights/data.

These checks validate software, never medical checkpoint identity or accuracy.
No downloads or text-encoder construction are allowed in adapter/training tests.
"""

import copy
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from foundation_sft.config import load_config, _merge
from foundation_sft.data import _resize_image
from foundation_sft.models.common import FoundationClassifier, PreprocessSpec, load_checkpoint_payload
from foundation_sft.models.extra_vlm_common import strict_tensors
from foundation_sft.models import mgca, biovil_t, medklip
from foundation_sft.pixel_cache import PixelCache, _resize_signature
from foundation_sft.workflow import build_plan


def tiny_adapter(name):
    provenance = {"model": name, "fixture": "synthetic local software test; not medical weights"}
    if name == "mgca":
        encoder = mgca.create_mgca_encoder(img_size=32, patch_size=8, embed_dim=24, depth=2, num_heads=3)
        return mgca.MGCAAdapter(encoder, PreprocessSpec(32), provenance)
    if name == "biovil_t":
        return biovil_t.BioViLTAdapter(biovil_t.BioViLTImageModel(), PreprocessSpec(32), provenance)
    return medklip.MedKLIPAdapter(medklip.MedKLIPVisualModel(), torch.randn(75, 768),
                                  PreprocessSpec(32), provenance)


class ExtraVLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_plans_keep_existing_cohort_and_only_schedule_supported_modes(self):
        reference = load_config("foundation_sft/configs/eva_x.json")
        lp = build_plan("foundation_sft/configs/experiments_vlm_extra.json")
        lora = build_plan("foundation_sft/configs/experiments_vlm_extra_lora.json")
        self.assertEqual({row["model"] for row in lp["jobs"]}, {"cxr_clip", "mgca", "biovil_t", "medklip"})
        self.assertEqual(len(lp["jobs"]), 4)
        self.assertEqual({row["model"] for row in lora["jobs"]}, {"cxr_clip", "mgca"})
        self.assertEqual({row["mode"] for row in lp["jobs"]}, {"linear_probe"})
        self.assertEqual({row["mode"] for row in lora["jobs"]}, {"lora"})
        for row in lp["jobs"] + lora["jobs"]:
            for key in ("data", "split_manifest", "split_seed", "split_ratios"):
                self.assertEqual(row["config"][key], reference[key])
            self.assertEqual(row["config"]["output_root"], "runs/foundation_sft_vlm_extra")
            self.assertFalse(row["config"]["training"]["amp"])
            training = row["config"]["training"]
            self.assertEqual(training["batch_size"] * training["accumulation_steps"], 64)

    def test_real_adapters_features_freezing_gradients_and_exact_parameter_reload(self):
        for name in ("mgca", "biovil_t", "medklip"):
            for mode in (("linear_probe", "lora") if name == "mgca" else ("linear_probe",)):
                with self.subTest(name=name, mode=mode):
                    adapter = tiny_adapter(name)
                    clean_adapter = copy.deepcopy(adapter)
                    frozen = [(p, p.detach().clone()) for p in adapter.parameters()]
                    model = FoundationClassifier(adapter, mode=mode, hidden_dims=[12])
                    model.train()
                    images = torch.randn(2, 3, 32, 32)
                    result = model(images, return_features=True)
                    self.assertEqual(result["logits"].shape, (2,))
                    self.assertEqual(result["local_features"].shape[-1], result["global_features"].shape[-1])
                    self.assertEqual(np.prod(result["grid_shape"]), result["local_features"].shape[1])
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                    result["logits"].square().mean().backward()
                    optimizer.step()
                    self.assertTrue(all(torch.equal(p, old) and p.grad is None for p, old in frozen))
                    if mode == "linear_probe":
                        self.assertEqual(set(model.trainable_state_dict()), {"head.weight", "head.bias"})
                        self.assertTrue(all(not module.training for module in adapter.modules()))
                    else:
                        self.assertEqual(len(model.lora_targets), 2)
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for key, p in model.named_parameters() if "lora_B" in key))
                    model.eval()
                    restored = FoundationClassifier(clean_adapter, mode=mode, hidden_dims=[12]).eval()
                    restored.load_trainable_state_dict(model.trainable_state_dict())
                    with torch.no_grad():
                        torch.testing.assert_close(model(images), restored(images), rtol=1e-6, atol=1e-7)

    def test_mgca_loader_selects_full_official_query_encoder_and_refuses_missing_or_foreign_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mgca.ckpt"
            encoder = mgca.create_mgca_encoder(img_size=32, patch_size=8, embed_dim=24, depth=2, num_heads=3)
            state = {"img_encoder_q.model." + key: value for key, value in encoder.state_dict().items()}
            state.update({"img_encoder_q.global_embed.head.0.weight": torch.ones(2, 24),
                          "img_encoder_q.local_embed.head.0.weight": torch.ones(2, 24, 1),
                          "text_encoder_q.synthetic_unused": torch.ones(1)})
            torch.save({"state_dict": state, "epoch": 3}, path)
            config = {"model": "mgca", "checkpoint_source": mgca.SOURCE, "checkpoint": str(path)}
            with patch.object(mgca, "create_mgca_encoder", side_effect=lambda: copy.deepcopy(encoder)):
                loaded = mgca.build_mgca(config)
                for key, value in encoder.state_dict().items():
                    self.assertTrue(torch.equal(loaded.encoder.state_dict()[key], value))
                del state["img_encoder_q.model.norm.weight"]
                torch.save({"state_dict": state}, path)
                with self.assertRaisesRegex(ValueError, "missing"):
                    mgca.build_mgca(config)
            with self.assertRaisesRegex(ValueError, "generic ViT"):
                mgca.build_mgca({**config, "checkpoint_source": "ImageNet"})

    def test_strict_loading_rejects_shape_dtype_and_nonfinite_values_before_mutation(self):
        layer = nn.Linear(2, 3)
        initial = copy.deepcopy(layer.state_dict())
        for bad in (torch.zeros(4, 2), initial["weight"].double(), initial["weight"] * float("nan")):
            with self.assertRaises(ValueError):
                strict_tensors(layer, {**initial, "weight": bad}, "fixture")
            self.assertTrue(torch.equal(layer.weight, initial["weight"]))

    def test_biovil_single_image_retains_missing_prior_and_skips_temporal_attention(self):
        adapter = tiny_adapter("biovil_t").eval()
        with patch.object(adapter.image_model.encoder.vit_pooler, "forward",
                          side_effect=AssertionError("Temporal attention must not run without previous image")):
            with torch.no_grad():
                image = torch.randn(2, 3, 32, 32)
                native = adapter.image_model.projector.model(adapter.image_model.encoder(image))
                features = adapter.forward_features(image)
            torch.testing.assert_close(features.global_features, native.mean((2, 3)), rtol=0, atol=0)
            torch.testing.assert_close(features.local_features, native.flatten(2).transpose(1, 2), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "bypasses temporal"):
            adapter.lora_target_names()
        with self.assertRaisesRegex(ValueError, "linear_probe"):
            biovil_t.build_biovil_t({"model": "biovil_t", "mode": "lora"})

    def test_medklip_decoder_matches_independent_pre_norm_reference(self):
        layer = medklip.KnowledgeDecoderLayer(dim=12, heads=3, hidden=24).eval()
        queries, memory = torch.randn(5, 2, 12), torch.randn(7, 2, 12)
        normalized = layer.norm1(queries)
        expected = queries + layer.self_attn(normalized, normalized, normalized)[0]
        expected = expected + layer.multihead_attn(layer.norm2(expected), memory, memory)[0]
        expected = expected + layer.linear2(torch.relu(layer.linear1(layer.norm3(expected))))
        torch.testing.assert_close(layer(queries, memory), expected, rtol=0, atol=0)

    def test_medklip_global_uses_knowledge_queries_while_local_features_are_spatial(self):
        adapter = tiny_adapter("medklip").eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            result = adapter.forward_features(images)
            queries = adapter.forward_query_features(images)
        self.assertEqual(queries.shape, (2, 75, 256))
        self.assertEqual(result.local_features.shape, (2, 4, 256))
        self.assertEqual(result.grid_shape, (2, 2))
        torch.testing.assert_close(result.global_features, queries.mean(1), rtol=0, atol=0)
        altered = copy.deepcopy(adapter)
        altered.disease_book.zero_()
        with torch.no_grad():
            self.assertFalse(torch.allclose(result.global_features, altered.forward_features(images).global_features))
        with self.assertRaisesRegex(ValueError, "linear_probe"):
            medklip.build_medklip({"model": "medklip", "mode": "lora"})

    def test_medklip_book_refuses_changed_order_recipe_and_bad_embeddings(self):
        with tempfile.TemporaryDirectory() as folder:
            book, embeddings = Path(folder)/"book.json", Path(folder)/"book.pt"
            names = [f"disease{i}" for i in range(75)]
            book.write_text(json.dumps(dict.fromkeys(names, "synthetic explanation")))
            checksum = hashlib.sha256(book.read_bytes()).hexdigest()
            config = {"model": "medklip", "disease_book": str(book), "disease_embeddings": str(embeddings)}
            payload = {"embeddings": torch.randn(75, 768), "names": names, "book_sha256": checksum,
                       "protocol": medklip.BOOK_PROTOCOL, "text_encoder_source": "emilyalsentzer/Bio_ClinicalBERT",
                       "tokenizer_do_lower_case": True,
                       "text_encoder_files_sha256": {"pytorch_model.bin": "0"*64}}
            with patch.object(medklip, "BOOK_SHA256", checksum):
                torch.save(payload, embeddings)
                result, metadata = medklip.load_book(config)
                self.assertEqual(result.shape, (75, 768))
                for change in ({"names": names[::-1]}, {"protocol": "unknown"}, {"embeddings": torch.ones(74, 768)},
                               {"embeddings": torch.full((75, 768), float("nan"))}):
                    torch.save({**payload, **change}, embeddings)
                    with self.assertRaises(ValueError):
                        medklip.load_book(config)

    def test_book_file_writing_supports_torch20_and_preserves_an_existing_output(self):
        from prepare_medklip_book import _save_book

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"book.pt"
            payload = {"embeddings": torch.randn(75,768), "protocol": medklip.BOOK_PROTOCOL}
            _save_book(payload,path)
            restored = load_checkpoint_payload(path)
            torch.testing.assert_close(restored["embeddings"],payload["embeddings"],rtol=0,atol=0)
            previous = path.read_bytes()
            with self.assertRaises(FileExistsError):
                _save_book({"embeddings": torch.zeros(75,768)},path)
            self.assertEqual(path.read_bytes(),previous)
            self.assertEqual(list(Path(folder).iterdir()),[path])


class ExtraVLMGeometryTests(unittest.TestCase):
    def test_mgca_exactly_matches_author_opencv_pad_and_crop_with_cache_hits(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.dcm"
            source.write_bytes(b"Synthetic source signature only; decoder is an explicit callback")
            cache = PixelCache(Path(folder)/"cache")
            for shape in ((137, 201), (201, 137), (301, 301)):
                pixels = np.random.default_rng(17).random(shape).astype(np.float32)
                height, width = shape
                scale = 256/max(shape)
                dimensions = (int(width*scale), int(height*scale))
                resized = cv2.resize((pixels*255).astype(np.uint8), dimensions, interpolation=cv2.INTER_AREA)
                top, left = (256-resized.shape[0])//2, (256-resized.shape[1])//2
                expected = np.pad(resized, ((top, 256-resized.shape[0]-top),
                                           (left, 256-resized.shape[1]-left)))[16:240,16:240].astype(np.float32)/255
                spec = PreprocessSpec(224, resize_mode="mgca_pad256_center_crop", interpolation="bilinear")
                direct = _resize_image(pixels, spec)
                np.testing.assert_array_equal(direct, expected)
                # Each geometry uses a separate cache source fingerprint.
                source.write_bytes(str(shape).encode())
                first = cache.get(source, spec, lambda: direct)
                hit = cache.get(source, spec, lambda: self.fail("Cache hit must not decode"))
                np.testing.assert_array_equal(first, hit)
                self.assertIn("opencv_version", _resize_signature(spec))

    def test_biovil_geometry_matches_torchvision_and_preserves_legacy_cache_signature(self):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
        pixels = np.random.default_rng(19).random((137, 202)).astype(np.float32)
        spec = PreprocessSpec(448, mean=(0.,)*3, std=(1.,)*3, resize_mode="biovil_center_crop",
                              resize_shortest_edge=512, interpolation="bilinear")
        expected = Compose([Resize(512), CenterCrop(448), ToTensor()])(Image.fromarray(pixels))[0].numpy()
        np.testing.assert_array_equal(_resize_image(pixels, spec), expected)
        self.assertEqual(_resize_signature(PreprocessSpec(224)), {
            "image_size": 224, "resize_mode": "resize", "resize_shortest_edge": None,
            "interpolation": "bicubic", "do_hist_equalization": False})


if __name__ == "__main__":
    unittest.main()
