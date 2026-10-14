"""Full shared-trainer runs for four real, locally initialized VLM encoders.

Tiny encoders and synthetic DICOM labels validate training/evaluation plumbing,
not medical accuracy. No published weights, text towers, or network are used.
"""

import copy
from contextlib import redirect_stdout
from dataclasses import replace
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.config import _merge
from foundation_sft.metrics import aggregate_patient_predictions, binary_metrics, select_threshold
from foundation_sft.models.common import PreprocessSpec
from foundation_sft.models.chexzero import CheXzeroAdapter, CheXzeroVisionTransformer
from foundation_sft.models.cxr_clip import CXRClipAdapter
from foundation_sft.models import openclip_vlm
from foundation_sft import train as trainer

# Import the module, not its TestCase class, so unittest does not discover its
# tests twice. Only its synthetic-data setup is reused below.
import test_foundation_training as shared_fixture


def _has_verified_vlm_runtime():
    try:
        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("open_clip") is not None
                and importlib.metadata.version("open_clip_torch") == "2.23.0"
                and importlib.metadata.version("timm") == "0.9.12")
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(_has_verified_vlm_runtime(), "Run in the isolated pinned OpenCLIP/timm VLM environment")
class VLMTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        shared_fixture.FoundationTrainingTests.setUp(self)
        self.addCleanup(self.temporary.cleanup)

    def _adapter(self, model_name):
        torch.manual_seed(317)
        provenance = {"model": model_name, "fixture": "locally initialized tiny software-test encoder",
                      "preprocessing_scope": "TEST ONLY: 32-pixel geometry, not the released medical protocol"}
        if model_name == "cxr_clip":
            from transformers import SwinConfig, SwinModel

            encoder = SwinModel(SwinConfig(
                image_size=32, patch_size=2, embed_dim=8, depths=[1, 1, 1, 1],
                num_heads=[1, 2, 4, 8], window_size=2, drop_path_rate=0.0,
            ))
            preprocess = PreprocessSpec(32, mean=(0.5,) * 3, std=(0.5,) * 3,
                                        resize_mode="shortest_edge_center_crop", resize_shortest_edge=32,
                                        interpolation="bilinear")
            return CXRClipAdapter(encoder, preprocess, provenance)
        if model_name == "chexzero":
            encoder = CheXzeroVisionTransformer(input_resolution=32, patch_size=8, width=12,
                                                layers=2, heads=3, output_dim=8)
            adapter = CheXzeroAdapter(encoder, provenance)
            # The released pad320 pathway deliberately only accepts 224px.
            # Override geometry explicitly for this tiny software-only test.
            adapter.preprocess = replace(adapter.preprocess, image_size=32, resize_mode="resize")
            return adapter
        if model_name == "xray_siglip":
            from transformers import SiglipVisionConfig, SiglipVisionModel
            from foundation_sft.models.hf_siglip import HFSiglipAdapter

            encoder = SiglipVisionModel(SiglipVisionConfig(
                image_size=32, patch_size=8, hidden_size=12, intermediate_size=24,
                num_hidden_layers=2, num_attention_heads=3,
            ))
            preprocess = PreprocessSpec(32, mean=(0.5,) * 3, std=(0.5,) * 3,
                                        resize_mode="resize", interpolation="bicubic")
            adapter = HFSiglipAdapter(encoder, preprocess, provenance, normalize=False)
            images = torch.randn(2, 3, 32, 32)
            encoder.eval()
            adapter.eval()
            with torch.no_grad():
                expected = encoder(pixel_values=images)
                features = adapter.forward_features(images).validate()
            torch.testing.assert_close(features.global_features, expected.pooler_output, rtol=0, atol=0)
            torch.testing.assert_close(features.local_features, expected.last_hidden_state, rtol=0, atol=0)
            self.assertEqual(features.grid_shape, (4, 4))
            return adapter
        from open_clip.timm_model import TimmModel
        from timm.models.vision_transformer import VisionTransformer

        self.assertEqual(model_name, "biomedclip")
        release = openclip_vlm.RELEASES[model_name]
        trunk = VisionTransformer(
            img_size=32, patch_size=8, embed_dim=12, depth=2, num_heads=3, num_classes=0,
            class_token=True, global_pool=release["timm_pool"],
        )
        trunk.default_cfg = {"pool_size": None}
        output_dim = 8
        vision = {"timm_model_name": "synthetic-local-tiny-vit", "image_size": 32,
                  "timm_pool": release["timm_pool"], "timm_proj": release["timm_proj"],
                  "timm_model_pretrained": False}
        # Retain OpenCLIP's actual TimmModel and all real pooling/projection;
        # substitute only the large architecture selected by the timm factory.
        with patch("timm.create_model", return_value=trunk) as factory:
            visual = openclip_vlm._create_visual_tower(vision, output_dim)
        self.assertIsInstance(visual, TimmModel)
        self.assertFalse(factory.call_args.kwargs["pretrained"])
        self.assertEqual(visual.trunk.global_pool, release["timm_pool"])
        self.assertIsNotNone(visual.trunk.cls_token)
        self.assertIsInstance(visual.head.proj, nn.Linear)
        preprocess = PreprocessSpec(32, mean=(0.2, 0.3, 0.4), std=(0.5, 0.6, 0.7),
                                    resize_mode="shortest_edge_center_crop", resize_shortest_edge=32)
        adapter = openclip_vlm.OpenCLIPVisionAdapter(
            visual, preprocess, provenance, embed_dim=output_dim,
            prefix_tokens=release["prefix_tokens"], normalize=False,
        )
        images = torch.randn(2, 3, 32, 32)
        visual.eval()
        adapter.eval()
        with torch.no_grad():
            features = adapter.forward_features(images).validate()
            torch.testing.assert_close(features.global_features, visual(images), rtol=0, atol=0)
        self.assertEqual(features.grid_shape, (4, 4))
        return adapter

    def test_four_real_vlm_adapters_complete_lp_and_lora_training_and_exact_reload(self):
        for model_name in ("cxr_clip", "chexzero", "xray_siglip", "biomedclip"):
            for mode in ("linear_probe", "lora"):
                with self.subTest(model=model_name, mode=mode):
                    config = _merge(self.config, {
                        "model": model_name, "mode": mode,
                        "checkpoint_source": "test-fixture:synthetic-" + model_name,
                        "training": {"epochs": 2, "batch_size": 4, "accumulation_steps": 2},
                    })
                    config.pop("model_dir", None)
                    adapter = self._adapter(model_name)
                    clean_adapter = copy.deepcopy(adapter)
                    frozen = [(parameter, parameter.detach().clone()) for parameter in adapter.parameters()]
                    initial, captured = {}, {}
                    original_classifier = trainer._classifier

                    def capture_classifier(*args):
                        classifier = original_classifier(*args)
                        captured["classifier"] = classifier
                        initial.update({name: tensor.clone() for name, tensor in classifier.trainable_state_dict().items()})
                        return classifier

                    with redirect_stdout(io.StringIO()), patch.object(
                            trainer, "_classifier", side_effect=capture_classifier), patch.object(
                            trainer, "select_threshold", wraps=select_threshold) as selector:
                        run = trainer.run_training(config, adapter=adapter)
                    self.assertEqual(selector.call_count, 1)
                    self.assertTrue(all(torch.equal(parameter.detach(), before) for parameter, before in frozen))
                    self.assertTrue(all(parameter.grad is None for parameter, _ in frozen))
                    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(best["config"]["model"], model_name)
                    self.assertEqual(best["model_metadata"]["mode"], mode)
                    self.assertEqual(set(best["model"]), set(initial))
                    self.assertTrue(any(not torch.equal(tensor, initial[name]) for name, tensor in best["model"].items()))
                    if mode == "linear_probe":
                        self.assertEqual(set(best["model"]), {"head.weight", "head.bias"})
                    else:
                        lora = {name: tensor for name, tensor in best["model"].items() if name.endswith("lora_B")}
                        self.assertEqual(len(lora), len(best["model_metadata"]["lora_targets"]))
                        self.assertTrue(any(torch.count_nonzero(tensor).item() > 0 for tensor in lora.values()))
                        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in lora.values()))
                    for name, tensor in captured["classifier"].trainable_state_dict().items():
                        torch.testing.assert_close(tensor, best["model"][name], rtol=0, atol=0)

                    history = json.loads((run / "history.json").read_text())
                    self.assertEqual([row["epoch"] for row in history], [1, 2])
                    self.assertTrue(all(np.isfinite(row["train_loss"]) and row["train_loss"] >= 0
                                        and np.isfinite(row["val_loss"]) and row["val_loss"] >= 0 for row in history))
                    self.assertEqual(best["best_val_patient_roc_auc"], max(row["val_patient_roc_auc"] for row in history))
                    for filename in ("training_loss.png", "validation_auc.png"):
                        self.assertGreater((run / filename).stat().st_size, 1000)

                    validation = pd.read_csv(run / "validation_predictions.csv", float_precision="round_trip")
                    patients = aggregate_patient_predictions(validation)
                    selection = json.loads((run / "selection.json").read_text())
                    self.assertEqual(selection["source"], "validation patient predictions")
                    self.assertEqual(selection["threshold"], best["threshold"])
                    self.assertEqual(select_threshold(patients.label, patients.probability), best["threshold"])
                    np.testing.assert_array_equal(selector.call_args.args[0], patients.label)
                    np.testing.assert_allclose(selector.call_args.args[1], patients.probability, rtol=0, atol=1e-14)

                    predictions = pd.read_csv(run / "test" / "predictions.csv", float_precision="round_trip")
                    patient_predictions = pd.read_csv(run / "test" / "patient_predictions.csv", float_precision="round_trip")
                    metrics = json.loads((run / "test" / "metrics.json").read_text())
                    self.assertFalse(set(validation.patient_id) & set(predictions.patient_id))
                    self.assertTrue(np.isfinite(predictions.probability).all())
                    self.assertTrue(predictions.probability.between(0, 1).all())
                    for name, frame in (("image", predictions), ("patient", patient_predictions)):
                        expected = binary_metrics(frame.label, frame.probability, threshold=best["threshold"])
                        self.assertEqual(metrics[name]["threshold"], best["threshold"])
                        self.assertEqual(sum(metrics[name][field] for field in ("tp", "fp", "tn", "fn")), len(frame))
                        for field in ("tp", "fp", "tn", "fn", "n_samples", "n_positive", "n_negative"):
                            self.assertEqual(metrics[name][field], expected[field])
                    for filename in ("roc_curve.png", "precision_recall_curve.png", "calibration_curve.png"):
                        self.assertGreater((run / "test" / filename).stat().st_size, 1000)

                    # Reload only trainable state into fresh copies of the actual
                    # encoder, checking both strict metadata and every tensor.
                    restored = original_classifier(copy.deepcopy(clean_adapter), config).eval()
                    trainer._load_checkpoint(run / "best.pt", restored, config, best["cohort_hash"])
                    for name, tensor in restored.trainable_state_dict().items():
                        torch.testing.assert_close(tensor, best["model"][name], rtol=0, atol=0)
                    with redirect_stdout(io.StringIO()):
                        split = trainer.prepare(config)
                    reloaded_predictions = trainer.predict(
                        restored, trainer._loader(split, "test", restored.adapter, config), torch.device("cpu"))
                    pd.testing.assert_frame_equal(predictions, reloaded_predictions, check_exact=True)
                    # The public evaluation path must reuse the stored validation
                    # threshold rather than selecting one using test labels.
                    with redirect_stdout(io.StringIO()), patch.object(
                            trainer, "build_adapter", return_value=copy.deepcopy(clean_adapter)), patch.object(
                            trainer, "select_threshold", wraps=select_threshold) as evaluation_selector:
                        evaluation = trainer.evaluate(config, run / "best.pt")
                    self.assertEqual(evaluation_selector.call_count, 0)
                    pd.testing.assert_frame_equal(
                        predictions, pd.read_csv(evaluation / "predictions.csv", float_precision="round_trip"), check_exact=True)
                    self.assertEqual(metrics, json.loads((evaluation / "metrics.json").read_text()))


if __name__ == "__main__":
    unittest.main()
