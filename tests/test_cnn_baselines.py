"""Frozen CNN backbones expose native features and only train a linear head."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torchvision.models import densenet121, resnet18

from foundation_sft.cnn_sources import CNN_SOURCES
from foundation_sft.models import build_adapter
from foundation_sft.models.cnn import CNNAdapter, _dense_keys, _moco_encoder, _strict_features
from foundation_sft.models.common import FoundationClassifier, PreprocessSpec
from foundation_sft.workflow import build_plan


class CNNBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cnn-baseline-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_source_types_channels_and_normalization_are_distinct(self):
        self.assertEqual(CNN_SOURCES["densenet121_imagenet"]["input_channels"], 3)
        self.assertEqual(CNN_SOURCES["densenet121_cxr"]["input_channels"], 1)
        self.assertEqual(CNN_SOURCES["resnet18_cxr"]["preprocessing"]["image_size"], 320)
        gray = torch.tensor([0.0, 0.5, 1.0])
        self.assertEqual(((gray - 0.5) / (1 / 2048)).tolist(), [-1024.0, 0.0, 1024.0])
        self.assertNotEqual(CNN_SOURCES["densenet121_imagenet"]["pretraining"],
                            CNN_SOURCES["densenet121_cxr"]["pretraining"])

    def test_three_backbone_variants_freeze_all_parameters_and_batchnorm_buffers(self):
        for architecture, channels in (("densenet121", 3), ("densenet121", 1), ("resnet18", 3)):
            with self.subTest(architecture=architecture, channels=channels):
                encoder = densenet121(weights=None) if architecture == "densenet121" else resnet18(weights=None)
                if architecture == "densenet121":
                    if channels == 1:
                        encoder.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
                    encoder.classifier = nn.Identity()
                else:
                    encoder.fc = nn.Identity()
                adapter = CNNAdapter(encoder, architecture, channels, PreprocessSpec(64), {"test": "synthetic"})
                model = FoundationClassifier(adapter, mode="linear_probe")
                before = {key: value.clone() for key, value in adapter.state_dict().items()}
                model.train()
                self.assertFalse(any(module.training for module in adapter.modules()))
                images = torch.rand(2, 1, 64, 64).repeat(1, 3, 1, 1)
                result = model(images, return_features=True)
                self.assertEqual(result["logits"].shape, (2,))
                self.assertEqual(result["local_features"].shape[-1], adapter.feature_dim)
                self.assertEqual(result["grid_shape"], (2, 2))
                loss = result["logits"].square().mean()
                loss.backward()
                self.assertIsNotNone(model.head.weight.grad)
                self.assertTrue(torch.isfinite(model.head.weight.grad).all())
                self.assertTrue(all(parameter.grad is None for parameter in adapter.parameters()))
                self.assertEqual(set(model.trainable_state_dict()), {"head.weight", "head.bias"})
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                                 adapter.feature_dim + 1)
                for key, original in before.items():
                    torch.testing.assert_close(adapter.state_dict()[key], original, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "linear_probe"):
                    FoundationClassifier(adapter, mode="lora")

    def test_moco_query_encoder_is_loaded_exactly_without_projector_or_other_encoder(self):
        torch.manual_seed(1)
        original = resnet18(weights=None, num_classes=128)
        state = {"module.encoder_q." + key: value for key, value in original.state_dict().items()}
        state["module.encoder_k.conv1.weight"] = torch.zeros_like(original.conv1.weight)
        state["module.queue"] = torch.ones(128, 4)
        path = self.root / "moco.pth.tar"
        torch.save({"epoch": 5, "state_dict": state}, path)
        with mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("No network")):
            adapter = build_adapter({"model": "resnet18_cxr", "mode": "linear_probe", "checkpoint": str(path)})
        for key, value in adapter.encoder.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(adapter.feature_dim, 512)
        self.assertEqual(adapter.preprocess.image_size, 320)
        self.assertNotIn("fc.weight", adapter.encoder.state_dict())
        self.assertIn("CheXpert", adapter.provenance["pretraining"])

    def test_imagenet_state_cannot_be_mislabeled_as_cxr_resnet18(self):
        original = resnet18(weights=None)
        with self.assertRaisesRegex(ValueError, "MoCo"):
            _moco_encoder(original.state_dict())

    def test_legacy_dense_parameter_names_are_mapped_without_duplicate_or_missing_features(self):
        weight = torch.ones(3)
        legacy = {"features.denseblock1.denselayer1.norm.1.weight": weight,
                  "features.conv0.weight": weight}
        converted = _dense_keys(legacy)
        self.assertEqual(set(converted), {"features.denseblock1.denselayer1.norm1.weight", "features.conv0.weight"})
        legacy["features.denseblock1.denselayer1.norm1.weight"] = weight
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _dense_keys(legacy)

    def test_backbone_parameter_or_running_statistics_mismatch_is_rejected(self):
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        for problem in ("missing", "unexpected", "shape", "dtype", "nonfinite", "bn_counter"):
            with self.subTest(problem=problem):
                state = dict(encoder.state_dict())
                if problem == "missing":
                    state.pop("bn1.running_mean")
                elif problem == "unexpected":
                    state["other_parameter"] = torch.ones(1)
                elif problem == "shape":
                    state["conv1.weight"] = state["conv1.weight"][:1]
                elif problem == "dtype":
                    state["conv1.weight"] = state["conv1.weight"].double()
                elif problem == "bn_counter":
                    state.pop("bn1.num_batches_tracked")
                else:
                    state["conv1.weight"] = torch.full_like(state["conv1.weight"], float("nan"))
                with self.assertRaises(ValueError):
                    _strict_features(encoder, state)

    def test_only_legacy_bn_counters_may_be_filled_for_old_imagenet_export(self):
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        state = {key: value for key, value in encoder.state_dict().items() if not key.endswith("num_batches_tracked")}
        _strict_features(encoder, state, allow_legacy_bn_counts=True)
        state.pop("bn1.running_var")
        with self.assertRaises(ValueError):
            _strict_features(encoder, state, allow_legacy_bn_counts=True)

    def test_wrong_dense_source_hash_and_non_linear_mode_fail_before_loading(self):
        path = self.root / "wrong.pth"
        path.write_bytes(b"not the official checkpoint")
        with mock.patch("torch.load", side_effect=AssertionError("Should not deserialize")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_adapter({"model": "densenet121_cxr", "mode": "linear_probe", "checkpoint": str(path)})
            with self.assertRaisesRegex(ValueError, "linear_probe"):
                build_adapter({"model": "densenet121_imagenet", "mode": "lora", "checkpoint": str(path)})

    def test_cnn_experiment_plan_has_three_heads_and_same_shared_split(self):
        plan = build_plan("foundation_sft/configs/experiments_cnn.json")
        self.assertEqual(len(plan["jobs"]), 3)
        self.assertEqual({job["mode"] for job in plan["jobs"]}, {"linear_probe"})
        self.assertEqual({job["config"]["split_manifest"] for job in plan["jobs"]},
                         {"runs/foundation_sft/shared_patient_split.csv"})
        self.assertEqual({job["config"]["split_seed"] for job in plan["jobs"]}, {42})
        self.assertTrue(all(job["config"]["training"]["amp"] is False for job in plan["jobs"]))
        self.assertEqual({job["env"] for job in plan["jobs"]}, {"evax_lora"})


if __name__ == "__main__":
    unittest.main()
