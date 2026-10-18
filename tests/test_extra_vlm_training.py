"""End-to-end LP/LoRA training, validation selection and test reload on fake data."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from foundation_sft.config import _merge
from foundation_sft.metrics import select_threshold
from foundation_sft import train as trainer
import test_foundation_training as shared_fixture
from test_extra_vlm import tiny_adapter


class ExtraVLMTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        shared_fixture.FoundationTrainingTests.setUp(self)
        self.addCleanup(self.temporary.cleanup)

    def test_three_new_models_complete_training_and_reload_identical_test_predictions(self):
        for name, modes in (("mgca", ("linear_probe", "lora")), ("biovil_t", ("linear_probe",)),
                            ("medklip", ("linear_probe",))):
            for mode in modes:
                with self.subTest(model=name, mode=mode):
                    torch.manual_seed(192)
                    adapter = tiny_adapter(name)
                    clean = copy.deepcopy(adapter)
                    frozen = [(p, p.detach().clone()) for p in adapter.parameters()]
                    config = _merge(self.config, {"model": name, "mode": mode,
                                                  "training": {"epochs": 2, "hidden_dims": [12], "amp": False},
                                                  "cache_dir": str(self.root/"cache")})
                    config.pop("model_dir", None)
                    with redirect_stdout(io.StringIO()), patch.object(trainer, "select_threshold", wraps=select_threshold) as selector:
                        run = trainer.run_training(config, adapter=adapter)
                    self.assertEqual(selector.call_count, 1)
                    self.assertTrue(all(torch.equal(p, old) and p.grad is None for p, old in frozen))
                    previous = pd.read_csv(run/"test/predictions.csv")
                    metrics = json.loads((run/"test/metrics.json").read_text())
                    with redirect_stdout(io.StringIO()), patch.object(trainer, "build_adapter", return_value=clean), patch.object(
                            trainer, "select_threshold", side_effect=AssertionError("Test must retain the validation threshold")):
                        output = trainer.evaluate(config, run/"best.pt")
                    pd.testing.assert_frame_equal(previous, pd.read_csv(output/"predictions.csv"), check_exact=True)
                    self.assertEqual(metrics, json.loads((output/"metrics.json").read_text()))
                    metadata = json.loads((run/"model_metadata.json").read_text())
                    self.assertEqual(metadata["pretrained"]["model"], name)
                    for key in ("best.pt", "last.pt", "history.json", "training_loss.png", "validation_auc.png"):
                        self.assertTrue((run/key).is_file())


if __name__ == "__main__":
    unittest.main()
