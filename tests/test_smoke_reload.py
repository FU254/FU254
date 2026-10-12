"""Reload checks retain exact weights while allowing CUDA rounding noise."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.train import _verify_smoke_reload


class SavedAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25, 0.5]))

    def trainable_state_dict(self):
        return {"weight": self.weight.detach().cpu()}


class SmokeReloadTests(unittest.TestCase):
    def setUp(self):
        self.model = SavedAdapter()
        self.saved = {name: value.clone()
                      for name, value in self.model.trainable_state_dict().items()}
        self.before = torch.tensor([0.0, 0.1, -0.1, 0.4])

    def verify(self, after, *, device="cuda:4", saved=None):
        # The device argument selects the numerical policy; tensors stay on CPU
        # so CI can exercise the real CUDA log's error without requiring a GPU.
        return _verify_smoke_reload(self.model, self.saved if saved is None else saved,
                                    self.before, after, torch.device(device))

    def test_cuda_near_zero_rounding_noise_is_reported(self):
        after = self.before.clone()
        after[0] = 4.470348358154297e-8
        report = self.verify(after)
        self.assertTrue(report["trainable_state_exact"])
        self.assertEqual(report["rtol"], 1e-5)
        self.assertEqual(report["atol"], 1e-6)
        self.assertEqual(report["max_abs_diff"], after[0].item())

    def test_cuda_large_prediction_difference_is_rejected(self):
        after = self.before.clone()
        after[0] = 1e-3
        with self.assertRaises(AssertionError):
            self.verify(after)

    def test_non_finite_predictions_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                after = self.before.clone()
                after[0] = value
                with self.assertRaisesRegex(AssertionError, "non-finite"):
                    self.verify(after)

    def test_small_parameter_change_is_rejected_even_with_exact_logits(self):
        with torch.no_grad():
            self.model.weight[0].add_(1e-7)
        with self.assertRaisesRegex(AssertionError, "parameter differs"):
            self.verify(self.before.clone())

    def test_parameter_keys_dtype_and_shape_must_match(self):
        for saved in ({}, {"weight": self.saved["weight"].double()},
                      {"weight": self.saved["weight"].reshape(1, 2)}):
            with self.subTest(saved=saved):
                with self.assertRaises(AssertionError):
                    self.verify(self.before.clone(), saved=saved)

    def test_cpu_predictions_still_require_exact_equality(self):
        report = self.verify(self.before.clone(), device="cpu")
        self.assertEqual((report["rtol"], report["atol"]), (0.0, 0.0))
        self.assertEqual(report["max_abs_diff"], 0.0)
        after = self.before.clone()
        after[0] = 4.470348358154297e-8
        with self.assertRaises(AssertionError):
            self.verify(after, device="cpu")

    def test_output_shape_and_dtype_must_match(self):
        for after in (self.before.reshape(1, 4), self.before.double()):
            with self.subTest(after=after):
                with self.assertRaises(AssertionError):
                    self.verify(after)

    def test_cuda_tolerance_depends_on_output_dtype(self):
        for dtype, tolerances in ((torch.float64, (1e-7, 1e-9)),
                                  (torch.float16, (1e-3, 1e-4)),
                                  (torch.bfloat16, (1e-2, 1e-3))):
            with self.subTest(dtype=dtype):
                self.before = self.before.to(dtype)
                report = self.verify(self.before.clone())
                self.assertEqual((report["rtol"], report["atol"]), tolerances)


if __name__ == "__main__":
    unittest.main()
