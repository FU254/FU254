"""Validation loss uses the existing prediction pass and actual sample counts."""

import unittest

import pandas as pd
import torch
from torch import nn

from foundation_sft.train import FocalLoss, predict


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return image[:, 0]


def batch(logits, labels, offset):
    return {
        "image": torch.tensor(logits, dtype=torch.float32)[:, None],
        "label": torch.tensor(labels, dtype=torch.float32),
        "sample_id": [f"s{offset+i}" for i in range(len(labels))],
        "patient_id": [f"p{offset+i}" for i in range(len(labels))],
    }


class ValidationLossTests(unittest.TestCase):
    def test_unequal_batches_have_sample_weighted_loss_in_single_pass(self):
        model = RecordingModel()
        criterion = FocalLoss(gamma=2.0, scale=0.8)
        batches = [batch([0.1, 2.0, -0.5], [0, 1, 0], 0), batch([-3.0], [1], 3)]
        frame, loss = predict(model, batches, torch.device("cpu"), criterion=criterion)
        expected = criterion(torch.tensor([0.1, 2.0, -0.5, -3.0]),
                             torch.tensor([0.0, 1.0, 0.0, 1.0])).item()
        self.assertAlmostEqual(loss, expected, places=7)
        self.assertEqual(model.calls, 2)
        self.assertFalse(model.training)
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.label.tolist(), [0, 1, 0, 1])
        model.calls = 0
        predictions_only = predict(model, batches, torch.device("cpu"))
        self.assertEqual(model.calls, 2)
        pd.testing.assert_frame_equal(frame, predictions_only)

    def test_nonfinite_loss_is_rejected_and_empty_validation_is_explicit(self):
        class InvalidLoss(nn.Module):
            def forward(self, logits, labels):
                return logits.new_tensor(float("nan"))

        with self.assertRaisesRegex(RuntimeError, "Non-finite validation loss"):
            predict(RecordingModel(), [batch([1.0], [1], 0)], torch.device("cpu"), criterion=InvalidLoss())
        with self.assertRaisesRegex(ValueError, "no images"):
            predict(RecordingModel(), [], torch.device("cpu"), criterion=FocalLoss())


if __name__ == "__main__":
    unittest.main()
