"""Numerical regression tests for scaled, sample-weighted accumulation.

The half-precision test model intentionally casts its linear operation on CPU.
Together with the real CPU GradScaler this exercises overflow handling without
CUDA, pretrained weights, clinical images, or a mocked scaler state machine.
"""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft import train as trainer


class BinaryHead(nn.Module):
    def __init__(self, *, half_linear=False, zero=False):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.half_linear = half_linear
        if zero:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, images):
        if self.half_linear:
            # Parameters remain FP32 as in AMP; backward through the cast must
            # first survive the FP16 linear operation's gradient calculation.
            logits = F.linear(images.half(), self.linear.weight.half(),
                              self.linear.bias.half()).float()
        else:
            logits = self.linear(images)
        return logits.squeeze(-1)


class InvalidBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class InvalidBackwardHead(BinaryHead):
    def __init__(self):
        super().__init__(zero=True)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return InvalidBackward.apply(super().forward(images))


def _batch(count, value=8.0):
    return {"image": torch.full((count, 2), value),
            "label": torch.zeros(count)}


def _disabled_scaler():
    # The CUDA class exists in the supported Torch 2.0 server environment and
    # remains usable without CUDA when disabled.
    return torch.cuda.amp.GradScaler(enabled=False)


def _cpu_scaler():
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        raise unittest.SkipTest("This Torch release has no public CPU GradScaler")
    try:
        scaler = torch.amp.GradScaler("cpu")
    except (TypeError, RuntimeError) as exc:
        raise unittest.SkipTest(f"CPU GradScaler unavailable: {exc}") from exc
    if not scaler.is_enabled():
        raise unittest.SkipTest("CPU GradScaler is disabled in this Torch build")
    return scaler


def _epoch(model, batches, scaler, *, accumulation=8):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = trainer._train_epoch(
        model, batches, optimizer, trainer.FocalLoss(), torch.device("cpu"),
        {"amp": False, "accumulation_steps": accumulation, "gradient_clip": 1e6},
        scaler,
    )
    return result


class AccumulationObjectiveTests(unittest.TestCase):
    def test_unequal_batches_and_short_window_match_full_window_objective(self):
        torch.manual_seed(781)
        actual = BinaryHead()
        reference = copy.deepcopy(actual)
        images = torch.randn(8, 2)
        labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        # Two optimizer windows contain 5 and 3 images. The first includes an
        # unequal middle batch, and the second has fewer batches than requested.
        indices = ((0, 1), (2,), (3, 4), (5,), (6, 7))
        batches = [{"image": images[list(rows)], "label": labels[list(rows)]}
                   for rows in indices]
        objective = trainer.FocalLoss()
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        expected_loss = 0.0
        for rows in (slice(0, 5), slice(5, 8)):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(reference(images[rows]), labels[rows])
            expected_loss += loss.item() * len(labels[rows])
            loss.backward()
            optimizer.step()
        actual_loss = _epoch(actual, batches, _disabled_scaler(), accumulation=3)
        self.assertAlmostEqual(actual_loss, expected_loss / 8, places=7)
        for parameter, expected in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)

    def test_finite_loss_with_invalid_fp32_backward_still_fails(self):
        model = InvalidBackwardHead()
        original = copy.deepcopy(model.state_dict())
        self.assertTrue(torch.isfinite(trainer.FocalLoss()(
            model(_batch(1)["image"]), _batch(1)["label"])).item())
        with self.assertRaisesRegex(RuntimeError, "(?i)(non.?finite|invalid).*grad"):
            _epoch(model, [_batch(1)], _disabled_scaler(), accumulation=1)
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_finite_gradient_entries_with_overflowed_norm_are_rejected(self):
        model = BinaryHead(zero=True)
        original = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "(?i)non.?finite"):
            _epoch(model, [_batch(1, 1e20)], _disabled_scaler(), accumulation=1)
        self.assertTrue(all(torch.isfinite(parameter.grad).all().item()
                            for parameter in model.parameters()))
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_short_window_normalization_overflow_is_rejected_before_update(self):
        model = BinaryHead(zero=True)
        original = copy.deepcopy(model.state_dict())

        def linear_loss(logits, labels):
            return (2.0 * logits).mean()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # The forward loss is zero and the gradient before the final window
        # correction is finite. Restoring the actual one-image objective makes
        # its derivative exceed FP32, so updating parameters would be invalid.
        with self.assertRaisesRegex(RuntimeError, "(?i)non.?finite"):
            trainer._train_epoch(
                model, [_batch(1, 3e38)], optimizer, linear_loss,
                torch.device("cpu"),
                {"amp": False, "accumulation_steps": 16, "gradient_clip": 1e6},
                _disabled_scaler(),
            )
        self.assertFalse(torch.isfinite(model.linear.weight.grad).all().item())
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


class RealGradScalerAccumulationTests(unittest.TestCase):
    def test_initial_scale_updates_half_linear_without_accumulation_overflow(self):
        scaler = _cpu_scaler()
        model = BinaryHead(half_linear=True, zero=True)
        reference = BinaryHead(zero=True)
        batch = _batch(64)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        loss = trainer.FocalLoss()(reference(batch["image"]), batch["label"])
        loss.backward()
        optimizer.step()

        actual_loss = _epoch(model, [_batch(8) for _ in range(8)], scaler)
        self.assertAlmostEqual(actual_loss, loss.item(), places=7)
        self.assertEqual(scaler.get_scale(), 65536.0)
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.isfinite(parameter).all().item())
            torch.testing.assert_close(parameter, expected, rtol=1e-3, atol=1e-4)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_overflow_skips_whole_window_then_recovers_without_stale_gradients(self):
        scaler = _cpu_scaler()
        model = BinaryHead(half_linear=True, zero=True)
        reference = BinaryHead(zero=True)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        valid = _batch(2, 0.5)
        loss = trainer.FocalLoss()(reference(valid["image"]), valid["label"])
        loss.backward()
        optimizer.step()
        output = io.StringIO()
        with redirect_stdout(output):
            actual_loss = _epoch(
                model, [_batch(1, 10000.0), _batch(1, 10000.0),
                        _batch(1, 0.5), _batch(1, 0.5)],
                scaler, accumulation=2,
            )
        self.assertTrue(torch.isfinite(torch.tensor(actual_loss)).item())
        self.assertEqual(scaler.get_scale(), 32768.0)
        self.assertRegex(output.getvalue(), "(?i)(overflow|skip)")
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=1e-3, atol=1e-4)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_scaler_does_not_bypass_nonfinite_norm_for_finite_gradient_entries(self):
        scaler = _cpu_scaler()
        model = BinaryHead(zero=True)
        original = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "(?i)non.?finite"):
            _epoch(model, [_batch(1, 1e20)], scaler, accumulation=1)
        self.assertTrue(all(torch.isfinite(parameter.grad).all().item()
                            for parameter in model.parameters()))
        self.assertEqual(scaler.get_scale(), 65536.0)
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)

    def test_entire_epoch_without_optimizer_update_is_rejected(self):
        scaler = _cpu_scaler()
        model = BinaryHead(half_linear=True, zero=True)
        original = copy.deepcopy(model.state_dict())
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            _epoch(model, [_batch(1, 10000.0)], scaler, accumulation=1)
        self.assertEqual(scaler.get_scale(), 32768.0)
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_persistent_invalid_backward_is_rejected_before_exhausting_epoch(self):
        scaler = _cpu_scaler()
        model = InvalidBackwardHead()
        original = copy.deepcopy(model.state_dict())
        batches = [_batch(1, 0.5) for _ in range(24)]
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            _epoch(model, batches, scaler, accumulation=1)
        self.assertLess(model.calls, len(batches))
        self.assertLess(scaler.get_scale(), 65536.0)
        for name, expected in original.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
