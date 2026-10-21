"""Separate exact checkpoint identity from bounded FP32 inference roundoff."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as fixtures

from foundation_sft import distillation


REPORTED_ERROR = 1.3113021850585938e-6


class StateWithUnusedBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("unused_counter", torch.zeros(1, dtype=torch.int64))

    def forward(self, values):
        return values * self.weight


class DistillationRoundtripTests(unittest.TestCase):
    def test_reported_near_zero_difference_passes_while_old_tolerance_fails(self):
        expected = torch.tensor([0.024039], dtype=torch.float32)
        actual = expected + REPORTED_ERROR
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        result = distillation._verify_roundtrip_outputs(expected, actual, batch_size=1)
        self.assertEqual(result, {
            "roundtrip_max_abs_error": REPORTED_ERROR,
            "roundtrip_rtol": 1e-4,
            "roundtrip_atol": 1e-5,
        })
        relative_error = result["roundtrip_max_abs_error"] / abs(expected.item())
        self.assertAlmostEqual(relative_error, 5.455e-5, delta=1e-8)

    def test_larger_logit_shift_is_rejected(self):
        expected = torch.tensor([0.024039, -0.01], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            distillation._verify_roundtrip_outputs(expected, expected + 1e-3, batch_size=2)

    def test_nonfinite_logits_are_rejected_even_when_both_sides_match(self):
        good = torch.tensor([0.024039], dtype=torch.float32)
        for value in (float("nan"), float("inf"), float("-inf")):
            bad = torch.tensor([value], dtype=torch.float32)
            for placement, expected, actual in (("both", bad, bad), ("expected", bad, good), ("actual", good, bad)):
                with self.subTest(value=value, placement=placement), self.assertRaises(ValueError):
                    distillation._verify_roundtrip_outputs(expected, actual, batch_size=1)

    def test_shape_batch_size_and_dtype_checks_remain_strict(self):
        cases = [
            (torch.zeros(2, 1), torch.zeros(2, 1), 2),
            (torch.zeros(2), torch.zeros(3), 2),
            (torch.zeros(2), torch.zeros(2), 3),
            (torch.tensor(0.0), torch.tensor(0.0), 1),
            (torch.zeros(2), torch.zeros(2), 0),
            (torch.zeros(2), torch.zeros(2), -1),
            (torch.zeros(1), torch.zeros(1), True),
            (torch.zeros(1), torch.zeros(1), 1.0),
        ]
        for expected, actual, batch_size in cases:
            with self.subTest(expected=expected.shape, actual=actual.shape, batch_size=batch_size), self.assertRaises(ValueError):
                distillation._verify_roundtrip_outputs(expected, actual, batch_size=batch_size)
        with self.assertRaises(AssertionError):
            distillation._verify_roundtrip_outputs(torch.zeros(2, dtype=torch.float32),
                                                   torch.zeros(2, dtype=torch.float64), batch_size=2)

    def test_identical_complete_state_returns_matching_sha256_evidence(self):
        student = StateWithUnusedBuffer()
        restored = deepcopy(student)
        expected_hash = distillation._state_hash(student.state_dict())
        result = distillation._verify_roundtrip_state(student, restored)
        self.assertEqual(result, {"student_state_sha256": expected_hash,
                                  "restored_state_sha256": expected_hash,
                                  "student_state_exact": True})
        self.assertRegex(expected_hash, r"^[0-9a-f]{64}$")

    def test_buffer_or_tiny_parameter_change_is_rejected_despite_identical_logits(self):
        student = StateWithUnusedBuffer()
        inputs = torch.zeros(2)
        for changed in ("unused_buffer", "one_ulp_parameter"):
            restored = deepcopy(student)
            with torch.no_grad():
                if changed == "unused_buffer":
                    restored.unused_counter.add_(1)
                else:
                    restored.weight.copy_(torch.nextafter(restored.weight, torch.full_like(restored.weight, 2.0)))
            torch.testing.assert_close(student(inputs), restored(inputs), rtol=0, atol=0)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                distillation._verify_roundtrip_state(student, restored)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Synthetic smoke integration needs pinned Transformers")
class DistillationRoundtripSmokeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SingleTeacherDistillationTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def test_real_smoke_accepts_small_output_drift_after_exact_state_check(self):
        original_load = distillation.load_student_checkpoint
        original_verify = distillation._verify_roundtrip_state
        verified, recorded_evidence = set(), []

        def near_zero_student(*args, **kwargs):
            student = fixtures.SmallTokenStudent(*args, **kwargs)
            # The actual source/reloaded head has identical state. A constant
            # near-zero logit reproduces the previous absolute tolerance miss;
            # the real global/local KD paths still supply backbone gradients.
            with torch.no_grad():
                student.encoder.head.weight.zero_()
                student.encoder.head.bias.fill_(0.024039)
            return student

        def verify_before_forward(student, restored):
            evidence = original_verify(student, restored)
            verified.add(id(restored))
            recorded_evidence.append(evidence)
            return evidence

        def load_with_bounded_forward_drift(*args, **kwargs):
            restored, payload = original_load(*args, **kwargs)
            native_forward = restored.forward

            def forward(images, return_features=False):
                self.assertIn(id(restored), verified, "State identity must be checked before restored inference")
                logits = native_forward(images, return_features=return_features)
                self.assertFalse(return_features)
                return logits + REPORTED_ERROR

            restored.forward = forward
            return restored, payload

        with redirect_stdout(io.StringIO()), mock.patch.object(
            distillation, "build_vit_tiny_student", side_effect=near_zero_student
        ), mock.patch.object(
            distillation, "load_student_checkpoint", side_effect=load_with_bounded_forward_drift
        ) as load, mock.patch.object(
            distillation, "_verify_roundtrip_state", side_effect=verify_before_forward
        ) as verify:
            directory = Path(distillation.check(self.fixture.config, project=self.fixture.root))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(verify.call_count, 1)
        result = json.loads((directory / "check.json").read_text())
        self.assertTrue(result["check_passed"])
        self.assertTrue(result["student_full_checkpoint_roundtrip"])
        self.assertTrue(result["teacher_unchanged"])
        self.assertFalse(result["training_started"])
        self.assertEqual(result["roundtrip_max_abs_error"], REPORTED_ERROR)
        self.assertEqual(result["roundtrip_rtol"], 1e-4)
        self.assertEqual(result["roundtrip_atol"], 1e-5)
        for key, value in recorded_evidence[0].items():
            self.assertEqual(result[key], value)
        self.assertFalse((directory / "roundtrip.pt").exists())


if __name__ == "__main__":
    unittest.main()
