"""Actual CPU optimizer/checkpoint coverage on a synthetic medical KD task."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest import mock

import torch

from foundation_sft import distillation as single
from foundation_sft import random_student_distillation as trainer
from tests import test_random_student_training as synthetic


class WeightDecayTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = synthetic.RandomStudentTrainingTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_actual_adamw_candidates_keep_initial_states_and_survive_strict_reload(self):
        fixture = self.fixture
        initial_states, projectors = set(), set()
        optimizer = single._optimizer
        observed = []

        def capture(student, kd, settings):
            value = optimizer(student, kd, settings)
            self.assertIsInstance(value, torch.optim.AdamW)
            self.assertEqual(len(value.param_groups), 2 if kd is None else 3)
            self.assertTrue(all(group["weight_decay"] == settings["weight_decay"] for group in value.param_groups))
            observed.append((kd is not None, settings["weight_decay"]))
            return value

        for decay in (.001, .01):
            for condition in ("supervised", "distillation"):
                config = deepcopy(fixture.config)
                config["training"]["weight_decay"] = decay
                config["training"]["epochs"] = 1
                destination = fixture.root / f"wd_{decay}_{condition}"
                destination.mkdir()
                with redirect_stdout(io.StringIO()), mock.patch.object(single, "_optimizer", side_effect=capture):
                    receipt = trainer._run_condition(condition, destination, fixture.teacher,
                        fixture.split, config, fixture.info)
                run = destination / condition
                payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(payload["config"]["training"]["weight_decay"], decay)
                last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
                self.assertTrue(all(group["weight_decay"] == decay for group in last["optimizer"]["param_groups"]))
                saved = json.loads((run / "config.json").read_text())
                check = {"config": saved, "teacher": fixture.info,
                    "model_metadata": json.loads((run / "model_metadata.json").read_text()),
                    "shared_view": json.loads((run / "shared_view.json").read_text()),
                    "initial_state_sha256": receipt["initial_state_sha256"],
                    "distillation_initial_state_sha256": receipt["distillation_initial_state_sha256"],
                    "cohort_hash": receipt["cohort_hash"]}
                job = {"seed": config["seed"], "condition": condition, "student_mode": "full_finetune"}
                verified = trainer.verify_completed_run(run, check, job)
                self.assertEqual(verified["checkpoint_sha256"], single.sha256_file(run / "best.pt"))
                initial_states.add(receipt["initial_state_sha256"])
                if condition == "distillation":
                    projectors.add(receipt["distillation_initial_state_sha256"])
                fixture.assert_teacher_unchanged()
        self.assertEqual(len(initial_states), 1)
        self.assertEqual(len(projectors), 1)
        self.assertEqual(set(observed), {(False, .001), (True, .001), (False, .01), (True, .01)})


if __name__ == "__main__":
    unittest.main()
