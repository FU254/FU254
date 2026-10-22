"""CPU protocol checks using synthetic DICOMs and small trainable models."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

import pandas as pd
import torch
from torch import nn

from foundation_sft import equal_teacher_distillation as equal
from foundation_sft import distillation as single
from foundation_sft import train as supervised
from foundation_sft.equal_teacher_losses import TEACHER_NAMES
from foundation_sft.metrics import aggregate_patient_predictions, select_threshold
from foundation_sft.models.common import sha256_file
from tests import test_single_teacher_distillation as single_fixture


SmallTokenStudent = single_fixture.SmallTokenStudent


class EqualTeacherTrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = single_fixture.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        with redirect_stdout(io.StringIO()):
            original, self.teacher_config, self.split, payload = single.load_teacher(
                fixture.teacher_checkpoint, device="cpu")
        self.teachers = {name: deepcopy(original).requires_grad_(False).eval() for name in TEACHER_NAMES}
        # Different task heads make the three soft targets distinct.
        with torch.no_grad():
            for index, teacher in enumerate(self.teachers.values()):
                list(teacher.head.parameters())[-1].add_(0.1 * index)
        self.before = {name: single._cpu_state(teacher) for name, teacher in self.teachers.items()}
        self.config = deepcopy(fixture.config)
        self.config["student"] = {"mode": "full_finetune"}
        self.config["training"] = single._training_settings(self.teacher_config, self.config)
        self.config["evaluation"] = deepcopy(self.teacher_config["evaluation"])
        self.config["teacher_config"] = deepcopy(self.teacher_config)
        self.config["training"].update(epochs=2, patience=4, batch_size=3, accumulation_steps=2)
        self.anchor = {
            "checkpoint": str(fixture.teacher_checkpoint),
            "checkpoint_sha256": sha256_file(fixture.teacher_checkpoint),
            "cohort_hash": payload["cohort_hash"], "model_metadata": payload["model_metadata"],
            "best_val_patient_roc_auc": payload["best_val_patient_roc_auc"],
            "selection": "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC",
        }
        self.infos = [{**deepcopy(self.anchor), "model": name, "weight": 1 / 3,
                       "feature_dim": self.teachers[name].adapter.feature_dim}
                      for name in TEACHER_NAMES]

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0, msg=key)

    def assert_teachers_unchanged(self):
        for name, teacher in self.teachers.items():
            self.assertFalse(teacher.training)
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in teacher.parameters()))
            self.assert_state_equal(self.before[name], teacher.state_dict())

    def test_batch_adds_hard_loss_once_and_backpropagates_all_independent_projectors(self):
        torch.manual_seed(64)
        student = SmallTokenStudent(self.fixture.student_checkpoint)
        kd = equal._build_loss(student, self.teachers, self.config)
        batch = next(iter(equal._loader(self.split, "train", self.teachers, student, self.config,
                                        torch.Generator().manual_seed(38))))
        criterion = mock.Mock(wraps=nn.BCEWithLogitsLoss())
        values = equal._batch_loss(student, self.teachers, kd, batch, criterion, torch.device("cpu"))
        self.assertEqual(criterion.call_count, 1)
        torch.testing.assert_close(values["total"], values["hard"] + values["kd"])
        torch.testing.assert_close(values["kd"], torch.stack(
            [values[f"{name}_total"] for name in TEACHER_NAMES]).mean())
        values["total"].backward()
        self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            and parameter.grad.abs().sum() > 0 for parameter in student.parameters()))
        for name in TEACHER_NAMES:
            params = list(kd.criteria[name].parameters())
            self.assertTrue(params)
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                and parameter.grad.abs().sum() > 0 for parameter in params))
        self.assert_teachers_unchanged()

    def test_accumulation_matches_sample_mean_for_unequal_batches_and_short_final_window(self):
        class ScalarStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

        student, expected = ScalarStudent(), ScalarStudent()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
        batches = [{"label": torch.tensor(labels)} for labels in ([1.0, 3.0], [5.0], [9.0, 11.0])]

        def loss(model, teachers, kd, batch, criterion, device):
            return {"total": (model.weight - batch["label"]).square().mean()}

        for targets in (torch.cat([batches[0]["label"], batches[1]["label"]]), batches[2]["label"]):
            expected_optimizer.zero_grad(set_to_none=True)
            (expected.weight - targets).square().mean().backward()
            expected_optimizer.step()
        with mock.patch.object(equal, "_batch_loss", side_effect=loss):
            result = equal._train_epoch(student, self.teachers, nn.Identity(), batches, optimizer,
                                        nn.Identity(), torch.device("cpu"),
                                        {"accumulation_steps": 2, "gradient_clip": 1000})
        self.assertEqual(result["optimizer_steps"], 2)
        torch.testing.assert_close(student.weight, expected.weight, rtol=0, atol=1e-7)
        self.assert_teachers_unchanged()

    def test_training_saves_validation_best_projectors_and_evaluates_source_free_student_once(self):
        suite = self.root / "equal_suite"
        suite.mkdir()
        snapshots, projector_snapshots, accesses = [], [], []
        original_epoch = equal._train_epoch
        original_predict = supervised.predict
        original_metrics = equal.binary_metrics

        def capture_epoch(student, teachers, kd, *args):
            result = original_epoch(student, teachers, kd, *args)
            snapshots.append(student.trainable_state_dict())
            projector_snapshots.append(single._cpu_state(kd))
            self.assert_teachers_unchanged()
            return result

        def controlled_validation(*args, **kwargs):
            result = original_metrics(*args, **kwargs)
            # Force selection of epoch one to test a stable best snapshot.
            result["roc_auc"] = (0.9, 0.7)[len(snapshots) - 1]
            return result

        def audit_predict(model, loader, device, **kwargs):
            dataset = getattr(loader.dataset, "base", loader.dataset)
            partition = set(dataset.cohort["split"])
            accesses.append(partition)
            if partition == {"test"}:
                run = suite / "equal_teacher"
                self.assertEqual(selector.call_count, 1)
                self.assertTrue((run / "selection.json").is_file())
                saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(saved["best_val_patient_roc_auc"], 0.9)
                self.assert_state_equal(saved["model"], model.trainable_state_dict())
            return original_predict(model, loader, device, **kwargs)

        with redirect_stdout(io.StringIO()), mock.patch.object(
            single, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ), mock.patch.object(equal, "_train_epoch", side_effect=capture_epoch), mock.patch.object(
            equal, "binary_metrics", side_effect=controlled_validation
        ), mock.patch.object(supervised, "predict", side_effect=audit_predict), mock.patch.object(
            equal, "select_threshold", wraps=select_threshold
        ) as selector, mock.patch("foundation_sft.training_history.save_history_plots"):
            receipt = equal._run_equal(suite, self.config, self.teachers, self.split, self.infos, self.anchor)

        self.assertEqual(accesses, [{"val"}, {"val"}, {"val"}, {"test"}])
        self.assert_teachers_unchanged()
        run = suite / "equal_teacher"
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        self.assert_state_equal(best["model"], snapshots[0])
        self.assert_state_equal(best["distillation_state"], projector_snapshots[0])
        self.assert_state_equal(last["model"], snapshots[1])
        self.assert_state_equal(last["best_model"], snapshots[0])
        self.assert_state_equal(last["best_distillation_state"], projector_snapshots[0])
        self.assertTrue(any(not torch.equal(value, snapshots[1][key]) for key, value in snapshots[0].items()))
        self.assertTrue(any(not torch.equal(value, projector_snapshots[1][key])
                            for key, value in projector_snapshots[0].items()))
        for name in TEACHER_NAMES:
            self.assertTrue(any(key.startswith(f"criteria.{name}.") for key in best["distillation_state"]))
        validation = pd.read_csv(run / "validation_predictions.csv")
        patients = aggregate_patient_predictions(validation)
        expected_threshold = select_threshold(patients.label, patients.probability)
        self.assertAlmostEqual(receipt["threshold"], expected_threshold, places=13)

        check = {"config": self.config, "teacher": self.anchor, "teachers": self.infos,
                 "model_metadata": json.loads((run / "model_metadata.json").read_text()),
                 "shared_view": best["shared_view"], "teacher_views": best["teacher_views"],
                 "initial_state_sha256": receipt["initial_state_sha256"],
                 "cohort_hash": receipt["cohort_hash"]}
        job = {"condition": "equal_teacher", "seed": self.config["seed"], "student_mode": "full_finetune"}
        self.fixture.teacher_checkpoint.unlink()
        self.fixture.student_checkpoint.unlink()
        shutil.rmtree(self.teacher_config["model_dir"])
        with mock.patch.object(single, "build_vit_tiny_student", side_effect=SmallTokenStudent), mock.patch.object(
            equal, "load_teacher", side_effect=AssertionError("Completed reuse must not reconstruct teachers")
        ), mock.patch.object(supervised, "prepare", side_effect=AssertionError("Completed reuse must not rebuild a cohort")):
            restored, saved = single.load_student_checkpoint(run / "best.pt", "cpu")
            self.assert_state_equal(best["model"], restored.trainable_state_dict())
            verified = equal.verify_completed_run(run, check, job)
            self.assertEqual(verified["status"], "completed")
            self.assertEqual(verified["experiment"], equal.EXPERIMENT)
            projector_key = next(iter(best["distillation_state"]))
            for defect in ("missing", "extra", "shape", "dtype", "nonfinite"):
                with self.subTest(defect=defect):
                    corrupted = deepcopy(best)
                    state = corrupted["distillation_state"]
                    if defect == "missing":
                        del state[projector_key]
                    elif defect == "extra":
                        state["unrecorded_projector"] = torch.ones(1)
                    elif defect == "shape":
                        state[projector_key] = state[projector_key].reshape(-1)[:1]
                    elif defect == "dtype":
                        state[projector_key] = state[projector_key].double()
                    else:
                        state[projector_key].reshape(-1)[0] = float("nan")
                    with self.assertRaisesRegex(ValueError, "All three global projectors"):
                        equal.verify_completed_run(run, check, job,
                                                   checkpoint_loader=lambda path: corrupted)
            altered = deepcopy(check)
            altered["teachers"][1]["checkpoint_sha256"] = "another-source"
            with self.assertRaisesRegex(ValueError, "teacher checkpoint provenance"):
                equal.verify_completed_run(run, altered, job)
            (run / "test" / "patient_predictions.csv").unlink()
            with self.assertRaisesRegex(ValueError, "missing test/patient_predictions.csv"):
                equal.verify_completed_run(run, check, job)


if __name__ == "__main__":
    unittest.main()
