"""Random-initialization A/B protocol using synthetic DICOMs and a tiny teacher."""

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
from torch.utils.data import SequentialSampler

from foundation_sft import distillation as single
from foundation_sft import random_student_distillation as random_kd
from foundation_sft import train as supervised
from foundation_sft.metrics import aggregate_patient_predictions, select_threshold
from tests import test_single_teacher_distillation as fixture_module


class SmallRandomStudent(fixture_module.SmallTokenStudent):
    """Exercise complete random student metadata without a large native model."""

    def __init__(self, checkpoint=None, *, initialization=None, random_seed=None):
        if random_seed is not None:
            if checkpoint is not None or initialization is not None:
                raise ValueError("Synthetic random student takes one initialization")
            initialization = {"architecture": "synthetic-six-channel-student", "pretraining": "none",
                              "initialization_mode": "random", "seed": random_seed,
                              "source_num_classes": 1, "variant": "single_cls"}
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(random_seed)
                super().__init__(initialization=initialization)
        else:
            if checkpoint is not None or initialization is None:
                raise ValueError("Synthetic restore needs saved metadata and no source file")
            super().__init__(initialization=initialization)


class RandomStudentTrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        self.raw = deepcopy(fixture.config)
        self.raw.update(student_checkpoint=None, student_initialization="random",
                        student={"mode": "full_finetune"}, output_root=str(self.root / "random_runs"))
        settings = deepcopy(self.raw["distillation"])
        self.loss_patch = mock.patch.object(random_kd, "LOSS_SETTINGS", settings)
        self.loss_patch.start()
        self.addCleanup(self.loss_patch.stop)
        self.factory_patch = mock.patch.object(random_kd, "build_vit_tiny_student", side_effect=SmallRandomStudent)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.restore_patch = mock.patch.object(single, "build_vit_tiny_student", side_effect=SmallRandomStudent)
        self.restore_patch.start()
        self.addCleanup(self.restore_patch.stop)
        with redirect_stdout(io.StringIO()):
            self.config, self.teacher, self.split, self.info = random_kd._prepare(self.raw, self.root)
        self.before_teacher = single._cpu_state(self.teacher)

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0, msg=name)

    def assert_teacher_unchanged(self, teacher=None):
        teacher = self.teacher if teacher is None else teacher
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assert_state_equal(self.before_teacher, teacher.state_dict())

    def completed_run(self, condition="distillation"):
        suite = self.root / ("suite_" + condition)
        suite.mkdir()
        with redirect_stdout(io.StringIO()):
            receipt = random_kd._run_condition(condition, suite, self.teacher, self.split, self.config, self.info)
        run = suite / condition
        check = {"config": self.config, "teacher": self.info,
                 "model_metadata": json.loads((run / "model_metadata.json").read_text()),
                 "shared_view": json.loads((run / "shared_view.json").read_text()),
                 "initial_state_sha256": receipt["initial_state_sha256"], "cohort_hash": receipt["cohort_hash"]}
        job = {"condition": condition, "seed": self.config["seed"], "student_mode": "full_finetune"}
        return run, receipt, check, job

    def test_prepare_needs_no_imagenet_student_file_and_rejects_changed_protocol_before_load(self):
        self.fixture.student_checkpoint.unlink()
        with redirect_stdout(io.StringIO()):
            resolved, teacher, split, info = random_kd._prepare(self.raw, self.root)
        self.assertIsNone(resolved["student_checkpoint"])
        self.assertEqual(resolved["student_initialization"], "random")
        self.assertEqual(info["cohort_hash"], supervised._split_hash(split))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        defects = [dict(student_initialization="imagenet"),
                   dict(student_checkpoint=str(self.fixture.student_checkpoint)),
                   dict(student={"mode": "lora"}), dict(seed=True), dict(seed=-1),
                   dict(distillation={**self.raw["distillation"], "soft_weight": .6})]
        for changes in defects:
            altered = deepcopy(self.raw)
            altered.update(changes)
            with self.subTest(changes=changes), mock.patch.object(
                single, "load_teacher", side_effect=AssertionError("Reject before restoring a large teacher")
            ), self.assertRaises(ValueError):
                random_kd._prepare(altered, self.root)

    def test_check_uses_only_train_pixels_and_preserves_teacher_and_exact_random_student(self):
        original_loader, original_build = single._loader, random_kd._build_student
        students, restored_teachers = [], []
        original_teacher_load = single.load_teacher
        original_loss = random_kd.SingleTeacherDistillationLoss
        projectors = []

        def training_loader(split, name, *args, **kwargs):
            self.assertEqual(name, "train")
            loader = original_loader(split, name, *args, **kwargs)
            self.assertEqual(set(loader.dataset.base.cohort["split"]), {"train"})
            return loader

        def capture_student(config):
            student = original_build(config)
            students.append(student)
            return student

        def capture_teacher(*args, **kwargs):
            value = original_teacher_load(*args, **kwargs)
            restored_teachers.append(value[0])
            return value

        def capture_loss(*args, **kwargs):
            loss = original_loss(*args, **kwargs)
            projectors.append(loss)
            return loss

        self.fixture.student_checkpoint.unlink()
        with redirect_stdout(io.StringIO()), mock.patch.object(single, "_loader", side_effect=training_loader), mock.patch.object(
            random_kd, "_build_student", side_effect=capture_student
        ), mock.patch.object(single, "load_teacher", side_effect=capture_teacher), mock.patch.object(
            random_kd, "SingleTeacherDistillationLoss", side_effect=capture_loss
        ), mock.patch.object(
            supervised, "predict", side_effect=AssertionError("Smoke does not predict validation/test")
        ):
            directory = random_kd.check(self.raw, self.root)
        record = json.loads((directory / "check.json").read_text())
        self.assertTrue(record["check_passed"])
        self.assertTrue(record["student_state_exact"])
        self.assertTrue(record["student_full_checkpoint_roundtrip"])
        self.assertFalse(record["training_started"])
        self.assertEqual(record["model_metadata"]["initialization"]["seed"], self.config["seed"])
        self.assertFalse(list(directory.glob("*.pt")))
        self.assertFalse((directory / "test").exists())
        self.assertGreater(students[0].encoder.patch_embed.weight.grad.abs().sum().item(), 0)
        self.assertGreater(projectors[-1].projector.weight.grad.abs().sum().item(), 0)
        self.assert_teacher_unchanged(restored_teachers[0])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in record["losses"].values()))

    def test_train_eval_is_unaugmented_sequential_and_preserves_training_rng(self):
        model = random_kd._build_student(self.config)
        train = single._loader(self.split, "train", self.teacher.adapter.preprocess, model.preprocess,
                               self.config, torch.Generator().manual_seed(self.config["seed"]))
        loader = random_kd._train_eval_loader(self.split, train.dataset.metadata(), self.config)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertFalse(loader.dataset.base.training)
        self.assertEqual(loader.dataset.base.rotation_degrees, 0)
        self.assertEqual(loader.dataset.base.horizontal_flip_probability, 0)
        self.assertEqual(set(loader.dataset.base.cohort["split"]), {"train"})
        self.assertIsNot(loader.generator, train.generator)
        torch.manual_seed(489)
        before, generator_before = torch.get_rng_state().clone(), train.generator.get_state().clone()
        first, second = list(loader), list(loader)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(train.generator.get_state(), generator_before, rtol=0, atol=0)
        for a, b in zip(first, second):
            self.assertEqual(a["sample_id"], b["sample_id"])
            torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
            self.assertNotIn("teacher_image", a)

    def test_a_and_b_share_random_initial_weights_first_training_views_and_seed(self):
        original_epoch = single._train_epoch
        starts, batches = {}, {}

        def capture_epoch(model, teacher, kd, loader, *args):
            condition = "supervised" if kd is None else "distillation"
            if condition not in starts:
                starts[condition] = model.trainable_state_dict()
                generator = getattr(loader, "loader", loader).generator
                rng = supervised._rng_state(generator)
                batch = next(iter(loader))
                batches[condition] = {"ids": batch["sample_id"], "image": batch["image"].clone()}
                supervised._restore_rng(rng, generator)
            return original_epoch(model, teacher, kd, loader, *args)

        suite = self.root / "paired"
        suite.mkdir()
        with redirect_stdout(io.StringIO()), mock.patch.object(single, "_train_epoch", side_effect=capture_epoch):
            a = random_kd._run_condition("supervised", suite, self.teacher, self.split, self.config, self.info)
            b = random_kd._run_condition("distillation", suite, self.teacher, self.split, self.config, self.info,
                                        expected_initial=a["initial_state_sha256"])
        self.assertEqual(a["initial_state_sha256"], b["initial_state_sha256"])
        self.assert_state_equal(starts["supervised"], starts["distillation"])
        self.assertEqual(batches["supervised"]["ids"], batches["distillation"]["ids"])
        torch.testing.assert_close(batches["supervised"]["image"], batches["distillation"]["image"], rtol=0, atol=0)
        self.assert_teacher_unchanged()

    def test_validation_best_student_and_projector_are_reloaded_before_one_test_evaluation(self):
        suite = self.root / "best_suite"
        suite.mkdir()
        snapshots, projector_snapshots, accesses, actual_eval_losses = [], [], [], []
        original_epoch, original_predict, original_metrics = single._train_epoch, supervised.predict, random_kd.binary_metrics
        metrics_calls = 0

        def capture_epoch(model, teacher, kd, *args):
            result = original_epoch(model, teacher, kd, *args)
            snapshots.append(model.trainable_state_dict())
            projector_snapshots.append(single._cpu_state(kd))
            return result

        def controlled_metrics(*args, **kwargs):
            nonlocal metrics_calls
            result = original_metrics(*args, **kwargs)
            metrics_calls += 1
            if metrics_calls % 2 == 0:
                result["roc_auc"] = (.9, .7)[len(snapshots) - 1]
            return result

        def audit_prediction(model, loader, device, **kwargs):
            partition = set(loader.dataset.base.cohort["split"])
            accesses.append(partition)
            if partition == {"test"}:
                self.assertEqual(selector.call_count, 1)
                run = suite / "distillation"
                self.assertTrue((run / "selection.json").is_file())
                saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(saved["best_val_patient_roc_auc"], .9)
                self.assert_state_equal(saved["model"], model.state_dict())
            value = original_predict(model, loader, device, **kwargs)
            if partition == {"train"}:
                self.assertFalse(loader.dataset.base.training)
                actual_eval_losses.append(value[1])
            return value

        with redirect_stdout(io.StringIO()), mock.patch.object(single, "_train_epoch", side_effect=capture_epoch), mock.patch.object(
            supervised, "predict", side_effect=audit_prediction
        ), mock.patch.object(random_kd, "binary_metrics", side_effect=controlled_metrics), mock.patch.object(
            random_kd, "select_threshold", wraps=select_threshold
        ) as selector:
            receipt = random_kd._run_condition("distillation", suite, self.teacher, self.split, self.config, self.info)
        self.assertEqual(accesses, [{"train"}, {"val"}, {"train"}, {"val"}, {"val"}, {"test"}])
        run = suite / "distillation"
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        self.assert_state_equal(best["model"], snapshots[0])
        self.assert_state_equal(best["distillation_state"], projector_snapshots[0])
        self.assert_state_equal(last["model"], snapshots[1])
        self.assert_state_equal(last["best_model"], snapshots[0])
        self.assert_state_equal(last["best_distillation_state"], projector_snapshots[0])
        history = json.loads((run / "history.json").read_text())
        self.assertEqual([row["train_eval_loss"] for row in history], actual_eval_losses)
        self.assertTrue(all(row["train_eval_patient_roc_auc"] is not None for row in history))
        self.assertTrue(all(row["train_loss"] > row["train_hard_loss"] for row in history))
        plot_meta = json.loads((run / "history_plot_metadata.json").read_text())
        self.assertIn("train_eval_loss", plot_meta["training_loss"])
        validation = pd.read_csv(run / "validation_predictions.csv")
        patients = aggregate_patient_predictions(validation)
        self.assertAlmostEqual(receipt["threshold"], select_threshold(patients.label, patients.probability), places=13)
        self.assert_teacher_unchanged()
        self.fixture.student_checkpoint.unlink()
        self.fixture.teacher_checkpoint.unlink()
        shutil.rmtree(self.config["teacher_config"]["model_dir"])
        with mock.patch.object(single, "load_teacher", side_effect=AssertionError("Inference needs no teacher")):
            restored, payload = single.load_student_checkpoint(run / "best.pt", "cpu")
        self.assert_state_equal(restored.state_dict(), best["model"])
        self.assertEqual(payload["student_initialization"], "random")

    def test_completed_verification_rejects_initialization_spoofs_and_corrupt_kd_state(self):
        run, receipt, check, job = self.completed_run()
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(random_kd.verify_completed_run(run, check, job)["status"], "completed")
        mutations = [
            ("wrong_initialization", lambda p: p["model_metadata"]["initialization"].update(initialization_mode="imagenet")),
            ("wrong_seed", lambda p: p["model_metadata"]["initialization"].update(seed=self.config["seed"] + 1)),
            ("false_source", lambda p: p["model_metadata"]["initialization"].update(checkpoint="missing-imagenet.pth")),
            ("spoofed_config", lambda p: p["config"].update(student_initialization="imagenet")),
            ("spoofed_marker", lambda p: p.update(student_initialization="imagenet")),
            ("wrong_experiment", lambda p: p.update(experiment="single_teacher_kd_v1")),
        ]
        for name, mutate in mutations:
            corrupted = deepcopy(payload)
            mutate(corrupted)
            with self.subTest(defect=name), self.assertRaises(ValueError):
                random_kd.verify_completed_run(run, check, job, checkpoint_loader=lambda path: corrupted)
        key = next(iter(payload["distillation_state"]))
        for defect in ("missing", "extra", "shape", "dtype", "nonfinite"):
            corrupted = deepcopy(payload)
            state = corrupted["distillation_state"]
            if defect == "missing":
                del state[key]
            elif defect == "extra":
                state["unrecorded_projection"] = torch.ones(1)
            elif defect == "shape":
                state[key] = state[key].reshape(-1)[:1]
            elif defect == "dtype":
                state[key] = state[key].double()
            else:
                state[key].reshape(-1)[0] = float("nan")
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, "global projector"):
                random_kd.verify_completed_run(run, check, job, checkpoint_loader=lambda path: corrupted)

    def test_supervised_verification_rejects_kd_state_and_incomplete_evaluation(self):
        run, receipt, check, job = self.completed_run("supervised")
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(random_kd.verify_completed_run(run, check, job)["status"], "completed")
        corrupted = deepcopy(payload)
        corrupted["distillation_state"] = {"projector.weight": torch.ones(12, 6)}
        with self.assertRaisesRegex(ValueError, "must not contain"):
            random_kd.verify_completed_run(run, check, job, checkpoint_loader=lambda path: corrupted)
        (run / "test" / "patient_predictions.csv").unlink()
        with self.assertRaisesRegex(ValueError, "patient_predictions.csv"):
            random_kd.verify_completed_run(run, check, job)

    def test_teacher_frozen_and_checkpoint_roundtrip_guards_are_strict(self):
        for defect in ("training", "trainable", "stale_gradient"):
            teacher = deepcopy(self.teacher)
            parameter = next(teacher.parameters())
            if defect == "training":
                teacher.train()
            elif defect == "trainable":
                parameter.requires_grad_(True)
            else:
                parameter.grad = torch.ones_like(parameter)
            with self.subTest(defect=defect), self.assertRaisesRegex(RuntimeError, "frozen"):
                random_kd._assert_frozen(teacher)
        model = random_kd._build_student(self.config)
        restored = deepcopy(model)
        with torch.no_grad():
            restored.encoder.head.weight.add_(1e-6)
        with self.assertRaisesRegex(ValueError, "state changed"):
            random_kd._roundtrip(model, restored)
        restored = deepcopy(model)
        with self.assertRaisesRegex(ValueError, "finite"):
            random_kd._roundtrip(model, restored, torch.zeros(2), torch.tensor([0., float("nan")]))


if __name__ == "__main__":
    unittest.main()
