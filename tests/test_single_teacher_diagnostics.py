"""Opt-in clean training/validation diagnostics on synthetic radiographs.

The production student factory is replaced with the complete small token
student from the existing integration fixture; all cohort, loader, optimizer,
checkpoint and evaluation code still runs against local synthetic DICOMs.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as single_fixture

SmallTokenStudent = single_fixture.SmallTokenStudent

from foundation_sft import train as trainer
from foundation_sft.metrics import aggregate_patient_predictions, binary_metrics


DIAGNOSTICS = {"protocol": "single_teacher_overfit_v1", "evaluation": "validation_only"}


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF teacher needs pinned Transformers")
class SingleTeacherDiagnosticTests(unittest.TestCase):
    def setUp(self):
        fixture = single_fixture.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.kdtrain = fixture.kdtrain
        self.root = fixture.root
        self.split = fixture.split
        self.config = deepcopy(fixture.config)
        self.config["seed"] = 42
        # Nonzero augmentation makes RNG/order differences change training.
        self.config["training"].update(brightness_jitter=0.12, contrast_jitter=0.15)

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                with self.subTest(key=key):
                    self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for first, second in zip(expected, actual):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(expected, actual)

    def run_suite(self, config, conditions):
        with redirect_stdout(io.StringIO()), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ):
            return Path(self.kdtrain.run_experiments(config, project=self.root,
                                                    conditions=conditions))

    def test_diagnostic_schema_rejected_before_teacher_or_student_loading(self):
        malformed = (None, False, [], {}, {"protocol": "single_teacher_overfit_v1"},
                     {**DIAGNOSTICS, "extra": True}, {**DIAGNOSTICS, "evaluation": "test"},
                     {**DIAGNOSTICS, "protocol": "other_protocol"})
        for value in malformed:
            config = deepcopy(self.config)
            config["diagnostics"] = value
            with self.subTest(value=value), mock.patch.object(
                self.kdtrain, "load_teacher", side_effect=AssertionError("invalid config loaded teacher")
            ), self.assertRaises(ValueError):
                self.kdtrain._prepare(config, self.root)
        for mode in ("lora", "partial_finetune"):
            config = deepcopy(self.config)
            config["diagnostics"] = deepcopy(DIAGNOSTICS)
            config["student"] = {"mode": mode}
            if mode == "partial_finetune":
                config["student"]["train_last_blocks"] = 2
            with self.subTest(mode=mode), mock.patch.object(
                self.kdtrain, "load_teacher", side_effect=AssertionError("invalid mode loaded teacher")
            ), self.assertRaises(ValueError):
                self.kdtrain._prepare(config, self.root)

    def test_validation_only_check_proves_clean_geometry_without_evaluation(self):
        config = deepcopy(self.config)
        config["diagnostics"] = deepcopy(DIAGNOSTICS)
        clean = []
        original_inference = self.kdtrain.inference_dataset

        def audit_inference(cohort, payload, *args, **kwargs):
            self.assertEqual(set(cohort["split"]), {"train"})
            dataset = original_inference(cohort, payload, *args, **kwargs)
            self.assertFalse(dataset.base.training)
            self.assertEqual(dataset.base.rotation_degrees, 0.0)
            self.assertEqual(dataset.base.horizontal_flip_probability, 0.0)
            self.assertEqual(dataset.base.brightness_jitter, 0.0)
            self.assertEqual(dataset.base.contrast_jitter, 0.0)
            self.assertEqual(dataset.metadata(), payload["shared_view"])
            self.assertEqual(dataset.base.preprocess.resize_mode, "shortest_edge_center_crop")
            clean.append(dataset)
            return dataset

        with redirect_stdout(io.StringIO()), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ), mock.patch.object(self.kdtrain, "inference_dataset", side_effect=audit_inference), mock.patch.object(
            self.kdtrain.supervised, "predict", side_effect=AssertionError("check must not evaluate")
        ):
            directory = Path(self.kdtrain.check(config, project=self.root))
        result = json.loads((directory / "check.json").read_text())
        self.assertTrue(result["check_passed"])
        self.assertTrue(result["student_state_exact"])
        self.assertEqual(result["evaluation_phase"], "validation_only")
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["diagnostics"]["protocol"], DIAGNOSTICS["protocol"])
        proof = result["diagnostics"]["clean_train_eval"]
        self.assertFalse(proof["training"])
        self.assertTrue(proof["private_generator"])
        self.assertFalse(proof["shuffle"])
        self.assertTrue(all(value == 0.0 for value in proof["augmentation"].values()))
        self.assertGreaterEqual(len(clean), 1)
        self.assertFalse(list(directory.rglob("*test*")))

    def test_clean_diagnostics_preserve_training_weights_optimizer_and_rng(self):
        hashes = []
        for variant, condition in (("supervised", "supervised"), ("full_kd", "distillation"),
                                   ("soft_only", "distillation")):
            with self.subTest(variant=variant):
                config = deepcopy(self.config)
                if variant == "soft_only":
                    config["distillation"].update(global_weight=0.0, local_weight=0.0)
                    # Worker initialization must also use its private RNG.
                    config["training"]["num_workers"] = 1
                config["output_root"] = str(self.root / (variant + "_legacy"))
                legacy = self.run_suite(config, [condition]) / condition
                config["diagnostics"] = deepcopy(DIAGNOSTICS)
                config["output_root"] = str(self.root / (variant + "_diagnostic"))
                clean_loaders = []
                original_predict = trainer.predict

                def audit_predict(model, loader, device, **kwargs):
                    dataset = loader.dataset
                    self.assertNotIn("test", set(dataset.base.cohort["split"]))
                    if set(dataset.base.cohort["split"]) == {"train"}:
                        self.assertFalse(dataset.base.training)
                        self.assertIsNotNone(loader.generator)
                        self.assertEqual(dataset.base.brightness_jitter, 0.0)
                        self.assertEqual(dataset.base.contrast_jitter, 0.0)
                        self.assertEqual(dataset.base.rotation_degrees, 0.0)
                        self.assertEqual(dataset.base.horizontal_flip_probability, 0.0)
                        self.assertEqual(dataset.base.preprocess.resize_mode, "shortest_edge_center_crop")
                        clean_loaders.append(loader)
                    return original_predict(model, loader, device, **kwargs)

                with mock.patch.object(trainer, "predict", side_effect=audit_predict):
                    diagnostic = self.run_suite(config, [condition]) / condition
                before = torch.load(legacy / "last.pt", map_location="cpu", weights_only=True)
                after = torch.load(diagnostic / "last.pt", map_location="cpu", weights_only=True)
                for field in ("model", "optimizer", "scheduler", "rng", "best_model",
                              "distillation_state", "best_distillation_state", "initial_state_sha256"):
                    self.assert_nested_equal(before[field], after[field])
                hashes.append(after["initial_state_sha256"])
                self.assertEqual(len(clean_loaders), config["training"]["epochs"] + 1)
                self.assertTrue((legacy / "test" / "metrics.json").is_file())
                self.assertNotIn("diagnostics", before)
                self.assertNotIn("train_eval_loss", before["history"][0])
                receipt = json.loads((diagnostic / "result.json").read_text())
                self.assertEqual(receipt["evaluation_phase"], "validation_only")
                self.assertFalse(receipt["test_evaluated"])
                self.assertNotIn("test_patient", receipt)
                self.assertFalse((diagnostic / "test").exists())
                self.assertEqual(after["diagnostics"]["protocol"], DIAGNOSTICS["protocol"])
                for row in after["history"]:
                    for field in ("train_eval_loss", "train_eval_patient_roc_auc",
                                  "train_eval_patient_pr_auc", "epoch_seconds"):
                        self.assertTrue(np.isfinite(row[field]), field)
                    self.assertGreaterEqual(row["epoch_seconds"], 0)
                    if variant == "soft_only":
                        self.assertEqual(row["train_global_loss"], 0.0)
                        self.assertEqual(row["train_local_loss"], 0.0)
                if variant == "soft_only":
                    self.assertEqual(receipt["distillation_trainable_parameters"], 0)
        self.assertEqual(len(set(hashes)), 1)

    def test_selected_checkpoint_recomputed_clean_losses_metrics_and_artifacts(self):
        config = deepcopy(self.config)
        config["diagnostics"] = deepcopy(DIAGNOSTICS)
        config["output_root"] = str(self.root / "selected_checkpoint")
        snapshots = {}
        original_save = trainer._torch_save
        original_predict = trainer.predict
        original_binary_metrics = self.kdtrain.binary_metrics
        reloaded_predictions = []
        validation_calls = 0

        def select_first_epoch(labels, probabilities, *args, **kwargs):
            nonlocal validation_calls
            result = original_binary_metrics(labels, probabilities, *args, **kwargs)
            if len(labels) == self.split.loc[self.split.split == "val", "patient_id"].nunique():
                validation_calls += 1
                # Make the earlier checkpoint unambiguously win, so inference
                # on a stale final-epoch student cannot satisfy this test.
                if validation_calls <= config["training"]["epochs"]:
                    result["roc_auc"] = 1.0 if validation_calls == 1 else 0.0
            return result

        def capture_save(path, payload):
            original_save(path, payload)
            if Path(path).name == "last.pt":
                snapshots[payload["epoch"] + 1] = deepcopy(payload["model"])

        def audit_reloaded(model, loader, device, **kwargs):
            best_files = list(Path(config["output_root"]).rglob("best.pt"))
            if best_files:
                saved = torch.load(best_files[0], map_location="cpu", weights_only=True)
                self.assert_nested_equal(saved["model"], model.trainable_state_dict())
                reloaded_predictions.append(set(loader.dataset.base.cohort["split"]).pop())
            return original_predict(model, loader, device, **kwargs)

        with mock.patch.object(trainer, "_torch_save", side_effect=capture_save), mock.patch.object(
            trainer, "predict", side_effect=audit_reloaded
        ), mock.patch.object(self.kdtrain, "binary_metrics", side_effect=select_first_epoch
        ):
            suite = self.run_suite(config, ["distillation"])
        directory = suite / "distillation"
        receipt = json.loads((directory / "result.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        best_epoch = max(history, key=lambda row: row["val_patient_roc_auc"])["epoch"]
        self.assertEqual(receipt["best_epoch"], best_epoch)
        self.assertEqual(best_epoch, 1)
        self.assertLess(best_epoch, receipt["last_epoch"])
        self.assertEqual(receipt["last_epoch"], history[-1]["epoch"])
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        self.assert_nested_equal(snapshots[best_epoch], best["model"])
        self.assertEqual(set(reloaded_predictions), {"train", "val"})
        with mock.patch.object(self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent):
            restored, payload = self.kdtrain.load_student_checkpoint(directory / "best.pt", "cpu")
        criterion = trainer.FocalLoss(best["config"]["training"]["focal_gamma"],
                                      best["config"]["training"]["focal_scale"])
        for split_name, output, loss_key, metric_key in (
            ("train", "train_eval", "best_train_eval_loss", "best_train_patient"),
            ("val", "validation", "best_val_loss", "validation_patient"),
        ):
            dataset = self.kdtrain.inference_dataset(self.split.loc[self.split.split == split_name], payload)
            loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                                generator=torch.Generator().manual_seed(123))
            predictions, loss = trainer.predict(restored, loader, torch.device("cpu"), criterion=criterion)
            saved_predictions = pd.read_csv(directory / output / "predictions.csv")
            pd.testing.assert_frame_equal(saved_predictions, predictions, check_exact=False, rtol=0, atol=1e-14)
            self.assertAlmostEqual(receipt[loss_key], loss, places=12)
            patients = aggregate_patient_predictions(predictions)
            expected = binary_metrics(patients.label, patients.probability, payload["threshold"])
            self.assert_nested_equal(expected, receipt[metric_key])
            evaluation = json.loads((directory / output / "metrics.json").read_text())
            self.assert_nested_equal(expected, evaluation["patient"])
            for name in ("patient_predictions.csv", "roc_curve.png", "precision_recall_curve.png",
                         "calibration_curve.png"):
                self.assertTrue((directory / output / name).is_file(), name)
        self.assertFalse((directory / "test").exists())
        self.assertIn("best_gap", receipt)
        self.assertIn("last_gap", receipt)
        self.assertTrue((suite / "comparison.csv").is_file())
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(
            torch, "load", side_effect=AssertionError("reporting must not load models")
        ):
            self.kdtrain.show_metrics(config, self.root, suite=suite)
        self.assertIn("distillation", output.getvalue())

    def test_l2_sp_full_kd_saves_anchor_proof_and_separate_penalties(self):
        config = deepcopy(self.config)
        config["diagnostics"] = {**DIAGNOSTICS,
                                 "l2_sp": {"coefficient": 0.0001, "reduction": "sum"}}
        suite = self.run_suite(config, ["distillation"])
        directory = suite / "distillation"
        receipt = json.loads((directory / "result.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        proof = receipt["diagnostics"]["l2_sp"]
        self.assertEqual(proof["coefficient"], 0.0001)
        self.assertEqual(proof["reduction"], "sum")
        self.assertTrue(proof["anchor_unchanged"])
        self.assertTrue(proof["includes_final_norm"])
        self.assertTrue(all(name.startswith("encoder.") and not name.startswith("encoder.head.")
                            for name in proof["anchor_parameter_names"]))
        self.assertTrue(any(row["train_l2_sp_loss"] > 0 for row in history))
        for row in history:
            self.assertAlmostEqual(row["train_l2_sp_penalty"], 0.0001 * row["train_l2_sp_loss"], places=10)
            self.assertAlmostEqual(row["train_loss"], row["train_hard_loss"] + row["train_kd_loss"]
                                   + row["train_l2_sp_penalty"], places=7)
        for checkpoint in ("best.pt", "last.pt"):
            saved = torch.load(directory / checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["diagnostics"]["l2_sp"]["anchor_sha256"], proof["anchor_sha256"])
        self.assertFalse((directory / "test").exists())


class ImageNetL2SPTests(unittest.TestCase):
    def setUp(self):
        from foundation_sft.single_teacher_diagnostics import ImageNetL2SP, validate_options
        self.anchor_type = ImageNetL2SP
        self.validate_options = validate_options

    def test_raw_sum_fixed_anchor_excludes_head_includes_norm_and_pulls_back(self):
        torch.manual_seed(42)
        student = SmallTokenStudent("synthetic_ImageNet")
        anchor = self.anchor_type(student, {"coefficient": 0.0001, "reduction": "sum"})
        original = deepcopy(anchor.anchors)
        self.assertIn("encoder.norm.weight", anchor.anchors)
        self.assertIn("encoder.norm.bias", anchor.anchors)
        self.assertFalse(any(name.startswith("encoder.head.") for name in anchor.anchors))
        raw, weighted = anchor.losses(student)
        self.assertEqual(raw.item(), 0.0)
        self.assertEqual(weighted.item(), 0.0)
        with torch.no_grad():
            student.encoder.head.weight.add_(5)
            student.encoder.head.bias.add_(2)
        self.assertEqual(anchor.losses(student)[0].item(), 0.0)
        target = student.encoder.norm.weight
        with torch.no_grad():
            target.add_(0.25)
        raw, weighted = anchor.losses(student)
        self.assertAlmostEqual(raw.item(), 6 * 0.25 ** 2, places=7)
        self.assertAlmostEqual(weighted.item(), 0.0001 * raw.item(), places=11)
        weighted.backward()
        torch.testing.assert_close(target.grad, torch.full_like(target, 0.0001 * 2 * 0.25),
                                   rtol=1e-6, atol=1e-10)
        self.assertIsNone(student.encoder.head.weight.grad)
        self.assertIsNone(student.encoder.head.bias.grad)
        with torch.no_grad():
            target.sub_(target.grad)
        self.assertLess(anchor.losses(student)[0].item(), raw.item())
        for name, value in original.items():
            torch.testing.assert_close(anchor.anchors[name], value, rtol=0, atol=0)
            self.assertFalse(anchor.anchors[name].requires_grad)
        self.assertTrue(anchor.metadata()["anchor_unchanged"])

    def test_l2_sp_option_is_exact_and_reserved_for_full_distillation(self):
        config = {"diagnostics": {**DIAGNOSTICS,
                                  "l2_sp": {"coefficient": 0.0001, "reduction": "sum"}},
                  "distillation": {"soft_weight": 0.5, "global_weight": 0.1, "local_weight": 1.0}}
        self.validate_options(config, student_mode="full_finetune", condition="distillation")
        for value in (None, {}, {"coefficient": True, "reduction": "sum"},
                      {"coefficient": 0.01, "reduction": "sum"},
                      {"coefficient": 0.0001, "reduction": "mean"},
                      {"coefficient": 0.0001, "reduction": "sum", "extra": 0}):
            invalid = deepcopy(config)
            invalid["diagnostics"]["l2_sp"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate_options(invalid, student_mode="full_finetune", condition="distillation")
        with self.assertRaises(ValueError):
            self.validate_options(config, student_mode="full_finetune", condition="supervised")
        config["distillation"].update(global_weight=0.0, local_weight=0.0)
        with self.assertRaises(ValueError):
            self.validate_options(config, student_mode="full_finetune", condition="distillation")


if __name__ == "__main__":
    unittest.main()
