"""Single-teacher KD integration with synthetic DICOMs and a real tiny HF teacher.

The student factory is replaced by a six-channel token model to keep this
end-to-end protocol test independent of timm and public pretrained downloads.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_foundation_training as teacher_fixture

from foundation_sft import train as trainer
from foundation_sft.metrics import aggregate_patient_predictions, select_threshold
from foundation_sft.models import build_adapter
from foundation_sft.models.common import BackboneFeatures, PreprocessSpec


class SmallTokenStudent(nn.Module):
    """A complete trainable student exposing the production student's API."""

    def __init__(self, checkpoint=None, *, initialization=None):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.patch_embed = nn.Conv2d(3, 6, kernel_size=4, stride=4)
        self.encoder.norm = nn.LayerNorm(6)
        self.encoder.head = nn.Linear(6, 1)
        self.feature_dim = 6
        self.preprocess = PreprocessSpec(8)
        self.initialization = deepcopy(initialization) if initialization is not None else {
            "source": "synthetic-six-channel-token-student", "checkpoint": str(checkpoint),
        }

    def forward(self, images, return_features=False):
        spatial = self.encoder.patch_embed(images)
        local = self.encoder.norm(spatial.flatten(2).transpose(1, 2))
        features = BackboneFeatures(local.mean(1), local, tuple(spatial.shape[-2:])).validate()
        logits = self.encoder.head(features.global_features).squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def trainable_state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state, strict=True)

    def training_metadata(self):
        return {"model": "synthetic_student", "mode": "full_finetune", "feature_dim": 6,
                "preprocessing": asdict(self.preprocess), "initialization": deepcopy(self.initialization),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF teacher needs pinned Transformers")
class SingleTeacherDistillationTests(unittest.TestCase):
    def setUp(self):
        from foundation_sft import distillation

        self.kdtrain = distillation
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        fixture = teacher_fixture.FoundationTrainingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.root = fixture.root
        self.teacher_config = deepcopy(fixture.config)
        self.teacher_config["mode"] = "lora"
        # Deliberately make teacher geometry differ from a direct student
        # resize: standalone inference must retain this first center crop.
        processor_path = Path(self.teacher_config["model_dir"]) / "preprocessor_config.json"
        processor = json.loads(processor_path.read_text())
        processor.update(size={"shortest_edge": 12}, crop_size={"height": 8, "width": 8},
                         do_center_crop=True)
        processor_path.write_text(json.dumps(processor))
        with redirect_stdout(io.StringIO()):
            self.split = trainer.prepare(self.teacher_config)
        torch.manual_seed(4)
        teacher = trainer._classifier(build_adapter(self.teacher_config), self.teacher_config)
        # Restore nonzero learned adapters/head before freezing. Loading only
        # the initial HF weights cannot accidentally satisfy this comparison.
        with torch.no_grad():
            for name, parameter in teacher.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(torch.randn_like(parameter) * 0.03)
        self.teacher_state = teacher.trainable_state_dict()
        payload = trainer._checkpoint(teacher, self.teacher_config, trainer._split_hash(self.split))
        payload.update(threshold=0.43, best_val_patient_roc_auc=0.71)
        self.teacher_checkpoint = self.root / "teacher" / "best.pt"
        self.teacher_checkpoint.parent.mkdir()
        torch.save(payload, self.teacher_checkpoint)
        self.student_checkpoint = self.root / "synthetic_student.pth"
        self.student_checkpoint.write_bytes(b"synthetic fixture; factory is patched")
        self.config = {
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "student_checkpoint": str(self.student_checkpoint),
            "device": "cpu", "seed": 38, "output_root": str(self.root / "kd_runs"),
            "training": {"epochs": 2, "patience": 4, "batch_size": 3,
                         "accumulation_steps": 2, "num_workers": 0,
                         "backbone_lr": 0.002, "head_lr": 0.01,
                         "projection_lr": 0.002, "amp": False},
            "distillation": {"temperature": 2.0, "soft_weight": 0.5,
                             "global_weight": 0.1, "local_weight": 1.0, "pool_grid": [2, 2]},
        }

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name, value in expected.items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0, msg=name)

    def test_teacher_restores_original_lora_and_head_then_freezes(self):
        with redirect_stdout(io.StringIO()):
            teacher, config, split, payload = self.kdtrain.load_teacher(self.teacher_checkpoint, device="cpu")
        self.assertEqual(config["model"], "rad_dino")
        self.assertEqual(config["mode"], "lora")
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        restored = teacher.state_dict()
        self.assert_state_equal(self.teacher_state, {name: restored[name] for name in self.teacher_state})
        self.assertEqual(trainer._split_hash(split), payload["cohort_hash"])
        self.assertEqual(payload["threshold"], 0.43)

    def test_teacher_rejects_nonbest_and_changed_cohort_checkpoints(self):
        payload = torch.load(self.teacher_checkpoint, map_location="cpu", weights_only=True)
        for defect in ("no_threshold", "wrong_cohort", "wrong_mode",
                       "task_nan", "task_inf", "task_dtype", "task_shape"):
            altered = deepcopy(payload)
            if defect == "no_threshold":
                del altered["threshold"]
            elif defect == "wrong_cohort":
                altered["cohort_hash"] = "different-frozen-patient-manifest"
            elif defect == "wrong_mode":
                altered["config"]["mode"] = "linear_probe"
            else:
                key = next(name for name, value in altered["model"].items() if value.is_floating_point())
                value = altered["model"][key]
                if defect == "task_nan":
                    value.reshape(-1)[0] = float("nan")
                elif defect == "task_inf":
                    value.reshape(-1)[0] = float("inf")
                elif defect == "task_dtype":
                    altered["model"][key] = value.double()
                else:
                    altered["model"][key] = value.reshape(-1)[:-1]
            path = self.root / (defect + ".pt")
            torch.save(altered, path)
            with self.subTest(defect=defect), redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                self.kdtrain.load_teacher(path, device="cpu")

    def test_check_smoke_preserves_teacher_removes_roundtrip_and_uses_training_only(self):
        teachers, before = [], []
        original_load_teacher = self.kdtrain.load_teacher
        original_loader = self.kdtrain._loader

        def capture_teacher(*args, **kwargs):
            result = original_load_teacher(*args, **kwargs)
            teachers.append(result[0])
            before.append({name: value.detach().clone() for name, value in result[0].state_dict().items()})
            return result

        def training_loader(split, name, *args, **kwargs):
            self.assertEqual(name, "train")
            loader = original_loader(split, name, *args, **kwargs)
            self.assertEqual(set(loader.dataset.base.cohort["split"]), {"train"})
            return loader

        with redirect_stdout(io.StringIO()), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ), mock.patch.object(self.kdtrain, "load_teacher", side_effect=capture_teacher), mock.patch.object(
            self.kdtrain, "_loader", side_effect=training_loader
        ) as loaders, mock.patch.object(
            self.kdtrain.supervised, "predict", side_effect=AssertionError("Smoke must not evaluate held-out data")
        ) as prediction:
            directory = Path(self.kdtrain.check(self.config, project=self.root))
        self.assertEqual(loaders.call_count, 1)
        self.assertEqual(prediction.call_count, 0)
        result = json.loads((directory / "check.json").read_text())
        self.assertTrue(result["check_passed"])
        self.assertTrue(result["teacher_unchanged"])
        self.assertTrue(result["student_full_checkpoint_roundtrip"])
        self.assertFalse(result["training_started"])
        self.assertEqual(result["teacher_features"], 12)
        self.assertEqual(result["student_features"], 6)
        self.assertTrue(all(np.isfinite(value) for value in result["losses"].values()))
        self.assertFalse((directory / "roundtrip.pt").exists())
        self.assertFalse(list(directory.glob("*.pt")))
        self.assertFalse((directory / "test").exists())
        for teacher, state in zip(teachers, before):
            self.assert_state_equal(state, teacher.state_dict())
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                for parameter in teacher.parameters()))

    def test_show_metrics_prints_both_groups_without_teacher_or_student_sources(self):
        suite = Path(self.config["output_root"]) / "suites" / "suite_saved_fixture"
        for condition in ("supervised", "distillation"):
            directory = suite / condition
            directory.mkdir(parents=True)
            receipt = {
                "condition": condition, "seed": self.config["seed"],
                "best_val_patient_roc_auc": 0.73,
                "test_patient": {"roc_auc": 0.75, "pr_auc": 0.8, "average_precision": 0.82,
                                 "sensitivity": 0.7, "specificity": 0.9},
                "roc_auc_ci": {"lower": 0.6, "upper": 0.9},
            }
            (directory / "result.json").write_text(json.dumps(receipt))
        self.teacher_checkpoint.unlink()
        self.student_checkpoint.unlink()
        shutil.rmtree(self.teacher_config["model_dir"])
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(
            self.kdtrain, "load_teacher", side_effect=AssertionError("Reporting must not load teacher")
        ), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=AssertionError("Reporting must not load student")
        ), mock.patch.object(
            self.kdtrain.supervised, "prepare", side_effect=AssertionError("Reporting must not read patients")
        ), mock.patch.object(torch, "load", side_effect=AssertionError("Reporting must not load weights")):
            self.kdtrain.show_metrics(self.config, project=self.root)
        self.assertIn("supervised", output.getvalue())
        self.assertIn("distillation", output.getvalue())
        self.assertIn("0.7500", output.getvalue())

    def test_same_initialization_frozen_teacher_validation_threshold_and_complete_checkpoints(self):
        initial_states, teachers, teacher_states = {}, [], []
        test_conditions = []
        original_train_epoch = self.kdtrain._train_epoch
        original_load_teacher = self.kdtrain.load_teacher
        original_predict = self.kdtrain.supervised.predict

        def capture_teacher(*args, **kwargs):
            result = original_load_teacher(*args, **kwargs)
            teacher = result[0]
            teachers.append(teacher)
            teacher_states.append({name: value.detach().clone() for name, value in teacher.state_dict().items()})
            return result

        def inspect_epoch(student, teacher, kd, loader, optimizer, hard_loss, device, train):
            group = "supervised" if kd is None else "distillation"
            if group not in initial_states:
                initial_states[group] = student.trainable_state_dict()
            if teacher is not None:
                self.assertFalse(teacher.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            result = original_train_epoch(student, teacher, kd, loader, optimizer, hard_loss, device, train)
            if teacher is not None:
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            return result

        def audit_predict(model, loader, device, **kwargs):
            dataset = getattr(loader.dataset, "base", loader.dataset)
            if set(dataset.cohort["split"]) == {"test"}:
                condition = ("supervised", "distillation")[len(test_conditions)]
                best_files = list(Path(self.config["output_root"]).rglob(f"{condition}/best.pt"))
                self.assertEqual(len(best_files), 1)
                saved = torch.load(best_files[0], map_location="cpu", weights_only=True)
                self.assertIn("threshold", saved)
                self.assertEqual(selector.call_count, len(test_conditions) + 1)
                self.assert_state_equal(saved["model"], model.trainable_state_dict())
                test_conditions.append(condition)
            return original_predict(model, loader, device, **kwargs)

        original_config = deepcopy(self.config)
        with redirect_stdout(io.StringIO()), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ), mock.patch.object(self.kdtrain, "load_teacher", side_effect=capture_teacher), mock.patch.object(
            self.kdtrain, "_train_epoch", side_effect=inspect_epoch
        ), mock.patch.object(self.kdtrain.supervised, "predict", side_effect=audit_predict
        ), mock.patch.object(self.kdtrain, "select_threshold", wraps=select_threshold) as selector:
            suite = Path(self.kdtrain.run_experiments(self.config, project=self.root))
        self.assertEqual(self.config, original_config)
        self.assertEqual(set(initial_states), {"supervised", "distillation"})
        self.assert_state_equal(initial_states["supervised"], initial_states["distillation"])
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(test_conditions, ["supervised", "distillation"])
        for teacher, before in zip(teachers, teacher_states):
            self.assert_state_equal(before, teacher.state_dict())
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

        # Standalone deployment must need neither the teacher checkpoint nor
        # its original HF snapshot, while the source DICOMs remain available.
        self.teacher_checkpoint.unlink()
        shutil.rmtree(self.teacher_config["model_dir"])
        fingerprints = []
        for index, group in enumerate(("supervised", "distillation")):
            with self.subTest(group=group):
                directory = suite / group
                best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
                fingerprints.append(best["initial_state_sha256"])
                self.assertRegex(fingerprints[-1], r"^[0-9a-f]{64}$")
                self.assertEqual(best["format_version"], "single_teacher_kd_v1")
                self.assertEqual(set(best["model"]), set(initial_states[group]))
                self.assertTrue(any(not torch.equal(value, initial_states[group][name])
                                    for name, value in best["model"].items()))
                self.assertEqual(last["initial_state_sha256"], best["initial_state_sha256"])
                for name in ("distillation_state", "best_model", "best_distillation_state", "rng"):
                    self.assertIn(name, last)
                history = json.loads((directory / "history.json").read_text())
                self.assertEqual(len(history), 2)
                self.assertTrue(all(np.isfinite(row["val_loss"]) for row in history))

                validation = pd.read_csv(directory / "validation_predictions.csv")
                patients = aggregate_patient_predictions(validation)
                expected_threshold = select_threshold(patients.label, patients.probability)
                self.assertAlmostEqual(best["threshold"], expected_threshold, places=14)
                np.testing.assert_array_equal(selector.call_args_list[index].args[0], patients.label)
                np.testing.assert_allclose(selector.call_args_list[index].args[1], patients.probability,
                                           rtol=0, atol=1e-14)
                test = pd.read_csv(directory / "test" / "predictions.csv")
                self.assertFalse(set(validation.patient_id) & set(test.patient_id))
                self.assertEqual(set(test.patient_id), set(self.split.loc[self.split.split == "test", "patient_id"]))
                metrics = json.loads((directory / "test" / "metrics.json").read_text())
                self.assertEqual(metrics["patient"]["threshold"], best["threshold"])

                # Both best and last are self-contained student artifacts.
                # Delete the initial weight placeholder before rebuilding.
                self.student_checkpoint.unlink(missing_ok=True)
                for checkpoint in (directory / "best.pt", directory / "last.pt"):
                    with mock.patch.object(self.kdtrain, "build_vit_tiny_student", side_effect=SmallTokenStudent) as factory:
                        restored, payload = self.kdtrain.load_student_checkpoint(checkpoint, device="cpu")
                    self.assertIsNone(factory.call_args.kwargs.get("checkpoint"))
                    self.assertIn("initialization", factory.call_args.kwargs)
                    self.assert_state_equal(payload["model"], restored.state_dict())
                    if checkpoint.name == "best.pt":
                        cohort = self.split.loc[self.split.split == "test"]
                        dataset = self.kdtrain.inference_dataset(cohort, payload)
                        self.assertNotIn("teacher_image", dataset[0])
                        self.assertEqual(dataset.base.preprocess.resize_mode, "shortest_edge_center_crop")
                        loader = DataLoader(dataset, batch_size=self.config["training"]["batch_size"], shuffle=False)
                        inferred = trainer.predict(restored, loader, torch.device("cpu"))
                        pd.testing.assert_frame_equal(
                            test.sort_values("sample_id").reset_index(drop=True),
                            inferred.sort_values("sample_id").reset_index(drop=True),
                            check_exact=False, rtol=0, atol=1e-14,
                        )
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_unequal_microbatches_match_full_batch_for_student_and_global_projector(self):
        torch.manual_seed(71)
        template = SmallTokenStudent("synthetic")
        teacher = SmallTokenStudent("synthetic-teacher").requires_grad_(False).eval()
        before_teacher = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        images, teacher_images = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
        labels = torch.tensor([0.0, 1.0, 1.0])

        def batch(section):
            return {"image": images[section], "teacher_image": teacher_images[section], "label": labels[section]}

        for distill in (False, True):
            with self.subTest(distillation=distill):
                first, second = deepcopy(template), deepcopy(template)
                kd_first = (self.kdtrain.SingleTeacherDistillationLoss(
                    student_dim=6, teacher_dim=6, pool_grid=(2, 2)
                ) if distill else None)
                kd_second = deepcopy(kd_first)
                before_projector = kd_first.projector.weight.detach().clone() if distill else None
                parameters_first = list(first.parameters()) + (list(kd_first.parameters()) if distill else [])
                parameters_second = list(second.parameters()) + (list(kd_second.parameters()) if distill else [])
                optimizer_first = torch.optim.SGD(parameters_first, lr=0.01)
                optimizer_second = torch.optim.SGD(parameters_second, lr=0.01)
                hard_loss = nn.BCEWithLogitsLoss()
                uneven = self.kdtrain._train_epoch(
                    first, teacher, kd_first, [batch(slice(0, 2)), batch(slice(2, 3))],
                    optimizer_first, hard_loss, torch.device("cpu"),
                    {"accumulation_steps": 2, "gradient_clip": 100.0, "amp": False},
                )
                whole = self.kdtrain._train_epoch(
                    second, teacher, kd_second, [batch(slice(None))],
                    optimizer_second, hard_loss, torch.device("cpu"),
                    {"accumulation_steps": 1, "gradient_clip": 100.0, "amp": False},
                )
                self.assertEqual(uneven["optimizer_steps"], 1)
                self.assertEqual(whole["optimizer_steps"], 1)
                for name, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[name], rtol=1e-5, atol=1e-7, msg=name)
                for name in ("total", "hard", "kd", "soft", "global", "local"):
                    self.assertAlmostEqual(uneven[name], whole[name], places=6)
                if distill:
                    for name, value in kd_first.state_dict().items():
                        torch.testing.assert_close(value, kd_second.state_dict()[name], rtol=1e-5, atol=1e-7, msg=name)
                    self.assertFalse(torch.equal(before_projector, kd_first.projector.weight.detach()))
                    self.assertIsNone(kd_first.projector.weight.grad)  # Cleared after the optimizer update.
        self.assert_state_equal(before_teacher, teacher.state_dict())
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
