"""Clean train/validation auditing of six saved, complete legacy students."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_single_teacher_distillation import SmallTokenStudent
import test_single_teacher_test_evaluation as dicom_fixture

import check_distillation_overfitting as checker
from foundation_sft import distillation, distillation_ablation, distillation_suite, metrics
from foundation_sft import train as trainer
from foundation_sft.data import CXRDataset
from foundation_sft.distillation_data import SharedViewDataset
from foundation_sft.models.common import PreprocessSpec


class PosthocAblationAuditTests(unittest.TestCase):
    """Use real DICOM decoding and independent logits/loss calculations."""

    _json = staticmethod(dicom_fixture.SingleTeacherTestEvaluationTests._json)
    _sha = staticmethod(dicom_fixture.SingleTeacherTestEvaluationTests._sha)
    _dicoms = dicom_fixture.SingleTeacherTestEvaluationTests._dicoms

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="posthoc-ablation-audit-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = self.root / "suite"
        self.suite.mkdir()
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        self.split = self._dicoms()
        self.teacher_spec = PreprocessSpec(8, resize_mode="shortest_edge_center_crop", resize_shortest_edge=12)
        self.geometry = SharedViewDataset(CXRDataset(self.split, self.teacher_spec, training=False),
                                         student_size=8, include_teacher=False).metadata()
        self.runs, self.payloads = {}, {}
        self.state = self._completed_suite()
        self._save_state()

    def _completed_suite(self):
        metadata = SmallTokenStudent(initialization={"pretraining": "ImageNet-1k",
            "checkpoint": "/missing/original-imagenet.pth", "checkpoint_sha256": "b" * 64}).training_metadata()
        teacher = {"checkpoint": "/missing/teacher.pt", "checkpoint_sha256": "d" * 64}
        plan = {"format_version": distillation_ablation.FORMAT, "project": str(self.root),
                "groups": list(distillation_ablation.GROUPS), "seeds": [42], "jobs": []}
        state = {"format_version": distillation_ablation.FORMAT, "status": "completed", "plan": plan, "jobs": {}}
        history = [{"epoch": epoch, "train_loss": 9.0 + epoch, "train_hard_loss": 4.0 + epoch,
                    "train_kd_loss": 5.0, "val_loss": loss, "val_patient_roc_auc": auc,
                    "val_patient_pr_auc": pr} for epoch, loss, auc, pr in
                   ((1, .23, .72, .69), (2, .21, .81, .79), (3, .25, .78, .74))]
        for index, (name, (condition, *weights)) in enumerate(distillation_ablation.GROUPS.items()):
            group = "ABCDEF"[index]
            config = {"seed": 42, "device": "cpu", "student": {"mode": "full_finetune"},
                "student_checkpoint": "/missing/original-imagenet.pth", "teacher_checkpoint": "/missing/teacher.pt",
                "training": {"batch_size": 2, "num_workers": 0, "epochs": 3, "amp": False,
                             "backbone_lr": 5e-5, "head_lr": 1e-4, "projection_lr": 1e-4,
                             "focal_gamma": 1.3, "focal_scale": .45},
                "evaluation": {"patient_aggregation": "mean", "bootstrap_samples": 0},
                "distillation": {"temperature": 2.0, "pool_grid": [7, 7],
                                 **dict(zip(distillation_ablation.WEIGHTS, weights))}}
            job = {"id": name + "_seed42", "group": name, "condition": condition, "seed": 42,
                   "student_mode": "full_finetune", "device": "cpu", "config": config}
            run = self.suite / "runs" / name
            run.mkdir(parents=True)
            torch.manual_seed(50 + index)
            student = SmallTokenStudent(initialization=metadata["initialization"])
            best = distillation._student_payload(student, config, "c" * 64, condition, "a" * 64, teacher, self.geometry)
            best.update(threshold=.43, best_val_patient_roc_auc=.81, distillation_state=None)
            # Native legacy best checkpoints omit epoch; last.model is the final
            # student and differs from last.best_model, which must not be used.
            with torch.no_grad():
                student.encoder.head.weight.mul_(-1.5)
                student.encoder.head.bias.add_(.7)
            last = distillation._student_payload(student, config, "c" * 64, condition, "a" * 64, teacher, self.geometry)
            last.update(epoch=2, history=deepcopy(history), best_auc=.81, bad_epochs=1,
                        best_model=deepcopy(best["model"]), best_distillation_state=None,
                        distillation_state=None, optimizer={}, scheduler={}, rng=None)
            torch.save(best, run / "best.pt")
            torch.save(last, run / "last.pt")
            result = {"status": "completed", "run": str(run), "seed": 42, "condition": condition,
                "student_mode": "full_finetune", "initial_state_sha256": "a" * 64, "cohort_hash": "c" * 64,
                "teacher_unchanged": True, "threshold": .43, "best_val_patient_roc_auc": .81,
                "test_patient": {"roc_auc": .75}, "roc_auc_ci": None}
            for filename, value in (("config", config), ("result", result), ("model_metadata", metadata),
                    ("teacher", teacher), ("shared_view", self.geometry), ("history", history),
                    ("split_metadata", {"_manifest_hash": "c" * 64})):
                self._json(run / (filename + ".json"), value)
            self.split.to_csv(run / "split.csv", index=False)
            # Historical test artifacts are deliberately present but opaque.
            self._json(run / "test" / "metrics.json", {"patient": result["test_patient"]})
            (run / "test" / "predictions.csv").write_text("historical test predictions; do not read\n")
            check = {"check_passed": True, "condition": condition, "config": deepcopy(config), "model_metadata": metadata,
                     "initial_state_sha256": "a" * 64, "cohort_hash": "c" * 64,
                     "teacher": teacher, "shared_view": self.geometry}
            receipt = deepcopy(result)
            receipt.update(checkpoint_sha256=self._sha(run / "best.pt"),
                           history_sha256=self._sha(run / "history.json"))
            state["jobs"][job["id"]] = {"job": deepcopy(job), "status": "completed", "check": check, "result": receipt}
            plan["jobs"].append(job)
            self.runs[group], self.payloads[group] = run, {"best": best, "last": last}
        return state

    def _save_state(self):
        self.state["protocol_sha256"] = distillation_suite.protocol_hash(self.state["plan"])
        self._json(self.suite / "suite.json", self.state)

    def _source_bytes(self):
        return {path: path.read_bytes() for path in self.root.rglob("*")
                if path.is_file() and "report" not in path.relative_to(self.root).parts}

    def _audit(self):
        def complete_student(checkpoint=None, *, initialization=None):
            self.assertIsNone(checkpoint, "audit tried to reload original ImageNet weights")
            return SmallTokenStudent(initialization=initialization)

        with redirect_stdout(io.StringIO()), mock.patch.object(
            distillation, "build_vit_tiny_student", side_effect=complete_student
        ):
            return checker.audit(self.suite, device="cpu")

    def _expected(self, group, checkpoint, partition):
        payload = self.payloads[group][checkpoint]
        student = SmallTokenStudent(initialization=payload["model_metadata"]["initialization"])
        student.load_trainable_state_dict(payload["model"])
        dataset = SharedViewDataset(CXRDataset(self.split.loc[self.split["split"] == partition],
            self.teacher_spec, training=False), student_size=8, include_teacher=False)
        probabilities, labels, rows, losses = [], [], [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=2):
                logits = student(batch["image"])
                bce = F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="none")
                losses.extend((.45 * (1 - torch.exp(-bce)).pow(1.3) * bce).tolist())
                probabilities.extend(logits.sigmoid().tolist())
                labels.extend(batch["label"].tolist())
                rows.extend(zip(batch["sample_id"], batch["patient_id"]))
        predictions = pd.DataFrame([{"sample_id": sample, "patient_id": patient, "label": int(label),
                                     "probability": probability}
            for (sample, patient), label, probability in zip(rows, labels, probabilities)])
        patients = metrics.aggregate_patient_predictions(predictions)
        return predictions, np.mean(losses), metrics.binary_metrics(patients.label, patients.probability)

    def test_real_clean_inference_uses_saved_final_models_focal_and_patient_means_without_mutation(self):
        before = self._source_bytes()
        opened = []
        original = CXRDataset.__getitem__

        def clean_only(dataset, index):
            row = dataset.cohort.iloc[index]
            self.assertIn(row["split"], ("train", "val"))
            self.assertFalse(dataset.training)
            opened.append(row["sample_id"])
            return original(dataset, index)

        with ExitStack() as stack:
            for module, name in ((distillation, "load_teacher"), (distillation, "_train_epoch"),
                    (trainer, "_train_epoch"), (distillation, "select_threshold"),
                    (trainer, "select_threshold"), (metrics, "select_threshold")):
                stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError("audit trained or selected a threshold")))
            stack.enter_context(mock.patch.object(CXRDataset, "__getitem__", new=clean_only))
            read = checker._read

            def no_test_metrics(path):
                self.assertNotIn("test", Path(path).relative_to(self.root).parts)
                return read(path)

            stack.enter_context(mock.patch.object(checker, "_read", side_effect=no_test_metrics))
            output = self._audit()
        self.assertIsInstance(output, Path)
        table = pd.read_csv(output / "comparison.csv")
        self.assertEqual(len(table), 12)
        self.assertEqual(set(zip(table.group, table.checkpoint)), {(g, c) for g in "ABCDEF" for c in ("best", "last")})
        self.assertEqual(len(opened), 12 * len(self.split.loc[self.split["split"].isin(["train", "val"])]))
        for row in table.itertuples(index=False):
            self.assertEqual(row.epoch, 2 if row.checkpoint == "best" else 3)
            self.assertEqual(row.ablation, self.runs[row.group].name)
            for partition in ("train", "val"):
                expected, loss, patient = self._expected(row.group, row.checkpoint, partition)
                saved = pd.read_csv(output / f"{row.group}_{row.checkpoint}_{partition}_predictions.csv")
                pd.testing.assert_frame_equal(saved, expected, check_exact=False, rtol=0, atol=1e-14)
                saved_patients = pd.read_csv(output / f"{row.group}_{row.checkpoint}_{partition}_patient_predictions.csv")
                pd.testing.assert_frame_equal(saved_patients, metrics.aggregate_patient_predictions(expected),
                                              check_exact=False, rtol=0, atol=1e-14)
                self.assertAlmostEqual(getattr(row, partition + "_loss"), loss, places=7)
                self.assertEqual(getattr(row, partition + "_n_images"), 3)
                self.assertEqual(getattr(row, partition + "_n_patients"), 2)
                for name in ("roc_auc", "pr_auc", "average_precision"):
                    self.assertAlmostEqual(getattr(row, partition + "_" + name), patient[name], places=12)
            self.assertAlmostEqual(row.loss_gap, row.val_loss - row.train_loss, places=12)
            self.assertAlmostEqual(row.auc_gap, row.train_roc_auc - row.val_roc_auc, places=12)
        self.assertTrue((table.train_loss < 4).all(), "historical optimization/KD loss was reused")
        best = table.loc[table.checkpoint == "best"].set_index("group")
        last = table.loc[table.checkpoint == "last"].set_index("group")
        self.assertFalse(np.allclose(best.train_loss, last.train_loss), "last.best_model was loaded instead of last.model")
        shared = SharedViewDataset(CXRDataset(self.split, self.teacher_spec, training=False), student_size=8, include_teacher=False)
        direct = CXRDataset(self.split, SmallTokenStudent().preprocess, training=False)
        self.assertFalse(torch.allclose(shared[0]["image"], direct[0]["image"]))
        self.assertEqual(before, self._source_bytes())

    def test_invalid_sources_are_rejected_before_any_pixels_or_student_reconstruction(self):
        state, files = deepcopy(self.state), self._source_bytes()
        job_id = list(self.state["jobs"])[-1]
        for defect in ("missing_last", "checkpoint_sha", "cohort", "patient_overlap", "conflicting_labels", "last_epoch", "best_score"):
            with self.subTest(defect=defect):
                self.state = deepcopy(state)
                for path, contents in files.items():
                    path.write_bytes(contents)
                run = self.runs["F"]
                if defect == "missing_last":
                    (run / "last.pt").unlink()
                elif defect == "checkpoint_sha":
                    self.state["jobs"][job_id]["result"]["checkpoint_sha256"] = "0" * 64
                elif defect == "cohort":
                    self._json(run / "split_metadata.json", {"_manifest_hash": "wrong"})
                elif defect in ("patient_overlap", "conflicting_labels"):
                    split = self.split.copy()
                    if defect == "patient_overlap":
                        split.loc[split["split"] == "val", "patient_id"] = "train-patient-0"
                    else:
                        split.loc[(split["split"] == "train") & (split["label"] == 1), "patient_id"] = "train-patient-0"
                    for directory in self.runs.values():
                        split.to_csv(directory / "split.csv", index=False)
                elif defect == "last_epoch":
                    payload = deepcopy(self.payloads["F"]["last"])
                    payload["epoch"] = 3
                    torch.save(payload, run / "last.pt")
                else:
                    payload = deepcopy(self.payloads["F"]["best"])
                    payload["best_val_patient_roc_auc"] = .99
                    torch.save(payload, run / "best.pt")
                    self.state["jobs"][job_id]["result"]["checkpoint_sha256"] = self._sha(run / "best.pt")
                self._save_state()
                with mock.patch.object(CXRDataset, "__getitem__", side_effect=AssertionError("invalid suite opened pixels")), mock.patch.object(
                    distillation, "build_vit_tiny_student", side_effect=AssertionError("invalid suite reconstructed student")
                ), redirect_stdout(io.StringIO()), self.assertRaises((ValueError, FileNotFoundError)):
                    checker.audit(self.suite, device="cpu")

    def test_plots_keep_validation_history_and_only_two_unconnected_clean_train_observations(self):
        from matplotlib.figure import Figure

        plots = []
        original = Figure.savefig

        def capture(figure, filename, *args, **kwargs):
            axes = []
            for axis in figure.axes:
                lines = [(line.get_label().lower(), len(line.get_xdata()), line.get_linestyle()) for line in axis.lines]
                points = [(artist.get_label().lower(), len(artist.get_offsets())) for artist in axis.collections]
                axes.append((lines, points))
            plots.append((Path(filename), axes))
            return original(figure, filename, *args, **kwargs)

        with mock.patch.object(Figure, "savefig", new=capture):
            output = self._audit()
        pngs = [(path, axes) for path, axes in plots if path.suffix == ".png"]
        self.assertEqual(len(pngs), 3)
        for path, axes in pngs:
            self.assertTrue(path.is_relative_to(output))
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(len(axes), 6)
            for lines, points in axes:
                self.assertTrue(any(length == 3 and "val" in label for label, length, style in lines))
                train_lines = [(length, style) for label, length, style in lines if "train" in label]
                self.assertTrue(all(length <= 2 and style in ("None", "none", "", " ") for length, style in train_lines))
                train_points = [length for label, length in points if "train" in label]
                self.assertEqual(sum(train_points) + sum(length for length, style in train_lines), 2)

    def test_cli_resolves_project_relative_suite_and_keeps_automatic_device_selection(self):
        with mock.patch.object(checker, "audit", return_value=self.suite / "report") as audit, redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(["--project", str(self.root), "--suite", "suite"]), 0)
        audit.assert_called_once_with(self.suite, device="auto")


if __name__ == "__main__":
    unittest.main()
