"""Held-out inference from complete synthetic students, without training a teacher."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_single_teacher_distillation import SmallTokenStudent

import run_single_teacher_test_evaluation as evaluator
from foundation_sft import distillation, metrics, single_teacher_overfit
from foundation_sft import train as trainer
from foundation_sft.data import CXRDataset
from foundation_sft.distillation_data import SharedViewDataset
from foundation_sft.models.common import PreprocessSpec


class SingleTeacherTestEvaluationTests(unittest.TestCase):
    """Exercise real decoding, shared cropping, student reload and patient metrics."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="single-teacher-test-eval-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.split = self._dicoms()
        self.teacher_spec = PreprocessSpec(8, resize_mode="shortest_edge_center_crop", resize_shortest_edge=12)
        self.geometry = SharedViewDataset(
            CXRDataset(self.split, self.teacher_spec, training=False), student_size=8,
            include_teacher=False,
        ).metadata()
        self.thresholds = dict(zip("ABCD", (.31, .42, .53, .64)))
        self.payloads = {}
        self.runs = {}
        self.state = self._completed_study()
        self._save_state()

    @staticmethod
    def _json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _dicoms(self):
        directory = self.root / "dicoms"
        directory.mkdir()
        rows = []
        # Unequal image counts exercise patient means and cluster bootstrapping.
        for partition in ("train", "val", "test"):
            for label in (0, 1):
                patient = f"{partition}-patient-{label}"
                for image_index in range(1 + label):
                    sample = f"{patient}-image-{image_index}"
                    path = directory / (sample + ".dcm")
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
                    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                    dataset.PatientID = patient
                    dataset.AccessionNumber = f"{partition[0]}-{label}-{image_index}"
                    dataset.Rows, dataset.Columns = 16, 24
                    dataset.SamplesPerPixel = 1
                    dataset.PhotometricInterpretation = "MONOCHROME2"
                    dataset.BitsAllocated = dataset.BitsStored = 16
                    dataset.HighBit, dataset.PixelRepresentation = 15, 0
                    y, x = np.mgrid[:16, :24]
                    pixels = (30 * x + 7 * y + 150 * label + image_index * x * y).astype(np.uint16)
                    dataset.PixelData = pixels.tobytes()
                    options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                               else {"write_like_original": False})
                    pydicom.dcmwrite(path, dataset, **options)
                    rows.append({"sample_id": sample, "patient_id": patient, "accession": str(dataset.AccessionNumber),
                                 "path": str(path), "label": label, "split": partition})
        return pd.DataFrame(rows)

    def _completed_study(self):
        source = self.root / "historical_source"
        source.mkdir()
        self.split.to_csv(source / "split.csv", index=False)
        metadata = SmallTokenStudent(initialization={"pretraining": "ImageNet-1k",
            "checkpoint": "/unavailable/original-imagenet.pth", "checkpoint_sha256": "b" * 64}).training_metadata()
        source_info = {"cohort_hash": "c" * 64, "initial_state_sha256": "a" * 64,
                       "model_metadata": metadata, "shared_view": self.geometry,
                       "files": {str(source / "split.csv"): self._sha(source / "split.csv")}}
        plan = {"project": str(self.root), "source_suite": str(source), "source": source_info, "jobs": []}
        state = {"format_version": single_teacher_overfit.FORMAT, "status": "completed", "plan": plan, "jobs": {}}
        teacher = {"checkpoint": "/unavailable/teacher.pt", "checkpoint_sha256": "d" * 64}
        for index, group in enumerate("ABCD"):
            condition = "supervised" if group == "A" else "distillation"
            diagnostics = {"protocol": "single_teacher_overfit_v1", "evaluation": "validation_only"}
            if group == "D":
                diagnostics["l2_sp"] = {"coefficient": .0001, "reduction": "sum"}
            config = {"seed": 42, "device": "cpu", "student": {"mode": "full_finetune"},
                "student_checkpoint": "/unavailable/original-imagenet.pth", "teacher_checkpoint": "/unavailable/teacher.pt",
                "training": {"batch_size": 2, "num_workers": 0}, "diagnostics": diagnostics,
                "evaluation": {"bootstrap_samples": 12, "patient_aggregation": "mean"},
                "distillation": dict(zip(single_teacher_overfit.WEIGHTS, single_teacher_overfit.GROUPS[group][1]))}
            job = {"id": group + "_seed42", "group": group, "condition": condition, "seed": 42,
                   "student_mode": "full_finetune", "device": "cpu", "config": config}
            run = self.study / "runs" / group
            run.mkdir(parents=True)
            torch.manual_seed(50 + index)
            student = SmallTokenStudent(initialization=metadata["initialization"])
            payload = distillation._student_payload(student, config, source_info["cohort_hash"], condition,
                source_info["initial_state_sha256"], teacher, self.geometry, diagnostics_proof=diagnostics)
            payload.update(threshold=self.thresholds[group], best_epoch=2, last_epoch=3, best_val_patient_roc_auc=.81)
            torch.save(payload, run / "best.pt")
            receipt = {name: deepcopy(payload[name]) for name in ("condition", "cohort_hash", "initial_state_sha256",
                "threshold", "best_epoch", "last_epoch", "best_val_patient_roc_auc", "evaluation_phase", "test_evaluated", "diagnostics")}
            receipt.update(status="completed", seed=42, run=str(run), teacher_unchanged=True)
            result = deepcopy(receipt)
            history = [{"epoch": epoch, "val_patient_roc_auc": score} for epoch, score in ((1, .72), (2, .81), (3, .78))]
            for name, value in (("config", config), ("result", result), ("model_metadata", metadata),
                                ("shared_view", self.geometry), ("split_metadata", {"_manifest_hash": source_info["cohort_hash"]}),
                                ("history", history), ("selection", {"threshold": self.thresholds[group],
                                 "source": "validation patient predictions", "best_val_patient_roc_auc": .81})):
                self._json(run / (name + ".json"), value)
            self.split.to_csv(run / "split.csv", index=False)
            receipt.update(checkpoint_sha256=self._sha(run / "best.pt"), history_sha256=self._sha(run / "history.json"))
            state["jobs"][job["id"]] = {"job": deepcopy(job), "status": "completed", "receipt": receipt}
            plan["jobs"].append(job)
            self.payloads[group], self.runs[group] = payload, run
        return state

    def _save_state(self):
        self.state["protocol_sha256"] = single_teacher_overfit.protocol_hash(self.state["plan"])
        self._json(self.study / "study.json", self.state)

    def _source_bytes(self):
        return {path: path.read_bytes() for path in self.root.rglob("*")
                if path.is_file() and "report" not in path.relative_to(self.root).parts}

    def _evaluate(self):
        with redirect_stdout(io.StringIO()), mock.patch.object(
            distillation, "build_vit_tiny_student", side_effect=SmallTokenStudent
        ):
            return evaluator.evaluate(self.study, device="cpu")

    def test_real_test_inference_keeps_saved_threshold_geometry_and_sources_then_reuses_cache(self):
        before = self._source_bytes()
        opened = []
        original_item = CXRDataset.__getitem__

        def test_only(dataset, index):
            row = dataset.cohort.iloc[index]
            self.assertEqual(row["split"], "test")
            self.assertFalse(dataset.training)
            opened.append(row["sample_id"])
            return original_item(dataset, index)

        with ExitStack() as stack:
            for module, name in ((distillation, "load_teacher"), (distillation, "_train_epoch"),
                                 (distillation, "select_threshold"), (trainer, "select_threshold"),
                                 (metrics, "select_threshold")):
                stack.enter_context(mock.patch.object(module, name, side_effect=AssertionError("evaluation trained or selected on test")))
            stack.enter_context(mock.patch.object(CXRDataset, "__getitem__", new=test_only))
            table = self._evaluate()
        self.assertEqual(table["group"].tolist(), list("ABCD"))
        self.assertEqual(len(opened), 4 * len(self.split.loc[self.split["split"] == "test"]))
        output = self.study / "report" / "test_evaluation_seed42"
        test = self.split.loc[self.split["split"] == "test"]
        for group in "ABCD":
            run = self.runs[group]
            self.assertFalse((run / "test").exists())
            saved = pd.read_csv(output / group / "predictions.csv")
            # Construct the historical two-stage image path independently of
            # the evaluator's inference_dataset helper, then compare actual logits.
            expected_student = SmallTokenStudent(initialization=self.payloads[group]["model_metadata"]["initialization"])
            expected_student.load_trainable_state_dict(self.payloads[group]["model"])
            shared = SharedViewDataset(CXRDataset(test, self.teacher_spec, training=False),
                                       student_size=8, include_teacher=False)
            expected = trainer.predict(expected_student, DataLoader(shared, batch_size=2), torch.device("cpu"))
            pd.testing.assert_frame_equal(saved, expected, check_exact=False, rtol=0, atol=1e-14)
            direct = CXRDataset(test, expected_student.preprocess, training=False)
            self.assertFalse(torch.allclose(shared[0]["image"], direct[0]["image"]))
            report = json.loads((output / group / "metrics.json").read_text())
            self.assertEqual(report["patient"]["threshold"], self.thresholds[group])
            self.assertEqual(report["patient"]["n_samples"], 2)
            self.assertEqual(report["image"]["n_samples"], 3)
            self.assertEqual(report["bootstrap"]["unit"], "patient")
            self.assertEqual(report["bootstrap"]["samples"], 12)
            for name in evaluator.ARTIFACTS:
                self.assertGreater((output / group / name).stat().st_size, 0)
        self.assertEqual(before, self._source_bytes())
        all_files = {path: path.read_bytes() for path in self.study.rglob("*") if path.is_file()}
        with mock.patch.object(evaluator, "_load_payload", side_effect=AssertionError("cache reloaded checkpoint")), mock.patch.object(
            distillation, "load_student_checkpoint", side_effect=AssertionError("cache reconstructed model")
        ), mock.patch.object(trainer, "predict", side_effect=AssertionError("cache repeated inference")), mock.patch.object(
            CXRDataset, "__getitem__", side_effect=AssertionError("cache opened image")
        ):
            pd.testing.assert_frame_equal(evaluator.evaluate(self.study, device="cpu"), table)
            pd.testing.assert_frame_equal(evaluator.show_metrics(self.study), table)
        self.assertEqual(all_files, {path: path.read_bytes() for path in self.study.rglob("*") if path.is_file()})

    def test_invalid_sources_fail_before_any_test_dataset_is_opened(self):
        baseline_state = deepcopy(self.state)
        baseline_files = self._source_bytes()
        for defect in ("incomplete", "checkpoint_sha", "metadata", "threshold", "different_split", "patient_overlap"):
            with self.subTest(defect=defect):
                self.state = deepcopy(baseline_state)
                for path, contents in baseline_files.items():
                    path.write_bytes(contents)
                if defect == "incomplete":
                    self.state["jobs"]["D_seed42"]["status"] = "training"
                elif defect == "checkpoint_sha":
                    self.state["jobs"]["D_seed42"]["receipt"]["checkpoint_sha256"] = "0" * 64
                elif defect == "metadata":
                    altered = deepcopy(self.state["plan"]["source"]["model_metadata"])
                    altered["feature_dim"] += 1
                    self._json(self.runs["D"] / "model_metadata.json", altered)
                elif defect == "threshold":
                    payload = deepcopy(self.payloads["D"])
                    payload["threshold"] = .99
                    torch.save(payload, self.runs["D"] / "best.pt")
                    self.state["jobs"]["D_seed42"]["receipt"]["checkpoint_sha256"] = self._sha(self.runs["D"] / "best.pt")
                else:
                    # Remove the original byte pin here to specifically test
                    # cross-arm membership and patient leakage validation.
                    self.state["plan"]["source"]["files"] = {}
                    split = self.split.copy()
                    if defect == "different_split":
                        split.loc[split["split"] == "test", "path"] = "/different/image.dcm"
                        split.to_csv(self.runs["D"] / "split.csv", index=False)
                    else:
                        split.loc[split["split"] == "test", "patient_id"] = split.loc[split["split"] == "train", "patient_id"].iloc[0]
                        for run in self.runs.values():
                            split.to_csv(run / "split.csv", index=False)
                self._save_state()
                with mock.patch.object(CXRDataset, "__getitem__", side_effect=AssertionError("invalid study opened pixels")), mock.patch.object(
                    evaluator, "_predict", side_effect=AssertionError("invalid study reached inference")
                ), self.assertRaises(ValueError):
                    evaluator.evaluate(self.study, device="cpu")

    def test_show_metrics_imports_no_torch_and_cached_artifact_changes_are_rejected(self):
        self._evaluate()
        script = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('basic reporting imported Torch')
sys.meta_path.insert(0, NoTorch())
import run_single_teacher_test_evaluation as evaluator
evaluator.main(['--study', sys.argv[1], '--show-metrics'])
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.study)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TEST", result.stdout)
        self.assertIn("sensitivity", result.stdout)
        path = self.study / "report" / "test_evaluation_seed42" / "B" / "predictions.csv"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with mock.patch.object(evaluator, "_predict", side_effect=AssertionError("changed cache repeated inference")), self.assertRaises(ValueError):
            evaluator.evaluate(self.study, device="cpu")


if __name__ == "__main__":
    unittest.main()
