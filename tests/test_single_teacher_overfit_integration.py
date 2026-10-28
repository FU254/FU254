"""Real worker/coordinator integration on local synthetic model weights and DICOMs.

The small student declares the ImageNet provenance fields needed to exercise
the production coordinator contract. Its weights remain synthetic test data.
No worker histories, evaluation results, verifiers or checkpoints are faked.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as single_fixture

from foundation_sft import distillation, distillation_suite, single_teacher_overfit
from foundation_sft import train as trainer
from foundation_sft.models import build_adapter
from foundation_sft.models.common import PreprocessSpec, sha256_file


class SevenBySevenStudent(single_fixture.SmallTokenStudent):
    """Complete tiny model with a seven-by-seven local-token grid."""

    def __init__(self, checkpoint=None, *, initialization=None):
        super().__init__(checkpoint, initialization=initialization)
        self.preprocess = PreprocessSpec(28)
        if initialization is None:
            self.initialization.update(pretraining="ImageNet-1k",
                                       checkpoint_sha256=sha256_file(checkpoint))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF teacher needs pinned Transformers")
class SingleTeacherOverfitIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture = single_fixture.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.split = fixture.split
        self.config = deepcopy(fixture.config)
        self.config.update(seed=42, student={"mode": "full_finetune"},
                           output_root=str(self.root / "original_worker"))
        self.config["training"].update(epochs=100, patience=15, backbone_lr=5e-5,
                                        head_lr=1e-4, projection_lr=1e-4,
                                        brightness_jitter=0.12, contrast_jitter=0.15)
        self.config["distillation"]["pool_grid"] = [7, 7]
        processor_path = Path(fixture.teacher_config["model_dir"]) / "preprocessor_config.json"
        processor = json.loads(processor_path.read_text())
        processor.update(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28})
        processor_path.write_text(json.dumps(processor))
        teacher = trainer._classifier(build_adapter(fixture.teacher_config), fixture.teacher_config)
        teacher.load_trainable_state_dict(fixture.teacher_state)
        payload = trainer._checkpoint(teacher, fixture.teacher_config, trainer._split_hash(self.split))
        payload.update(threshold=0.43, best_val_patient_roc_auc=0.71)
        torch.save(payload, fixture.teacher_checkpoint)

    @staticmethod
    def json_file(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def file_hashes(paths):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def test_genuine_source_four_workers_default_verification_and_completed_resume(self):
        calls, data_splits = [], []
        original_item = distillation.CXRDataset.__getitem__

        def forbid_test_images(dataset, index):
            split_name = dataset.cohort.iloc[index]["split"]
            self.assertNotEqual(split_name, "test", "diagnostic study opened a test image")
            data_splits.append(split_name)
            return original_item(dataset, index)

        def launch(job, action, project, directory):
            self.assertEqual(project, self.root)
            directory.mkdir(parents=True, exist_ok=False)
            config = deepcopy(job["config"])
            config["output_root"] = str(directory / "outputs")
            self.json_file(directory / "config.json", config)
            calls.append((job["id"], action))
            with (directory / (action + ".log")).open("w", encoding="utf-8") as log, redirect_stdout(log):
                if action == "check":
                    checked = distillation.check(config, project, condition=job["condition"])
                    result = json.loads((checked / "check.json").read_text())
                    result.update(action="check", directory=str(checked))
                else:
                    suite = distillation.run_experiments(config, project, conditions=[job["condition"]])
                    result = {"suite": str(suite), **json.loads((suite / "suite.json").read_text())}
            self.json_file(directory / "result.json", result)
            return result

        with redirect_stdout(io.StringIO()), mock.patch.object(
            distillation, "build_vit_tiny_student", side_effect=SevenBySevenStudent
        ):
            # A real unmodified native run supplies the historical source.
            # Its legacy test evaluation completes before the study's guard.
            source = distillation.run_experiments(self.config, self.root, conditions=["distillation"])
            self.assertTrue((source / "distillation" / "test" / "metrics.json").is_file())
            settings = self.root / "overfit_study.json"
            self.json_file(settings, {"source_suite": str(source), "seed": 42,
                "devices": ["cpu"], "max_parallel": 1,
                "output_root": str(self.root / "fresh_overfit_study"), "l2_sp_coefficient": 0.0001})
            plan = single_teacher_overfit.make_plan(settings, self.root)
            pinned_paths = {Path(path) for path in plan["source"]["files"]}
            pinned_paths.update(path for path in source.rglob("*") if path.is_file())
            before = self.file_hashes(pinned_paths)
            with mock.patch.object(distillation.CXRDataset, "__getitem__", new=forbid_test_images), mock.patch.object(
                distillation_suite, "_default_checkpoint_loader", wraps=distillation_suite._default_checkpoint_loader
            ) as checkpoint_loader:
                study = single_teacher_overfit.run_study(plan, launcher=launch)
                completed = json.loads((study / "study.json").read_text())
                self.assertEqual(completed["status"], "completed", completed)
                self.assertEqual([record["status"] for record in completed["jobs"].values()],
                                 ["completed"] * 4)
                self.assertEqual(checkpoint_loader.call_count, 4)
                self.assertEqual(calls, [(group + "_seed42", "check") for group in "ABCD"]
                                       + [(group + "_seed42", "train") for group in "ABCD"])
                for group in "ABCD":
                    record = completed["jobs"][group + "_seed42"]
                    receipt = record["receipt"]
                    self.assertEqual(receipt["evaluation_phase"], "validation_only")
                    self.assertFalse(receipt["test_evaluated"])
                    self.assertNotIn("test_patient", receipt)
                    run = Path(receipt["run"])
                    self.assertFalse((run / "test").exists())
                    history = json.loads((run / "history.json").read_text())
                    self.assertGreaterEqual(len(history), 16)
                    self.assertEqual(history[-1]["epoch"], receipt["last_epoch"])
                    self.assertEqual(receipt["last_epoch"] - receipt["best_epoch"], 15)
                    self.assertEqual(receipt["initial_state_sha256"], plan["source"]["initial_state_sha256"])
                    if group == "C":
                        self.assertTrue(all(row["train_global_loss"] == row["train_local_loss"] == 0
                                            for row in history))
                    if group == "D":
                        self.assertTrue(any(row["train_l2_sp_penalty"] > 0 for row in history))
                        self.assertTrue(receipt["diagnostics"]["l2_sp"]["anchor_unchanged"])
                        self.assertEqual(receipt["diagnostics"]["l2_sp"]["coefficient"], 0.0001)
                    self.assertTrue((run / "best.pt").is_file())
                    self.assertTrue((run / "last.pt").is_file())
                    for evaluated in ("train_eval", "validation"):
                        self.assertTrue((run / evaluated / "metrics.json").is_file())
                        self.assertTrue((run / evaluated / "predictions.csv").is_file())
                training_calls = [call for call in calls if call[1] == "train"]
                # A completed resume reruns real preflight and reloads each
                # saved best checkpoint, with no optimizer or evaluation pass.
                with mock.patch.object(trainer, "predict", side_effect=AssertionError("resume evaluated data")), mock.patch.object(
                    distillation, "_train_epoch", side_effect=AssertionError("resume restarted training")
                ):
                    resumed_study = single_teacher_overfit.run_study(plan, resume_study=study, launcher=launch)
                self.assertEqual(resumed_study, study)
                resumed = json.loads((study / "study.json").read_text())
                self.assertEqual(resumed["status"], "completed", resumed)
                self.assertTrue(all(record["status"] == "resumed" for record in resumed["jobs"].values()))
                self.assertEqual([call for call in calls if call[1] == "train"], training_calls)
                self.assertEqual(checkpoint_loader.call_count, 8)
                self.assertEqual(len([call for call in calls if call[1] == "check"]), 8)
            self.assertEqual(before, self.file_hashes(pinned_paths))
            self.assertEqual(set(data_splits), {"train", "val"})


if __name__ == "__main__":
    unittest.main()
