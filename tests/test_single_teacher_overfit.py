"""File-backed provenance and worker orchestration without model dependencies."""

from contextlib import redirect_stdout
from copy import deepcopy
import builtins
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from foundation_sft import single_teacher_overfit as study_api


class SingleTeacherOverfitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.source = self.project / "runs" / "single_teacher_distillation" / "suites" / "suite_20261019T110343Z_3b1acd96"
        self.source_run = self.source / "distillation"
        self.source_run.mkdir(parents=True)
        self.teacher_file = self.project / "weights" / "rad_dino.pt"
        self.student_file = self.project / "weights" / "deit_tiny.pth"
        self.split_file = self.project / "frozen_split.csv"
        self._bytes(self.teacher_file, b"the fixed RAD-DINO teacher")
        self._bytes(self.student_file, b"the fixed ImageNet encoder")
        self._bytes(self.split_file, b"sample_id,patient_id,label,split\nt1,p1,0,train\nv1,p2,1,val\ns1,p3,0,test\n")
        self.cohort_hash = "c" * 64
        self.initial_hash = "a" * 64
        self.training = {
            "epochs": 100, "patience": 15, "batch_size": 8, "accumulation_steps": 8,
            "num_workers": 4, "backbone_lr": 5e-5, "head_lr": 1e-4,
            "projection_lr": 1e-4, "amp": False, "weight_decay": .037,
            "minimum_lr": 1e-7, "focal_gamma": 2., "focal_scale": .8,
            "gradient_clip": 1., "rotation_degrees": 7.5,
            "horizontal_flip_probability": .25, "brightness_jitter": .12,
            "contrast_jitter": .15,
        }
        self.loss = {"temperature": 2., "soft_weight": .5, "global_weight": .1,
                     "local_weight": 1., "pool_grid": [7, 7], "eps": 1e-6}
        self.metadata = {"model": "vit_tiny_student", "mode": "full_finetune",
            "architecture": "deit_tiny_patch16_224", "feature_dim": 192,
            "initialization": {"pretraining": "ImageNet-1k",
                "checkpoint": str(self.student_file),
                "checkpoint_sha256": self._sha(self.student_file)},
            "trainable_parameters": 5524609, "total_parameters": 5524609}
        self.teacher = {"checkpoint": str(self.teacher_file),
                        "checkpoint_sha256": self._sha(self.teacher_file),
                        "feature_dim": 768}
        self.geometry = {"shared_field_of_view": True,
            "teacher_preprocessing": {"image_size": 518, "resize_mode": "shortest_edge_center_crop"},
            "student_preprocessing": {"image_size": 224, "resize_mode": "resize_shared_teacher_view"}}
        self.config = {"seed": 42, "device": "cuda:0", "cache_dir": str(self.project / "old_cache"),
            "output_root": str(self.project / "historical_outputs"),
            "teacher_checkpoint": str(self.teacher_file), "student_checkpoint": str(self.student_file),
            "student": {"mode": "full_finetune"}, "training": deepcopy(self.training),
            "distillation": deepcopy(self.loss),
            "evaluation": {"bootstrap_samples": 25, "patient_aggregation": "mean"},
            "teacher_config": {"split_manifest": str(self.split_file), "split_seed": 42}}
        self.source_result = {"status": "completed", "condition": "distillation",
            "run": str(self.source_run), "seed": 42, "initial_state_sha256": self.initial_hash,
            "cohort_hash": self.cohort_hash, "teacher_unchanged": True,
            "best_val_patient_roc_auc": .83}
        for name, value in (("config", self.config), ("model_metadata", self.metadata),
                            ("teacher", self.teacher), ("shared_view", self.geometry),
                            ("split_metadata", {"_manifest_hash": self.cohort_hash}),
                            ("result", self.source_result)):
            self._json(self.source_run / (name + ".json"), value)
        self._bytes(self.source_run / "split.csv", self.split_file.read_bytes())
        self._bytes(self.source_run / "best.pt", b"historical trained student: never imported")
        self._json(self.source_run / "history.json", [{"epoch": 1, "val_patient_roc_auc": .83}])
        self._json(self.source / "suite.json", {"status": "completed",
            "conditions": {"distillation": self.source_result}})
        self.settings_file = self.project / "study_config.json"
        self.settings = {"source_suite": str(self.source), "seed": 42,
            "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], "max_parallel": 4,
            "output_root": str(self.project / "new_study"), "l2_sp_coefficient": 1e-4}
        self._json(self.settings_file, self.settings)
        self.calls = []
        self.call_lock = threading.Lock()
        self.payloads = {}
        self.fail_training = set()
        self.bad_initial = set()
        self.bad_projector = set()

    @staticmethod
    def _bytes(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def plan(self, **kwargs):
        return study_api.make_plan(self.settings_file, self.project, **kwargs)

    def check(self, job):
        value = {"check_passed": True, "condition": job["condition"],
            "config": deepcopy(job["config"]), "model_metadata": deepcopy(self.metadata),
            "initial_state_sha256": self.initial_hash, "cohort_hash": self.cohort_hash,
            "teacher": deepcopy(self.teacher), "shared_view": deepcopy(self.geometry),
            "teacher_unchanged": True, "student_full_checkpoint_roundtrip": True,
            "student_state_exact": True, "evaluation_phase": "validation_only",
            "test_evaluated": False,
            "initial_backbone_state_sha256": "f" * 64,
            "initial_head_state_sha256": "1" * 64,
            "diagnostics": deepcopy(job["config"]["diagnostics"])}
        value["diagnostics"].update(training_rng_preserved=True, clean_train_eval={
            "split": "train", "training": False, "shuffle": False, "include_teacher": False,
            "private_generator": True, "shared_view": deepcopy(self.geometry),
            "augmentation": {"rotation_degrees": 0., "horizontal_flip_probability": 0.,
                             "brightness_jitter": 0., "contrast_jitter": 0.}})
        if job["group"] == "D":
            value["diagnostics"]["l2_sp"].update(anchor_sha256="f" * 64, anchor_unchanged=True)
        if job["group"] in ("B", "D"):
            value["distillation_initial_state_sha256"] = "d" * 64
        elif job["group"] == "C":
            value["distillation_initial_state_sha256"] = hashlib.sha256(b"").hexdigest()
        if job["id"] in self.bad_initial:
            value["initial_state_sha256"] = "b" * 64
        if job["id"] in self.bad_projector:
            value["distillation_initial_state_sha256"] = "e" * 64
        return value

    def write_run(self, job, directory):
        directory.mkdir(parents=True, exist_ok=False)
        run = directory / "outputs" / "suite" / job["condition"]
        run.mkdir(parents=True)
        check = self.check(job)
        history = [{"epoch": epoch, "train_loss": 99. - epoch,
            "train_eval_loss": .2 - epoch * .01, "train_eval_patient_roc_auc": .8 + epoch * .01,
            "train_eval_patient_pr_auc": .6 + epoch * .01,
            "val_loss": .22 + epoch * .01, "val_patient_roc_auc": auc,
            "val_patient_pr_auc": .55, "optimizer_steps": 2}
            for epoch, auc in ((epoch, .80 if epoch == 1 else .83) for epoch in range(1, 18))]
        if job["group"] == "D":
            for row in history:
                row.update(train_l2_sp_loss=float(row["epoch"]), train_l2_sp_penalty=1e-4 * row["epoch"])
        patient = {"roc_auc": .83, "pr_auc": .55, "n_samples": 10,
                   "n_positive": 4, "n_negative": 6, "threshold": .4,
                   "average_precision": .57, "sensitivity": .75, "specificity": .67}
        train_patient = {**patient, "roc_auc": .82, "pr_auc": .62}
        result = {"status": "completed", "condition": job["condition"], "run": str(run),
            "seed": 42, "initial_state_sha256": self.initial_hash, "cohort_hash": self.cohort_hash,
            "teacher_unchanged": True, "evaluation_phase": "validation_only", "test_evaluated": False,
            "best_epoch": 2, "last_epoch": 17, "best_val_patient_roc_auc": .83,
            "threshold": .4, "validation_patient": patient, "best_train_patient": train_patient,
            "diagnostics": deepcopy(check["diagnostics"])}
        for key in ("initial_backbone_state_sha256", "initial_head_state_sha256", "distillation_initial_state_sha256"):
            if key in check:
                result[key] = check[key]
        for name, row in (("best_gap", history[1]), ("last_gap", history[-1])):
            result[name] = {
                "auc_gap": row["train_eval_patient_roc_auc"] - row["val_patient_roc_auc"],
                "loss_gap": row["val_loss"] - row["train_eval_loss"],
                "auc_gap_definition": "clean train patient ROC-AUC minus validation patient ROC-AUC",
                "loss_gap_definition": "validation Focal loss minus clean train Focal loss"}
        for name, value in (("config", job["config"]), ("teacher", self.teacher),
                            ("model_metadata", self.metadata), ("shared_view", self.geometry),
                            ("history", history), ("result", result)):
            self._json(run / (name + ".json"), value)
        for split, metrics in (("validation", patient), ("train_eval", train_patient)):
            self._json(run / split / "metrics.json", {"patient": metrics})
            self._bytes(run / split / "predictions.csv", b"sample_id,patient_id,label,probability\na,p,0,0.2\n")
            self._bytes(run / split / "patient_predictions.csv", b"patient_id,label,probability\np,0,0.2\n")
        self._bytes(run / "best.pt", ("saved model " + job["id"]).encode())
        self._bytes(run / "last.pt", ("last model " + job["id"]).encode())
        self.payloads[str(run.resolve())] = {"config": deepcopy(job["config"]),
            "condition": job["condition"], "initial_state_sha256": self.initial_hash,
            "cohort_hash": self.cohort_hash, "teacher": deepcopy(self.teacher),
            "model_metadata": deepcopy(self.metadata), "shared_view": deepcopy(self.geometry),
            "best_val_patient_roc_auc": .83, "threshold": .4,
            "diagnostics": deepcopy(result["diagnostics"]),
            "evaluation_phase": "validation_only", "test_evaluated": False}
        for key in ("initial_backbone_state_sha256", "initial_head_state_sha256", "distillation_initial_state_sha256"):
            if key in check:
                self.payloads[str(run.resolve())][key] = check[key]
        self._json(directory / "result.json", result)
        return result

    def launch(self, job, action, project, directory):
        self.assertEqual(project, self.project)
        with self.call_lock:
            self.calls.append((job["id"], action))
        if action == "check":
            directory.mkdir(parents=True, exist_ok=False)
            value = self.check(job)
            self._json(directory / "result.json", value)
            return value
        if job["id"] in self.fail_training:
            directory.mkdir(parents=True, exist_ok=False)
            raise RuntimeError("synthetic worker failure for " + job["id"])
        return self.write_run(job, directory)

    def verify(self, run, check, job):
        return study_api.verify_run(run, check, job,
            checkpoint_loader=lambda path: deepcopy(self.payloads[str(Path(path).parent.resolve())]))

    def run_study(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return study_api.run_study(plan or self.plan(), launcher=self.launch,
                verifier=self.verify, reporter=lambda *_: None, **kwargs)

    def state(self, study):
        return json.loads((study / "study.json").read_text())

    def test_planning_is_read_only_and_copies_original_effective_recipe(self):
        files_before = {path: path.read_bytes() for path in self.project.rglob("*") if path.is_file()}
        plan = self.plan()
        self.assertEqual(files_before, {path: path.read_bytes() for path in self.project.rglob("*") if path.is_file()})
        self.assertEqual(plan["evaluation_policy"], "validation_only")
        self.assertEqual(plan["new_optimization_trainings"], 4)
        self.assertEqual([job["id"] for job in plan["jobs"]], ["A_seed42", "B_seed42", "C_seed42", "D_seed42"])
        self.assertEqual([job["device"] for job in plan["jobs"]], self.settings["devices"])
        for job in plan["jobs"]:
            config = deepcopy(job["config"])
            expected = deepcopy(self.config)
            for key in ("device", "output_root"):
                config.pop(key)
                expected.pop(key)
            diagnostics = config.pop("diagnostics")
            self.assertEqual(diagnostics["evaluation"], "validation_only")
            self.assertEqual("l2_sp" in diagnostics, job["group"] == "D")
            if job["group"] == "D":
                self.assertEqual(diagnostics["l2_sp"], {"coefficient": 1e-4, "reduction": "sum"})
            expected["distillation"].update(dict(zip(study_api.WEIGHTS, study_api.GROUPS[job["group"]][1])))
            self.assertEqual(config, expected)
            self.assertEqual(config["training"]["weight_decay"], .037)
            self.assertEqual(config["training"]["brightness_jitter"], .12)

    def test_scheduling_and_output_changes_preserve_scientific_hash(self):
        original = self.plan()
        changed = self.plan(devices=["cuda:7"], max_parallel=1, cache_dir="different_cache")
        changed["output_root"] = str(self.project / "other_outputs")
        for job in changed["jobs"]:
            job["config"]["output_root"] = changed["output_root"]
        self.assertEqual(study_api.protocol_hash(original), study_api.protocol_hash(changed))
        changed["jobs"][0]["config"]["training"]["weight_decay"] = .01
        self.assertNotEqual(study_api.protocol_hash(original), study_api.protocol_hash(changed))

    def test_planning_never_imports_tensor_or_model_libraries(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "timm", "transformers"}:
                raise AssertionError("Planning tried to import model runtime " + name)
            return original_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 4)

    def test_source_mutation_is_rejected_before_workers_start(self):
        plan = self.plan()
        self.teacher_file.write_bytes(b"different teacher")
        with self.assertRaisesRegex(ValueError, "source artifact changed"):
            self.run_study(plan)
        self.assertEqual(self.calls, [])

    def test_no_template_fallback_for_missing_or_unresolved_source(self):
        config_path = self.source_run / "config.json"
        config_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        incomplete = deepcopy(self.config)
        incomplete["training"].pop("weight_decay")
        self._json(config_path, incomplete)
        with self.assertRaisesRegex(ValueError, "complete resolved effective"):
            self.plan()

    def test_source_random_student_and_altered_checkpoint_are_rejected(self):
        changed = deepcopy(self.metadata)
        changed["initialization"]["pretraining"] = "none"
        self._json(self.source_run / "model_metadata.json", changed)
        with self.assertRaisesRegex(ValueError, "original ImageNet"):
            self.plan()
        self._json(self.source_run / "model_metadata.json", self.metadata)
        self.student_file.write_bytes(b"different ImageNet weights")
        with self.assertRaisesRegex(ValueError, "ImageNet checkpoint checksum"):
            self.plan()

    def test_recipe_overrides_and_output_inside_source_are_rejected(self):
        for changes in ({"training": {"weight_decay": .01}}, {"seed": 43},
                        {"l2_sp_coefficient": .001}, {"output_root": str(self.source / "new")}):
            with self.subTest(changes=changes):
                self._json(self.settings_file, {**self.settings, **changes})
                with self.assertRaises(ValueError):
                    self.plan()

    def test_four_checks_precede_training_and_completed_resume_does_not_retrain(self):
        plan = self.plan()
        study = self.run_study(plan)
        state = self.state(study)
        self.assertEqual(state["status"], "completed")
        self.assertEqual({record["status"] for record in state["jobs"].values()}, {"completed"})
        first_training = next(index for index, (_, action) in enumerate(self.calls) if action == "train")
        self.assertEqual(first_training, 4)
        self.assertEqual(len(self.calls), 8)
        first_receipts = {key: record["receipt"] for key, record in state["jobs"].items()}
        self.calls.clear()
        updated_plan = self.plan(devices=["cuda:5", "cuda:6"], max_parallel=2)
        resumed = self.run_study(updated_plan, resume_study=study)
        self.assertEqual(resumed, study)
        self.assertEqual([action for _, action in self.calls], ["check"] * 4)
        state = self.state(study)
        self.assertEqual(state["status"], "completed")
        self.assertEqual({record["status"] for record in state["jobs"].values()}, {"resumed"})
        self.assertEqual({key: record["receipt"] for key, record in state["jobs"].items()}, first_receipts)

    def test_check_only_can_resume_into_exactly_four_trainings(self):
        plan = self.plan()
        study = self.run_study(plan, check_only=True)
        self.assertEqual(self.state(study)["status"], "checked")
        self.assertEqual([action for _, action in self.calls], ["check"] * 4)
        self.calls.clear()
        self.run_study(plan, resume_study=study)
        self.assertEqual(self.state(study)["status"], "completed")
        self.assertEqual(sum(action == "train" for _, action in self.calls), 4)

    def test_partial_worker_failure_preserves_success_and_only_retries_failure(self):
        plan = self.plan()
        self.fail_training = {"C_seed42"}
        study = self.run_study(plan)
        state = self.state(study)
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["jobs"]["C_seed42"]["status"], "failed")
        completed = {key: record["receipt"] for key, record in state["jobs"].items() if record["status"] == "completed"}
        self.assertEqual(len(completed), 3)
        self.calls.clear()
        self.fail_training.clear()
        self.run_study(plan, resume_study=study)
        state = self.state(study)
        self.assertEqual(state["status"], "completed")
        self.assertEqual([(key, action) for key, action in self.calls if action == "train"], [("C_seed42", "train")])
        for key, receipt in completed.items():
            self.assertEqual(state["jobs"][key]["receipt"], receipt)
            self.assertEqual(state["jobs"][key]["status"], "resumed")

    def test_resume_adopts_completed_child_if_coordinator_lost_its_receipt(self):
        plan = self.plan()
        study = self.run_study(plan)
        state = self.state(study)
        lost = state["jobs"]["B_seed42"].pop("receipt")
        state["jobs"]["B_seed42"]["status"] = "training"
        state["status"] = "interrupted"
        self._json(study / "study.json", state)
        self.calls.clear()
        self.run_study(plan, resume_study=study)
        recovered = self.state(study)
        self.assertEqual(recovered["status"], "completed")
        self.assertFalse(any(action == "train" for _, action in self.calls))
        self.assertEqual(recovered["jobs"]["B_seed42"]["receipt"], lost)
        self.assertEqual(recovered["jobs"]["B_seed42"]["status"], "resumed")

    def test_recovery_rejects_incomplete_attempt_and_retrains_only_that_arm(self):
        plan = self.plan()
        study = self.run_study(plan)
        state = self.state(study)
        lost = state["jobs"]["B_seed42"].pop("receipt")
        state["jobs"]["B_seed42"]["status"] = "training"
        state["status"] = "interrupted"
        (Path(lost["run"]) / "last.pt").unlink()
        self._json(study / "study.json", state)
        self.calls.clear()
        self.run_study(plan, resume_study=study)
        recovered = self.state(study)
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual([(key, action) for key, action in self.calls if action == "train"], [("B_seed42", "train")])
        self.assertNotEqual(recovered["jobs"]["B_seed42"]["receipt"]["run"], lost["run"])
        self.assertIn("Missing final checkpoint", recovered["jobs"]["B_seed42"]["recovery_rejections"][0]["reason"])

    def test_completed_artifact_change_is_rejected_without_silent_retraining(self):
        plan = self.plan()
        study = self.run_study(plan)
        state = self.state(study)
        best = Path(state["jobs"]["A_seed42"]["receipt"]["run"]) / "best.pt"
        best.write_bytes(b"altered completed checkpoint")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "completed artifact checkpoint_sha256"):
            self.run_study(plan, resume_study=study)
        self.assertFalse(any(action == "train" for _, action in self.calls))
        self.assertEqual(self.state(study)["status"], "interrupted")

    def test_mismatched_initial_student_never_trains_that_arm(self):
        self.bad_initial = {"C_seed42"}
        study = self.run_study()
        state = self.state(study)
        self.assertEqual(state["status"], "partial")
        self.assertIn("initial_state_sha256", state["jobs"]["C_seed42"]["error"])
        self.assertNotIn(("C_seed42", "train"), self.calls)

    def test_mismatched_B_D_projector_stops_all_training(self):
        self.bad_projector = {"D_seed42"}
        with self.assertRaisesRegex(ValueError, "paired initial KD projector"):
            self.run_study()
        self.assertFalse(any(action == "train" for _, action in self.calls))

    def completed_fixture(self):
        job = self.plan()["jobs"][1]
        result = self.write_run(job, self.project / "standalone_attempt")
        return job, self.check(job), Path(result["run"])

    def test_native_verifier_accepts_first_validation_maximum_without_test(self):
        job, check, run = self.completed_fixture()
        result = self.verify(run, check, job)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["last_epoch"], 17)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(result["checkpoint_sha256"], self._sha(run / "best.pt"))

    def test_native_verifier_rejects_test_artifacts_and_claims(self):
        job, check, run = self.completed_fixture()
        pristine = json.loads((run / "result.json").read_text())
        for changes in ({"test_evaluated": True}, {"evaluation_phase": "test"}, {"test_patient": {}}):
            with self.subTest(changes=changes):
                self._json(run / "result.json", {**pristine, **changes})
                with self.assertRaises(ValueError):
                    self.verify(run, check, job)
        self._json(run / "result.json", pristine)
        (run / "test").mkdir()
        with self.assertRaisesRegex(ValueError, "test evaluations"):
            self.verify(run, check, job)

    def test_native_verifier_rejects_wrong_selection_or_unmatched_artifacts(self):
        job, check, run = self.completed_fixture()
        result = json.loads((run / "result.json").read_text())
        for changes in ({"best_epoch": 3}, {"last_epoch": 2}, {"best_val_patient_roc_auc": .9}):
            with self.subTest(changes=changes):
                self._json(run / "result.json", {**result, **changes})
                with self.assertRaises(ValueError):
                    self.verify(run, check, job)
        self._json(run / "result.json", result)
        self._json(run / "validation" / "metrics.json", {"patient": {"roc_auc": .99}})
        with self.assertRaisesRegex(ValueError, "validation metrics"):
            self.verify(run, check, job)

    def test_native_verifier_rejects_missing_clean_metric_or_wrong_checkpoint(self):
        job, check, run = self.completed_fixture()
        history = json.loads((run / "history.json").read_text())
        pristine = deepcopy(history)
        history[0].pop("train_eval_loss")
        self._json(run / "history.json", history)
        with self.assertRaisesRegex(ValueError, "same-definition"):
            self.verify(run, check, job)
        self._json(run / "history.json", pristine)
        self.payloads[str(run)]["threshold"] = .7
        with self.assertRaisesRegex(ValueError, "checkpoint threshold"):
            self.verify(run, check, job)

    def test_native_verifier_rejects_nonfinite_or_noncontiguous_history(self):
        job, check, run = self.completed_fixture()
        original = json.loads((run / "history.json").read_text())
        for field, value in (("train_eval_loss", float("nan")), ("val_patient_roc_auc", float("inf")),
                             ("epoch", 12)):
            with self.subTest(field=field, value=value):
                history = deepcopy(original)
                history[3][field] = value
                self._json(run / "history.json", history)
                with self.assertRaisesRegex(ValueError, "history epochs/metrics"):
                    self.verify(run, check, job)

    def test_native_verifier_rejects_self_consistent_but_wrong_selected_validation(self):
        job, check, run = self.completed_fixture()
        result = json.loads((run / "result.json").read_text())
        result["validation_patient"]["roc_auc"] = .99
        self._json(run / "result.json", result)
        self._json(run / "validation" / "metrics.json", {"patient": result["validation_patient"]})
        with self.assertRaisesRegex(ValueError, "selected validation ROC-AUC"):
            self.verify(run, check, job)

    def test_native_verifier_rejects_incorrect_gap_and_incomplete_history(self):
        job, check, run = self.completed_fixture()
        result = json.loads((run / "result.json").read_text())
        pristine = deepcopy(result)
        result["best_gap"]["loss_gap"] = .99
        self._json(run / "result.json", result)
        with self.assertRaisesRegex(ValueError, "gaps do not match history"):
            self.verify(run, check, job)
        history = json.loads((run / "history.json").read_text())[:-1]
        self._json(run / "history.json", history)
        pristine["last_epoch"] = 16
        self._json(run / "result.json", pristine)
        with self.assertRaisesRegex(ValueError, "early-stopping budget"):
            self.verify(run, check, job)


if __name__ == "__main__":
    unittest.main()
