"""One new soft-only B run, pinned old controls, audited resume and reporting."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from foundation_sft import random_student_soft_only as study
from foundation_sft import random_student_soft_report as report
from tests.test_droppath_baseline import make_baseline, put


class RandomStudentSoftStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.source_study = make_baseline(self.project)
        self.config_path = self.project / "experiment.json"
        self.settings = {"baseline_study": str(self.source_study), "seeds": [42],
            "weight_decay": .01, "drop_path_rate": 0., "output_root": "runs/soft-only",
            "distillation_profile": "soft_only", "devices": ["cuda:0"], "max_parallel": 1}
        put(self.config_path, self.settings)
        self.baseline = study.baseline_api.load_baseline(self.source_study, self.project)
        self.calls, self.baseline_checks, self.reports = [], [], []
        self.failure = None
        self.check_defect = None

    def plan(self, **kwargs):
        return study.make_plan(self.config_path, self.project, **kwargs)

    def baseline_verifier(self, run, check, job):
        self.baseline_checks.append(job["id"])
        return deepcopy(self.baseline["jobs"][job["id"]]["result"])

    def check(self, job):
        value = deepcopy(self.baseline["jobs"]["B_seed42"]["check"])
        value["config"] = deepcopy(job["config"])
        value["config"]["evaluation"] = deepcopy(self.baseline["jobs"]["B_seed42"]["config"]["evaluation"])
        value.update(distillation_profile="soft_only", soft_kd_gradient_verified=True,
            distillation_trainable_parameters=0,
            distillation_initial_state_sha256=study.EMPTY_STATE_SHA256,
            losses={"hard": .03, "soft": .02, "global": 0., "local": 0., "kd": .01, "total": .04})
        if self.check_defect:
            self.check_defect(value)
        return value

    def save_run(self, directory, job):
        check = self.check(job)
        source = self.baseline["jobs"]["B_seed42"]
        result = deepcopy(source["result"])
        result.update(run=str(directory), distillation_profile="soft_only",
            distillation_initial_state_sha256=study.EMPTY_STATE_SHA256,
            distillation_trainable_parameters=0)
        result.pop("checkpoint_sha256", None)
        result.pop("model_metadata", None)
        payloads = {"config.json": check["config"], "result.json": result,
            "model_metadata.json": check["model_metadata"], "teacher.json": check["teacher"],
            "shared_view.json": check["shared_view"],
            "test/metrics.json": {"patient": result["test_patient"],
                "bootstrap": {"patient": {"roc_auc": result["roc_auc_ci"]}}},
            "history.json": json.loads((Path(source["run"]) / "history.json").read_text())}
        for name, value in payloads.items():
            put(directory / name, value)
        (directory / "best.pt").write_bytes(b"complete soft-only fixture student")
        for name in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
            (directory / name).write_text("label,probability\n1,0.8\n0,0.2\n")
        return result

    def verifier(self, run, check, job):
        run = Path(run)
        result = study.random.legacy._read(run / "result.json")
        payload = {"condition": result["condition"], "config": study.random.legacy._read(run / "config.json"),
            "model_metadata": study.random.legacy._read(run / "model_metadata.json"),
            "initial_state_sha256": result["initial_state_sha256"], "cohort_hash": result["cohort_hash"],
            "teacher": study.random.legacy._read(run / "teacher.json"),
            "shared_view": study.random.legacy._read(run / "shared_view.json"),
            "threshold": result["threshold"], "best_val_patient_roc_auc": result["best_val_patient_roc_auc"]}
        return study.random.legacy.verify_run(run, check, job, checkpoint_loader=lambda _: payload)

    def launcher(self, job, action, project, directory):
        self.calls.append((action, job["id"]))
        self.assertEqual(job["id"], "B_seed42")
        child = study.random.legacy._read(directory.parents[2] / "suite.json")
        self.assertEqual(child["jobs"][job["id"]]["attempts"][-1]["status"], "running")
        if self.failure == action:
            raise RuntimeError("fixture CUDA out of memory")
        return self.check(job) if action == "check" else self.save_run(directory / "run", job)

    def run_plan(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(), launcher=self.launcher,
                verifier=self.verifier, baseline_verifier=self.baseline_verifier,
                reporter=lambda path, state: self.reports.append((path, deepcopy(state))), **kwargs)

    def test_plan_exactly_one_B_seed42_only_two_loss_coefficients_change(self):
        before = {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        plan = self.plan(devices=["cuda:4", "cuda:2"], max_parallel=2)
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["seeds"], [42])
        self.assertFalse(plan["selection_rule"]["test_scores_used"])
        job = plan["jobs"][0]
        self.assertEqual((job["id"], job["group"], job["device"]), ("B_seed42", "B", "cuda:4"))
        source = self.baseline["jobs"]["B_seed42"]["config"]
        self.assertEqual(job["config"]["training"], source["training"])
        self.assertEqual(job["config"]["distillation"], {**source["distillation"], "global_weight": 0., "local_weight": 0.})
        self.assertEqual(job["config"]["distillation_profile"], "soft_only")
        self.assertIsNone(job["config"]["student_checkpoint"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()})

    def test_rejects_changed_declared_science(self):
        for key, value in (("seeds", [42, 43]), ("seeds", [42.0]), ("weight_decay", .001),
                ("weight_decay", True), ("drop_path_rate", .1), ("drop_path_rate", False),
                ("distillation_profile", "three_term")):
            with self.subTest(key=key):
                settings = {**self.settings, key: value}
                put(self.config_path, settings)
                with self.assertRaises(ValueError):
                    self.plan()
        put(self.config_path, self.settings)

    def test_runtime_changes_preserve_signature_and_science_changes_do_not(self):
        first = self.plan()
        self.assertEqual(study.protocol_hash(first), study.protocol_hash(self.plan(devices=["cuda:4"], cache_dir="new-cache")))
        changed = deepcopy(first)
        changed["candidate"]["plan"]["jobs"][0]["config"]["training"]["head_lr"] *= 2
        self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))

    def test_only_one_training_immutable_source_and_completed_resume(self):
        before = {str(p): p.read_bytes() for p in self.source_study.rglob("*") if p.is_file()}
        path = self.run_plan()
        state = study.random.legacy._read(path / "study.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.calls, [("check", "B_seed42"), ("train", "B_seed42")])
        self.assertEqual(self.baseline_checks, ["A_seed42", "B_seed42"])
        self.calls.clear()
        self.run_plan(self.plan(devices=["cuda:4"]), resume_study=path)
        self.assertEqual(self.calls, [("check", "B_seed42")])
        child = study.random.legacy._read(path / "suite/suite.json")
        self.assertEqual(child["jobs"]["B_seed42"]["status"], "resumed")
        self.assertEqual(child["jobs"]["B_seed42"]["job"]["device"], "cuda:4")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.source_study.rglob("*") if p.is_file()})

    def test_check_only_then_train_same_parent_and_failed_retry_keeps_attempts(self):
        path = self.run_plan(check_only=True)
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "checked")
        self.assertEqual(self.calls, [("check", "B_seed42")])
        self.calls.clear()
        self.failure = "train"
        self.run_plan(resume_study=path)
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "failed")
        self.failure = None
        self.calls.clear()
        self.run_plan(resume_study=path)
        self.assertEqual(self.calls, [("check", "B_seed42"), ("train", "B_seed42")])
        child = study.random.legacy._read(path / "suite/suite.json")
        self.assertTrue(any(attempt["status"] == "failed" for attempt in child["jobs"]["B_seed42"]["attempts"]))

    def test_changed_teacher_view_training_initialization_blocks_new_training(self):
        defects = [lambda c: c.update(shared_view={"changed": True}),
            lambda c: c.update(initial_state_sha256="0"*64),
            lambda c: c["teacher"].update(checkpoint_sha256="1"*64),
            lambda c: c["config"]["training"].update(head_lr=.9),
            lambda c: c.update(distillation_initial_state_sha256="1"*64),
            lambda c: c["losses"].update({"global": .1}),
            lambda c: c.update(soft_kd_gradient_verified=False)]
        for defect in defects:
            self.check_defect = defect
            self.calls.clear()
            path = self.run_plan()
            self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "failed")
            self.assertEqual(self.calls, [("check", "B_seed42")])
        self.check_defect = None

    def test_source_or_generated_candidate_tamper_prevents_resume(self):
        path = self.run_plan(check_only=True)
        config_path = path / "config.json"
        original = config_path.read_bytes()
        candidate = json.loads(original)
        candidate["training"]["backbone_lr"] *= 2
        put(config_path, candidate)
        with self.assertRaisesRegex(ValueError, "saved candidate scientific protocol"):
            self.run_plan(resume_study=path)
        config_path.write_bytes(original)
        source_history = Path(self.baseline["jobs"]["A_seed42"]["run"]) / "history.json"
        source_history.write_bytes(source_history.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.run_plan(resume_study=path)

    def test_source_changed_during_check_prevents_new_training(self):
        original = self.launcher
        def mutate_after_check(job, action, project, directory):
            result = original(job, action, project, directory)
            if action == "check":
                path = Path(self.baseline["jobs"]["A_seed42"]["run"]) / "history.json"
                path.write_bytes(path.read_bytes() + b" ")
            return result
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            study.run_study(self.plan(), launcher=mutate_after_check,
                verifier=self.verifier, baseline_verifier=self.baseline_verifier,
                reporter=lambda *_: None)
        self.assertEqual(self.calls, [("check", "B_seed42")])

    def test_missing_started_manifest_and_relocated_candidate_are_rejected(self):
        path = self.run_plan(check_only=True)
        manifest = path / "suite/suite.json"
        raw = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, "missing its pinned child manifest"):
            self.run_plan(resume_study=path)
        manifest.write_bytes(raw)
        parent = study.random.legacy._read(path / "study.json")
        parent["candidate"]["suite"] = str(self.project / "wrong-suite")
        put(path / "study.json", parent)
        with self.assertRaisesRegex(ValueError, "pinned candidate suite"):
            self.run_plan(resume_study=path)

    def test_report_labels_reused_controls_and_same_definition_gaps(self):
        path = self.run_plan()
        state = study.random.legacy._read(path / "study.json")
        result = report.write_report(path, state)
        self.assertEqual([row["arm"] for row in result["rows"]], ["supervised", "full_kd", "soft_only"])
        self.assertEqual(len(result["overfitting"]), 3)
        self.assertEqual(result["selection"]["selected_arm"], "full_kd")
        self.assertEqual(len(result["paired"]), 3)
        self.assertFalse(result["selection"]["test_metrics_used"])
        source = result["overfitting"][1]
        self.assertAlmostEqual(source["best_auc_gap"], source["best_train_auc"] - source["best_val_auc"])
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_metrics(path, state)
        self.assertIn("PR-AUC and AP are distinct", output.getvalue())
        self.assertIn("Single seed42", output.getvalue())

    def test_selection_uses_validation_not_test_and_ties_keep_full(self):
        rows = [{"arm": arm, "seed": 42, "status": "completed", "val_auc": .75,
                 "roc_auc": .95 if arm == "soft_only" else .60} for arm in report.LABELS]
        self.assertEqual(report.validation_selection(rows)["selected_arm"], "full_kd")
        rows[-1]["val_auc"] = .76
        self.assertEqual(report.validation_selection(rows)["selected_arm"], "soft_only")
        rows[-1]["status"] = "failed"
        self.assertEqual(report.validation_selection(rows)["status"], "incomplete")
        rows[-1]["status"] = "completed"
        for value in (-.1, 1.1, float("nan"), True):
            rows[-1]["val_auc"] = value
            self.assertEqual(report.validation_selection(rows)["status"], "incomplete")

    def test_plot_reads_comparable_fields_and_preserves_existing_figures(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib required for curve rendering")
        path = self.run_plan()
        state = study.random.legacy._read(path / "study.json")
        captured = []
        original = plt.Axes.plot
        def capture(axis, x, y, *args, **kwargs):
            captured.append((list(x), list(y)))
            return original(axis, x, y, *args, **kwargs)
        with patch.object(plt.Axes, "plot", capture):
            image = report.plot_curves(path, state)
        self.assertTrue(image.is_file())
        self.assertEqual(len(captured), 12)
        rows = report.metric_rows(path, state)
        history, _ = report.history_report._load_history(rows[2])
        self.assertEqual(captured[8][1], [row["train_eval_loss"] for row in history])
        self.assertEqual(captured[10][1], [row["train_eval_patient_roc_auc"] for row in history])
        first = image.read_bytes()
        second = report.plot_curves(path, state)
        self.assertNotEqual(image, second)
        self.assertEqual(image.read_bytes(), first)

    def test_cli_plan_and_empty_reporting_are_standard_library_only(self):
        program = self.repo / "run_random_student_soft_only.py"
        result = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.project),
            "--config", str(self.config_path), "--plan"], cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["jobs"]), 1)
        result = subprocess.run([sys.executable, "-S", "-c",
            "import sys; from foundation_sft import random_student_soft_only; assert 'torch' not in sys.modules"],
            cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.project),
            "--config", str(self.config_path), "--show-metrics"], cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No saved soft-only studies found.", result.stdout)


if __name__ == "__main__":
    unittest.main()
