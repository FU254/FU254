"""Tiny artifact fixtures exercise scheduling/provenance without Torch or a GPU."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from foundation_sft import equal_teacher_suite as suite


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class EqualTeacherSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        repo = Path(__file__).resolve().parents[1]
        self.settings = json.loads((repo / suite.DEFAULT_CONFIG).read_text())
        self.settings.update(template="template.json", output_root="runs/equal", reuse_suites=[])
        self.template = json.loads((repo / "foundation_sft/configs/distillation_rad_dino_vit_tiny.json").read_text())
        self.template["training"]["epochs"] = 2
        put(self.root / "template.json", self.template)
        self.save_settings()
        self.calls = []
        self.verifications = []
        self.failures = set()
        self.active = set()
        self.lock = threading.Lock()

    def save_settings(self):
        put(self.root / "experiment.json", self.settings)

    def plan(self, **kwargs):
        return suite.make_plan("experiment.json", self.root, **kwargs)

    def check(self, job):
        config = deepcopy(job["config"])
        config["evaluation"] = {"bootstrap_samples": 10}
        return {"check_passed": True, "condition": job["condition"], "config": config,
                "model_metadata": {"mode": "full_finetune", "initialization": "fixture", "total_parameters": 20, "trainable_parameters": 20},
                "initial_state_sha256": f"initial-seed{job['seed']}", "cohort_hash": "frozen-task",
                "teacher": {"checkpoint_sha256": "rad-sha"}, "shared_view": {"geometry": "same-view"},
                "teachers": [{"model": name, "weight": 1 / 3, "checkpoint_sha256": name + "-sha"} for name in suite.TEACHERS],
                "teacher_views": {name: {"size": 224} for name in suite.TEACHERS}}

    def saved_run(self, directory, job):
        check = self.check(job)
        condition = "supervised" if job["condition"] == "supervised" else ("distillation" if job["condition"] == "single_teacher" else "equal_teacher")
        result = {"status": "completed", "condition": condition, "seed": job["seed"], "run": str(directory),
                  "initial_state_sha256": check["initial_state_sha256"], "cohort_hash": check["cohort_hash"],
                  "teacher_unchanged": True, "threshold": .4, "best_val_patient_roc_auc": .8,
                  "test_patient": {"roc_auc": .75, "pr_auc": .5, "average_precision": .52, "sensitivity": .6, "specificity": .9},
                  "roc_auc_ci": {"lower": .65, "upper": .85}}
        payload = {"condition": condition, "config": check["config"], "model_metadata": check["model_metadata"],
                   "initial_state_sha256": check["initial_state_sha256"], "cohort_hash": check["cohort_hash"],
                   "teacher": check["teacher"], "shared_view": check["shared_view"], "threshold": .4,
                   "best_val_patient_roc_auc": .8}
        for name, value in (("result.json", result), ("config.json", check["config"]),
                            ("model_metadata.json", check["model_metadata"]), ("teacher.json", check["teacher"]),
                            ("shared_view.json", check["shared_view"]), ("best.pt", payload),
                            ("test/metrics.json", {"patient": result["test_patient"], "bootstrap": {"patient": {"roc_auc": result["roc_auc_ci"]}}})):
            put(directory / name, value)
        for name in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
            (directory / name).write_text("sample,label,probability\nfixture,1,.7\n", encoding="utf-8")
        return result

    def verifier(self, run, check, job):
        self.verifications.append((str(run), job["id"]))
        if job["condition"] == "equal_teacher":
            result = suite.legacy._read(Path(run) / "result.json")
            suite.legacy._same("seed", result["seed"], job["seed"])
            suite.legacy._same("cohort", result["cohort_hash"], check["cohort_hash"])
            result.update(checkpoint_sha256=suite.legacy._file_hash(Path(run) / "best.pt"), model_metadata=check["model_metadata"])
            return result
        with mock.patch.object(suite.legacy, "_default_checkpoint_loader", side_effect=suite.legacy._read):
            return suite.verify_run(run, check, job)

    def launcher(self, job, action, project, directory):
        with self.lock:
            self.assertNotIn(job["device"], self.active)
            self.active.add(job["device"])
        try:
            self.calls.append((action, job["id"]))
            manifest = suite.legacy._read(directory.parents[2] / "suite.json")
            record = manifest["jobs"][job["id"]]
            self.assertEqual(record["status"], "checking" if action == "check" else "training")
            self.assertEqual(record["attempts"][-1]["status"], "running")
            if (action, job["id"]) in self.failures:
                raise RuntimeError("fixture failure")
            time.sleep(.002)
            if action == "check":
                return self.check(job)
            result = self.saved_run(directory / "run", job)
            # The trainer wraps legacy control condition names in its CLI receipt.
            return dict(result, condition=job["condition"])
        finally:
            with self.lock:
                self.active.remove(job["device"])

    def run_plan(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return suite.run_suite(plan or self.plan(), launcher=self.launcher, verifier=self.verifier, **kwargs)

    def source_controls(self):
        source = self.root / "source-suite"
        records = {}
        for job in self.plan()["jobs"]:
            if job["condition"] == "equal_teacher":
                continue
            result = self.saved_run(source / job["id"], job)
            group = suite.SOURCE_GROUPS[job["condition"]]
            records[job["id"]] = {"status": "completed", "job": {
                "group": group, "condition": result["condition"], "student_mode": "full_finetune", "seed": job["seed"]}, "result": result}
        # This nonmatching ablation must not become a control, even if it is first.
        decoy = deepcopy(next(record for record in records.values() if record["job"]["group"] == "full_kd"))
        decoy["job"]["group"] = "soft_only"
        decoy["result"]["run"] = str(self.root / "must-not-use")
        put(source / "suite.json", {"jobs": {"decoy": decoy, **records}})
        self.settings["reuse_suites"] = [str(source)]
        self.save_settings()
        return source

    def test_plan_is_stdlib_only_and_has_nine_fixed_jobs(self):
        before = set(self.root.rglob("*"))
        plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 9)
        self.assertEqual(plan["devices"], ["cuda:0"])
        self.assertEqual(plan["max_parallel"], 1)
        self.assertEqual(set(self.root.rglob("*")), before)
        for job in plan["jobs"]:
            self.assertEqual(job["student_mode"], "full_finetune")
            self.assertEqual([teacher["weight"] for teacher in job["config"]["teachers"]], [1 / 3] * 3)
        repo = Path(__file__).resolve().parents[1]
        code = "import sys; from foundation_sft.equal_teacher_suite import make_plan; make_plan(sys.argv[1],sys.argv[2]); assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-S", "-c", code, str(self.root / "experiment.json"), str(self.root)], cwd=repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_rejects_unequal_weights_reordered_teacher_or_lora_student(self):
        self.settings["teachers"][0]["weight"] = .5
        self.save_settings()
        with self.assertRaisesRegex(ValueError, "1/3"):
            self.plan()
        self.settings["teachers"][0]["weight"] = 1 / 3
        self.settings["teachers"].reverse()
        self.save_settings()
        with self.assertRaisesRegex(ValueError, "order"):
            self.plan()
        self.settings["teachers"].reverse()
        self.save_settings()
        self.template["student"] = {"mode": "lora"}
        put(self.root / "template.json", self.template)
        with self.assertRaisesRegex(ValueError, "full_finetune"):
            self.plan()

    def test_only_current_explicit_ablation_controls_are_verified_and_reused(self):
        self.source_controls()
        path = self.run_plan()
        state = suite.legacy._read(path / "suite.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(sum(record["status"] == "reused" for record in state["jobs"].values()), 6)
        self.assertEqual([name for action, name in self.calls if action == "train"], [f"equal_teacher_seed{seed}" for seed in (42, 43, 44)])
        self.assertFalse(any("must-not-use" in run for run, _ in self.verifications))
        self.assertEqual(len(self.verifications), 9)

    def test_reuse_checks_real_control_artifacts_and_rejects_scientific_mismatch(self):
        job = self.plan(conditions=["single_teacher"], seeds=[42])["jobs"][0]
        run = self.root / "saved"
        self.saved_run(run, job)
        check = self.check(job)
        with mock.patch.object(suite.legacy, "_default_checkpoint_loader", side_effect=suite.legacy._read):
            receipt = suite.verify_run(run, check, job)
            self.assertEqual(receipt["condition"], "single_teacher")
            self.assertEqual(receipt["source_condition"], "distillation")
            for name, mutate in (
                ("initial student state", lambda c: c.update(initial_state_sha256="wrong")),
                ("cohort hash", lambda c: c.update(cohort_hash="wrong")),
                ("teacher checkpoint", lambda c: c["teacher"].update(checkpoint_sha256="wrong")),
                ("shared field of view", lambda c: c["shared_view"].update(geometry="wrong")),
                ("effective training", lambda c: c["config"]["training"].update(epochs=999)),
                ("evaluation", lambda c: c["config"]["evaluation"].update(bootstrap_samples=999)),
                ("student metadata", lambda c: c["model_metadata"].update(trainable_parameters=999))):
                bad = deepcopy(check)
                mutate(bad)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    suite.verify_run(run, bad, job)

    def test_equal_verifier_dispatches_to_trainer_artifact_validator(self):
        job = self.plan(conditions=["equal_teacher"], seeds=[42])["jobs"][0]
        validator = mock.Mock(return_value={"valid": True})
        module = types.ModuleType("foundation_sft.equal_teacher_distillation")
        module.verify_completed_run = validator
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            self.assertEqual(suite.verify_run("run", self.check(job), job), {"valid": True})
        validator.assert_called_once()

    def test_failures_preserve_completed_jobs_and_resume_can_change_gpu(self):
        plan = self.plan(conditions=["equal_teacher"], seeds=[42, 43])
        self.failures.add(("train", "equal_teacher_seed43"))
        path = self.run_plan(plan)
        state = suite.legacy._read(path / "suite.json")
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["jobs"]["equal_teacher_seed42"]["status"], "completed")
        self.assertEqual(state["jobs"]["equal_teacher_seed43"]["attempts"][-1]["status"], "failed")
        old_result = state["jobs"]["equal_teacher_seed42"]["result"]
        self.calls.clear()
        self.failures.clear()
        changed = self.plan(conditions=["equal_teacher"], seeds=[42, 43], devices=["cuda:4"], max_parallel=3)
        self.assertEqual(suite.protocol_hash(plan), suite.protocol_hash(changed))
        self.run_plan(changed, resume_suite=path)
        state = suite.legacy._read(path / "suite.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["jobs"]["equal_teacher_seed42"]["status"], "resumed")
        self.assertEqual(state["jobs"]["equal_teacher_seed42"]["result"], old_result)
        self.assertEqual([name for action, name in self.calls if action == "train"], ["equal_teacher_seed43"])
        self.assertEqual(state["execution_history"][-1]["devices"], ["cuda:4"])

    def test_check_only_performs_no_training_or_control_reuse(self):
        self.source_controls()
        path = self.run_plan(check_only=True)
        state = suite.legacy._read(path / "suite.json")
        self.assertEqual(state["status"], "checked")
        self.assertEqual(len(self.calls), 9)
        self.assertTrue(all(action == "check" for action, _ in self.calls))
        self.assertFalse(self.verifications)

    def test_launcher_streams_epoch_progress_and_retains_complete_log(self):
        script = self.root / "run_equal_teacher_distillation.py"
        script.write_text("import json, sys\nfrom pathlib import Path\n"
                          "print('full auxiliary log line', flush=True)\n"
                          "print(json.dumps({'epoch': 3, 'val_patient_roc_auc': 0.8}), flush=True)\n"
                          "target = Path(sys.argv[sys.argv.index('--result-json') + 1])\n"
                          "target.write_text(json.dumps({'status': 'checked'}))\n", encoding="utf-8")
        job = self.plan(seeds=[42])["jobs"][0]
        output = io.StringIO()
        with redirect_stdout(output):
            value = suite._launch(job, "check", self.root, self.root / "attempt")
        self.assertEqual(value, {"status": "checked"})
        self.assertIn('[supervised seed42] {"epoch": 3', output.getvalue())
        self.assertNotIn("full auxiliary log line", output.getvalue())
        self.assertIn("full auxiliary log line", (self.root / "attempt" / "check.log").read_text())

    def test_scheduler_serializes_same_gpu_even_with_larger_parallel_limit(self):
        path = self.run_plan(self.plan(conditions=["equal_teacher"], max_parallel=3))
        self.assertEqual(suite.legacy._read(path / "suite.json")["status"], "completed")

    def test_cross_condition_initial_state_or_teacher_mismatch_blocks_job(self):
        original = self.check
        def check(job):
            value = original(job)
            if job["condition"] == "equal_teacher":
                value["initial_state_sha256"] = "wrong"
            return value
        with mock.patch.object(self, "check", side_effect=check):
            path = self.run_plan(self.plan(seeds=[42]))
        state = suite.legacy._read(path / "suite.json")
        self.assertEqual(state["jobs"]["equal_teacher_seed42"]["status"], "failed")
        self.assertIn("paired initial", state["jobs"]["equal_teacher_seed42"]["error"])
        self.assertNotIn(("train", "equal_teacher_seed42"), self.calls)

    def test_report_keeps_pr_auc_separate_from_ap_and_uses_sample_sd(self):
        path = self.run_plan(self.plan(seeds=[42, 43]))
        state = suite.legacy._read(path / "suite.json")
        rows = suite.metric_rows(state)
        self.assertTrue(all(row["pr_auc"] == .5 and row["ap"] == .52 for row in rows))
        summaries = suite.legacy.summary_rows(rows)
        self.assertTrue(all(row["pr_auc_mean"] == .5 and row["ap_mean"] == .52 and row["roc_auc_sd"] == 0 for row in summaries))
        paired = suite.paired_rows(rows)
        self.assertEqual(len(paired), 6)
        self.assertTrue((path / "report" / "matched_seed_differences.csv").is_file())
        value = (path / "report" / "comparison.csv").read_text()
        self.assertIn("pr_auc,ap", value)


if __name__ == "__main__":
    unittest.main()
