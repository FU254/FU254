"""Random-only planning, provenance, scheduling, and recovery without Torch."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
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

from foundation_sft import random_student_suite as suite


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RandomStudentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.settings = json.loads((self.repo / suite.DEFAULT_CONFIG).read_text())
        self.settings.update(output_root="runs/random", groups=["A", "B"], seeds=[42, 43, 44])
        self.settings["training"]["epochs"] = 2
        self.save_settings()
        self.calls, self.verify_calls = [], []
        self.failures, self.active = set(), set()
        self.mutate_check = None
        self.lock = threading.Lock()

    def save_settings(self):
        put(self.root / "experiment.json", self.settings)

    def plan(self, **kwargs):
        return suite.make_plan("experiment.json", self.root, **kwargs)

    def check(self, job):
        config = deepcopy(job["config"])
        config["evaluation"] = {"bootstrap_samples": 10}
        check = {"check_passed": True, "condition": job["condition"], "config": config,
            "model_metadata": {"mode": "full_finetune", "feature_dim": 6,
                "initialization": {"architecture": "deit_tiny_patch16_224", "pretraining": "none",
                    "initialization_mode": "random", "seed": job["seed"], "source_num_classes": 1,
                    "variant": "single_cls"}, "total_parameters": 20, "trainable_parameters": 20},
            "initial_state_sha256": f"random-seed{job['seed']}", "cohort_hash": "frozen-task",
            "teacher": {"checkpoint_sha256": "rad-sha", "selection": "validation"},
            "teacher_unchanged": True, "shared_view": {"geometry": "same-view"}}
        if self.mutate_check:
            self.mutate_check(check, job)
        return check

    def save_run(self, directory, job):
        check = self.check(job)
        result = {"status": "completed", "condition": job["condition"], "seed": job["seed"],
            "run": str(directory), "student_mode": "full_finetune", "initialization": "random",
            "initial_state_sha256": check["initial_state_sha256"], "cohort_hash": check["cohort_hash"],
            "teacher_unchanged": True, "threshold": .4, "best_val_patient_roc_auc": .8,
            "test_patient": {"roc_auc": .75 + (job["seed"] - 42) * .01,
                "pr_auc": .5, "average_precision": .52, "sensitivity": .6, "specificity": .9},
            "roc_auc_ci": {"lower": .65, "upper": .85}}
        for name, value in (("result.json", result), ("config.json", check["config"]),
                            ("metadata.json", check["model_metadata"]), ("check.json", check)):
            put(directory / name, value)
        (directory / "best.pt").write_bytes(b"fixture complete student state")
        return result

    def verifier(self, run, check, job):
        self.verify_calls.append((str(run), job["id"]))
        root = Path(run)
        result = suite.legacy._read(root / "result.json")
        saved = suite.legacy._read(root / "config.json")
        suite.legacy._validate_protocol(saved, check["config"], job["condition"])
        suite.legacy._same("random initialization", saved.get("student_initialization"), "random")
        suite.legacy._same("random checkpoint", saved.get("student_checkpoint"), None)
        suite.legacy._same("metadata", suite.legacy._read(root / "metadata.json"), check["model_metadata"])
        suite.legacy._same("initialized state", result["initial_state_sha256"], check["initial_state_sha256"])
        suite.legacy._same("cohort", result["cohort_hash"], check["cohort_hash"])
        result.update(model_metadata=check["model_metadata"], checkpoint_sha256=suite.legacy._file_hash(root / "best.pt"))
        return result

    def launcher(self, job, action, project, directory):
        with self.lock:
            self.assertNotIn(job["device"], self.active)
            self.active.add(job["device"])
            self.calls.append((action, job["id"]))
        try:
            manifest = suite.legacy._read(directory.parents[2] / "suite.json")
            record = manifest["jobs"][job["id"]]
            self.assertEqual(record["status"], "checking" if action == "check" else "training")
            self.assertEqual(record["attempts"][-1]["status"], "running")
            if (action, job["id"]) in self.failures:
                raise RuntimeError("fixture failure")
            time.sleep(.002)
            return self.check(job) if action == "check" else self.save_run(directory / "run", job)
        finally:
            with self.lock:
                self.active.remove(job["device"])

    def run_plan(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return suite.run_suite(plan or self.plan(), launcher=self.launcher, verifier=self.verifier, **kwargs)

    def test_default_plan_has_six_full_random_jobs_without_a_weight_file(self):
        plan = self.plan(devices=["cuda:0", "cuda:1", "cuda:2"], max_parallel=3)
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual(plan["groups"], ["A", "B"])
        self.assertNotIn("reuse_suites", plan)
        for index, job in enumerate(plan["jobs"]):
            self.assertEqual(job["initialization"], "random")
            self.assertEqual(job["student_mode"], "full_finetune")
            self.assertEqual(job["config"]["student"], {"mode": "full_finetune"})
            self.assertEqual(job["config"]["student_initialization"], "random")
            self.assertIsNone(job["config"]["student_checkpoint"])
            self.assertEqual(job["device"], f"cuda:{index % 3}")
        self.assertFalse((self.root / "runs").exists())

    def test_plan_rejects_imagenet_sources_historical_reuse_and_unrequested_lora(self):
        for defect, value in (("student_initialization", "imagenet"),
                              ("student_checkpoint", "weights/old.pth"),
                              ("reuse_suites", ["old/imagenet/results"]),
                              ("groups", ["C", "D"]), ("seeds", [42, 42]),
                              ("devices", ["cuda:0", "cuda:0"]), ("max_parallel", 0)):
            with self.subTest(defect=defect):
                settings = deepcopy(self.settings)
                settings[defect] = value
                put(self.root / "experiment.json", settings)
                with self.assertRaises(ValueError):
                    self.plan()
        self.save_settings()

    def test_cli_plan_and_reporting_are_standard_library_only(self):
        program = self.repo / "run_random_student_experiments.py"
        completed = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.root),
            "--config", "experiment.json", "--plan"], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(json.loads(completed.stdout)["jobs"]), 6)
        self.assertFalse((self.root / "runs").exists())
        completed = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.root),
            "--config", "experiment.json", "--show-metrics"], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No saved random-student", completed.stdout)
        self.assertFalse((self.root / "runs").exists())

    def test_check_only_does_not_train_or_import_historical_results(self):
        root = self.run_plan(check_only=True)
        state = suite.legacy._read(root / "suite.json")
        self.assertEqual(state["status"], "checked")
        self.assertTrue(all(action == "check" for action, name in self.calls))
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(self.verify_calls, [])
        self.assertTrue(all(record["status"] == "checked" for record in state["jobs"].values()))

    def test_metadata_rejects_pretrained_or_wrong_seed_even_when_config_says_random(self):
        job = self.plan()["jobs"][0]
        for defect in ("pretraining", "initialization_mode", "seed", "checkpoint_sha256"):
            with self.subTest(defect=defect):
                check = self.check(job)
                check["model_metadata"]["initialization"][defect] = "invalid"
                with self.assertRaises(ValueError):
                    suite._checked(check, job)

    def test_paired_initial_state_or_frozen_task_mismatch_blocks_the_changed_job(self):
        for field in ("initial_state_sha256", "cohort_hash", "teacher", "shared_view"):
            def altered(check, job):
                if job["group"] == "B":
                    check[field] = {"changed": True} if field in ("teacher", "shared_view") else "changed"
            self.mutate_check = altered
            root = self.run_plan(self.plan(seeds=[42]))
            state = suite.legacy._read(root / "suite.json")
            self.assertEqual(state["jobs"]["B_seed42"]["status"], "failed", field)
            self.assertEqual(state["status"], "partial")
            self.assertNotIn(("train", "B_seed42"), self.calls)
            self.calls.clear()
        self.mutate_check = None

    def test_partial_failure_resume_keeps_completed_job_and_can_move_gpus(self):
        self.failures.add(("train", "B_seed42"))
        plan = self.plan(seeds=[42])
        root = self.run_plan(plan)
        self.assertEqual(suite.legacy._read(root / "suite.json")["status"], "partial")
        self.failures.clear()
        self.calls.clear()
        changed = self.plan(seeds=[42], devices=["cuda:3", "cuda:4"], max_parallel=2)
        self.assertEqual(suite.protocol_hash(changed), suite.protocol_hash(plan))
        self.run_plan(changed, resume_suite=root)
        state = suite.legacy._read(root / "suite.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["jobs"]["A_seed42"]["status"], "resumed")
        self.assertEqual(state["jobs"]["B_seed42"]["status"], "completed")
        self.assertNotIn(("train", "A_seed42"), self.calls)
        self.assertIn(("train", "B_seed42"), self.calls)
        self.assertEqual(state["execution_history"][-1]["devices"], ["cuda:3", "cuda:4"])

    def test_resume_rejects_different_science_and_imagenet_batch_formats(self):
        root = self.run_plan(self.plan(seeds=[42]), check_only=True)
        changed = self.plan(seeds=[43])
        with self.assertRaisesRegex(ValueError, "scientific protocol"):
            self.run_plan(changed, resume_suite=root)
        state = suite.legacy._read(root / "suite.json")
        state["format_version"] = "distillation_abcd_v1"
        put(root / "suite.json", state)
        with self.assertRaisesRegex(ValueError, "scientific protocol"):
            self.run_plan(self.plan(seeds=[42]), resume_suite=root)
        self.assertEqual(suite.saved_batches(root.parents[1]), [])

    def test_complete_random_checkpoint_is_reverified_on_resume(self):
        root = self.run_plan(self.plan(seeds=[42]))
        state = suite.legacy._read(root / "suite.json")
        old = Path(state["jobs"]["A_seed42"]["result"]["run"])
        saved = suite.legacy._read(old / "config.json")
        saved["student_initialization"] = "imagenet"
        put(old / "config.json", saved)
        self.calls.clear()
        self.run_plan(self.plan(seeds=[42]), resume_suite=root)
        state = suite.legacy._read(root / "suite.json")
        self.assertIn(("train", "A_seed42"), self.calls)
        self.assertNotIn(("train", "B_seed42"), self.calls)
        self.assertIn("resume_rejections", state["jobs"]["A_seed42"])

    def test_report_labels_random_keeps_ap_distinct_and_uses_sample_sd(self):
        root = self.run_plan()
        with (root / "report" / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["initialization"] == "random" for row in rows))
        self.assertEqual(float(rows[0]["pr_auc"]), .5)
        self.assertEqual(float(rows[0]["ap"]), .52)
        with (root / "report" / "summary.csv").open() as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(summary[0]["initialization"], "random")
        self.assertAlmostEqual(float(summary[0]["roc_auc_sd"]), .01)
        with (root / "report" / "matched_seed_differences.csv").open() as handle:
            paired = list(csv.DictReader(handle))
        self.assertEqual(len(paired), 3)
        self.assertTrue(all(row["comparison"] == "full_kd_gain" for row in paired))

    def test_same_gpu_is_serial_even_when_global_parallel_limit_is_larger(self):
        self.run_plan(self.plan(devices=["cuda:0"], max_parallel=4))
        self.assertEqual(len(self.calls), 12)
        self.assertEqual(self.active, set())

    def test_verifier_dispatches_to_random_trainer_not_legacy_imagenet_loader(self):
        module = types.ModuleType("foundation_sft.random_student_distillation")
        module.verify_completed_run = mock.Mock(return_value={"status": "completed"})
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            result = suite.verify_run("run", {"check": True}, {"id": "A_seed42"})
        self.assertEqual(result, {"status": "completed"})
        module.verify_completed_run.assert_called_once_with("run", {"check": True}, {"id": "A_seed42"})

    def test_real_child_epoch_stream_and_full_log_are_retained(self):
        script = self.root / "run_random_student_distillation.py"
        script.write_text(
            "import argparse,json,sys\n"
            "p=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--config');"
            "p.add_argument('--condition');p.add_argument('--result-json');p.add_argument('--check-only',action='store_true');a=p.parse_args()\n"
            "print('full diagnostic line',flush=True)\n"
            "print(json.dumps({'epoch':1,'train_loss':.5}),flush=True)\n"
            "open(a.result_json,'w').write(json.dumps({'status':'completed','condition':a.condition,'run':'fixture'}))\n",
            encoding="utf-8")
        output = io.StringIO()
        directory = self.root / "attempt"
        with redirect_stdout(output):
            result = suite._launch(self.plan(seeds=[42])["jobs"][0], "train", self.root, directory)
        self.assertEqual(result["status"], "completed")
        self.assertIn('"epoch": 1', output.getvalue())
        self.assertNotIn("full diagnostic line", output.getvalue())
        self.assertIn("full diagnostic line", (directory / "train.log").read_text())
        saved = suite.legacy._read(directory / "config.json")
        self.assertIsNone(saved["student_checkpoint"])
        self.assertEqual(saved["student_initialization"], "random")


if __name__ == "__main__":
    unittest.main()
