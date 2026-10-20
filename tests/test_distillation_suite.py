"""ABCD coordination uses tiny on-disk fixtures and no GPU or patient data."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from foundation_sft import distillation_suite as suite


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class DistillationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        training = {"backbone_lr": 5e-5, "head_lr": 1e-4, "projection_lr": 1e-4,
                    "epochs": 2, "weight_decay": .01, "amp": False}
        self.template = {"student_checkpoint": "initial.pt", "teacher_checkpoint": "teacher.pt",
                         "training": training, "distillation": {"temperature": 2, "soft_weight": .5,
                         "global_weight": .1, "local_weight": 1, "pool_grid": [7, 7]}}
        put(self.root / "full.json", self.template)
        lora = deepcopy(self.template)
        lora.update(student={"mode": "lora", "rank": 8, "alpha": 16})
        lora["training"]["lora_lr"] = 5e-5
        put(self.root / "lora.json", lora)
        self.settings = {"output_root": "runs/abcd", "seeds": [42, 43, 44],
                         "max_parallel": 4, "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                         "reuse_suites": [], "groups": {
                             group: {"student_mode": mode, "condition": condition,
                                     "config": "lora.json" if mode == "lora" else "full.json"}
                             for group, (mode, condition) in suite.GROUPS.items()}}
        put(self.root / "abcd.json", self.settings)
        self.calls = []
        self.lock = threading.Lock()
        self.active = set()

    def plan(self, **kwargs):
        return suite.make_plan("abcd.json", self.root, **kwargs)

    def check(self, job):
        config = deepcopy(job["config"])
        config["evaluation"] = {"bootstrap_samples": 10}
        return {"check_passed": True, "condition": job["condition"], "config": config,
                "model_metadata": {"mode": job["student_mode"], "initialization": "fixture",
                                   "total_parameters": 20, "trainable_parameters": 20 if job["student_mode"] == "full_finetune" else 4},
                "initial_state_sha256": f"init-{job['student_mode']}-{job['seed']}",
                "cohort_hash": "frozen", "teacher": {"checkpoint_sha256": "teacher-sha"},
                "shared_view": {"geometry": "same-crop"}}

    def saved_run(self, directory, job):
        check = self.check(job)
        directory.mkdir(parents=True)
        result = {"condition": job["condition"], "seed": job["seed"], "status": "completed",
                  "run": str(directory), "initial_state_sha256": check["initial_state_sha256"],
                  "cohort_hash": check["cohort_hash"], "teacher_unchanged": True,
                  "threshold": .4, "best_val_patient_roc_auc": .8,
                  "test_patient": {"roc_auc": .75, "pr_auc": .5, "average_precision": .52,
                                   "sensitivity": .6, "specificity": .9},
                  "roc_auc_ci": {"lower": .65, "upper": .85}}
        payload = {"condition": job["condition"], "config": check["config"],
                   "model_metadata": check["model_metadata"],
                   "initial_state_sha256": check["initial_state_sha256"],
                   "cohort_hash": check["cohort_hash"], "teacher": check["teacher"],
                   "shared_view": check["shared_view"], "threshold": .4,
                   "best_val_patient_roc_auc": .8, "model": {"weight": [1, 2, 3]}}
        for name, value in (("result.json", result), ("config.json", check["config"]),
                            ("model_metadata.json", check["model_metadata"]),
                            ("teacher.json", check["teacher"]), ("shared_view.json", check["shared_view"]),
                            ("best.pt", payload), ("test/metrics.json", {"patient": result["test_patient"],
                              "bootstrap": {"patient": {"roc_auc": result["roc_auc_ci"]}}})):
            put(directory / name, value)
        for name in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
            (directory / name).write_text("sample_id,label,probability\nsample,1,.8\n")
        return result

    def verifier(self, run, check, job):
        return suite.verify_run(run, check, job, checkpoint_loader=suite._read)

    def launcher(self, job, action, project, directory):
        with self.lock:
            self.assertNotIn(job["device"], self.active, "same GPU was oversubscribed")
            self.active.add(job["device"])
            self.calls.append((action, job["id"]))
        try:
            time.sleep(.005)
            directory.mkdir(parents=True)
            if action == "check":
                return self.check(job)
            result = self.saved_run(directory / "run", job)
            return {"conditions": {job["condition"]: result}, "status": "completed"}
        finally:
            with self.lock:
                self.active.remove(job["device"])

    def execute(self, plan, **kwargs):
        with redirect_stdout(io.StringIO()):
            path = suite.run_suite(plan, launcher=self.launcher, verifier=self.verifier, **kwargs)
        return path, suite._read(path / "suite.json")

    def test_plan_subset_stable_defaults_and_no_write(self):
        before = set(self.root.rglob("*"))
        plan = self.plan(groups=["B", "D"], seeds=[43])
        self.assertEqual([job["id"] for job in plan["jobs"]], ["B_seed43", "D_seed43"])
        self.assertEqual([job["device"] for job in plan["jobs"]], ["cuda:1", "cuda:3"])
        self.assertEqual(before, set(self.root.rglob("*")))
        with redirect_stdout(io.StringIO()), mock.patch.dict("sys.modules", {"torch": None}):
            self.assertEqual(suite.main(["--project", str(self.root), "--config", "abcd.json", "--plan"]), 0)
            self.assertEqual(suite.main(["--project", str(self.root), "--config", "abcd.json", "--show-metrics"]), 0)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_checkpoint_tuple_metadata_has_same_json_meaning(self):
        suite._same("preprocessing", {"mean": (.1, .2), "grid": (7, 7)},
                    {"mean": [.1, .2], "grid": [7.0, 7.0]})
        with self.assertRaisesRegex(ValueError, "preprocessing"):
            suite._same("preprocessing", {"mean": (.1, .2)}, {"mean": [.1, .3]})

    def test_all_checks_before_training_gpu_exclusion_and_summary(self):
        plan = self.plan(devices=["cpu", "cpu"], seeds=[42, 43])
        path, state = self.execute(plan)
        self.assertEqual(state["status"], "completed")
        first_train = next(i for i, call in enumerate(self.calls) if call[0] == "train")
        self.assertEqual(first_train, 8)
        summaries = suite.summary_rows(suite.metric_rows(state))
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(row["n_completed"] == 2 for row in summaries))
        self.assertEqual(summaries[0]["roc_auc_sd"], 0)
        self.assertEqual(suite.metric_rows(state)[0]["ap"], .52)
        before = {file: file.read_bytes() for file in path.rglob("*") if file.is_file()}
        with redirect_stdout(io.StringIO()), mock.patch.dict("sys.modules", {"torch": None}):
            suite.print_metrics(suite.saved_batches(path.parent.parent))
        self.assertTrue(all(file.read_bytes() == data for file, data in before.items()))

    def test_check_only_never_trains(self):
        _, state = self.execute(self.plan(groups=["C", "D"], seeds=[42]), check_only=True)
        self.assertEqual(state["status"], "checked")
        self.assertEqual([action for action, _ in self.calls], ["check", "check"])
        self.assertTrue(all(row["n_completed"] == 0 for row in suite.summary_rows(suite.metric_rows(state))))

    def test_legacy_full_reuse_and_reject_provenance_differences(self):
        job = self.plan(groups=["B"], seeds=[42])["jobs"][0]
        run = self.root / "old" / "distillation"
        self.saved_run(run, job)  # no config.student: the historical full format
        valid = self.verifier(run, self.check(job), job)
        self.assertEqual(valid["status"], "completed")
        for key in ("cohort_hash", "initial_state_sha256", "model_metadata", "teacher", "shared_view"):
            with self.subTest(key=key):
                check = self.check(job)
                check[key] = "changed" if key in {"cohort_hash", "initial_state_sha256"} else {"changed": True}
                with self.assertRaises((ValueError, AttributeError)):
                    self.verifier(run, check, job)
        for section in ("training", "evaluation", "distillation"):
            check = self.check(job)
            check["config"][section]["changed"] = True
            with self.assertRaisesRegex(ValueError, "differs"):
                self.verifier(run, check, job)
        original = suite._read(run / "result.json")
        changed = deepcopy(original)
        changed["roc_auc_ci"]["lower"] = .1
        put(run / "result.json", changed)
        with self.assertRaisesRegex(ValueError, "bootstrap interval"):
            self.verifier(run, self.check(job), job)
        put(run / "result.json", original)
        raw = suite._read(run / "best.pt")
        raw["model_metadata"] = {"changed": True}
        put(run / "best.pt", raw)
        with self.assertRaisesRegex(ValueError, "checkpoint student metadata"):
            self.verifier(run, self.check(job), job)

    def test_explicit_legacy_suite_reused_low_lr_run_rejected(self):
        plan = self.plan(groups=["A", "B"], seeds=[42])
        old = self.root / "old_suite"
        for job in plan["jobs"]:
            self.saved_run(old / job["condition"], job)
        plan["reuse_suites"] = [str(old)]
        _, state = self.execute(plan)
        self.assertTrue(all(record["status"] == "reused" for record in state["jobs"].values()))
        self.assertTrue(all(action == "check" for action, _ in self.calls))
        # Deliberately modified low LR must train afresh, regardless of its test metric.
        config = suite._read(old / "supervised" / "config.json")
        config["training"]["backbone_lr"] = 1e-5
        put(old / "supervised" / "config.json", config)
        _, state = self.execute(plan)
        self.assertEqual(state["jobs"]["A_seed42"]["status"], "completed")
        self.assertEqual(state["jobs"]["B_seed42"]["status"], "reused")

    def test_damaged_old_manifest_does_not_stop_new_jobs(self):
        plan = self.plan(groups=["A"], seeds=[42])
        old = self.root / "old_broken_suite"
        old.mkdir()
        plan["reuse_suites"] = [str(old)]
        for text in ('{"conditions":', '{"conditions":[]}',
                     '{"conditions":{"supervised":{"run":123}}}',
                     '{"conditions":{"supervised":{"run":[]}}}'):
            with self.subTest(manifest=text):
                (old / "suite.json").write_text(text)
                _, state = self.execute(plan)
                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["jobs"]["A_seed42"]["status"], "completed")
                self.assertTrue(state["jobs"]["A_seed42"]["reuse_rejections"])

    def test_resume_rechecks_complete_jobs_retries_invalid_and_protocol_guard(self):
        plan = self.plan(groups=["A", "C"], seeds=[42])
        path, original = self.execute(plan)
        self.calls.clear()
        _, state = self.execute(plan, resume_suite=path)
        self.assertTrue(all(record["status"] == "resumed" for record in state["jobs"].values()))
        self.assertEqual(len(self.calls), 2)
        bad = Path(state["jobs"]["C_seed42"]["result"]["run"]) / "best.pt"
        bad.write_text("truncated")
        self.calls.clear()
        _, state = self.execute(plan, resume_suite=path)
        self.assertEqual(state["jobs"]["A_seed42"]["status"], "resumed")
        self.assertEqual(state["jobs"]["C_seed42"]["status"], "completed")
        self.assertEqual(bad.read_text(), "truncated", "resume overwrote the prior run")
        altered = deepcopy(plan)
        altered["jobs"][0]["config"]["distillation"]["temperature"] = 3
        with self.assertRaisesRegex(ValueError, "differs"):
            self.execute(altered, resume_suite=path)

    def test_check_only_resume_preserves_verified_completed_jobs(self):
        plan = self.plan(groups=["A", "C"], seeds=[42])
        path, initial = self.execute(plan)
        original_runs = {key: record["result"]["run"] for key, record in initial["jobs"].items()}
        self.calls.clear()
        _, checked = self.execute(plan, resume_suite=path, check_only=True)
        self.assertEqual(checked["status"], "checked")
        self.assertTrue(all(value["status"] == "resumed" for value in checked["jobs"].values()))
        self.assertTrue(all(action == "check" for action, _ in self.calls))
        self.calls.clear()
        _, resumed = self.execute(plan, resume_suite=path)
        self.assertTrue(all(action == "check" for action, _ in self.calls))
        self.assertEqual(original_runs, {key: record["result"]["run"] for key, record in resumed["jobs"].items()})

    def test_failed_check_and_train_keep_partial_report(self):
        normal = self.launcher
        def failing(job, action, project, directory):
            if (job["group"], action) in {("A", "check"), ("B", "train")}:
                raise RuntimeError("synthetic failure")
            return normal(job, action, project, directory)
        with redirect_stdout(io.StringIO()):
            path = suite.run_suite(self.plan(seeds=[42]), launcher=failing, verifier=self.verifier)
        state = suite._read(path / "suite.json")
        self.assertEqual(state["status"], "partial")
        self.assertEqual([state["jobs"][f"{group}_seed42"]["status"] for group in "ABCD"],
                         ["failed", "failed", "completed", "completed"])
        self.assertTrue((path / "report" / "comparison.csv").is_file())
        self.assertIn("synthetic failure", (path / "report" / "comparison.csv").read_text())

    def test_cross_group_cohort_change_blocks_comparison(self):
        normal = self.launcher
        def changed(job, action, project, directory):
            result = normal(job, action, project, directory)
            if action == "check" and job["group"] == "D":
                result["cohort_hash"] = "different-patients"
            return result
        with redirect_stdout(io.StringIO()):
            path = suite.run_suite(self.plan(groups=["A", "D"], seeds=[42]), launcher=changed, verifier=self.verifier)
        state = suite._read(path / "suite.json")
        self.assertEqual(state["jobs"]["D_seed42"]["status"], "failed")
        self.assertEqual(state["jobs"]["A_seed42"]["status"], "completed")

    def test_cross_group_teacher_initialization_and_effective_training_must_match(self):
        normal = self.launcher
        for defect in ("teacher", "initialization", "training", "evaluation"):
            with self.subTest(defect=defect):
                def changed(job, action, project, directory):
                    result = normal(job, action, project, directory)
                    if action == "check" and job["group"] == "D":
                        if defect == "teacher":
                            result["teacher"]["checkpoint_sha256"] = "changed"
                        elif defect == "initialization":
                            result["model_metadata"]["initialization"] = "different-ImageNet-source"
                        else:
                            result["config"][defect]["changed"] = True
                    return result
                with redirect_stdout(io.StringIO()):
                    path = suite.run_suite(self.plan(groups=["A", "D"], seeds=[42]), launcher=changed, verifier=self.verifier)
                state = suite._read(path / "suite.json")
                self.assertEqual(state["jobs"]["D_seed42"]["status"], "failed")

    def test_running_resume_retries_and_does_not_treat_stale_result_as_completed(self):
        plan = self.plan(groups=["A"], seeds=[42])
        path, state = self.execute(plan)
        state["jobs"]["A_seed42"]["status"] = "running"
        put(path / "suite.json", state)
        self.calls.clear()
        _, updated = self.execute(plan, resume_suite=path)
        self.assertEqual(updated["jobs"]["A_seed42"]["status"], "completed")
        self.assertEqual(self.calls, [("check", "A_seed42"), ("train", "A_seed42")])


if __name__ == "__main__":
    unittest.main()
