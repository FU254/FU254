"""Four-job WD-only screening, source audits, locks and restart behavior."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from foundation_sft import weight_decay_study as study
from tests.test_weight_decay_baseline import make_baseline, put


class WeightDecayStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.baseline_suite = make_baseline(self.project)
        self.settings = {"baseline_suite": str(self.baseline_suite),
            "weight_decays": [.001, .01], "seeds": [42], "output_root": "runs/wd-study",
            "devices": ["cuda:0"], "max_parallel": 1}
        self.config_path = self.project / "experiment.json"
        put(self.config_path, self.settings)
        self.baseline = study.baseline_api.load_baseline(self.baseline_suite, self.project)
        self.calls, self.baseline_checks = [], []
        self.failures, self.check_defects, self.active = set(), {}, set()
        self.lock = threading.Lock()
        self.reports = []

    def plan(self, **kwargs):
        return study.make_plan(self.config_path, self.project, **kwargs)

    def baseline_verifier(self, run, check, job):
        self.baseline_checks.append(job["id"])
        return deepcopy(self.baseline["jobs"][job["id"]]["result"])

    def check(self, job):
        check = deepcopy(self.baseline["jobs"][job["id"]]["check"])
        check["config"] = deepcopy(job["config"])
        check["config"]["evaluation"] = deepcopy(self.baseline["jobs"][job["id"]]["config"]["evaluation"])
        defect = self.check_defects.get((job["config"]["training"]["weight_decay"], job["id"]))
        if defect:
            defect(check)
        return check

    def save_run(self, directory, job):
        check = self.check(job)
        source = self.baseline["jobs"][job["id"]]
        result = deepcopy(source["result"])
        result.update(run=str(directory))
        result.pop("checkpoint_sha256", None)
        result.pop("model_metadata", None)
        payloads = {"config.json": check["config"], "result.json": result,
            "model_metadata.json": check["model_metadata"], "teacher.json": check["teacher"],
            "shared_view.json": check["shared_view"],
            "test/metrics.json": {"patient": result["test_patient"],
                "bootstrap": {"patient": {"roc_auc": result["roc_auc_ci"]}}},
            "history.json": json.loads((Path(source["run"]) / "history.json").read_text())}
        for relative, value in payloads.items():
            put(directory / relative, value)
        (directory / "best.pt").write_bytes(b"new complete student fixture")
        for relative in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
            (directory / relative).write_text("label,probability\n1,0.8\n0,0.2\n")
        return result

    def verifier(self, run, check, job):
        run = Path(run)
        result = study.random.legacy._read(run / "result.json")
        saved = study.random.legacy._read(run / "config.json")
        payload = {"condition": result["condition"], "config": saved,
            "model_metadata": study.random.legacy._read(run / "model_metadata.json"),
            "initial_state_sha256": result["initial_state_sha256"], "cohort_hash": result["cohort_hash"],
            "teacher": study.random.legacy._read(run / "teacher.json"),
            "shared_view": study.random.legacy._read(run / "shared_view.json"),
            "threshold": result["threshold"],
            "best_val_patient_roc_auc": result["best_val_patient_roc_auc"]}
        return study.random.legacy.verify_run(run, check, job, checkpoint_loader=lambda _: payload)

    def launcher(self, job, action, project, directory):
        wd = job["config"]["training"]["weight_decay"]
        with self.lock:
            self.assertNotIn(job["device"], self.active)
            self.active.add(job["device"])
            self.calls.append((wd, action, job["id"]))
        try:
            parent = study.random.legacy._read(directory.parents[2] / "suite.json")
            self.assertEqual(parent["jobs"][job["id"]]["attempts"][-1]["status"], "running")
            if (wd, action, job["id"]) in self.failures:
                raise RuntimeError("fixture CUDA out of memory")
            return self.check(job) if action == "check" else self.save_run(directory / "run", job)
        finally:
            with self.lock:
                self.active.remove(job["device"])

    def run_plan(self, plan=None, **kwargs):
        def report(path, state):
            self.reports.append((path, deepcopy(state)))
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(), launcher=self.launcher,
                verifier=self.verifier, baseline_verifier=self.baseline_verifier,
                reporter=report, **kwargs)

    def test_plan_has_four_seed42_jobs_and_only_weight_decay_changes(self):
        before = {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        plan = self.plan(devices=["cuda:0", "cuda:2"], max_parallel=2)
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual(plan["seeds"], [42])
        self.assertEqual(list(plan["arms"]), ["wd_0p001", "wd_0p01"])
        self.assertEqual(plan["baseline"]["weight_decay"], .0001)
        self.assertFalse(plan["selection_rule"]["test_scores_used"])
        for job in plan["jobs"]:
            self.assertEqual(job["seed"], 42)
            actual = deepcopy(job["config"]["training"])
            actual["weight_decay"] = .0001
            self.assertEqual(actual, self.baseline["jobs"][job["id"]]["config"]["training"])
            self.assertIsNone(job["config"]["student_checkpoint"])
            self.assertEqual(job["config"]["student_initialization"], "random")
            self.assertEqual(job["student_mode"], "full_finetune")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()})

    def test_plan_rejects_undeclared_or_duplicate_baseline_weight_decay(self):
        for values in ([.001], [.001, .002], [.001, float("nan")], [True, .01]):
            settings = deepcopy(self.settings)
            settings["weight_decays"] = values
            put(self.config_path, settings)
            with self.assertRaises(ValueError):
                self.plan()
        put(self.config_path, self.settings)
        source = deepcopy(self.baseline)
        source["weight_decay"] = .001
        with self.assertRaisesRegex(ValueError, "duplicates"):
            study.make_plan(self.config_path, self.project, baseline_loader=lambda *_: source)

    def test_plan_rejects_extra_or_changed_screening_seeds(self):
        for seeds in ([42, 43, 44], [43], [], [42.0], "42"):
            settings = deepcopy(self.settings)
            settings["seeds"] = seeds
            put(self.config_path, settings)
            with self.assertRaisesRegex(ValueError, "seeds=\\[42\\]"):
                self.plan()
        put(self.config_path, self.settings)

    def test_runtime_changes_preserve_signature_and_science_changes_do_not(self):
        first = self.plan()
        moved = self.plan(devices=["cuda:3", "cuda:4"], max_parallel=2, cache_dir="runs/other-cache")
        self.assertEqual(study.protocol_hash(first), study.protocol_hash(moved))
        changed = deepcopy(first)
        changed["arms"]["wd_0p001"]["plan"]["jobs"][0]["config"]["training"]["head_lr"] *= 2
        self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))
        changed = deepcopy(first)
        changed["selection_rule"]["test_scores_used"] = True
        self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))
        changed = deepcopy(first)
        changed["seeds"] = [43]
        self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))

    def test_complete_execution_preserves_baseline_and_records_pinned_children(self):
        before = {str(p): p.read_bytes() for p in self.baseline_suite.rglob("*") if p.is_file()}
        path = self.run_plan(self.plan(devices=["cuda:0", "cuda:1"], max_parallel=2))
        state = study.random.legacy._read(path / "study.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len([c for c in self.calls if c[1] == "check"]), 4)
        self.assertEqual(len([c for c in self.calls if c[1] == "train"]), 4)
        self.assertEqual(state["seeds"], [42])
        self.assertEqual(len(self.baseline_checks), 6)
        for arm_id, arm in state["arms"].items():
            self.assertEqual(Path(arm["suite"]), path / "arms" / arm_id / "suite")
            self.assertEqual(arm["status"], "completed")
            self.assertTrue((Path(arm["config_path"])).is_file())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.baseline_suite.rglob("*") if p.is_file()})

    def test_check_only_runs_no_training_and_can_continue_same_study(self):
        path = self.run_plan(check_only=True)
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "checked")
        self.assertTrue(all(action == "check" for _, action, _ in self.calls))
        self.calls.clear()
        self.run_plan(resume_study=path)
        self.assertEqual(len([c for c in self.calls if c[1] == "train"]), 4)

    def test_resume_keeps_three_completed_jobs_and_changes_gpu_safely(self):
        self.failures.add((.001, "train", "B_seed42"))
        path = self.run_plan()
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "partial")
        self.failures.clear()
        self.calls.clear()
        self.run_plan(self.plan(devices=["cuda:4"], max_parallel=1), resume_study=path)
        self.assertEqual([c for c in self.calls if c[1] == "train"], [(.001, "train", "B_seed42")])
        state = study.random.legacy._read(path / "study.json")
        self.assertEqual(state["status"], "completed")
        for arm in state["arms"].values():
            child = study.random.legacy._read(Path(arm["suite"]) / "suite.json")
            self.assertTrue(all(job["device"] == "cuda:4" for job in child["plan"]["jobs"]))

    def test_source_protocol_check_failure_blocks_training_that_job(self):
        self.check_defects[(.001, "B_seed42")] = lambda check: check["shared_view"].update(student_size=112)
        path = self.run_plan()
        self.assertNotIn((.001, "train", "B_seed42"), self.calls)
        state = study.random.legacy._read(path / "study.json")
        child = study.random.legacy._read(Path(state["arms"]["wd_0p001"]["suite"]) / "suite.json")
        self.assertEqual(child["jobs"]["B_seed42"]["status"], "failed")

    def test_resume_rejects_modified_source_snapshot_and_generated_config(self):
        path = self.run_plan(check_only=True)
        changed = deepcopy(self.plan())
        changed["baseline"]["weight_decay"] = .002
        with self.assertRaisesRegex(ValueError, "frozen weight-decay study"):
            self.run_plan(changed, resume_study=path)
        state = study.random.legacy._read(path / "study.json")
        config_path = Path(state["arms"]["wd_0p001"]["config_path"])
        config = study.random.legacy._read(config_path)
        config["training"]["backbone_lr"] *= 2
        put(config_path, config)
        with self.assertRaisesRegex(ValueError, "saved arm scientific protocol"):
            self.run_plan(resume_study=path)

    def test_missing_started_child_manifest_is_not_silently_recreated(self):
        path = self.run_plan(check_only=True)
        state = study.random.legacy._read(path / "study.json")
        (Path(state["arms"]["wd_0p001"]["suite"]) / "suite.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing its pinned child manifest"):
            self.run_plan(resume_study=path)

    def test_saved_arm_labels_and_selection_policy_are_frozen(self):
        path = self.run_plan(check_only=True)
        original = study.random.legacy._read(path / "study.json")
        for defect in ("arm_weight", "candidates", "selection", "seeds"):
            with self.subTest(defect=defect):
                state = deepcopy(original)
                if defect == "arm_weight":
                    state["arms"]["wd_0p001"]["weight_decay"] = .05
                elif defect == "candidates":
                    state["weight_decays"] = [.002, .01]
                elif defect == "seeds":
                    state["seeds"] = [42, 43, 44]
                else:
                    state["selection_rule"]["test_scores_used"] = True
                put(path / "study.json", state)
                with self.assertRaises(ValueError):
                    self.run_plan(resume_study=path)
        put(path / "study.json", original)

    def test_study_lock_blocks_duplicates_and_releases_without_stale_cleanup(self):
        path = self.project / "lock-fixture"
        path.mkdir()
        with study._study_lock(path):
            with self.assertRaisesRegex(ValueError, "already running"):
                with study._study_lock(path):
                    self.fail("duplicate lock acquired")
        with study._study_lock(path):
            pass

    def test_cli_plan_and_import_are_standard_library_only(self):
        program = self.repo / "run_weight_decay_experiments.py"
        completed = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.project),
            "--config", str(self.config_path), "--plan"], cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        planned = json.loads(completed.stdout)
        self.assertEqual(len(planned["jobs"]), 4)
        self.assertEqual(planned["seeds"], [42])
        completed = subprocess.run([sys.executable, "-S", "-c",
            "import sys; from foundation_sft import weight_decay_study; assert 'torch' not in sys.modules"],
            cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
