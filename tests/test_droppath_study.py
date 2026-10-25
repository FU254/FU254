"""Two-job DropPath-only planning, source audit and restart behavior."""

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

from foundation_sft import droppath_study as study
from tests.test_droppath_baseline import make_baseline, put


class DropPathStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.source_study = make_baseline(self.project)
        self.settings = {"baseline_study": str(self.source_study), "seeds": [42],
            "weight_decay": .01, "drop_path_rate": .1, "output_root": "runs/drop-study",
            "devices": ["cuda:0"], "max_parallel": 1}
        self.config_path = self.project / "experiment.json"
        put(self.config_path, self.settings)
        self.baseline = study.baseline_api.load_baseline(self.source_study, self.project)
        self.calls, self.baseline_checks, self.reports = [], [], []
        self.failures, self.check_defects, self.active = set(), {}, set()
        self.lock = threading.Lock()

    def plan(self, **kwargs):
        return study.make_plan(self.config_path, self.project, **kwargs)

    def baseline_verifier(self, run, check, job):
        self.baseline_checks.append(job["id"])
        return deepcopy(self.baseline["jobs"][job["id"]]["result"])

    def check(self, job):
        check = deepcopy(self.baseline["jobs"][job["id"]]["check"])
        check["config"] = deepcopy(job["config"])
        check["config"]["evaluation"] = deepcopy(self.baseline["jobs"][job["id"]]["config"]["evaluation"])
        check["model_metadata"]["regularization"] = {"drop_path_rate": .1,
            "schedule": "linear_12_blocks", "rng": "isolated_torch", "rng_seed": 42 ^ 0x44504154}
        defect = self.check_defects.get(job["id"])
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
        (directory / "best.pt").write_bytes(b"complete DropPath fixture student")
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
            "threshold": result["threshold"], "best_val_patient_roc_auc": result["best_val_patient_roc_auc"]}
        return study.random.legacy.verify_run(run, check, job, checkpoint_loader=lambda _: payload)

    def launcher(self, job, action, project, directory):
        with self.lock:
            self.assertNotIn(job["device"], self.active)
            self.active.add(job["device"])
            self.calls.append((action, job["id"]))
        try:
            state = study.random.legacy._read(directory.parents[2] / "suite.json")
            self.assertEqual(state["jobs"][job["id"]]["attempts"][-1]["status"], "running")
            if (action, job["id"]) in self.failures:
                raise RuntimeError("fixture CUDA out of memory")
            return self.check(job) if action == "check" else self.save_run(directory / "run", job)
        finally:
            with self.lock:
                self.active.remove(job["device"])

    def run_plan(self, plan=None, **kwargs):
        def reporter(path, state):
            self.reports.append((path, deepcopy(state)))
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(), launcher=self.launcher,
                verifier=self.verifier, baseline_verifier=self.baseline_verifier,
                reporter=reporter, **kwargs)

    def test_plan_has_exactly_two_seed42_jobs_and_only_droppath_changes(self):
        before = {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        plan = self.plan(devices=["cuda:0", "cuda:2"], max_parallel=2)
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertEqual(plan["seeds"], [42])
        self.assertEqual(list(plan["arms"]), ["drop_path_0p1"])
        self.assertEqual(plan["weight_decay"], .01)
        self.assertEqual(plan["drop_path_rate"], .1)
        self.assertFalse(plan["selection_rule"]["test_scores_used"])
        self.assertEqual([j["device"] for j in plan["jobs"]], ["cuda:0", "cuda:2"])
        for job in plan["jobs"]:
            actual = deepcopy(job["config"]["training"])
            self.assertEqual(actual.pop("drop_path_rate"), .1)
            expected = deepcopy(self.baseline["jobs"][job["id"]]["config"]["training"])
            expected.pop("drop_path_rate", None)
            self.assertEqual(actual, expected)
            self.assertIsNone(job["config"]["student_checkpoint"])
            self.assertEqual(job["seed"], 42)
            self.assertEqual(job["student_mode"], "full_finetune")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()})

    def test_plan_rejects_other_rates_weights_and_seeds(self):
        for field, value in (("drop_path_rate", .2), ("drop_path_rate", float("nan")),
                ("drop_path_rate", True), ("drop_path_rate", 0), ("weight_decay", .001),
                ("weight_decay", True), ("seeds", [42,43,44]), ("seeds", [43]), ("seeds", [42.0])):
            with self.subTest(field=field, value=value):
                config = deepcopy(self.settings)
                config[field] = value
                put(self.config_path, config)
                with self.assertRaises(ValueError):
                    self.plan()
        put(self.config_path, self.settings)

    def test_runtime_changes_preserve_signature_and_science_changes_do_not(self):
        first = self.plan()
        changed_runtime = self.plan(devices=["cuda:4"], max_parallel=1, cache_dir="runs/new-cache")
        self.assertEqual(study.protocol_hash(first), study.protocol_hash(changed_runtime))
        for key, value in (("drop_path_rate", .2), ("weight_decay", .001), ("seeds", [43])):
            changed = deepcopy(first)
            changed[key] = value
            self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))
        changed = deepcopy(first)
        changed["arms"][study.ARM_ID]["plan"]["jobs"][0]["config"]["training"]["head_lr"] *= 2
        self.assertNotEqual(study.protocol_hash(first), study.protocol_hash(changed))

    def test_execution_keeps_source_immutable_and_pins_the_child_suite(self):
        before = {str(p): p.read_bytes() for p in self.source_study.rglob("*") if p.is_file()}
        path = self.run_plan(self.plan(devices=["cuda:0", "cuda:1"], max_parallel=2))
        state = study.random.legacy._read(path / "study.json")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(self.baseline_checks, ["A_seed42", "B_seed42"])
        self.assertEqual(len([v for v in self.calls if v[0] == "train"]), 2)
        arm = state["arms"][study.ARM_ID]
        self.assertEqual(Path(arm["suite"]), path / "arms" / study.ARM_ID / "suite")
        self.assertEqual(arm["status"], "completed")
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.source_study.rglob("*") if p.is_file()})

    def test_check_only_can_continue_same_study_to_training(self):
        path = self.run_plan(check_only=True)
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "checked")
        self.assertTrue(all(action == "check" for action, _ in self.calls))
        self.calls.clear()
        self.run_plan(resume_study=path)
        self.assertEqual([name for action, name in self.calls if action == "train"], ["A_seed42", "B_seed42"])

    def test_resume_after_one_failure_preserves_completed_controlled_candidate(self):
        self.failures.add(("train", "B_seed42"))
        path = self.run_plan()
        self.assertEqual(study.random.legacy._read(path / "study.json")["status"], "partial")
        self.failures.clear()
        self.calls.clear()
        self.run_plan(self.plan(devices=["cuda:4"], max_parallel=1), resume_study=path)
        self.assertEqual([v for v in self.calls if v[0] == "train"], [("train", "B_seed42")])
        state = study.random.legacy._read(path / "study.json")
        self.assertEqual(state["status"], "completed")
        child = study.random.legacy._read(Path(state["arms"][study.ARM_ID]["suite"]) / "suite.json")
        self.assertEqual(child["jobs"]["A_seed42"]["status"], "resumed")
        self.assertTrue(all(j["device"] == "cuda:4" for j in child["plan"]["jobs"]))

    def test_changed_view_or_student_or_projector_initialization_blocks_training(self):
        for field, value in (("shared_view", {"changed": True}),
                             ("initial_state_sha256", "0"*64),
                             ("distillation_initial_state_sha256", "1"*64)):
            with self.subTest(field=field):
                self.check_defects["B_seed42"] = lambda check: check.update({field: value})
                self.calls.clear()
                path = self.run_plan()
                self.assertNotIn(("train", "B_seed42"), self.calls)
                state = study.random.legacy._read(path / "study.json")
                self.assertEqual(state["status"], "partial")
        self.check_defects.clear()

    def test_generated_config_changes_are_rejected_on_resume(self):
        path = self.run_plan(check_only=True)
        state = study.random.legacy._read(path / "study.json")
        config_path = Path(state["arms"][study.ARM_ID]["config_path"])
        config = study.random.legacy._read(config_path)
        config["training"]["weight_decay"] = .001
        put(config_path, config)
        with self.assertRaisesRegex(ValueError, "saved DropPath arm scientific protocol"):
            self.run_plan(resume_study=path)

    def test_source_artifact_changes_are_rejected_before_resuming(self):
        path = self.run_plan(check_only=True)
        run = Path(self.baseline["jobs"]["A_seed42"]["run"])
        with (run / "history.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            self.run_plan(self.plan(), resume_study=path)

    def test_saved_arm_labels_scientific_settings_and_selection_rule_are_frozen(self):
        path = self.run_plan(check_only=True)
        original = study.random.legacy._read(path / "study.json")
        for defect in ("arm", "rate", "weight", "seeds", "selection"):
            state = deepcopy(original)
            if defect == "arm":
                state["arms"][study.ARM_ID]["drop_path_rate"] = .2
            elif defect == "rate":
                state["drop_path_rate"] = .2
            elif defect == "weight":
                state["weight_decay"] = .001
            elif defect == "seeds":
                state["seeds"] = [43]
            else:
                state["selection_rule"]["test_scores_used"] = True
            put(path / "study.json", state)
            with self.assertRaises(ValueError):
                self.run_plan(resume_study=path)
        put(path / "study.json", original)

    def test_missing_started_child_manifest_is_not_silently_recreated(self):
        path = self.run_plan(check_only=True)
        state = study.random.legacy._read(path / "study.json")
        (Path(state["arms"][study.ARM_ID]["suite"]) / "suite.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing its pinned child manifest"):
            self.run_plan(resume_study=path)

    def test_lock_blocks_duplicates_and_releases_without_stale_cleanup(self):
        path = self.project / "lock-fixture"
        path.mkdir()
        with study._study_lock(path):
            with self.assertRaisesRegex(ValueError, "already running"):
                with study._study_lock(path):
                    self.fail("duplicate lock acquired")
        with study._study_lock(path):
            pass

    def test_cli_plan_import_and_empty_reporting_are_standard_library_only(self):
        program = self.repo / "run_droppath_experiments.py"
        completed = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.project),
            "--config", str(self.config_path), "--plan"], cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(json.loads(completed.stdout)["jobs"]), 2)
        completed = subprocess.run([sys.executable, "-S", "-c",
            "import sys; from foundation_sft import droppath_study; assert 'torch' not in sys.modules"],
            cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run([sys.executable, "-S", str(program), "--project", str(self.project),
            "--config", str(self.config_path), "--show-metrics"], cwd=self.repo, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("No saved DropPath studies found.", completed.stdout)


if __name__ == "__main__":
    unittest.main()
