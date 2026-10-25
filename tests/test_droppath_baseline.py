"""Pin the selected weight-decay control and audit only the DropPath change."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from foundation_sft import droppath_baseline as audit
from foundation_sft import weight_decay_study as wd_study
from tests.test_weight_decay_baseline import make_baseline as make_random_baseline, put


def make_baseline(project):
    """Build an actual completed parent WD study with two two-job child suites."""
    project = Path(project)
    original = make_random_baseline(project)
    config_file = project / "fixture_weight_decay.json"
    put(config_file, {"baseline_suite": str(original), "weight_decays": [.001, .01],
        "seeds": [42], "output_root": "runs/random_student_weight_decay",
        "devices": ["cuda:0"], "max_parallel": 1})
    plan = wd_study.make_plan(config_file, project)
    study = project / "runs/random_student_weight_decay/studies/study_fixture"
    study.mkdir(parents=True)
    state = {"format_version": wd_study.FORMAT, "protocol_sha256": wd_study.protocol_hash(plan),
        "baseline": deepcopy(plan["baseline"]), "weight_decays": list(plan["weight_decays"]),
        "seeds": [42], "selection_rule": deepcopy(plan["selection_rule"]),
        "arms": {}, "status": "completed", "plan": deepcopy(plan)}
    wd_study._materialize(study, plan, state)
    for arm in state["arms"].values():
        suite = Path(arm["suite"])
        child = json.loads((suite / "suite.json").read_text())
        child.update(status="completed", plan=deepcopy(arm["plan"]))
        for name, record in child["jobs"].items():
            source = plan["baseline"]["jobs"][name]
            run = suite / "jobs" / name / "train_fixture" / "run"
            shutil.copytree(source["run"], run)
            job = record["job"]
            saved = deepcopy(source["config"])
            saved.update(device=job["device"], output_root=str(run.parent), cache_dir=job["config"].get("cache_dir"))
            saved["training"] = deepcopy(job["config"]["training"])
            check = deepcopy(source["check"])
            check["config"] = deepcopy(saved)
            result = deepcopy(source["result"])
            result["run"] = str(run)
            receipt = json.loads((run / "result.json").read_text())
            receipt["run"] = str(run)
            put(run / "config.json", saved)
            put(run / "result.json", receipt)
            record.update(status="completed", check=check, result=result)
        put(suite / "suite.json", child)
        arm["status"] = "completed"
    put(study / "study.json", state)
    return study


class DropPathBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.study = make_baseline(self.project)

    def load(self):
        return audit.load_baseline(self.study, self.project)

    def parent(self):
        return json.loads((self.study / "study.json").read_text())

    def child(self):
        suite = Path(self.parent()["arms"]["wd_0p01"]["suite"])
        return suite, json.loads((suite / "suite.json").read_text())

    def candidate(self, name="B_seed42"):
        baseline = self.load()
        source = baseline["jobs"][name]
        job, check = deepcopy(source["job"]), deepcopy(source["check"])
        job["config"] = deepcopy(source["config"])
        for config in (job["config"], check["config"]):
            config["training"]["drop_path_rate"] = .1
            config.update(device="cuda:4", output_root="runs/new_droppath", cache_dir="runs/new_cache")
        job["device"] = "cuda:4"
        check["model_metadata"]["regularization"] = deepcopy(audit.REGULARIZATION)
        return baseline, job, check

    def test_pins_two_wd001_controls_read_only_and_records_exact_files(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.project.rglob("*") if p.is_file()}
        snapshot = self.load()
        self.assertEqual(set(snapshot["jobs"]), {"A_seed42", "B_seed42"})
        self.assertEqual(snapshot["weight_decay"], .01)
        self.assertEqual(snapshot["drop_path_rate"], 0.)
        self.assertEqual(Path(snapshot["suite"]), self.study / "arms/wd_0p01/suite")
        self.assertEqual(set(snapshot["fingerprints"]), {"study.json", "suite.json"})
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.project.rglob("*") if p.is_file()})
        audit.validate_snapshot(snapshot)

    def test_rejects_unfinished_parent_wrong_seed_wrong_arm_and_wrong_protocol(self):
        original = self.parent()
        for defect in ("unfinished", "seed", "weight", "arm_status", "suite", "signature", "selection"):
            with self.subTest(defect=defect):
                state = deepcopy(original)
                if defect == "unfinished":
                    state["status"] = "running"
                elif defect == "seed":
                    state["seeds"] = [43]
                elif defect == "weight":
                    state["arms"]["wd_0p01"]["weight_decay"] = .001
                elif defect == "arm_status":
                    state["arms"]["wd_0p01"]["status"] = "partial"
                elif defect == "suite":
                    state["arms"]["wd_0p01"]["suite"] = state["arms"]["wd_0p001"]["suite"]
                elif defect == "signature":
                    state["protocol_sha256"] = "wrong"
                else:
                    state["selection_rule"]["test_scores_used"] = True
                put(self.study / "study.json", state)
                with self.assertRaises(ValueError):
                    self.load()
        put(self.study / "study.json", original)

    def test_rejects_failed_missing_wrong_seed_and_changed_job_protocol(self):
        suite, original = self.child()
        for defect in ("failed", "missing", "seed", "protocol", "training"):
            with self.subTest(defect=defect):
                state = deepcopy(original)
                if defect == "failed":
                    state["jobs"]["B_seed42"]["status"] = "failed"
                elif defect == "missing":
                    state["jobs"].pop("A_seed42")
                elif defect == "seed":
                    state["jobs"]["B_seed42"]["job"]["seed"] = 43
                elif defect == "protocol":
                    state["protocol_sha256"] = "wrong"
                else:
                    state["jobs"]["B_seed42"]["job"]["config"]["training"]["head_lr"] *= 2
                put(suite / "suite.json", state)
                with self.assertRaises(ValueError):
                    self.load()
        put(suite / "suite.json", original)

    def test_control_requires_actual_fixed_weight_decay_and_zero_droppath(self):
        source = self.load()["jobs"]["A_seed42"]
        run = Path(source["run"])
        original = json.loads((run / "config.json").read_text())
        for key, value in (("weight_decay", .001), ("drop_path_rate", .1), ("drop_path_rate", False)):
            saved = deepcopy(original)
            saved["training"][key] = value
            put(run / "config.json", saved)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load()
        put(run / "config.json", original)

    def test_consistently_changed_control_cannot_claim_nonzero_droppath_or_regularization(self):
        suite, original = self.child()
        for defect in ("rate", "metadata"):
            state = deepcopy(original)
            record = state["jobs"]["B_seed42"]
            run = Path(record["result"]["run"])
            config_path, metadata_path = run / "config.json", run / "model_metadata.json"
            before_config, before_metadata = config_path.read_bytes(), metadata_path.read_bytes()
            config = json.loads(before_config)
            metadata = json.loads(before_metadata)
            if defect == "rate":
                config["training"]["drop_path_rate"] = .1
                record["check"]["config"]["training"]["drop_path_rate"] = .1
            else:
                metadata["regularization"] = deepcopy(audit.REGULARIZATION)
                record["check"]["model_metadata"] = deepcopy(metadata)
            put(config_path, config)
            put(metadata_path, metadata)
            put(suite / "suite.json", state)
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, "control"):
                self.load()
            config_path.write_bytes(before_config)
            metadata_path.write_bytes(before_metadata)
        put(suite / "suite.json", original)

    def test_rejects_changed_fingerprints_and_teacher_file(self):
        snapshot = self.load()
        history = Path(snapshot["jobs"]["A_seed42"]["run"]) / "history.json"
        original = history.read_bytes()
        history.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            audit.validate_snapshot(snapshot)
        history.write_bytes(original)
        teacher = Path(snapshot["jobs"]["A_seed42"]["check"]["teacher"]["checkpoint"])
        teacher.write_bytes(b"unexpected local teacher")
        with self.assertRaisesRegex(ValueError, "teacher fingerprint"):
            audit.validate_snapshot(snapshot, verify_runs=True,
                verifier=lambda run, check, job: deepcopy(snapshot["jobs"][job["id"]]["result"]))

    def test_only_explicit_droppath_change_is_allowed_for_both_groups(self):
        for name in audit.JOB_IDS:
            snapshot, job, check = self.candidate(name)
            self.assertIs(audit.audit_check(check, job, snapshot), check)
            self.assertEqual(check["config"]["training"]["weight_decay"], .01)

    def test_candidate_rejects_other_settings_and_changed_initial_states(self):
        for defect in ("wd", "lr", "loss", "view", "teacher", "student", "projector", "rate", "metadata"):
            snapshot, job, check = self.candidate()
            if defect == "wd":
                for cfg in (job["config"], check["config"]):
                    cfg["training"]["weight_decay"] = .001
            elif defect == "lr":
                for cfg in (job["config"], check["config"]):
                    cfg["training"]["head_lr"] *= 2
            elif defect == "loss":
                for cfg in (job["config"], check["config"]):
                    cfg["distillation"]["soft_weight"] = .6
            elif defect == "view":
                check["shared_view"]["student_size"] = 112
            elif defect == "teacher":
                check["teacher"]["checkpoint_sha256"] = "0" * 64
            elif defect == "student":
                check["initial_state_sha256"] = "0" * 64
            elif defect == "projector":
                check["distillation_initial_state_sha256"] = "0" * 64
            elif defect == "rate":
                for cfg in (job["config"], check["config"]):
                    cfg["training"]["drop_path_rate"] = .2
            else:
                check["model_metadata"]["regularization"]["schedule"] = "constant_per_layer"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                audit.audit_check(check, job, snapshot)

    def test_source_free_planning_import_and_snapshot_use_no_torch(self):
        repo = Path(__file__).resolve().parents[1]
        code = ("import sys; from foundation_sft.droppath_baseline import load_baseline; "
                "x=load_baseline(sys.argv[1], sys.argv[2]); "
                "assert set(x['jobs'])=={'A_seed42','B_seed42'}; "
                "assert 'torch' not in sys.modules; print('stdlib baseline audit passed')")
        result = subprocess.run([sys.executable, "-S", "-c", code, str(self.study), str(self.project)],
            cwd=repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stdlib baseline audit passed", result.stdout)

    def test_execution_verifies_both_complete_control_students(self):
        snapshot = self.load()
        calls = []
        def verifier(run, check, job):
            calls.append(job["id"])
            return deepcopy(snapshot["jobs"][job["id"]]["result"])
        audit.validate_snapshot(snapshot, verify_runs=True, verifier=verifier)
        self.assertEqual(calls, list(audit.JOB_IDS))
        with self.assertRaises(ValueError):
            audit.validate_snapshot(snapshot, verify_runs=True, verifier=lambda *_: {})


if __name__ == "__main__":
    unittest.main()
