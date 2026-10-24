"""Frozen-baseline provenance and WD-only preflight contracts, without Torch."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from foundation_sft import weight_decay_baseline as audit


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def make_baseline(project):
    """Create the six-run JSON/file fixture shared by coordinator tests."""
    project = Path(project)
    suite = project / "runs" / "baseline" / "suite_fixture"
    teacher = project / "teacher" / "best.pt"
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"frozen official backbone plus validated LoRA/head fixture")
    teacher_sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
    training = {"epochs": 100, "patience": 15, "batch_size": 8, "accumulation_steps": 8,
        "num_workers": 0, "backbone_lr": 5e-5, "head_lr": 1e-4, "projection_lr": 1e-4,
        "weight_decay": 1e-4, "amp": False, "focal_gamma": 2.0, "focal_scale": .8,
        "minimum_lr": 1e-7, "gradient_clip": 1.0,
        "horizontal_flip_probability": 0.0, "rotation_degrees": 10.0}
    jobs = {}
    for name in audit.JOB_IDS:
        group, seed_text = name.split("_seed")
        seed = int(seed_text)
        condition = "supervised" if group == "A" else "distillation"
        run = suite / "jobs" / name / "train_original" / "run"
        config = {"student_initialization": "random", "student_checkpoint": None,
            "student": {"mode": "full_finetune"}, "seed": seed, "device": "cuda:0",
            "teacher_suite": "runs/foundation_sft/suites/source", "teacher_checkpoint": str(teacher),
            "cache_dir": "runs/foundation_sft/data_cache", "output_root": str(run.parent),
            "training": deepcopy(training), "distillation": deepcopy(audit.LOSS_SETTINGS),
            "evaluation": {"bootstrap_samples": 100}}
        job = {"id": name, "group": group, "seed": seed, "condition": condition,
               "student_mode": "full_finetune", "device": "cuda:0", "initialization": "random",
               "config": deepcopy(config)}
        # The historical top-level plan did not specify effective inherited WD.
        job["config"]["training"].pop("weight_decay")
        metadata = {"model": "vit_tiny_student", "mode": "full_finetune", "feature_dim": 192,
            "initialization": {"architecture": "deit_tiny_patch16_224", "initialization_mode": "random",
                "pretraining": "none", "seed": seed, "source_num_classes": 1, "variant": "single_cls"},
            "total_parameters": 20, "trainable_parameters": 20}
        teacher_info = {"checkpoint": str(teacher), "checkpoint_sha256": teacher_sha,
            "feature_dim": 768, "selection": "validation patient ROC-AUC"}
        geometry = {"geometry": "one shared immutable view", "student_size": 224}
        kd_initial = fingerprint("projector" + str(seed)) if group == "B" else None
        check = {"check_passed": True, "condition": condition, "config": deepcopy(config),
            "model_metadata": metadata, "initial_state_sha256": fingerprint("student" + str(seed)),
            "cohort_hash": fingerprint("frozen cohort"), "teacher": teacher_info,
            "teacher_unchanged": True, "shared_view": geometry,
            "distillation_initial_state_sha256": kd_initial}
        metrics = {"roc_auc": .75, "pr_auc": .4, "average_precision": .41,
                   "sensitivity": .6, "specificity": .8}
        interval = {"lower": .65, "upper": .85}
        receipt = {"status": "completed", "experiment": "random_student_kd_v1", "condition": condition,
            "run": str(run), "seed": seed, "student_initialization": "random", "student_mode": "full_finetune",
            "initial_state_sha256": check["initial_state_sha256"], "cohort_hash": check["cohort_hash"],
            "best_val_patient_roc_auc": .8, "distillation_initial_state_sha256": kd_initial,
            "threshold": .5, "test_patient": metrics, "roc_auc_ci": interval, "teacher_unchanged": True,
            "student_trainable_parameters": 20, "distillation_trainable_parameters": 0 if group == "A" else 12}
        histories = [{"epoch": epoch, "train_eval_loss": .1 / epoch, "val_loss": .11 / epoch,
            "train_eval_patient_roc_auc": .75 + .05 * epoch,
            "val_patient_roc_auc": .6 + .1 * epoch} for epoch in (1, 2)]
        histories[-1]["val_patient_roc_auc"] = .8
        files = {"config.json": config, "result.json": receipt, "history.json": histories,
            "model_metadata.json": metadata, "teacher.json": teacher_info, "shared_view.json": geometry,
            "test/metrics.json": {"patient": metrics, "bootstrap": {"patient": {"roc_auc": interval}}}}
        for relative, payload in files.items():
            put(run / relative, payload)
        (run / "best.pt").write_bytes(("complete student " + name).encode())
        for relative in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
            (run / relative).write_text("label,probability\n1,0.8\n0,0.2\n")
        result = deepcopy(receipt)
        result.update(checkpoint_sha256=hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
                      model_metadata=deepcopy(metadata))
        jobs[name] = {"job": job, "status": "resumed" if group == "A" else "completed",
                      "check": check, "result": result, "attempts": []}
    put(suite / "suite.json", {"format_version": "random_student_suite_v1", "status": "completed", "jobs": jobs})
    return suite


class WeightDecayBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.suite = make_baseline(self.project)

    def source(self):
        return json.loads((self.suite / "suite.json").read_text())

    def save(self, source):
        put(self.suite / "suite.json", source)

    def load(self):
        return audit.load_baseline(self.suite, self.project)

    def candidate(self, name="B_seed42", wd=.001):
        baseline = self.load()
        original = baseline["jobs"][name]
        job, check = deepcopy(original["job"]), deepcopy(original["check"])
        job["config"] = deepcopy(original["config"])
        for config in (job["config"], check["config"]):
            config["training"]["weight_decay"] = wd
            config.update(device="cuda:4", output_root="runs/new-wd", cache_dir="new-cache")
        job["device"] = "cuda:4"
        return baseline, job, check

    def test_reads_actual_inherited_wd_and_snapshots_all_six_without_writes(self):
        before = {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        baseline = self.load()
        self.assertEqual(baseline["weight_decay"], .0001)
        self.assertEqual(set(baseline["jobs"]), set(audit.JOB_IDS))
        self.assertEqual(baseline, self.load())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.project.rglob("*") if p.is_file()})
        self.assertNotIn("weight_decay", baseline["jobs"]["A_seed42"]["job"]["config"]["training"])
        self.assertEqual(baseline["jobs"]["A_seed42"]["config"]["training"]["weight_decay"], .0001)

    def test_rejects_incomplete_missing_extra_and_wrong_task_identity(self):
        original = self.source()
        for kind in ("failed", "missing", "extra", "wrong_seed", "wrong_mode", "wrong_format"):
            with self.subTest(kind=kind):
                source = deepcopy(original)
                if kind == "failed":
                    source["jobs"]["A_seed44"]["status"] = "failed"
                elif kind == "missing":
                    source["jobs"].pop("B_seed42")
                elif kind == "extra":
                    source["jobs"]["C_seed42"] = source["jobs"]["A_seed42"]
                elif kind == "wrong_seed":
                    source["jobs"]["A_seed42"]["job"]["seed"] = 43
                elif kind == "wrong_mode":
                    source["jobs"]["A_seed42"]["job"]["student_mode"] = "lora"
                else:
                    source["format_version"] = "distillation_abcd_v1"
                self.save(source)
                with self.assertRaises(ValueError):
                    self.load()
        self.save(original)

    def test_rejects_nonrandom_imagenet_config_and_nonfinite_wd(self):
        run = Path(self.source()["jobs"]["A_seed42"]["result"]["run"])
        original = json.loads((run / "config.json").read_text())
        for key, value in (("student_initialization", "imagenet"), ("student_checkpoint", "old.pth"),
                           ("weight_decay", float("nan")), ("weight_decay", -1), ("weight_decay", True)):
            with self.subTest(key=key, value=value):
                saved = deepcopy(original)
                if key == "weight_decay":
                    saved["training"][key] = value
                else:
                    saved[key] = value
                put(run / "config.json", saved)
                with self.assertRaises(ValueError):
                    self.load()
        put(run / "config.json", original)

    def test_rejects_different_effective_training_between_baseline_seeds(self):
        source = self.source()
        record = source["jobs"]["A_seed43"]
        run = Path(record["result"]["run"])
        config = json.loads((run / "config.json").read_text())
        config["training"]["weight_decay"] = .001
        record["check"]["config"]["training"]["weight_decay"] = .001
        put(run / "config.json", config)
        self.save(source)
        with self.assertRaisesRegex(ValueError, "shared baseline effective protocol"):
            self.load()

    def test_rejects_result_metrics_projector_and_checkpoint_mismatch(self):
        original = self.source()
        for field in ("initial_state_sha256", "checkpoint_sha256", "distillation_initial_state_sha256", "test_patient"):
            with self.subTest(field=field):
                source = deepcopy(original)
                source["jobs"]["B_seed42"]["result"][field] = fingerprint("changed")
                self.save(source)
                with self.assertRaises(ValueError):
                    self.load()
        self.save(original)

    def test_validate_snapshot_rejects_changed_source_file_and_source_manifest(self):
        snapshot = self.load()
        run = Path(snapshot["jobs"]["A_seed42"]["run"])
        history_file = run / "history.json"
        original = history_file.read_bytes()
        history_file.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "immutable baseline snapshot"):
            audit.validate_snapshot(snapshot)
        history_file.write_bytes(original)
        source = self.source()
        source["updated_at"] = "changed by another coordinator"
        self.save(source)
        with self.assertRaisesRegex(ValueError, "immutable baseline snapshot"):
            audit.validate_snapshot(snapshot)

    def test_full_verification_calls_every_run_and_checks_actual_teacher_hash(self):
        snapshot = self.load()
        calls = []
        def verifier(run, check, job):
            calls.append(job["id"])
            return deepcopy(snapshot["jobs"][job["id"]]["result"])
        audit.validate_snapshot(snapshot, verify_runs=True, verifier=verifier)
        self.assertEqual(calls, list(audit.JOB_IDS))
        teacher = Path(snapshot["jobs"]["A_seed42"]["check"]["teacher"]["checkpoint"])
        teacher.write_bytes(b"another teacher")
        with self.assertRaisesRegex(ValueError, "actual teacher checkpoint fingerprint"):
            audit.validate_snapshot(snapshot, verify_runs=True, verifier=verifier)

    def test_only_explicit_wd_change_with_runtime_locations_is_accepted(self):
        for name in audit.JOB_IDS:
            baseline, job, check = self.candidate(name)
            self.assertEqual(audit.audit_check({"check": check}, job, baseline), check)

    def test_rejects_teacher_data_view_metadata_and_other_training_changes(self):
        for field in ("teacher", "cohort_hash", "shared_view", "model_metadata", "training", "evaluation"):
            with self.subTest(field=field):
                baseline, job, check = self.candidate()
                if field == "teacher":
                    check[field]["checkpoint_sha256"] = fingerprint("another teacher")
                elif field == "cohort_hash":
                    check[field] = fingerprint("other cohort")
                elif field == "shared_view":
                    check[field]["student_size"] = 384
                elif field == "model_metadata":
                    check[field]["total_parameters"] += 1
                elif field == "evaluation":
                    check["config"][field]["bootstrap_samples"] += 1
                else:
                    for config in (job["config"], check["config"]):
                        config["training"]["backbone_lr"] = .0002
                with self.assertRaises(ValueError):
                    audit.audit_check(check, job, baseline)

    def test_rejects_student_and_projector_initialization_changes(self):
        for field in ("initial_state_sha256", "distillation_initial_state_sha256"):
            with self.subTest(field=field):
                baseline, job, check = self.candidate()
                check[field] = fingerprint("new initial parameters")
                with self.assertRaises(ValueError):
                    audit.audit_check(check, job, baseline)

    def test_rejects_implicit_or_wrong_candidate_weight_decay(self):
        baseline, job, check = self.candidate()
        job["config"]["training"].pop("weight_decay")
        with self.assertRaises(ValueError):
            audit.audit_check(check, job, baseline)
        baseline, job, check = self.candidate()
        check["config"]["training"]["weight_decay"] = .01
        with self.assertRaises(ValueError):
            audit.audit_check(check, job, baseline)

    def test_rejects_candidate_group_identity_change(self):
        baseline, job, check = self.candidate()
        job["group"] = "A"
        with self.assertRaisesRegex(ValueError, "candidate job group"):
            audit.audit_check(check, job, baseline)

    def test_import_and_loading_work_with_site_packages_disabled(self):
        code = "from foundation_sft.weight_decay_baseline import load_baseline; import sys; x=load_baseline(sys.argv[1],sys.argv[2]); assert len(x['jobs'])==6"
        result = subprocess.run([sys.executable, "-S", "-c", code, str(self.suite), str(self.project)],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
