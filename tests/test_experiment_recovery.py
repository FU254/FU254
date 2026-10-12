"""Exercise failed-job recovery through a real, lightweight subprocess.

The fake conda executable supplies training artifacts and receipts, without
importing torch or requiring a GPU. Fixtures retain the original failed suite.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


FAKE_CONDA = r'''
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

settings = json.loads(Path(__file__).with_suffix(".json").read_text())
root = Path(settings["root"])
args = sys.argv[1:]
assert args[0] == "run", args
assert "--no-capture-output" in args, args
assert "-n" in args, args
assert "LD_LIBRARY_PATH" not in os.environ
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED"):
    assert os.environ.get(name) == "1", (name, os.environ.get(name))
with (root / "fake_calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")

def argument(name):
    return args[args.index(name) + 1]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")

if "foundation_sft.report" in args:
    assert "--suite" not in args
    start = args.index("--runs") + 1
    end = next((i for i in range(start, len(args)) if args[i].startswith("--")), len(args))
    runs = args[start:end]
    assert runs and all(Path(run).is_absolute() for run in runs), runs
    output = Path(argument("--output"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_metrics.csv").write_text("model,mode,seed\neva_x,lora,42\n")
    write(output / "protocols.json", [])
    print("Synthetic comparison report completed", flush=True)
    raise SystemExit(0)

assert "foundation_sft.train" in args, args
assert argument("--action") == "train"
assert "--resume-full-precision" in args
source = Path(argument("--resume"))
assert source.is_absolute() and source.name == "last.pt" and source.is_file()
config = json.loads(Path(argument("--config")).read_text())
assert config["training"]["amp"] is True
if settings.get("behavior") == "fail":
    print("Synthetic training failure after startup", flush=True)
    raise SystemExit(23)

run = root / "model_runs" / ("recovered_" + uuid.uuid4().hex)
(run / "test").mkdir(parents=True)
saved = copy.deepcopy(config)
saved["training"]["amp"] = False
fingerprints = json.loads((source.parent / "split_metadata.json").read_text())
write(run / "config.json", saved)
write(run / "split_metadata.json", fingerprints)
write(run / "environment.json", {"cohort_hash": fingerprints["_cohort_hash"]})
write(run / "model_metadata.json", {
    "mode": saved["mode"], "preprocessing": {"image_size": 224},
    "pretrained": {"model": saved["model"], "checkpoint_sha256": "fixture-weights"},
})
write(run / "history.json", [{"epoch": 2, "train_loss": 0.5}])
write(run / "selection.json", {"threshold": 0.5, "source": "validation patient predictions"})
write(run / "test" / "metrics.json", {"patient": {"roc_auc": 0.75}})
resume = {"checkpoint": str(source), "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "precision_changed": True, "source_amp": True, "target_amp": False}
if settings.get("behavior") == "wrong_resume_hash":
    resume["checkpoint_sha256"] = "0" * 64
elif settings.get("behavior") == "wrong_source_amp":
    resume["source_amp"] = False
elif settings.get("behavior") == "wrong_target_amp":
    resume["target_amp"] = True
write(run / "resume.json", resume)
for filename in ("best.pt", "last.pt"):
    (run / filename).write_bytes(b"synthetic recovered checkpoint")
(run / "test" / "predictions.csv").write_text(
    "sample_id,patient_id,label,probability\na,pa,0,0.2\nb,pb,1,0.8\n"
)
(run / "test" / "patient_predictions.csv").write_text(
    "patient_id,label,probability,n_images\npa,0,0.2,1\npb,1,0.8,1\n"
)
receipt = {"action": "train", "model": saved["model"], "mode": saved["mode"],
           "seed": saved["seed"], "run_dir": str(run.resolve())}
if settings.get("behavior") == "wrong_receipt":
    receipt["seed"] += 1
write(Path(argument("--result-json")), receipt)
print("Run directory: " + str(run.resolve()), flush=True)
print("Synthetic training progress: epoch 2 completed", flush=True)
'''


class ExperimentRecoveryTests(unittest.TestCase):
    def setUp(self):
        from foundation_sft.recover_experiment import main
        self.main = main
        self.temporary = tempfile.TemporaryDirectory(prefix="experiment-recovery-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        self.project.mkdir()
        self.suite = self.root / "suite"
        (self.suite / "logs").mkdir(parents=True)
        self.failed = self.root / "failed_run"
        self.failed.mkdir()
        self.config = {
            "model": "eva_x", "mode": "lora", "seed": 42, "device": "cuda:0",
            "output_root": str(self.root / "model_runs"),
            "split_manifest": str(self.root / "frozen.csv"),
            "data": {"dicom_dir": str(self.root / "dicoms")},
            "training": {"amp": True, "epochs": 4, "head_lr": 0.001,
                         "lora_rank": 8, "batch_size": 2},
        }
        self.fingerprints = {"_cohort_hash": "frozen-cohort", "_source_hash": "labels-source",
                             "_manifest_hash": "frozen-assignment"}
        self.snapshot = self.suite / "configs" / "eva_x_lora_seed42.json"
        write_json(self.snapshot, self.config)
        write_json(self.failed / "config.json", self.config)
        write_json(self.failed / "split_metadata.json", self.fingerprints)
        write_json(self.failed / "environment.json", {"cohort_hash": "frozen-cohort"})
        (self.failed / "last.pt").write_bytes(b"original failed checkpoint")
        self.train_log = self.suite / "logs" / "eva_x_lora_seed42_train.log"
        self.train_log.write_text("Training started\nRun directory: " + str(self.failed) + "\nAMP failed\n")
        self.job = {
            "id": "eva_x_lora_seed42", "model": "eva_x", "mode": "lora", "seed": 42,
            "env": "evax_lora", "device": "cuda:0", "status": "failed",
            "train_status": "failed", "config_file": str(self.snapshot),
            "train_log": str(self.train_log),
        }
        self.state = {"format_version": 1, "status": "partial", "project_dir": str(self.project),
                      "cohort_hash": "frozen-cohort", "jobs": [self.job]}
        self.save_suite()
        self.fake_conda = self.root / "fake conda"
        self.fake_conda.write_text("#!" + sys.executable + "\n" + textwrap.dedent(FAKE_CONDA))
        self.fake_conda.chmod(0o700)
        self.configure_fake()

    def configure_fake(self, behavior="success"):
        write_json(self.fake_conda.with_suffix(".json"), {"root": str(self.root), "behavior": behavior})

    def save_suite(self):
        write_json(self.suite / "suite.json", self.state)

    def calls(self):
        path = self.root / "fake_calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def recovery_dirs(self):
        return sorted(path for path in self.suite.glob("recovery_*") if path.is_dir())

    def invoke(self, *extra):
        stdout = io.StringIO()
        args = ["--suite", str(self.suite), "--job", self.job["id"],
                "--conda-executable", str(self.fake_conda), *extra]
        with redirect_stdout(stdout), patch.dict(os.environ, {"LD_LIBRARY_PATH": "inherited-bad-path"}):
            result = self.main(args)
        return result, stdout.getvalue()

    def recover(self):
        previous = set(self.recovery_dirs())
        result, stdout = self.invoke("--resume-full-precision")
        self.assertEqual(result, 0)
        created = set(self.recovery_dirs()) - previous
        self.assertEqual(len(created), 1)
        recovery = created.pop()
        manifest = json.loads((recovery / "recovery_manifest.json").read_text())
        return recovery, manifest, stdout

    def add_other_job(self, *, status="completed", train_status="passed", device="cuda:1"):
        other_run = self.root / "other_run"
        other_run.mkdir(exist_ok=True)
        other_config = deepcopy(self.config)
        other_config.update(model="rad_dino", device=device)
        write_json(other_run / "config.json", other_config)
        write_json(other_run / "split_metadata.json", self.fingerprints)
        write_json(other_run / "environment.json", {"cohort_hash": "frozen-cohort"})
        for name in ("best.pt", "last.pt"):
            (other_run / name).write_bytes(b"other completed checkpoint")
        write_json(other_run / "model_metadata.json", {"mode": "lora"})
        write_json(other_run / "history.json", [{"epoch": 4}])
        write_json(other_run / "selection.json", {
            "threshold": 0.5, "source": "validation patient predictions", "best_val_patient_roc_auc": 0.8125,
        })
        write_json(other_run / "test" / "metrics.json", {
            "patient": {"roc_auc": 0.75, "pr_auc": 0.8, "average_precision": 0.7,
                        "sensitivity": 0.8, "specificity": 0.9},
            "bootstrap": {"patient": {"roc_auc": {"lower": 0.6, "upper": 0.9}}},
        })
        (other_run / "test" / "predictions.csv").write_text(
            "sample_id,patient_id,label,probability\na,pa,0,0.2\nb,pb,1,0.8\n"
        )
        (other_run / "test" / "patient_predictions.csv").write_text(
            "patient_id,label,probability,n_images\npa,0,0.2,1\npb,1,0.8,1\n"
        )
        self.state["jobs"].append({
            "id": "rad_dino_lora_seed42", "model": "rad_dino", "mode": "lora", "seed": 42,
            "env": "rad_dino", "device": device, "status": status, "train_status": train_status,
            "run_dir": str(other_run),
        })
        self.save_suite()
        return other_run

    def assert_not_launched(self):
        self.assertEqual(self.calls(), [])

    def test_real_subprocess_preserves_failed_job_and_changes_only_precision(self):
        preserved = {path: path.read_bytes() for path in (
            self.suite / "suite.json", self.snapshot, self.failed / "config.json",
            self.failed / "last.pt", self.train_log,
        )}
        recovery, manifest, stdout = self.recover()
        self.assertEqual(recovery.parent, self.suite)
        self.assertRegex(recovery.name, r"^recovery_.+")
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["suite"], str(self.suite))
        self.assertEqual(manifest["job_id"], self.job["id"])
        self.assertEqual(manifest["source_checkpoint"], str(self.failed / "last.pt"))
        self.assertEqual(manifest["receipt_file"], str(recovery / "train_receipt.json"))
        run = Path(manifest["recovered_run_dir"])
        self.assertTrue(run.is_absolute())
        self.assertNotEqual(run, self.failed)
        expected = deepcopy(self.config)
        expected["training"]["amp"] = False
        self.assertEqual(json.loads((run / "config.json").read_text()), expected)
        resume = json.loads((run / "resume.json").read_text())
        self.assertEqual(resume["checkpoint"], str(self.failed / "last.pt"))
        self.assertEqual(resume["checkpoint_sha256"], hashlib.sha256((self.failed / "last.pt").read_bytes()).hexdigest())
        self.assertIs(resume["precision_changed"], True)
        self.assertIs(resume["source_amp"], True)
        self.assertIs(resume["target_amp"], False)
        self.assertIn("Synthetic training progress", stdout)
        self.assertIn("Synthetic training progress", (recovery / "train.log").read_text())
        for path, original in preserved.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.suite / f".recovery_{self.job['id']}.lock").exists())

    def test_repeated_recoveries_use_distinct_output_directories(self):
        first, first_manifest, _ = self.recover()
        second, second_manifest, _ = self.recover()
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_manifest["recovered_run_dir"], second_manifest["recovered_run_dir"])
        self.assertEqual(len(self.calls()), 2)

    def test_running_suite_permits_recovery_on_an_idle_different_gpu(self):
        self.state["status"] = "training"
        self.add_other_job(status="pending", train_status="running", device="cuda:1")
        self.recover()
        self.assertEqual(len(self.calls()), 1)

    def test_running_job_on_same_gpu_blocks_recovery_before_launch(self):
        self.state["status"] = "training"
        self.add_other_job(status="pending", train_status="running", device="cuda:0")
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke("--resume-full-precision")
        self.assert_not_launched()

    def test_exclusive_job_lock_is_preserved_when_already_held(self):
        lock = self.suite / f".recovery_{self.job['id']}.lock"
        lock.write_text("another recovery owns this lock")
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            self.invoke("--resume-full-precision")
        self.assertEqual(lock.read_text(), "another recovery owns this lock")
        self.assert_not_launched()

    def test_subprocess_failure_keeps_log_and_releases_lock(self):
        self.configure_fake("fail")
        with self.assertRaises(RuntimeError):
            self.invoke("--resume-full-precision")
        recovery = self.recovery_dirs()[0]
        self.assertIn("Synthetic training failure", (recovery / "train.log").read_text())
        self.assertFalse((self.suite / f".recovery_{self.job['id']}.lock").exists())
        self.assertEqual(json.loads((self.suite / "suite.json").read_text())["jobs"][0]["status"], "failed")

    def test_mismatched_receipt_is_rejected_and_outputs_are_retained(self):
        self.configure_fake("wrong_receipt")
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke("--resume-full-precision")
        recovery = self.recovery_dirs()[0]
        receipt = json.loads((recovery / "train_receipt.json").read_text())
        self.assertEqual(receipt["seed"], 43)
        self.assertTrue(Path(receipt["run_dir"]).is_dir())
        self.assertFalse((self.suite / f".recovery_{self.job['id']}.lock").exists())

    def test_resume_provenance_rejects_changed_hash_or_precision_direction(self):
        for behavior in ("wrong_resume_hash", "wrong_source_amp", "wrong_target_amp"):
            with self.subTest(behavior=behavior):
                self.configure_fake(behavior)
                with self.assertRaisesRegex(ValueError, "checkpoint SHA256|precision change"):
                    self.invoke("--resume-full-precision")
                self.assertFalse((self.suite / f".recovery_{self.job['id']}.lock").exists())
        self.assertEqual(len(self.calls()), 3)

    def test_ambiguous_run_directory_log_is_rejected_before_launch(self):
        with self.train_log.open("a") as handle:
            handle.write("Run directory: " + str(self.root / "another_failed_run") + "\n")
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke("--resume-full-precision")
        self.assert_not_launched()

    def test_report_only_uses_explicit_recovery_and_preserves_original_report(self):
        other = self.add_other_job()
        recovery, manifest, _ = self.recover()
        original_report = self.suite / "report" / "run_metrics.csv"
        original_report.parent.mkdir()
        original_report.write_text("original partial suite report\n")
        result, stdout = self.invoke("--report-only", "--recovery-dir", str(recovery))
        self.assertEqual(result, 0)
        self.assertIn("Synthetic comparison report completed", stdout)
        self.assertTrue((recovery / "report" / "run_metrics.csv").is_file())
        self.assertEqual(original_report.read_text(), "original partial suite report\n")
        report_command = self.calls()[-1]
        self.assertIn("foundation_sft.report", report_command)
        start = report_command.index("--runs") + 1
        end = next((i for i in range(start, len(report_command)) if report_command[i].startswith("--")), len(report_command))
        self.assertEqual(set(report_command[start:end]), {str(other), manifest["recovered_run_dir"]})
        self.assertNotIn(str(self.failed), report_command)
        # A second report request must preserve the existing report.
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            self.invoke("--report-only", "--recovery-dir", str(recovery))

    def test_report_only_rejects_a_still_active_suite(self):
        recovery, _, _ = self.recover()
        self.state["status"] = "training"
        self.save_suite()
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke("--report-only", "--recovery-dir", str(recovery))
        self.assertEqual(len(self.calls()), 1)

    def test_report_only_rejects_unfinished_other_jobs(self):
        recovery, _, _ = self.recover()
        self.add_other_job(status="pending", train_status="pending")
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke("--report-only", "--recovery-dir", str(recovery))
        self.assertEqual(len(self.calls()), 1)

    def test_report_only_checks_all_split_fingerprints(self):
        other = self.add_other_job()
        recovery, _, _ = self.recover()
        for key in self.fingerprints:
            with self.subTest(fingerprint=key):
                changed = dict(self.fingerprints, **{key: "different-frozen-value"})
                write_json(other / "split_metadata.json", changed)
                with self.assertRaisesRegex((ValueError, RuntimeError), "fingerprint"):
                    self.invoke("--report-only", "--recovery-dir", str(recovery))
                self.assertEqual(len(self.calls()), 1)
        write_json(other / "split_metadata.json", self.fingerprints)

    def test_show_metrics_is_read_only_without_conda_and_uses_only_explicit_recovery(self):
        from foundation_sft.recover_experiment import show_metrics
        self.add_other_job()
        recovery, _, _ = self.recover()
        self.job["run_dir"] = str(self.failed)
        write_json(self.failed / "test" / "metrics.json", {"patient": {"roc_auc": 0.99999}})
        self.state["jobs"].append({
            "id": "cxformer_lora_seed42", "model": "cxformer", "mode": "lora", "seed": 42,
            "env": "cxformer", "device": "cuda:2", "status": "pending", "train_status": "running",
            "run_dir": str(self.root / "not_yet_created_running_run"),
        })
        self.state["status"] = "training"
        self.save_suite()
        preserved = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        stdout = io.StringIO()
        args = ["--suite", str(self.suite), "--show-metrics"]
        with redirect_stdout(stdout), patch("foundation_sft.recover_experiment.subprocess.Popen") as launch, \
                patch("foundation_sft.recover_experiment.shutil.which") as detect_conda:
            self.assertEqual(self.main(args), 0)
            rows = show_metrics(self.suite)
            self.assertEqual(self.main([*args, "--recovery-dir", str(recovery)]), 0)
            recovered_rows = show_metrics(self.suite, recovery)
            launch.assert_not_called()
            detect_conda.assert_not_called()
        output = stdout.getvalue()
        for header in ("Model", "Mode", "Seed", "Status", "Val-AUC", "ROC-AUC", "ROC95%CI",
                       "PR-AUC", "AP", "Sensitivity", "Specificity"):
            self.assertIn(header, output)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][:4], ("eva_x", "lora", "42", "failed"))
        self.assertEqual(rows[0][4:], ("-",) * 7)
        self.assertEqual(rows[1], ("rad_dino", "lora", "42", "completed", "0.8125", "0.7500",
                                   "[0.6000,0.9000]", "0.8000", "0.7000", "0.8000", "0.9000"))
        self.assertEqual(rows[2][:4], ("cxformer", "lora", "42", "running"))
        self.assertEqual(rows[2][4:], ("-",) * 7)
        self.assertEqual(len(recovered_rows), 4)
        self.assertEqual(recovered_rows[-1][:4], ("eva_x", "lora", "42", "recovered"))
        self.assertEqual(recovered_rows[-1][5], "0.7500")
        self.assertNotIn("1.0000", output)
        self.assertEqual(preserved, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
