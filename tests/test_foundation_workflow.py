"""Workflow barriers, subprocess receipts and GPU scheduling without real GPUs.

These tests exercise the orchestration contract. Actual training and evaluation
on a tiny local DINOv2 encoder are covered by test_foundation_training.py.
"""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.config import DEFAULTS


class FakeConda:
    """Create the same receipts/artifacts as the CLI, recording real overlap."""

    def __init__(self, root, *, fail=None, omit_receipt=None, corrupt=None,
                 training_delay=0.0, report_failed=False):
        self.root = Path(root)
        self.fail = fail
        self.omit_receipt = omit_receipt
        self.corrupt = corrupt
        self.training_delay = training_delay
        self.report_failed = report_failed
        self.calls = []
        self.events = []
        self.active_devices = set()
        self.same_device_overlap = False
        self.max_active_devices = 0
        self.lock = threading.Lock()
        self.cohort_hash = "synthetic-frozen-patient-cohort"

    @staticmethod
    def argument(command, name):
        return command[command.index(name) + 1]

    def __call__(self, command, **kwargs):
        command = list(command)
        if "foundation_sft.report" in command:
            with self.lock:
                self.calls.append(("report", None, command))
            report = Path(self.argument(command, "--suite")) / "report"
            if not self.report_failed:
                report.mkdir(parents=True, exist_ok=True)
                (report / "run_metrics.csv").write_text("model,mode,seed\n")
            return subprocess.CompletedProcess(command, 19 if self.report_failed else 0)
        action = self.argument(command, "--action")
        config = json.loads(Path(self.argument(command, "--config")).read_text())
        receipt = Path(self.argument(command, "--result-json"))
        identity = (action, config["model"], config["mode"], config["seed"])
        device = config["device"]
        with self.lock:
            self.calls.append((action, config, command))
            self.events.append(("start", identity))
            if action == "train":
                if device in self.active_devices:
                    self.same_device_overlap = True
                self.active_devices.add(device)
                self.max_active_devices = max(self.max_active_devices, len(self.active_devices))
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write(f"Synthetic subprocess: {action} {config['model']} {config['mode']}\n")
            stdout.flush()
        if action == "train" and self.training_delay:
            time.sleep(self.training_delay)
        failed = self.fail is not None and self.fail(action, config)
        omitted = self.omit_receipt is not None and self.omit_receipt(action, config)
        if not failed and not omitted:
            payload = {"action": action}
            if action == "prepare":
                payload["cohort_hash"] = self.cohort_hash
            elif action == "smoke":
                payload["smoke_passed"] = True
            elif action == "train":
                run_dir = receipt.parent / f"{receipt.stem}_synthetic_model_run"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "test").mkdir(exist_ok=True)
                (run_dir / "best.pt").write_bytes(b"synthetic checkpoint")
                (run_dir / "last.pt").write_bytes(b"synthetic checkpoint")
                (run_dir / "config.json").write_text(json.dumps(config))
                (run_dir / "environment.json").write_text(json.dumps({"cohort_hash": self.cohort_hash}))
                (run_dir / "selection.json").write_text(json.dumps({"threshold": 0.5}))
                (run_dir / "test" / "metrics.json").write_text(json.dumps({"patient": {"roc_auc": 0.75}}))
                (run_dir / "test" / "patient_predictions.csv").write_text(
                    "patient_id,label,probability\nsynthetic-a,0,0.25\nsynthetic-b,1,0.75\n"
                )
                payload.update(run_dir=str(run_dir.resolve()), model=config["model"],
                               mode=config["mode"], seed=config["seed"])
            if self.corrupt is not None:
                self.corrupt(action, config, payload)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(payload))
        with self.lock:
            if action == "train":
                self.active_devices.remove(device)
            self.events.append(("end", identity))
        return subprocess.CompletedProcess(command, 17 if failed else 0)


class FoundationWorkflowTests(unittest.TestCase):
    def setUp(self):
        from foundation_sft import workflow
        self.workflow = workflow
        self.temporary = tempfile.TemporaryDirectory(prefix="foundation-workflow-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, jobs=None):
        specifications = jobs or [
            ("rad_dino", "linear_probe", "cuda:0", 42),
            ("rad_dino", "lora", "cuda:0", 42),
            ("cxformer", "lora", "cuda:1", 42),
        ]
        result = {"suite_root": str(self.root / "suites"),
                  "conda_executable": "/synthetic/conda", "jobs": []}
        for index, (model, mode, device, seed) in enumerate(specifications):
            config = deepcopy(DEFAULTS)
            config.update(model=model, mode=mode, device=device, seed=seed,
                          output_root=str(self.root / "model_runs"),
                          split_manifest=str(self.root / "shared_patient_split.csv"))
            config["data"].update(excel_path=str(self.root / "synthetic.xlsx"),
                                   dicom_dir=str(self.root / "synthetic_dicoms"))
            result["jobs"].append({"id": f"job-{index}", "model": model,
                                    "mode": mode, "device": device, "seed": seed,
                                    "env": "synthetic_env", "config": config})
        return result

    def run_suite(self, plan, runner, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(
            self.workflow.subprocess, "run", side_effect=runner
        ):
            suite_dir = self.workflow.run_suite(plan, **kwargs)
        self.assertIsInstance(suite_dir, Path)
        manifest = json.loads((suite_dir / "suite.json").read_text())
        return suite_dir, manifest

    def test_prepare_and_every_smoke_finish_before_training_and_receipts_are_independent(self):
        runner = FakeConda(self.root)
        suite_dir, manifest = self.run_suite(self.plan(), runner)
        actions = [action for action, _, _ in runner.calls]
        self.assertEqual(actions.count("prepare"), 1)
        self.assertEqual(actions.count("smoke"), 3)
        self.assertEqual(actions.count("train"), 3)
        self.assertEqual(actions[-1], "report")
        completed_smokes = set()
        for event, identity in runner.events:
            if event == "end" and identity[0] == "smoke":
                completed_smokes.add(identity[1:])
            if event == "start" and identity[0] == "train":
                self.assertEqual(len(completed_smokes), 3)
        self.assertTrue(all(job["status"] == "completed" for job in manifest["jobs"]))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["report_status"], "completed")
        outputs = [Path(job["run_dir"]) for job in manifest["jobs"]]
        self.assertEqual(len(set(outputs)), 3)
        for job, output in zip(manifest["jobs"], outputs):
            actual = json.loads((output / "config.json").read_text())
            for field in ("model", "mode", "seed"):
                self.assertEqual(actual[field], job[field])
        # The subprocess console text must be preserved for server debugging.
        logs = list(suite_dir.rglob("*.log"))
        self.assertTrue(logs)
        self.assertTrue(any("Synthetic subprocess:" in path.read_text() for path in logs))
        for action, _, command in runner.calls:
            if action != "report":
                self.assertEqual(command[0], "/synthetic/conda")
                self.assertIn("--no-capture-output", command)
                self.assertIn("-n", command)

    def test_check_only_never_trains_or_reports(self):
        runner = FakeConda(self.root)
        _, manifest = self.run_suite(self.plan(), runner, check_only=True)
        self.assertEqual([call[0] for call in runner.calls].count("prepare"), 1)
        self.assertEqual([call[0] for call in runner.calls].count("smoke"), 3)
        self.assertNotIn("train", [call[0] for call in runner.calls])
        self.assertNotIn("report", [call[0] for call in runner.calls])
        self.assertFalse(any(job.get("run_dir") for job in manifest["jobs"]))
        self.assertEqual(manifest["status"], "checked")

    def test_failed_smoke_blocks_all_training(self):
        runner = FakeConda(self.root, fail=lambda action, config:
                           action == "smoke" and config["model"] == "cxformer")
        suite_dir, manifest = self.run_suite(self.plan(), runner)
        self.assertNotIn("train", [call[0] for call in runner.calls])
        self.assertNotIn("report", [call[0] for call in runner.calls])
        self.assertTrue(any(job["status"] == "failed" for job in manifest["jobs"]))
        self.assertTrue(any("Synthetic subprocess: smoke cxformer" in path.read_text()
                            for path in suite_dir.rglob("*.log")))

    def test_success_exit_without_receipt_does_not_pass_smoke(self):
        runner = FakeConda(self.root, omit_receipt=lambda action, config:
                           action == "smoke" and config["model"] == "cxformer")
        _, manifest = self.run_suite(self.plan(), runner)
        self.assertNotIn("train", [call[0] for call in runner.calls])
        self.assertTrue(any(job["status"] == "failed" for job in manifest["jobs"]))

    def test_one_training_failure_does_not_mislabel_other_jobs_or_lose_outputs(self):
        runner = FakeConda(self.root, fail=lambda action, config:
                           action == "train" and config["mode"] == "linear_probe")
        _, manifest = self.run_suite(self.plan(), runner)
        self.assertEqual([call[0] for call in runner.calls].count("train"), 3)
        failures = [job for job in manifest["jobs"] if job["status"] == "failed"]
        successes = [job for job in manifest["jobs"] if job["status"] == "completed"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(successes), 2)
        self.assertFalse(failures[0].get("run_dir"))
        self.assertEqual(len({job["run_dir"] for job in successes}), 2)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["report_status"], "completed")

    def test_data_or_frozen_split_disagreement_rejected_before_subprocesses(self):
        for field in ("dicom_dir", "split_manifest", "split_seed", "split_ratios"):
            with self.subTest(field=field):
                plan = self.plan()
                other = plan["jobs"][1]["config"]
                if field == "dicom_dir":
                    other["data"][field] = str(self.root / "different_dicoms")
                elif field == "split_manifest":
                    other[field] = str(self.root / "different_split.csv")
                elif field == "split_seed":
                    other[field] += 1
                else:
                    other[field] = [0.6, 0.2, 0.2]
                runner = FakeConda(self.root)
                with patch.object(self.workflow.subprocess, "run", side_effect=runner):
                    with self.assertRaises(ValueError):
                        self.workflow.run_suite(plan)
                self.assertFalse(runner.calls)

    def test_gpu_jobs_are_serial_per_device_but_parallel_across_devices(self):
        runner = FakeConda(self.root, training_delay=0.12)
        _, manifest = self.run_suite(self.plan(), runner, max_parallel=3)
        self.assertFalse(runner.same_device_overlap)
        self.assertGreaterEqual(runner.max_active_devices, 2)
        self.assertTrue(all(job["status"] == "completed" for job in manifest["jobs"]))

    def test_training_receipt_cannot_claim_another_model(self):
        def corrupt(action, config, payload):
            if action == "train" and config["model"] == "cxformer":
                payload["model"] = "rad_dino"

        runner = FakeConda(self.root, corrupt=corrupt)
        _, manifest = self.run_suite(self.plan(), runner)
        cxformer = [job for job in manifest["jobs"] if job["model"] == "cxformer"]
        self.assertEqual(cxformer[0]["status"], "failed")
        self.assertFalse(cxformer[0].get("run_dir"))

    def test_failed_prepare_never_smokes_or_trains(self):
        runner = FakeConda(self.root, fail=lambda action, config: action == "prepare")
        _, manifest = self.run_suite(self.plan(), runner)
        self.assertEqual([call[0] for call in runner.calls], ["prepare"])
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(any(job.get("run_dir") for job in manifest["jobs"]))

    def test_wrong_cohort_or_missing_artifact_cannot_complete_training(self):
        for problem in ("cohort", "artifact"):
            with self.subTest(problem=problem):
                def corrupt(action, config, payload):
                    if action != "train" or config["model"] != "cxformer":
                        return
                    run = Path(payload["run_dir"])
                    if problem == "cohort":
                        (run / "environment.json").write_text(json.dumps({"cohort_hash": "different-patients"}))
                    else:
                        (run / "test" / "patient_predictions.csv").unlink()

                runner = FakeConda(self.root, corrupt=corrupt)
                _, manifest = self.run_suite(self.plan(), runner)
                cxformer = [job for job in manifest["jobs"] if job["model"] == "cxformer"][0]
                self.assertEqual(cxformer["status"], "failed")
                self.assertFalse(cxformer.get("run_dir"))

    def test_report_failure_preserves_completed_training_but_fails_suite(self):
        runner = FakeConda(self.root, report_failed=True)
        suite, manifest = self.run_suite(self.plan(), runner)
        self.assertTrue(all(job["status"] == "completed" for job in manifest["jobs"]))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["report_status"], "failed")
        self.assertIn("19", manifest["report_error"])
        self.assertTrue((suite / "logs" / "report.log").is_file())


if __name__ == "__main__":
    unittest.main()
