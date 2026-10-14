"""Inventory all saved experiment sources without mixing or modifying runs."""

import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import show_foundation_results as viewer


class AllResultsViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="results-view-")
        self.project = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_artifacts(self, root, model="eva_x", mode="lora", seed=42, name="run_01", auc=0.8,
                      cohort="cohort-a", confidence=0.95, evaluation=False, amp=True):
        run = self.project / "runs" / root / model / mode / name
        config = {"model": model, "mode": mode, "seed": seed, "device": "cuda:0", "training": {"amp": amp}}
        self.write(run / "config.json", config)
        self.write(run / "model_metadata.json", {})
        self.write(run / "environment.json", {"cohort_hash": cohort})
        self.write(run / "split_metadata.json", {"_manifest_hash": cohort, "_cohort_hash": cohort + "-data",
                                                  "_source_hash": "label-source"})
        self.write(run / "selection.json", {"best_val_patient_roc_auc": 0.85})
        metrics = {"patient": {"roc_auc": auc, "pr_auc": 0.45, "average_precision": 0.46,
                               "sensitivity": 0.75, "specificity": 0.8},
                   "bootstrap": {"confidence_level": confidence,
                                 "patient": {"roc_auc": {"lower": 0.7, "upper": 0.9}}}}
        self.write(run / ("metrics.json" if evaluation else "test/metrics.json"), metrics)
        return run, config

    def suite(self, root, jobs, name="suite_01"):
        suite = self.project / "runs" / root / "suites" / name
        self.write(suite / "suite.json", {"project_dir": str(self.project), "jobs": jobs})
        return suite

    def job(self, run, config, status="completed", job_id=None):
        return {"id": job_id or config["model"] + "_" + config["mode"],
                **{key: config[key] for key in ("model", "mode", "seed")},
                "status": status, "train_status": status, "run_dir": str(run)}

    def test_all_roots_and_standalone_evaluation_are_visible(self):
        a, ac = self.run_artifacts("foundation_sft")
        b, bc = self.run_artifacts("foundation_sft_vlm", "xray_siglip")
        c, _ = self.run_artifacts("foundation_sft_imagenet", "radjepa", cohort="cohort-b")
        d, _ = self.run_artifacts("foundation_sft", name="evaluation", evaluation=True)
        self.suite("foundation_sft", [self.job(a, ac)])
        self.suite("foundation_sft_vlm", [self.job(b, bc)])
        rows, info = viewer.collect_results(self.project)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["run_dir"] for row in rows}, {str(a), str(b), str(c), str(d)})
        self.assertEqual({row["kind"] for row in rows}, {"suite", "standalone", "evaluation"})
        self.assertEqual({row["cohort_hash"] for row in rows}, {"cohort-a-data", "cohort-b-data"})
        self.assertEqual({row["split_hash"] for row in rows}, {"cohort-a", "cohort-b"})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            viewer.print_results(rows, info)
        self.assertIn("xray_siglip", stdout.getvalue())
        self.assertIn("radjepa", stdout.getvalue())
        self.assertIn("distinct completed/evaluated runs: 4", stdout.getvalue())

    def test_failed_original_and_each_recovery_are_kept_without_selecting_best(self):
        original, config = self.run_artifacts("foundation_sft", "cxformer", name="failed_original")
        (original / "test/metrics.json").unlink()
        job = self.job(original, config, "failed", "cxformer_lora_seed42")
        del job["run_dir"]
        log = self.project / "train.log"
        log.write_text("Run directory: " + str(original) + "\nNon-finite training loss\n")
        job["train_log"] = str(log)
        suite = self.suite("foundation_sft", [job])
        for index, auc in enumerate((0.72, 0.89)):
            recovered, _ = self.run_artifacts("foundation_sft", "cxformer", name="recovered_" + str(index),
                                              auc=auc, amp=False)
            self.write(recovered / "resume.json", {"precision_changed": True, "source_amp": True, "target_amp": False})
            self.write(suite / ("recovery_" + str(index)) / "recovery_manifest.json",
                       {"suite": str(suite), "job_id": job["id"], "status": "completed",
                        "source_run_dir": str(original), "recovered_run_dir": str(recovered)})
        rows, _ = viewer.collect_results(self.project)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["status"] for row in rows], ["failed", "recovered", "recovered"])
        self.assertEqual(rows[0]["run_dir"], str(original))
        self.assertEqual([row["roc_auc"] for row in rows[1:]], [0.72, 0.89])
        self.assertTrue(all(row["precision"] == "AMP->FP32" for row in rows[1:]))

    def test_overlapping_roots_do_not_duplicate_records(self):
        run, config = self.run_artifacts("foundation_sft_vlm", "xray_siglip")
        self.suite("foundation_sft_vlm", [self.job(run, config)])
        rows, info = viewer.collect_results(self.project, [self.project / "runs", self.project / "runs/foundation_sft_vlm"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(info["suites"], 1)

    def test_failed_running_and_pending_jobs_in_training_batch_remain_visible(self):
        run, config = self.run_artifacts("foundation_sft")
        jobs = [self.job(run, config)]
        for status in ("failed", "running", "pending"):
            jobs.append({"id": status, "model": status, "mode": "lora", "seed": 42,
                         "status": status, "train_status": status})
        self.suite("foundation_sft", jobs)
        rows, _ = viewer.collect_results(self.project)
        self.assertEqual({row["status"] for row in rows}, {"completed", "failed", "running", "pending"})
        for row in rows[1:]:
            self.assertEqual(row["roc_auc"], "")

    def test_check_only_batches_are_explicitly_counted_and_optionally_printed(self):
        self.suite("foundation_sft", [{"id": "check", "model": "eva_x", "mode": "lora", "seed": 42,
                                       "status": "checked"}])
        rows, info = viewer.collect_results(self.project)
        self.assertEqual(rows, [])
        self.assertEqual(info["skipped_check_suites"], 1)
        rows, info = viewer.collect_results(self.project, include_checks=True)
        self.assertEqual(rows[0]["status"], "checked")
        self.assertEqual(info["skipped_check_suites"], 0)

    def test_saved_confidence_level_is_used_and_zero_seed_preserved(self):
        self.run_artifacts("foundation_sft", seed=0, confidence=0.9)
        rows, info = viewer.collect_results(self.project)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            viewer.print_results(rows, info)
        self.assertIn("90%[0.7000,0.9000]", stdout.getvalue())
        self.assertEqual(rows[0]["seed"], 0)
        self.assertNotIn("95%", stdout.getvalue())

    def test_malformed_missing_or_mismatched_artifacts_do_not_hide_other_batches(self):
        run, config = self.run_artifacts("foundation_sft", name="mismatched")
        wrong_job = self.job(run, config)
        wrong_job["model"] = "other_model"
        self.suite("foundation_sft", [wrong_job])
        valid, _ = self.run_artifacts("foundation_sft_vlm", "xray_siglip")
        malformed = self.project / "runs/foundation_sft/suites/bad/suite.json"
        malformed.parent.mkdir(parents=True)
        malformed.write_text("not JSON")
        rows, info = viewer.collect_results(self.project)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["status"], "missing_artifacts")
        self.assertEqual(rows[0]["roc_auc"], "")
        self.assertIn(str(valid), [row["run_dir"] for row in rows])
        self.assertTrue(any(row["model"] == "eva_x" and row["status"] == "completed" for row in rows))
        self.assertEqual(len(info["diagnostics"]), 1)

    def test_malformed_recovery_does_not_become_a_successful_recovery(self):
        run, config = self.run_artifacts("foundation_sft")
        suite = self.suite("foundation_sft", [self.job(run, config)])
        self.write(suite / "recovery_wrong/recovery_manifest.json",
                   {"suite": str(suite), "job_id": "wrong-job", "status": "completed", "recovered_run_dir": str(run)})
        rows, info = viewer.collect_results(self.project)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(info["diagnostics"]), 1)

    def test_rejected_recovery_mapping_cannot_hide_valid_standalone_results(self):
        original, config = self.run_artifacts("foundation_sft")
        other, _ = self.run_artifacts("foundation_sft_vlm", "xray_siglip")
        suite = self.suite("foundation_sft", [self.job(original, config)])
        for name, job_id in (("wrong-job", "not-in-suite"), ("wrong-identity", "eva_x_lora")):
            self.write(suite / ("recovery_" + name) / "recovery_manifest.json",
                       {"suite": str(suite), "job_id": job_id, "status": "completed", "recovered_run_dir": str(other)})
        rows, info = viewer.collect_results(self.project)
        self.assertEqual(sum(row["model"] == "xray_siglip" and row["status"] == "completed" for row in rows), 1)
        self.assertEqual(sum(row["status"] == "missing_artifacts" for row in rows), 1)
        self.assertEqual(len(info["diagnostics"]), 1)

    def test_split_fingerprint_fallback_keeps_same_digest_type(self):
        run, _ = self.run_artifacts("foundation_sft", cohort="assignment-digest")
        (run / "environment.json").unlink()
        rows, _ = viewer.collect_results(self.project)
        self.assertEqual(rows[0]["split_hash"], "assignment-digest")
        self.assertEqual(rows[0]["cohort_hash"], "assignment-digest-data")
        self.assertEqual(rows[0]["source_hash"], "label-source")

    def test_csv_is_optional_new_file_and_original_results_unchanged(self):
        self.run_artifacts("foundation_sft")
        before = {str(path): path.read_bytes() for path in self.project.rglob("*.json")}
        output = self.project / "all_results.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(viewer.main(["--project", str(self.project), "--csv", str(output)]), 0)
        after = {str(path): path.read_bytes() for path in self.project.rglob("*.json")}
        self.assertEqual(before, after)
        with output.open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        self.assertEqual(exported[0]["roc_auc"], "0.8")
        original = output.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            viewer.main(["--project", str(self.project), "--csv", str(output)])
        self.assertEqual(output.read_bytes(), original)

    def test_configured_models_with_no_saved_result_are_shown_explicitly(self):
        original, config = self.run_artifacts("foundation_sft", "eva_x")
        self.suite("foundation_sft", [self.job(original, config)])
        self.write(self.project / "foundation_sft/configs/experiments.json",
                   {"jobs": [{"model": model} for model in ("eva_x", "rad_dino", "cxformer", "chexfound", "radjepa")],
                    "modes": ["linear_probe", "lora"], "seeds": [42]})
        self.write(self.project / "foundation_sft/configs/experiments_vlm.json",
                   {"jobs": [{"model": model} for model in ("cxr_clip", "xray_siglip", "biomedclip", "chexzero")],
                    "modes": ["linear_probe", "lora"], "seeds": [42]})
        rows, _ = viewer.collect_results(self.project)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({row["model"] for row in rows}), 9)
        self.assertEqual(sum(row["status"] == "no_saved_run" for row in rows), 17)
        for row in rows:
            if row["status"] == "no_saved_run":
                self.assertEqual(row["roc_auc"], "")
        output = self.project / "eight_models.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            viewer.main(["--project", str(self.project), "--csv", str(output), "--models",
                         "EVA-X", "rad_dino", "cxformer", "CheXFound", "radjepa",
                         "xray_siglip", "biomedclip", "chexzero"])
        with output.open(encoding="utf-8-sig", newline="") as stream:
            filtered = list(csv.DictReader(stream))
        self.assertEqual(len(filtered), 16)
        self.assertEqual(len({row["model"] for row in filtered}), 8)
        self.assertFalse(any(row["model"] == "cxr_clip" for row in filtered))

    def test_script_has_python37_compatible_syntax_and_no_ml_imports(self):
        source = Path(viewer.__file__).read_text()
        tree = ast.parse(source, feature_version=(3, 7))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        self.assertFalse(imported & {"torch", "numpy", "pandas", "pydicom", "foundation_sft"})


if __name__ == "__main__":
    unittest.main()
