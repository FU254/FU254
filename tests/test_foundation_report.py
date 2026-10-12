"""Ensure comparison reports preserve the declared runs and held-out cohort."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.metrics import aggregate_patient_predictions, evaluate_predictions
from foundation_sft.report import summarize_runs, summarize_suite


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class FoundationReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counter = 0

    def run_fixture(self, model="rad_dino", mode="lora", seed=42,
                    probabilities=(0.1, 0.8, 0.7, 0.9), bootstrap_samples=12,
                    research=False, cohort_hash="same-cohort"):
        self.counter += 1
        run = self.root / f"run{self.counter}"
        (run / "test").mkdir(parents=True)
        config = {"model": model, "mode": mode, "seed": seed,
                  "training": {"epochs": 3, "head_lr": 0.001}}
        if research:
            config["preprocessing_reference"] = {"status": "research_choice_not_author_confirmed"}
        write_json(run / "config.json", config)
        write_json(run / "environment.json", {"cohort_hash": cohort_hash})
        write_json(run / "selection.json", {"threshold": 0.5, "source": "validation patient predictions",
                                            "best_val_patient_roc_auc": 0.8125})
        write_json(run / "history.json", [{"epoch": 1, "train_loss": 0.8}])
        write_json(run / "model_metadata.json", {
            "mode": mode, "rank": 8, "alpha": 16,
            "preprocessing": {"image_size": 224, "mean": [0.5] * 3, "std": [0.2] * 3},
            "pretrained": {"model": model, "checkpoint_sha256": "fixture-sha"},
            "total_parameters": 100, "trainable_parameters": 10,
        })
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"],
            # Leading zeros and the literal NA must survive CSV loading.
            "patient_id": ["001", "002", "003", "NA"],
            "label": [0, 0, 1, 1], "probability": probabilities,
        })
        metrics = evaluate_predictions(frame, threshold=0.5, bootstrap_samples=bootstrap_samples, seed=seed)
        write_json(run / "test" / "metrics.json", metrics)
        aggregate_patient_predictions(frame).to_csv(run / "test" / "patient_predictions.csv", index=False)
        return run, metrics

    def test_declared_seeds_all_appear_with_correct_ci_and_sample_std(self):
        first, first_metrics = self.run_fixture()
        second, _ = self.run_fixture(seed=43, probabilities=(0.1, 0.2, 0.8, 0.9))
        # An unlisted better-scoring directory must not be discovered or selected.
        self.run_fixture(seed=44, probabilities=(0.01, 0.02, 0.98, 0.99))
        output = summarize_runs([first, second], self.root / "report")
        per_run = pd.read_csv(output / "run_metrics.csv")
        self.assertEqual(per_run.seed.tolist(), [42, 43])
        self.assertEqual(per_run.roc_auc.tolist(), [0.75, 1.0])
        ci = first_metrics["bootstrap"]["patient"]["roc_auc"]
        self.assertEqual(per_run.iloc[0].roc_auc_ci_low, ci["lower"])
        self.assertEqual(per_run.iloc[0].roc_auc_ci_high, ci["upper"])
        summary = pd.read_csv(output / "model_mode_summary.csv")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0].n_seeds, 2)
        self.assertAlmostEqual(summary.iloc[0].roc_auc_mean, 0.875)
        self.assertAlmostEqual(summary.iloc[0].roc_auc_std, np.std([0.75, 1], ddof=1))
        self.assertTrue((output / "patient_roc_lora.png").is_file())
        self.assertTrue((output / "patient_pr_lora.png").is_file())
        self.assertIn("不按测试分数挑选", (output / "README.md").read_text(encoding="utf-8"))

    def test_single_seed_missing_ci_and_research_protocol_remain_explicit(self):
        first, _ = self.run_fixture(model="radjepa", bootstrap_samples=0)
        second, _ = self.run_fixture(model="radjepa", seed=43, bootstrap_samples=0, research=True)
        output = summarize_runs([first, second], self.root / "report")
        summary = pd.read_csv(output / "model_mode_summary.csv")
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary.roc_auc_std.isna().all())
        self.assertEqual(set(summary.normalization_status), {"model_adapter_default", "research_choice_not_author_confirmed"})
        per_run = pd.read_csv(output / "run_metrics.csv")
        self.assertTrue(per_run.roc_auc_ci_low.isna().all())
        self.assertIn("尚未核实为作者", (output / "README.md").read_text(encoding="utf-8"))
        protocols = json.loads((output / "protocols.json").read_text())
        self.assertEqual(len(protocols), 2)

    def test_changed_hyperparameters_are_not_averaged_as_repeat_seeds(self):
        first, _ = self.run_fixture()
        second, _ = self.run_fixture(seed=43)
        config = json.loads((second / "config.json").read_text())
        config["training"]["head_lr"] = 0.002
        write_json(second / "config.json", config)
        output = summarize_runs([first, second], self.root / "report")
        self.assertEqual(len(pd.read_csv(output / "model_mode_summary.csv")), 2)

    def test_distinct_cohort_hashes_fail_before_creating_report(self):
        first, _ = self.run_fixture()
        second, _ = self.run_fixture(model="cxformer", cohort_hash="different-cohort")
        output = self.root / "report"
        with self.assertRaisesRegex(ValueError, "cohort_hash"):
            summarize_runs([first, second], output)
        self.assertFalse(output.exists())

    def test_actual_test_patient_ids_and_labels_are_checked_even_with_same_hash(self):
        first, _ = self.run_fixture()
        second, _ = self.run_fixture(model="cxformer")
        path = second / "test" / "patient_predictions.csv"
        original = pd.read_csv(path, dtype={"patient_id": str}, keep_default_na=False)
        reordered = original.iloc[::-1]
        reordered.to_csv(path, index=False)
        # Row order does not change the cohort.
        summarize_runs([first, second], self.root / "aligned")
        changed = original.copy()
        changed.loc[0, "patient_id"] = "another-patient"
        changed.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "test patient IDs"):
            summarize_runs([first, second], self.root / "wrong_ids")
        # Relabel and rewrite metrics consistently: comparing actual labels must
        # still detect the difference even if the saved scores match predictions.
        changed = original.copy()
        changed["label"] = [1, 0, 0, 1]
        changed.to_csv(path, index=False)
        image_frame = changed.assign(sample_id=["a", "b", "c", "d"])
        write_json(second / "test" / "metrics.json", evaluate_predictions(image_frame, bootstrap_samples=0))
        with self.assertRaisesRegex(ValueError, "test patient labels"):
            summarize_runs([first, second], self.root / "wrong_labels")

    def test_duplicate_seed_and_incomplete_or_inconsistent_run_are_errors(self):
        first, _ = self.run_fixture()
        second, _ = self.run_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate model/mode/seed"):
            summarize_runs([first, second], self.root / "duplicate")
        selection = json.loads((second / "selection.json").read_text())
        selection["threshold"] = 0.95
        write_json(second / "selection.json", selection)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            summarize_runs([second], self.root / "inconsistent")
        (second / "selection.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize_runs([second], self.root / "incomplete")
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_runs([], self.root / "empty")

    def test_suite_includes_completed_and_records_failed_without_reading_its_run(self):
        first, _ = self.run_fixture()
        suite = self.root / "suite"
        suite.mkdir()
        jobs = [
            {"id": "rad-lora-42", "model": "rad_dino", "mode": "lora", "seed": 42,
             "status": "completed", "run_dir": "../run1"},
            {"id": "cx-lora-42", "model": "cxformer", "mode": "lora", "seed": 42,
             "status": "failed", "run_dir": "nonexistent", "error": "GPU out of memory"},
        ]
        write_json(suite / "suite.json", {"jobs": jobs})
        output = summarize_suite(suite)
        self.assertEqual(pd.read_csv(output / "run_metrics.csv").job_id.tolist(), ["rad-lora-42"])
        excluded = pd.read_csv(output / "excluded_jobs.csv")
        self.assertEqual(excluded.status.tolist(), ["failed"])
        self.assertEqual(excluded.error.tolist(), ["GPU out of memory"])
        inconsistent = deepcopy(jobs)
        inconsistent[0]["model"] = "cxformer"
        write_json(suite / "suite.json", {"jobs": inconsistent})
        with self.assertRaisesRegex(ValueError, "inconsistent model"):
            summarize_suite(suite, suite / "inconsistent")

    def test_test_selected_threshold_and_non95_interval_are_rejected(self):
        run, metrics = self.run_fixture()
        selection_path = run / "selection.json"
        selection = json.loads(selection_path.read_text())
        selection["source"] = "test patient predictions"
        write_json(selection_path, selection)
        with self.assertRaisesRegex(ValueError, "validation patients"):
            summarize_runs([run], self.root / "test_tuned")
        selection["source"] = "validation patient predictions"
        write_json(selection_path, selection)
        metrics["bootstrap"]["confidence_level"] = 0.90
        write_json(run / "test" / "metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "95%"):
            summarize_runs([run], self.root / "wrong_ci")

    def test_suite_with_no_completed_runs_reports_exclusions_and_never_overwrites(self):
        suite = self.root / "suite"
        suite.mkdir()
        write_json(suite / "suite.json", {"jobs": [{"id": "cx", "status": "failed", "error": "No weights"}]})
        output = summarize_suite(suite)
        self.assertTrue(pd.read_csv(output / "run_metrics.csv").empty)
        self.assertIn("尚无完成训练", (output / "README.md").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            summarize_suite(suite)


if __name__ == "__main__":
    unittest.main()
