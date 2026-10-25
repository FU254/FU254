"""The fixed-WD DropPath screen uses validation and same-seed comparisons."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from foundation_sft import droppath_baseline as source
from foundation_sft import droppath_report as report
from tests.test_droppath_baseline import make_baseline


class DropPathReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = make_baseline(self.root)
        baseline = source.load_baseline(original, self.root)
        self.suite = self.root / "new_dp" / "suite"
        self.state = {"baseline": baseline, "arms": {"drop_path_0p1": {
            "drop_path_rate": .1, "suite": str(self.suite)}}}
        jobs = {}
        for name, item in baseline["jobs"].items():
            record = {key: deepcopy(item[key]) for key in ("job", "check", "result")}
            record["status"] = "completed"
            run = self.root / "new_dp" / name
            run.mkdir(parents=True)
            record["result"].update(run=str(run), best_val_patient_roc_auc=.82)
            record["result"]["test_patient"].update(roc_auc=.60, pr_auc=.30, average_precision=.31)
            record["job"]["config"]["training"]["drop_path_rate"] = .1
            history = [
                {"epoch": 1, "train_loss": 900., "train_eval_loss": .10, "val_loss": .12,
                 "train_eval_patient_roc_auc": .81, "val_patient_roc_auc": .80, "val_patient_pr_auc": .40},
                {"epoch": 2, "train_loss": 899., "train_eval_loss": .08, "val_loss": .11,
                 "train_eval_patient_roc_auc": .85, "val_patient_roc_auc": .82, "val_patient_pr_auc": .45},
                {"epoch": 3, "train_loss": 898., "train_eval_loss": .06, "val_loss": .13,
                 "train_eval_patient_roc_auc": .87, "val_patient_roc_auc": .81, "val_patient_pr_auc": .43},
            ]
            (run / "history.json").write_text(json.dumps(history))
            jobs[name] = record
        self.child = {"jobs": jobs}
        self.suite.mkdir(parents=True, exist_ok=True)
        self.save_child()

    def save_child(self):
        (self.suite / "suite.json").write_text(json.dumps(self.child))

    def rows(self):
        return report.metric_rows(self.root, self.state)

    def test_comparison_contains_only_four_seed42_records_at_fixed_weight_decay(self):
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["weight_decay"] for row in rows}, {.01})
        self.assertEqual({row["seed"] for row in rows}, {42})
        self.assertEqual({row["drop_path_rate"] for row in rows}, {0., .1})
        self.assertEqual({row["val_pr_auc"] for row in rows if row["arm"] != "baseline"}, {.45})
        self.assertTrue(all(row["roc_auc_sd"] is None and row["n_completed"] == 1
                            for row in report.summary_rows(rows)))

    def test_selection_uses_validation_even_when_test_scores_are_lower(self):
        selected = report.validation_selection(self.rows())
        self.assertEqual(selected["status"], "complete")
        self.assertEqual(selected["selected_drop_path_rate"], .1)
        self.assertFalse(selected["test_metrics_used"])
        rows = self.rows()
        for row in rows:
            row["roc_auc"] = 1. if row["arm"] == "baseline" else 0.
        self.assertEqual(report.validation_selection(rows)["selected_arm"], "drop_path_0p1")

    def test_ties_keep_baseline_and_incomplete_studies_have_no_recommendation(self):
        rows = self.rows()
        for row in rows:
            row["val_auc"] = .8
        self.assertEqual(report.validation_selection(rows)["selected_arm"], "baseline")
        self.assertEqual(report.validation_selection(rows[:-1])["status"], "incomplete")
        rows[-1]["status"] = "training"
        self.assertIsNone(report.validation_selection(rows)["selected_arm"])

    def test_paired_differences_match_the_same_seed_and_group(self):
        paired = report.paired_rows(self.rows())
        self.assertEqual(len(paired), 4)
        changes = [row for row in paired if row["comparison"] == "drop_path_gain"]
        self.assertEqual(len(changes), 2)
        for row in changes:
            self.assertEqual(row["group"], row["reference_group"])
            self.assertEqual(row["seed"], 42)
            self.assertEqual(row["weight_decay"], row["reference_weight_decay"])
            self.assertAlmostEqual(row["delta_roc_auc"], -.15)
            self.assertEqual(row["reference_drop_path_rate"], 0.)

    def test_diagnostics_compare_eval_focal_at_best_and_last_epochs(self):
        records = report.overfitting_rows(self.rows())
        selected = [row for row in records if row["arm"] == "drop_path_0p1"]
        self.assertEqual(len(selected), 2)
        for row in selected:
            self.assertEqual(row["best_epoch"], 2)
            self.assertEqual(row["last_epoch"], 3)
            self.assertAlmostEqual(row["best_loss_gap"], .03)
            self.assertAlmostEqual(row["last_loss_gap"], .07)
            self.assertEqual(row["best_val_pr_auc"], .45)

    def test_report_files_and_printing_preserve_source_artifacts(self):
        suite = Path(self.state["baseline"]["suite"])
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in suite.rglob('*') if p.is_file()}
        report.write_report(self.root / "report_study", self.state)
        with (self.root / "report_study/report/comparison.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)
        self.assertTrue((self.root / "report_study/report/overfitting.csv").is_file())
        with redirect_stdout(io.StringIO()) as output:
            report.print_metrics(self.root, self.state)
        self.assertIn("DropPath=0.1; weight_decay=0.01", output.getvalue())
        self.assertIn("seed SD are not estimable", output.getvalue())
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in suite.rglob('*') if p.is_file()})


if __name__ == "__main__":
    unittest.main()
