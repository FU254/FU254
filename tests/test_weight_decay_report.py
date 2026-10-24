"""Weight-decay comparisons preserve matching and never select by test scores."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from foundation_sft import weight_decay_report as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class WeightDecayReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {"baseline": {"suite": str(self.root / "baseline"),
            "weight_decay": .0001, "jobs": {}}, "arms": {}}
        self.suites = {}
        for arm, decay, validation in (("baseline", .0001, .80), ("wd_0p001", .001, .85), ("wd_0p01", .01, .82)):
            jobs = {}
            for group in ("A", "B"):
                for seed in ((42, 43, 44) if arm == "baseline" else (42,)):
                    name = f"{group}_seed{seed}"
                    run = self.root / arm / name
                    auc = validation - (.1 if group == "A" else 0) + (seed - 42) * .001
                    history = [
                        {"epoch": 1, "train_loss": 999., "train_eval_loss": .08, "val_loss": .09,
                         "train_eval_patient_roc_auc": auc-.01, "val_patient_roc_auc": auc-.02,
                         "val_patient_pr_auc": .30},
                        {"epoch": 2, "train_loss": 998., "train_eval_loss": .06, "val_loss": .085,
                         "train_eval_patient_roc_auc": auc+.03, "val_patient_roc_auc": auc,
                         "val_patient_pr_auc": .35},
                        {"epoch": 3, "train_loss": 997., "train_eval_loss": .05, "val_loss": .10,
                         "train_eval_patient_roc_auc": auc+.05, "val_patient_roc_auc": auc-.01,
                         "val_patient_pr_auc": .32},
                    ]
                    put(run / "history.json", history)
                    job = {"id": name, "group": group, "seed": seed,
                        "student_mode": "full_finetune", "condition": "supervised" if group == "A" else "distillation",
                        "config": {"distillation": {"soft_weight": .5, "global_weight": .1, "local_weight": 1.}}}
                    test_auc = .70 + (0 if arm == "baseline" else .01) + (seed-42)*.001
                    result = {"run": str(run), "best_val_patient_roc_auc": auc,
                        "test_patient": {"roc_auc": test_auc, "pr_auc": .4, "average_precision": .41,
                                         "sensitivity": .6, "specificity": .8}}
                    jobs[name] = {"job": job, "check": {}, "result": result, "status": "completed"}
            if arm == "baseline":
                self.state["baseline"]["jobs"] = deepcopy(jobs)
            else:
                suite = self.root / "arms" / arm
                put(suite / "suite.json", {"jobs": jobs})
                self.state["arms"][arm] = {"suite": str(suite), "weight_decay": decay}
                self.suites[arm] = {"jobs": jobs}

    def rows(self):
        return report.metric_rows(self.root, self.state)

    def test_each_decay_and_group_uses_only_the_confirmed_single_seed(self):
        rows = self.rows()
        self.assertEqual(len(rows), 6)
        summaries = report.summary_rows(rows)
        self.assertEqual(len(summaries), 6)
        self.assertEqual({(r["arm"], r["group"]) for r in summaries},
                         {(arm, group) for arm in ("baseline", "wd_0p001", "wd_0p01") for group in "AB"})
        self.assertTrue(all(r["n_completed"] == 1 and r["roc_auc_sd"] is None for r in summaries))
        self.assertEqual({r["seed"] for r in rows}, {42})
        self.assertTrue(all(r["initialization"] == "random" for r in rows))
        self.assertTrue(all(r["ap"] == .41 and r["pr_auc"] == .4 for r in rows))
        self.assertTrue(all(r["val_pr_auc"] == .35 for r in rows))
        self.assertTrue(all(abs(r["val_pr_auc_mean"]-.35) < 1e-12 for r in summaries))

    def test_validation_selection_ignores_all_test_metrics(self):
        rows = self.rows()
        for row in rows:
            row["roc_auc"] = 1. if row["arm"] == "wd_0p01" else 0.
            row["pr_auc"] = row["roc_auc"]
        selected = report.validation_selection(rows)
        self.assertEqual(selected["selected_arm"], "wd_0p001")
        self.assertFalse(selected["test_metrics_used"])
        self.assertEqual(selected["status"], "complete")

    def test_missing_or_failed_seed_prevents_parameter_recommendation(self):
        rows = self.rows()
        for defective in (rows[:-1], [{**r, "status": "failed"} if i == 0 else r for i, r in enumerate(rows)]):
            selected = report.validation_selection(defective)
            self.assertEqual(selected["status"], "incomplete")
            self.assertIsNone(selected["selected_arm"])

    def test_ties_prefer_baseline_then_smaller_decay(self):
        rows = self.rows()
        for row in rows:
            row["val_auc"] = .8
        self.assertEqual(report.validation_selection(rows)["selected_arm"], "baseline")
        for row in rows:
            if row["arm"] != "baseline":
                row["val_auc"] = .9
        self.assertEqual(report.validation_selection(list(reversed(rows)))["selected_arm"], "wd_0p001")

    def test_only_matching_seeds_are_subtracted(self):
        rows = self.rows()
        pairs = report.paired_rows(rows)
        self.assertEqual(len(pairs), 7)
        selected = [r for r in pairs if r["comparison"] == "weight_decay_gain"]
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(abs(r["delta_roc_auc"]-.01) < 1e-12 for r in selected))
        fewer = [r for r in rows if not (r["arm"] == "baseline" and r["group"] == "A" and r["seed"] == 42)]
        pairs = report.paired_rows(fewer)
        self.assertFalse(any(r["reference"] == "baseline" and r["reference_group"] == "A"
                             and r["seed"] == 42 for r in pairs))

    def test_diagnostics_use_eval_focal_and_saved_best_epoch(self):
        records = report.overfitting_rows(self.rows())
        self.assertEqual(len(records), 6)
        for row in records:
            self.assertEqual(row["best_epoch"], 2)
            self.assertEqual(row["last_epoch"], 3)
            self.assertAlmostEqual(row["best_loss_gap"], .025)
            self.assertAlmostEqual(row["last_loss_gap"], .05)
            self.assertAlmostEqual(row["best_auc_gap"], .03)
            self.assertAlmostEqual(row["last_auc_gap"], .06)
            self.assertEqual(row["best_val_pr_auc"], .35)
            self.assertEqual(row["last_val_pr_auc"], .32)

    def test_incompatible_or_missing_history_cannot_silently_produce_a_diagnosis(self):
        rows = self.rows()
        corrupt = deepcopy(rows)
        corrupt[0]["val_auc"] = .1
        with self.assertRaisesRegex(ValueError, "selection differs"):
            report.overfitting_rows(corrupt)
        run = Path(rows[0]["run"])
        history = json.loads((run / "history.json").read_text())
        del history[0]["train_eval_loss"]
        put(run / "history.json", history)
        with self.assertRaisesRegex(ValueError, "train_eval_loss"):
            report.overfitting_rows(rows)

    def test_report_writes_comparisons_and_keeps_baseline_history_unchanged(self):
        baseline_histories = list((self.root / "baseline").glob("*/history.json"))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_histories}
        result = report.write_report(self.root / "study", self.state)
        self.assertEqual(result["selection"]["selected_arm"], "wd_0p001")
        with (self.root / "study/report/summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)
        with (self.root / "study/report/overfitting.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)
        self.assertTrue((self.root / "study/report/validation_selection.json").is_file())
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_histories})
        with redirect_stdout(io.StringIO()) as output:
            report.print_metrics(self.root / "study", self.state)
        self.assertIn("Validation-only selected setting: wd_0p001", output.getvalue())
        self.assertIn("actual weight_decay=0.0001", output.getvalue())


if __name__ == "__main__":
    unittest.main()
