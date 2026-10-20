"""Pair the same seeds, retaining missing results and the direction of effects."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from foundation_sft import distillation_ablation_report as report


def record(group, seed, auc, pr=None, status="completed"):
    return {"job": {"group": group, "seed": seed, "student_mode": "full_finetune", "condition": "supervised" if group == "supervised" else "distillation",
                    "config": {"distillation": {"soft_weight": .5, "global_weight": .1, "local_weight": 1.0}}},
            "status": status, "result": {"best_val_patient_roc_auc": auc,
            "cohort_hash": "fixed", "initial_state_sha256": f"initial-{seed}",
            "test_patient": {"roc_auc": auc, "pr_auc": pr, "average_precision": pr}},
            "check": {"teacher": {"checkpoint_sha256": "fixed-teacher"}}}


class AblationPairedReportTests(unittest.TestCase):
    def state(self):
        return {"jobs": {"s42": record("supervised", 42, .7, .3),
                         "s43": record("supervised", 43, .6, .2),
                         "s44": record("supervised", 44, .8, .4),
                         "k42": record("full_kd", 42, .8, .5),
                         "k43": record("full_kd", 43, .9, .6),
                         "k44": record("full_kd", 44, .99, .99, "failed"),
                         "soft42": record("soft_only", 42, .75, .4),
                         "soft44": record("soft_only", 44, .85, .5),
                         "ng42": record("no_global", 42, .82, .51)}}

    def test_same_seed_deltas_do_not_subtract_unmatched_group_means(self):
        summaries = {row["contrast"]: row for row in report.paired_summaries(self.state())}
        full = summaries["all_kd_gain"]
        self.assertEqual((full["n_pairs"], full["n_requested"], full["seeds"]), (2, 3, "42,43"))
        self.assertAlmostEqual(full["delta_roc_auc_mean"], .2)
        self.assertAlmostEqual(full["delta_roc_auc_sd"], 2**.5 * .1)
        beyond = summaries["beyond_soft_kd"]
        self.assertEqual(beyond["n_pairs"], 1)
        self.assertAlmostEqual(beyond["delta_roc_auc_mean"], .05)
        self.assertIsNone(beyond["delta_roc_auc_sd"])
        self.assertLess(summaries["global_contribution"]["delta_roc_auc_mean"], 0)

    def test_missing_metrics_and_failed_pairs_are_not_zero_imputed(self):
        state = self.state()
        state["jobs"]["k42"]["result"]["test_patient"]["pr_auc"] = None
        full = next(row for row in report.paired_summaries(state) if row["contrast"] == "all_kd_gain")
        self.assertEqual(full["n_pairs"], 2)
        self.assertEqual(full["delta_pr_auc_n"], 1)
        self.assertAlmostEqual(full["delta_pr_auc_mean"], .4)
        self.assertIsNone(full["delta_pr_auc_sd"])
        state["jobs"]["k42"]["status"] = state["jobs"]["k43"]["status"] = "failed"
        full = next(row for row in report.paired_summaries(state) if row["contrast"] == "all_kd_gain")
        self.assertEqual(full["n_pairs"], 0)
        self.assertIsNone(full["delta_roc_auc_mean"])

    def test_pairing_requires_the_same_cohort_initial_state_and_teacher(self):
        for field in ("cohort_hash", "initial_state_sha256", "teacher"):
            state = self.state()
            if field == "teacher":
                state["jobs"]["k42"]["check"]["teacher"]["checkpoint_sha256"] = "different"
            else:
                state["jobs"]["k42"]["result"][field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Cannot pair"):
                report.paired_differences(state)

    def test_duplicate_seed_group_is_rejected(self):
        state = self.state()
        state["jobs"]["duplicate"] = deepcopy(state["jobs"]["k42"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.paired_differences(state)

    def test_csv_records_and_printing_are_file_only_and_keep_state_unchanged(self):
        state, original = self.state(), deepcopy(self.state())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report.write_paired_reports(root, state)
            self.assertTrue((root / "report" / "paired_differences.csv").is_file())
            self.assertIn("all_kd_gain", (root / "report" / "paired_summary.csv").read_text())
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with redirect_stdout(io.StringIO()) as output:
                report.print_paired_metrics(state)
            self.assertIn("n=2/3", output.getvalue())
            self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})
        self.assertEqual(state, original)

    def test_multiple_batches_label_their_own_paired_comparisons(self):
        from foundation_sft.distillation_ablation import main

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            config = project / "config.json"
            config.write_text(json.dumps({"output_root": "runs/ablation"}))
            for name in ("suite_first", "suite_second"):
                path = project / "runs" / "ablation" / "suites" / name
                path.mkdir(parents=True)
                (path / "suite.json").write_text(json.dumps(self.state()))
            before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--project", str(project), "--config", str(config), "--show-metrics"]), 0)
            self.assertEqual(output.getvalue().count("Paired batch:"), 2)
            self.assertIn("Paired batch: " + str(project / "runs/ablation/suites/suite_first"), output.getvalue())
            self.assertIn("Paired batch: " + str(project / "runs/ablation/suites/suite_second"), output.getvalue())
            self.assertEqual(before, {p: p.read_bytes() for p in project.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
