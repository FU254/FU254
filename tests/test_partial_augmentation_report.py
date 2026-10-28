"""The shared partial-finetuning report supports a validation-only jitter arm."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from foundation_sft import partial_finetune_report as report
from tests.test_partial_finetune_report import make_state, put


def make_augmentation_state(root):
    old, histories = make_state(root)
    baseline = deepcopy(old["candidates"]["last8"])
    baseline["status"] = "reused"
    baseline["job"]["id"] = "baseline"
    baseline["job"]["config"]["student"] = {"mode": "partial_finetune", "train_last_blocks": 8}
    baseline["result"].update(phase="main", test_patient=deepcopy(old["baseline"]["result"]["test_patient"]),
                              roc_auc_ci={"lower": .78, "upper": .9})
    augmented = deepcopy(old["candidates"]["last4"])
    augmented["job"]["id"] = "augmented"
    augmented["job"]["config"]["student"] = {"mode": "partial_finetune", "train_last_blocks": 8}
    augmented["job"]["config"]["training"].update(brightness_jitter=.1, contrast_jitter=.1)
    augmented["check"]["model_metadata"]["trainable_parameters"] = 3_559_489
    augmented["result"]["student_trainable_parameters"] = 3_559_489
    state = {"format_version": "partial_finetune_study_v1", "status": "completed",
        "plan": {"study_type": "intensity_augmentation", "evaluation_policy": "validation_only",
            "jobs": [deepcopy(augmented["job"])]},
        "baseline": baseline, "candidates": {"augmented": augmented},
        "selection": {"status": "selected", "arm": "augmented", "best_val_patient_roc_auc": .89},
        "selected_evaluation": {"status": "not_requested"}}
    return state, {"baseline": histories["last8"], "augmented": histories["last4"]}


class PartialAugmentationReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="partial-augmentation-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state, self.histories = make_augmentation_state(self.root)

    def test_rows_show_two_last8_arms_and_their_jitter_strengths(self):
        rows = report.screening_rows(self.root, self.state)
        self.assertEqual([row["arm"] for row in rows], ["baseline", "augmented"])
        self.assertEqual([row["student_update"] for row in rows],
            ["Last8 existing", "Last8 + shared intensity jitter"])
        self.assertEqual([row["train_last_blocks"] for row in rows], [8, 8])
        self.assertEqual([row["trainable_student_parameters"] for row in rows], [3_559_489] * 2)
        for key in ("brightness_jitter", "contrast_jitter"):
            self.assertEqual([row[key] for row in rows], [0.0, .1])
        self.assertEqual([row["val_auc"] for row in rows], [.88, .89])
        self.assertEqual(report.ARMS, ("baseline", "last8", "last4", "last2"))
        for row in rows:
            self.assertNotIn("roc_auc", row)
            self.assertNotIn("sensitivity", row)

    def test_only_historical_baseline_test_metrics_are_reported(self):
        # Stray evaluation files must never promote the augmented arm to test reporting.
        run = Path(self.state["candidates"]["augmented"]["result"]["run"])
        put(run / "test/metrics.json", {"patient": {"roc_auc": 1.0}})
        rows = report.test_rows(self.root, self.state)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["arm"], "baseline")
        self.assertEqual(rows[0]["student_update"], "Last8 existing")
        self.assertEqual(rows[0]["source"], "historical baseline")
        self.assertFalse(rows[0]["selected_by_validation"])
        self.assertEqual(rows[0]["roc_auc"], .84)
        state = deepcopy(self.state)
        state["selection"]["arm"] = "baseline"
        self.assertTrue(report.test_rows(self.root, state)[0]["selected_by_validation"])

    def test_test_contamination_and_new_selected_evaluation_are_rejected(self):
        for key in ("test_patient", "roc_auc_ci"):
            state = deepcopy(self.state)
            state["candidates"]["augmented"]["result"][key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "must not contain test"):
                report.build_report(self.root, state)
        for evaluation in ({"status": "completed"},
                           {"status": "not_requested", "result": deepcopy(self.state["baseline"]["result"])}):
            state = deepcopy(self.state)
            state["selected_evaluation"] = evaluation
            with self.subTest(evaluation=evaluation), self.assertRaisesRegex(ValueError, "validation-only"):
                report.build_report(self.root, state)
        state = deepcopy(self.state)
        state["selection"]["arm"] = "last4"
        with self.assertRaisesRegex(ValueError, "Unknown validation-selected"):
            report.build_report(self.root, state)

    def test_completed_arms_must_retain_equal_last8_parameter_counts(self):
        for arm in ("baseline", "augmented"):
            state = deepcopy(self.state)
            entry = state["baseline"] if arm == "baseline" else state["candidates"][arm]
            entry["result"]["student_trainable_parameters"] = 1_780_033
            entry["check"]["model_metadata"]["trainable_parameters"] = 1_780_033
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "3559489"):
                report.build_report(self.root, state)

    def test_same_focal_gaps_and_matched_epochs_cover_both_arms(self):
        result = report.build_report(self.root, self.state)
        self.assertEqual([row["arm"] for row in result["overfitting"]], ["baseline", "augmented"])
        gaps = result["overfitting"][1]
        history = self.histories["augmented"]
        self.assertEqual((gaps["best_epoch"], gaps["last_epoch"]), (2, 3))
        self.assertAlmostEqual(gaps["best_loss_gap"], history[1]["val_loss"] - history[1]["train_eval_loss"])
        self.assertAlmostEqual(gaps["last_auc_gap"], history[-1]["train_eval_patient_roc_auc"] - history[-1]["val_patient_roc_auc"])
        self.assertEqual(len(result["matched_epochs"]), 3)
        for row, candidate, baseline in zip(result["matched_epochs"], history, self.histories["baseline"]):
            self.assertEqual((row["arm"], row["reference"], row["epoch"]), ("augmented", "baseline", candidate["epoch"]))
            self.assertAlmostEqual(row["delta_val_auc"], candidate["val_patient_roc_auc"] - baseline["val_patient_roc_auc"])

    def test_pending_candidate_does_not_load_a_history_or_test_result(self):
        state = deepcopy(self.state)
        state["candidates"]["augmented"].update(status="training", result={"run": "/missing/never/read"})
        state["selection"] = {"status": "incomplete"}
        state["selected_evaluation"] = {"status": "pending"}
        result = report.build_report(self.root, state)
        self.assertEqual(result["rows"][1]["status"], "training")
        self.assertIsNone(result["rows"][1]["val_auc"])
        self.assertEqual([row["arm"] for row in result["test_rows"]], ["baseline"])
        self.assertEqual(len(result["overfitting"]), 1)
        self.assertEqual(result["matched_epochs"], [])

    def test_written_report_and_console_explain_validation_only_selection(self):
        selection = deepcopy(self.state["selection"])
        result = report.write_report(self.root, self.state)
        self.assertEqual(result["selection"], selection)
        self.assertIn("exact ties retain the existing baseline", result["selection_rule"])
        self.assertIn("no new test evaluation or test-based winner selection or promotion", result["selection_rule"])
        saved = json.loads((self.root / "report/selection.json").read_text())
        self.assertEqual(saved["rule"], result["selection_rule"])
        self.assertFalse(saved["test_metrics_used"])
        screening = (self.root / "report/screening.csv").read_text()
        self.assertIn("brightness_jitter", screening)
        self.assertNotIn("sensitivity", screening)
        test_csv = (self.root / "report/test_comparison.csv").read_text()
        self.assertNotIn("augmented", test_csv)
        self.assertIn("historical baseline", test_csv)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_metrics(self.root, self.state)
        for expected in ("Intensity augmentation study", "historical baseline only", "no new test evaluation",
                         "test-based winner selection or promotion", "Single seed42"):
            self.assertIn(expected, output.getvalue())

    def test_curves_use_two_panels_and_comparable_eval_losses(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib required for plotting")
        observed, panel_counts, titles = [], [], []
        original_plot, original_subplots, original_title = plt.Axes.plot, plt.subplots, plt.Axes.set_title

        def capture_plot(axis, x, y, *args, **kwargs):
            observed.append((list(x), list(y), kwargs.get("label")))
            return original_plot(axis, x, y, *args, **kwargs)

        def capture_panels(*args, **kwargs):
            panel_counts.append(args[:2])
            return original_subplots(*args, **kwargs)

        def capture_title(axis, title, *args, **kwargs):
            titles.append(title)
            return original_title(axis, title, *args, **kwargs)

        with mock.patch.object(plt.Axes, "plot", capture_plot), mock.patch.object(plt, "subplots", capture_panels), \
             mock.patch.object(plt.Axes, "set_title", capture_title):
            paths = report.plot_curves(self.root, self.state)
        self.assertEqual(panel_counts, [(1, 2)] * 4)
        self.assertEqual(titles, ["Last8 existing", "Last8 + shared intensity jitter"] * 4)
        self.assertEqual(observed[0][1], [row["train_eval_loss"] for row in self.histories["baseline"]])
        self.assertEqual(observed[2][1], [row["train_eval_loss"] for row in self.histories["augmented"]])
        self.assertEqual(observed[4][1], [row["train_eval_patient_roc_auc"] for row in self.histories["baseline"]])
        for key in ("loss_curves", "roc_auc_curves", "pr_auc_curves", "optimization_losses"):
            self.assertTrue(Path(paths[key]).is_file())
        metadata = json.loads((Path(paths["directory"]) / "plot_metadata.json").read_text())
        self.assertEqual(metadata["available_arms"], ["baseline", "augmented"])
        self.assertEqual(metadata["missing_arms"], [])
        self.assertFalse(metadata["test_curves"])


if __name__ == "__main__":
    unittest.main()
