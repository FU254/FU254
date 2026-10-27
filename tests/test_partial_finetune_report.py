"""Partial-finetuning screening stays validation-only and plots comparable loss."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from foundation_sft import partial_finetune_report as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_state(root):
    root = Path(root)
    state = {"format_version": "partial_finetune_study_v1", "status": "completed",
        "baseline": {}, "candidates": {},
        "selection": {"status": "selected", "arm": "last4", "best_val_patient_roc_auc": .89},
        "selected_evaluation": {"status": "completed"}}
    histories = {}
    for index, arm in enumerate(report.ARMS):
        phase = "main" if arm == "baseline" else "pilot"
        run = root / "runs" / arm
        score = {"baseline": .8864, "last8": .88, "last4": .89, "last2": .87}[arm]
        history = [{"epoch": epoch, "train_eval_loss": .05 - epoch * .005,
            "val_loss": .08 - epoch * .001, "train_eval_patient_roc_auc": .9 + epoch * .02,
            "val_patient_roc_auc": auc, "train_eval_patient_pr_auc": .6 + epoch * .04,
            "val_patient_pr_auc": .4 + epoch * .02, "train_loss": 2 + epoch,
            "train_hard_loss": .03, "train_kd_loss": .06}
            for epoch, auc in ((1, score - .02), (2, score), (3, score - .01))]
        put(run / "history.json", history)
        histories[arm] = history
        count = {"baseline": 5_524_609, "last8": 3_559_489,
                 "last4": 1_780_033, "last2": 890_305}[arm]
        receipt = {"status": "completed", "phase": phase, "run": str(run),
            "student_trainable_parameters": count, "checkpoint_sha256": str(index + 1) * 64,
            "best_val_patient_roc_auc": score}
        metric_index = {"baseline": 0, "last8": 3, "last4": 1, "last2": 2}[arm]
        patient = {"roc_auc": .84 + .01 * metric_index, "pr_auc": .5 + .01 * metric_index,
            "average_precision": .51 + .01 * metric_index, "sensitivity": .6 + .01 * metric_index,
            "specificity": .8 - .01 * metric_index}
        if arm == "baseline":
            receipt.update(test_patient=patient, roc_auc_ci={"lower": .78, "upper": .9})
        entry = {"status": "reused" if arm == "baseline" else "completed",
            "job": {"config": {"seed": 42, "training": {"backbone_lr": 5e-5}}},
            "check": {"model_metadata": {"trainable_parameters": count}}, "result": receipt}
        if arm == "baseline":
            state["baseline"] = entry
        else:
            state["candidates"][arm] = entry
        if arm == "last4":
            final = deepcopy(receipt)
            final.update(phase="main", source_phase="pilot", test_patient=patient,
                roc_auc_ci={"lower": .79, "upper": .91})
            state["selected_evaluation"]["result"] = final
    return state, histories


class PartialFinetuneReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="partial-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state, self.histories = make_state(self.root)

    def test_validation_rows_have_parameter_counts_and_no_test_columns(self):
        rows = report.screening_rows(self.root, self.state)
        self.assertEqual([row["arm"] for row in rows], ["baseline", "last8", "last4", "last2"])
        self.assertEqual([row["train_last_blocks"] for row in rows], [12, 8, 4, 2])
        self.assertEqual([row["trainable_student_parameters"] for row in rows],
                         [5_524_609, 3_559_489, 1_780_033, 890_305])
        self.assertEqual(rows[2]["val_auc"], .89)
        self.assertEqual(rows[2]["val_pr_auc"], self.histories["last4"][1]["val_patient_pr_auc"])
        for row in rows:
            self.assertNotIn("roc_auc", row)
            self.assertNotIn("sensitivity", row)

    def test_only_baseline_and_selected_candidate_test_are_exposed_with_all_four_metrics(self):
        rows = report.test_rows(self.root, self.state)
        self.assertEqual([row["arm"] for row in rows], ["baseline", "last4"])
        self.assertFalse(rows[0]["selected_by_validation"])
        self.assertTrue(rows[1]["selected_by_validation"])
        self.assertEqual(rows[1]["roc_auc"], .85)
        self.assertEqual(rows[1]["pr_auc"], .51)
        self.assertEqual(rows[1]["ap"], .52)
        self.assertEqual(rows[1]["sensitivity"], .61)
        self.assertEqual(rows[1]["specificity"], .79)
        self.assertEqual(rows[1]["roc_ci_lower"], .79)
        state = deepcopy(self.state)
        state["selection"] = {"status": "selected", "arm": "baseline"}
        state["selected_evaluation"]["result"] = deepcopy(state["baseline"]["result"])
        rows = report.test_rows(self.root, state)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["selected_by_validation"])

    def test_incomplete_selection_or_candidate_does_not_read_unselected_test_files(self):
        state = deepcopy(self.state)
        state["selected_evaluation"] = {"status": "pending"}
        state["selection"] = {"status": "incomplete"}
        state["candidates"]["last2"]["status"] = "training"
        state["candidates"]["last2"]["result"]["run"] = "/missing/candidate/must/not/read"
        # An unrelated test file under the completed unselected candidate must
        # never become a source for the report or contaminate pilot receipts.
        put(Path(state["candidates"]["last4"]["result"]["run"]) / "test/metrics.json", {"patient": {"roc_auc": 1}})
        result = report.build_report(self.root, state)
        self.assertEqual([row["arm"] for row in result["test_rows"]], ["baseline"])
        self.assertIsNone(result["rows"][3]["val_auc"])
        self.assertEqual(len(result["overfitting"]), 3)

    def test_candidate_test_contamination_and_selected_checkpoint_mismatch_fail_closed(self):
        for key in ("test_patient", "roc_auc_ci"):
            state = deepcopy(self.state)
            state["candidates"]["last2"]["result"][key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "must not contain test"):
                report.build_report(self.root, state)
        for field, value in (("run", "/wrong/run"), ("checkpoint_sha256", "f" * 64),
                             ("best_val_patient_roc_auc", .99)):
            state = deepcopy(self.state)
            state["selected_evaluation"]["result"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.build_report(self.root, state)
        state = deepcopy(self.state)
        state["selection"] = {"status": "incomplete"}
        with self.assertRaisesRegex(ValueError, "requires frozen"):
            report.build_report(self.root, state)
        state = deepcopy(self.state)
        state["candidates"]["last4"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "requires a completed screening arm"):
            report.build_report(self.root, state)

    def test_gaps_use_same_focal_at_first_roc_best_and_comparable_epochs(self):
        result = report.build_report(self.root, self.state)
        gaps = result["overfitting"][2]
        history = self.histories["last4"]
        self.assertEqual((gaps["best_epoch"], gaps["last_epoch"]), (2, 3))
        self.assertAlmostEqual(gaps["best_loss_gap"], history[1]["val_loss"] - history[1]["train_eval_loss"])
        self.assertAlmostEqual(gaps["last_auc_gap"], history[-1]["train_eval_patient_roc_auc"] - history[-1]["val_patient_roc_auc"])
        self.assertNotEqual(gaps["best_train_eval_loss"], history[1]["train_loss"])
        self.assertEqual(len(result["matched_epochs"]), 9)
        first = result["matched_epochs"][0]
        self.assertEqual((first["arm"], first["epoch"]), ("last8", 1))
        self.assertAlmostEqual(first["delta_val_auc"], -.0064)

    def test_bad_history_or_parameter_counts_are_rejected(self):
        path = Path(self.state["candidates"]["last4"]["result"]["run"]) / "history.json"
        raw = path.read_bytes()
        for defect in ("missing_eval", "wrong_best", "bad_epoch", "nan"):
            history = deepcopy(self.histories["last4"])
            if defect == "missing_eval":
                history[0].pop("train_eval_loss")
            elif defect == "wrong_best":
                history[1]["val_patient_roc_auc"] = .99
            elif defect == "bad_epoch":
                history[1]["epoch"] = 1
            else:
                history[0]["val_loss"] = float('nan')
            put(path, history)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                report.build_report(self.root, self.state)
        path.write_bytes(raw)
        state = deepcopy(self.state)
        state["candidates"]["last4"]["check"]["model_metadata"]["trainable_parameters"] += 1
        with self.assertRaisesRegex(ValueError, "counts differ"):
            report.build_report(self.root, state)

    def test_reporting_does_not_reselect_from_test_and_saved_csvs_remain_separate(self):
        state = deepcopy(self.state)
        state["selected_evaluation"]["result"]["test_patient"]["roc_auc"] = .1
        selection = deepcopy(state["selection"])
        result = report.write_report(self.root, state)
        self.assertEqual(result["selection"], selection)
        for name in ("screening.csv", "test_comparison.csv", "overfitting.csv", "matched_epochs.csv", "selection.json"):
            self.assertTrue((self.root / 'report' / name).is_file())
        screening = (self.root / 'report/screening.csv').read_text()
        self.assertNotIn('sensitivity', screening)
        self.assertIn('sensitivity', (self.root / 'report/test_comparison.csv').read_text())
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_metrics(self.root, state)
        for expected in ('validation only', 'Single seed42', 'PR-AUC and AP are distinct', 'sensitivity', 'specificity'):
            self.assertIn(expected, output.getvalue())

    def test_plots_use_eval_loss_and_preserve_prior_figures(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('matplotlib required for plotting')
        observed = []
        original = plt.Axes.plot
        def capture(axis, x, y, *args, **kwargs):
            observed.append((list(x), list(y), kwargs.get('label')))
            return original(axis, x, y, *args, **kwargs)
        with mock.patch.object(plt.Axes, 'plot', capture):
            first = report.plot_curves(self.root, self.state)
        self.assertEqual(observed[0][1], [row['train_eval_loss'] for row in self.histories['baseline']])
        self.assertEqual(observed[1][1], [row['val_loss'] for row in self.histories['baseline']])
        self.assertEqual(observed[8][1], [row['train_eval_patient_roc_auc'] for row in self.histories['baseline']])
        before = Path(first['loss_curves']).read_bytes()
        state = deepcopy(self.state)
        state['candidates']['last2']['status'] = 'training'
        state['selected_evaluation'] = {'status':'pending'}
        for row in self.histories['last4']:
            row.pop('train_eval_patient_pr_auc')
            row.pop('val_patient_pr_auc')
        put(Path(state['candidates']['last4']['result']['run']) / 'history.json', self.histories['last4'])
        second = report.plot_curves(self.root, state)
        self.assertNotEqual(first['directory'], second['directory'])
        self.assertEqual(Path(first['loss_curves']).read_bytes(), before)
        for key in ('loss_curves','roc_auc_curves','pr_auc_curves','optimization_losses'):
            self.assertTrue(Path(second[key]).is_file())
        metadata = json.loads((Path(second['directory']) / 'plot_metadata.json').read_text())
        self.assertEqual(metadata['available_arms'], ['baseline','last8','last4'])
        self.assertEqual(metadata['missing_arms'], ['last2'])

    def test_import_is_standard_library_only(self):
        result = subprocess.run([sys.executable, '-S', '-c',
            "import sys; from foundation_sft import partial_finetune_report; assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
