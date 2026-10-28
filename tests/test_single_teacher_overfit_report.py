"""Exercise saved-history diagnostics without trained models or patient data."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from foundation_sft import single_teacher_overfit_report as report


def history_row(epoch, train_auc, validation_auc, train_loss, validation_loss):
    return {
        "epoch": epoch, "train_loss": 100. + epoch, "train_hard_loss": .2,
        "train_kd_loss": 100., "train_soft_loss": 30., "train_global_loss": 40.,
        "train_local_loss": 50., "train_eval_loss": train_loss,
        "train_eval_patient_roc_auc": train_auc, "train_eval_patient_pr_auc": train_auc - .1,
        "val_loss": validation_loss, "val_patient_roc_auc": validation_auc,
        "val_patient_pr_auc": validation_auc - .2, "learning_rates": [1e-4, 1e-3],
    }


def saved_study(root):
    state = {"format_version": "single_teacher_overfit_study_v1", "status": "completed",
             "plan": {"source_suite": "/read-only/source/suite", "jobs": []}, "jobs": {}}
    definitions = {
        "A": [(1, .70, .65, .4, .5), (2, .80, .70, .3, .6), (3, .90, .68, .2, .8)],
        "B": [(1, .80, .60, .3, .6), (2, .90, .80, .2, .5), (3, .98, .70, .1, .9)],
        "C": [(1, .75, .72, .4, .5), (2, .85, .76, .3, .5), (3, .95, .74, .2, .6)],
        "D": [(1, .73, .68, .5, .6), (2, .83, .75, .4, .5), (3, .88, .77, .3, .4)],
    }
    for group, values in definitions.items():
        name = group + "_seed42"
        run = root / name
        run.mkdir()
        history = [history_row(*values) for values in values]
        if group == "D":
            for item in history:
                item.update(train_l2_sp_loss=1000. * item["epoch"], train_l2_sp_penalty=.1 * item["epoch"])
        (run / "history.json").write_text(json.dumps(history))
        # These files must remain entirely unread by reports.
        (run / "best.pt").write_bytes(b"not-a-checkpoint")
        (run / "last.pt").write_bytes(b"not-a-checkpoint")
        cfg = {"student_initialization": "imagenet", "student": {"mode": "full_finetune"},
               "diagnostics": {"protocol": "single_teacher_overfit_v1", "evaluation": "validation_only"}}
        if group == "D":
            cfg["diagnostics"]["l2_sp"] = {"coefficient": 1e-4, "reduction": "sum"}
        job = {"id": name, "group": group, "condition": "supervised" if group == "A" else "distillation",
               "seed": 42, "config": cfg}
        best = max(history, key=lambda item: item["val_patient_roc_auc"])
        state["plan"]["jobs"].append(job)
        state["jobs"][name] = {"status": "completed", "receipt": {
            "run": str(run), "evaluation_phase": "validation_only", "test_evaluated": False,
            "best_epoch": best["epoch"], "last_epoch": 3,
            "best_val_patient_roc_auc": best["val_patient_roc_auc"],
            "validation_patient": {"roc_auc": best["val_patient_roc_auc"], "pr_auc": best["val_patient_pr_auc"],
                                   "average_precision": .4321, "sensitivity": .625, "specificity": .875},
            "best_train_patient": {"roc_auc": best["train_eval_patient_roc_auc"], "pr_auc": best["train_eval_patient_pr_auc"]},
            "best_train_eval_loss": best["train_eval_loss"], "best_val_loss": best["val_loss"],
            "test_patient": {"roc_auc": .99999999, "sensitivity": .99999999},
        }}
    return state


class SingleTeacherOverfitReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = saved_study(self.root)

    def test_best_and_final_gaps_use_clean_focal_instead_of_kd_training_objective(self):
        rows = {row["group"]: row for row in report.comparison_rows(self.state)}
        row = rows["B"]
        self.assertEqual((row["best_epoch"], row["last_epoch"]), (2, 3))
        self.assertAlmostEqual(row["best_auc_gap"], .1)
        self.assertAlmostEqual(row["last_auc_gap"], .28)
        self.assertAlmostEqual(row["best_loss_gap"], .3)
        self.assertAlmostEqual(row["last_loss_gap"], .8)
        self.assertAlmostEqual(row["last_minus_best_validation_patient_roc_auc"], -.1)
        self.assertAlmostEqual(row["best_train_eval_loss"], .2)
        self.assertEqual(rows["D"]["l2_sp_reduction"], "sum")
        self.assertEqual(rows["D"]["l2_sp_coefficient"], .0001)

    def test_shared_epoch_contrasts_include_all_controls_and_do_not_compare_different_best_epochs(self):
        rows = report.matched_epoch_rows(self.state)
        self.assertEqual(len(rows), 15)
        self.assertEqual({row["comparison"] for row in rows},
                         {"B_minus_A", "C_minus_A", "D_minus_A", "B_minus_C", "B_minus_D"})
        row = next(row for row in rows if row["comparison"] == "B_minus_D" and row["epoch"] == 2)
        self.assertEqual((row["candidate_best_epoch"], row["reference_best_epoch"]), (2, 3))
        self.assertAlmostEqual(row["delta_validation_patient_roc_auc"], .05)
        self.assertAlmostEqual(row["delta_train_eval_loss"], -.2)
        self.assertAlmostEqual(row["delta_loss_gap"], .2)
        # Remove epoch 2 from D: it cannot be substituted by D's best epoch 3.
        path = self.root / "D_seed42" / "history.json"
        history = json.loads(path.read_text())
        path.write_text(json.dumps([history[0], history[2]]))
        rows = report.matched_epoch_rows(self.state)
        self.assertFalse(any(row["comparison"] == "B_minus_D" and row["epoch"] == 2 for row in rows))

    def test_partial_and_failed_jobs_keep_status_and_only_use_available_history(self):
        failed = self.state["jobs"]["A_seed42"]
        failed.update(status="failed", error="worker stopped")
        failed.pop("receipt")
        training = self.state["jobs"]["C_seed42"]
        training.update(status="training", run=training.pop("receipt")["run"])
        rows = {row["group"]: row for row in report.comparison_rows(self.state)}
        self.assertEqual(rows["A"]["status"], "failed")
        self.assertIsNone(rows["A"]["best_validation_patient_roc_auc"])
        self.assertEqual(rows["C"]["status"], "training")
        self.assertEqual(rows["C"]["best_epoch"], 2)
        self.assertEqual(rows["C"]["best_epoch_basis"], "best observed validation ROC-AUC")
        self.assertFalse(any(row["reference"] == "A" for row in report.matched_epoch_rows(self.state)))
        paths = report.write_reports(self.root, self.state)
        self.assertIn("worker stopped", paths["markdown"].read_text())

    def test_nonfinite_and_missing_metrics_remain_missing_in_csv_and_differences(self):
        path = self.root / "B_seed42" / "history.json"
        history = json.loads(path.read_text())
        history[2]["val_patient_pr_auc"] = float("nan")
        history[2]["train_eval_loss"] = float("inf")
        path.write_text(json.dumps(history))
        self.state["jobs"]["B_seed42"]["receipt"]["validation_patient"]["sensitivity"] = True
        row = next(row for row in report.comparison_rows(self.state) if row["group"] == "B")
        self.assertIsNone(row["last_validation_patient_pr_auc"])
        self.assertIsNone(row["last_loss_gap"])
        self.assertIsNone(row["validation_patient_sensitivity"])
        matched = next(row for row in report.matched_epoch_rows(self.state)
                       if row["comparison"] == "B_minus_A" and row["epoch"] == 3)
        self.assertIsNone(matched["delta_validation_patient_pr_auc"])
        paths = report.write_reports(self.root, self.state)
        with paths["comparison"].open(newline="") as handle:
            saved = next(row for row in csv.DictReader(handle) if row["group"] == "B")
        self.assertEqual(saved["last_validation_patient_pr_auc"], "")
        self.assertEqual(saved["validation_patient_sensitivity"], "")

    def test_live_history_is_read_only_from_the_latest_declared_training_attempt(self):
        record = self.state["jobs"]["B_seed42"]
        record.pop("receipt")
        record["status"] = "training"
        record["job"] = self.state["plan"]["jobs"][1]
        old, latest = self.root / "old_attempt", self.root / "latest_attempt"
        record["attempts"] = [{"phase": "train", "directory": str(old)},
                              {"phase": "train", "directory": str(latest)}]
        old_run = old / "outputs/suites/suite_old/distillation"
        latest_run = latest / "outputs/suites/suite_live/distillation"
        old_run.mkdir(parents=True)
        latest_run.mkdir(parents=True)
        saved = (self.root / "B_seed42/history.json").read_text()
        (old_run / "history.json").write_text(saved)
        # Never fall back to an older attempt just because the current one is empty.
        row = report.comparison_rows(self.state)[1]
        self.assertIsNone(row["best_epoch"])
        (latest_run / "history.json").write_text(saved)
        row = report.comparison_rows(self.state)[1]
        self.assertEqual(row["best_epoch"], 2)
        self.assertEqual(row["run"], str(latest_run))
        ambiguous = latest / "outputs/suites/suite_other/distillation"
        ambiguous.mkdir(parents=True)
        (ambiguous / "history.json").write_text(saved)
        with self.assertRaisesRegex(ValueError, "Ambiguous histories"):
            report.comparison_rows(self.state)

    def test_print_is_read_only_and_explicitly_labels_validation_sensitivity_specificity(self):
        original = deepcopy(self.state)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with redirect_stdout(io.StringIO()) as output:
            report.print_metrics(self.state)
        text = output.getvalue()
        self.assertIn("validation_Se=0.625", text)
        self.assertIn("validation_Sp=0.875", text)
        self.assertIn("validation_AP=0.4321", text)
        self.assertNotIn("0.99999999", text)
        self.assertIn("Source suite: /read-only/source/suite", text)
        self.assertIn("no universal gap cutoff", text)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})
        self.assertEqual(self.state, original)

    def test_reports_keep_ap_distinct_and_never_export_test_metrics(self):
        paths = report.write_reports(self.root, self.state)
        self.assertEqual({path.name for path in paths.values() if path.is_file()},
                         {"comparison.csv", "matched_epoch.csv", "report.md"})
        with paths["comparison"].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[1]["validation_patient_ap"], "0.4321")
        self.assertAlmostEqual(float(rows[1]["best_validation_patient_pr_auc"]), .6)
        text = paths["markdown"].read_text()
        self.assertIn("Validation Se", text)
        self.assertIn("Validation Sp", text)
        self.assertIn("does not establish", text)
        self.assertNotIn("test_patient", paths["comparison"].read_text())
        self.assertNotIn("0.99999999", text)

    def test_basic_report_imports_and_printing_do_not_import_torch_or_matplotlib(self):
        state_path = self.root / "study.json"
        state_path.write_text(json.dumps(self.state))
        code = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'matplotlib'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from foundation_sft import single_teacher_overfit_report as report
state = json.load(open(sys.argv[1]))
report.print_metrics(state)
report.write_reports(sys.argv[2], state)
"""
        result = subprocess.run([sys.executable, "-c", code, str(state_path), str(self.root)],
                                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_receipts_and_duplicate_epochs_are_rejected_before_writing(self):
        self.state["jobs"]["A_seed42"]["receipt"]["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "validation-only"):
            report.write_reports(self.root, self.state)
        self.assertFalse((self.root / "report").exists())
        self.state["jobs"]["A_seed42"]["receipt"]["test_evaluated"] = False
        path = self.root / "B_seed42" / "history.json"
        history = json.loads(path.read_text())
        path.write_text(json.dumps(history + [history[0]]))
        with self.assertRaisesRegex(ValueError, "Duplicate epoch"):
            report.comparison_rows(self.state)

    def test_plot_axes_use_correct_metrics_and_keep_raw_l2_sp_separate(self):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
        except ImportError:
            self.skipTest("matplotlib is optional")
        figures = []
        original_save = Figure.savefig
        def save(figure, *args, **kwargs):
            figures.append(figure)
            return original_save(figure, *args, **kwargs)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with mock.patch.object(Figure, "savefig", save):
            paths = report.plot_curves(self.root, self.state)
        self.assertEqual(set(paths), {"directory", "focal_loss", "roc_auc", "pr_auc", "optimization", "l2_sp"})
        self.assertTrue(all(path.is_file() and path.stat().st_size > 1000
                            for name, path in paths.items() if name != "directory"))
        self.assertEqual([len(figure.axes) for figure in figures], [4, 4, 4, 4, 2])
        focal_b = figures[0].axes[1]
        train = next(line for line in focal_b.lines if line.get_label() == "Clean train (eval)")
        self.assertEqual(list(train.get_ydata()), [.3, .2, .1])
        optimization_b = figures[3].axes[1]
        self.assertNotIn("Validation", [line.get_label() for line in optimization_b.lines])
        self.assertIn("Weighted KD", [line.get_label() for line in optimization_b.lines])
        self.assertFalse(any("raw L2-SP" in line.get_label() for line in figures[3].axes[3].lines))
        self.assertEqual(list(figures[4].axes[0].lines[0].get_ydata()), [1000., 2000., 3000.])
        for value, expected in zip(figures[4].axes[1].lines[0].get_ydata(), [.1, .2, .3]):
            self.assertAlmostEqual(value, expected)
        # Missing jobs can be plotted, and a second plot preserves prior figures.
        self.state["jobs"]["A_seed42"] = {"status": "failed", "receipt": None, "error": "unavailable"}
        second = report.plot_curves(self.root, self.state)
        self.assertNotEqual(paths["directory"], second["directory"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
