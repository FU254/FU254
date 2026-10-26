"""Validation-only pilot tables and same-definition six-group learning curves."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from foundation_sft import teacher_weight_report as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_state(root):
    root = Path(root)
    state = {"format_version": "teacher_weight_study_v1", "status": "completed", "pilots": {}, "main": {},
        "selection": {initialization: {"status": "selected", "job_id": initialization + "_1", "backbone_lr": 1e-4}
                      for initialization in ("random", "imagenet")}}
    histories = {}
    for index, group in enumerate(report.GROUPS):
        initialization = "random" if group.startswith("R-") else "imagenet"
        method = {"E": "equal", "C": "ca", "R": "rl"}[group[-1]]
        job = {"id": group, "group": group, "student_initialization": initialization, "method": method,
            "phase": "main", "seed": 42, "config": {"seed": 42, "student_initialization": initialization,
                "method": method, "phase": "main", "training": {"backbone_lr": 1e-4}}}
        run = root / "runs" / group
        history = [{"epoch": epoch, "train_eval_loss": .12 - epoch * .01,
            "val_loss": .15 - epoch * .005, "train_eval_patient_roc_auc": .7 + epoch * .02,
            "val_patient_roc_auc": auc + index * .005, "train_eval_patient_pr_auc": .3 + epoch * .02,
            "val_patient_pr_auc": .25 + epoch * .01, "train_loss": 5.0 + epoch,
            "train_hard_loss": .1, "train_kd_loss": .2, "train_soft_loss": .02,
            "train_global_loss": .4, "train_local_loss": .01,
            "teacher_weights": dict(zip(report.TEACHERS, (.4, .35, .25))), "weight_entropy": 1.,
            "reward": None if epoch == 1 or method != "rl" else -.3 + epoch * .01,
            "policy_loss": None if epoch == 1 or method != "rl" else -.02 + epoch * .001}
            for epoch, auc in ((1, .72), (2, .76), (3, .74))]
        put(run / "history.json", history)
        histories[group] = history
        result = {"status": "completed", "phase": "main", "run": str(run),
            "best_val_patient_roc_auc": history[1]["val_patient_roc_auc"],
            "test_patient": {"roc_auc": .77 + index * .005, "pr_auc": .4 + index * .005,
                "average_precision": .41 + index * .005, "sensitivity": .6, "specificity": .8},
            "roc_auc_ci": {"lower": .7, "upper": .85}}
        state["main"][group] = {"job": job, "status": "completed", "result": result}
    for initialization in ("random", "imagenet"):
        for index, lr in enumerate((1e-5, 1e-4, 3e-4)):
            identifier = initialization + "_" + str(index)
            run = root / "pilots" / identifier
            history = deepcopy(histories["R-E"])
            put(run / "history.json", history)
            state["pilots"][identifier] = {"status": "completed",
                "job": {"id": identifier, "group": "R-E" if initialization == "random" else "P-E",
                    "student_initialization": initialization, "method": "equal", "seed": 42,
                    "phase": "pilot", "config": {"training": {"backbone_lr": lr}}},
                "result": {"status": "completed", "phase": "pilot", "run": str(run),
                    "best_val_patient_roc_auc": history[1]["val_patient_roc_auc"]}}
    return state, histories


class TeacherWeightReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="teacher-weight-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state, self.histories = make_state(self.root)

    def test_formal_groups_order_metrics_and_pr_auc_ap_are_distinct(self):
        rows = report.metric_rows(self.root, self.state)
        self.assertEqual([row["group"] for row in rows], list(report.GROUPS))
        self.assertEqual(rows[0]["val_auc"], .76)
        self.assertEqual(rows[0]["val_pr_auc"], .27)
        self.assertEqual(rows[0]["pr_auc"], .4)
        self.assertEqual(rows[0]["ap"], .41)
        self.assertEqual((rows[0]["roc_ci_lower"], rows[0]["roc_ci_upper"]), (.7, .85))

    def test_pilot_rows_expose_no_test_columns_and_refuse_test_receipt(self):
        rows = report.pilot_rows(self.root, self.state)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertNotIn("roc_auc", row)
            self.assertNotIn("pr_auc", row)
            self.assertNotIn("test_patient", row)
        for key, value in (("test_patient", {"roc_auc": .99}), ("roc_auc_ci", [0.9, 1.])):
            state = deepcopy(self.state)
            next(iter(state["pilots"].values()))["result"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pilot must not contain test"):
                report.pilot_rows(self.root, state)

    def test_same_definition_gaps_use_first_best_epoch_and_final_separately(self):
        result = report.build_report(self.root, self.state)
        first = result["overfitting"][0]
        history = self.histories["R-E"]
        self.assertEqual((first["best_epoch"], first["last_epoch"]), (2, 3))
        self.assertAlmostEqual(first["best_loss_gap"], history[1]["val_loss"] - history[1]["train_eval_loss"])
        self.assertAlmostEqual(first["last_auc_gap"], history[2]["train_eval_patient_roc_auc"] - history[2]["val_patient_roc_auc"])
        self.assertNotEqual(first["best_train_eval_loss"], history[1]["train_loss"])
        self.assertEqual(len(result["paired"]), 9)
        recipe = [row for row in result["paired"] if row["comparison"] == "imagenet_recipe_minus_random_recipe"]
        self.assertEqual(len(recipe), 3)

    def test_incomplete_groups_have_unknown_metrics_and_do_not_read_old_result(self):
        state = deepcopy(self.state)
        state["main"]["R-R"]["status"] = "failed"
        state["main"]["R-R"]["error"] = "CUDA fixture failure"
        state["main"]["R-R"]["result"]["run"] = "/does/not/exist"
        del state["main"]["P-C"]
        rows = report.metric_rows(self.root, state)
        self.assertIsNone(rows[2]["roc_auc"])
        self.assertEqual(rows[2]["error"], "CUDA fixture failure")
        self.assertEqual(rows[4]["status"], "pending")
        self.assertIsNone(rows[4]["val_auc"])
        self.assertEqual(len(report.overfitting_rows(rows)), 4)

    def test_missing_history_invalid_fields_or_changed_best_are_refused(self):
        run = Path(self.state["main"]["R-E"]["result"]["run"])
        path = run / "history.json"
        original = path.read_bytes()
        for defect in ("missing_train_eval", "bad_epoch", "nan", "range", "changed_best"):
            history = deepcopy(self.histories["R-E"])
            if defect == "missing_train_eval":
                history[0].pop("train_eval_loss")
            elif defect == "bad_epoch":
                history[1]["epoch"] = 1
            elif defect == "nan":
                history[0]["val_loss"] = float("nan")
            elif defect == "range":
                history[0]["val_patient_roc_auc"] = 1.2
            else:
                history[1]["val_patient_roc_auc"] = .75
            put(path, history)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                report.build_report(self.root, self.state)
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            report.build_report(self.root, self.state)

    def test_report_files_are_saved_without_reselecting_by_test(self):
        state = deepcopy(self.state)
        before = deepcopy(state["selection"])
        state["main"]["R-C"]["result"]["test_patient"]["roc_auc"] = 1.
        result = report.write_report(self.root, state)
        self.assertEqual(result["selection"], before)
        expected = ("lr_pilots.csv", "comparison.csv", "overfitting.csv", "matched_differences.csv", "learning_rate_selection.json")
        self.assertTrue(all((self.root / "report" / name).is_file() for name in expected))
        pilot_csv = (self.root / "report/lr_pilots.csv").read_text()
        self.assertNotIn("test_patient", pilot_csv)
        self.assertNotIn("roc_auc,", pilot_csv)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_metrics(self.root, state)
        self.assertIn("no test scores", output.getvalue())
        self.assertIn("independently validation-selected learning-rate recipes", output.getvalue())
        self.assertIn("PR-AUC and AP are distinct", output.getvalue())

    def test_plot_comparable_fields_missing_diagnostics_and_no_overwrite(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib required for rendering")
        captured = []
        original = plt.Axes.plot
        def capture(axis, x, y, *args, **kwargs):
            captured.append((list(x), list(y), kwargs.get("label")))
            return original(axis, x, y, *args, **kwargs)
        with patch.object(plt.Axes, "plot", capture):
            files = report.plot_curves(self.root, self.state)
        self.assertEqual(captured[0][1], [h["train_eval_loss"] for h in self.histories["R-E"]])
        self.assertEqual(captured[1][1], [h["val_loss"] for h in self.histories["R-E"]])
        self.assertEqual(captured[12][1], [h["train_eval_patient_roc_auc"] for h in self.histories["R-E"]])
        rewards = [line for line in captured if line[2] == "reward"]
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(str(line[1][0]) == "nan" for line in rewards))
        for name in ("loss_curves", "roc_auc_curves", "pr_auc_curves", "optimization_losses",
                     "teacher_weight_curves", "weight_entropy_curves", "rl_diagnostics"):
            self.assertTrue(Path(files[name]).is_file())
        first = Path(files["loss_curves"]).read_bytes()
        state = deepcopy(self.state)
        for group in report.GROUPS[1:]:
            state["main"][group]["status"] = "training"
        run = Path(state["main"]["R-E"]["result"]["run"])
        history = deepcopy(self.histories["R-E"])
        for row in history:
            for key in ("teacher_weights", "weight_entropy", "reward", "policy_loss",
                        "train_eval_patient_pr_auc", "val_patient_pr_auc"):
                row.pop(key, None)
        put(run / "history.json", history)
        second = report.plot_curves(self.root, state)
        self.assertNotEqual(files["directory"], second["directory"])
        self.assertEqual(Path(files["loss_curves"]).read_bytes(), first)
        metadata = json.loads((Path(second["directory"]) / "plot_metadata.json").read_text())
        self.assertEqual(metadata["available_groups"], ["R-E"])
        self.assertEqual(len(metadata["missing_groups"]), 5)

    def test_plot_without_completed_main_fails_before_making_output(self):
        state = deepcopy(self.state)
        state["main"] = {}
        with self.assertRaisesRegex(ValueError, "No completed formal group"):
            report.plot_curves(self.root, state)
        self.assertFalse((self.root / "report").exists())

    def test_import_and_printing_use_only_standard_library(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import sys; from foundation_sft import teacher_weight_report; assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
