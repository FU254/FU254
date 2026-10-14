"""Saved training curves use real epoch records and preserve existing artifacts."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.training_history import main, save_history_plots


class TrainingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="training-history-test-")
        self.root = Path(self.temporary.name)
        self.history = [
            {"epoch": 1, "train_loss": 0.9, "val_loss": 0.85,
             "val_patient_roc_auc": 0.6, "val_patient_pr_auc": 0.55},
            {"epoch": 2, "train_loss": 0.6, "val_loss": 0.7,
             "val_patient_roc_auc": 0.8, "val_patient_pr_auc": 0.77},
            {"epoch": 3, "train_loss": 0.35, "val_loss": 1.1,
             "val_patient_roc_auc": 0.72, "val_patient_pr_auc": 0.66},
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def assert_png(self, path):
        path = Path(path)
        self.assertGreater(path.stat().st_size, 1000)
        self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        pixels = mpimg.imread(path)
        self.assertGreater(pixels.shape[0], 500)
        self.assertGreater(pixels.shape[1], 800)
        self.assertGreater(float(pixels.std()), 0.01)

    def test_real_pngs_use_unsmoothed_recorded_values_and_validation_best_epoch(self):
        captured = []
        original_save = Figure.savefig

        def capture(figure, *args, **kwargs):
            captured.append({
                "lines": {line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                          for line in figure.axes[0].lines},
                "annotations": [text.get_text() for text in figure.axes[0].texts],
            })
            return original_save(figure, *args, **kwargs)

        before = deepcopy(self.history)
        with patch.object(Figure, "savefig", capture):
            result = save_history_plots(self.history, self.root / "curves")
        self.assertEqual(self.history, before)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(captured[0]["lines"]["Train loss"], ([1, 2, 3], [0.9, 0.6, 0.35]))
        self.assertEqual(captured[0]["lines"]["Validation loss"], ([1, 2, 3], [0.85, 0.7, 1.1]))
        self.assertEqual(captured[1]["lines"]["Validation patient ROC-AUC"],
                         ([1, 2, 3], [0.6, 0.8, 0.72]))
        self.assertIn("Best validation ROC: epoch 2", captured[1]["annotations"])
        self.assertIn("best epoch=2 train_loss=0.6 val_loss=0.7 valROC=0.8 valPR=0.77", result["summary"])
        self.assertIn("last epoch=3 train_loss=0.35 val_loss=1.1 valROC=0.72 valPR=0.66", result["summary"])
        self.assert_png(result["training_loss"])
        self.assert_png(result["validation_auc"])

    def test_legacy_and_resumed_missing_validation_loss_stays_unknown(self):
        for record in self.history[:2]:
            record.pop("val_loss")
        captured = []
        original_save = Figure.savefig

        def capture(figure, *args, **kwargs):
            captured.append((figure.axes[0].lines,
                             [text.get_text() for text in figure.texts]))
            return original_save(figure, *args, **kwargs)

        with patch.object(Figure, "savefig", capture):
            result = save_history_plots(self.history, self.root / "partial")
        validation = next(line for line in captured[0][0] if line.get_label() == "Validation loss")
        self.assertTrue(math.isnan(validation.get_ydata()[0]))
        self.assertTrue(math.isnan(validation.get_ydata()[1]))
        self.assertEqual(validation.get_ydata()[2], 1.1)
        self.assertIn("Validation loss not recorded for 2/3 epochs; missing values are not imputed.", captured[0][1])
        self.assertIn("first epoch=1 train_loss=0.9 val_loss=unknown", result["summary"])
        self.assertIn("best epoch=2 train_loss=0.6 val_loss=unknown", result["summary"])
        self.assertNotIn("val_loss", self.history[0])
        for record in self.history:
            record.pop("val_loss", None)
        result = save_history_plots(self.history, self.root / "legacy")
        self.assertIn("3/3 epochs", result["summary"])
        self.assert_png(result["training_loss"])

    def test_invalid_history_is_rejected_before_creating_output(self):
        invalid = [[], {}, [None], [{"epoch": 1}],
                   [*self.history, self.history[-1]], list(reversed(self.history))]
        for field, value in (("epoch", True), ("epoch", 0), ("train_loss", -1),
                             ("train_loss", "0.5"), ("train_loss", float("nan")),
                             ("train_loss", 10 ** 400), ("val_loss", float("inf")),
                             ("val_patient_roc_auc", 1.1), ("val_patient_roc_auc", True),
                             ("val_patient_pr_auc", -0.1)):
            invalid.append([{**self.history[0], field: value}])
        for index, history in enumerate(invalid):
            with self.subTest(index=index):
                output = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    save_history_plots(history, output)
                self.assertFalse(output.exists())

    def write_run(self, name, history=None):
        run = self.root / name
        run.mkdir()
        (run / "history.json").write_text(json.dumps(history or self.history), encoding="utf-8")
        return run

    def test_run_cli_preserves_records_and_existing_plots_and_prints_first_best_last(self):
        run = self.write_run("run")
        original = (run / "history.json").read_bytes()
        old = run / "training_curves"
        old.mkdir()
        (old / "training_loss.png").write_bytes(b"Existing user plot")
        console = io.StringIO()
        with redirect_stdout(console):
            output = main(["--run", str(run)])
        self.assertNotEqual(output, old)
        self.assertEqual((old / "training_loss.png").read_bytes(), b"Existing user plot")
        self.assertEqual((run / "history.json").read_bytes(), original)
        for label in ("first", "best", "last"):
            self.assertIn(f"{label} epoch=", console.getvalue())
        self.assert_png(output / "training_loss.png")
        with self.assertRaises(FileExistsError):
            main(["--run", str(run), "--output", str(output)])
        with self.assertRaises(FileExistsError):
            save_history_plots(self.history, output)

    def test_suite_cli_includes_unambiguous_failed_history_and_excludes_ambiguous_or_pending(self):
        completed = self.write_run("completed")
        failed = self.write_run("failed")
        suite = self.root / "suite"
        suite.mkdir()
        log = suite / "failed.log"
        log.write_text(f"Run directory: {failed}\nSynthetic failure\n", encoding="utf-8")
        ambiguous = suite / "ambiguous.log"
        ambiguous.write_text(f"Run directory: {completed}\nRun directory: {failed}\n", encoding="utf-8")
        manifest = {"jobs": [
            {"id": "done", "status": "completed", "run_dir": str(completed)},
            {"id": "failed", "status": "failed", "train_log": str(log)},
            {"id": "ambiguous", "status": "failed", "train_log": str(ambiguous)},
            {"id": "pending", "status": "pending", "run_dir": str(completed)},
        ]}
        (suite / "suite.json").write_text(json.dumps(manifest), encoding="utf-8")
        tracked = {path: path.read_bytes() for path in
                   [suite / "suite.json", log, ambiguous, completed / "history.json", failed / "history.json"]}
        console = io.StringIO()
        with redirect_stdout(console):
            output = main(["--suite", str(suite), "--output", str(self.root / "suite-plots")])
        self.assertEqual(set(path.name for path in output.iterdir()), {"done", "failed"})
        self.assertIn("failed; completed epochs only", console.getvalue())
        self.assertIn("[skip] ambiguous", console.getvalue())
        self.assert_png(output / "failed" / "training_loss.png")
        for path, before in tracked.items():
            self.assertEqual(path.read_bytes(), before)

    def test_suite_ids_cannot_escape_output_directory(self):
        run = self.write_run("run")
        suite = self.root / "suite"
        suite.mkdir()
        (suite / "suite.json").write_text(json.dumps({"jobs": [
            {"id": "../escape", "status": "completed", "run_dir": str(run)},
        ]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "safe IDs"):
            main(["--suite", str(suite)])
        self.assertFalse((suite / "training_curves").exists())


if __name__ == "__main__":
    unittest.main()
