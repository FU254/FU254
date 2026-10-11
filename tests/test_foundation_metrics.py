"""Regression checks for patient-based evaluation and validation selection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.metrics import (
    aggregate_patient_predictions,
    binary_metrics,
    evaluate_predictions,
    save_evaluation,
    select_threshold,
)


def predictions():
    return pd.DataFrame({
        "sample_id": ["a1", "a2", "b1", "c1", "c2", "c3", "d1"],
        "patient_id": ["a", "a", "b", "c", "c", "c", "d"],
        "label": [0, 0, 0, 1, 1, 1, 1],
        "probability": [0.1, 0.3, 0.6, 0.2, 0.7, 0.9, 0.8],
    })


class FoundationMetricsTests(unittest.TestCase):
    def test_known_confusion_matrix_and_distinct_pr_definitions(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["ppv"], 2 / 3)
        self.assertAlmostEqual(metrics["brier_score"], (0.01 + 0.64 + 0.09 + 0.01) / 4)
        self.assertNotAlmostEqual(metrics["pr_auc"], metrics["average_precision"])
        json.dumps(metrics, allow_nan=False)

    def test_single_class_metrics_are_explicitly_undefined(self):
        metrics = binary_metrics([0, 0], [0.1, 0.2])
        for name in ("roc_auc", "pr_auc", "average_precision", "sensitivity", "ppv", "f1"):
            self.assertIsNone(metrics[name], name)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["brier_score"], 0.025000000000000005)
        self.assertTrue(metrics["notes"])
        json.dumps(metrics, allow_nan=False)

    def test_invalid_numbers_and_shapes_fail(self):
        cases = (([0, 1], [0.3, np.nan]), ([0], [1.01]), ([2], [0.3]), ([], []), ([0], [0.1, 0.2]), ([[0]], [[0.1]]))
        for labels, probabilities in cases:
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                binary_metrics(labels, probabilities)
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [0.1, 0.9], threshold=float("inf"))

    def test_patient_mean_and_conflicting_labels(self):
        grouped = aggregate_patient_predictions(predictions()).set_index("patient_id")
        self.assertEqual(len(grouped), 4)
        self.assertEqual(grouped.loc["c", "n_images"], 3)
        self.assertAlmostEqual(grouped.loc["c", "probability"], 0.6)
        conflict = predictions()
        conflict.loc[1, "label"] = 1
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            aggregate_patient_predictions(conflict)
        duplicate = predictions()
        duplicate.loc[1, "sample_id"] = "a1"
        with self.assertRaisesRegex(ValueError, "sample_id"):
            evaluate_predictions(duplicate, bootstrap_samples=0)

    def test_threshold_selection_matches_exhaustive_validation_objective(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.3, 0.8, 0.9])
        candidates = np.unique(np.r_[0.0, probabilities, 1.0])
        objectives = []
        for threshold in candidates:
            predicted = probabilities >= threshold
            sensitivity = predicted[labels == 1].mean()
            specificity = (~predicted[labels == 0]).mean()
            objectives.append((sensitivity + specificity - 1, f1_score(labels, predicted), -abs(threshold - 0.5), threshold))
        expected = float(candidates[max(range(len(candidates)), key=lambda index: objectives[index])])
        self.assertEqual(select_threshold(labels, probabilities), expected)
        self.assertEqual(select_threshold(labels, probabilities), select_threshold(labels[::-1], probabilities[::-1]))
        with self.assertRaisesRegex(ValueError, "both label classes"):
            select_threshold([0, 0], [0.1, 0.2])
        # Evaluation accepts a fixed validation-selected threshold and never tunes it.
        frame = predictions()
        result = evaluate_predictions(frame, threshold=expected, bootstrap_samples=0)
        self.assertEqual(result["image"]["threshold"], expected)
        self.assertEqual(result["patient"]["threshold"], expected)

    def test_patient_cluster_bootstrap_preserves_images_and_is_deterministic(self):
        frame = predictions()
        result = evaluate_predictions(frame, bootstrap_samples=30, seed=42)
        self.assertEqual(result, evaluate_predictions(frame, bootstrap_samples=30, seed=42))
        self.assertEqual(result["image"]["n_samples"], 7)
        self.assertEqual(result["patient"]["n_samples"], 4)
        self.assertEqual(result["bootstrap"]["unit"], "patient")
        # Independently reproduce bootstrap accuracy using patient-cluster draws.
        rng = np.random.default_rng(42)
        groups = [frame.index[frame.patient_id == patient].to_numpy() for patient in ["a", "b", "c", "d"]]
        accuracies = []
        for _ in range(30):
            draws = rng.integers(0, 4, size=4)
            selected = frame.iloc[np.concatenate([groups[index] for index in draws])]
            accuracies.append(((selected.probability >= 0.5) == selected.label).mean())
        ci = result["bootstrap"]["image"]["accuracy"]
        self.assertAlmostEqual(ci["lower"], float(np.quantile(accuracies, 0.025)))
        self.assertAlmostEqual(ci["upper"], float(np.quantile(accuracies, 0.975)))
        self.assertEqual(ci["valid_samples"], 30)
        self.assertLess(result["bootstrap"]["image"]["roc_auc"]["valid_samples"], 30)
        json.dumps(result, allow_nan=False)

    def test_headless_artifacts_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = save_evaluation(predictions(), temporary, bootstrap_samples=10, seed=4)
            root = Path(temporary)
            saved = json.loads((root / "metrics.json").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(len(pd.read_csv(root / "patient_predictions.csv")), 4)
            for filename in ("roc_curve.png", "precision_recall_curve.png", "calibration_curve.png"):
                self.assertGreater((root / filename).stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                save_evaluation(predictions(), temporary, bootstrap_samples=0)

    def test_single_class_bootstrap_and_curve_artifacts(self):
        frame = predictions().query("label == 0").copy()
        with tempfile.TemporaryDirectory() as temporary:
            result = save_evaluation(frame, temporary, bootstrap_samples=5, seed=8)
            self.assertIsNone(result["image"]["roc_auc"])
            for level in ("image", "patient"):
                ci = result["bootstrap"][level]["roc_auc"]
                self.assertEqual(ci, {"lower": None, "upper": None, "valid_samples": 0})
            self.assertTrue((Path(temporary) / "roc_curve.png").exists())
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
