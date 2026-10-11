"""Binary chest-radiograph evaluation with patient-cluster uncertainty.

Thresholds are selected on validation predictions and then kept fixed for test
evaluation. Image-level and mean-probability patient-level results are reported
separately; the bootstrap samples patients, retaining all their images together.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


_RATE_METRICS = (
    "roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity",
    "ppv", "npv", "f1", "accuracy", "brier_score",
)
_ARTIFACT_NAMES = (
    "predictions.csv", "patient_predictions.csv", "metrics.json",
    "roc_curve.png", "precision_recall_curve.png", "calibration_curve.png",
)


def _arrays(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have the same nonzero length.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    return labels.astype(np.int64), probabilities


def _threshold(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("The probability threshold must be finite and lie in [0, 1].")
    return value


def binary_metrics(labels, probabilities, threshold=0.5):
    """Return JSON-safe metrics; undefined rates are None with explanatory notes.

    ``pr_auc`` integrates the precision-recall curve by the trapezoidal rule.
    ``average_precision`` is reported separately and is not called PR-AUC.
    Predictions equal to the threshold are assigned to the positive class.
    """
    labels, probabilities = _arrays(labels, probabilities)
    threshold = _threshold(threshold)
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    notes = []

    def ratio(numerator, denominator, name):
        if denominator == 0:
            notes.append(f"{name} is undefined because its denominator is zero.")
            return None
        return float(numerator / denominator)

    result = {
        "n_samples": int(len(labels)),
        "n_positive": int(positive.sum()),
        "n_negative": int((~positive).sum()),
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": ratio(tp, tp + fn, "sensitivity"),
        "specificity": ratio(tn, tn + fp, "specificity"),
        "ppv": ratio(tp, tp + fp, "ppv"),
        "npv": ratio(tn, tn + fn, "npv"),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "f1"),
        "accuracy": float((tp + tn) / len(labels)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }
    if np.unique(labels).size == 2:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        result.update(
            roc_auc=float(roc_auc_score(labels, probabilities)),
            pr_auc=float(auc(recall, precision)),
            average_precision=float(average_precision_score(labels, probabilities)),
        )
    else:
        result.update(roc_auc=None, pr_auc=None, average_precision=None)
        notes.append("ROC-AUC, PR-AUC and average precision require both label classes.")
    result["notes"] = notes
    return result


def select_threshold(labels, probabilities):
    """Select a validation threshold by Youden J, breaking ties by F1.

    Final ties prefer the value closest to 0.5 and then the higher threshold.
    This function must receive validation predictions, never test predictions.
    Both validation classes are required. No metric is recomputed per threshold.
    """
    labels, probabilities = _arrays(labels, probabilities)
    if np.unique(labels).size != 2:
        raise ValueError("Validation threshold selection requires both label classes.")
    order = np.argsort(probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    positives_before = np.r_[0, np.cumsum(labels[order])]
    thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    starts = np.searchsorted(sorted_probabilities, thresholds, side="left")
    total_positive = int(labels.sum())
    total_negative = len(labels) - total_positive
    tp = total_positive - positives_before[starts]
    fp = len(labels) - starts - tp
    fn = total_positive - tp
    youden = tp / total_positive - fp / total_negative
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    candidates = np.flatnonzero(np.isclose(youden, youden.max(), rtol=0, atol=1e-12))
    candidates = candidates[np.isclose(f1[candidates], f1[candidates].max(), rtol=0, atol=1e-12)]
    best = min(candidates, key=lambda index: (abs(thresholds[index] - 0.5), -thresholds[index]))
    return float(thresholds[best])


def _prediction_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Predictions must be a pandas DataFrame.")
    required = {"sample_id", "patient_id", "label", "probability"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction columns are missing: {missing}")
    frame = frame.copy()
    for column in ("sample_id", "patient_id"):
        if frame[column].isna().any():
            raise ValueError(f"{column} must not be missing.")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} must not be empty.")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must uniquely identify each evaluated image.")
    labels, probabilities = _arrays(frame["label"], frame["probability"])
    frame["label"] = labels
    frame["probability"] = probabilities
    conflicts = frame.groupby("patient_id", sort=False)["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting labels within patients: {int((conflicts > 1).sum())} affected patients.")
    return frame


def aggregate_patient_predictions(frame, method="mean"):
    """Aggregate image probabilities per patient; conflicting labels fail.

    The primary protocol uses the arithmetic mean, specified explicitly in the
    evaluation report. Other aggregation methods require a separate protocol.
    """
    if method != "mean":
        raise ValueError("Only the predefined mean patient aggregation is supported.")
    frame = _prediction_frame(frame)
    return (
        frame.groupby("patient_id", sort=False, as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"), n_images=("sample_id", "size"))
    )


def evaluate_predictions(frame, threshold=0.5, bootstrap_samples=1000, seed=42):
    """Evaluate fixed-threshold image and patient predictions with 95% CIs.

    Each replicate draws the original number of patients with replacement.
    Repeated patient draws retain all corresponding images and their multiplicity.
    Single-class replicates have undefined AUCs and are excluded for those CIs;
    ``valid_samples`` records the number of usable replicates for each metric.
    The threshold is never reselected during evaluation or bootstrapping.
    """
    frame = _prediction_frame(frame)
    threshold = _threshold(threshold)
    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) != bootstrap_samples or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer.")
    bootstrap_samples = int(bootstrap_samples)
    patients = aggregate_patient_predictions(frame)
    labels = frame["label"].to_numpy()
    probabilities = frame["probability"].to_numpy()
    patient_labels = patients["label"].to_numpy()
    patient_probabilities = patients["probability"].to_numpy()
    result = {
        "threshold_source": "fixed externally; select on validation only",
        "patient_aggregation": "mean image probability",
        "image": binary_metrics(labels, probabilities, threshold),
        "patient": binary_metrics(patient_labels, patient_probabilities, threshold),
        "bootstrap": {
            "unit": "patient", "samples": bootstrap_samples,
            "confidence_level": 0.95, "seed": int(seed), "image": {}, "patient": {},
        },
    }
    if bootstrap_samples == 0:
        return result
    # Group indices and patient rows share the same stable order.
    groups = list(frame.groupby("patient_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    values = {level: {name: [] for name in _RATE_METRICS} for level in ("image", "patient")}
    for _ in range(bootstrap_samples):
        draws = rng.integers(0, len(groups), size=len(groups))
        image_indices = np.concatenate([groups[index] for index in draws])
        replicate_metrics = {
            "image": binary_metrics(labels[image_indices], probabilities[image_indices], threshold),
            "patient": binary_metrics(patient_labels[draws], patient_probabilities[draws], threshold),
        }
        for level, metrics in replicate_metrics.items():
            for name in _RATE_METRICS:
                if metrics[name] is not None:
                    values[level][name].append(metrics[name])
    for level in values:
        for name, samples in values[level].items():
            # One valid replicate cannot provide an uncertainty interval.
            bounds = np.quantile(samples, [0.025, 0.975]) if len(samples) >= 2 else [None, None]
            result["bootstrap"][level][name] = {
                "lower": None if bounds[0] is None else float(bounds[0]),
                "upper": None if bounds[1] is None else float(bounds[1]),
                "valid_samples": len(samples),
            }
    return result


def _save_plots(frame, patients, output_dir):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    for plot_name, filename in (
        ("roc", "roc_curve.png"),
        ("pr", "precision_recall_curve.png"),
        ("calibration", "calibration_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        drawn = False
        try:
            for name, predictions in (("Image", frame), ("Patient (mean)", patients)):
                labels = predictions["label"].to_numpy()
                probabilities = predictions["probability"].to_numpy()
                if plot_name in ("roc", "pr") and np.unique(labels).size < 2:
                    continue
                if plot_name == "roc":
                    x, y, _ = roc_curve(labels, probabilities)
                    label = f"{name}: AUC={roc_auc_score(labels, probabilities):.3f}"
                elif plot_name == "pr":
                    y, x, _ = precision_recall_curve(labels, probabilities)
                    label = f"{name}: trapezoidal AUC={auc(x, y):.3f}"
                else:
                    y, x = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
                    label = name
                axis.plot(x, y, marker="o" if plot_name == "calibration" else None, label=label)
                drawn = True
            if plot_name == "roc":
                axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
                axis.set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC curve")
            elif plot_name == "pr":
                axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
            else:
                axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
                axis.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration (10 uniform bins)")
            if drawn:
                axis.legend(loc="best")
            else:
                axis.text(0.5, 0.5, "Curve undefined: only one label class", ha="center", va="center", transform=axis.transAxes)
            axis.set(xlim=(0, 1), ylim=(0, 1))
            axis.grid(alpha=0.2)
            figure.tight_layout()
            with (output_dir / filename).open("xb") as handle:
                figure.savefig(handle, format="png", dpi=160)
        finally:
            plt.close(figure)


def save_evaluation(frame, output_dir, threshold=0.5, bootstrap_samples=1000, seed=42):
    """Save predictions, JSON metrics and headless ROC/PR/calibration plots.

    Callers allocate a distinct output directory per run and split. Existing
    evaluation artifacts cause FileExistsError instead of silent replacement.
    """
    frame = _prediction_frame(frame)
    patients = aggregate_patient_predictions(frame)
    output_dir = Path(output_dir)
    existing = [name for name in _ARTIFACT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Evaluation artifacts already exist in {output_dir}: {existing}")
    result = evaluate_predictions(frame, threshold, bootstrap_samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("predictions.csv", frame), ("patient_predictions.csv", patients)):
        with (output_dir / filename).open("x", encoding="utf-8", newline="") as handle:
            data.to_csv(handle, index=False)
    with (output_dir / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    _save_plots(frame, patients, output_dir)
    return result
