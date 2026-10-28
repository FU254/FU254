"""File-only validation diagnostics for the four single-teacher seed42 jobs.

Basic reports use only the standard library. Plotting imports matplotlib lazily;
neither path loads checkpoints, patient data, Torch, or test evaluations.
"""

import csv
import json
import math
from pathlib import Path
import tempfile


LABELS = {
    "A": "supervised",
    "B": "full three-component KD",
    "C": "soft-only KD",
    "D": "full KD + L2-SP",
}
CONTRASTS = (("B_minus_A", "B", "A"), ("C_minus_A", "C", "A"),
             ("D_minus_A", "D", "A"), ("B_minus_C", "B", "C"),
             ("B_minus_D", "B", "D"))
EPOCH_METRICS = {
    "train_eval_loss": "train_eval_loss",
    "validation_loss": "val_loss",
    "train_patient_roc_auc": "train_eval_patient_roc_auc",
    "validation_patient_roc_auc": "val_patient_roc_auc",
    "train_patient_pr_auc": "train_eval_patient_pr_auc",
    "validation_patient_pr_auc": "val_patient_pr_auc",
}
MATCHED_METRICS = (*EPOCH_METRICS, "auc_gap", "loss_gap")
NOTES = (
    "Validation only; no test evaluation or test-based selection.",
    "ImageNet initialization; all student parameters trainable; one RAD-DINO teacher; seed42 only.",
    "AUC gap = clean train patient ROC-AUC minus validation patient ROC-AUC. "
    "Loss gap = validation Focal minus clean train-eval Focal.",
    "Train-eval and validation use the same Focal definition. The augmented training "
    "objective includes KD and, for D, L2-SP; its magnitude is not compared with validation Focal.",
    "Focal loss averages images, whereas ROC-AUC and PR-AUC aggregate predictions by patient. "
    "PR-AUC depends on prevalence, so train/validation PR differences are descriptive.",
    "L2-SP is 0.0001 times the raw SUM of squared backbone distances to the initial "
    "ImageNet weights. This is a fixed exploratory coefficient, not an optimized value. "
    "Raw distance and weighted penalty have separate displays.",
    "Compare validation quality, same-epoch gaps, and best-to-final changes together. "
    "A smaller gap alone does not establish that overfitting has been resolved; no universal gap cutoff is used.",
    "PR-AUC and average precision (AP) are distinct. A single seed cannot estimate seed variability or seed SD.",
    "Validation is used for checkpoint selection; comparing arms on it is exploratory. "
    "Selected validation scores and these curves/gaps do not provide an unbiased estimate of generalization.",
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _number(value):
    return float(value) if _finite(value) else None


def _difference(first, second):
    return float(first) - float(second) if _finite(first) and _finite(second) else None


def _epoch(value):
    return int(value) if _finite(value) and value >= 1 and int(value) == value else None


def _format(value):
    return f"{value:.6g}" if _finite(value) else "-"


def _jobs(state):
    jobs = state.get("plan", {}).get("jobs", [])
    records = state.get("jobs", {})
    seen = set()
    for job in jobs:
        key = job.get("group"), job.get("seed")
        if key in seen:
            raise ValueError(f"Duplicate diagnostic group/seed: {key}")
        seen.add(key)
        yield job, records.get(job["id"], {})


def _run_path(study, state, record):
    run = (record.get("receipt") or {}).get("run") or record.get("run")
    if not run:
        attempts = [attempt for attempt in record.get("attempts", []) if attempt.get("phase") == "train"]
        condition = record.get("job", {}).get("condition")
        if not attempts or condition not in {"supervised", "distillation"}:
            return None
        directory = Path(attempts[-1]["directory"])
        if not directory.is_absolute():
            base = study or state.get("study")
            directory = Path(base) / directory if base else directory
        # A live worker has not emitted a receipt yet. Inspect only its latest
        # declared train attempt and require an unambiguous condition directory.
        candidates = [path.parent for path in directory.glob(f"outputs/suites/suite_*/{condition}/history.json")
                      if path.is_file()]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous histories in declared training attempt {directory}")
        return candidates[0] if candidates else None
    path = Path(run)
    if path.is_absolute():
        return path
    base = study or state.get("study")
    return Path(base) / path if base else path


def _history(study, state, record):
    run = _run_path(study, state, record)
    if run is None or not (run / "history.json").is_file():
        return [], "history unavailable"
    try:
        saved = json.loads((run / "history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], f"history unreadable: {error}"
    if not isinstance(saved, list):
        return [], "history unavailable: expected a list"
    history, epochs = [], set()
    for item in saved:
        if not isinstance(item, dict) or _epoch(item.get("epoch")) is None:
            continue
        epoch = _epoch(item["epoch"])
        if epoch in epochs:
            raise ValueError(f"Duplicate epoch {epoch} in {run / 'history.json'}")
        epochs.add(epoch)
        history.append({**item, "epoch": epoch})
    return sorted(history, key=lambda row: row["epoch"]), "" if history else "history has no valid epochs"


def _metrics(record):
    values = {name: _number(record.get(key)) for name, key in EPOCH_METRICS.items()}
    values["auc_gap"] = _difference(values["train_patient_roc_auc"], values["validation_patient_roc_auc"])
    values["loss_gap"] = _difference(values["validation_loss"], values["train_eval_loss"])
    return values


def _select_best(history, receipt):
    selected = _epoch(receipt.get("best_epoch"))
    if selected is not None:
        return next((item for item in history if item["epoch"] == selected), {})
    candidates = [item for item in history if _finite(item.get("val_patient_roc_auc"))]
    # Sorted history and max preserve the first epoch on an exact tie.
    return max(candidates, key=lambda item: item["val_patient_roc_auc"], default={})


def _data(study, state):
    rows, histories = [], {}
    for job, record in _jobs(state):
        receipt = record.get("receipt") or {}
        if (receipt.get("test_evaluated") is True
                or receipt.get("evaluation_phase", "validation_only") != "validation_only"):
            raise ValueError(f"{job['id']}: expected a validation-only diagnostic receipt")
        history, note = _history(study, state, record)
        histories[job["id"]] = history
        best = _select_best(history, receipt)
        last = history[-1] if history else {}
        cfg = job.get("config", {})
        anchor = cfg.get("diagnostics", {}).get("l2_sp", {})
        row = {
            "id": job["id"], "group": job.get("group"), "condition": job.get("condition"),
            "seed": job.get("seed"), "status": record.get("status", "pending"),
            "initialization": cfg.get("student_initialization", "imagenet"),
            "student_checkpoint": cfg.get("student_checkpoint", ""),
            "student_mode": cfg.get("student", {}).get("mode", "full_finetune"),
            "source_suite": state.get("plan", {}).get("source_suite", ""),
            "run": str(_run_path(study, state, record) or ""),
            "l2_sp_coefficient": _number(anchor.get("coefficient")),
            "l2_sp_reduction": anchor.get("reduction", ""),
            "best_epoch": _epoch(receipt.get("best_epoch")) or best.get("epoch"),
            "last_epoch": last.get("epoch") or _epoch(receipt.get("last_epoch")),
            "best_epoch_basis": "saved best checkpoint" if receipt.get("best_epoch") else "best observed validation ROC-AUC",
        }
        best_metrics, last_metrics = _metrics(best), _metrics(last)
        # A receipt can still describe the best checkpoint when history is absent.
        val = receipt.get("validation_patient") or {}
        train = receipt.get("best_train_patient") or {}
        fallback = {
            "train_eval_loss": receipt.get("best_train_eval_loss"),
            "validation_loss": receipt.get("best_val_loss"),
            "train_patient_roc_auc": train.get("roc_auc"),
            "validation_patient_roc_auc": val.get("roc_auc", receipt.get("best_val_patient_roc_auc")),
            "train_patient_pr_auc": train.get("pr_auc"),
            "validation_patient_pr_auc": val.get("pr_auc"),
        }
        for key, value in fallback.items():
            if best_metrics[key] is None:
                best_metrics[key] = _number(value)
        best_metrics["auc_gap"] = _difference(best_metrics["train_patient_roc_auc"], best_metrics["validation_patient_roc_auc"])
        best_metrics["loss_gap"] = _difference(best_metrics["validation_loss"], best_metrics["train_eval_loss"])
        for prefix, metrics in (("best", best_metrics), ("last", last_metrics)):
            for key, value in metrics.items():
                row[prefix + "_" + key] = value
        for key in MATCHED_METRICS:
            row["last_minus_best_" + key] = _difference(last_metrics[key], best_metrics[key])
        for name, key in (("ap", "average_precision"), ("sensitivity", "sensitivity"), ("specificity", "specificity")):
            row["validation_patient_" + name] = _number(val.get(key))
        row["error"] = str(record.get("error") or "")
        row["report_note"] = note
        rows.append(row)
    return rows, histories


def comparison_rows(state, study=None):
    """One row per planned job, retaining failures and unavailable metrics."""
    return _data(study, state)[0]


def matched_epoch_rows(state, study=None):
    """Candidate minus reference at shared observed epochs of the same seed."""
    rows, histories = _data(study, state)
    indexed = {(row["group"], row["seed"]): row for row in rows}
    differences = []
    for name, candidate, reference in CONTRASTS:
        seeds = sorted({seed for group, seed in indexed if group == candidate}
                       & {seed for group, seed in indexed if group == reference})
        for seed in seeds:
            first, second = indexed[candidate, seed], indexed[reference, seed]
            left = {item["epoch"]: item for item in histories[first["id"]]}
            right = {item["epoch"]: item for item in histories[second["id"]]}
            for epoch in sorted(left.keys() & right.keys()):
                a, b = _metrics(left[epoch]), _metrics(right[epoch])
                row = {"comparison": name, "candidate": candidate, "reference": reference,
                       "seed": seed, "epoch": epoch,
                       "candidate_status": first["status"], "reference_status": second["status"],
                       "candidate_best_epoch": first["best_epoch"], "reference_best_epoch": second["best_epoch"]}
                for metric in MATCHED_METRICS:
                    row["candidate_" + metric] = a[metric]
                    row["reference_" + metric] = b[metric]
                    row["delta_" + metric] = _difference(a[metric], b[metric])
                differences.append(row)
    return differences


def print_metrics(state):
    """Print saved validation metrics and comparable gaps without any writes."""
    rows = comparison_rows(state)
    print("Single-teacher overfitting diagnostics | seed42 | validation only")
    print("Source suite: " + str(state.get("plan", {}).get("source_suite", "unavailable")))
    for row in rows:
        print(f"{row['group']} {LABELS.get(row['group'], row['condition'])} seed{row['seed']} status={row['status']} "
              f"initialization={row['initialization']} mode={row['student_mode']} best_epoch={row['best_epoch']} last_epoch={row['last_epoch']}")
        print("  " + " ".join(f"{label}={_format(row[key])}" for label, key in (
            ("validation_ROC-AUC", "best_validation_patient_roc_auc"),
            ("validation_PR-AUC", "best_validation_patient_pr_auc"),
            ("validation_AP", "validation_patient_ap"),
            ("validation_Se", "validation_patient_sensitivity"),
            ("validation_Sp", "validation_patient_specificity"))))
        print("  " + " ".join(f"{key}={_format(row[key])}" for key in (
            "best_auc_gap", "last_auc_gap", "best_loss_gap", "last_loss_gap",
            "last_minus_best_validation_patient_roc_auc")))
        print("  " + " ".join(f"{key}={_format(row[key])}" for key in (
            "best_train_eval_loss", "best_validation_loss", "last_train_eval_loss", "last_validation_loss",
            "last_validation_patient_roc_auc", "last_validation_patient_pr_auc")))
        if row["error"] or row["report_note"]:
            print("  " + "; ".join(value for value in (row["error"], row["report_note"]) if value))
    for note in NOTES:
        print(note)
    return rows


def _write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else columns)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(state, rows, matched):
    lines = ["# Single-teacher overfitting diagnostics", "",
             "Source suite: " + str(state.get("plan", {}).get("source_suite", "unavailable")), "",
             "Study status: " + str(state.get("status", "unknown")), "",
             "| Group | Status | Best / last epoch | Validation ROC-AUC | Validation PR-AUC | Validation AP | Validation Se | Validation Sp |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = [row["group"], row["status"], f"{row['best_epoch']} / {row['last_epoch']}"]
        values.extend(_format(row[key]) for key in ("best_validation_patient_roc_auc", "best_validation_patient_pr_auc",
                      "validation_patient_ap", "validation_patient_sensitivity", "validation_patient_specificity"))
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines.extend(["", "| Group | Best AUC gap | Final AUC gap | Best Focal gap | Final Focal gap | Final minus best validation ROC-AUC |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for row in rows:
        lines.append("| " + " | ".join([str(row["group"]), *(_format(row[key]) for key in (
            "best_auc_gap", "last_auc_gap", "best_loss_gap", "last_loss_gap", "last_minus_best_validation_patient_roc_auc"))]) + " |")
    lines.extend(["", "Same-epoch contrasts (candidate minus reference): B−A, C−A, D−A, B−C, B−D. "
                  f"`matched_epoch.csv` contains {len(matched)} shared-epoch rows; missing observations are not filled.", ""])
    for note in NOTES:
        lines.append("- " + note)
    for row in rows:
        if row["error"] or row["report_note"]:
            lines.extend(["", f"{row['id']}: " + "; ".join(value for value in (row["error"], row["report_note"]) if value)])
    return "\n".join(lines) + "\n"


def write_reports(study, state):
    """Write comparison.csv, matched_epoch.csv, and a concise report.md."""
    rows = comparison_rows(state, study)
    matched = matched_epoch_rows(state, study)
    root = Path(study) / "report"
    root.mkdir(parents=True, exist_ok=True)
    paths = {"directory": root, "comparison": root / "comparison.csv",
             "matched_epoch": root / "matched_epoch.csv", "markdown": root / "report.md"}
    _write_csv(paths["comparison"], rows, ["id", "group", "seed", "status"])
    _write_csv(paths["matched_epoch"], matched, ["comparison", "candidate", "reference", "seed", "epoch"])
    paths["markdown"].write_text(_markdown(state, rows, matched), encoding="utf-8")
    return paths


def plot_curves(study, state):
    """Plot saved histories in a unique directory, including partial studies."""
    rows, histories = _data(study, state)
    root = Path(study) / "report"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="curves_", dir=root))
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths = {"directory": output}
    indexed = {row["group"]: row for row in rows}
    specifications = (
        ("focal_loss", "Focal loss (same definition)", "train_eval_loss", "val_loss"),
        ("roc_auc", "Patient ROC-AUC", "train_eval_patient_roc_auc", "val_patient_roc_auc"),
        ("pr_auc", "Patient PR-AUC", "train_eval_patient_pr_auc", "val_patient_pr_auc"),
        ("optimization", "Augmented training objective and components", None, None),
    )
    for name, title, train, validation in specifications:
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        try:
            for axis, group in zip(axes.flat, LABELS):
                row = indexed.get(group)
                history = histories[row["id"]] if row else []
                axis.set(title=f"{group}: {LABELS[group]} | seed42 | {row['status'] if row else 'unplanned'}",
                         xlabel="Epoch", ylabel=title)
                if not history:
                    axis.text(.5, .5, "Saved history unavailable", ha="center", va="center", transform=axis.transAxes)
                else:
                    if train:
                        series = ((train, "Clean train (eval)"), (validation, "Validation"))
                    else:
                        series = (("train_loss", "Total optimization objective"), ("train_hard_loss", "Hard Focal"),
                                  ("train_kd_loss", "Weighted KD"), ("train_soft_loss", "Soft KD (raw)"),
                                  ("train_global_loss", "Global KD (raw)"), ("train_local_loss", "Local KD (raw)"),
                                  ("train_l2_sp_penalty", "Weighted L2-SP penalty"))
                    for key, label in series:
                        points = [(item["epoch"], item[key]) for item in history if _finite(item.get(key))]
                        if points:
                            axis.plot(*zip(*points), label=label)
                    if row["best_epoch"] is not None:
                        axis.axvline(row["best_epoch"], color="black", linestyle="--", alpha=.5,
                                     label=f"Best validation ROC epoch {row['best_epoch']}")
                    if axis.get_legend_handles_labels()[0]:
                        axis.legend(fontsize=8)
                axis.grid(alpha=.25)
            figure.suptitle(title + " | ImageNet initialization; full finetuning")
            if name == "optimization":
                figure.text(.5, .01, "Raw KD components and weighted objective contributions have different scales. "
                            "Do not compare total/KD with validation Focal. L2-SP raw sum is plotted separately.",
                            ha="center", fontsize=8)
            figure.tight_layout(rect=(0, .035, 1, .95))
            path = output / (name + ".png")
            figure.savefig(path, dpi=160)
            paths[name] = path
        finally:
            plt.close(figure)

    # Separate axes keep the raw distance from concealing its 1e-4 contribution.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    try:
        row = indexed.get("D")
        history = histories[row["id"]] if row else []
        for axis, key, title in zip(axes, ("train_l2_sp_loss", "train_l2_sp_penalty"),
                                   ("D: raw L2-SP SUM", "D: weighted L2-SP penalty (0.0001 × raw SUM)")):
            points = [(item["epoch"], item[key]) for item in history if _finite(item.get(key))]
            if points:
                axis.plot(*zip(*points), label=title)
            else:
                axis.text(.5, .5, "Saved L2-SP history unavailable", ha="center", va="center", transform=axis.transAxes)
            axis.set(title=title, xlabel="Epoch", ylabel="Squared distance" if key.endswith("loss") else "Objective contribution")
            axis.grid(alpha=.25)
        figure.tight_layout()
        paths["l2_sp"] = output / "l2_sp.png"
        figure.savefig(paths["l2_sp"], dpi=160)
    finally:
        plt.close(figure)
    return paths
