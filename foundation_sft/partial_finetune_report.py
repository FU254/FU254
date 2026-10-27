"""Validation-only partial-finetuning screening and comparable learning curves.

The existing full-finetuning baseline and the one validation-selected candidate
may expose test metrics. Unselected candidate receipts remain validation-only.
Importing this module does not load Torch, image data or plotting libraries.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

from . import distillation_suite as records
from . import teacher_weight_report as common


SUCCESS = {"completed", "resumed", "reused"}
ARMS = ("baseline", "last8", "last4", "last2")
LABELS = {"baseline": "Full 12 blocks (existing P-E)",
          "last8": "Last 8 blocks", "last4": "Last 4 blocks", "last2": "Last 2 blocks"}
SELECTION_RULE = (
    "Maximize best validation patient ROC-AUC after all three candidates complete; "
    "exact ties prefer fewer trainable student parameters. Test metrics are not used."
)
FIT_NOTE = (
    "Train-eval and validation losses are the same Focal loss in eval mode, without random augmentation. "
    "Optimization losses include KD and are plotted separately. Compare complete curves and matched epochs; "
    "smaller gaps at different best epochs alone do not establish reduced overfitting."
)


def _entries(state):
    return {"baseline": state.get("baseline", {}),
            **{arm: state.get("candidates", {}).get(arm, {}) for arm in ARMS[1:]}}


def _count(entry, result):
    values = [result.get("student_trainable_parameters"),
              entry.get("check", {}).get("model_metadata", {}).get("trainable_parameters")]
    available = [value for value in values if value is not None]
    if not available:
        raise ValueError("Completed partial-finetuning record lacks a trainable student parameter count")
    if any(type(value) is not int or value <= 0 for value in available):
        raise ValueError("Trainable student parameter count must be a positive integer")
    if len(set(available)) != 1:
        raise ValueError("Saved trainable student parameter counts differ")
    return available[0]


def _completed(study, arm, entry):
    result = entry.get("result")
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("A completed partial-finetuning arm needs its completed receipt")
    expected_phase = "main" if arm == "baseline" else "pilot"
    if result.get("phase") != expected_phase:
        raise ValueError("Partial-finetuning screening receipt has the wrong phase")
    if arm != "baseline" and any(key in result for key in ("test_patient", "roc_auc_ci")):
        raise ValueError("A screening candidate must not contain test metrics")
    auc = common._number(result.get("best_val_patient_roc_auc"), "best validation ROC-AUC", auc=True)
    run = common._run_path(study, result)
    history, best = common._history(run, auc)
    return result, run, history, best


def _base_row(arm, entry):
    config = entry.get("job", {}).get("config", {})
    return {"arm": arm, "student_update": LABELS[arm],
        "train_last_blocks": 12 if arm == "baseline" else int(arm[4:]),
        "seed": config.get("seed", 42), "backbone_lr": config.get("training", {}).get("backbone_lr"),
        "status": entry.get("status", "pending"), "trainable_student_parameters": None,
        "val_auc": None, "val_pr_auc": None, "best_epoch": None, "last_epoch": None,
        "run": None, "error": entry.get("error")}


def screening_rows(study, state):
    rows = []
    entries = _entries(state)
    for arm in ARMS:
        entry = entries[arm]
        row = _base_row(arm, entry)
        if row["status"] in SUCCESS:
            result, run, history, best = _completed(study, arm, entry)
            row.update(run=str(run), trainable_student_parameters=_count(entry, result),
                val_auc=best["val_patient_roc_auc"], val_pr_auc=best.get("val_patient_pr_auc"),
                best_epoch=best["epoch"], last_epoch=history[-1]["epoch"])
        rows.append(row)
    return rows


def _test_row(arm, receipt, *, source, selected):
    if receipt.get("phase") != "main" or receipt.get("status") != "completed":
        raise ValueError("Test reporting requires a completed main evaluation receipt")
    patient = receipt.get("test_patient")
    if not isinstance(patient, dict):
        raise ValueError("A reported test evaluation needs saved patient metrics")
    row = {"arm": arm, "student_update": LABELS[arm], "source": source,
        "selected_by_validation": bool(selected), "roc_auc": None, "pr_auc": None,
        "ap": None, "sensitivity": None, "specificity": None,
        "roc_ci_lower": None, "roc_ci_upper": None, "run": receipt.get("run")}
    for target, key in (("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"),
                        ("ap", "average_precision"), ("sensitivity", "sensitivity"),
                        ("specificity", "specificity")):
        row[target] = common._number(patient.get(key), "test " + key, optional=target == "ap", auc=True)
    interval = receipt.get("roc_auc_ci")
    if isinstance(interval, dict):
        row.update(roc_ci_lower=interval.get("lower"), roc_ci_upper=interval.get("upper"))
    elif isinstance(interval, (list, tuple)) and len(interval) == 2:
        row.update(roc_ci_lower=interval[0], roc_ci_upper=interval[1])
    for key in ("roc_ci_lower", "roc_ci_upper"):
        common._number(row[key], key, optional=True, auc=True)
    if row["roc_ci_lower"] is not None and row["roc_ci_upper"] is not None and row["roc_ci_lower"] > row["roc_ci_upper"]:
        raise ValueError("Invalid saved ROC-AUC interval")
    return row


def test_rows(study, state):
    entries = _entries(state)
    selection = state.get("selection", {})
    selected = selection.get("arm") if selection.get("status") == "selected" else None
    if selected is not None and selected not in ARMS:
        raise ValueError("Unknown validation-selected partial-finetuning arm")
    output = []
    baseline = entries["baseline"]
    if baseline.get("status") in SUCCESS:
        result, _, _, _ = _completed(study, "baseline", baseline)
        output.append(_test_row("baseline", result, source="historical baseline", selected=selected == "baseline"))
    evaluation = state.get("selected_evaluation", {})
    if evaluation.get("status") not in SUCCESS:
        return output
    if selected is None:
        raise ValueError("A completed selected evaluation requires frozen validation selection")
    if entries[selected].get("status") not in SUCCESS:
        raise ValueError("Selected test evaluation requires a completed screening arm")
    result = evaluation.get("result")
    if not isinstance(result, dict):
        raise ValueError("Selected evaluation lacks its completed receipt")
    source, run, _, _ = _completed(study, selected, entries[selected])
    records._same("selected evaluation run", str(common._run_path(study, result)), str(run))
    records._same("selected evaluation validation best", result.get("best_val_patient_roc_auc"), source.get("best_val_patient_roc_auc"))
    if source.get("checkpoint_sha256") is not None:
        records._same("selected evaluation checkpoint", result.get("checkpoint_sha256"), source["checkpoint_sha256"])
    if selected == "baseline":
        records._same("selected baseline result", result, source)
        return output
    output.append(_test_row(selected, result, source="validation-selected candidate", selected=True))
    return output


def overfitting_rows(rows):
    output = []
    for row in rows:
        if row["status"] not in SUCCESS:
            continue
        history, best = common._history(row["run"], row["val_auc"])
        value = {key: row[key] for key in ("arm", "seed", "backbone_lr", "trainable_student_parameters", "run")}
        for prefix, epoch in (("best", best), ("last", history[-1])):
            value.update({prefix + "_epoch": epoch["epoch"],
                prefix + "_train_eval_loss": epoch["train_eval_loss"],
                prefix + "_val_loss": epoch["val_loss"],
                prefix + "_loss_gap": epoch["val_loss"] - epoch["train_eval_loss"],
                prefix + "_train_auc": epoch["train_eval_patient_roc_auc"],
                prefix + "_val_auc": epoch["val_patient_roc_auc"],
                prefix + "_auc_gap": epoch["train_eval_patient_roc_auc"] - epoch["val_patient_roc_auc"],
                prefix + "_train_pr_auc": epoch.get("train_eval_patient_pr_auc"),
                prefix + "_val_pr_auc": epoch.get("val_patient_pr_auc")})
        output.append(value)
    return output


def matched_epoch_rows(rows):
    available = {row["arm"]: row for row in rows if row["status"] in SUCCESS}
    if "baseline" not in available:
        return []
    base_history, _ = common._history(available["baseline"]["run"], available["baseline"]["val_auc"])
    baseline = {row["epoch"]: row for row in base_history}
    output = []
    for arm in ARMS[1:]:
        if arm not in available:
            continue
        history, _ = common._history(available[arm]["run"], available[arm]["val_auc"])
        for row in history:
            epoch = row["epoch"]
            if epoch not in baseline:
                continue
            base = baseline[epoch]
            output.append({"arm": arm, "reference": "baseline", "epoch": epoch,
                "candidate_val_auc": row["val_patient_roc_auc"], "baseline_val_auc": base["val_patient_roc_auc"],
                "delta_val_auc": row["val_patient_roc_auc"] - base["val_patient_roc_auc"],
                "candidate_auc_gap": row["train_eval_patient_roc_auc"] - row["val_patient_roc_auc"],
                "baseline_auc_gap": base["train_eval_patient_roc_auc"] - base["val_patient_roc_auc"],
                "candidate_loss_gap": row["val_loss"] - row["train_eval_loss"],
                "baseline_loss_gap": base["val_loss"] - base["train_eval_loss"]})
    return output


def build_report(study, state):
    rows = screening_rows(study, state)
    return {"rows": rows, "test_rows": test_rows(study, state),
        "overfitting": overfitting_rows(rows), "matched_epochs": matched_epoch_rows(rows),
        "selection": deepcopy(state.get("selection", {})), "selection_rule": SELECTION_RULE,
        "fit_note": FIT_NOTE, "single_seed": True}


def write_report(study, state):
    report = build_report(study, state)
    root = Path(study) / "report"
    for name, key in (("screening", "rows"), ("test_comparison", "test_rows"),
                      ("overfitting", "overfitting"), ("matched_epochs", "matched_epochs")):
        records._csv(root / (name + ".csv"), report[key])
    records._json(root / "selection.json", {"selection": report["selection"],
        "rule": SELECTION_RULE, "test_metrics_used": False})
    return report


def print_metrics(study, state):
    report = build_report(study, state)
    print(f"\nPartial-finetuning study: {study}")
    print("Screening is validation only. Unselected candidate test metrics are not evaluated or reported.")
    columns = ("arm", "train_last_blocks", "status", "trainable_student_parameters", "val_auc", "val_pr_auc", "best_epoch", "last_epoch")
    print("  ".join(columns))
    for row in report["rows"]:
        print("  ".join(records._number(row.get(key)) if key in ("val_auc", "val_pr_auc")
                        else str(row.get(key) if row.get(key) is not None else "-") for key in columns))
        if row.get("error"):
            print("  " + str(row["error"]))
    print("Frozen validation selection: " + json.dumps(report["selection"], ensure_ascii=False, sort_keys=True))
    print("Saved patient test metrics: existing baseline and the validation-selected candidate only.")
    columns = ("arm", "source", "roc_auc", "pr_auc", "ap", "sensitivity", "specificity", "roc_ci_lower", "roc_ci_upper")
    print("  ".join(columns))
    for row in report["test_rows"]:
        print("  ".join(str(row[key]) if key in ("arm", "source") else records._number(row.get(key)) for key in columns))
    for row in report["overfitting"]:
        print(f"{row['arm']} best={row['best_epoch']} last={row['last_epoch']} "
            f"best_auc_gap={row['best_auc_gap']:.4f} last_auc_gap={row['last_auc_gap']:.4f} "
            f"best_loss_gap={row['best_loss_gap']:.6f} last_loss_gap={row['last_loss_gap']:.6f}")
    print(SELECTION_RULE)
    print(FIT_NOTE)
    print("Single seed42: seed variability and seed SD are not estimable. PR-AUC and AP are distinct.")
    return report


def plot_curves(study, state):
    report = build_report(study, state)
    available = {row["arm"]: row for row in report["rows"] if row["status"] in SUCCESS}
    if not available:
        raise ValueError("No completed screening history is available")
    histories = {arm: common._history(row["run"], row["val_auc"]) for arm, row in available.items()}
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    root = Path(study) / "report"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="curves_", dir=root))
    paths = {"directory": str(output)}

    def panels(name, painter, ylabel, bounds=None):
        figure, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
        try:
            for axis, arm in zip(axes, ARMS):
                axis.set(title=LABELS[arm], xlabel="Epoch", ylabel=ylabel)
                axis.grid(alpha=.25)
                if bounds is not None:
                    axis.set_ylim(*bounds)
                if arm not in histories:
                    axis.text(.5, .5, "No completed history\n" + report["rows"][ARMS.index(arm)]["status"],
                              ha="center", va="center", transform=axis.transAxes)
                    continue
                history, best = histories[arm]
                epochs = [row["epoch"] for row in history]
                painter(axis, epochs, history)
                axis.axvline(best["epoch"], color="black", linestyle="--", alpha=.5,
                             label=f"Best validation ROC epoch {best['epoch']}")
                axis.legend(fontsize=8)
            figure.tight_layout()
            path = output / (name + ".png")
            figure.savefig(path, dpi=160)
            paths[name] = str(path)
        finally:
            plt.close(figure)

    def paired(axis, epochs, history, train, validation):
        for key, label in ((train, "Train (eval)"), (validation, "Validation")):
            values = common._optional_series(history, key, auc="auc" in key)
            if any(math.isfinite(value) for value in values):
                axis.plot(epochs, values, label=label)
            else:
                axis.text(.02, .05 if label.startswith("Train") else .12, label + " not recorded",
                          fontsize=8, transform=axis.transAxes)

    panels("loss_curves", lambda a, e, h: paired(a, e, h, "train_eval_loss", "val_loss"), "Focal loss")
    panels("roc_auc_curves", lambda a, e, h: paired(a, e, h, "train_eval_patient_roc_auc", "val_patient_roc_auc"),
           "Patient ROC-AUC", (0, 1))
    panels("pr_auc_curves", lambda a, e, h: paired(a, e, h, "train_eval_patient_pr_auc", "val_patient_pr_auc"),
           "Patient PR-AUC", (0, 1))

    def optimization(axis, epochs, history):
        for key, label in (("train_loss", "Total"), ("train_hard_loss", "Hard"), ("train_kd_loss", "Weighted KD")):
            values = common._optional_series(history, key)
            if any(math.isfinite(value) for value in values):
                axis.plot(epochs, values, label=label)
        axis.text(.02, .02, "Optimization loss includes KD", fontsize=7, transform=axis.transAxes)
    panels("optimization_losses", optimization, "Training objective")
    records._json(output / "plot_metadata.json", {"available_arms": [a for a in ARMS if a in histories],
        "missing_arms": [a for a in ARMS if a not in histories], "loss_definition": FIT_NOTE,
        "best_epoch": "First epoch attaining maximum validation patient ROC-AUC", "test_curves": False})
    return paths
