"""Describe soft-only random-student training alongside its frozen A/B runs."""

import math
from pathlib import Path
import tempfile

from . import random_student_suite as random
from . import weight_decay_report as history_report
from . import droppath_baseline as baseline_api


SUCCESS = history_report.SUCCESS
LABELS = ("supervised", "full_kd", "soft_only")
SELECTION_RULE = "Maximize seed42 B validation patient ROC-AUC; exact ties retain existing full KD. Test scores are not used."


def metric_rows(study, state):
    rows = []
    for label, name in (("supervised", "A_seed42"), ("full_kd", "B_seed42")):
        source = state["baseline"]["jobs"][name]
        batch = {"jobs": {name: {"job": source["job"], "check": source["check"],
            "result": source["result"], "status": "completed", "source": "frozen control"}}}
        rows.extend({"arm": label, "weight_decay": .01, "drop_path_rate": 0., **row}
                    for row in random.metric_rows(batch))
    suite = Path(state["candidate"]["suite"])
    if not suite.is_absolute():
        suite = Path(study) / suite
    batch = random.legacy._read(suite / "suite.json")
    rows.extend({"arm": "soft_only", "weight_decay": .01, "drop_path_rate": 0., **row}
                for row in random.metric_rows(batch))
    for row in rows:
        row["val_pr_auc"] = None
        if row["status"] in SUCCESS:
            _, best = history_report._load_history(row)
            row["val_pr_auc"] = history_report._optional_pr(best)
    return rows


def paired_rows(rows):
    usable = {row["arm"]: row for row in rows if row["status"] in SUCCESS}
    result = []
    for label, candidate, reference in (("full_kd_gain", "full_kd", "supervised"),
            ("soft_kd_gain", "soft_only", "supervised"),
            ("soft_minus_full", "soft_only", "full_kd")):
        if candidate not in usable or reference not in usable:
            continue
        left, right = usable[candidate], usable[reference]
        row = {"comparison": label, "candidate": candidate, "reference": reference, "seed": 42}
        for key in (*random.METRICS, "val_pr_auc"):
            a, b = left.get(key), right.get(key)
            row["delta_" + key] = a-b if type(a) in (float, int) and type(b) in (float, int) else None
        result.append(row)
    return result


def validation_selection(rows):
    value = {"rule": SELECTION_RULE, "test_metrics_used": False, "seed": 42,
        "status": "incomplete", "selected_arm": None, "candidates": []}
    for arm in LABELS:
        found = [row for row in rows if row["arm"] == arm]
        if len(found) != 1 or found[0]["status"] not in SUCCESS:
            return value
        row = found[0]
        if (row["seed"] != 42 or type(row.get("val_auc")) not in (int, float)
                or not math.isfinite(row["val_auc"]) or not 0 <= row["val_auc"] <= 1):
            return value
        if arm != "supervised":
            value["candidates"].append({"arm": arm, "val_auc": row["val_auc"]})
    winner = min(value["candidates"], key=lambda row: (-row["val_auc"], row["arm"] != "full_kd"))
    value.update(status="complete", selected_arm=winner["arm"])
    return value


def build_report(study, state):
    baseline_api.validate_snapshot(state["baseline"])
    rows = metric_rows(study, state)
    return {"rows": rows, "paired": paired_rows(rows),
        "overfitting": history_report.overfitting_rows(rows),
        "selection": validation_selection(rows)}


def write_report(study, state):
    result = build_report(study, state)
    root = Path(study) / "report"
    for filename, key in (("comparison", "rows"), ("matched_seed_differences", "paired"), ("overfitting", "overfitting")):
        random.legacy._csv(root / (filename + ".csv"), result[key])
    random.legacy._json(root / "validation_selection.json", result["selection"])
    return result


def print_metrics(study, state):
    result = build_report(study, state)
    print(f"\nSoft-only study: {study}")
    print(f"Source WD study: {state['baseline']['study']}; random initialization; seed42; WD=0.01; DropPath=0")
    print("Frozen controls: A supervised, B full KD; exactly one new B soft-only training.")
    columns = ("arm", "group", "seed", "status", "val_auc", "val_pr_auc", "roc_auc", "pr_auc", "ap", "sensitivity", "specificity")
    print("  ".join(columns))
    for row in result["rows"]:
        print("  ".join(random.legacy._number(row.get(key)) if key in (*random.METRICS, "val_pr_auc")
                        else str(row.get(key, "-")) for key in columns))
        if row.get("error"):
            print("  " + row["error"])
    for row in result["paired"]:
        print(f"{row['comparison']} seed42: delta_val_auc={random.legacy._number(row['delta_val_auc'])}; "
            f"delta_roc_auc={random.legacy._number(row['delta_roc_auc'])}; delta_pr_auc={random.legacy._number(row['delta_pr_auc'])}")
    print("Same-definition Focal losses and patient AUC at selected best and final epochs:")
    for row in result["overfitting"]:
        print(f"{row['arm']} seed42 best={row['best_epoch']} last={row['last_epoch']} "
            f"best_auc_gap={row['best_auc_gap']:.4f} last_auc_gap={row['last_auc_gap']:.4f} "
            f"best_loss_gap={row['best_loss_gap']:.6f} last_loss_gap={row['last_loss_gap']:.6f}")
    if result["selection"]["status"] == "complete":
        print("Validation-only selected setting: " + result["selection"]["selected_arm"])
    else:
        print("Validation selection pending: all three saved records must be complete.")
    print("Single seed42: seed variability and seed SD are not estimable.")
    print("PR-AUC and AP are distinct. Inspect full histories: different best-epoch gaps alone do not establish reduced overfitting.")
    print(SELECTION_RULE)
    return result


def plot_curves(study, state):
    """Read saved histories only; preserve prior figures in unique directories."""
    result = build_report(study, state)
    rows = result["rows"]
    if len(rows) != 3 or any(row["status"] not in SUCCESS for row in rows):
        raise ValueError("All three saved records must be complete before plotting")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharey="col")
    try:
        for index, row in enumerate(rows):
            history, best = history_report._load_history(row)
            epochs = [record["epoch"] for record in history]
            for column, (train, val, ylabel) in enumerate((
                    ("train_eval_loss", "val_loss", "Focal loss"),
                    ("train_eval_patient_roc_auc", "val_patient_roc_auc", "Patient ROC-AUC"))):
                axis = axes[index, column]
                axis.plot(epochs, [record[train] for record in history], label="Train (eval)")
                axis.plot(epochs, [record[val] for record in history], label="Validation")
                axis.axvline(best["epoch"], color="black", linestyle="--", alpha=.5,
                    label=f"Best val ROC epoch {best['epoch']}")
                axis.set(title=f"{row['arm']} | seed42 | WD=0.01 | DropPath=0", xlabel="Epoch", ylabel=ylabel)
                axis.grid(alpha=.25)
                axis.legend(fontsize=9)
        figure.tight_layout()
        root = Path(study) / "report"
        root.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="curves_", dir=root)) / "training_validation.png"
        figure.savefig(output, dpi=180)
        return output
    finally:
        plt.close(figure)
