"""Describe a fixed weight-decay study; parameter selection uses validation only."""

import math
from pathlib import Path
import statistics

from . import distillation_suite as legacy
from . import random_student_suite as random_suite


SEEDS = {42}
SUCCESS = {"completed", "resumed", "reused"}
SELECTION_RULE = (
    "Maximize validation patient ROC-AUC of B at seed42; "
    "exact ties prefer baseline, then smaller weight decay. "
    "Keep the same selected setting for its paired A control. Test metrics are not used."
)


def _arms(study, state):
    baseline = state["baseline"]
    jobs = {}
    for name, entry in baseline["jobs"].items():
        if entry["job"]["seed"] not in SEEDS:
            continue
        jobs[name] = {"job": entry["job"], "check": entry["check"],
                      "result": entry["result"], "status": "completed", "source": "baseline"}
    yield "baseline", baseline["weight_decay"], {"jobs": jobs}
    for name, arm in state["arms"].items():
        suite = Path(arm["suite"])
        if not suite.is_absolute():
            suite = Path(study) / suite
        yield name, arm["weight_decay"], legacy._read(suite / "suite.json")


def metric_rows(study, state):
    result = []
    for arm, decay, batch in _arms(study, state):
        for row in random_suite.metric_rows(batch):
            value = {"arm": arm, "weight_decay": decay, **row, "val_pr_auc": None}
            if row["status"] in SUCCESS:
                _, best = _load_history(row)
                value["val_pr_auc"] = _optional_pr(best)
            result.append(value)
    return result


def summary_rows(rows):
    result = []
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        for summary in random_suite.summary_rows(selected):
            values = [row["val_pr_auc"] for row in selected if row["group"] == summary["group"]
                      and row["status"] in SUCCESS and row.get("val_pr_auc") is not None]
            summary.update(val_pr_auc_n=len(values),
                val_pr_auc_mean=statistics.mean(values) if values else None,
                val_pr_auc_sd=statistics.stdev(values) if len(values) > 1 else None)
            result.append({"arm": arm, "weight_decay": selected[0]["weight_decay"], **summary})
    return result


def paired_rows(rows):
    usable = {(r["arm"], r["group"], r["seed"]): r for r in rows if r["status"] in SUCCESS}
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    pairs = []
    for arm in arms:
        for seed in sorted(SEEDS):
            pairs.append(("distillation_gain", arm, "B", arm, "A", seed))
        if arm != "baseline":
            for group in ("A", "B"):
                for seed in sorted(SEEDS):
                    pairs.append(("weight_decay_gain", arm, group, "baseline", group, seed))
    output = []
    for comparison, candidate, group, reference, reference_group, seed in pairs:
        left, right = usable.get((candidate, group, seed)), usable.get((reference, reference_group, seed))
        if left is None or right is None:
            continue
        value = {"comparison": comparison, "candidate": candidate, "reference": reference,
                 "group": group, "reference_group": reference_group, "seed": seed,
                 "weight_decay": left["weight_decay"], "reference_weight_decay": right["weight_decay"]}
        for metric in legacy.METRICS:
            a, b = left.get(metric), right.get(metric)
            value["delta_" + metric] = a-b if type(a) in (float, int) and type(b) in (float, int) else None
        output.append(value)
    return output


def _history_rows(history):
    if not isinstance(history, list) or not history:
        raise ValueError("Completed weight-decay run needs a nonempty history.json")
    previous = 0
    for row in history:
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch <= previous:
            raise ValueError("History epochs must be positive and strictly increasing")
        previous = epoch
        for name in ("train_eval_loss", "val_loss", "train_eval_patient_roc_auc", "val_patient_roc_auc"):
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing or invalid same-definition history field: " + name)
            if name.endswith("roc_auc") and value > 1:
                raise ValueError("History AUC must be in [0,1]")
    return history


def _optional_pr(row):
    value = row.get("val_patient_pr_auc")
    if value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid validation PR-AUC in history")
    return value


def _load_history(row):
    history = _history_rows(legacy._read(Path(row["run"]) / "history.json"))
    best = max(history, key=lambda r: r["val_patient_roc_auc"])
    if not math.isclose(best["val_patient_roc_auc"], row["val_auc"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved validation selection differs from recorded history: " + row["run"])
    return history, best


def overfitting_rows(rows):
    output = []
    for row in rows:
        if row["status"] not in SUCCESS:
            continue
        history, best = _load_history(row)
        record = {key: row[key] for key in ("arm", "weight_decay", "group", "seed", "run")}
        for prefix, source in (("best", best), ("last", history[-1])):
            record.update({prefix + "_epoch": source["epoch"],
                prefix + "_train_eval_loss": source["train_eval_loss"],
                prefix + "_val_loss": source["val_loss"],
                prefix + "_loss_gap": source["val_loss"] - source["train_eval_loss"],
                prefix + "_train_auc": source["train_eval_patient_roc_auc"],
                prefix + "_val_auc": source["val_patient_roc_auc"],
                prefix + "_val_pr_auc": _optional_pr(source),
                prefix + "_auc_gap": source["train_eval_patient_roc_auc"] - source["val_patient_roc_auc"]})
        output.append(record)
    return output


def validation_selection(rows):
    arms = list(dict.fromkeys(row["arm"] for row in rows))
    output = {"rule": SELECTION_RULE, "test_metrics_used": False, "status": "incomplete",
              "selected_arm": None, "selected_weight_decay": None, "candidates": []}
    if len(arms) != 3 or "baseline" not in arms:
        return output
    complete = True
    for arm in arms:
        for group in ("A", "B"):
            selected = [row for row in rows if row["arm"] == arm and row["group"] == group]
            valid = (len(selected) == len(SEEDS) and {row["seed"] for row in selected} == SEEDS
                     and all(row["status"] in SUCCESS and type(row.get("val_auc")) in (int, float)
                             and math.isfinite(row["val_auc"]) and 0 <= row["val_auc"] <= 1
                             for row in selected))
            complete = complete and valid
        group_b = [row for row in rows if row["arm"] == arm and row["group"] == "B"]
        values = [row["val_auc"] for row in group_b if row["status"] in SUCCESS
                  and type(row.get("val_auc")) in (int, float) and math.isfinite(row["val_auc"])]
        output["candidates"].append({"arm": arm, "weight_decay": group_b[0]["weight_decay"] if group_b else None,
            "B_n": len(values), "B_mean_val_auc": statistics.mean(values) if values else None})
    if not complete:
        return output
    winner = min(output["candidates"], key=lambda r: (-r["B_mean_val_auc"],
        r["arm"] != "baseline", r["weight_decay"]))
    output.update(status="complete", selected_arm=winner["arm"], selected_weight_decay=winner["weight_decay"])
    return output


def build_report(study, state):
    rows = metric_rows(study, state)
    return {"rows": rows, "summary": summary_rows(rows), "paired": paired_rows(rows),
            "overfitting": overfitting_rows(rows), "selection": validation_selection(rows)}


def write_report(study, state):
    report = build_report(study, state)
    root = Path(study) / "report"
    for name, key in (("comparison", "rows"), ("summary", "summary"),
                      ("matched_seed_differences", "paired"), ("overfitting", "overfitting")):
        legacy._csv(root / (name + ".csv"), report[key])
    legacy._json(root / "validation_selection.json", report["selection"])
    return report


def print_metrics(study, state):
    report = build_report(study, state)
    print(f"\nWeight-decay study: {study}")
    print(f"Baseline: {state['baseline']['suite']}; actual weight_decay={state['baseline']['weight_decay']:g}")
    columns = ("arm", "weight_decay", "group", "seed", "status", "val_auc", "val_pr_auc", "roc_auc", "pr_auc", "ap", "sensitivity", "specificity")
    print("  ".join(columns))
    for row in report["rows"]:
        print("  ".join(legacy._number(row.get(key)) if key in (*legacy.METRICS, "val_pr_auc")
                        else str(row.get(key, "-")) for key in columns))
        if row.get("error"):
            print("  " + row["error"])
    print("Single seed42: seed variability and seed SD are not estimable.")
    for row in report["summary"]:
        metrics = "; ".join(f"{key}={legacy._number(row[key+'_mean'])}"
                            for key in (*legacy.METRICS, "val_pr_auc"))
        print(f"{row['arm']} wd={row['weight_decay']:g} {row['group']} n={row['n_completed']}/{row['n_requested']}: {metrics}")
    for row in report["paired"]:
        print(f"{row['comparison']} {row['candidate']}/{row['group']} - "
              f"{row['reference']}/{row['reference_group']} seed{row['seed']}: "
              f"delta_val_auc={legacy._number(row['delta_val_auc'])}; "
              f"delta_roc_auc={legacy._number(row['delta_roc_auc'])}; "
              f"delta_pr_auc={legacy._number(row['delta_pr_auc'])}")
    print("Same-definition Focal losses and patient AUC at the selected best epoch and final epoch:")
    for row in report["overfitting"]:
        print(f"{row['arm']} {row['group']} seed{row['seed']} "
              f"best={row['best_epoch']} last={row['last_epoch']} "
              f"best_auc_gap={row['best_auc_gap']:.4f} last_auc_gap={row['last_auc_gap']:.4f} "
              f"best_loss_gap={row['best_loss_gap']:.6f} last_loss_gap={row['last_loss_gap']:.6f}")
    selection = report["selection"]
    if selection["status"] == "complete":
        print(f"Validation-only selected setting: {selection['selected_arm']} "
              f"weight_decay={selection['selected_weight_decay']:g}. "
              "This selection is for validation ROC-AUC; inspect the saved curves and gaps separately.")
    else:
        print("Validation selection pending: all 6 seed42 baseline/candidate records must be complete.")
    print("Selection rule: " + SELECTION_RULE)
    print("Patient test metrics are descriptive; PR-AUC and AP are distinct. "
          "Gaps have no universal cutoff for overfitting. No test scores select parameters or seeds.")
    return report
