"""Single-seed DropPath comparisons at a fixed, validation-selected WD setting."""

import math
from pathlib import Path

from . import distillation_suite as legacy
from . import random_student_suite as random_suite
from . import weight_decay_report as history_report


SELECTION_RULE = (
    "Maximize validation patient ROC-AUC of B at seed42, with weight_decay=0.01; "
    "exact ties prefer the existing DropPath=0 baseline. "
    "Use the same setting for paired A. Test scores are not used."
)


def _arms(study, state):
    jobs = {name: {"job": entry["job"], "result": entry["result"], "check": entry["check"],
                   "status": "completed", "source": "baseline"}
            for name, entry in state["baseline"]["jobs"].items()}
    yield "baseline", 0., {"jobs": jobs}
    for arm, record in state["arms"].items():
        suite = Path(record["suite"])
        if not suite.is_absolute():
            suite = Path(study) / suite
        yield arm, record["drop_path_rate"], legacy._read(suite / "suite.json")


def metric_rows(study, state):
    result = []
    for arm, rate, batch in _arms(study, state):
        for row in random_suite.metric_rows(batch):
            value = {"arm": arm, "drop_path_rate": rate, "weight_decay": .01,
                     **row, "val_pr_auc": None}
            if row["status"] in history_report.SUCCESS:
                _, best = history_report._load_history(row)
                value["val_pr_auc"] = history_report._optional_pr(best)
            result.append(value)
    return result


def summary_rows(rows):
    rates = {row["arm"]: row["drop_path_rate"] for row in rows}
    return [{"drop_path_rate": rates[row["arm"]], **row}
            for row in history_report.summary_rows(rows)]


def paired_rows(rows):
    rates = {row["arm"]: row["drop_path_rate"] for row in rows}
    result = []
    for row in history_report.paired_rows(rows):
        row["comparison"] = "drop_path_gain" if row["comparison"] == "weight_decay_gain" else row["comparison"]
        result.append({"drop_path_rate": rates[row["candidate"]],
                       "reference_drop_path_rate": rates[row["reference"]], **row})
    return result


def overfitting_rows(rows):
    rates = {row["arm"]: row["drop_path_rate"] for row in rows}
    return [{"drop_path_rate": rates[row["arm"]], **row}
            for row in history_report.overfitting_rows(rows)]


def validation_selection(rows):
    arms = list(dict.fromkeys(row["arm"] for row in rows))
    result = {"rule": SELECTION_RULE, "seed": 42, "weight_decay": .01,
              "test_metrics_used": False, "status": "incomplete", "selected_arm": None,
              "selected_drop_path_rate": None, "candidates": []}
    if len(arms) != 2 or "baseline" not in arms:
        return result
    for arm in arms:
        for group in ("A", "B"):
            found = [row for row in rows if row["arm"] == arm and row["group"] == group]
            if (len(found) != 1 or found[0]["seed"] != 42
                    or found[0]["status"] not in history_report.SUCCESS
                    or type(found[0].get("val_auc")) not in (float, int)
                    or not math.isfinite(found[0]["val_auc"]) or not 0 <= found[0]["val_auc"] <= 1):
                return result
        group_b = next(row for row in rows if row["arm"] == arm and row["group"] == "B")
        result["candidates"].append({"arm": arm, "drop_path_rate": group_b["drop_path_rate"],
                                     "B_val_auc": group_b["val_auc"]})
    winner = min(result["candidates"], key=lambda row: (-row["B_val_auc"], row["arm"] != "baseline"))
    result.update(status="complete", selected_arm=winner["arm"], selected_drop_path_rate=winner["drop_path_rate"])
    return result


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
    print(f"\nDropPath study: {study}")
    print(f"Source WD study: {state['baseline']['study']}; fixed weight_decay=0.01; seed42")
    print("DropPath=0.1 means a native linear 12-block schedule, maximum probability 0.1, active during training.")
    columns = ("arm", "drop_path_rate", "weight_decay", "group", "seed", "status", "val_auc", "val_pr_auc",
               "roc_auc", "pr_auc", "ap", "sensitivity", "specificity")
    print("  ".join(columns))
    for row in report["rows"]:
        print("  ".join(legacy._number(row.get(key)) if key in (*legacy.METRICS, "val_pr_auc")
                        else str(row.get(key, "-")) for key in columns))
        if row.get("error"):
            print("  " + row["error"])
    print("Single seed42: seed variability and seed SD are not estimable.")
    for row in report["paired"]:
        print(f"{row['comparison']} {row['candidate']}/{row['group']} - "
              f"{row['reference']}/{row['reference_group']} seed42: "
              f"delta_val_auc={legacy._number(row['delta_val_auc'])}; "
              f"delta_roc_auc={legacy._number(row['delta_roc_auc'])}; "
              f"delta_pr_auc={legacy._number(row['delta_pr_auc'])}")
    print("Same-definition Focal losses and patient AUC at selected best and final epochs:")
    for row in report["overfitting"]:
        print(f"{row['arm']} {row['group']} seed42 best={row['best_epoch']} last={row['last_epoch']} "
              f"best_auc_gap={row['best_auc_gap']:.4f} last_auc_gap={row['last_auc_gap']:.4f} "
              f"best_loss_gap={row['best_loss_gap']:.6f} last_loss_gap={row['last_loss_gap']:.6f}")
    selected = report["selection"]
    if selected["status"] == "complete":
        print(f"Validation-only selected setting: {selected['selected_arm']} "
              f"DropPath={selected['selected_drop_path_rate']:g}; weight_decay=0.01.")
    else:
        print("Validation selection pending: all 4 baseline/candidate records must be complete.")
    print("Selection rule: " + SELECTION_RULE)
    print("Patient test metrics are descriptive; PR-AUC and AP are distinct. "
          "Inspect full histories: gaps at different best epochs alone do not establish reduced overfitting.")
    return report
