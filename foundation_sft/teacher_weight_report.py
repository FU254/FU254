"""Read-only validation search reports and comparable six-group learning curves.

Learning-rate pilots expose validation metrics only. The six formal groups are
reported separately after each initialization's learning rate has been frozen.
Reporting imports only the standard library; Matplotlib is loaded by plotting.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

from . import distillation_suite as records


SUCCESS = {"completed", "resumed", "reused"}
GROUPS = ("R-E", "R-C", "R-R", "P-E", "P-C", "P-R")
TEACHERS = ("rad_dino", "chexfound", "chexzero")
METRICS = ("val_auc", "val_pr_auc", "roc_auc", "pr_auc", "ap", "sensitivity", "specificity")
SELECTION_RULE = (
    "For each initialization, maximize the equal-weight pilot's best validation patient ROC-AUC; "
    "exact ties prefer the smaller backbone learning rate. Pilots never evaluate test."
)
INITIALIZATION_NOTE = (
    "Random versus ImageNet comparisons include independently validation-selected learning-rate recipes; "
    "they do not isolate initialization alone."
)


def _number(value, label, *, optional=False, auc=False, nonnegative=True):
    if value is None and optional:
        return None
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (auc and not 0 <= value <= 1) or (nonnegative and value < 0)):
        raise ValueError("Invalid saved " + label)
    return value


def _run_path(study, result):
    value = result.get("run")
    if not isinstance(value, str) or not value:
        raise ValueError("A completed teacher-weight record must name its run")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(study) / path).resolve()


def _history(run, val_auc):
    history = records._read(Path(run) / "history.json")
    if not isinstance(history, list) or not history:
        raise ValueError("Completed teacher-weight run needs a nonempty history.json")
    previous = 0
    for row in history:
        if not isinstance(row, dict):
            raise ValueError("History epoch records must be objects")
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch <= previous:
            raise ValueError("History epochs must be positive and strictly increasing")
        previous = epoch
        for key in ("train_eval_loss", "val_loss", "train_eval_patient_roc_auc", "val_patient_roc_auc"):
            _number(row.get(key), "history " + key, auc=key.endswith("roc_auc"))
        for key in ("train_eval_patient_pr_auc", "val_patient_pr_auc"):
            _number(row.get(key), "history " + key, optional=True, auc=True)
    best = max(history, key=lambda row: row["val_patient_roc_auc"])
    if not math.isclose(best["val_patient_roc_auc"], val_auc, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved validation selection differs from recorded history: " + str(run))
    return history, best


def _base_row(identifier, entry, *, phase):
    job = entry.get("job", {})
    config = job.get("config", {})
    group = job.get("group", identifier if identifier in GROUPS else None)
    initialization = job.get("student_initialization", config.get("student_initialization"))
    if initialization is None and group in GROUPS:
        initialization = "random" if group.startswith("R-") else "imagenet"
    method = job.get("method", config.get("method"))
    if method is None and group in GROUPS:
        method = {"E": "equal", "C": "ca", "R": "rl"}[group[-1]]
    return {"id": identifier, "group": group, "student_initialization": initialization,
        "method": method, "phase": phase, "seed": job.get("seed", config.get("seed", 42)),
        "backbone_lr": config.get("training", {}).get("backbone_lr"),
        "status": entry.get("status", "pending"), "val_auc": None, "val_pr_auc": None,
        "run": None, "error": entry.get("error")}


def _completed(study, entry, row, *, pilot):
    result = entry.get("result")
    if not isinstance(result, dict):
        raise ValueError("Completed teacher-weight job lacks a result")
    if result.get("status") != "completed":
        raise ValueError("Completed teacher-weight record has an incomplete result")
    if pilot and (result.get("test_patient") is not None or result.get("roc_auc_ci") is not None):
        raise ValueError("A learning-rate pilot must not contain test metrics")
    phase = result.get("phase")
    if phase is not None and phase != ("pilot" if pilot else "main"):
        raise ValueError("Completed teacher-weight result has the wrong phase")
    val_auc = _number(result.get("best_val_patient_roc_auc"), "best validation ROC-AUC", auc=True)
    run = _run_path(study, result)
    history, best = _history(run, val_auc)
    row.update(run=str(run), val_auc=val_auc, val_pr_auc=best.get("val_patient_pr_auc"))
    return result, history, best


def pilot_rows(study, state):
    result = []
    for identifier, entry in state.get("pilots", {}).items():
        row = _base_row(identifier, entry, phase="pilot")
        if row["status"] in SUCCESS:
            _completed(study, entry, row, pilot=True)
        result.append(row)
    return result


def metric_rows(study, state):
    result = []
    for group in GROUPS:
        entry = state.get("main", {}).get(group, {})
        row = _base_row(group, entry, phase="main")
        row.update({key: None for key in ("roc_auc", "roc_ci_lower", "roc_ci_upper", "pr_auc", "ap",
                                         "sensitivity", "specificity")})
        if row["status"] in SUCCESS:
            receipt, _, _ = _completed(study, entry, row, pilot=False)
            patient = receipt.get("test_patient")
            if not isinstance(patient, dict):
                raise ValueError("A completed formal group needs saved test patient metrics")
            for target, source in (("roc_auc", "roc_auc"), ("pr_auc", "pr_auc"), ("ap", "average_precision"),
                                   ("sensitivity", "sensitivity"), ("specificity", "specificity")):
                row[target] = _number(patient.get(source), "test " + source, optional=True, auc=True)
            interval = receipt.get("roc_auc_ci")
            if isinstance(interval, dict):
                row.update(roc_ci_lower=interval.get("lower"), roc_ci_upper=interval.get("upper"))
            elif isinstance(interval, (list, tuple)) and len(interval) == 2:
                row.update(roc_ci_lower=interval[0], roc_ci_upper=interval[1])
            for key in ("roc_ci_lower", "roc_ci_upper"):
                _number(row[key], key, optional=True, auc=True)
            if row["roc_ci_lower"] is not None and row["roc_ci_upper"] is not None:
                if row["roc_ci_lower"] > row["roc_ci_upper"]:
                    raise ValueError("Invalid saved ROC-AUC confidence interval")
        result.append(row)
    return result


def overfitting_rows(rows):
    output = []
    for row in rows:
        if row["status"] not in SUCCESS:
            continue
        history, best = _history(row["run"], row["val_auc"])
        result = {key: row[key] for key in ("group", "student_initialization", "method", "seed", "backbone_lr", "run")}
        for prefix, record in (("best", best), ("last", history[-1])):
            result.update({prefix + "_epoch": record["epoch"],
                prefix + "_train_eval_loss": record["train_eval_loss"],
                prefix + "_val_loss": record["val_loss"],
                prefix + "_loss_gap": record["val_loss"] - record["train_eval_loss"],
                prefix + "_train_auc": record["train_eval_patient_roc_auc"],
                prefix + "_val_auc": record["val_patient_roc_auc"],
                prefix + "_auc_gap": record["train_eval_patient_roc_auc"] - record["val_patient_roc_auc"],
                prefix + "_train_pr_auc": record.get("train_eval_patient_pr_auc"),
                prefix + "_val_pr_auc": record.get("val_patient_pr_auc")})
        output.append(result)
    return output


def paired_rows(rows):
    usable = {row["group"]: row for row in rows if row["status"] in SUCCESS}
    comparisons = [(prefix + "-C", prefix + "-E", "ca_minus_equal") for prefix in ("R", "P")]
    comparisons += [(prefix + "-R", prefix + "-E", "rl_minus_equal") for prefix in ("R", "P")]
    comparisons += [(prefix + "-R", prefix + "-C", "rl_minus_ca") for prefix in ("R", "P")]
    comparisons += [("P-" + method, "R-" + method, "imagenet_recipe_minus_random_recipe") for method in ("E", "C", "R")]
    result = []
    for candidate, reference, name in comparisons:
        if candidate not in usable or reference not in usable:
            continue
        first, second = usable[candidate], usable[reference]
        row = {"comparison": name, "candidate": candidate, "reference": reference, "seed": 42,
            "candidate_backbone_lr": first["backbone_lr"], "reference_backbone_lr": second["backbone_lr"]}
        for metric in METRICS:
            a, b = first.get(metric), second.get(metric)
            row["delta_" + metric] = a-b if a is not None and b is not None else None
        result.append(row)
    return result


def build_report(study, state):
    pilots, rows = pilot_rows(study, state), metric_rows(study, state)
    return {"pilots": pilots, "rows": rows, "overfitting": overfitting_rows(rows),
        "paired": paired_rows(rows), "selection": deepcopy(state.get("selection", {})),
        "selection_rule": SELECTION_RULE, "initialization_note": INITIALIZATION_NOTE}


def write_report(study, state):
    report = build_report(study, state)
    root = Path(study) / "report"
    for filename, key in (("lr_pilots", "pilots"), ("comparison", "rows"),
                          ("overfitting", "overfitting"), ("matched_differences", "paired")):
        records._csv(root / (filename + ".csv"), report[key])
    records._json(root / "learning_rate_selection.json", {
        "rule": SELECTION_RULE, "selection": report["selection"], "test_metrics_used": False})
    return report


def print_metrics(study, state):
    report = build_report(study, state)
    print(f"\nTeacher-weight study: {study}")
    print("Learning-rate pilots: validation only; no test scores are evaluated or used for selection.")
    pilot_columns = ("id", "student_initialization", "backbone_lr", "status", "val_auc", "val_pr_auc")
    print("  ".join(pilot_columns))
    for row in report["pilots"]:
        print("  ".join(records._number(row.get(key)) if key in ("val_auc", "val_pr_auc")
                        else str(row.get(key, "-")) for key in pilot_columns))
        if row.get("error"):
            print("  " + row["error"])
    print("Frozen learning-rate selections: " + json.dumps(report["selection"], ensure_ascii=False, sort_keys=True))
    columns = ("group", "student_initialization", "method", "seed", "backbone_lr", "status", "val_auc",
               "val_pr_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper", "pr_auc", "ap", "sensitivity", "specificity")
    print("Formal six-group patient metrics:")
    print("  ".join(columns))
    for row in report["rows"]:
        print("  ".join(records._number(row.get(key)) if key in (*METRICS, "roc_ci_lower", "roc_ci_upper")
                        else str(row.get(key, "-")) for key in columns))
        if row.get("error"):
            print("  " + row["error"])
    print("Same-definition train-eval/validation Focal losses and patient AUC gaps:")
    for row in report["overfitting"]:
        print(f"{row['group']} best={row['best_epoch']} last={row['last_epoch']} "
            f"best_auc_gap={row['best_auc_gap']:.4f} last_auc_gap={row['last_auc_gap']:.4f} "
            f"best_loss_gap={row['best_loss_gap']:.6f} last_loss_gap={row['last_loss_gap']:.6f}")
    for row in report["paired"]:
        print(f"{row['comparison']} {row['candidate']} - {row['reference']}: "
            f"delta_val_auc={records._number(row['delta_val_auc'])}; "
            f"delta_roc_auc={records._number(row['delta_roc_auc'])}; "
            f"delta_pr_auc={records._number(row['delta_pr_auc'])}")
    print(SELECTION_RULE)
    print(INITIALIZATION_NOTE)
    print("Single seed42: seed SD and seed variability are not estimable. PR-AUC and AP are distinct.")
    print("Smaller train/validation gaps alone do not establish reduced overfitting; inspect complete curves and validation performance.")
    return report


def _optional_series(history, key, *, auc=False, signed=False):
    return [math.nan if row.get(key) is None else _number(row[key], key, auc=auc, nonnegative=not signed)
            for row in history]


def plot_curves(study, state):
    """Draw the six formal histories without training or overwriting old figures."""
    report = build_report(study, state)
    available = {row["group"]: row for row in report["rows"] if row["status"] in SUCCESS}
    if not available:
        raise ValueError("No completed formal group is available for learning curves")
    histories = {group: _history(row["run"], row["val_auc"]) for group, row in available.items()}
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    root = Path(study) / "report"
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="curves_", dir=root))
    paths = {"directory": str(output)}

    def six_panel(name, painter, ylabel, *, bounds=None):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        try:
            for index, group in enumerate(GROUPS):
                axis = axes[index // 3, index % 3]
                row = report["rows"][index]
                axis.set(title=f"{group} | backbone LR={row['backbone_lr']}", xlabel="Epoch", ylabel=ylabel)
                axis.grid(alpha=.25)
                if bounds:
                    axis.set_ylim(*bounds)
                if group not in histories:
                    axis.text(.5, .5, "No completed history\n" + row["status"],
                              ha="center", va="center", transform=axis.transAxes)
                    continue
                history, best = histories[group]
                epochs = [record["epoch"] for record in history]
                painter(axis, epochs, history)
                axis.axvline(best["epoch"], color="black", linestyle="--", alpha=.5,
                             label=f"Best val ROC epoch {best['epoch']}")
                axis.legend(fontsize=8)
            figure.tight_layout()
            path = output / (name + ".png")
            figure.savefig(path, dpi=160)
            paths[name] = str(path)
        finally:
            plt.close(figure)

    def paired(axis, epochs, history, train, val, *, optional=False):
        for key, label in ((train, "Train (eval)"), (val, "Validation")):
            series = _optional_series(history, key, auc="auc" in key) if optional else [record[key] for record in history]
            if any(math.isfinite(value) for value in series):
                axis.plot(epochs, series, label=label)
            else:
                axis.text(.02, .05 if label.startswith("Train") else .12, label + " not recorded",
                          fontsize=8, transform=axis.transAxes)

    six_panel("loss_curves", lambda a, e, h: paired(a, e, h, "train_eval_loss", "val_loss"), "Focal loss")
    six_panel("roc_auc_curves", lambda a, e, h: paired(a, e, h, "train_eval_patient_roc_auc", "val_patient_roc_auc"),
              "Patient ROC-AUC", bounds=(0, 1))
    six_panel("pr_auc_curves", lambda a, e, h: paired(a, e, h, "train_eval_patient_pr_auc", "val_patient_pr_auc", optional=True),
              "Patient PR-AUC", bounds=(0, 1))

    def optimization(axis, epochs, history):
        for key, label in (("train_loss", "Total"), ("train_hard_loss", "Hard"), ("train_kd_loss", "Weighted KD"),
                           ("train_soft_loss", "Soft (raw)"), ("train_global_loss", "Global (raw)"),
                           ("train_local_loss", "Local (raw)")):
            series = _optional_series(history, key)
            if any(math.isfinite(value) for value in series):
                axis.plot(epochs, series, label=label)
        axis.text(.02, .02, "Optimization losses; not validation Focal", fontsize=7, transform=axis.transAxes)
    six_panel("optimization_losses", optimization, "Training objective")

    def weights(axis, epochs, history):
        present = False
        for name in TEACHERS:
            series = []
            for record in history:
                mapping = record.get("teacher_weights")
                value = mapping.get(name) if isinstance(mapping, dict) else None
                series.append(math.nan if value is None else _number(value, "teacher weight " + name, auc=True))
            if any(math.isfinite(value) for value in series):
                axis.plot(epochs, series, label=name)
                present = True
        if not present:
            axis.text(.5, .5, "Teacher weights not recorded", ha="center", transform=axis.transAxes)
    six_panel("teacher_weight_curves", weights, "Mean teacher weight", bounds=(0, 1))

    def entropy(axis, epochs, history):
        series = _optional_series(history, "weight_entropy")
        if any(math.isfinite(value) for value in series):
            axis.plot(epochs, series, label="Teacher-weight entropy")
        else:
            axis.text(.5, .5, "Weight entropy not recorded", ha="center", transform=axis.transAxes)
    six_panel("weight_entropy_curves", entropy, "Weight entropy")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="row")
    try:
        for index, group in enumerate(("R-R", "P-R")):
            for column, key in enumerate(("reward", "policy_loss")):
                axis = axes[index, column]
                axis.set(title=f"{group} | {key}", xlabel="Epoch", ylabel=key)
                axis.grid(alpha=.25)
                if group not in histories:
                    axis.text(.5, .5, "No completed RL history", ha="center", transform=axis.transAxes)
                    continue
                history, best = histories[group]
                series = _optional_series(history, key, signed=True)
                if any(math.isfinite(value) for value in series):
                    axis.plot([row["epoch"] for row in history], series, label=key)
                else:
                    axis.text(.5, .5, key + " not recorded", ha="center", transform=axis.transAxes)
                axis.axvline(best["epoch"], color="black", linestyle="--", alpha=.5, label="Best validation ROC")
                axis.legend(fontsize=8)
        figure.tight_layout()
        path = output / "rl_diagnostics.png"
        figure.savefig(path, dpi=160)
        paths["rl_diagnostics"] = str(path)
    finally:
        plt.close(figure)
    records._json(output / "plot_metadata.json", {
        "available_groups": list(available), "missing_groups": [group for group in GROUPS if group not in available],
        "loss_fields": ["train_eval_loss", "val_loss"], "loss_definition": "Same-definition label-only Focal in eval mode without random augmentation",
        "roc_fields": ["train_eval_patient_roc_auc", "val_patient_roc_auc"],
        "pr_fields": ["train_eval_patient_pr_auc", "val_patient_pr_auc"],
        "missing_diagnostics": "Unknown values are left missing; warmup or absent policy values are never replaced by zero.",
        "initialization_note": INITIALIZATION_NOTE, "selection_rule": SELECTION_RULE})
    return paths
