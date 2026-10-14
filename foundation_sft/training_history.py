"""Plot saved epoch records without rerunning training or image preprocessing.

Missing validation loss in older histories remains unknown. These plots show
the recorded epoch values directly, without smoothing or filling missing values.
"""

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import uuid


def _number(value, name, *, optional=False, auc=False):
    if value is None and optional:
        return None
    try:
        valid = (type(value) in (int, float) and math.isfinite(value)
                 and (0 <= value <= 1 if auc else value >= 0))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"History {name} must be a finite {'[0, 1] value' if auc else 'nonnegative number'}")
    return value


def _records(history):
    if not isinstance(history, list) or not history:
        raise ValueError("Training history must be a nonempty list of epoch records")
    result = []
    previous = 0
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("Each training history record must be an object")
        epoch = record.get("epoch")
        if type(epoch) is not int or epoch <= previous:
            raise ValueError("History epochs must be positive, unique and strictly increasing")
        previous = epoch
        result.append({
            "epoch": epoch,
            "train_loss": _number(record.get("train_loss"), "train_loss"),
            "val_loss": _number(record.get("val_loss"), "val_loss", optional=True),
            "val_patient_roc_auc": _number(record.get("val_patient_roc_auc"),
                                           "val_patient_roc_auc", auc=True),
            "val_patient_pr_auc": _number(record.get("val_patient_pr_auc"),
                                          "val_patient_pr_auc", auc=True, optional=True),
        })
    return result


def _summary(records, best):
    def value(number):
        return "unknown" if number is None else f"{number:.8g}"

    lines = []
    for label, record in (("first", records[0]), ("best", best), ("last", records[-1])):
        lines.append(f"{label} epoch={record['epoch']} "
                     f"train_loss={value(record['train_loss'])} "
                     f"val_loss={value(record['val_loss'])} "
                     f"valROC={value(record['val_patient_roc_auc'])} "
                     f"valPR={value(record['val_patient_pr_auc'])}")
    missing = sum(record["val_loss"] is None for record in records)
    if missing:
        lines.append(f"Validation loss was not recorded for {missing}/{len(records)} epochs; "
                     "those values remain unknown.")
    return "\n".join(lines)


def save_history_plots(history, output_dir):
    """Save two PNGs from actual records, refusing to replace existing plots.

    Return their paths, the first best validation-ROC epoch, and a text summary.
    The caller may use a populated run directory; only the two plot filenames
    must be absent. Missing validation loss or PR-AUC is shown as missing data.
    """
    records = _records(history)
    best = max(records, key=lambda row: row["val_patient_roc_auc"])
    output = Path(output_dir).expanduser().resolve()
    paths = {"training_loss": output / "training_loss.png",
             "validation_auc": output / "validation_auc.png"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Training plots already exist; choose a new output directory")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in records]
    encoded = {}
    for kind in paths:
        figure, axis = plt.subplots(figsize=(8, 5))
        try:
            if kind == "training_loss":
                axis.plot(epochs, [row["train_loss"] for row in records],
                          marker="o", label="Train loss")
                missing = sum(row["val_loss"] is None for row in records)
                if missing < len(records):
                    axis.plot(epochs, [math.nan if row["val_loss"] is None else row["val_loss"]
                                       for row in records], marker="o", label="Validation loss")
                if missing:
                    figure.text(0.5, 0.015,
                                f"Validation loss not recorded for {missing}/{len(records)} epochs; "
                                "missing values are not imputed.",
                                ha="center", fontsize=9)
                axis.set(title="Training and validation loss", ylabel="Loss")
            else:
                axis.plot(epochs, [row["val_patient_roc_auc"] for row in records],
                          marker="o", label="Validation patient ROC-AUC")
                missing = sum(row["val_patient_pr_auc"] is None for row in records)
                if missing < len(records):
                    axis.plot(epochs, [math.nan if row["val_patient_pr_auc"] is None
                                       else row["val_patient_pr_auc"] for row in records],
                              marker="o", label="Validation patient PR-AUC")
                if missing:
                    figure.text(0.5, 0.015,
                                f"Validation PR-AUC not recorded for {missing}/{len(records)} epochs; "
                                "missing values are not imputed.", ha="center", fontsize=9)
                axis.scatter([best["epoch"]], [best["val_patient_roc_auc"]],
                             marker="*", s=160, color="black", zorder=4)
                axis.annotate(f"Best validation ROC: epoch {best['epoch']}",
                              (best["epoch"], best["val_patient_roc_auc"]),
                              xytext=(0, -18), textcoords="offset points", ha="center", fontsize=9)
                axis.set(title="Validation patient AUC", ylabel="AUC", ylim=(-0.02, 1.08))
            axis.axvline(best["epoch"], color="black", linestyle="--", alpha=0.4,
                         label=f"Best validation ROC epoch {best['epoch']}")
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout(rect=(0, 0.05, 1, 1))
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=160)
            encoded[kind] = buffer.getvalue()
        finally:
            plt.close(figure)

    output.mkdir(parents=True, exist_ok=True)
    for kind, path in paths.items():
        with path.open("xb") as handle:
            handle.write(encoded[kind])
    return {**{name: str(path) for name, path in paths.items()},
            "best_epoch": best["epoch"], "summary": _summary(records, best)}


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing required run or log path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _suite_runs(suite):
    manifest = _read_json(suite / "suite.json")
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, list):
        raise ValueError("suite.json must contain a jobs list")
    selected, ids = [], set()
    for job in jobs:
        if (not isinstance(job, dict) or not isinstance(job.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]) or job["id"] in ids):
            raise ValueError("Suite jobs must have unique, safe IDs")
        ids.add(job["id"])
        if job.get("status") == "completed":
            run = _path(job.get("run_dir"), suite)
            if not (run / "history.json").is_file():
                raise ValueError(f"Completed job has no history.json: {job['id']}")
            selected.append((job["id"], run, "completed"))
        elif job.get("status") == "failed" and job.get("train_log"):
            log = _path(job["train_log"], suite)
            if not log.is_file():
                print(f"[skip] {job['id']}: training log is unavailable.", flush=True)
                continue
            with log.open(encoding="utf-8", errors="replace") as handle:
                directories = {_path(match[1].strip(), suite)
                               for line in handle
                               if (match := re.fullmatch(r"Run directory:\s*(.+)", line.strip()))}
            if len(directories) != 1:
                print(f"[skip] {job['id']}: training log has no unique Run directory.", flush=True)
                continue
            run = directories.pop()
            if not (run / "history.json").is_file():
                print(f"[skip] {job['id']}: no completed-epoch history is available.", flush=True)
                continue
            selected.append((job["id"], run, "failed; completed epochs only"))
    if not selected:
        raise ValueError("No completed or unambiguously logged failed runs have training history")
    return selected


def _new_output(base):
    if not base.exists():
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base.with_name(f"{base.name}_{stamp}_{uuid.uuid4().hex[:8]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Run directory containing history.json")
    source.add_argument("--suite", help="Suite directory containing suite.json")
    parser.add_argument("--output", help="New output directory; existing directories are not overwritten")
    args = parser.parse_args(argv)
    source_dir = Path(args.run or args.suite).expanduser().resolve()
    runs = [(None, source_dir, "recorded epochs")] if args.run else _suite_runs(source_dir)
    # Validate every selected history before producing any suite output.
    loaded = [(job_id, run, status, _read_json(run / "history.json"))
              for job_id, run, status in runs]
    for _, _, _, history in loaded:
        _records(history)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError("--output must name a new directory")
    else:
        output = _new_output(source_dir / "training_curves")
    for job_id, run, status, history in loaded:
        result = save_history_plots(history, output / job_id if job_id else output)
        print(f"[{job_id or 'run'}] {status}: {run}\n{result['summary']}\n"
              f"training_loss: {result['training_loss']}\n"
              f"validation_auc: {result['validation_auc']}", flush=True)
    print(f"Training curves directory: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
