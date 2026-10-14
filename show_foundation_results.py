#!/usr/bin/env python3
"""Read saved foundation-model results across batches, without starting training.

Uses only the standard library and supports Python 3.7+. Original suite records,
FP32 recoveries, standalone training and independent evaluations remain separate.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sys


FIELDS = (
    "batch", "kind", "model", "mode", "seed", "status", "recorded_status",
    "precision", "best_val_patient_roc_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper",
    "confidence_level", "pr_auc", "average_precision", "sensitivity", "specificity",
    "cohort_hash", "source_hash", "split_hash", "normalization_status", "run_dir", "notes",
)


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Saved JSON is not an object")
    return value


def _optional_json(path):
    try:
        return _json(path)
    except (OSError, ValueError, TypeError):
        return {}


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _label(path, project):
    try:
        return str(path.relative_to(project))
    except ValueError:
        return str(path)


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _row(batch, kind, job, status, run=None):
    row = dict.fromkeys(FIELDS, "")
    row.update(batch=batch, kind=kind, model=job.get("model", "?"), mode=job.get("mode", "?"),
               seed=job.get("seed", "?"), status=status, recorded_status=status,
               run_dir=str(run) if run else "")
    if status not in ("completed", "recovered", "evaluated") or run is None:
        if status in ("completed", "recovered", "evaluated"):
            row.update(status="missing_artifacts", notes="Recorded completion has no run directory")
        return row
    try:
        config = _json(run / "config.json")
        if any(config.get(key) != job.get(key) for key in ("model", "mode", "seed")):
            raise ValueError("Run identity differs from saved job")
        metrics = _json(run / ("metrics.json" if kind == "evaluation" else "test/metrics.json"))
        patient = metrics.get("patient")
        if not isinstance(patient, dict):
            raise ValueError("Patient metrics are missing")
        for name in ("roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity"):
            row[name] = _number(patient.get(name))
        selection = _optional_json(run / "selection.json")
        row["best_val_patient_roc_auc"] = _number(selection.get("best_val_patient_roc_auc"))
        bootstrap = metrics.get("bootstrap", {})
        confidence = _number(bootstrap.get("confidence_level"))
        interval = bootstrap.get("patient", {}).get("roc_auc", {})
        if confidence is not None and 0 < confidence < 1:
            row.update(confidence_level=confidence, roc_ci_lower=_number(interval.get("lower")),
                       roc_ci_upper=_number(interval.get("upper")))
        resume = _optional_json(run / "resume.json")
        amp = config.get("training", {}).get("amp", False)
        row["precision"] = "AMP" if amp and str(config.get("device", "")).startswith("cuda") else "FP32"
        if resume.get("precision_changed") and resume.get("source_amp") and not resume.get("target_amp"):
            row["precision"] = "AMP->FP32"
        environment = _optional_json(run / "environment.json")
        split = _optional_json(run / "split_metadata.json")
        row["cohort_hash"] = split.get("_cohort_hash", "")
        row["source_hash"] = split.get("_source_hash", "")
        # The legacy environment field named cohort_hash stores _manifest_hash.
        row["split_hash"] = environment.get("cohort_hash") or split.get("_manifest_hash", "")
        for name in ("cohort_hash", "source_hash", "split_hash"):
            if not isinstance(row[name], str):
                row[name] = ""
        row["normalization_status"] = config.get("preprocessing_reference", {}).get("status", "")
    except (OSError, ValueError, TypeError, AttributeError):
        for name in ("best_val_patient_roc_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper",
                     "confidence_level", "pr_auc", "average_precision", "sensitivity", "specificity"):
            row[name] = ""
        row.update(status="missing_artifacts", notes="Saved metrics/config are unreadable or inconsistent")
    return row


def _logged_run(log, base):
    if log is None:
        return None
    try:
        paths = set()
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.fullmatch(r"Run directory:\s*(.+)", line.strip())
            if match:
                paths.add(_path(match.group(1), base))
        return next(iter(paths)) if len(paths) == 1 else None
    except OSError:
        return None


def _missing_planned_results(project, rows, diagnostics):
    covered = {(row["model"], row["mode"], row["seed"]) for row in rows if row["kind"] != "evaluation"}
    planned = set()
    missing = []
    for config_file in sorted((project / "foundation_sft/configs").glob("experiments*.json")):
        try:
            config = _json(config_file)
            for job in config.get("jobs", []):
                for mode in config.get("modes", ["linear_probe", "lora"]):
                    for seed in config.get("seeds", [42]):
                        identity = (job["model"], mode, seed)
                        if identity in covered or identity in planned:
                            continue
                        planned.add(identity)
                        row = _row("Configured experiments without a saved training record", "planned",
                                   {"model": identity[0], "mode": mode, "seed": seed}, "no_saved_run")
                        row["notes"] = "No matching saved training record under the scanned roots"
                        missing.append(row)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            diagnostics.append(_label(config_file, project) + ": unreadable experiment plan")
    return missing


def collect_results(project, roots=None, include_checks=False):
    """Inventory records; never average or select runs by test performance."""
    project = Path(project).expanduser().resolve()
    roots = ([Path(value).expanduser().resolve() for value in roots] if roots is not None
             else sorted(path for path in (project / "runs").glob("foundation_sft*") if path.is_dir()))
    rows, reserved, seen_suites, seen_runs = [], set(), set(), set()
    info = {"suites": 0, "skipped_check_suites": 0, "diagnostics": []}
    for root in roots:
        for manifest in sorted(root.rglob("suite.json")):
            manifest = manifest.resolve()
            if manifest in seen_suites:
                continue
            seen_suites.add(manifest)
            suite, batch = manifest.parent, _label(manifest.parent, project)
            info["suites"] += 1
            try:
                saved = _json(manifest)
                jobs = saved.get("jobs")
                if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
                    raise ValueError("Invalid saved jobs")
            except (OSError, ValueError):
                info["diagnostics"].append(batch + ": unreadable suite record")
                continue
            base = _path(saved.get("project_dir"), project) or project
            for job in jobs:
                run = (_path(job.get("run_dir"), suite)
                       or _logged_run(_path(job.get("train_log"), base), base))
                if run:
                    identity = _optional_json(run / "config.json")
                    if all(key in job and identity.get(key) == job[key] for key in ("model", "mode", "seed")):
                        reserved.add(run)
            trained = any(job.get("train_status") or job.get("run_dir") for job in jobs)
            if include_checks or trained:
                for job in jobs:
                    run = (_path(job.get("run_dir"), suite)
                           or _logged_run(_path(job.get("train_log"), base), base))
                    status = "running" if job.get("train_status") == "running" else job.get("status", "unknown")
                    rows.append(_row(batch, "suite", job, status, run))
            else:
                info["skipped_check_suites"] += 1
            for recovery_file in sorted(suite.glob("recovery_*/recovery_manifest.json")):
                recovery_batch = _label(recovery_file.parent, project)
                try:
                    recovery = _json(recovery_file)
                    matches = [job for job in jobs if job.get("id") == recovery.get("job_id")]
                    if _path(recovery.get("suite"), base) != suite or len(matches) != 1:
                        raise ValueError("Recovery mapping differs from original suite")
                    status = recovery.get("status", "unknown")
                    run = _path(recovery.get("recovered_run_dir"), base)
                    rows.append(_row(recovery_batch, "recovery", matches[0],
                                     "recovered" if status == "completed" else status, run))
                    candidates = [_path(recovery.get(name), base)
                                  for name in ("source_run_dir", "recovered_run_dir")]
                    candidates.append(_logged_run(recovery_file.parent / "train.log", base))
                    for path in candidates:
                        if path:
                            identity = _optional_json(path / "config.json")
                            if all(identity.get(key) == matches[0].get(key) for key in ("model", "mode", "seed")):
                                reserved.add(path)
                except (OSError, ValueError):
                    info["diagnostics"].append(recovery_batch + ": unreadable or mismatched recovery record")
    for root in roots:
        for config_file in sorted(root.rglob("config.json")):
            run = config_file.parent.resolve()
            if run in reserved or run in seen_runs:
                continue
            seen_runs.add(run)
            if not ((run / "test/metrics.json").is_file() or (run / "metrics.json").is_file()
                    or (run / "model_metadata.json").is_file()):
                continue
            try:
                config = _json(config_file)
                if not all(key in config for key in ("model", "mode", "seed")):
                    raise ValueError("Run identity is missing")
                kind = "evaluation" if (run / "metrics.json").is_file() else "standalone"
                status = ("evaluated" if kind == "evaluation" else "completed"
                          if (run / "test/metrics.json").is_file() else "incomplete")
                rows.append(_row(_label(run, project), kind, config, status, run))
            except (OSError, ValueError):
                info["diagnostics"].append(_label(run, project) + ": unreadable standalone run")
    rows.extend(_missing_planned_results(project, rows, info["diagnostics"]))
    return rows, info


def print_results(rows, info, completed_only=False):
    successful = {"completed", "recovered", "evaluated"}
    visible = [row for row in rows if not completed_only or row["status"] in successful]
    completed = {row["run_dir"] for row in visible if row["status"] in successful and row["run_dir"]}
    print("Saved records: {}; planned entries without a saved run: {}; distinct completed/evaluated runs: {}".format(
        sum(row["kind"] != "planned" for row in visible),
        sum(row["kind"] == "planned" for row in visible), len(completed)))
    print("Check-only/preflight suites omitted: {} (use --include-checks to show them)".format(info["skipped_check_suites"]))
    groups = defaultdict(list)
    for row in visible:
        groups[row["batch"]].append(row)
    headers = ("Model", "Mode", "Seed", "Status", "Precision", "Val-AUC", "ROC-AUC",
               "ROC-CI", "PR-AUC", "AP", "Sensitivity", "Specificity")

    def number(value):
        value = _number(value)
        return "{:.4f}".format(value) if value is not None else "-"

    for batch, records in groups.items():
        print("\nBatch: " + batch)
        table = []
        for row in records:
            ci = "-"
            if all(_number(row[name]) is not None for name in ("confidence_level", "roc_ci_lower", "roc_ci_upper")):
                ci = "{:g}%[{},{}]".format(100 * row["confidence_level"], number(row["roc_ci_lower"]), number(row["roc_ci_upper"]))
            table.append(tuple(str(row[name]) if row[name] not in ("", None) else "-"
                               for name in ("model", "mode", "seed", "status", "precision"))
                         + (number(row["best_val_patient_roc_auc"]), number(row["roc_auc"]), ci)
                         + tuple(number(row[name]) for name in ("pr_auc", "average_precision", "sensitivity", "specificity")))
        widths = [max([len(header)] + [len(row[index]) for row in table]) for index, header in enumerate(headers)]
        print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
        for row in table:
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
        fingerprints = {(row["cohort_hash"], row["source_hash"], row["split_hash"])
                        for row in records if row["cohort_hash"] or row["source_hash"] or row["split_hash"]}
        for cohort, source, split in sorted(fingerprints):
            print("Input fingerprints: cohort={} source={} split={}".format(cohort or "-", source or "-", split or "-"))
        for row in records:
            if row["notes"]:
                print("{} {}: {}".format(row["model"], row["mode"], row["notes"]))
    for message in info["diagnostics"]:
        print("Diagnostic: " + message)
    print("\nPatient test metrics; Val-AUC uses validation. PR-AUC is trapezoidal area; AP is separate.")
    print("Batches/recovery histories are separate records. No runs or seeds are selected by test scores.")
    return visible


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--root", action="append", help="Explicit results root; repeat for additional roots")
    parser.add_argument("--models", nargs="+", help="Show only these model IDs, across every batch and mode")
    parser.add_argument("--include-checks", action="store_true", help="Also display batches that never started training")
    parser.add_argument("--completed-only", action="store_true", help="Display only finished/evaluated records")
    parser.add_argument("--csv", help="Export the displayed inventory to a NEW CSV; original results stay untouched")
    args = parser.parse_args(argv)
    roots = None
    if args.root:
        roots = [str(_path(value, Path(args.project).expanduser().resolve())) for value in args.root]
    rows, info = collect_results(args.project, roots, args.include_checks)
    if args.models:
        selected = set(name.lower().replace("-", "_") for name in args.models)
        rows = [row for row in rows if str(row["model"]).lower().replace("-", "_") in selected]
    visible = print_results(rows, info, args.completed_only)
    if args.csv:
        destination = Path(args.csv).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(visible)
        print("CSV: " + str(destination.resolve()))
    return 0 if visible else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("Result inventory failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
