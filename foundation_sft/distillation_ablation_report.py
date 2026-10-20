"""Matched-seed loss ablation differences; no model or patient data are loaded.

Differences describe the fixed experiment. Seed SD is not a patient confidence
interval, a significance test, or an ensemble evaluation.
"""

import csv
import math
from pathlib import Path
import statistics

from .distillation_suite import METRICS, metric_rows


CONTRASTS = (
    ("all_kd_gain", "full_kd", "supervised"),
    ("soft_kd_gain", "soft_only", "supervised"),
    ("beyond_soft_kd", "full_kd", "soft_only"),
    ("soft_contribution", "full_kd", "no_soft"),
    ("global_contribution", "full_kd", "no_global"),
    ("local_contribution", "full_kd", "no_local"),
)
SUCCESS = {"completed", "reused", "resumed"}


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def paired_differences(state):
    """Candidate minus reference for completed matching seeds, never fill failures."""
    rows = metric_rows(state)
    records = {}
    for record in state.get("jobs", {}).values():
        job = record["job"]
        key = job["group"], job["seed"]
        if key in records:
            raise ValueError(f"Duplicate ablation group/seed: {key}")
        records[key] = record
    available = {(row["group"], row["seed"]): row for row in rows if row["status"] in SUCCESS}
    differences = []
    for name, candidate, reference in CONTRASTS:
        seeds = sorted({seed for group, seed in available if group == candidate}
                       & {seed for group, seed in available if group == reference})
        for seed in seeds:
            first, second = available[candidate, seed], available[reference, seed]
            a, b = records[candidate, seed], records[reference, seed]
            for field in ("cohort_hash", "initial_state_sha256"):
                value = a.get("result", {}).get(field)
                other = b.get("result", {}).get(field)
                if not value or value != other:
                    raise ValueError(f"Cannot pair {candidate}/{reference} seed{seed}: {field} differs or is missing")
            ta = a.get("check", {}).get("teacher", {}).get("checkpoint_sha256")
            tb = b.get("check", {}).get("teacher", {}).get("checkpoint_sha256")
            if ta != tb:
                raise ValueError(f"Cannot pair {candidate}/{reference} seed{seed}: teacher differs")
            difference = {"contrast": name, "candidate": candidate, "reference": reference,
                          "seed": seed}
            for metric in METRICS:
                x, y = first.get(metric), second.get(metric)
                difference["delta_" + metric] = float(x) - float(y) if _finite(x) and _finite(y) else None
            differences.append(difference)
    return differences


def paired_summaries(state):
    differences = paired_differences(state)
    summaries = []
    jobs = [record["job"] for record in state.get("jobs", {}).values()]
    for name, candidate, reference in CONTRASTS:
        requested = ({job["seed"] for job in jobs if job["group"] == candidate}
                     & {job["seed"] for job in jobs if job["group"] == reference})
        if not requested:
            continue
        pairs = [row for row in differences if row["contrast"] == name]
        row = {"contrast": name, "candidate": candidate, "reference": reference,
               "n_pairs": len(pairs), "n_requested": len(requested),
               "seeds": ",".join(str(pair["seed"]) for pair in pairs)}
        for metric in METRICS:
            values = [pair["delta_" + metric] for pair in pairs if _finite(pair["delta_" + metric])]
            row["delta_" + metric + "_n"] = len(values)
            row["delta_" + metric + "_mean"] = statistics.mean(values) if values else None
            row["delta_" + metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(row)
    return summaries


def _write(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else columns)
        writer.writeheader()
        writer.writerows(rows)


def write_paired_reports(suite, state):
    directory = Path(suite) / "report"
    # Validate both tables before writing either one.
    differences, summaries = paired_differences(state), paired_summaries(state)
    _write(directory / "paired_differences.csv", differences,
           ["contrast", "candidate", "reference", "seed", *("delta_" + metric for metric in METRICS)])
    _write(directory / "paired_summary.csv", summaries,
           ["contrast", "candidate", "reference", "n_pairs", "n_requested", "seeds"])


def print_paired_metrics(state):
    summaries = paired_summaries(state)
    if not summaries:
        return
    print("Matched-seed differences: candidate minus reference (mean +/- sample SD)")
    for row in summaries:
        values = []
        for metric in ("roc_auc", "pr_auc"):
            mean, sd = row["delta_" + metric + "_mean"], row["delta_" + metric + "_sd"]
            text = "-" if mean is None else f"{mean:+.4f}"
            text += " +/- " + ("-" if sd is None else f"{sd:.4f}")
            values.append(f"delta_{metric}={text}")
        print(f"{row['contrast']}: {row['candidate']} - {row['reference']} "
              f"n={row['n_pairs']}/{row['n_requested']} seeds=[{row['seeds']}] " + "; ".join(values))
    print("These are seed-level paired differences; patient bootstrap intervals remain in each run.")
