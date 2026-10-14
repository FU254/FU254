"""Reuse only complete, successful pixel audits of unchanged frozen inputs.

The cache contains an opaque input fingerprint and a technical report. Patient
identifiers, file paths and the full frozen cohort are hashed in memory and are
never written into cached reports or printed. The source files are checked
again before reuse and after a fresh audit, including files outside the cohort
builder's header cache. Runtime and code signatures isolate decoder environments.
"""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path

from .cache import digest, file_signature, load_json, private_directory, runtime_signature, save_json
from .data import PREPROCESS_VERSION


AUDIT_CACHE_VERSION = "full_pixel_audit_cache_v1"
_PARTITIONS = ("train", "val", "test")
_PRIVATE_COLUMNS = (
    "split", "row", "path", "error_type", "error_category", "transfer_syntax_uid",
    "transfer_syntax_name", "error_detail",
)


def _validate_split_and_output(split, output_dir):
    if {"path", "split"}.difference(split.columns):
        raise ValueError("Pixel audit requires frozen split columns path and split.")
    if split.empty or not split["split"].isin(_PARTITIONS).all():
        raise ValueError("Pixel audit requires a nonempty frozen train/val/test split.")
    output = Path(output_dir)
    if (output / "report.json").exists() or (output / "failures_private.csv").exists():
        raise FileExistsError("Pixel audit output already exists; use a fresh audit directory.")
    return output


def _cohort_fingerprint(split):
    # Include every provenance/metadata column and attr, rather than trusting a
    # caller-supplied cohort_hash or only hashing the image paths. The row order
    # also determines per-partition diagnostic row numbers in a cached report.
    try:
        return digest({
            "columns": list(split.columns),
            "dtypes": [str(dtype) for dtype in split.dtypes],
            "rows": split.values.tolist(),
            "attrs": {name: value for name, value in split.attrs.items()
                      if name != "cohort_cache_hit"},
        })
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Pixel audit cache requires finite JSON-compatible frozen cohort metadata.") from None


def _source_snapshot(split):
    result = []
    for path in split["path"]:
        try:
            result.append(file_signature(path))
        except (OSError, RuntimeError) as error:
            # Let the ordinary full audit produce the sanitized per-image
            # missing/unreadable-file failure report. Such a report is never
            # cached; a transition to/from this state invalidates the snapshot.
            result.append({"path": str(Path(path).expanduser().absolute()),
                           "unavailable": type(error).__name__})
    return result


def _stable_inputs(split, cohort, sources, runtime):
    if (_cohort_fingerprint(split) != cohort
            or _source_snapshot(split) != sources
            or runtime_signature() != runtime):
        raise RuntimeError("Pixel audit inputs changed during verification; rerun with stable source files and environment.")


def _valid_report(report, split):
    if not isinstance(report, dict):
        return False
    if (report.get("audit_passed") is not True
            or report.get("preprocess_version") != PREPROCESS_VERSION
            or report.get("failures") != []):
        return False
    total = len(split)
    for name, expected in (("checked_images", total), ("passed_images", total), ("failed_images", 0)):
        if type(report.get(name)) is not int or report[name] != expected:
            return False
    expected_counts = {
        name: {"checked": int(split["split"].eq(name).sum()),
               "passed": int(split["split"].eq(name).sum()), "failed": 0}
        for name in _PARTITIONS
    }
    if report.get("split_counts") != expected_counts:
        return False
    syntaxes = report.get("transfer_syntax_counts")
    if not isinstance(syntaxes, dict) or not syntaxes:
        return False
    if any(not isinstance(value, dict) or type(value.get("images")) is not int
           or value["images"] <= 0 for value in syntaxes.values()):
        return False
    return sum(value["images"] for value in syntaxes.values()) == total


def _write_reused_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive writes preserve the ordinary audit's evidence protection. The
    # local failure artifact is always present and owner-only, even on a hit.
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    descriptor = os.open(output / "failures_private.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=_PRIVATE_COLUMNS).writeheader()


def cached_audit(split, output_dir, cache_dir, force, run_audit):
    """Return a fresh technical report using a verified cache hit or full scan."""
    output = _validate_split_and_output(split, output_dir)
    if not isinstance(force, bool):
        raise ValueError("force_audit must be a boolean.")
    cohort = _cohort_fingerprint(split)
    sources = _source_snapshot(split)
    runtime = runtime_signature()
    identity = digest({"version": AUDIT_CACHE_VERSION, "cohort": cohort,
                       "sources": sources, "runtime": runtime})
    cache = private_directory(Path(cache_dir) / "audits")
    entry = cache / (identity + ".json")
    payload = None if force else load_json(entry)
    _stable_inputs(split, cohort, sources, runtime)
    if (isinstance(payload, dict) and payload.get("version") == AUDIT_CACHE_VERSION
            and payload.get("identity") == identity
            and _valid_report(payload.get("report"), split)):
        report = deepcopy(payload["report"])
        report["audit_cached"] = True
        _write_reused_report(report, output)
        _stable_inputs(split, cohort, sources, runtime)
        print(f"[audit-data] Reused successful pixel audit for {len(split)} frozen-split images.", flush=True)
        return report

    report = run_audit(split, output)
    _stable_inputs(split, cohort, sources, runtime)
    if _valid_report(report, split):
        save_json(entry, {"version": AUDIT_CACHE_VERSION, "identity": identity,
                          "report": report})
    return report
