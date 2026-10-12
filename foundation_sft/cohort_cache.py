"""Private, opt-in cache for validated workbook/DICOM header cohorts.

Every use rechecks the complete DICOM inventory and label workbook. Only
unchanged sources can reuse header parsing; the frozen split still validates
the returned scientific provenance independently.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import digest, file_signature, load_json, private_directory, runtime_signature, save_json


COHORT_CACHE_VERSION = "workbook_dicom_header_cohort_v1"
_SOURCE_CHANGED = "Cohort sources changed during cache preparation; retry with stable source files."


def _workbook_sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _source_snapshot(config):
    settings = dict(config.get("data", config))
    for name in ("excel_path", "dicom_dir"):
        if not settings.get(name):
            raise ValueError(f"Missing required data configuration: {name}.")
    try:
        workbook = Path(settings["excel_path"]).expanduser().resolve()
        dicom_dir = Path(settings["dicom_dir"]).expanduser().resolve()
        if not workbook.is_file() or not dicom_dir.is_dir():
            raise FileNotFoundError
        before = file_signature(workbook)
        workbook_hash = _workbook_sha256(workbook)
        if before != file_signature(workbook):
            raise ValueError(_SOURCE_CHANGED)
        # Match build_cohort's case-insensitive suffix and recursive inventory,
        # including unmatched and currently unreadable files. Directory mtime
        # alone misses edits to existing nested files.
        inventory = [
            {"relative_path": path.relative_to(dicom_dir).as_posix(),
             "file": file_signature(path)}
            for path in sorted(dicom_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() == ".dcm"
        ]
    except (OSError, TypeError):
        raise FileNotFoundError(
            "Configured workbook or DICOM sources are unavailable during cohort cache preparation."
        ) from None
    return {
        "version": COHORT_CACHE_VERSION,
        "settings": settings,
        "runtime": runtime_signature(),
        "dicom_dir": str(dicom_dir),
        "workbook": {"file": before, "sha256": workbook_hash},
        "inventory": inventory,
    }


def _pack(value):
    """Encode typed attrs/records as JSON without executable deserialization."""
    if value is pd.NA:
        return {"type": "missing"}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"type": "scalar", "value": value}
        return {"type": "float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [_pack(v) for v in value]}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[_pack(key), _pack(item)] for key, item in value.items()]}
    raise ValueError("Cohort cache contains an unsupported metadata value type.")


def _unpack(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid cohort cache value.")
    kind = value.get("type")
    if kind == "scalar" and set(value) == {"type", "value"}:
        item = value["value"]
        if item is None or isinstance(item, (str, bool, int)) or (isinstance(item, float) and math.isfinite(item)):
            return item
    elif kind == "missing" and set(value) == {"type"}:
        return pd.NA
    elif kind == "float" and set(value) == {"type", "value"}:
        return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[value["value"]]
    elif kind in {"list", "tuple"} and set(value) == {"type", "items"}:
        items = [_unpack(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    elif kind == "dict" and set(value) == {"type", "items"}:
        return {_unpack(key): _unpack(item) for key, item in value["items"]}
    raise ValueError("Invalid cohort cache value.")


def _cohort_payload(cohort):
    if not isinstance(cohort, pd.DataFrame) or not cohort.columns.is_unique:
        raise ValueError("Cohort cache requires a DataFrame with unique columns.")
    # build_cohort returns a RangeIndex. Preserve it explicitly rather than
    # relying on JSON/Pandas inference to reconstruct the original rows.
    if not isinstance(cohort.index, pd.RangeIndex):
        raise ValueError("Cohort cache requires a cohort RangeIndex.")
    return {
        "columns": list(cohort.columns),
        "dtypes": [str(dtype) for dtype in cohort.dtypes],
        "records": _pack(cohort.to_dict(orient="records")),
        "attrs": _pack({name: value for name, value in cohort.attrs.items() if name != "cohort_cache_hit"}),
        "index": {"start": cohort.index.start, "stop": cohort.index.stop,
                  "step": cohort.index.step, "name": _pack(cohort.index.name)},
    }


def _restore_cohort(payload):
    if not isinstance(payload, dict) or set(payload) != {"columns", "dtypes", "records", "attrs", "index"}:
        raise ValueError("Invalid cohort cache payload.")
    columns, dtypes = payload["columns"], payload["dtypes"]
    if (not isinstance(columns, list) or not columns or len(set(columns)) != len(columns)
            or not all(isinstance(name, str) for name in columns)
            or not isinstance(dtypes, list) or len(dtypes) != len(columns)):
        raise ValueError("Invalid cohort cache column metadata.")
    records = _unpack(payload["records"])
    if not isinstance(records, list) or any(not isinstance(row, dict) or set(row) != set(columns) for row in records):
        raise ValueError("Invalid cohort cache records.")
    result = pd.DataFrame(records, columns=columns)
    for name, dtype in zip(columns, dtypes):
        result[name] = result[name].astype(dtype)
    index = payload["index"]
    result.index = pd.RangeIndex(index["start"], index["stop"], index["step"], name=_unpack(index["name"]))
    attrs = _unpack(payload["attrs"])
    if not isinstance(attrs, dict):
        raise ValueError("Invalid cohort cache attributes.")
    result.attrs = attrs
    return result


def cached_cohort(config, cache_dir, builder):
    """Build once and reuse headers only while every source remains unchanged.

    Cache paths contain hashed keys, and records remain in a private local
    directory. Corrupt or incompatible cache entries are rebuilt. Source edits
    during either a cache hit or initial build fail before returning a cohort.
    """
    before = _source_snapshot(config)
    directory = private_directory(Path(cache_dir) / "cohorts")
    key = digest(before)
    path = directory / f"{key}.json"
    payload = load_json(path)
    result = None
    if isinstance(payload, dict) and payload.get("version") == COHORT_CACHE_VERSION and payload.get("source") == before:
        try:
            result = _restore_cohort(payload["cohort"])
        except (ValueError, TypeError, KeyError, OverflowError):
            result = None
    hit = result is not None
    if result is None:
        result = builder(config)
    if _source_snapshot(config) != before:
        raise ValueError(_SOURCE_CHANGED)
    if not hit:
        save_json(path, {"version": COHORT_CACHE_VERSION, "source": before, "cohort": _cohort_payload(result)})
    result.attrs["cohort_cache_hit"] = hit
    return result
