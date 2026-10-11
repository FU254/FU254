"""Create and reuse a frozen, patient-isolated classification split.

The manifest contains one row per sample and all provenance in one CSV so it
can be published atomically. Patient stratification uses the maximum observed
label; individual examination labels are preserved when a patient's outcome
changes over time. No patient can occur in more than one partition.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Sequence

import pandas as pd


SPLIT_VERSION = "foundation_patient_stratified_v1"
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = ("sample_id", "patient_id", "accession", "path", "label")
_PROVENANCE_COLUMNS = (
    "label_source_signature", "label_definition", "label_definition_signature",
)
_PROVENANCE_ATTRS = _PROVENANCE_COLUMNS + ("preprocess_version",)
_FILE_STAT_COLUMNS = ("dicom_size", "dicom_mtime_ns")
_STABLE_COLUMNS = _PROVENANCE_COLUMNS + _FILE_STAT_COLUMNS + ("patient_id_source",)
_GROUPING = "patient_id;stratum=max(label);examination_labels_preserved"
_ALLOCATION = "classwise_nearest_ratio;minimum_one_patient_per_class_per_split"
_METADATA_COLUMNS = (
    "_version", "_seed", "_ratios", "_cohort_hash", "_source_hash",
    "_grouping", "_allocation", "_provenance_columns", "_manifest_hash",
)
_MISSING_TEXT = {"", "nan", "none", "null", "<na>", "na", "n/a"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _prepare_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cohort, pd.DataFrame):
        raise ValueError("cohort must be a pandas DataFrame.")
    if not cohort.columns.is_unique:
        raise ValueError("Cohort column names must be unique.")
    if any(column not in cohort.columns for column in REQUIRED_COLUMNS):
        raise ValueError("Cohort requires sample_id, patient_id, accession, path and label columns.")
    if cohort.empty:
        raise ValueError("Cannot split an empty cohort.")
    if any(column in cohort.columns for column in _METADATA_COLUMNS):
        raise ValueError("Cohort contains reserved manifest metadata columns.")

    columns = list(REQUIRED_COLUMNS) + [
        column for column in _STABLE_COLUMNS if column in cohort.columns
    ]
    clean = cohort.loc[:, columns].copy().reset_index(drop=True)
    for column in REQUIRED_COLUMNS[:-1]:
        missing = clean[column].isna()
        clean[column] = clean[column].astype(str).str.strip()
        missing |= clean[column].str.lower().isin(_MISSING_TEXT)
        if missing.any():
            raise ValueError(f"Cohort contains {int(missing.sum())} missing {column} values.")
    if clean["sample_id"].duplicated().any():
        raise ValueError("Cohort sample_id values must be unique.")

    numeric_labels = pd.to_numeric(clean["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin((0, 1)).all():
        raise ValueError("Cohort labels must be strictly binary 0 or 1 without missing values.")
    clean["label"] = numeric_labels.astype("int64")
    for column in columns[len(REQUIRED_COLUMNS):]:
        if clean[column].isna().any():
            raise ValueError("Per-sample label provenance cannot contain missing values.")
        if column in _FILE_STAT_COLUMNS:
            text = clean[column].astype(str).str.strip()
            if not text.str.fullmatch(r"[0-9]+").all():
                raise ValueError("DICOM file-size and modification metadata must be nonnegative integers.")
            clean[column] = text.map(int)
        else:
            clean[column] = clean[column].astype(str)
            if column == "patient_id_source" and clean[column].str.strip().str.lower().isin(_MISSING_TEXT).any():
                raise ValueError("Patient identifier provenance must be nonempty when supplied.")
    return clean


def _configuration(seed: int, ratios: Sequence[float]) -> tuple[int, tuple[float, float, float]]:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("Split seed must be an integer.")
    seed = int(seed)
    try:
        values = tuple(float(value) for value in ratios)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Split ratios must be three positive finite numbers summing to one.") from None
    if (
        len(values) != 3
        or any(not math.isfinite(value) or value <= 0 for value in values)
        or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Split ratios must be three positive finite numbers summing to one.")
    # Eliminate harmless floating-point summation differences before hashing.
    total = sum(values)
    return seed, tuple(value / total for value in values)


def _source_hash(cohort: pd.DataFrame) -> str:
    source = {key: cohort.attrs[key] for key in _PROVENANCE_ATTRS if key in cohort.attrs}
    try:
        return _digest(source)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Cohort label provenance attributes must contain finite JSON-compatible values.") from None


def _cohort_hash(clean: pd.DataFrame) -> str:
    ordered = clean.sort_values("sample_id", kind="stable")
    return _digest({"columns": list(ordered.columns), "rows": ordered.values.tolist()})


def _allocation_counts(count: int, ratios: Sequence[float]) -> list[int]:
    """Round class quotas subject to every partition receiving that class."""
    targets = [count * ratio for ratio in ratios]
    counts = [max(1, int(math.floor(target))) for target in targets]
    while sum(counts) > count:
        eligible = [index for index, value in enumerate(counts) if value > 1]
        index = max(eligible, key=lambda i: (counts[i] - targets[i], -i))
        counts[index] -= 1
    while sum(counts) < count:
        index = max(range(3), key=lambda i: (targets[i] - counts[i], -i))
        counts[index] += 1
    return counts


def _assign_patients(clean: pd.DataFrame, seed: int, ratios: Sequence[float]) -> pd.Series:
    patient_labels = clean.groupby("patient_id", sort=True)["label"].max()
    class_counts = patient_labels.value_counts().reindex((0, 1), fill_value=0)
    if (class_counts < 3).any():
        raise ValueError(
            "Patient stratification requires at least three patients in each max(label) "
            "stratum to put both classes in train, val and test; "
            f"patient class counts are 0={int(class_counts[0])}, 1={int(class_counts[1])}. "
            "Add patients or revise the prespecified cohort; sample labels remain unchanged."
        )
    generator = random.Random(seed)
    split_by_patient: dict[str, str] = {}
    for label in (0, 1):
        patients = patient_labels.index[patient_labels.eq(label)].tolist()
        generator.shuffle(patients)
        counts = _allocation_counts(len(patients), ratios)
        offset = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            for patient in patients[offset:offset + count]:
                split_by_patient[patient] = split_name
            offset += count
    return clean["patient_id"].map(split_by_patient)


def _manifest_hash(manifest: pd.DataFrame) -> str:
    rows = manifest.loc[:, ["sample_id", "split"]].sort_values("sample_id", kind="stable")
    return _digest(rows.values.tolist())


def _expected_metadata(clean: pd.DataFrame, cohort: pd.DataFrame, seed: int, ratios: Sequence[float]) -> dict[str, str]:
    return {
        "_version": SPLIT_VERSION,
        "_seed": str(seed),
        "_ratios": _json(list(ratios)),
        "_cohort_hash": _cohort_hash(clean),
        "_source_hash": _source_hash(cohort),
        "_grouping": _GROUPING,
        "_allocation": _ALLOCATION,
        "_provenance_columns": _json([column for column in _STABLE_COLUMNS if column in clean]),
    }


def _validate_assignment(manifest: pd.DataFrame) -> None:
    if not manifest["split"].isin(SPLIT_NAMES).all():
        raise ValueError("Split manifest contains missing or invalid partition names.")
    if (manifest.groupby("patient_id")["split"].nunique() != 1).any():
        raise ValueError("Split manifest leaks patients across partitions.")
    for split_name in SPLIT_NAMES:
        partition = manifest.loc[manifest["split"].eq(split_name)]
        if partition.empty or set(partition["label"]) != {0, 1}:
            raise ValueError(f"Split manifest {split_name} must be nonempty and contain both label classes.")


def _load_manifest(path: Path, clean: pd.DataFrame, expected: dict[str, str], seed: int, ratios: Sequence[float]) -> pd.DataFrame:
    try:
        manifest = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError):
        raise ValueError("Cannot read the frozen split manifest; repair or explicitly regenerate it.") from None
    required = set(clean.columns) | set(_METADATA_COLUMNS) | {"split"}
    if required != set(manifest.columns) or not manifest.columns.is_unique or manifest.empty:
        raise ValueError("Split manifest is empty or missing required cohort/provenance metadata.")
    for column, value in expected.items():
        if not manifest[column].eq(value).all():
            raise ValueError(
                "Frozen split manifest does not match the cohort, label source, split version, "
                "seed or ratios. Reuse the original inputs or explicitly regenerate the manifest."
            )
    stored = _prepare_cohort(manifest.drop(columns=list(_METADATA_COLUMNS) + ["split"]))
    if len(stored) != len(clean) or _cohort_hash(stored) != expected["_cohort_hash"]:
        raise ValueError("Frozen split manifest sample membership, patient, label, accession or path has changed.")
    manifest["label"] = stored["label"].to_numpy()
    for column in REQUIRED_COLUMNS[:-1]:
        manifest[column] = stored[column].to_numpy()
    _validate_assignment(manifest)
    if not manifest["_manifest_hash"].eq(_manifest_hash(manifest)).all():
        raise ValueError("Frozen split manifest partition assignments have been modified.")
    expected_splits = _assign_patients(stored, seed, ratios)
    if not manifest["split"].reset_index(drop=True).equals(expected_splits.reset_index(drop=True)):
        raise ValueError("Frozen split manifest assignments are incompatible with its saved split configuration.")
    return manifest


def _write_manifest(manifest: pd.DataFrame, path: Path, overwrite: bool) -> bool:
    """Atomically publish one complete CSV, without clobbering a race winner."""
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".foundation-split-", suffix=".tmp", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            manifest.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                return False
        return True
    except OSError:
        raise ValueError("Cannot atomically save the split manifest in its configured directory.") from None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def load_or_create_split(
    cohort: pd.DataFrame,
    manifest_path: str,
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return the input cohort with a reproducible ``split`` column.

    ``patient_id`` must be a verified, nonempty patient identifier, rather than
    an examination/accession identifier. All observations of one patient stay
    together. Patient strata use ``max(label)``; a patient's labels may differ
    across examinations. Each class needs at least three distinct patients.
    Class-wise quotas approximate ``ratios`` with at least one patient of each
    stratum in every partition, so small cohorts cannot match ratios exactly.

    A saved manifest is immutable unless ``overwrite=True`` is explicitly
    selected. Changes to sample membership, patient identifiers, accession,
    image paths, labels, seed, ratios or provenance are rejected. Provenance is
    taken from optional per-row columns and DataFrame attrs named
    ``label_source_signature``, ``label_definition`` or
    ``label_definition_signature``. Record a source content digest and label
    definition there before calling this function. Optional ``dicom_size``,
    ``dicom_mtime_ns`` and ``patient_id_source`` columns also enter the cohort
    fingerprint to detect file replacements and identifier-source changes.
    Input order does not affect
    assignment, and returned rows/index/extra columns retain the input order.

    Manifest rows contain sensitive cohort identifiers and paths: keep this
    artifact in the configured private output directory, outside source control.
    Exceptions report counts and configuration fields without patient values.
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    seed, ratios = _configuration(seed, ratios)
    clean = _prepare_cohort(cohort)
    expected = _expected_metadata(clean, cohort, seed, ratios)
    try:
        path = Path(manifest_path)
    except (TypeError, ValueError):
        raise ValueError("manifest_path must be a valid filesystem path.") from None
    if not str(manifest_path).strip() or path.name in ("", ".", ".."):
        raise ValueError("manifest_path must identify a CSV file.")
    if path.exists() and not overwrite:
        manifest = _load_manifest(path, clean, expected, seed, ratios)
    else:
        manifest = clean.sort_values("sample_id", kind="stable").reset_index(drop=True)
        manifest["split"] = _assign_patients(manifest, seed, ratios)
        _validate_assignment(manifest)
        for column, value in expected.items():
            manifest[column] = value
        manifest["_manifest_hash"] = _manifest_hash(manifest)
        if not _write_manifest(manifest, path, overwrite):
            manifest = _load_manifest(path, clean, expected, seed, ratios)

    split_by_sample = manifest.set_index("sample_id")["split"]
    output = cohort.copy()
    output["split"] = clean["sample_id"].map(split_by_sample).to_numpy()
    output.attrs["split_metadata"] = {
        **expected,
        "_manifest_hash": str(manifest["_manifest_hash"].iloc[0]),
        "patient_counts": {
            name: int(manifest.loc[manifest["split"].eq(name), "patient_id"].nunique())
            for name in SPLIT_NAMES
        },
    }
    return output
