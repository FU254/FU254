"""Decode every frozen-split image before any model computation or training.

Reports contain technical diagnostics only. A separate local CSV contains file
paths so the data owner can locate failures without exposing them in receipts,
console output, or the technical report. The audit never drops samples, changes
the split, or computes model predictions.
"""

from contextlib import contextmanager
import csv
import json
import logging
import os
from pathlib import Path
import re
import warnings

import pydicom
from pydicom.uid import UID, UID_dictionary

from .data import PREPROCESS_VERSION, _DicomDataError, _dicom_pixels


_PARTITIONS = ("train", "val", "test")
_DECODER_PACKAGES = (
    "pylibjpeg-openjpeg", "pylibjpeg-libjpeg", "pylibjpeg-rle", "pylibjpeg",
    "python-gdcm", "gdcm", "pillow", "numpy",
)
_DEPENDENCY_PATTERN = re.compile(
    r"(?<![a-z0-9_-])(" + "|".join(re.escape(name) for name in _DECODER_PACKAGES)
    + r")(?![a-z0-9_-])(?:\s*(>=|==|<=|>|<)\s*(\d+(?:\.\d+){0,3}))?",
    re.IGNORECASE,
)
_KNOWN_ERROR_TYPES = {
    "RuntimeError", "ValueError", "AttributeError", "KeyError", "TypeError",
    "OSError", "FileNotFoundError", "PermissionError", "EOFError", "IndexError",
    "NotImplementedError", "ImportError", "ModuleNotFoundError", "OverflowError",
    "InvalidDicomError", "_DicomDataError",
}
_PREPROCESS_ERRORS = {
    "DICOM rescaling or window metadata is nonfinite.",
    "Expected one monochrome 2D DICOM image.",
    "DICOM image contains no finite pixels.",
    "DICOM image contains nonfinite pixels.",
}


@contextmanager
def _quiet_decode():
    """Prevent decoder warnings/tracebacks from printing file or header values.

The audit is sequential. Temporarily silence library logging as well as Python
warnings because decoder failures may be logged with their original exceptions.
Both settings are restored, including when decoding raises.
"""
    previous_disable = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(max(previous_disable, logging.CRITICAL))
        try:
            yield
        finally:
            logging.disable(previous_disable)


def _transfer_syntax(dataset):
    """Return known standard transfer syntax information, never private UIDs."""
    if dataset is None:
        return None, "Unavailable"
    value = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", ""))
    definition = UID_dictionary.get(value)
    if definition is None or definition[1] != "Transfer Syntax":
        return None, "Unknown or missing transfer syntax"
    return value, definition[0]


def _pixel_metadata(dataset):
    """Only numeric pixel layout and byte counts, never identifying headers."""
    if dataset is None:
        return {}
    raw = getattr(dataset, "_dict", {}).get(0x7FE00010)
    declared = getattr(raw, "length", None)
    result = {"PixelData_declared_bytes": declared if type(declared) is int else None}
    for name in ("Rows", "Columns", "SamplesPerPixel", "BitsAllocated", "BitsStored",
                 "HighBit", "PixelRepresentation", "NumberOfFrames"):
        value = getattr(dataset, name, None)
        text = str(value) if value is not None else ""
        result[name] = int(text) if re.fullmatch(r"[+-]?\d+", text) else (None if value is None else "invalid")
    for name in ("PixelData", "FloatPixelData", "DoubleFloatPixelData"):
        value = getattr(dataset, name, None)
        result[name + "_bytes"] = len(value) if isinstance(value, (bytes, bytearray)) else None
    pixel_element = dataset.get(0x7FE00010)
    result["PixelData_undefined_length"] = bool(pixel_element.is_undefined_length) if pixel_element else None
    pixel_bytes = getattr(dataset, "PixelData", b"")
    result["PixelData_encapsulated"] = isinstance(pixel_bytes, (bytes, bytearray)) and pixel_bytes.startswith(b"\xfe\xff\x00\xe0")
    rows, columns, samples, bits = (result[key] for key in ("Rows", "Columns", "SamplesPerPixel", "BitsAllocated"))
    frames = result["NumberOfFrames"] if result["NumberOfFrames"] is not None else 1
    result["expected_uncompressed_bytes"] = None
    if all(type(value) is int and value > 0 for value in (rows, columns, samples, bits, frames)):
        count = rows * columns * samples * frames
        if bits == 1:
            result["expected_uncompressed_bytes"] = (count + 7) // 8
        elif bits % 8 == 0:
            result["expected_uncompressed_bytes"] = count * (bits // 8)
    return result


def _native_pixel_shortfall(syntax_uid, metadata):
    stored = metadata.get("PixelData_bytes")
    declared = metadata.get("PixelData_declared_bytes")
    return bool(syntax_uid and not UID(syntax_uid).is_compressed
                and not metadata.get("PixelData_encapsulated")
                and type(stored) is int and type(declared) is int
                and declared != 0xFFFFFFFF and stored < declared)


def _safe_failure(error, stage):
    """Classify known decoder errors without returning arbitrary exception text.

Only package names and numeric version constraints from a fixed allowlist can
be extracted. Original error messages may include any number of patient fields,
so replacing only known fields or filenames would not be a sufficient guard.
"""
    kind = type(error).__name__
    kind = kind if kind in _KNOWN_ERROR_TYPES else "DecoderError"
    if stage == "read":
        return kind, "dicom_read_failed", "DICOM file could not be read.", []
    text = str(error)
    lowered = text.lower()
    if isinstance(error, _DicomDataError):
        detail = text if text in _PREPROCESS_ERRORS else "Shared DICOM pixel preprocessing failed."
        return kind, "pixel_preprocessing_failed", detail, []
    missing_decoder = (
        ("dependenc" in lowered and any(term in lowered for term in ("missing", "requires", "unavailable")))
        or "no pixel data handlers" in lowered
        or "no available plugins" in lowered
        or isinstance(error, (ImportError, ModuleNotFoundError))
    )
    if missing_decoder:
        dependencies = sorted({
            match[0].lower() + ((match[1] + match[2]) if match[1] else "")
            for match in _DEPENDENCY_PATTERN.findall(text)
        })
        detail = "A pixel decoder or its required dependencies are unavailable."
        if dependencies:
            detail += " Detected dependencies: " + ", ".join(dependencies) + "."
        return kind, "missing_pixel_decoder", detail, dependencies
    if any(term in lowered for term in (
            "less than expected", "fewer bytes", "pixel data is too short", "unexpected end",
            "truncated", "insufficient data", "not enough data", "invalid pixel data",
            "doesn't match the expected length", "does not match the expected length")):
        return kind, "invalid_pixel_data", "Pixel data is truncated or invalid for its declared format.", []
    if isinstance(error, (AttributeError, KeyError)):
        return kind, "invalid_pixel_metadata", "Required pixel data or metadata is missing or invalid.", []
    return kind, "pixel_decode_failed", "Pixel decoding failed; inspect the original file locally.", []


def audit_split(split, output_dir):
    """Audit all train/val/test pixel data and return a safe technical report.

``row`` is the zero-based index within each partition, matching ``CXRDataset``.
The input DataFrame and frozen manifest remain unchanged. Reusing an output
directory containing either report artifact is rejected to preserve evidence.
``failures_private.csv`` is created with owner-only filesystem permissions and
contains paths for local diagnosis; callers should not upload that file.
"""
    if {"path", "split"}.difference(split.columns):
        raise ValueError("Pixel audit requires frozen split columns path and split.")
    if split.empty or not split["split"].isin(_PARTITIONS).all():
        raise ValueError("Pixel audit requires a nonempty frozen train/val/test split.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    private_path = output_dir / "failures_private.csv"
    if report_path.exists() or private_path.exists():
        raise FileExistsError("Pixel audit output already exists; use a fresh audit directory.")

    counts = {partition: {"checked": 0, "passed": 0, "failed": 0} for partition in _PARTITIONS}
    transfer_counts = {}
    failures = []
    private_failures = []
    checked = 0
    total = len(split)
    print(f"[audit-data] Checking pixel data for {total} frozen-split images.", flush=True)
    for partition in _PARTITIONS:
        paths = split.loc[split["split"] == partition, "path"]
        for row, path in enumerate(paths):
            dataset = None
            pixel_metadata = {}
            failure = None
            stage = "read"
            with _quiet_decode():
                try:
                    dataset = pydicom.dcmread(path)
                    stage = "pixels"
                    pixel_metadata = _pixel_metadata(dataset)
                    syntax_uid, _ = _transfer_syntax(dataset)
                    if _native_pixel_shortfall(syntax_uid, pixel_metadata):
                        raise ValueError("Truncated native PixelData")
                    _dicom_pixels(dataset)
                except Exception as error:
                    failure = _safe_failure(error, stage)
                syntax_uid, syntax_name = _transfer_syntax(dataset)
            syntax_key = syntax_uid or "unknown"
            syntax_count = transfer_counts.setdefault(syntax_key, {"name": syntax_name, "images": 0})
            syntax_count["images"] += 1
            checked += 1
            counts[partition]["checked"] += 1
            if failure is None:
                counts[partition]["passed"] += 1
            else:
                kind, category, detail, dependencies = failure
                stored = pixel_metadata.get("PixelData_bytes")
                declared = pixel_metadata.get("PixelData_declared_bytes")
                if _native_pixel_shortfall(syntax_uid, pixel_metadata):
                    category = "truncated_pixel_data"
                    detail = (f"Uncompressed PixelData declares {declared} bytes but only {stored} bytes "
                              "are present. Restore the complete original DICOM file.")
                item = {
                    "split": partition,
                    "row": row,
                    "error_type": kind,
                    "error_category": category,
                    "transfer_syntax_uid": syntax_uid,
                    "transfer_syntax_name": syntax_name,
                    "error_detail": detail,
                    "decoder_dependencies": dependencies,
                    "pixel_metadata": pixel_metadata,
                }
                failures.append(item)
                private_failures.append({
                    **{key: value for key, value in item.items() if key not in {"decoder_dependencies", "pixel_metadata"}},
                    "path": str(path),
                })
                counts[partition]["failed"] += 1
            if checked % 100 == 0 or checked == total:
                print(f"[audit-data] Checked {checked}/{total}; failed={len(failures)}.", flush=True)

    report = {
        "audit_passed": not failures,
        "checked_images": checked,
        "passed_images": checked - len(failures),
        "failed_images": len(failures),
        "preprocess_version": PREPROCESS_VERSION,
        "split_counts": counts,
        "transfer_syntax_counts": transfer_counts,
        "failures": failures,
    }
    # Exclusive writes prevent a second audit from replacing earlier evidence.
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        columns = (
            "split", "row", "path", "error_type", "error_category", "transfer_syntax_uid",
            "transfer_syntax_name", "error_detail",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(private_failures)
    return report
