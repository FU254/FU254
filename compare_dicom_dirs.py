#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency at runtime
    pydicom = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two DICOM datasets by filename, path, size, and key header tags."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan one DICOM directory and save a manifest JSON file.",
    )
    scan_parser.add_argument("input", type=str, help="Directory containing DICOM files.")
    scan_parser.add_argument("output", type=str, help="Path to output manifest JSON.")
    scan_parser.add_argument(
        "--hash",
        action="store_true",
        help="Calculate SHA1 for every file. Slower, but useful for exact content comparisons.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two directories or two manifest JSON files.",
    )
    compare_parser.add_argument("left", type=str, help="Left directory or manifest JSON.")
    compare_parser.add_argument("right", type=str, help="Right directory or manifest JSON.")
    compare_parser.add_argument(
        "--output-dir",
        type=str,
        default="compare_dicom_outputs",
        help="Directory for summary JSON and CSV exports.",
    )
    compare_parser.add_argument(
        "--hash",
        action="store_true",
        help="When scanning live directories, calculate SHA1 for every file.",
    )
    compare_parser.add_argument(
        "--max-preview",
        type=int,
        default=20,
        help="Number of sample rows to print for each major difference bucket.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {"nan", "None", "<NA>"}:
        return None
    return text


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_dicom_header(path: Path) -> dict[str, Any]:
    if pydicom is None:
        return {"dicom_read_ok": False, "dicom_error": "pydicom not installed"}
    tags = [
        "AccessionNumber",
        "StudyDate",
        "PatientID",
        "PatientName",
        "Modality",
        "SeriesDescription",
        "SOPInstanceUID",
    ]
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=tags,
            force=True,
        )
        return {
            "dicom_read_ok": True,
            "accession": normalize_text(ds.get("AccessionNumber")),
            "study_date": normalize_text(ds.get("StudyDate")),
            "patient_id": normalize_text(ds.get("PatientID")),
            "patient_name": normalize_text(ds.get("PatientName")),
            "modality": normalize_text(ds.get("Modality")),
            "series_description": normalize_text(ds.get("SeriesDescription")),
            "sop_instance_uid": normalize_text(ds.get("SOPInstanceUID")),
        }
    except Exception as exc:  # pragma: no cover - runtime/environment dependent
        return {"dicom_read_ok": False, "dicom_error": str(exc)}


def scan_directory(root: Path, compute_hash: bool) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {root}")

    files = sorted(root.rglob("*.dcm"))
    records: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        header = read_dicom_header(path)
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "filename": path.name,
            "stem": path.stem,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        record.update(header)
        if compute_hash:
            record["sha1"] = sha1_file(path)
        records.append(record)

    return {
        "source_type": "directory",
        "root": str(root),
        "file_count": len(records),
        "hash_enabled": compute_hash,
        "records": records,
    }


def save_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest_or_scan(path_text: str, compute_hash: bool) -> dict[str, Any]:
    path = Path(path_text)
    if path.exists() and path.is_file() and path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["manifest_path"] = str(path)
        return manifest
    return scan_directory(path, compute_hash=compute_hash)


def build_index(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        value = normalize_text(row.get(key))
        if value is not None:
            grouped[value].append(row)
    return dict(grouped)


def first_or_none(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    return items[0] if items else None


def compare_manifests(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    left_records = left["records"]
    right_records = right["records"]

    left_by_name = build_index(left_records, "filename")
    right_by_name = build_index(right_records, "filename")
    left_by_stem = build_index(left_records, "stem")
    right_by_stem = build_index(right_records, "stem")
    left_by_acc = build_index(left_records, "accession")
    right_by_acc = build_index(right_records, "accession")

    left_name_set = set(left_by_name)
    right_name_set = set(right_by_name)
    left_stem_set = set(left_by_stem)
    right_stem_set = set(right_by_stem)
    left_acc_set = set(left_by_acc)
    right_acc_set = set(right_by_acc)

    only_left_names = sorted(left_name_set - right_name_set)
    only_right_names = sorted(right_name_set - left_name_set)
    only_left_stems = sorted(left_stem_set - right_stem_set)
    only_right_stems = sorted(right_stem_set - left_stem_set)
    only_left_acc = sorted(left_acc_set - right_acc_set)
    only_right_acc = sorted(right_acc_set - left_acc_set)

    duplicate_names_left = sorted(k for k, v in left_by_name.items() if len(v) > 1)
    duplicate_names_right = sorted(k for k, v in right_by_name.items() if len(v) > 1)
    duplicate_stems_left = sorted(k for k, v in left_by_stem.items() if len(v) > 1)
    duplicate_stems_right = sorted(k for k, v in right_by_stem.items() if len(v) > 1)
    duplicate_acc_left = sorted(k for k, v in left_by_acc.items() if len(v) > 1)
    duplicate_acc_right = sorted(k for k, v in right_by_acc.items() if len(v) > 1)

    common_unique_names = sorted(
        name
        for name in (left_name_set & right_name_set)
        if len(left_by_name[name]) == 1 and len(right_by_name[name]) == 1
    )

    same_name_different_size: list[dict[str, Any]] = []
    same_name_different_hash: list[dict[str, Any]] = []
    same_name_different_accession: list[dict[str, Any]] = []
    same_name_different_path: list[dict[str, Any]] = []
    same_name_different_study_date: list[dict[str, Any]] = []

    for name in common_unique_names:
        left_row = left_by_name[name][0]
        right_row = right_by_name[name][0]

        if left_row.get("size_bytes") != right_row.get("size_bytes"):
            same_name_different_size.append(
                {
                    "filename": name,
                    "left_size_bytes": left_row.get("size_bytes"),
                    "right_size_bytes": right_row.get("size_bytes"),
                    "left_relative_path": left_row.get("relative_path"),
                    "right_relative_path": right_row.get("relative_path"),
                }
            )

        left_hash = left_row.get("sha1")
        right_hash = right_row.get("sha1")
        if left_hash is not None and right_hash is not None and left_hash != right_hash:
            same_name_different_hash.append(
                {
                    "filename": name,
                    "left_sha1": left_hash,
                    "right_sha1": right_hash,
                }
            )

        if normalize_text(left_row.get("accession")) != normalize_text(right_row.get("accession")):
            same_name_different_accession.append(
                {
                    "filename": name,
                    "left_accession": left_row.get("accession"),
                    "right_accession": right_row.get("accession"),
                }
            )

        if normalize_text(left_row.get("relative_path")) != normalize_text(right_row.get("relative_path")):
            same_name_different_path.append(
                {
                    "filename": name,
                    "left_relative_path": left_row.get("relative_path"),
                    "right_relative_path": right_row.get("relative_path"),
                }
            )

        if normalize_text(left_row.get("study_date")) != normalize_text(right_row.get("study_date")):
            same_name_different_study_date.append(
                {
                    "filename": name,
                    "left_study_date": left_row.get("study_date"),
                    "right_study_date": right_row.get("study_date"),
                }
            )

    left_read_failures = [row for row in left_records if not row.get("dicom_read_ok", False)]
    right_read_failures = [row for row in right_records if not row.get("dicom_read_ok", False)]

    summary = {
        "left_source": left.get("root") or left.get("manifest_path") or "left",
        "right_source": right.get("root") or right.get("manifest_path") or "right",
        "left_file_count": len(left_records),
        "right_file_count": len(right_records),
        "left_unique_filenames": len(left_name_set),
        "right_unique_filenames": len(right_name_set),
        "left_unique_stems": len(left_stem_set),
        "right_unique_stems": len(right_stem_set),
        "left_unique_accessions": len(left_acc_set),
        "right_unique_accessions": len(right_acc_set),
        "only_left_filenames": len(only_left_names),
        "only_right_filenames": len(only_right_names),
        "only_left_stems": len(only_left_stems),
        "only_right_stems": len(only_right_stems),
        "only_left_accessions": len(only_left_acc),
        "only_right_accessions": len(only_right_acc),
        "duplicate_filenames_left": len(duplicate_names_left),
        "duplicate_filenames_right": len(duplicate_names_right),
        "duplicate_stems_left": len(duplicate_stems_left),
        "duplicate_stems_right": len(duplicate_stems_right),
        "duplicate_accessions_left": len(duplicate_acc_left),
        "duplicate_accessions_right": len(duplicate_acc_right),
        "same_name_different_size": len(same_name_different_size),
        "same_name_different_hash": len(same_name_different_hash),
        "same_name_different_accession": len(same_name_different_accession),
        "same_name_different_path": len(same_name_different_path),
        "same_name_different_study_date": len(same_name_different_study_date),
        "left_dicom_read_failures": len(left_read_failures),
        "right_dicom_read_failures": len(right_read_failures),
        "left_hash_enabled": bool(left.get("hash_enabled")),
        "right_hash_enabled": bool(right.get("hash_enabled")),
    }

    details = {
        "only_left_filenames": [
            first_or_none(left_by_name[name]) for name in only_left_names
        ],
        "only_right_filenames": [
            first_or_none(right_by_name[name]) for name in only_right_names
        ],
        "only_left_stems": [
            first_or_none(left_by_stem[name]) for name in only_left_stems
        ],
        "only_right_stems": [
            first_or_none(right_by_stem[name]) for name in only_right_stems
        ],
        "only_left_accessions": [
            first_or_none(left_by_acc[name]) for name in only_left_acc
        ],
        "only_right_accessions": [
            first_or_none(right_by_acc[name]) for name in only_right_acc
        ],
        "duplicate_filenames_left": [
            {"filename": name, "count": len(left_by_name[name])} for name in duplicate_names_left
        ],
        "duplicate_filenames_right": [
            {"filename": name, "count": len(right_by_name[name])} for name in duplicate_names_right
        ],
        "duplicate_stems_left": [
            {"stem": name, "count": len(left_by_stem[name])} for name in duplicate_stems_left
        ],
        "duplicate_stems_right": [
            {"stem": name, "count": len(right_by_stem[name])} for name in duplicate_stems_right
        ],
        "duplicate_accessions_left": [
            {"accession": name, "count": len(left_by_acc[name])} for name in duplicate_acc_left
        ],
        "duplicate_accessions_right": [
            {"accession": name, "count": len(right_by_acc[name])} for name in duplicate_acc_right
        ],
        "same_name_different_size": same_name_different_size,
        "same_name_different_hash": same_name_different_hash,
        "same_name_different_accession": same_name_different_accession,
        "same_name_different_path": same_name_different_path,
        "same_name_different_study_date": same_name_different_study_date,
        "left_dicom_read_failures": left_read_failures,
        "right_dicom_read_failures": right_read_failures,
    }
    return summary, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_preview(title: str, rows: list[dict[str, Any]], max_preview: int) -> None:
    print(f"\n[{title}] count={len(rows)}")
    for row in rows[:max_preview]:
        compact = ", ".join(f"{k}={row[k]}" for k in row.keys())
        print(f"  {compact}")
    if len(rows) > max_preview:
        print(f"  ... {len(rows) - max_preview} more")


def run_scan(args: argparse.Namespace) -> None:
    manifest = scan_directory(Path(args.input).expanduser(), compute_hash=args.hash)
    manifest["manifest_path"] = str(Path(args.output).expanduser())
    save_manifest(manifest, Path(args.output).expanduser())
    print(f"Saved manifest: {args.output}")
    print(f"file_count={manifest['file_count']}")
    print(f"hash_enabled={manifest['hash_enabled']}")


def run_compare(args: argparse.Namespace) -> None:
    left = load_manifest_or_scan(args.left, compute_hash=args.hash)
    right = load_manifest_or_scan(args.right, compute_hash=args.hash)

    summary, details = compare_manifests(left, right)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name, rows in details.items():
        write_csv(output_dir / f"{name}.csv", rows)

    print("[Summary]")
    for key, value in summary.items():
        print(f"{key}={value}")

    for bucket in [
        "only_left_filenames",
        "only_right_filenames",
        "only_left_accessions",
        "only_right_accessions",
        "same_name_different_size",
        "same_name_different_hash",
        "same_name_different_accession",
        "left_dicom_read_failures",
        "right_dicom_read_failures",
    ]:
        print_preview(bucket, details[bucket], max_preview=args.max_preview)

    print(f"\nOutputs written to: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "scan":
        run_scan(args)
    elif args.command == "compare":
        run_compare(args)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
