#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

from audit_pft_cxr_dataset import (
    COL_ACCESSION,
    COL_SEQ,
    DEFAULT_DICOM_DIR_CANDIDATES,
    DEFAULT_DR_XLSX_CANDIDATES,
    DEFAULT_PFT_XLSX_CANDIDATES,
    SCRIPT_DIR,
    clean_binary_like_label,
    clean_join_key,
    find_column,
    json_ready,
    load_excel_best_header,
)


DEFAULT_TRAIN_SCRIPTS = [
    SCRIPT_DIR / "EVA-X_train-tiny-弥散.py",
    SCRIPT_DIR / "EVA-X_train-tiny-气肿.py",
    SCRIPT_DIR / "EVA-X_train-tiny-阻塞.py",
    SCRIPT_DIR / "EVA-X_train-tiny-限制.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect image-label mappings used by EVA-X training scripts and count duplicates."
    )
    parser.add_argument(
        "--train-scripts",
        nargs="*",
        default=[str(path) for path in DEFAULT_TRAIN_SCRIPTS],
        help="Training script paths to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "label_duplicate_audit"),
        help="Directory used to save audit results.",
    )
    parser.add_argument(
        "--max-dicoms",
        type=int,
        default=None,
        help="Optional cap for quick validation.",
    )
    return parser.parse_args()


def extract_config(script_path: Path) -> dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    wanted = {"EXCEL_PATH1", "EXCEL_PATH2", "DICOM_DIR", "LABEL_COLUMN"}
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in wanted:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
        except Exception:
            continue
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"{script_path} missing config values: {sorted(missing)}")
    return values


def resolve_input_path(raw_value: str, fallback_candidates: list[Path], is_dir: bool) -> Path:
    candidates: list[Path] = []
    raw_path = Path(str(raw_value))
    candidates.append(raw_path)
    candidates.append(SCRIPT_DIR / raw_path.name)
    candidates.append(Path(r"D:\99") / raw_path.name)
    candidates.extend(fallback_candidates)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_dir() == is_dir:
            return candidate

    kind = "directory" if is_dir else "file"
    raise FileNotFoundError(f"Could not resolve {kind} from {raw_value!r}")


def load_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    dr_path = resolve_input_path(config["EXCEL_PATH1"], DEFAULT_DR_XLSX_CANDIDATES, is_dir=False)
    pft_path = resolve_input_path(config["EXCEL_PATH2"], DEFAULT_PFT_XLSX_CANDIDATES, is_dir=False)

    dr_df, dr_header = load_excel_best_header(
        dr_path,
        0,
        expected_terms=[COL_SEQ, COL_ACCESSION],
    )
    pft_df, pft_header = load_excel_best_header(
        pft_path,
        0,
        expected_terms=[COL_SEQ, str(config["LABEL_COLUMN"])],
    )
    return dr_df, pft_df, str(dr_header), str(pft_header)


def build_first_mapping(dr_df: pd.DataFrame, pft_df: pd.DataFrame, label_column: str) -> tuple[dict[str, str], dict[str, int], dict[str, Any]]:
    dr_accession_col = find_column(list(dr_df.columns), [COL_ACCESSION, "AccessionNumber"])
    dr_seq_col = find_column(list(dr_df.columns), [COL_SEQ])
    pft_seq_col = find_column(list(pft_df.columns), [COL_SEQ])
    pft_label_col = find_column(list(pft_df.columns), [label_column])

    dr_pairs = dr_df[[dr_accession_col, dr_seq_col]].copy()
    dr_pairs.columns = ["accession", "seq"]
    dr_pairs["accession"] = clean_join_key(dr_pairs["accession"])
    dr_pairs["seq"] = clean_join_key(dr_pairs["seq"])
    dr_pairs = dr_pairs.dropna(subset=["accession", "seq"])

    pft_pairs = pft_df[[pft_seq_col, pft_label_col]].copy()
    pft_pairs.columns = ["seq", "label_raw"]
    pft_pairs["seq"] = clean_join_key(pft_pairs["seq"])
    pft_pairs["label"] = clean_binary_like_label(pft_pairs["label_raw"])
    pft_pairs = pft_pairs.dropna(subset=["seq"])

    accession_to_seq: dict[str, str] = {}
    for row in dr_pairs.itertuples(index=False):
        if row.accession not in accession_to_seq:
            accession_to_seq[row.accession] = row.seq

    seq_to_label: dict[str, int] = {}
    for row in pft_pairs.itertuples(index=False):
        if row.seq in seq_to_label:
            continue
        if pd.isna(row.label):
            continue
        label_int = int(row.label)
        if label_int in {0, 1}:
            seq_to_label[row.seq] = label_int

    stats = {
        "dr_rows_with_accession_seq": int(len(dr_pairs)),
        "pft_rows_with_seq": int(len(pft_pairs)),
        "source_duplicate_accessions": int(dr_pairs.duplicated(subset=["accession"]).sum()),
        "source_accessions_with_multiple_rows": int((dr_pairs["accession"].value_counts() > 1).sum()),
        "source_accessions_with_multiple_seqs": int((dr_pairs.groupby("accession")["seq"].nunique() > 1).sum()),
        "source_duplicate_pft_seqs": int(pft_pairs.duplicated(subset=["seq"]).sum()),
        "source_pft_seqs_with_multiple_rows": int((pft_pairs["seq"].value_counts() > 1).sum()),
        "source_pft_seqs_with_multiple_valid_labels": int(
            (pft_pairs.dropna(subset=["label"]).groupby("seq")["label"].nunique() > 1).sum()
        ),
    }
    return accession_to_seq, seq_to_label, stats


def collect_training_rows(
    dicom_dir: Path,
    accession_to_seq: dict[str, str],
    seq_to_label: dict[str, int],
    max_dicoms: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {
        "dicoms_scanned": 0,
        "dicom_read_errors": 0,
        "missing_accession": 0,
        "missing_seq": 0,
        "missing_or_invalid_label": 0,
    }

    dicom_files = sorted(dicom_dir.rglob("*.dcm"))
    if max_dicoms is not None:
        dicom_files = dicom_files[:max_dicoms]

    for dicom_path in dicom_files:
        counts["dicoms_scanned"] += 1
        try:
            ds = pydicom.dcmread(
                str(dicom_path),
                stop_before_pixels=True,
                specific_tags=["AccessionNumber"],
                force=True,
            )
        except Exception:
            counts["dicom_read_errors"] += 1
            continue

        accession = str(ds.get("AccessionNumber", "")).strip()
        accession = accession or None
        if not accession:
            counts["missing_accession"] += 1
            continue

        seq = accession_to_seq.get(accession)
        if not seq:
            counts["missing_seq"] += 1
            continue

        label = seq_to_label.get(seq)
        if label is None:
            counts["missing_or_invalid_label"] += 1
            continue

        rows.append(
            {
                "dicom_path": str(dicom_path),
                "filename": dicom_path.name,
                "accession": accession,
                "seq": seq,
                "label": int(label),
            }
        )

    return pd.DataFrame(rows), counts


def duplicate_rows(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df.duplicated(subset=subset, keep=False)].copy()
    return out.sort_values(subset + ["dicom_path"]).reset_index(drop=True)


def build_task_summary(
    task_name: str,
    label_column: str,
    matched_df: pd.DataFrame,
    source_stats: dict[str, Any],
    scan_stats: dict[str, int],
) -> dict[str, Any]:
    if matched_df.empty:
        label_counts: dict[str, int] = {}
        accession_conflicts = 0
        seq_conflicts = 0
    else:
        label_counts = matched_df["label"].value_counts().sort_index().astype(int).to_dict()
        accession_conflicts = int((matched_df.groupby("accession")["label"].nunique() > 1).sum())
        seq_conflicts = int((matched_df.groupby("seq")["label"].nunique() > 1).sum())

    summary = {
        "task": task_name,
        "label_column": label_column,
        "matched_rows": int(len(matched_df)),
        "unique_filenames": int(matched_df["filename"].nunique()) if not matched_df.empty else 0,
        "unique_accessions": int(matched_df["accession"].nunique()) if not matched_df.empty else 0,
        "unique_seqs": int(matched_df["seq"].nunique()) if not matched_df.empty else 0,
        "duplicate_filename_rows": int(matched_df.duplicated(subset=["filename"]).sum()) if not matched_df.empty else 0,
        "duplicate_accession_rows": int(matched_df.duplicated(subset=["accession"]).sum()) if not matched_df.empty else 0,
        "duplicate_seq_rows": int(matched_df.duplicated(subset=["seq"]).sum()) if not matched_df.empty else 0,
        "duplicate_accession_label_rows": int(matched_df.duplicated(subset=["accession", "label"]).sum()) if not matched_df.empty else 0,
        "duplicate_seq_label_rows": int(matched_df.duplicated(subset=["seq", "label"]).sum()) if not matched_df.empty else 0,
        "accessions_with_multiple_labels": accession_conflicts,
        "seqs_with_multiple_labels": seq_conflicts,
        "label_counts": label_counts,
    }
    summary.update(scan_stats)
    summary.update(source_stats)
    return summary


def audit_script(script_path: Path, output_dir: Path, max_dicoms: int | None) -> dict[str, Any]:
    config = extract_config(script_path)
    dr_df, pft_df, dr_header, pft_header = load_tables(config)
    dicom_dir = resolve_input_path(config["DICOM_DIR"], DEFAULT_DICOM_DIR_CANDIDATES, is_dir=True)

    accession_to_seq, seq_to_label, source_stats = build_first_mapping(
        dr_df,
        pft_df,
        str(config["LABEL_COLUMN"]),
    )
    matched_df, scan_stats = collect_training_rows(dicom_dir, accession_to_seq, seq_to_label, max_dicoms)

    task_name = script_path.stem
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    matched_csv = task_dir / "matched_pairs.csv"
    dup_filename_csv = task_dir / "duplicate_filenames.csv"
    dup_accession_csv = task_dir / "duplicate_accessions.csv"
    dup_seq_csv = task_dir / "duplicate_seqs.csv"
    summary_json = task_dir / "summary.json"

    matched_df.to_csv(matched_csv, index=False, encoding="utf-8-sig")
    duplicate_rows(matched_df, ["filename"]).to_csv(dup_filename_csv, index=False, encoding="utf-8-sig")
    duplicate_rows(matched_df, ["accession"]).to_csv(dup_accession_csv, index=False, encoding="utf-8-sig")
    duplicate_rows(matched_df, ["seq"]).to_csv(dup_seq_csv, index=False, encoding="utf-8-sig")

    summary = build_task_summary(task_name, str(config["LABEL_COLUMN"]), matched_df, source_stats, scan_stats)
    summary["train_script"] = str(script_path)
    summary["dicom_dir"] = str(dicom_dir)
    summary["dr_header"] = dr_header
    summary["pft_header"] = pft_header
    summary["outputs"] = {
        "matched_pairs": str(matched_csv),
        "duplicate_filenames": str(dup_filename_csv),
        "duplicate_accessions": str(dup_accession_csv),
        "duplicate_seqs": str(dup_seq_csv),
    }

    summary_json.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    display = pd.DataFrame(
        [
            {
                "task": item["task"],
                "matched_rows": item["matched_rows"],
                "dup_accession_rows": item["duplicate_accession_rows"],
                "dup_seq_rows": item["duplicate_seq_rows"],
                "source_dup_accessions": item["source_duplicate_accessions"],
                "source_dup_pft_seqs": item["source_duplicate_pft_seqs"],
                "source_seq_multi_labels": item["source_pft_seqs_with_multiple_valid_labels"],
            }
            for item in summaries
        ]
    )
    print(display.to_string(index=False))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for script in args.train_scripts:
        script_path = Path(script)
        summary = audit_script(script_path, output_dir, args.max_dicoms)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(json_ready(summaries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary_table(summaries)
    print("")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
