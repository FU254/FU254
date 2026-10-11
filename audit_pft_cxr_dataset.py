#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom


SCRIPT_DIR = Path(__file__).resolve().parent

DR_XLSX_NAME = "DR\u5df2\u7b5b\u90092019-2023.xlsx"
PFT_XLSX_NAME = "\u526f\u672c2025.7.14-\u57fa\u7840\u4fe1\u606f+\u80ba\u529f\u80fd-\u90ed\u5c7f\u8001\u5e08.xlsx"
DEFAULT_LABEL_COLUMN = "\u5f25\u6563\u529f\u80fd\u6b63\u5e380\u5f02\u5e381"

COL_SEQ = "\u5e8f\u53f7"
COL_ACCESSION = "\u653e\u5c04\u7f16\u53f7"
COL_PATIENT_NAME = "\u75c5\u4eba\u59d3\u540d"
COL_NAME = "\u59d3\u540d"
COL_EXAM_TIME = "\u68c0\u67e5\u65f6\u95f4"
COL_EXAM_DATE = "\u68c0\u67e5\u65e5\u671f"
COL_EXAM_NAME = "\u68c0\u67e5"
COL_PATIENT_TYPE = "\u75c5\u4eba\u7c7b\u578b"
COL_DEPARTMENT = "\u7533\u8bf7\u90e8\u95e8"
COL_AGE = "\u5e74\u9f84"
COL_CLINICAL_DX = "\u4e34\u5e8a\u8bca\u65ad"
COL_REPORT_DX = "\u5f71\u50cf\u5b66\u8bca\u65ad"
COL_INPATIENT_ID = "\u4f4f\u9662\u53f7"
COL_SEX = "\u6027\u522b"
COL_HEIGHT = "\u8eab\u9ad8"
COL_WEIGHT = "\u4f53\u91cd"
COL_SMOKING = "\u5438\u70df"

TEXT_NORMAL = "\u6b63\u5e38"
TEXT_ABNORMAL = "\u5f02\u5e38"
TEXT_NEGATIVE = "\u9634\u6027"
TEXT_POSITIVE = "\u9633\u6027"
TEXT_NO = "\u65e0"
TEXT_YES = "\u6709"

DEFAULT_DR_XLSX_CANDIDATES = [
    Path(r"D:\99") / DR_XLSX_NAME,
    SCRIPT_DIR / DR_XLSX_NAME,
    Path("/home/cfu/.virtualenvs/DR/TEST") / DR_XLSX_NAME,
]
DEFAULT_PFT_XLSX_CANDIDATES = [
    Path(r"D:\99") / PFT_XLSX_NAME,
    SCRIPT_DIR / PFT_XLSX_NAME,
    Path("/tmp/pycharm_project_601") / PFT_XLSX_NAME,
]
DEFAULT_DICOM_DIR_CANDIDATES = [
    Path(r"D:\99\NEW.DR"),
    SCRIPT_DIR / "NEW.DR",
    Path("/home/cfu/.virtualenvs/DR/TEST/NEW.DR"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CXR-PFT pairing quality before model training."
    )
    parser.add_argument("--dr-xlsx", type=str, default=None, help="Chest x-ray metadata xlsx.")
    parser.add_argument("--pft-xlsx", type=str, default=None, help="Pulmonary function xlsx.")
    parser.add_argument("--dicom-dir", type=str, default=None, help="Directory containing DICOM files.")
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="Target label column name or substring.",
    )
    parser.add_argument(
        "--pft-sheet",
        type=str,
        default="0",
        help="PFT sheet name or integer index. Default: 0",
    )
    parser.add_argument(
        "--dr-sheet",
        type=str,
        default="0",
        help="DR sheet name or integer index. Default: 0",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "audit_outputs"),
        help="Directory for markdown/json/csv outputs.",
    )
    parser.add_argument(
        "--max-dicoms",
        type=int,
        default=None,
        help="Optional cap for quick audits.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N categories to include in report tables.",
    )
    parser.add_argument(
        "--pft-date-column",
        type=str,
        default=None,
        help="Optional PFT date column for time-gap auditing.",
    )
    return parser.parse_args()


def resolve_existing_path(explicit: str | None, candidates: list[Path], kind: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"{kind} not found: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No existing {kind} found in candidates: {candidates}")


def parse_sheet_arg(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def flatten_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        parts = []
        for item in column:
            text = str(item).strip()
            if not text or text.lower().startswith("unnamed:"):
                continue
            parts.append(text)
        return "|".join(dict.fromkeys(parts))
    text = str(column).strip()
    return "" if text.lower().startswith("unnamed:") else text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [flatten_column_name(col) for col in out.columns]
    out.columns = [re.sub(r"\s+", "", col) for col in out.columns]
    return out


def load_excel_best_header(
    path: Path,
    sheet: str | int,
    expected_terms: list[str],
) -> tuple[pd.DataFrame, str]:
    header_candidates: list[Any] = [0, 1, [0, 1]]
    best_df = None
    best_header = None
    best_score = -1
    for header in header_candidates:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=header)
        except Exception:
            continue
        df = normalize_columns(df)
        score = 0
        columns = list(df.columns)
        for term in expected_terms:
            score += int(any(term in col for col in columns))
        non_empty = sum(bool(col) for col in columns)
        score = score * 1000 + non_empty
        if score > best_score:
            best_df = df
            best_header = header
            best_score = score
    if best_df is None:
        raise RuntimeError(f"Failed to read {path}")
    return best_df, str(best_header)


def find_column(columns: list[str], patterns: list[str], required: bool = True) -> str | None:
    for pattern in patterns:
        for col in columns:
            if col == pattern:
                return col
    for pattern in patterns:
        for col in columns:
            if pattern in col:
                return col
    if required:
        raise KeyError(f"Could not find column matching any of: {patterns}")
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d\.\-]+", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_excel_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="datetime64[ns]")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio >= 0.8:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def safe_value_counts(series: pd.Series, top_n: int) -> dict[str, int]:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return {}
    return cleaned.value_counts().head(top_n).astype(int).to_dict()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


@dataclass
class ColumnMap:
    dr_seq: str
    dr_accession: str
    dr_name: str | None
    dr_exam_time: str | None
    dr_exam_name: str | None
    dr_patient_type: str | None
    dr_department: str | None
    dr_age: str | None
    dr_clinical_dx: str | None
    dr_report_dx: str | None
    pft_seq: str
    pft_name: str | None
    pft_inpatient_id: str | None
    pft_sex: str | None
    pft_age: str | None
    pft_height: str | None
    pft_weight: str | None
    pft_smoking: str | None
    pft_label: str
    pft_date: str | None


def infer_column_map(dr_df: pd.DataFrame, pft_df: pd.DataFrame, label_column: str, pft_date_column: str | None) -> ColumnMap:
    dr_columns = list(dr_df.columns)
    pft_columns = list(pft_df.columns)
    pft_label = find_column(pft_columns, [label_column])
    pft_date = None
    if pft_date_column:
        pft_date = find_column(pft_columns, [pft_date_column], required=False)
    return ColumnMap(
        dr_seq=find_column(dr_columns, [COL_SEQ]),
        dr_accession=find_column(dr_columns, [COL_ACCESSION, "AccessionNumber"]),
        dr_name=find_column(dr_columns, [COL_PATIENT_NAME, COL_NAME], required=False),
        dr_exam_time=find_column(dr_columns, [COL_EXAM_TIME, COL_EXAM_DATE], required=False),
        dr_exam_name=find_column(dr_columns, [COL_EXAM_NAME], required=False),
        dr_patient_type=find_column(dr_columns, [COL_PATIENT_TYPE], required=False),
        dr_department=find_column(dr_columns, [COL_DEPARTMENT], required=False),
        dr_age=find_column(dr_columns, [COL_AGE], required=False),
        dr_clinical_dx=find_column(dr_columns, [COL_CLINICAL_DX], required=False),
        dr_report_dx=find_column(dr_columns, [COL_REPORT_DX], required=False),
        pft_seq=find_column(pft_columns, [COL_SEQ]),
        pft_name=find_column(pft_columns, [COL_NAME], required=False),
        pft_inpatient_id=find_column(pft_columns, [COL_INPATIENT_ID], required=False),
        pft_sex=find_column(pft_columns, [COL_SEX], required=False),
        pft_age=find_column(pft_columns, [COL_AGE], required=False),
        pft_height=find_column(pft_columns, [COL_HEIGHT], required=False),
        pft_weight=find_column(pft_columns, [COL_WEIGHT], required=False),
        pft_smoking=find_column(pft_columns, [COL_SMOKING], required=False),
        pft_label=pft_label,
        pft_date=pft_date,
    )


def clean_binary_like_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    mapping = {
        TEXT_NORMAL: 0,
        TEXT_ABNORMAL: 1,
        TEXT_NEGATIVE: 0,
        TEXT_POSITIVE: 1,
        TEXT_NO: 0,
        TEXT_YES: 1,
    }
    out = series.astype(str).str.strip().replace(mapping)
    return pd.to_numeric(out, errors="coerce")


def clean_join_key(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return cleaned


def build_patient_key(df: pd.DataFrame) -> pd.Series:
    name = df.get("pft_name_final")
    inpatient = df.get("pft_inpatient_id")
    dr_name = df.get("dr_name")
    dicom_pid = df.get("dicom_patient_id")
    dicom_name = df.get("dicom_patient_name")

    if name is not None and inpatient is not None:
        key = name.fillna("").astype(str).str.strip() + "|" + inpatient.fillna("").astype(str).str.strip()
        key = key.str.strip("|")
        key = key.replace("", np.nan)
        if key.notna().any():
            return key

    for candidate in [name, dr_name, dicom_name, dicom_pid]:
        if candidate is not None:
            key = candidate.astype(str).str.strip().replace("", np.nan)
            if key.notna().any():
                return key

    return pd.Series(np.arange(len(df)), index=df.index).astype(str)


def scan_dicoms(dicom_dir: Path, max_dicoms: int | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    files = sorted(dicom_dir.rglob("*.dcm"))
    if max_dicoms is not None:
        files = files[:max_dicoms]
    tags = [
        "AccessionNumber",
        "StudyDate",
        "PatientID",
        "PatientName",
        "PatientSex",
        "PatientAge",
        "ViewPosition",
        "SeriesDescription",
        "BodyPartExamined",
        "Rows",
        "Columns",
        "Modality",
    ]
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=tags, force=True)
            rows.append(
                {
                    "dicom_path": str(path),
                    "accession": str(ds.get("AccessionNumber", "")).strip() or np.nan,
                    "study_date_raw": str(ds.get("StudyDate", "")).strip() or np.nan,
                    "dicom_patient_id": str(ds.get("PatientID", "")).strip() or np.nan,
                    "dicom_patient_name": str(ds.get("PatientName", "")).strip() or np.nan,
                    "dicom_patient_sex": str(ds.get("PatientSex", "")).strip() or np.nan,
                    "dicom_patient_age": str(ds.get("PatientAge", "")).strip() or np.nan,
                    "view_position": str(ds.get("ViewPosition", "")).strip() or np.nan,
                    "series_description": str(ds.get("SeriesDescription", "")).strip() or np.nan,
                    "body_part_examined": str(ds.get("BodyPartExamined", "")).strip() or np.nan,
                    "rows": ds.get("Rows", np.nan),
                    "cols": ds.get("Columns", np.nan),
                    "modality": str(ds.get("Modality", "")).strip() or np.nan,
                    "dicom_read_ok": True,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dicom_path": str(path),
                    "accession": np.nan,
                    "dicom_read_ok": False,
                    "dicom_error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    if "study_date_raw" in df.columns:
        df["study_date"] = pd.to_datetime(df["study_date_raw"], format="%Y%m%d", errors="coerce")
    return df


def summarize_numeric_by_label(df: pd.DataFrame, label_col: str, feature_col: str) -> dict[str, Any]:
    subset = df[[label_col, feature_col]].copy()
    subset = subset.dropna()
    if subset.empty:
        return {}
    result: dict[str, Any] = {}
    for label_value, group in subset.groupby(label_col):
        values = group[feature_col]
        result[str(label_value)] = {
            "n": int(values.shape[0]),
            "mean": float(values.mean()),
            "std": float(values.std()) if values.shape[0] > 1 else 0.0,
            "median": float(values.median()),
        }
    return result


def create_merged_table(
    dicom_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    pft_df: pd.DataFrame,
    columns: ColumnMap,
) -> pd.DataFrame:
    dr_subset = dr_df.rename(
        columns={
            columns.dr_seq: "dr_seq",
            columns.dr_accession: "accession",
        }
    ).copy()
    pft_subset = pft_df.rename(
        columns={
            columns.pft_seq: "pft_seq",
            columns.pft_label: "target_label_raw",
        }
    ).copy()

    optional_renames = {
        columns.dr_name: "dr_name",
        columns.dr_exam_time: "dr_exam_time_raw",
        columns.dr_exam_name: "dr_exam_name",
        columns.dr_patient_type: "dr_patient_type",
        columns.dr_department: "dr_department",
        columns.dr_age: "dr_age_raw",
        columns.dr_clinical_dx: "dr_clinical_dx",
        columns.dr_report_dx: "dr_report_dx",
        columns.pft_name: "pft_name_final",
        columns.pft_inpatient_id: "pft_inpatient_id",
        columns.pft_sex: "pft_sex",
        columns.pft_age: "pft_age_raw",
        columns.pft_height: "pft_height_raw",
        columns.pft_weight: "pft_weight_raw",
        columns.pft_smoking: "pft_smoking",
        columns.pft_date: "pft_date_raw",
    }
    optional_renames = {k: v for k, v in optional_renames.items() if k is not None}
    dr_subset = dr_subset.rename(columns=optional_renames)
    pft_subset = pft_subset.rename(columns=optional_renames)

    for frame, seq_col in [(dr_subset, "dr_seq"), (pft_subset, "pft_seq")]:
        frame[seq_col] = clean_join_key(frame[seq_col])
    if "accession" in dr_subset.columns:
        dr_subset["accession"] = clean_join_key(dr_subset["accession"])
    if "accession" in dicom_df.columns:
        dicom_df["accession"] = clean_join_key(dicom_df["accession"])

    merged = dicom_df.merge(dr_subset, on="accession", how="left", suffixes=("", "_dr"))
    merged = merged.merge(pft_subset, left_on="dr_seq", right_on="pft_seq", how="left", suffixes=("", "_pft"))

    if "dr_exam_time_raw" in merged.columns:
        merged["dr_exam_time"] = parse_excel_datetime(merged["dr_exam_time_raw"])
    if "pft_date_raw" in merged.columns:
        merged["pft_date"] = parse_excel_datetime(merged["pft_date_raw"])
    if "dr_exam_time" in merged.columns and "pft_date" in merged.columns:
        merged["time_gap_days"] = (merged["dr_exam_time"] - merged["pft_date"]).dt.days.abs()

    for raw_col, clean_col in [
        ("dr_age_raw", "dr_age"),
        ("pft_age_raw", "pft_age"),
        ("pft_height_raw", "pft_height"),
        ("pft_weight_raw", "pft_weight"),
        ("dicom_patient_age", "dicom_patient_age_num"),
    ]:
        if raw_col in merged.columns:
            merged[clean_col] = to_numeric(merged[raw_col])

    merged["target_label"] = clean_binary_like_label(merged["target_label_raw"])
    merged["patient_key"] = build_patient_key(merged)
    return merged


def build_summary(
    dicom_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    pft_df: pd.DataFrame,
    merged: pd.DataFrame,
    label_column: str,
    top_n: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    summary["paths"] = {
        "dicom_rows_scanned": int(len(dicom_df)),
        "dr_rows": int(len(dr_df)),
        "pft_rows": int(len(pft_df)),
    }

    summary["pairing"] = {
        "dicom_read_ok": int(dicom_df.get("dicom_read_ok", pd.Series(dtype=bool)).fillna(False).sum()),
        "dicoms_with_accession": int(merged["accession"].notna().sum()),
        "matched_to_dr": int(merged["dr_seq"].notna().sum()),
        "matched_to_pft": int(merged["pft_seq"].notna().sum()),
        "matched_with_valid_label": int(merged["target_label"].notna().sum()),
        "dr_duplicate_accessions": int(dr_df.duplicated(subset=[COL_ACCESSION]).sum()) if COL_ACCESSION in dr_df.columns else None,
    }

    labeled = merged[merged["target_label"].notna()].copy()
    label_counts = {}
    if not labeled.empty:
        counts = labeled["target_label"].value_counts(dropna=False).sort_index()
        label_counts = {str(int(k)) if float(k).is_integer() else str(k): int(v) for k, v in counts.items()}
    summary["label_distribution"] = {
        "label_column": label_column,
        "counts": label_counts,
        "positive_rate": float((labeled["target_label"] == 1).mean()) if set(labeled["target_label"].dropna().unique()).issubset({0, 1}) and not labeled.empty else None,
        "num_classes": int(labeled["target_label"].nunique()) if not labeled.empty else 0,
    }

    patient_counts = labeled["patient_key"].value_counts(dropna=True) if not labeled.empty else pd.Series(dtype=int)
    patient_conflicts = 0
    if not labeled.empty:
        patient_conflicts = int((labeled.groupby("patient_key")["target_label"].nunique() > 1).sum())
    summary["patients"] = {
        "unique_patients_labeled": int(labeled["patient_key"].nunique()) if not labeled.empty else 0,
        "images_per_patient_mean": float(patient_counts.mean()) if not patient_counts.empty else None,
        "images_per_patient_median": float(patient_counts.median()) if not patient_counts.empty else None,
        "images_per_patient_max": int(patient_counts.max()) if not patient_counts.empty else None,
        "patients_with_label_conflicts": patient_conflicts,
    }

    summary["dicom_metadata"] = {
        "view_position_top": safe_value_counts(merged.get("view_position", pd.Series(dtype=object)), top_n),
        "modality_top": safe_value_counts(merged.get("modality", pd.Series(dtype=object)), top_n),
        "patient_type_top": safe_value_counts(merged.get("dr_patient_type", pd.Series(dtype=object)), top_n),
        "department_top": safe_value_counts(merged.get("dr_department", pd.Series(dtype=object)), top_n),
        "study_date_min": dicom_df.get("study_date", pd.Series(dtype="datetime64[ns]")).min(),
        "study_date_max": dicom_df.get("study_date", pd.Series(dtype="datetime64[ns]")).max(),
    }

    missingness = {}
    for col in ["pft_sex", "pft_age", "pft_height", "pft_weight", "pft_smoking", "dr_patient_type", "dr_clinical_dx", "dr_report_dx"]:
        if col in labeled.columns:
            missingness[col] = float(labeled[col].isna().mean())
    summary["missingness_labeled"] = missingness

    numeric_profiles = {}
    for col in ["pft_age", "pft_height", "pft_weight", "dr_age", "dicom_patient_age_num"]:
        if col in labeled.columns:
            profile = summarize_numeric_by_label(labeled, "target_label", col)
            if profile:
                numeric_profiles[col] = profile
    summary["numeric_profiles_by_label"] = numeric_profiles

    if "time_gap_days" in labeled.columns:
        gap = labeled["time_gap_days"].dropna()
        summary["time_gap_days"] = {
            "available_pairs": int(gap.shape[0]),
            "median": float(gap.median()) if not gap.empty else None,
            "p90": float(gap.quantile(0.9)) if not gap.empty else None,
            "le_7_days": int((gap <= 7).sum()) if not gap.empty else None,
            "le_30_days": int((gap <= 30).sum()) if not gap.empty else None,
        }
    else:
        summary["time_gap_days"] = {"available_pairs": 0}

    warnings: list[str] = []
    valid_label_count = summary["pairing"]["matched_with_valid_label"]
    scanned = max(summary["paths"]["dicom_rows_scanned"], 1)
    if valid_label_count / scanned < 0.6:
        warnings.append("Matched labeled pairs are less than 60% of scanned DICOMs; attrition may be hurting training.")
    positive_rate = summary["label_distribution"].get("positive_rate")
    if positive_rate is not None and (positive_rate < 0.15 or positive_rate > 0.85):
        warnings.append("Binary label is strongly imbalanced; default threshold F1 will usually look poor.")
    if patient_conflicts > 0:
        warnings.append("Some patients carry more than one label across images; verify time alignment or label consistency.")
    smoking_missing = missingness.get("pft_smoking")
    if smoking_missing is not None and smoking_missing > 0.3:
        warnings.append("Smoking is missing in more than 30% of labeled pairs; tabular baselines will be unstable.")
    if summary["time_gap_days"].get("available_pairs", 0) == 0:
        warnings.append("No PFT date column was available, so time-gap noise could not be audited.")
    summary["warnings"] = warnings
    return summary


def render_markdown(summary: dict[str, Any], columns: ColumnMap, header_choices: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# CXR-PFT Data Audit")
    lines.append("")
    lines.append("## Header Selection")
    lines.append(f"- DR header strategy: `{header_choices['dr']}`")
    lines.append(f"- PFT header strategy: `{header_choices['pft']}`")
    lines.append(f"- Target label column: `{columns.pft_label}`")
    lines.append("")

    lines.append("## Pairing Summary")
    for key, value in summary["pairing"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Label Distribution")
    lines.append(f"- label_column: `{summary['label_distribution']['label_column']}`")
    lines.append(f"- counts: `{summary['label_distribution']['counts']}`")
    lines.append(f"- positive_rate: `{summary['label_distribution']['positive_rate']}`")
    lines.append(f"- num_classes: `{summary['label_distribution']['num_classes']}`")
    lines.append("")

    lines.append("## Patient Leakage Risk")
    for key, value in summary["patients"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## DICOM / Clinical Metadata")
    lines.append(f"- view_position_top: `{summary['dicom_metadata']['view_position_top']}`")
    lines.append(f"- patient_type_top: `{summary['dicom_metadata']['patient_type_top']}`")
    lines.append(f"- department_top: `{summary['dicom_metadata']['department_top']}`")
    lines.append(f"- study_date_min: `{summary['dicom_metadata']['study_date_min']}`")
    lines.append(f"- study_date_max: `{summary['dicom_metadata']['study_date_max']}`")
    lines.append("")

    lines.append("## Missingness On Labeled Pairs")
    for key, value in summary["missingness_labeled"].items():
        lines.append(f"- {key}: {value:.3f}")
    lines.append("")

    lines.append("## Time Gap")
    for key, value in summary["time_gap_days"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Warnings")
    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- No obvious high-severity data warnings were triggered by the audit heuristics.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    dr_path = resolve_existing_path(args.dr_xlsx, DEFAULT_DR_XLSX_CANDIDATES, "DR xlsx")
    pft_path = resolve_existing_path(args.pft_xlsx, DEFAULT_PFT_XLSX_CANDIDATES, "PFT xlsx")
    dicom_dir = resolve_existing_path(args.dicom_dir, DEFAULT_DICOM_DIR_CANDIDATES, "DICOM directory")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dr_sheet = parse_sheet_arg(args.dr_sheet)
    pft_sheet = parse_sheet_arg(args.pft_sheet)

    dr_df, dr_header_choice = load_excel_best_header(
        dr_path,
        dr_sheet,
        expected_terms=[COL_SEQ, COL_ACCESSION, COL_PATIENT_NAME, COL_EXAM_TIME, COL_PATIENT_TYPE],
    )
    pft_df, pft_header_choice = load_excel_best_header(
        pft_path,
        pft_sheet,
        expected_terms=[COL_SEQ, COL_INPATIENT_ID, COL_SMOKING, args.label_column],
    )
    columns = infer_column_map(dr_df, pft_df, args.label_column, args.pft_date_column)

    dicom_df = scan_dicoms(dicom_dir, args.max_dicoms)
    merged = create_merged_table(dicom_df, dr_df, pft_df, columns)
    summary = build_summary(dicom_df, dr_df, pft_df, merged, args.label_column, args.top_n)

    report_md = render_markdown(
        summary,
        columns,
        {"dr": dr_header_choice, "pft": pft_header_choice},
    )

    matched_csv = output_dir / "matched_records.csv"
    summary_json = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    merged.to_csv(matched_csv, index=False, encoding="utf-8-sig")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)
    report_path.write_text(report_md, encoding="utf-8")

    print(report_md)
    print("")
    print(f"Saved report to: {report_path}")
    print(f"Saved summary json to: {summary_json}")
    print(f"Saved merged table to: {matched_csv}")


if __name__ == "__main__":
    main()
