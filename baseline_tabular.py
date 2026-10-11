#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from audit_pft_cxr_dataset import (
    COL_ACCESSION,
    COL_EXAM_TIME,
    COL_INPATIENT_ID,
    COL_NAME,
    COL_PATIENT_NAME,
    COL_PATIENT_TYPE,
    COL_SEQ,
    COL_SMOKING,
    DEFAULT_DICOM_DIR_CANDIDATES,
    DEFAULT_DR_XLSX_CANDIDATES,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_PFT_XLSX_CANDIDATES,
    TEXT_NO,
    TEXT_YES,
    build_patient_key,
    clean_binary_like_label,
    create_merged_table,
    find_column,
    infer_column_map,
    json_ready,
    load_excel_best_header,
    normalize_columns,
    parse_sheet_arg,
    resolve_existing_path,
    scan_dicoms,
)


SCRIPT_DIR = Path(__file__).resolve().parent

TEXT_MALE = "\u7537"
TEXT_FEMALE = "\u5973"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal patient-level tabular baseline for CXR-PFT dataset."
    )
    parser.add_argument("--dr-xlsx", type=str, default=None)
    parser.add_argument("--pft-xlsx", type=str, default=None)
    parser.add_argument("--dicom-dir", type=str, default=None)
    parser.add_argument("--label-column", type=str, default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--dr-sheet", type=str, default="0")
    parser.add_argument("--pft-sheet", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the remaining train_val split used as validation.",
    )
    parser.add_argument("--max-dicoms", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR / "baseline_outputs"),
    )
    return parser.parse_args()


def expected_terms_for_dr() -> list[str]:
    return [COL_SEQ, COL_ACCESSION, COL_PATIENT_NAME, COL_EXAM_TIME, COL_PATIENT_TYPE]


def expected_terms_for_pft(label_column: str) -> list[str]:
    return [COL_SEQ, COL_INPATIENT_ID, COL_SMOKING, label_column]


def expected_terms_for_pft_direct(label_column: str) -> list[str]:
    return [COL_ACCESSION, "sex", "age", "height", label_column]


def prepare_patient_level_table(merged: pd.DataFrame) -> pd.DataFrame:
    labeled = merged[merged["target_label"].notna()].copy()
    if labeled.empty:
        raise RuntimeError("No labeled rows found after merging.")

    labeled["patient_key"] = build_patient_key(labeled)
    labeled["target_label"] = pd.to_numeric(labeled["target_label"], errors="coerce")
    labeled = labeled[labeled["target_label"].isin([0, 1])].copy()
    if labeled.empty:
        raise RuntimeError("No binary-labeled rows remained after filtering.")

    patient_conflicts = labeled.groupby("patient_key")["target_label"].nunique()
    conflicting = patient_conflicts[patient_conflicts > 1]
    if not conflicting.empty:
        raise RuntimeError(
            f"Found {len(conflicting)} patients with conflicting labels; fix pairing before running baseline."
        )

    sort_columns: list[str] = []
    if "dr_exam_time" in labeled.columns:
        sort_columns.append("dr_exam_time")
    elif "study_date" in labeled.columns:
        sort_columns.append("study_date")
    if sort_columns:
        labeled = labeled.sort_values(sort_columns)

    patient_level = labeled.groupby("patient_key", as_index=False).first()
    patient_level["target_label"] = patient_level["target_label"].astype(int)
    patient_level["patient_image_count"] = (
        labeled.groupby("patient_key").size().reindex(patient_level["patient_key"]).values
    )
    return patient_level


def build_direct_accession_merged_table(
    dicom_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    pft_df: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    dr_subset = dr_df.rename(
        columns={
            find_column(list(dr_df.columns), [COL_ACCESSION, "AccessionNumber"]): "accession",
            find_column(list(dr_df.columns), [COL_PATIENT_NAME, COL_NAME], required=False): "dr_name",
            find_column(list(dr_df.columns), [COL_EXAM_TIME], required=False): "dr_exam_time_raw",
        }
    ).copy()
    dr_subset["accession"] = dr_subset["accession"].astype(str).str.strip()

    pft_subset = normalize_columns(pft_df).copy()
    pft_subset = pft_subset.rename(
        columns={
            find_column(list(pft_subset.columns), [COL_ACCESSION]): "accession",
            find_column(list(pft_subset.columns), [label_column]): "target_label_raw",
        }
    )
    for raw_name, target_name in {
        "sex": "pft_sex",
        "age": "pft_age",
        "height": "pft_height",
        "weight": "pft_weight",
        "smoking": "pft_smoking",
        "吸烟": "pft_smoking",
    }.items():
        if raw_name in pft_subset.columns and target_name not in pft_subset.columns:
            pft_subset = pft_subset.rename(columns={raw_name: target_name})
    pft_subset["accession"] = pft_subset["accession"].astype(str).str.strip()

    merged = dicom_df.copy()
    merged["accession"] = merged["accession"].astype(str).str.strip()
    merged = merged.merge(dr_subset, on="accession", how="left", suffixes=("", "_dr"))
    merged = merged.merge(pft_subset, on="accession", how="inner", suffixes=("", "_pft"))

    if "dr_exam_time_raw" in merged.columns:
        merged["dr_exam_time"] = pd.to_datetime(
            pd.to_numeric(merged["dr_exam_time_raw"], errors="coerce"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    for raw_col in ["pft_age", "pft_height", "pft_weight"]:
        if raw_col in merged.columns:
            merged[raw_col] = pd.to_numeric(merged[raw_col], errors="coerce")

    merged["target_label"] = clean_binary_like_label(merged["target_label_raw"])
    merged["patient_key"] = build_patient_key(merged)
    return merged


def build_feature_table(
    patient_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_df = patient_df.copy()
    feature_df["pft_age"] = pd.to_numeric(feature_df.get("pft_age"), errors="coerce")
    feature_df["pft_height"] = pd.to_numeric(feature_df.get("pft_height"), errors="coerce")
    feature_df["pft_weight"] = pd.to_numeric(feature_df.get("pft_weight"), errors="coerce")

    if "pft_sex" in feature_df.columns:
        feature_df["pft_sex"] = (
            feature_df["pft_sex"]
            .astype(str)
            .str.strip()
            .replace({"0": "female", "1": "male", TEXT_FEMALE: "female", TEXT_MALE: "male"})
        )
    if "pft_smoking" in feature_df.columns:
        feature_df["pft_smoking"] = (
            feature_df["pft_smoking"]
            .astype(str)
            .str.strip()
            .replace({"0": "no", "1": "yes", TEXT_NO: "no", TEXT_YES: "yes"})
        )

    numeric_features = [col for col in ["pft_age", "pft_height", "pft_weight"] if col in feature_df.columns]
    categorical_features = [col for col in ["pft_sex", "pft_smoking"] if col in feature_df.columns]
    if not numeric_features and not categorical_features:
        raise RuntimeError("No usable tabular features were found in the selected workbook.")
    return feature_df, numeric_features, categorical_features


def stratified_patient_split(
    patient_df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        patient_df,
        test_size=test_size,
        random_state=seed,
        stratify=patient_df["target_label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=train_val["target_label"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=None,
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def find_best_threshold(
    y_true: pd.Series,
    probs: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = evaluate_predictions(y_true, probs, 0.5)
    best_f1 = best_metrics["f1"]
    for threshold in np.arange(0.05, 0.96, 0.01):
        metrics = evaluate_predictions(y_true, probs, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_patients": int(len(df)),
        "positive_rate": float(df["target_label"].mean()),
        "avg_images_per_patient": float(df["patient_image_count"].mean()),
        "max_images_per_patient": int(df["patient_image_count"].max()),
    }


def main() -> None:
    args = parse_args()

    dr_path = resolve_existing_path(args.dr_xlsx, DEFAULT_DR_XLSX_CANDIDATES, "DR xlsx")
    pft_path = resolve_existing_path(args.pft_xlsx, DEFAULT_PFT_XLSX_CANDIDATES, "PFT xlsx")
    dicom_dir = resolve_existing_path(args.dicom_dir, DEFAULT_DICOM_DIR_CANDIDATES, "DICOM directory")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dr_df, dr_header_choice = load_excel_best_header(
        dr_path,
        parse_sheet_arg(args.dr_sheet),
        expected_terms_for_dr(),
    )

    dicom_df = scan_dicoms(dicom_dir, args.max_dicoms)
    pft_df_direct, pft_header_choice_direct = load_excel_best_header(
        pft_path,
        parse_sheet_arg(args.pft_sheet),
        expected_terms_for_pft_direct(args.label_column),
    )
    direct_mode = COL_ACCESSION in pft_df_direct.columns and args.label_column in pft_df_direct.columns

    if direct_mode:
        pft_df = pft_df_direct
        pft_header_choice = pft_header_choice_direct
        merged = build_direct_accession_merged_table(dicom_df, dr_df, pft_df, args.label_column)
        input_mode = "direct_accession"
    else:
        pft_df, pft_header_choice = load_excel_best_header(
            pft_path,
            parse_sheet_arg(args.pft_sheet),
            expected_terms_for_pft(args.label_column),
        )
        column_map = infer_column_map(dr_df, pft_df, args.label_column, pft_date_column=None)
        merged = create_merged_table(dicom_df, dr_df, pft_df, column_map)
        input_mode = "seq_linked"

    patient_df = prepare_patient_level_table(merged)
    patient_df, numeric_features, categorical_features = build_feature_table(patient_df)

    train_df, val_df, test_df = stratified_patient_split(
        patient_df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    feature_columns = numeric_features + categorical_features
    model = build_model(numeric_features, categorical_features, seed=args.seed)
    model.fit(train_df[feature_columns], train_df["target_label"])

    val_probs = model.predict_proba(val_df[feature_columns])[:, 1]
    test_probs = model.predict_proba(test_df[feature_columns])[:, 1]

    threshold_05_metrics = evaluate_predictions(test_df["target_label"], test_probs, 0.5)
    best_threshold, best_val_metrics = find_best_threshold(val_df["target_label"], val_probs)
    tuned_test_metrics = evaluate_predictions(test_df["target_label"], test_probs, best_threshold)

    result = {
        "header_selection": {
            "dr": dr_header_choice,
            "pft": pft_header_choice,
        },
        "input_mode": input_mode,
        "label_column": args.label_column,
        "features": feature_columns,
        "dataset": {
            "n_labeled_images": int(merged["target_label"].notna().sum()),
            "n_patients": int(len(patient_df)),
            "split": {
                "train": summarize_split(train_df),
                "val": summarize_split(val_df),
                "test": summarize_split(test_df),
            },
        },
        "metrics": {
            "test_at_0_5": threshold_05_metrics,
            "val_best_threshold": best_threshold,
            "val_best_threshold_metrics": best_val_metrics,
            "test_at_best_val_threshold": tuned_test_metrics,
        },
    }

    summary_path = output_dir / "tabular_baseline_summary.json"
    pred_path = output_dir / "tabular_baseline_test_predictions.csv"

    test_output = test_df[["patient_key", "target_label", "patient_image_count"]].copy()
    test_output["prob"] = test_probs
    test_output["pred_0_5"] = (test_probs >= 0.5).astype(int)
    test_output["pred_best_val_threshold"] = (test_probs >= best_threshold).astype(int)
    test_output.to_csv(pred_path, index=False, encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(result), f, ensure_ascii=False, indent=2)

    print("# Tabular Baseline")
    print("")
    print(f"- label_column: `{args.label_column}`")
    print(f"- features: `{feature_columns}`")
    print(f"- patients: `{len(patient_df)}`")
    print(f"- test roc_auc @0.5 threshold: `{threshold_05_metrics['roc_auc']:.4f}`")
    print(f"- test pr_auc @0.5 threshold: `{threshold_05_metrics['pr_auc']:.4f}`")
    print(f"- test f1 @0.5 threshold: `{threshold_05_metrics['f1']:.4f}`")
    print(f"- best val threshold: `{best_threshold:.2f}`")
    print(f"- test f1 @best val threshold: `{tuned_test_metrics['f1']:.4f}`")
    print("")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved test predictions to: {pred_path}")


if __name__ == "__main__":
    main()
