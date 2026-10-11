"""Shared chest-X-ray preprocessing and patient-level split utilities."""

import os
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ACCESSION_COLUMN = "放射编号"
PATIENT_KEY_COLUMN = "patient_key"
SPLIT_VERSION = "patient_stratified_classification_z_70_10_20_v1"
PREPROCESS_VERSION = "letterbox_percentile_dicom_v3_shared_patient_manifest"
PATIENT_ID_COLUMNS = ("住院号", "患者ID", "病人ID", "PatientID")
PATIENT_NAME_COLUMNS = ("病人姓名", "姓名", "PatientName")


def _dicom_scalar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            value = value[0]
        except (IndexError, TypeError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dicom_text(value, default="unknown"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def preprocess_cxr_dicom(dicom, image_size=224, return_metadata=False):
    """Convert a CXR DICOM to robustly normalized, aspect-preserving pixels."""
    image = np.asarray(dicom.pixel_array, dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM image, got shape={image.shape}")

    finite_before = np.isfinite(image)
    if not finite_before.any():
        raise ValueError("DICOM pixel array contains no finite values")

    slope = _dicom_scalar(getattr(dicom, "RescaleSlope", 1.0))
    intercept = _dicom_scalar(getattr(dicom, "RescaleIntercept", 0.0))
    slope = 1.0 if slope is None else slope
    intercept = 0.0 if intercept is None else intercept
    image = image * slope + intercept

    photometric = _dicom_text(getattr(dicom, "PhotometricInterpretation", "unknown")).upper()
    if photometric == "MONOCHROME1":
        finite = image[np.isfinite(image)]
        image = finite.min() + finite.max() - image

    window_center = _dicom_scalar(getattr(dicom, "WindowCenter", None))
    window_width = _dicom_scalar(getattr(dicom, "WindowWidth", None))
    if window_center is not None and window_width is not None and window_width > 0:
        lower = window_center - window_width / 2.0
        upper = window_center + window_width / 2.0
        image = np.clip(image, lower, upper)

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("DICOM pixel array contains no finite values after rescaling")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    dynamic_range = float(upper - lower)
    if dynamic_range > 1e-8:
        image = np.clip(image, lower, upper)
        image = (image - lower) / dynamic_range
    else:
        image = np.zeros_like(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    height, width = image.shape
    scale = min(float(image_size) / height, float(image_size) / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized

    view_position = _dicom_text(getattr(dicom, "ViewPosition", None)).upper()
    patient_position = _dicom_text(getattr(dicom, "PatientPosition", None))
    ap_flag = view_position.startswith("AP")
    edge_width = max(1, int(round(0.02 * min(height, width))))
    bright_threshold = float(np.percentile(image, 95))
    bright_content = image >= bright_threshold
    edge_touch = (
        bright_content[:edge_width, :].mean() > 0.25
        or bright_content[-edge_width:, :].mean() > 0.25
        or bright_content[:, :edge_width].mean() > 0.25
        or bright_content[:, -edge_width:].mean() > 0.25
    )
    quality_flag = "poor" if dynamic_range <= 1e-8 else "acceptable" if ap_flag or edge_touch else "good"
    metadata = {
        "view_position": view_position,
        "patient_position": patient_position,
        "ap_flag": bool(ap_flag),
        "obvious_cropping": bool(edge_touch),
        "low_inspiration": "unknown",
        "quality_flag": quality_flag,
        "original_height": int(height),
        "original_width": int(width),
        "resized_height": int(resized_height),
        "resized_width": int(resized_width),
    }
    return (canvas, metadata) if return_metadata else canvas


def _clean_value(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none", "<na>"} else value


def normalize_accession(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r"\d+\.0", value) else value


def patient_key_from_sources(excel_row, dicom):
    """Prefer workbook identifiers, then fall back to DICOM identifiers."""
    if excel_row is not None:
        for column in PATIENT_ID_COLUMNS:
            if column in excel_row:
                value = _clean_value(excel_row[column])
                if value:
                    return f"excel_id:{value}"
        for column in PATIENT_NAME_COLUMNS:
            if column in excel_row:
                value = _clean_value(excel_row[column])
                if value:
                    return f"excel_name:{value}"

    if dicom is not None:
        value = _clean_value(getattr(dicom, "PatientID", ""))
        if value:
            return f"dicom_id:{value}"
        value = _clean_value(getattr(dicom, "PatientName", ""))
        if value:
            return f"dicom_name:{value}"
    return ""


def _joint_strata(patient_table):
    for bin_count in (4, 3, 2):
        try:
            z_bin = pd.qcut(
                patient_table["z_score"], q=bin_count, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        if z_bin.nunique() < 2:
            continue
        strata = patient_table["classification_label"].astype(str) + "_z" + z_bin.astype(str)
        if strata.value_counts().min() >= 6:
            return strata, z_bin.astype(int), f"classification + {z_bin.nunique()} z-score quantiles"
    return (
        patient_table["classification_label"].astype(str),
        pd.Series(-1, index=patient_table.index, dtype=np.int64),
        "classification only (insufficient joint-stratum samples)",
    )


def _create_patient_table(metadata, classification_labels, z_scores):
    sample_table = metadata[[PATIENT_KEY_COLUMN]].copy()
    sample_table["classification_label"] = np.asarray(classification_labels, dtype=np.int64)
    sample_table["z_score"] = np.asarray(z_scores, dtype=np.float64)
    if sample_table[PATIENT_KEY_COLUMN].eq("").any():
        raise RuntimeError("Empty patient key encountered; refusing a non-isolated split.")

    patient_table = sample_table.groupby(PATIENT_KEY_COLUMN, as_index=False).agg(
        classification_label=("classification_label", "max"),
        z_score=("z_score", "median"),
        image_count=("classification_label", "size"),
    )
    if patient_table["classification_label"].nunique() != 2:
        raise RuntimeError("Patient-level split requires both classification classes.")
    return patient_table


def _create_patient_manifest(metadata, classification_labels, z_scores, manifest_path, seed):
    patient_table = _create_patient_table(metadata, classification_labels, z_scores)

    strata, z_bin, strategy = _joint_strata(patient_table)
    patient_table["z_bin"] = z_bin.to_numpy()
    patient_table["stratum"] = strata.to_numpy()
    try:
        train_patients, temporary_patients = train_test_split(
            patient_table, test_size=0.30, random_state=seed,
            stratify=patient_table["stratum"],
        )
        val_patients, test_patients = train_test_split(
            temporary_patients, test_size=2.0 / 3.0, random_state=seed,
            stratify=temporary_patients["stratum"],
        )
    except ValueError as error:
        if strategy.startswith("classification only"):
            raise RuntimeError(f"Patient-level classification stratification failed: {error}") from error
        patient_table["z_bin"] = -1
        patient_table["stratum"] = patient_table["classification_label"].astype(str)
        strategy = "classification only (joint stratification unavailable)"
        train_patients, temporary_patients = train_test_split(
            patient_table, test_size=0.30, random_state=seed,
            stratify=patient_table["stratum"],
        )
        val_patients, test_patients = train_test_split(
            temporary_patients, test_size=2.0 / 3.0, random_state=seed,
            stratify=temporary_patients["stratum"],
        )

    split_by_patient = {}
    for split_name, split_table in (("train", train_patients), ("val", val_patients), ("test", test_patients)):
        split_by_patient.update(dict.fromkeys(split_table[PATIENT_KEY_COLUMN], split_name))
    patient_table["split"] = patient_table[PATIENT_KEY_COLUMN].map(split_by_patient)
    patient_table["seed"] = int(seed)
    patient_table["split_version"] = SPLIT_VERSION
    patient_table["stratification"] = strategy
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    patient_table.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return patient_table


def load_or_create_patient_split(metadata, classification_labels, z_scores, manifest_path, seed, log_fn=None):
    if PATIENT_KEY_COLUMN not in metadata.columns:
        raise RuntimeError(f"Metadata must include {PATIENT_KEY_COLUMN}.")
    if len(metadata) != len(classification_labels) or len(metadata) != len(z_scores):
        raise RuntimeError("Metadata, classification labels, and z-scores must have equal length.")

    if os.path.isfile(manifest_path):
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={PATIENT_KEY_COLUMN: str})
        required = {
            PATIENT_KEY_COLUMN, "classification_label", "z_score", "split", "seed",
            "split_version",
        }
        missing = required.difference(manifest.columns)
        if missing:
            raise RuntimeError(f"Patient split manifest is missing columns: {sorted(missing)}")
        if set(manifest["split_version"].astype(str)) != {SPLIT_VERSION}:
            raise RuntimeError("Patient split manifest version is incompatible.")
        if not (pd.to_numeric(manifest["seed"], errors="coerce") == int(seed)).all():
            raise RuntimeError("Patient split manifest seed is incompatible.")
        current_patients = set(metadata[PATIENT_KEY_COLUMN].astype(str))
        manifest_patients = set(manifest[PATIENT_KEY_COLUMN].astype(str))
        if current_patients != manifest_patients:
            raise RuntimeError(
                "Patient split manifest does not match current data; "
                f"missing={len(current_patients - manifest_patients)}, extra={len(manifest_patients - current_patients)}"
            )
        current_patient_table = _create_patient_table(metadata, classification_labels, z_scores)
        current_patient_table = current_patient_table.set_index(PATIENT_KEY_COLUMN)
        manifest_table = manifest.set_index(PATIENT_KEY_COLUMN).loc[current_patient_table.index]
        if not (
            current_patient_table["classification_label"].eq(manifest_table["classification_label"]).all()
            and np.allclose(
                current_patient_table["z_score"].to_numpy(dtype=np.float64),
                manifest_table["z_score"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise RuntimeError("Patient split manifest labels do not match current data.")
    else:
        manifest = _create_patient_manifest(
            metadata, classification_labels, z_scores, manifest_path, seed
        )

    split_by_patient = manifest.set_index(PATIENT_KEY_COLUMN)["split"]
    sample_splits = metadata[PATIENT_KEY_COLUMN].map(split_by_patient).to_numpy()
    if pd.isna(sample_splits).any():
        raise RuntimeError("Some samples were not assigned by the patient split manifest.")
    indices = tuple(np.flatnonzero(sample_splits == split) for split in ("train", "val", "test"))
    if any(len(index) == 0 for index in indices):
        raise RuntimeError("At least one split is empty.")
    if log_fn:
        log_fn(f"Patient split manifest: {manifest_path}")
        for split_name, index in zip(("train", "val", "test"), indices):
            labels = np.asarray(classification_labels)[index]
            log_fn(
                f"{split_name} | samples={len(index)} | patients={metadata.iloc[index][PATIENT_KEY_COLUMN].nunique()} | "
                f"positive={int(labels.sum())} ({labels.mean():.4f})"
            )
    return (*indices, sample_splits)
