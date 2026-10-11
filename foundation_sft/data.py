"""Build a shared local CXR cohort and decode DICOMs lazily during training."""

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset


PREPROCESS_VERSION = "dicom_rescale_window_percentile_monochrome_v2"
_NAME_COLUMNS = ("姓名", "病人姓名", "PatientName")
_ID_COLUMNS = ("住院号", "患者ID", "病人ID", "PatientID")
_MISSING = {"", "nan", "none", "null", "<na>"}


class _DicomDataError(ValueError):
    """A sanitized failure of the shared pixel-preprocessing contract."""


def normalize_identifier(value):
    """Normalize Excel integer artifacts without removing meaningful leading zeros."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in _MISSING:
        return ""
    return re.sub(r"\.0+$", "", value) if re.fullmatch(r"\d+\.0+", value) else value


def _binary_label(value):
    text = normalize_identifier(value)
    if not text:
        return None
    try:
        numeric = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("Workbook labels must be exactly numeric 0 or 1.") from error
    if not numeric.is_finite() or numeric not in (Decimal(0), Decimal(1)):
        raise ValueError("Workbook labels must be exactly numeric 0 or 1; fractional labels are invalid.")
    return int(numeric)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patient_identity(row, dicom, settings):
    column = settings.get("patient_column", "PatientID")
    source = settings.get("patient_id_source", "auto")
    if source not in {"auto", "excel", "dicom"}:
        raise ValueError("patient_id_source must be auto, excel, or dicom.")
    allow_names = bool(settings.get("allow_name_patient_id", False))
    if column in _NAME_COLUMNS and not allow_names:
        raise ValueError("Using a patient-name column requires allow_name_patient_id=true.")
    if source != "dicom":
        value = normalize_identifier(row.get(column))
        if value:
            return value, f"excel:{column}"
    if source == "dicom" or (source == "auto" and settings.get("dicom_patient_fallback", True)):
        value = normalize_identifier(getattr(dicom, "PatientID", None))
        if value:
            return value, "dicom:PatientID"
    if allow_names:
        if source != "dicom":
            requested = settings.get("patient_name_column")
            for name_column in ((requested,) if requested else _NAME_COLUMNS):
                value = normalize_identifier(row.get(name_column))
                if value:
                    return value, f"excel_name:{name_column}"
        if source != "excel":
            value = normalize_identifier(getattr(dicom, "PatientName", None))
            if value:
                return value, "dicom_name:PatientName"
    return "", ""


def build_cohort(config):
    """Match strict workbook labels to DICOM headers without loading a pixel cache.

    Only configured Excel patient IDs or DICOM PatientID identify patients by
    default. A cohort must use one identifier origin; mixed origins need an
    explicit common mapping or patient_id_source='dicom'. No patient values
    are included in diagnostics or exception messages.
    """
    settings = dict(config.get("data", config))
    for key in ("excel_path", "dicom_dir"):
        if not settings.get(key):
            raise ValueError(f"Missing required data configuration: {key}.")
    excel_path = Path(settings["excel_path"]).expanduser().resolve()
    dicom_dir = Path(settings["dicom_dir"]).expanduser().resolve()
    if not excel_path.is_file() or not dicom_dir.is_dir():
        raise FileNotFoundError("Configured workbook or DICOM directory is unavailable on this machine.")
    accession_column = settings.get("accession_column", "放射编号")
    label_column = settings.get("label_column", "阻塞性通气障碍")
    patient_column = settings.get("patient_column", "PatientID")
    excel_hash = _file_sha256(excel_path)
    workbook = pd.read_excel(excel_path, sheet_name=settings.get("sheet_name", 0), dtype=str)
    if patient_column == "auto":
        patient_column = next((column for column in _ID_COLUMNS if column in workbook.columns), "PatientID")
        settings["patient_column"] = patient_column
    missing = {accession_column, label_column}.difference(workbook.columns)
    if missing:
        raise ValueError(f"Workbook is missing configured columns: {sorted(missing)}.")
    candidates = [column for column in workbook.columns if column in _ID_COLUMNS + _NAME_COLUMNS]
    diagnostics = {
        "workbook_rows": len(workbook),
        "candidate_patient_columns": candidates,
        "selected_patient_column": patient_column,
        "configured_patient_column_present": patient_column in workbook.columns,
        "missing_accession_rows": 0,
        "missing_label_rows": 0,
        "duplicate_workbook_rows": 0,
        "dicom_files": 0,
        "unmatched_dicoms": 0,
        "unreadable_dicoms": 0,
        "unsupported_dicoms": 0,
        "unidentifiable_patients": 0,
    }
    rows_by_accession = {}
    for _, row in workbook.iterrows():
        accession = normalize_identifier(row[accession_column])
        if not accession:
            diagnostics["missing_accession_rows"] += 1
            continue
        label = _binary_label(row[label_column])
        if label is None:
            diagnostics["missing_label_rows"] += 1
            continue
        patient_value = normalize_identifier(row.get(patient_column))
        if accession in rows_by_accession:
            previous, previous_label = rows_by_accession[accession]
            previous_patient = normalize_identifier(previous.get(patient_column))
            if previous_label != label or previous_patient != patient_value:
                raise ValueError("Conflicting duplicate workbook accession labels or patient mappings.")
            diagnostics["duplicate_workbook_rows"] += 1
            continue
        rows_by_accession[accession] = (row, label)

    samples = []
    identity_sources = set()
    instance_ids = set()
    for path in sorted(dicom_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".dcm":
            continue
        diagnostics["dicom_files"] += 1
        try:
            dicom = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            diagnostics["unreadable_dicoms"] += 1
            continue
        accession = normalize_identifier(getattr(dicom, "AccessionNumber", None))
        match = rows_by_accession.get(accession)
        if match is None:
            diagnostics["unmatched_dicoms"] += 1
            continue
        row, label = match
        if (int(getattr(dicom, "Rows", 0)) < 1 or int(getattr(dicom, "Columns", 0)) < 1
                or int(getattr(dicom, "SamplesPerPixel", 1)) != 1
                or int(getattr(dicom, "NumberOfFrames", 1)) != 1):
            diagnostics["unsupported_dicoms"] += 1
            continue
        patient_id, origin = _patient_identity(row, dicom, settings)
        if not patient_id:
            diagnostics["unidentifiable_patients"] += 1
            continue
        identity_sources.add(origin)
        instance_id = normalize_identifier(getattr(dicom, "SOPInstanceUID", None))
        if instance_id and instance_id in instance_ids:
            raise ValueError("Duplicate DICOM SOPInstanceUID detected; deduplicate the cohort before comparison.")
        if instance_id:
            instance_ids.add(instance_id)
        stat = path.stat()
        sample_id = hashlib.sha256(path.relative_to(dicom_dir).as_posix().encode("utf-8")).hexdigest()
        samples.append({
            "sample_id": sample_id,
            "patient_id": patient_id,
            "patient_id_source": origin,
            "accession": accession,
            "path": str(path.resolve()),
            "label": label,
            "dicom_size": stat.st_size,
            "dicom_mtime_ns": stat.st_mtime_ns,
        })
    if diagnostics["unidentifiable_patients"]:
        raise ValueError(
            f"Reliable patient IDs are missing for {diagnostics['unidentifiable_patients']} matched images. "
            f"Configured patient column: {patient_column}; candidate columns: {candidates}. "
            "Configure a reliable workbook column or supply DICOM PatientID; accession is not a patient ID."
        )
    if len(identity_sources) > 1:
        raise ValueError(
            "Mixed patient identifier origins are ambiguous. Complete one explicit workbook mapping "
            "or configure patient_id_source='dicom' consistently for the entire cohort."
        )
    if not samples:
        raise ValueError(f"No eligible labeled DICOM images were found. Counts: {diagnostics}.")
    cohort = pd.DataFrame(samples).sort_values("sample_id").reset_index(drop=True)
    if cohort["path"].duplicated().any():
        raise ValueError("Multiple cohort samples resolve to the same DICOM file; deduplicate before comparison.")
    if cohort.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError(
            "The cohort contains conflicting labels within a patient. Define one paired CXR/PFT "
            "examination per patient according to the study protocol; do not choose an arbitrary new examination."
        )
    if _file_sha256(excel_path) != excel_hash:
        raise ValueError("Workbook changed during cohort preparation; retry with a stable label source.")
    provenance = {
        "excel_sha256": excel_hash,
        "label_column": label_column,
        "patient_column": patient_column,
        "patient_id_source": sorted(identity_sources),
        "allow_name_patient_id": bool(settings.get("allow_name_patient_id", False)),
    }
    cohort.attrs["label_source_signature"] = hashlib.sha256(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cohort.attrs["label_definition"] = settings.get(
        "label_definition",
        "Existing workbook binary label; LLN reference equation unavailable; labels are not recomputed.",
    )
    cohort.attrs["label_column"] = label_column
    cohort.attrs["excel_sha256"] = excel_hash
    cohort.attrs["cohort_diagnostics"] = diagnostics
    cohort.attrs["preprocess_version"] = PREPROCESS_VERSION
    return cohort


def _scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        value = value[0] if value else default
    if value is None:
        return default
    result = float(value)
    if not np.isfinite(result):
        raise _DicomDataError("DICOM rescaling or window metadata is nonfinite.")
    return result


def _dicom_pixels(dicom):
    image = np.asarray(dicom.pixel_array, dtype=np.float32)
    if image.ndim != 2 or int(getattr(dicom, "SamplesPerPixel", 1)) != 1:
        raise _DicomDataError("Expected one monochrome 2D DICOM image.")
    image = image * _scalar(getattr(dicom, "RescaleSlope", None), 1.0)
    image = image + _scalar(getattr(dicom, "RescaleIntercept", None), 0.0)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise _DicomDataError("DICOM image contains no finite pixels.")
    if finite.size != image.size:
        raise _DicomDataError("DICOM image contains nonfinite pixels.")
    center = _scalar(getattr(dicom, "WindowCenter", None))
    width = _scalar(getattr(dicom, "WindowWidth", None))
    if center is not None and width is not None and width > 0:
        image = np.clip(image, center - width / 2.0, center + width / 2.0)
    finite = image[np.isfinite(image)]
    lower, upper = np.percentile(finite, [0.5, 99.5])
    if upper - lower <= 1e-8:
        image = np.zeros_like(image, dtype=np.float32)
    else:
        image = (np.clip(image, lower, upper) - lower) / (upper - lower)
    # Window center/width refer to modality values. Apply the display polarity
    # after windowing, otherwise an asymmetric MONOCHROME1 window clips the
    # wrong intensity range.
    if str(getattr(dicom, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        image = 1.0 - image
    return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _spec_value(spec, name, default=None):
    return spec.get(name, default) if isinstance(spec, dict) else getattr(spec, name, default)


def _resize_image(pixels, preprocess):
    size = int(_spec_value(preprocess, "image_size"))
    if size < 1:
        raise ValueError("image_size must be positive.")
    mode = _spec_value(preprocess, "resize_mode", "resize")
    interpolation = _spec_value(preprocess, "interpolation", "bicubic")
    resamplers = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "box": Image.Resampling.BOX,
        "hamming": Image.Resampling.HAMMING,
    }
    if interpolation not in resamplers:
        raise ValueError("Unsupported image interpolation setting.")
    image = Image.fromarray(pixels)
    width, height = image.size
    resample = resamplers[interpolation]
    if mode == "resize":
        image = image.resize((size, size), resample=resample)
    elif mode == "shortest_edge_center_crop":
        edge = int(_spec_value(preprocess, "resize_shortest_edge") or size)
        if edge < 1:
            raise ValueError("resize_shortest_edge must be positive.")
        # Keep the short side exact; floor only the aspect-preserving long side.
        dimensions = ((edge, max(1, int(edge * height / width))) if width <= height
                      else (max(1, int(edge * width / height)), edge))
        image = image.resize(dimensions, resample=resample)
        left, top = (image.width - size) // 2, (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
    elif mode == "letterbox":
        factor = size / max(width, height)
        resized = image.resize((max(1, round(width * factor)), max(1, round(height * factor))), resample=resample)
        canvas = Image.new("F", (size, size), 0.0)
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        image = canvas
    else:
        raise ValueError("resize_mode must be resize, letterbox, or shortest_edge_center_crop.")
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


class CXRDataset(Dataset):
    """Return images and identities from one frozen cohort, without eager arrays."""

    def __init__(self, cohort, preprocess, training=False, seed=42,
                 horizontal_flip_probability=0.0, horizontal_flip_p=None, rotation_degrees=0.0):
        required = {"sample_id", "patient_id", "path", "label"}
        if required.difference(cohort.columns):
            raise ValueError("Dataset cohort is missing required columns.")
        self.cohort = cohort.reset_index(drop=True).copy()
        self.preprocess = preprocess
        self.training = bool(training)
        self.seed = int(seed)
        self.horizontal_flip_probability = float(
            horizontal_flip_probability if horizontal_flip_p is None else horizontal_flip_p
        )
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be between zero and one.")
        self.rotation_degrees = float(rotation_degrees)
        if not np.isfinite(self.rotation_degrees) or self.rotation_degrees < 0:
            raise ValueError("rotation_degrees must be finite and nonnegative.")
        self.mean = torch.as_tensor(_spec_value(preprocess, "mean"), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(_spec_value(preprocess, "std"), dtype=torch.float32).view(-1, 1, 1)
        if (self.mean.shape[0] != 3 or self.std.shape[0] != 3
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or not (self.std > 0).all()):
            raise ValueError("Preprocess mean/std must have three finite channels and positive standard deviations.")

    def __len__(self):
        return len(self.cohort)

    def __getitem__(self, index):
        row = self.cohort.iloc[index]
        try:
            pixels = _dicom_pixels(pydicom.dcmread(row["path"]))
        except _DicomDataError as error:
            raise RuntimeError(f"{error} Cohort row {index}.") from None
        except Exception:
            # Exceptions from decoders can include filenames or header values.
            raise RuntimeError(f"DICOM pixel decoding failed for cohort row {index}.") from None
        image = _resize_image(pixels, self.preprocess)
        if self.training and self.rotation_degrees > 0:
            angle = float(torch.rand(())) * (2 * self.rotation_degrees) - self.rotation_degrees
            image = np.asarray(
                Image.fromarray(image).rotate(
                    angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0.0
                ),
                dtype=np.float32,
            )
        image = torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0).repeat(3, 1, 1)
        # The caller seeds Torch and each DataLoader worker; no vertical flips.
        if self.training and self.horizontal_flip_probability > 0:
            if torch.rand(()) < self.horizontal_flip_probability:
                image = image.flip(-1)
        image = (image - self.mean) / self.std
        return {
            "image": image,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
        }
