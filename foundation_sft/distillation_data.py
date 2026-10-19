"""Derive a student image from one cached, augmented teacher field of view."""

from collections.abc import Mapping
import copy
from dataclasses import asdict, is_dataclass
import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalization(value, name, *, positive=False, exact_shape=False):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must contain three finite normalization values.") from error
    if ((exact_shape and tensor.shape != (3, 1, 1)) or tensor.numel() != 3
            or not torch.isfinite(tensor).all() or (positive and not (tensor > 0).all())):
        raise ValueError(
            f"{name} must contain three finite values; base normalization must have "
            "shape (3,1,1); std must be positive."
        )
    return tensor.reshape(3, 1, 1)


def _preprocessing_metadata(spec):
    """Preserve every preprocessing field, including future dataclass fields."""
    if is_dataclass(spec) and not isinstance(spec, type):
        metadata = asdict(spec)
    elif isinstance(spec, Mapping):
        metadata = copy.deepcopy(dict(spec))
    elif hasattr(spec, "__dict__"):
        metadata = copy.deepcopy(vars(spec))
    else:
        raise ValueError("Base preprocessing must be a dataclass, mapping, or attribute object.")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Teacher preprocessing metadata must contain finite JSON values.") from error


class SharedViewDataset(Dataset):
    """Reuse one base decode, cache lookup, and set of geometric transforms.

    CXRDataset repeats a grayscale image into three channels before applying
    its per-channel normalization. Inverting channel zero recovers that common
    display image. The original normalized teacher tensor is preserved; only
    the student's second resize is newly performed. No random draws are added.
    """

    def __init__(self, base, student_size=224, include_teacher=True,
                 student_mean=_IMAGENET_MEAN, student_std=_IMAGENET_STD):
        if not hasattr(base, "__getitem__") or not hasattr(base, "__len__"):
            raise TypeError("base must be an existing indexed dataset.")
        if isinstance(student_size, bool) or not isinstance(student_size, int) or student_size < 1:
            raise ValueError("student_size must be a positive integer.")
        if not isinstance(include_teacher, bool):
            raise ValueError("include_teacher must be a bool.")
        if not hasattr(base, "mean") or not hasattr(base, "std") or not hasattr(base, "preprocess"):
            raise ValueError("base must expose its teacher mean, std, and preprocessing specification.")
        self.base = base
        self.student_size = student_size
        self.include_teacher = include_teacher
        self.teacher_mean = _normalization(base.mean, "base.mean", exact_shape=True)
        self.teacher_std = _normalization(base.std, "base.std", positive=True, exact_shape=True)
        self.student_mean = _normalization(student_mean, "student_mean")
        self.student_std = _normalization(student_std, "student_std", positive=True)
        self.preprocessing_metadata = {
            "shared_field_of_view": True,
            "shared_view_source": "One augmented base CXRDataset sample; inverse normalization of grayscale channel 0.",
            "shared_geometry": "All crop, rotation, and flips are performed only by the base dataset.",
            "cache_policy": "Reuse the base dataset PixelCache; add no new cache or DICOM preprocessing.",
            "teacher_preprocessing": _preprocessing_metadata(base.preprocess),
            "teacher_normalization": {
                "mean": self.teacher_mean.flatten().tolist(),
                "std": self.teacher_std.flatten().tolist(),
            },
            "student_preprocessing": {
                "image_size": student_size,
                "mean": self.student_mean.flatten().tolist(),
                "std": self.student_std.flatten().tolist(),
                "resize_mode": "resize_shared_teacher_view",
                "interpolation": "bicubic",
                "align_corners": False,
                "antialias": True,
                "clamp": [0.0, 1.0],
            },
        }

    def __len__(self):
        return len(self.base)

    def metadata(self):
        """Return an independent, JSON-serializable preprocessing record."""
        return copy.deepcopy(self.preprocessing_metadata)

    def __getitem__(self, index):
        item = self.base[index]
        if not isinstance(item, Mapping) or {"image", "label", "sample_id", "patient_id"}.difference(item):
            raise ValueError("Base samples must contain image, label, sample_id, and patient_id.")
        teacher_image = item["image"]
        if (not isinstance(teacher_image, torch.Tensor) or not teacher_image.is_floating_point()
                or teacher_image.ndim != 3 or teacher_image.shape[0] != 3
                or min(teacher_image.shape[1:]) < 1 or not torch.isfinite(teacher_image).all()):
            raise ValueError("Base image must be a finite floating tensor with shape (3,H,W).")
        device = teacher_image.device
        # CXRDataset repeats the same display image before channel normalization.
        gray = (teacher_image[0:1].to(dtype=torch.float32)
                * self.teacher_std[0:1].to(device=device)
                + self.teacher_mean[0:1].to(device=device))
        if not torch.isfinite(gray).all():
            raise ValueError("Inverse teacher normalization produced nonfinite pixels.")
        resized = F.interpolate(
            gray.unsqueeze(0), size=(self.student_size, self.student_size), mode="bicubic",
            align_corners=False, antialias=True,
        ).squeeze(0).clamp(0.0, 1.0)
        student_image = (
            resized.repeat(3, 1, 1) - self.student_mean.to(device=device)
        ) / self.student_std.to(device=device)
        if not torch.isfinite(student_image).all():
            raise ValueError("Student preprocessing produced nonfinite pixels.")
        result = dict(item)
        result["image"] = student_image
        result.pop("teacher_image", None)
        if self.include_teacher:
            result["teacher_image"] = teacher_image
        return result
