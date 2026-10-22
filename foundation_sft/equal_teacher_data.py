"""Derive all teacher views from the existing augmented RAD-DINO field.

The anchor and student images are unchanged from SharedViewDataset. Additional
teachers receive deterministic resizes of that same field with their native
normalization. Native crop/padding is deliberately bypassed and recorded.
"""

from collections.abc import Mapping
import copy
from numbers import Integral

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .distillation_data import _normalization, _preprocessing_metadata
from .equal_teacher_losses import TEACHER_NAMES, _teacher_mapping


_NATIVE_RESIZE_MODES = {"resize", "shortest_edge_center_crop", "chexzero_pad320"}


class MultiTeacherViewDataset(Dataset):
    """Wrap SharedViewDataset without an extra decode or augmentation draw.

    ``metadata()`` intentionally remains the exact legacy anchor/student record
    for control reuse and standalone student inference. The additional teacher
    preprocessing protocol is exposed separately by ``teacher_view_metadata``.
    """

    def __init__(self, shared, teacher_specs, anchor="rad_dino"):
        if anchor != "rad_dino":
            raise ValueError("The shared-view anchor must be rad_dino")
        if (not hasattr(shared, "__getitem__") or not hasattr(shared, "__len__")
                or not callable(getattr(shared, "metadata", None))
                or getattr(shared, "include_teacher", None) is not True):
            raise ValueError("shared must expose SharedViewDataset with include_teacher=True")
        _teacher_mapping(teacher_specs, "teacher_specs")
        self.shared = shared
        self.anchor = anchor
        self.teacher_names = TEACHER_NAMES
        self.teacher_specs = {}
        self.teacher_mean, self.teacher_std, self.teacher_size = {}, {}, {}
        for name in self.teacher_names:
            spec = _preprocessing_metadata(teacher_specs[name])
            size = spec.get("image_size")
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
                raise ValueError("Each teacher image_size must be a positive integer")
            if spec.get("resize_mode", "resize") not in _NATIVE_RESIZE_MODES:
                raise ValueError("Unsupported native teacher resize_mode")
            if spec.get("interpolation", "bicubic") != "bicubic":
                raise ValueError("The shared teacher-view protocol requires bicubic interpolation")
            if spec.get("do_hist_equalization", False) is not False:
                raise ValueError("Additional per-teacher histogram equalization is unsupported")
            shortest = spec.get("resize_shortest_edge")
            if shortest is not None and (isinstance(shortest, bool)
                                         or not isinstance(shortest, Integral) or shortest < 1):
                raise ValueError("resize_shortest_edge must be a positive integer or None")
            self.teacher_mean[name] = _normalization(spec.get("mean"), f"{name} mean")
            self.teacher_std[name] = _normalization(spec.get("std"), f"{name} std", positive=True)
            self.teacher_size[name] = int(size)
            self.teacher_specs[name] = spec
        metadata = shared.metadata()
        if (not isinstance(metadata, Mapping)
                or metadata.get("teacher_preprocessing") != self.teacher_specs[anchor]):
            raise ValueError("RAD-DINO anchor profile differs from the existing shared dataset")
        self.anchor_mean = _normalization(shared.teacher_mean, "shared teacher_mean", exact_shape=True)
        self.anchor_std = _normalization(shared.teacher_std, "shared teacher_std", positive=True,
                                         exact_shape=True)
        if (not torch.equal(self.anchor_mean, self.teacher_mean[anchor])
                or not torch.equal(self.anchor_std, self.teacher_std[anchor])):
            raise ValueError("RAD-DINO anchor normalization differs from its native profile")
        self._teacher_view_metadata = {
            "anchor": anchor,
            "teacher_names": list(self.teacher_names),
            "shared_field_of_view": True,
            "shared_geometry": "Use the existing augmented RAD-DINO field; no additional crop, pad, rotation, or flip.",
            "reference_preprocessing": copy.deepcopy(metadata),
            "teacher_views": {},
        }
        for name in self.teacher_names:
            self._teacher_view_metadata["teacher_views"][name] = {
                "native_preprocessing": copy.deepcopy(self.teacher_specs[name]),
                "effective_preprocessing": {
                    "image_size": self.teacher_size[name],
                    "mean": self.teacher_mean[name].flatten().tolist(),
                    "std": self.teacher_std[name].flatten().tolist(),
                    "resize_mode": ("preserve_anchor_tensor" if name == anchor
                                    else "resize_shared_anchor_view"),
                    "interpolation": "bicubic",
                    "align_corners": False,
                    "antialias": True,
                    "clamp": (None if name == anchor else [0.0, 1.0]),
                },
                "native_geometry_applied": name == anchor,
            }

    def __len__(self):
        return len(self.shared)

    def metadata(self):
        return self.shared.metadata()

    def teacher_view_metadata(self):
        return copy.deepcopy(self._teacher_view_metadata)

    def __getitem__(self, index):
        item = self.shared[index]
        if not isinstance(item, Mapping) or "teacher_image" not in item:
            raise ValueError("Shared sample must contain its original teacher_image")
        anchor_image = item["teacher_image"]
        expected = self.teacher_size[self.anchor]
        if (not isinstance(anchor_image, torch.Tensor) or not anchor_image.is_floating_point()
                or anchor_image.layout != torch.strided
                or anchor_image.shape != (3, expected, expected)
                or not torch.isfinite(anchor_image).all()):
            raise ValueError("Anchor teacher_image must be a finite native-size (3,H,W) tensor")
        device = anchor_image.device
        gray = (anchor_image[0:1].float() * self.anchor_std[0:1].to(device=device)
                + self.anchor_mean[0:1].to(device=device))
        if not torch.isfinite(gray).all():
            raise ValueError("Inverse anchor normalization produced nonfinite pixels")
        images = {self.anchor: anchor_image}
        for name in self.teacher_names:
            if name == self.anchor:
                continue
            size = self.teacher_size[name]
            display = F.interpolate(
                gray.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False,
                antialias=True,
            ).squeeze(0).clamp(0.0, 1.0).repeat(3, 1, 1)
            image = (display - self.teacher_mean[name].to(device=device)) / self.teacher_std[name].to(device=device)
            if not torch.isfinite(image).all():
                raise ValueError("Teacher shared-view preprocessing produced nonfinite pixels")
            images[name] = image
        result = dict(item)
        result["teacher_images"] = images
        return result
