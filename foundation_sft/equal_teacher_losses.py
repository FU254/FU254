"""Equal averaging of three independent, frozen-teacher KD objectives.

The hard-label loss remains external and is added once. Each teacher gets its
own student-to-teacher global projector; global vectors are never averaged
across incompatible representation spaces.
"""

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import nn

from .distillation_losses import SingleTeacherDistillationLoss, _dimension


TEACHER_NAMES = ("rad_dino", "chexfound", "chexzero")


def _teacher_mapping(value, name):
    if not isinstance(value, Mapping) or set(value) != set(TEACHER_NAMES):
        raise ValueError(f"{name} must contain exactly rad_dino, chexfound, and chexzero")
    return value


class EqualTeacherDistillationLoss(nn.Module):
    """Return the mean KD and raw branch losses, plus per-teacher terms.

    Include ``parameters()`` in the optimizer and save ``state_dict()`` for
    resumption. The global projectors are training-only modules. Teacher
    tensors are detached inside the legacy single-teacher loss implementation.
    """

    def __init__(self, teacher_dims, student_dim=192, temperature=2.0,
                 soft_weight=0.5, global_weight=0.1, local_weight=1.0,
                 pool_grid=(7, 7), eps=1e-6, teacher_weights=None):
        super().__init__()
        _teacher_mapping(teacher_dims, "teacher_dims")
        self.teacher_names = TEACHER_NAMES
        self.student_dim = _dimension("student_dim", student_dim)
        self.teacher_dims = {
            name: _dimension(f"{name} teacher_dim", teacher_dims[name])
            for name in self.teacher_names
        }
        if teacher_weights is not None:
            _teacher_mapping(teacher_weights, "teacher_weights")
            for value in teacher_weights.values():
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                        or not math.isclose(float(value), 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("Each of the three teacher weights must be 1/3")
        self.teacher_weights = {name: 1.0 / 3.0 for name in self.teacher_names}
        self.criteria = nn.ModuleDict({
            name: SingleTeacherDistillationLoss(
                student_dim=self.student_dim, teacher_dim=self.teacher_dims[name],
                temperature=temperature, soft_weight=soft_weight,
                global_weight=global_weight, local_weight=local_weight,
                pool_grid=pool_grid, eps=eps,
            ) for name in self.teacher_names
        })
        first = self.criteria[self.teacher_names[0]]
        self.temperature, self.pool_grid, self.eps = first.temperature, first.pool_grid, first.eps
        self.soft_weight = first.soft_weight
        self.global_weight = first.global_weight
        self.local_weight = first.local_weight

    def forward(self, student_logits, references, *, student_global=None,
                student_patches=None, student_grid=None):
        _teacher_mapping(references, "references")
        per_teacher = {}
        for name in self.teacher_names:
            reference = references[name]
            if not isinstance(reference, Mapping):
                raise ValueError("Each teacher reference must be a feature mapping")
            required = set()
            if self.soft_weight > 0:
                required.add("logits")
            if self.global_weight > 0:
                required.add("global_features")
            if self.local_weight > 0:
                required.update(("local_features", "grid_shape"))
            if required.difference(reference):
                raise ValueError(f"Incomplete {name} teacher reference for enabled KD branches")
            per_teacher[name] = self.criteria[name](
                student_logits, reference.get("logits"),
                student_global=student_global, teacher_global=reference.get("global_features"),
                student_patches=student_patches, teacher_patches=reference.get("local_features"),
                student_grid=student_grid, teacher_grid=reference.get("grid_shape"),
            )
        result = {
            key: torch.stack([per_teacher[name][key] for name in self.teacher_names]).mean()
            for key in ("total", "soft", "global", "local")
        }
        result["per_teacher"] = per_teacher
        return result
