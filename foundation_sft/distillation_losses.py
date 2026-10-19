"""Tensor-only single-teacher KD; the original supervised loss stays external.

Teacher tensors are detached internally. Local inputs must contain only
row-major spatial patch tokens, with matching image-space fields of view.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


_F32 = torch.finfo(torch.float32)


def _number(name, value, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"Invalid {name}")
    return value


def _epsilon(value):
    value = _number("eps", value)
    if not _F32.tiny <= value <= _F32.max:
        raise ValueError("eps must be representable as positive float32")
    return value


def _temperature(value):
    value = _number("temperature", value)
    if not _F32.tiny <= value * value <= _F32.max:
        raise ValueError("temperature squared is outside float32 range")
    return value


def _dimension(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _grid(name, value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain height and width")
    return tuple(_dimension(name, dimension) for dimension in value)


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _tensor(name, value, ndim):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.layout != torch.strided or not value.is_floating_point():
        raise TypeError(f"{name} must be a dense floating-point Tensor")
    if value.ndim != ndim or any(size == 0 for size in value.shape):
        raise ValueError(f"Invalid {name} shape: {tuple(value.shape)}")
    return _finite(name, value.float())


def _logits(name, value):
    if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    return _tensor(name, value, 1)


def _same_batch_device(first, second):
    if first.shape[0] != second.shape[0] or first.device != second.device:
        raise ValueError("Student and teacher batch size/device must match")


def _normalize(name, value, eps):
    value = _finite(name, value.float())
    norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    _finite(f"{name} norms", norms)
    return _finite(name, value / norms.clamp_min(eps))


def binary_kl_loss(student_logits, teacher_logits, temperature=2.0):
    """Mean Bernoulli KL(teacher || student), including temperature squared."""
    temperature = _temperature(temperature)
    student = _logits("student_logits", student_logits)
    teacher = _logits("teacher_logits", teacher_logits).detach()
    _same_batch_device(student, teacher)
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _finite("scaled student logits", student / temperature)
        teacher = _finite("scaled teacher logits", teacher / temperature)
        # One binary logit means class logits [0, z], not a one-class softmax.
        student_pair = torch.stack((torch.zeros_like(student), student), dim=-1)
        teacher_pair = torch.stack((torch.zeros_like(teacher), teacher), dim=-1)
        result = F.kl_div(
            F.log_softmax(student_pair, dim=-1),
            F.log_softmax(teacher_pair, dim=-1),
            reduction="batchmean", log_target=True,
        ) * (temperature * temperature)
    return _finite("binary KL", result)


def global_cosine_loss(projected_student, teacher_global, eps=1e-6):
    """Mean cosine distance for the already projected student global feature."""
    eps = _epsilon(eps)
    student = _tensor("projected_student", projected_student, 2)
    teacher = _tensor("teacher_global", teacher_global, 2).detach()
    _same_batch_device(student, teacher)
    if student.shape[1] != teacher.shape[1]:
        raise ValueError("Projected student and teacher dimensions differ")
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _normalize("projected_student", student, eps)
        teacher = _normalize("teacher_global", teacher, eps)
        cosine = (student * teacher).sum(-1).clamp(-1, 1)
        result = (1 - cosine).mean()
    return _finite("global cosine loss", result)


def _pool_tokens(tokens, source_grid, pool_grid, name):
    height, width = source_grid
    if tokens.shape[1] != height * width:
        raise ValueError(f"{name}: token count must match pure-patch grid")
    if pool_grid[0] > height or pool_grid[1] > width:
        raise ValueError(f"{name}: pooling must not upsample the patch grid")
    batch, _, channels = tokens.shape
    spatial = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(spatial, pool_grid).flatten(2).transpose(1, 2)
    return _finite(f"{name} pooled tokens", pooled)


def local_relation_loss(student_patches, teacher_patches, student_grid,
                        teacher_grid, pool_grid=(7, 7), eps=1e-6):
    """Off-diagonal Gram MSE for pure patches [B, H*W, D].

    The caller must supply the same source view and spatial augmentation.
    CLS/register tokens must already be excluded. Channels may differ.
    """
    eps = _epsilon(eps)
    student_grid = _grid("student_grid", student_grid)
    teacher_grid = _grid("teacher_grid", teacher_grid)
    pool_grid = _grid("pool_grid", pool_grid)
    count = pool_grid[0] * pool_grid[1]
    if count < 2:
        raise ValueError("Local relations require at least two pooled positions")
    student = _tensor("student_patches", student_patches, 3)
    teacher = _tensor("teacher_patches", teacher_patches, 3).detach()
    _same_batch_device(student, teacher)
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _pool_tokens(student, student_grid, pool_grid, "student")
        teacher = _pool_tokens(teacher, teacher_grid, pool_grid, "teacher")
        student = _normalize("student pooled tokens", student, eps)
        teacher = _normalize("teacher pooled tokens", teacher, eps)
        student_relations = torch.bmm(student, student.transpose(1, 2))
        teacher_relations = torch.bmm(teacher, teacher.transpose(1, 2))
        squared_error = (student_relations - teacher_relations).square()
        mask = ~torch.eye(count, dtype=torch.bool, device=student.device)
        result = squared_error[:, mask].mean()
    return _finite("local relation loss", result)


class SingleTeacherDistillationLoss(nn.Module):
    """Return weighted KD only; disabled branches ignore their inputs.

    The optimizer must include the global projection parameters. Save this
    module when resuming training; discard it when deploying the student.
    All loss math, including projection and relation matmul, runs in FP32.
    """

    def __init__(self, student_dim=192, teacher_dim=768, temperature=2.0,
                 soft_weight=0.5, global_weight=0.1, local_weight=1.0,
                 pool_grid=(7, 7), eps=1e-6):
        super().__init__()
        self.student_dim = _dimension("student_dim", student_dim)
        self.teacher_dim = _dimension("teacher_dim", teacher_dim)
        self.temperature = _temperature(temperature)
        self.soft_weight = _number("soft_weight", soft_weight, True)
        self.global_weight = _number("global_weight", global_weight, True)
        self.local_weight = _number("local_weight", local_weight, True)
        self.pool_grid, self.eps = _grid("pool_grid", pool_grid), _epsilon(eps)
        if self.pool_grid[0] * self.pool_grid[1] < 2:
            raise ValueError("pool_grid requires at least two positions")
        self.projector = (
            nn.Linear(self.student_dim, self.teacher_dim, bias=False)
            if self.global_weight > 0 else None
        )

    def forward(self, student_logits, teacher_logits=None, *, student_global=None,
                teacher_global=None, student_patches=None, teacher_patches=None,
                student_grid=None, teacher_grid=None):
        reference = _logits("student_logits", student_logits)
        soft = global_term = local = reference.new_zeros(())
        if self.soft_weight > 0:
            soft = binary_kl_loss(reference, teacher_logits, self.temperature)
        if self.global_weight > 0:
            student = _tensor("student_global", student_global, 2)
            teacher = _tensor("teacher_global", teacher_global, 2).detach()
            _same_batch_device(reference, student)
            _same_batch_device(student, teacher)
            if student.shape[1] != self.student_dim or teacher.shape[1] != self.teacher_dim:
                raise ValueError("Unexpected global feature dimensions")
            if self.projector.weight.device != student.device:
                raise ValueError("Move the KD module to the student's device")
            with torch.autocast(device_type=student.device.type, enabled=False):
                projected = F.linear(student, self.projector.weight.float())
            global_term = global_cosine_loss(projected, teacher, self.eps)
        if self.local_weight > 0:
            student = _tensor("student_patches", student_patches, 3)
            _same_batch_device(reference, student)
            local = local_relation_loss(student, teacher_patches, student_grid,
                                        teacher_grid, self.pool_grid, self.eps)
        total = self.soft_weight * soft + self.global_weight * global_term + self.local_weight * local
        return {"total": _finite("weighted KD", total), "soft": soft,
                "global": global_term, "local": local}
