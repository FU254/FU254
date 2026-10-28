"""Opt-in validation diagnostics and an explicitly scaled ImageNet anchor.

The anchor is independent of the teacher and the distillation projector. It is
captured from the freshly initialized student before the first optimizer step.
"""

from copy import deepcopy
import hashlib
import math

import torch


def validate_options(config, *, student_mode, condition=None):
    if "diagnostics" not in config:
        return None
    options = config["diagnostics"]
    required = {"protocol", "evaluation"}
    if (not isinstance(options, dict) or not required.issubset(options)
            or set(options) - required - {"l2_sp"}
            or options["protocol"] != "single_teacher_overfit_v1"
            or options["evaluation"] != "validation_only"):
        raise ValueError("diagnostics requires protocol=single_teacher_overfit_v1 and evaluation=validation_only")
    if student_mode != "full_finetune":
        raise ValueError("single_teacher_overfit_v1 requires student.mode=full_finetune")
    if "l2_sp" in options:
        anchor = options["l2_sp"]
        if (not isinstance(anchor, dict) or set(anchor) != {"coefficient", "reduction"}
                or type(anchor["coefficient"]) not in (int, float)
                or not math.isfinite(anchor["coefficient"])
                or anchor["coefficient"] != 1e-4 or anchor["reduction"] != "sum"):
            raise ValueError("diagnostics.l2_sp requires coefficient=0.0001 and reduction=sum")
        loss = config.get("distillation", {})
        if (condition == "supervised" or any(
                type(loss.get(key)) not in (int, float) or loss[key] <= 0
                for key in ("soft_weight", "global_weight", "local_weight"))):
            raise ValueError("L2-SP diagnostics requires the full soft/global/local distillation condition")
    return deepcopy(options)


def _anchor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class ImageNetL2SP:
    """Raw sum of squared distances to every initial non-head encoder weight."""

    def __init__(self, student, options):
        self.coefficient = options["coefficient"]
        self.anchors = {
            name: parameter.detach().clone()
            for name, parameter in student.named_parameters()
            if name.startswith("encoder.") and not name.startswith("encoder.head.")
        }
        if not self.anchors:
            raise ValueError("L2-SP requires an initial ImageNet backbone")
        if any(not dict(student.named_parameters())[name].requires_grad for name in self.anchors):
            raise ValueError("L2-SP requires the complete trainable student backbone")
        self.initial_sha256 = _anchor_hash(self.anchors)
        self.initialization = deepcopy(student.training_metadata()["initialization"])

    def losses(self, student):
        parameters = dict(student.named_parameters())
        raw = sum((parameters[name] - anchor).square().sum()
                  for name, anchor in self.anchors.items())
        return raw, self.coefficient * raw

    def metadata(self):
        if _anchor_hash(self.anchors) != self.initial_sha256:
            raise RuntimeError("The fixed initial ImageNet L2-SP anchor changed")
        return {
            "coefficient": self.coefficient, "reduction": "sum",
            "formula": "0.0001 * sum((student_backbone - initial_ImageNet_backbone)^2)",
            "anchor_sha256": self.initial_sha256,
            "anchor_initialization": deepcopy(self.initialization),
            "anchor_parameter_names": sorted(self.anchors),
            "anchor_tensor_count": len(self.anchors),
            "anchor_parameter_count": sum(value.numel() for value in self.anchors.values()),
            "includes_final_norm": any(name.startswith("encoder.norm.") for name in self.anchors),
            "excludes": ["student classification head", "distillation projector", "teacher"],
            "anchor_unchanged": True,
        }
