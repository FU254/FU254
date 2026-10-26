"""Explicit loss profiles for the unchanged random-student training entry point.

An absent profile keeps the historical three-term protocol byte-for-byte in
plans and saved metadata. The approved soft-only trial is one B/seed42 job.
This module deliberately uses only the standard library for planning/reporting.
"""

from copy import deepcopy
import math


SOFT_ONLY = "soft_only"
THREE_TERM_LOSS_SETTINGS = {
    "temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
    "local_weight": 1.0, "pool_grid": [7, 7],
}
SOFT_ONLY_LOSS_SETTINGS = {
    **deepcopy(THREE_TERM_LOSS_SETTINGS), "global_weight": 0.0, "local_weight": 0.0,
}


def is_soft_only(config):
    if "distillation_profile" not in config:
        return False
    if config["distillation_profile"] != SOFT_ONLY:
        raise ValueError("Unknown random-student distillation_profile; use explicit 'soft_only' or omit it")
    return True


def loss_settings(config, *, legacy_settings=None):
    """Return a fresh loss mapping without adding defaults to the input config."""
    if is_soft_only(config):
        return deepcopy(SOFT_ONLY_LOSS_SETTINGS)
    return deepcopy(THREE_TERM_LOSS_SETTINGS if legacy_settings is None else legacy_settings)


def validate_soft_only(config, *, condition=None):
    """Keep the approved one-job trial narrow; legacy experiments are unchanged."""
    if not is_soft_only(config):
        return
    if condition is not None and condition != "distillation":
        raise ValueError("The soft-only trial adds only the B/distillation training")
    if type(config.get("seed")) is not int or config["seed"] != 42:
        raise ValueError("The confirmed soft-only trial requires seed42")
    if config.get("student_initialization") != "random" or config.get("student_checkpoint") is not None:
        raise ValueError("The soft-only trial requires a random student without a source checkpoint")
    if config.get("student", {}).get("mode", "full_finetune") != "full_finetune":
        raise ValueError("The soft-only trial fully trains the random student")
    train = config.get("training", {})
    for key, expected in (("weight_decay", 0.01), ("drop_path_rate", 0.0)):
        value = train.get(key, 0.0 if key == "drop_path_rate" else None)
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            raise ValueError(f"The confirmed soft-only trial requires training.{key}={expected}")
    if train.get("amp") is not False:
        raise ValueError("The soft-only trial retains FP32 training.amp=false")
    if config.get("distillation") != SOFT_ONLY_LOSS_SETTINGS:
        raise ValueError("The confirmed soft-only trial retains temperature=2 and soft_weight=0.5; global/local must be zero")
    if any(type(config["distillation"][key]) not in (int, float)
           for key in ("temperature", "soft_weight", "global_weight", "local_weight")):
        raise ValueError("Soft-only distillation coefficients must be numeric, not booleans")
