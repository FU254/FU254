#!/usr/bin/env python3
"""Install the matched validation-only single-teacher overfitting study."""

import hashlib
import importlib.util
from pathlib import Path
import sys


PAYLOAD_FILES = (
    "foundation_sft/distillation.py",
    "foundation_sft/single_teacher_diagnostics.py",
    "foundation_sft/single_teacher_overfit.py",
    "foundation_sft/single_teacher_overfit_report.py",
    "foundation_sft/configs/single_teacher_overfit.json",
    "foundation_sft/SINGLE_TEACHER_OVERFIT.md",
    "run_single_teacher_overfit.py",
)
BASE_REVISION = "950c7c65ca55764b51262a2e4de789200c8cc754"
PREVIOUS_SHA256 = {
    "foundation_sft/distillation.py": "8ac7c2ebca8f7459d0486d6374e14175ac83c8becbe3b69ac59df8afd5024272",
}
# Freeze this manifest only after reviewing every payload and this guide.
PAYLOAD_SHA256 = {
    "foundation_sft/distillation.py": "31318b0c603dc89f012ca1ba5525f778162749451ba2a42d08606addcd032306",
    "foundation_sft/single_teacher_diagnostics.py": "e7732b19a4c4266e74b6cfe303e7125d04d386c7a1ad1cc2d3d809849e5fdbe6",
    "foundation_sft/single_teacher_overfit.py": "eb36edb96181814de0eff3f3c91752c31fd342bedcd8f465460324046591d0be",
    "foundation_sft/single_teacher_overfit_report.py": "323f1547b39b53491a733d88547d9d6e4a43ee28aaee491e8110a55b0f05d0b8",
    "foundation_sft/configs/single_teacher_overfit.json": "6c7508c1021e7623d89ad07ae183caea59a4c5b3fa3bd92dfac0add638d02dcb",
    "foundation_sft/SINGLE_TEACHER_OVERFIT.md": "5689054621a606305ef9494b25ef6f79df43197f776f1417baf97b6cf08d41f8",
    "run_single_teacher_overfit.py": "d90199071f65786e32cd06d286ce0738b45c282c1de7953e183212df62f2582d"
}
PREREQUISITES = (
    "run_single_teacher_distillation.py",
    "foundation_sft/train.py",
    "foundation_sft/data.py",
    "foundation_sft/metrics.py",
    "foundation_sft/distillation_suite.py",
    "foundation_sft/distillation_data.py",
    "foundation_sft/distillation_losses.py",
    "foundation_sft/training_history.py",
    "foundation_sft/models/vit_tiny_student.py",
    "foundation_sft/configs/distillation_rad_dino_vit_tiny.json",
)
BACKUP_DIRECTORY = ".single-teacher-overfit-backups"
LOCK_NAME = ".single-teacher-overfit-update.lock"
ENGINE_FILE = "update_partial_finetune_study.py"
ENGINE_SHA256 = "aa1450ef15999ca309e12bf88463322879e94b457335deb33d265a0ea8ff2266"


def _load_engine():
    """Load the unchanged shared installer privately, preserving older updates."""
    path = Path(__file__).resolve().with_name(ENGINE_FILE)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("The included installer engine checksum does not match")
    spec = importlib.util.spec_from_file_location("_single_teacher_overfit_update_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


_engine = _load_engine()


def _configured_engine():
    for name in ("PAYLOAD_FILES", "PREVIOUS_SHA256", "PAYLOAD_SHA256", "PREREQUISITES",
                 "BACKUP_DIRECTORY", "LOCK_NAME"):
        setattr(_engine, name, globals()[name])
    _engine.__doc__ = __doc__
    return _engine


def install(project, source_dir, apply=False):
    return _configured_engine().install(project, source_dir, apply)


def rollback(project, backup_dir, apply=False):
    return _configured_engine().rollback(project, backup_dir, apply)


def main(argv=None):
    return _configured_engine().main(argv)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        print("Single-teacher overfit update installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
