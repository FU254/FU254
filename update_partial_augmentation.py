#!/usr/bin/env python3
"""Install the validation-only last-eight-block intensity augmentation update."""

import hashlib
import importlib.util
from pathlib import Path
import sys


PAYLOAD_FILES = (
    "foundation_sft/data.py",
    "foundation_sft/distillation.py",
    "foundation_sft/teacher_weight_training.py",
    "foundation_sft/partial_finetune_study.py",
    "foundation_sft/partial_finetune_report.py",
    "foundation_sft/configs/partial_finetune_augmentation.json",
    "foundation_sft/PARTIAL_FINETUNE_AUGMENTATION.md",
)
BASE_REVISION = "8300504692225b809ea625356da373a9da2a583d"
PREVIOUS_SHA256 = {
    "foundation_sft/data.py": "985a8b083928b27313c04cf7c916f9899d8fd6e4db4870f5786c5c20d748afc6",
    "foundation_sft/distillation.py": "e8a8d9ed294098c07cda26a1ec98a1f7a2a05d64f8817f55e59543d7b081ae78",
    "foundation_sft/teacher_weight_training.py": "93fd9b3638e73663eb984eec23158304ca4bcc7cc625d93c369fa27225500d43",
    "foundation_sft/partial_finetune_study.py": "05b5219ec3f7bcd08d3a3dbef72dc62c414a0e3238799b8d877c19c6f00fd329",
    "foundation_sft/partial_finetune_report.py": "5bea8f8977ecaa1c2ac75e2f7c0805dd62ec39106f435f455106762980ac841f",
}
# Frozen only after the final reviewed payload is ready for distribution.
PAYLOAD_SHA256 = {
    "foundation_sft/data.py": "6103d1073b5a71a0513915ae78d1ea0689971bb60889fa787e4f90d06087e535",
    "foundation_sft/distillation.py": "8ac7c2ebca8f7459d0486d6374e14175ac83c8becbe3b69ac59df8afd5024272",
    "foundation_sft/teacher_weight_training.py": "891264b94a7bec1fd71f7dc6089f6cde06c3cb8402abcab3c3039845246ba3db",
    "foundation_sft/partial_finetune_study.py": "d72b057771e8c57763868a4c0d94138586eb227633a36cad5ef6315b38e7e874",
    "foundation_sft/partial_finetune_report.py": "c3692558189813ed48a4f6b4dd8a22e39d3119c6f9c00f3acdf3b39ab8fda975",
    "foundation_sft/configs/partial_finetune_augmentation.json": "2264bab825ec7c3ed06f4b8146bf4f2a696dea76369f6e717ffe43a7f5d2518e",
    "foundation_sft/PARTIAL_FINETUNE_AUGMENTATION.md": "a20b8b8a3022880d05e8cb6d3c83dc0324f6d1e91c138399324ce7507449fc12"
}
PREREQUISITES = (
    "run_partial_finetune_experiments.py",
    "foundation_sft/partial_finetune_study.py",
    "foundation_sft/partial_finetune_report.py",
    "foundation_sft/models/vit_tiny_student.py",
    "foundation_sft/data.py",
    "foundation_sft/train.py",
    "foundation_sft/distillation_suite.py",
    "foundation_sft/equal_teacher_sources.py",
    "foundation_sft/equal_teacher_data.py",
    "foundation_sft/teacher_weight_study.py",
    "foundation_sft/teacher_weight_report.py",
    "foundation_sft/teacher_weight_losses.py",
    "foundation_sft/configs/teacher_weight_study.json",
    "foundation_sft/configs/partial_finetune_study.json",
    "run_teacher_weight_experiments.py",
)
BACKUP_DIRECTORY = ".partial-augmentation-backups"
LOCK_NAME = ".partial-augmentation-update.lock"
ENGINE_FILE = "update_partial_finetune_study.py"
ENGINE_SHA256 = "aa1450ef15999ca309e12bf88463322879e94b457335deb33d265a0ea8ff2266"


def _load_engine():
    """Use a private module so configuring this update leaves the old one intact."""
    path = Path(__file__).resolve().with_name(ENGINE_FILE)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("The included installer engine checksum does not match")
    spec = importlib.util.spec_from_file_location("_partial_augmentation_update_engine", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


_engine = _load_engine()


def _configured_engine():
    for name in ("PAYLOAD_FILES", "PREVIOUS_SHA256", "PAYLOAD_SHA256", "PREREQUISITES",
                 "BACKUP_DIRECTORY", "LOCK_NAME"):
        setattr(_engine, name, globals()[name])
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
        print("Partial augmentation update installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
