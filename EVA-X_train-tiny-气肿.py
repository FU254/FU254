#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = SCRIPT_DIR / "tiny-new-\u80ba\u6c14.py"


def load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("tiny_new_lung_gas_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_module(module) -> None:
    # Keep the historical EVA-X_train-tiny-气肿.py entrypoint, but run the
    # newer script that already supports label lookup by accession number.
    module.SCRIPT_DIR = str(SCRIPT_DIR)
    module.TASK_NAME = "tiny_main"
    module.OUTPUT_DIR = os.path.join(module.SCRIPT_DIR, "runs", module.TASK_NAME)
    module.EXCEL_PATH1 = "/home/cfu/.virtualenvs/DR/TEST/DR\u5df2\u7b5b\u90092019-2023.xlsx"
    module.EXCEL_PATH2 = "/tmp/pycharm_project_601/\u80ba\u529f\u80fd\u4e07\u80fd\u8868\u6700\u7ec8\u72481.xlsx"
    module.DICOM_DIR = "/tmp/pycharm_project_601/DR\u6570\u636e"
    module.LABEL_COLUMN = "\u6c14\u4f53\u6f74\u7559 / \u80ba\u8fc7\u5ea6\u5145\u6c14"
    module.FREEZE_BACKBONE = True
    module.UNFREEZE_LAST_N_BLOCKS = 1

    module.LABEL_TAG = module.make_safe_name(module.LABEL_COLUMN)
    module.CACHE_FILE = os.path.join(module.OUTPUT_DIR, f"cached_data_{module.LABEL_TAG}.npz")
    module.BEST_MODEL_FILE = os.path.join(
        module.OUTPUT_DIR,
        f"best_evax_tiny_model_{module.LABEL_TAG}.pth",
    )
    module.RUN_INFO_FILE = os.path.join(
        module.OUTPUT_DIR,
        f"run_info_{module.LABEL_TAG}.txt",
    )


def main() -> None:
    module = load_module(SOURCE_SCRIPT)
    configure_module(module)
    module.main()


if __name__ == "__main__":
    main()
