#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVA-X Tiny regression entry point for measured FVC."""

from tiny_regression_common import run_regression_task


if __name__ == "__main__":
    run_regression_task(
        label_column="fvc实际值",
        task_name="tiny_fvc_regression",
    )
