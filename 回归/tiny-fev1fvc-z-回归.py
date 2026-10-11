#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVA-X Tiny regression entry point for measured FEV1/FVC."""

from tiny_regression_common import run_regression_task


if __name__ == "__main__":
    run_regression_task(
        label_column="fev1fvc_z",
        task_name="tiny_fev1fvc_regression",
    )

