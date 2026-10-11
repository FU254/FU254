#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quickly verify that PyCharm can receive Matplotlib figures."""

from tiny_regression_common import (
    PLOT_DISPLAY_VERSION,
    PYCHARM_PLOT_BACKEND,
    PYCHARM_PLOT_BACKEND_ERROR,
    plt,
)


print(f"绘图模式: {PLOT_DISPLAY_VERSION}")
print(f"Matplotlib backend: {plt.get_backend()}")
print(f"PyCharm backend: {PYCHARM_PLOT_BACKEND or '未找到'}")
if PYCHARM_PLOT_BACKEND_ERROR:
    print(f"PyCharm backend 加载失败: {PYCHARM_PLOT_BACKEND_ERROR}")

plt.figure(figsize=(8, 5), dpi=140)
plt.plot([1, 2, 3, 4], [1, 4, 2, 5], marker="o", linewidth=2)
plt.title("PyCharm Plot Test", fontsize=15, fontweight="bold")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, alpha=0.3)
plt.tight_layout()

if "interagg" not in str(plt.get_backend()).lower():
    raise RuntimeError(
        "未检测到 PyCharm interagg 后端。请在 PyCharm 中启用 "
        "Settings -> Tools -> Python Scientific -> Show plots in tool window，"
        "然后重新运行本文件。"
    )

plt.show()
print("测试图已发送到 PyCharm Plots 工具窗口。")
