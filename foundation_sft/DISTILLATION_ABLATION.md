# 单教师蒸馏损失消融：六组 × 三种子

本实验固定 RAD-DINO LoRA 教师和全参数微调的 ViT-Tiny 学生，只改变三项蒸馏损失是否启用。每组都保留真实标签的 Focal Loss；教师处于评估模式且不更新参数，学生从同一份 ImageNet 权重初始化。

总损失为 `Focal + soft_weight × soft + global_weight × global + local_weight × local`：

| 组别 | 中文含义 | soft | global | local |
| --- | --- | --- | --- | --- |
| `supervised` | 纯标签监督 | 0 | 0 | 0 |
| `soft_only` | 标签监督 + 软预测蒸馏 | 0.5 | 0 | 0 |
| `no_local` | 去掉局部项 | 0.5 | 0.1 | 0 |
| `no_global` | 去掉全局项 | 0.5 | 0 | 1.0 |
| `no_soft` | 去掉软预测项 | 0 | 0.1 | 1.0 |
| `full_kd` | 完整三项蒸馏 | 0.5 | 0.1 | 1.0 |

删除一项时将该项权重置零，其余权重保持原值，不重新归一化或单独调参。软预测温度固定为 2，局部 patch 池化固定为 7×7，局部损失继续比较归一化 patch Gram 矩阵的非对角关系。学习率、权重衰减、分类头、数据划分、输入视野、空间增强、轮数和早停等其余训练设置与原全参实验保持一致。

默认种子为 42、43、44，共 18 个预先指定的任务。`supervised` 与 ABCD 的 A 组、`full_kd` 与 ABCD 的 B 组协议一致。若旧 A/B 的六个任务均完成且通过兼容性核对，可复用六个，只新增其余四组 × 三种子 = 12 次训练；缺失或核对失败的任务按预设组别和种子补训。

## 安装到已完成 ABCD 更新的项目

在 Windows 将 `FU254-distillation-ablation-update.zip` 解压到 `D:\99\EVA-X-main`，得到：

```text
D:\99\EVA-X-main\distillation_ablation_update\update_distillation_ablation.py
```

通过 PyCharm 上传整个 `distillation_ablation_update` 目录，服务器对应：

```text
/home/cfu/EVA-X-main/distillation_ablation_update/update_distillation_ablation.py
```

先预览，再安装：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_ablation_update/update_distillation_ablation.py --project /home/cfu/EVA-X-main
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_ablation_update/update_distillation_ablation.py --project /home/cfu/EVA-X-main --apply
```

增量包只升级经过 SHA256 核对的官方 `foundation_sft/distillation_suite.py`，并新增消融入口、配置、报告模块和本文档。其他本地修改会触发整批拒绝；目标已是同一新版本时跳过，重复安装幂等。

原训练器、蒸馏损失实现、学生实现、ABCD 配置、全参和低学习率配置、权重、缓存、数据划分及历史结果均保留，不升级 Python 依赖。完整旧协调器字节、权限和安装记录保存在项目 `.distillation-ablation-backups/<时间和标识>/`。安装中断会恢复本次已完成的修改；若期间出现用户新修改，会保留并报告冲突。

需要回退时，使用安装输出的实际备份路径；不加 `--apply` 只预览：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_ablation_update/update_distillation_ablation.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.distillation-ablation-backups/实际备份目录
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_ablation_update/update_distillation_ablation.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.distillation-ablation-backups/实际备份目录 --apply
```

回退校验当前代码与备份，不覆盖后续用户修改，不删除训练结果。

## 直接启动检查与训练

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_distillation_ablation.py
```

入口自带训练前检查、严格复用核对和训练调度。默认使用 GPU 0–3，同一张 GPU 同时只执行一个任务。每次运行创建独立的 `runs/distillation_ablation/suites/suite_时间_标识/`，不覆盖旧批次。

需要先单独查看计划或只做检查时：

```bash
python run_distillation_ablation.py --plan
python run_distillation_ablation.py --check-only
```

`--plan` 只解析配置和列任务；`--check-only` 检查实际训练批次的前向、反向和保存重载，不运行完整训练或全量像素审计。

默认配置 `foundation_sft/configs/distillation_ablation.json` 按以下明确顺序尝试复用：

1. `runs/distillation_abcd/suites/suite_20261020T081023Z_ac300699` 中已完成且匹配的 A/B 任务；即使整个批次是 `partial`，其中有效的完成任务仍可核对复用。
2. `runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96`，作为原 seed 42 的后备来源。

复用需要教师、学生初始化、数据划分、训练参数、损失权重、输入视野和实际保存结果一致。协调器不根据测试成绩选择来源，也不因为批次目录存在就宣称复用成功；失败原因写入新批次的任务记录。

查看结果：

```bash
python run_distillation_ablation.py --show-metrics
python run_distillation_ablation.py --show-metrics --suite /home/cfu/EVA-X-main/runs/distillation_ablation/suites/实际批次目录
```

继续同一个批次队列时：

```bash
python run_distillation_ablation.py --resume-suite /home/cfu/EVA-X-main/runs/distillation_ablation/suites/实际批次目录
```

继续队列会核对原协议并复用合格的已完成任务，未完成任务重新训练。`--groups`、`--seeds`、`--devices`、`--max-parallel` 可指定任务范围和设备，`--reuse-suite` 可追加明确的旧来源，`--no-reuse` 可选择全部重新训练。改变实验协议时应启动新批次。

## 查看每项损失的贡献

报告同时保留逐次结果和跨种子汇总：

| 文件 | 内容 |
| --- | --- |
| `suite.json` | 每个任务的状态、来源、尝试和核对失败原因 |
| `report/comparison.csv` | 六组逐种子患者指标、验证 ROC-AUC、测试 CI 和任务状态 |
| `report/summary.csv` | 每组跨种子的均值、样本标准差和实际完成数量 |
| `report/paired_differences.csv` | 同一种子下 `full_kd` 减去纯监督、软预测单项及各去项组的指标差值 |
| `report/paired_summary.csv` | 上述配对差值的跨种子均值、样本标准差和实际配对数量 |

例如，`full_kd - no_local` 表示保留其余损失时加入局部约束的差异；`full_kd - soft_only` 表示全局与局部约束联合加入的差异。各项可能相互作用，这些差值不应直接相加当作完整模型的总提升。ROC-AUC、PR-AUC 等越高越好的指标，正差值表示该指标提高。

配对比较只使用两边均已完成且种子相同的任务。失败、缺失或未配对任务不填零；一种子不能计算种子间标准差。每次患者 bootstrap 置信区间描述患者采样不确定性，跨种子标准差描述重复训练波动，两者分别报告，不根据测试结果筛掉不利种子。

原学生 checkpoint 和独立推理流程继续可用；复算时沿用保存的视野元数据及验证集阈值，不重新选择测试阈值。
