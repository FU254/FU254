# ViT-Tiny 全参 / LoRA × 监督 / 蒸馏四组实验

本流程将“哪些参数更新”和“用什么损失训练”分开比较：

| 组别 | 学生参数更新方式 | 训练目标 |
| --- | --- | --- |
| A | 全参数微调 | 真实标签监督 |
| B | 全参数微调 | 真实标签监督 + 单教师蒸馏 |
| C | LoRA + 同一线性分类头 | 真实标签监督 |
| D | LoRA + 同一线性分类头 | 真实标签监督 + 单教师蒸馏 |

“监督”表示利用真实标签计算任务损失，不代表冻结骨干。“全参 / LoRA”表示更新哪些参数。四组都从同一份 ImageNet 预训练 ViT-Tiny 初始化，全参组是预训练模型的完整微调，不是从零训练。

蒸馏教师固定为已完成该任务的 RAD-DINO LoRA 模型。B、D 两组使用相同教师，教师处于评估模式、关闭梯度，学生计算图保持完整。原有教师权重、数据划分、灰度像素缓存和历史结果继续使用。

## 训练设置和比较原则

LoRA 插入学生 12 个注意力层的 fused QKV，rank=8、alpha=16，并训练与全参组相同的单层线性分类头。LoRA 加分类头共有 73,921 个可训练参数；蒸馏使用的全局特征投影另外增加 147,456 个可训练参数，仅在训练时对齐特征。

四组控制相同的数据划分、初始学生状态、随机种子、批次与空间增强；骨干全参学习率及 LoRA 学习率都设为 `5e-5`，分类头和投影学习率为 `1e-4`，权重衰减及其余训练设置保持一致。默认每组最多 100 轮，早停 patience=15，使用 FP32。默认预先指定种子 42、43、44，共四组 × 三种子，而不是根据测试分数挑选一个种子。

蒸馏保留三项约束：温度 2 的二分类软预测 KL、全局特征余弦距离、局部 patch 关系损失，权重依次为 0.5、0.1、1.0。局部特征池化至 7×7，逐位置归一化后比较 Gram 相似度矩阵的非对角元素。教师和学生共享同一视野与空间增强，输入尺寸和归一化分别匹配模型。这些是实验预设，尚未证明为本数据的最优参数。

LoRA 减少训练时的可训练参数和优化器状态，可能改变模型拟合能力，但不保证防止过拟合。它也不会把同一个 ViT-Tiny 学生在部署时自动变成更小网络。模型选择继续依据验证集患者 ROC-AUC，阈值由验证集确定，测试集在选择完成后固定阈值评估。

## 把增量包装入现有项目

本更新适用于已经安装原单教师蒸馏流程的项目。将 `FU254-distillation-abcd-update.zip` 在 Windows 解压到 `D:\99\EVA-X-main`，得到：

```text
D:\99\EVA-X-main\distillation_abcd_update\update_distillation_abcd.py
```

通过 PyCharm 上传整个 `distillation_abcd_update` 文件夹，服务器应对应：

```text
/home/cfu/EVA-X-main/distillation_abcd_update/update_distillation_abcd.py
```

使用项目解释器先预览：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_abcd_update/update_distillation_abcd.py --project /home/cfu/EVA-X-main
```

执行安装：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/distillation_abcd_update/update_distillation_abcd.py --project /home/cfu/EVA-X-main --apply
```

安装器先验证全部来源 SHA256，再核对目标。只允许替换原发布版本的 `foundation_sft/distillation.py`、`foundation_sft/models/vit_tiny_student.py`；目标已与新版本相同时跳过，发现其他本地修改则整批拒绝。其余五个文件只新增，不覆盖用户已有的不同内容。

旧全参配置、旧低学习率配置、模型注册表、原监督训练器、权重、缓存和历史实验均不替换。安装器不联网、不升级依赖。重复安装内容相同的包不会重复修改文件。

原代码的完整字节、权限和安装记录备份在项目的 `.distillation-abcd-backups/<时间和标识>/`。安装中断会回退已完成的修改；若发现期间又有用户改动，则保留该改动并明确报出冲突。

需要恢复此次安装前的代码时，使用安装输出的实际 `backup` 目录，先预览后执行：

```bash
python /home/cfu/EVA-X-main/distillation_abcd_update/update_distillation_abcd.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.distillation-abcd-backups/实际备份目录
python /home/cfu/EVA-X-main/distillation_abcd_update/update_distillation_abcd.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.distillation-abcd-backups/实际备份目录 --apply
```

回退前会完整校验备份与当前代码，后来修改的代码或配置会被保留，不会强制覆盖；回退不删除训练结果。

## 先列计划，再检查和运行

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_distillation_experiments.py --plan
```

默认配置为 `foundation_sft/configs/distillation_abcd.json`。计划列出四组 × 三种子以及设备安排，默认使用 GPU 0–3。检查预训练权重、教师恢复、学生梯度和保存/重载流程：

```bash
python run_distillation_experiments.py --check-only
```

检查使用训练批次，不会执行完整训练或全量像素审计。确认检查成功后开始正式实验：

```bash
python run_distillation_experiments.py
```

查看已保存的结果：

```bash
python run_distillation_experiments.py --show-metrics
```

默认配置中明确指定的旧 A / B、seed 42 结果，来自 `suite_20261019T110343Z_3b1acd96`，必须通过教师、初始化、数据划分、训练配置、输入视野等兼容性核对后才会复用。未通过会报告具体原因，并按预先指定的组别和种子重新训练；不会扫描历史目录寻找测试成绩最高的批次。原历史目录保持不变。

每次运行创建 `runs/distillation_abcd/suites/suite_时间_标识/`。其中 `suite.json` 记录每个任务的状态、复用来源、核对失败原因和尝试记录；`report/comparison.csv` 列出逐组逐种子指标，`report/summary.csv` 列出跨种子的均值、样本标准差和实际完成数量。单一种子的标准差留空，失败任务不伪装为完成或填零。

查看单次批次或继续其任务队列时，指定实际批次路径：

```bash
python run_distillation_experiments.py --show-metrics --suite /home/cfu/EVA-X-main/runs/distillation_abcd/suites/实际批次目录
python run_distillation_experiments.py --resume-suite /home/cfu/EVA-X-main/runs/distillation_abcd/suites/实际批次目录
```

继续队列要求原实验协议一致。已完成学生经过重新核对后复用，未完成任务重新训练；这项功能不是恢复某个学生训练到一半的优化器状态。需要调整任务范围时，`--groups A B C D`、`--seeds 42 43 44`、`--devices cuda:0 cuda:1 cuda:2 cuda:3` 和 `--max-parallel 4` 可以显式指定计划；`--plan` 始终只显示计划，不启动训练。

每个学生均保存完整状态和输入视野元数据，LoRA 学生恢复包含骨干、LoRA 参数与分类头，推理可以独立于教师和 ImageNet 初始化文件完成。恢复图像输入时仍应使用 `foundation_sft.distillation.inference_dataset`，保持保存的共享视野处理。

## 仅单独运行 LoRA 两组

旧单教师入口仍然可用。指定新增 LoRA 配置时，依次运行 C / D 两组，并使用独立结果目录 `runs/single_teacher_distillation_lora`：

```bash
python run_single_teacher_distillation.py --config foundation_sft/configs/distillation_rad_dino_vit_tiny_lora.json --check-only
python run_single_teacher_distillation.py --config foundation_sft/configs/distillation_rad_dino_vit_tiny_lora.json
python run_single_teacher_distillation.py --config foundation_sft/configs/distillation_rad_dino_vit_tiny_lora.json --show-metrics
```

不指定新 LoRA 配置的旧单教师命令继续保留原来的全参行为。原全参 checkpoint 也继续支持恢复。

四组比较时，应同时查看每次实验、三种子的均值/标准差，以及每次患者 bootstrap 置信区间。跨种子波动和患者采样不确定性表达的是不同含义；不能用其中一次测试结果决定保留哪个种子。
