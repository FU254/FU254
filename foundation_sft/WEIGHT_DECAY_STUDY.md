# 随机学生的权重衰减实验

本轮以已完成的随机初始化 ViT-Tiny A/B 实验为基线，新增 AdamW 权重衰减 `0.001`、`0.01` 两档。每档仅使用 **seed42**，分别运行 A/B，共4项新训练。

| 条件 | 学生初始化与更新 | 训练信号 | 权重衰减 |
| --- | --- | --- | --- |
| 既有 A/B，基线 | 随机初始化、全参数训练 | A：真实标签；B：真实标签＋三项蒸馏 | 读取原结果中的实际值 |
| 新 A/B，候选一 | 相同 | 相同 | 0.001 |
| 新 A/B，候选二 | 相同 | 相同 | 0.01 |

基线固定为：

```text
runs/random_student_distillation/suites/suite_20261023T131255Z_64117fb7
```

程序仍核验基线六个完成结果的完整性，并从其 `config.json` 读取**实际生效的完整训练设置**，记录原权重衰减；本轮比较只使用已有的 **A seed42、B seed42** 两项结果作为对照，不重训基线。新任务仅替换 `weight_decay`。原始顶层配置未必显式写出该值，因此本实验不假定基线一定是 `0.0001`。基线缺项、来源不一致或与候选值重复时会停止，避免把不同实验误作同一对照。

教师仍为同一个冻结的 RAD-DINO LoRA 最佳验证检查点，学生保持随机初始化和全参数训练。学习率、早停、数据划分、共享视野、评价规则与三项蒸馏系数保持原设置。学生初始参数及 B 组投影器初始参数均与同种子基线核对。

当前优化器对其全部参数组统一使用 `weight_decay`，因此 B 组的本轮设置同时作用于**学生主干、分类头和可训练蒸馏投影器**。本轮检验的是这个全局 AdamW 权重衰减设置，不将结果解释为仅对主干施加正则化的独立效应。

## 安装

将 `FU254-weight-decay-study.zip` 解压到 Windows 项目目录，得到：

```text
D:\99\EVA-X-main\weight_decay_update\update_weight_decay_study.py
```

通过 PyCharm 上传整个 `weight_decay_update` 文件夹到 `/home/cfu/EVA-X-main/`，再在服务器终端执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/weight_decay_update/update_weight_decay_study.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/weight_decay_update --apply
```

安装器只新增本轮入口、配置、审计、报告和说明文件；已有训练代码不替换。全部载荷经 SHA256 和语法检查后原子安装，发现同名本地修改时保留文件并停止。本包不包含患者数据、权重或训练结果。学生无需 ImageNet 权重；原教师基础权重和已微调的 LoRA/分类头检查点仍须保留。

## 检查计划

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_weight_decay_experiments.py --plan
```

计划会列出基线实际权重衰减与4项 seed42 新任务。只做数据、前后向和来源检查时使用 `--check-only`；该参数不进行正式训练。实际训练前会核验基线完整检查点及评估文件，并在每项预检查中核对除权重衰减外的实验设置。

## 后台启动

以下整段默认只使用一张当前剩余显存至少20GB、GPU利用率不超过5%的卡，并发数为1。它会检查是否已有同类进程，再用 `nohup` 运行：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
mkdir -p runs/random_student_weight_decay

weight_decay_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_(weight_decay_experiments|random_student_experiments|random_student_distillation)\.py'; then
    echo "已有相关实验进程，未重复启动。"
elif [ -z "$weight_decay_gpu" ]; then
    echo "当前没有符合条件的空闲卡，未启动。"
    nvidia-smi
else
    weight_decay_log="runs/random_student_weight_decay/launch_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_weight_decay_experiments.py \
        --devices "cuda:$weight_decay_gpu" --max-parallel 1 \
        --cache-dir runs/foundation_sft/data_cache \
        > "$weight_decay_log" 2>&1 < /dev/null &
    echo "使用 GPU $weight_decay_gpu；PID：$!；日志：$weight_decay_log"
    tail -n 30 -f "$weight_decay_log"
fi
```

按 `Ctrl+C` 退出日志查看，后台实验继续。训练沿用原随机学生入口，会输出每轮及批次进度、同口径训练集评估损失和验证损失。

两档权重衰减依次执行；每档内部的两项任务可分配到两张空闲卡，每张卡同时最多一个任务。如果确认 GPU 0、1 都有足够空余，可在启动前将设备与并发参数改为：

```text
--devices cuda:0 cuda:1 --max-parallel 2
```

## 结果与恢复

父实验目录为 `runs/random_student_weight_decay/studies/study_.../`，其 `study.json` 固定记录基线指纹、两档子批次路径和执行状态。基线六项结果只读，新训练不会覆盖它们；新批次也不会导入其他历史训练结果。

打印已保存的研究结果：

```bash
python run_weight_decay_experiments.py --show-metrics
```

只打印指定研究时，把路径末尾替换为启动时输出的真实目录：

```bash
python run_weight_decay_experiments.py --show-metrics --study /home/cfu/EVA-X-main/runs/random_student_weight_decay/studies/实际研究目录
```

报告目录包含逐任务 `comparison.csv`、按方案汇总的 `summary.csv`、同种子相对基线和 A/B 对照的 `matched_seed_differences.csv`、同轮过拟合诊断 `overfitting.csv`，以及依据验证集制定的 `validation_selection.json`。本轮共比较6项结果，即基线 seed42 两项和新增四项；每个方案只有一个种子，不能估计跨种子标准差。PR-AUC 与 AP 分别报告。

若进程已经结束而仍有失败任务，先用启动块中的 `pgrep` 检查确认相关任务不再运行，再通过 `nohup` 和新日志启动下列恢复命令：

```bash
python -u run_weight_decay_experiments.py --resume-study /home/cfu/EVA-X-main/runs/random_student_weight_decay/studies/实际研究目录 --devices cuda:0 --max-parallel 1 --cache-dir runs/foundation_sft/data_cache
```

恢复时可以更换空闲 GPU 和并发数。程序重新检查并保留核验通过的完成任务；未完成任务从头训练，而非从最后一个 epoch 续训。源基线和科学设置必须与研究记录一致。每档子批次的固定路径在启动前即被记录，恢复不依据测试分数或最新目录猜测来源。

## 如何判断是否改善

待基线 seed42 两项和两档候选四项的全部6项结果完整后，按每档 **B seed42 的验证集 ROC-AUC**推荐一档；A seed42 作为该档相同训练配方下的监督对照，保持同档 A/B 配对。精确并列时优先保留基线，再优先较小的候选权重衰减。辅助观察验证 PR-AUC 和训练—验证分化；基线也参与比较，新增设置未必更好。结果尚未完整时不推荐参数。测试指标完整保留用于最终报告，不用于挑选权重衰减。

这是单种子探索筛选，不能评价随机种子稳定性，也不能据此宣称统计显著；后续必要时用多个种子确认选定方案。本轮保持原模型配置，不加入 DropPath；若权重衰减效果不足，再单独设计 DropPath 实验。

过拟合诊断比较同一轮的 `train_eval_loss` 与 `val_loss`，两者都是真实标签 Focal 损失、评估模式、无随机增强；不要把包含蒸馏项的优化总损失 `train_loss` 与验证损失作差。`train_eval_patient_roc_auc` 与 `val_patient_roc_auc` 使用患者级口径。

各完成任务已有 `training_loss.png` 和 `history.json`。报告分别记录最佳验证轮与停止轮；实际测试来自最佳验证检查点。判断过拟合需看连续多轮的损失和 AUC 走势，不能仅比较最大验证 AUC 与最后一轮。

权重衰减更大不能保证消除过拟合，也可能限制随机学生的学习能力。只有验证表现保持或提高、并且持续分化减轻时，才支持这项调整有效；训练和验证成绩一起下降而差距缩小，可能只是学习不足。
