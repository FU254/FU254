# 随机学生的 DropPath 对照实验

本轮固定 `weight_decay=0.01`、随机初始化、全参数训练和 seed42，只新增 A/B 两项 `DropPath=0.1` 训练。A 使用真实标签，B 使用真实标签＋软标签、全局特征和局部关系三项蒸馏。

| 条件 | weight_decay | DropPath | 任务 |
| --- | --- | --- | --- |
| 已完成对照 | 0.01 | 0 | 原 A/B seed42，只读核验 |
| 新实验 | 0.01 | 0.1 | 新增 A/B seed42，两项 |

对照固定来自已完成的权重衰减实验：

```text
runs/random_student_weight_decay/studies/study_20261025T045150Z_b978d3c4
```

程序读取该研究中 `wd_0p01` 子批次的 A/B seed42 完成记录和实际生效配置，仅增加学生的 `training.drop_path_rate=0.1`。教师仍为同一个冻结的 RAD-DINO LoRA 最佳验证检查点；学生参数和 B 组投影器的初始 SHA256、患者划分、共享视野、学习率、增强和蒸馏系数均与对应对照核对。

这里的 `0.1` 是12层中**线性递增的最大 DropPath 概率**，从首层0递增至末层约0.1；仅在训练模式启用。它属于随机深度，不是普通 Dropout，也不是每层固定10%的丢弃率。教师保持评估模式。

DropPath 使用隔离的 Torch 随机流，并把状态保存到检查点额外字段中，避免改变数据增强的外部随机数序列。模型参数状态不增加随机流 buffer，所以可以继续核对相同初始参数。新检查点明确保存 DropPath 配置，重载时重建同一模型并恢复独立随机流；已有 DropPath=0 检查点的元数据格式保持兼容。

## 安装

本包包含三个经版本核对的兼容性更新，以及本轮新增的入口、配置、来源审计、报告和说明文件。安装器先核验 SHA256 和语法，备份后原子更新；未知本地修改会保留并停止安装。包内没有患者数据、权重或训练结果。

先在服务器检查原任务是否已经结束：

```bash
pgrep -u "$(id -u)" -af '[r]un_(droppath_experiments|weight_decay_experiments|random_student_experiments|random_student_distillation)\.py'
```

有相关训练进程时，先等该任务完成再安装，避免同一运行过程混用训练模块版本。

将 `FU254-droppath-study.zip` 解压到 Windows 项目目录，得到 `D:\99\EVA-X-main\droppath_update\`。通过 PyCharm 上传整个 `droppath_update` 文件夹到 `/home/cfu/EVA-X-main/`，再在服务器执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/droppath_update/update_droppath_study.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/droppath_update --apply
```

## 检查与后台启动

查看计划时读取原研究和文件指纹，不加载 GPU 或开始训练：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_droppath_experiments.py --plan
```

计划应为两个新任务：A seed42、B seed42，`weight_decay=0.01`、`drop_path_rate=0.1`。实际启动前会核验原完整检查点、来源和配置，并进行前后向及保存重载检查。单独使用 `--check-only` 时只做检查。

以下整段会自动选择一张当前剩余显存至少20GB、利用率不超过5%的卡，串行运行两项新训练：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
mkdir -p runs/random_student_droppath

droppath_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_(droppath_experiments|weight_decay_experiments|random_student_experiments|random_student_distillation)\.py'; then
    echo "已有相关实验进程，未重复启动。"
elif [ -z "$droppath_gpu" ]; then
    echo "当前没有符合条件的空闲卡，未启动。"
    nvidia-smi
else
    droppath_log="runs/random_student_droppath/launch_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_droppath_experiments.py \
        --devices "cuda:$droppath_gpu" --max-parallel 1 \
        --cache-dir runs/foundation_sft/data_cache \
        > "$droppath_log" 2>&1 < /dev/null &
    echo "使用 GPU $droppath_gpu；PID：$!；日志：$droppath_log"
    tail -n 30 -f "$droppath_log"
fi
```

按 `Ctrl+C` 退出日志查看，后台训练继续。若确认两张卡都有足够空余，可使用 `--devices cuda:0 cuda:1 --max-parallel 2`，每卡最多同时一个独立任务。

## 查看结果与恢复

新研究输出到 `runs/random_student_droppath/studies/study_.../`，其 `study.json` 固定记录对照指纹、子批次路径和执行状态。旧对照结果只读，新结果单独保存。

```bash
python run_droppath_experiments.py --show-metrics
```

只查看指定研究时，将路径末尾替换为启动时打印的真实研究目录：

```bash
python run_droppath_experiments.py --show-metrics --study /home/cfu/EVA-X-main/runs/random_student_droppath/studies/实际研究目录
```

**单项训练日志正常结束后，也可能以 `[evaluation] Random student reloaded; evaluating test once` 作为最后一行。** 单个旧日志停止更新不能用于判断当前任务是否卡住；用上述总状态查看命令确认 `completed`、`training` 或 `failed`，并结合进程状态判断。

进程已结束且有失败项时，可在确认没有同类进程后，通过 `nohup` 和新日志运行下面的恢复命令：

```bash
python -u run_droppath_experiments.py --resume-study /home/cfu/EVA-X-main/runs/random_student_droppath/studies/实际研究目录 --devices cuda:0 --max-parallel 1 --cache-dir runs/foundation_sft/data_cache
```

恢复允许更换 GPU、并发数和缓存位置。完成任务经核验后保留；未完成任务从头训练，不是从最后一个 epoch 续训。科学设置、来源和固定子批次路径必须一致。

## 评价规则

报告包含 `comparison.csv`、`summary.csv`、`matched_seed_differences.csv`、`overfitting.csv` 和 `validation_selection.json`。只比较原 A/B seed42 与新 A/B seed42，共四条结果；单个种子无法估计种子标准差或稳定性。

四条结果完整后，按 B seed42 的验证集 ROC-AUC 比较 DropPath=0 与0.1，精确并列时保留原对照。A 使用同一选定配方作为配对监督对照。验证 PR-AUC 作为辅助指标，测试指标完整报告，不用于选参。

过拟合诊断使用同一轮的 `train_eval_loss` 与 `val_loss`，两者均为评估模式、无随机增强下的真实标签 Focal 损失；不把含蒸馏项的优化总损失混入。另比较训练评估与验证的患者 ROC-AUC。报告分别记录最佳验证轮与停止轮，测试使用最佳验证检查点。

DropPath 不保证消除过拟合。应结合完整逐轮曲线，确认训练—验证分化是否减轻、验证表现是否保持或提高；不同设置的最佳轮可能不同。本轮只检验在 `weight_decay=0.01` 配方下增加 DropPath 的作用，不对随机种子稳定性或更广泛设置作结论。
