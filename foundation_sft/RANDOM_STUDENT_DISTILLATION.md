# 随机初始化 ViT-Tiny 学生实验

本轮固定已选的 RAD-DINO LoRA 教师，使用 seed42 的验证集最优 `best.pt`，教师参数冻结。学生为随机初始化的 ViT-Tiny，采用全参数训练。

| 组别 | 学生初始化 | 学生更新方式 | 训练信号 | 种子 |
| --- | --- | --- | --- | --- |
| A | 随机初始化 | 全参数训练 | 真实标签 | 42、43、44 |
| B | 随机初始化 | 全参数训练 | 真实标签＋三项蒸馏 | 42、43、44 |

共六次新训练。每个种子的 A、B 学生从完全相同的初始参数开始，使用相同的患者划分、图像视野、增强和评价规则。程序保存学生初始参数的 SHA256，B 组另保存蒸馏投影器初始参数的 SHA256。教师仍需要原始 RAD-DINO 基础权重及已训练的 LoRA/分类头检查点；学生无需下载或提供 ImageNet 权重。

蒸馏设置保持温度 2、局部关系池化 7×7：

```text
L = Focal + 0.5 × soft + 0.1 × global + 1.0 × local
```

第一轮沿用原全参数实验的训练配方：最多100轮、验证集 ROC-AUC 连续15轮未提高时早停、FP32、batch size 8、累积8步，主干学习率 `5e-5`，分类头和投影头学习率 `1e-4`。随机初始化可能需要不同的学习率或训练长度，需先检查收敛情况；进一步调参使用训练和验证数据，并匹配各方案的搜索预算。

本轮 A/B 是受控的随机学生监督与蒸馏比较。旧 ImageNet 实验可作历史参照，但其模型构造过程消耗随机数不同，蒸馏投影器初值也可能不同。若要把两批差异严格归因于学生初始化，还需同步核对或控制这些初始状态。

随机初始化不会减少学生参数数量，也不能保证解决过拟合。旧 ImageNet 初始化结果继续保留，不能将旧表改名后当作随机初始化成绩。

## 安装

将 `FU254-random-student-update.zip` 解压到 Windows 项目目录，得到：

```text
D:\99\EVA-X-main\random_student_update\update_random_student.py
```

通过 PyCharm 上传整个 `random_student_update` 文件夹到 `/home/cfu/EVA-X-main/`，再在服务器执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/random_student_update/update_random_student.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/random_student_update --apply
```

更新包含一个兼容旧检查点的学生模型工厂和六个新文件。安装器核对 SHA256、语法及既有模型工厂的版本，自动备份后原子更新；发现未知本地修改时保留文件并停止安装。旧训练入口、权重、数据和历史结果均保留。

## 后台启动

以下命令默认使用 GPU 0，六个任务依次执行。它先检查是否已有同类进程，再用 `nohup` 后台运行，日志查看与训练进程分离：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
mkdir -p runs/random_student_distillation

if pgrep -u "$(id -u)" -af '[r]un_random_student_(experiments|distillation)\.py'; then
    echo "发现随机学生实验进程，未重复启动。"
else
    random_student_log="runs/random_student_distillation/launch_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_random_student_experiments.py --devices cuda:0 --max-parallel 1 --cache-dir runs/foundation_sft/data_cache > "$random_student_log" 2>&1 < /dev/null &
    echo "后台进程 PID：$!；日志：$random_student_log"
    tail -n 30 -f "$random_student_log"
fi
```

按 `Ctrl+C` 退出日志查看，后台训练继续。入口自动执行前后向及检查点保存重载检查，然后开始训练；每轮会输出进度和真实标签、蒸馏损失。

如果多张 GPU 空闲，在启动前将上面命令的设备与并发参数改为，例如：

```text
--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 --max-parallel 5
```

每张卡同时最多运行一个独立任务，无需修改代码。只查看计划或只做短检查时使用：

```bash
python run_random_student_experiments.py --plan
python run_random_student_experiments.py --devices cuda:0 --max-parallel 1 --check-only
```

## 教师来源与结果

配置文件为 `foundation_sft/configs/random_student_distillation.json`。默认从指定的旧图像基础模型批次解析 RAD-DINO LoRA seed42 的 `best.pt`：

```text
runs/foundation_sft/suites/suite_20261013T024024Z_92ac596d
```

如果教师恢复训练后路径改变，可在配置中的 `teacher_checkpoint` 填入实际微调结果的 `best.pt`。程序验证教师、患者划分及检查点来源，使用验证集选择模型；不按测试成绩挑选教师或种子。

随机学生结果写入独立的 `runs/random_student_distillation/suites/suite_...`。报告明确记录随机初始化，保留逐种子结果、跨种子均值及样本标准差、配对差值。PR-AUC 和 AP 分开输出。

```bash
python run_random_student_experiments.py --show-metrics
```

只查看指定批次时，将下列路径末尾替换为启动时打印的真实批次目录：

```bash
python run_random_student_experiments.py --show-metrics --suite /home/cfu/EVA-X-main/runs/random_student_distillation/suites/实际批次目录
```

新批次不会复用任何历史训练结果。恢复同一批次时，已完成任务经过核验后保留，未完成任务重新开始；这不是从最后一个 epoch 续训。先用上述进程检查确认旧协调器已退出，再使用 `nohup` 和日志重定向运行下列恢复命令：

```bash
python -u run_random_student_experiments.py --resume-suite /home/cfu/EVA-X-main/runs/random_student_distillation/suites/实际批次目录 --devices cuda:0 --max-parallel 1 --cache-dir runs/foundation_sft/data_cache
```

继续队列时可更换 GPU 和并发数，初始化、种子及科学设置必须一致。

## 检查过拟合

每轮优化后，程序会用 `eval` 模式和无随机增强的训练集做一次额外的学生前向评估，保存 `train_eval_loss`、`train_eval_patient_roc_auc` 和 `train_eval_patient_pr_auc`。该评估使用独立的数据加载随机数生成器，不改变优化阶段的抽样或增强随机序列；每轮会增加一次训练集评估用时。

- `training_loss.png` 比较 `train_eval_loss` 与 `val_loss`，两者均为真实标签 Focal 损失；其计算口径记录在 `history_plot_metadata.json`。
- `history.json` 同时保留优化阶段的真实标签、蒸馏及总损失，以及无增强的训练集评估和验证集指标。查看训练集评估损失持续下降时，验证损失是否持续上升，以及训练与验证 AUC 差距是否扩大。
- 单个轮次的波动或验证/测试分数差距不足以诊断过拟合。蒸馏优化总损失含额外 KD 项，不能直接与仅含 Focal 的验证损失作差。
