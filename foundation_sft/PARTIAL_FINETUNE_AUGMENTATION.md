# Last8 共享亮度／对比度增强验证

本轮只新增一次训练：在已完成的 last8 配方上加入轻度亮度、对比度扰动，其他设置保持一致。复用来源固定为：

```text
/home/cfu/EVA-X-main/runs/partial_finetune_study/studies/study_20261027T150707Z_1ad30256
```

| 对照 | 初始化与更新范围 | 本轮处理 |
|---|---|---|
| Last8 existing | ImageNet DeiT-Tiny，最后8块、最终归一化层、分类头与三个投影器可训练 | 核验后复用已完成结果 |
| Last8 + shared intensity jitter | 同一 ImageNet 初始权重与 last8 范围 | 只新增1项 seed42 训练 |

新增候选从原始 ImageNet 权重重新开始，沿用原分类头／投影器初始化，不从旧 last8 的 `best.pt` 接着训练。沿用同一患者划分、三个冻结教师、等权三项蒸馏、Focal 标签损失及软标签／全局／局部系数 `0.5/0.1/1.0`，沿用主干学习率 `5e-5`、头／投影器 `1e-4`、WD `0.01`、DropPath `0`、batch、梯度累积、FP32、最多100轮和 patience15。

`brightness_jitter=0.1`、`contrast_jitter=0.1` 是振幅，分别产生独立均匀采样的 `[0.9, 1.1]` 系数。仅训练时，在既有共享几何增强后，对同一灰度图先围绕图像均值调整对比度，再乘亮度系数，最后截断至 `[0, 1]`；之后才分别生成学生和三位教师所需的尺寸与标准化输入。四个分支共享本次增强视野。训练集 eval、验证和历史测试不加随机强度扰动；旧配置未声明这两个字段时保持原数据流程与随机数消耗。

本轮按最佳验证患者 ROC-AUC 比较；精确相同时保留已有 last8。新候选只使用训练集和验证集，**即使验证胜出也不追加任何测试评估**。报告中的既有 last8 测试指标仅作历史描述，测试分数不参与筛选。只看不同最佳轮次的 train-val gap 变小不足以说明过拟合改善；需同时看验证 ROC/PR、同口径损失、完整曲线与匹配轮次。seed42 单次筛选不能估计随机种子间 SD，也不能宣称稳定或显著改善。

## 上传、安装与计划

解压 `FU254-partial-augmentation.zip`，把整个 `partial_augmentation_update` 文件夹上传到 `/home/cfu/EVA-X-main/`，保持文件夹结构。使用现有环境中的绝对 Python 路径执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/partial_augmentation_update/update_partial_augmentation.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/partial_augmentation_update --apply
```

本增量包只替换已发布版本的 `data.py`、`distillation.py`、`teacher_weight_training.py`、`partial_finetune_study.py`、`partial_finetune_report.py` 五个文件，新增本说明和增强配置。安装器直接复用包内原版安装引擎，不重新实现安装流程。先校验全部来源 SHA256 和语法，再备份并原子发布；未知本地改动会保留并停止安装。独立备份目录为 `.partial-augmentation-backups`，相同版本重复安装无须重复写入。安装不改原 12/8/4/2 研究、旧入口、模型定义、旧配置、缓存、教师权重、已恢复 P-R 结果或并行调度代码。

继续使用原入口，**每条本轮命令都显式传增强配置**。默认不传 `--config` 的原计划仍属于原 12/8/4/2 研究。

```bash
cd /home/cfu/EVA-X-main
/opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --config foundation_sft/configs/partial_finetune_augmentation.json --plan
```

计划应显示已完成 last8 对照及唯一新增增强候选，不应创建新的12/4/2训练。正式运行的预检查会核验来源配方、已完成结果、冻结范围、教师不变、损失梯度与检查点恢复。原 `--check-only` 参数也可配合本增强配置使用；它会建立预检查研究目录，之后应恢复该准确目录继续训练。

## 选一张空闲卡并后台启动

整段复制执行，只选一张空闲显存至少20GB、利用率不超过5%的GPU，`max_parallel=1`。重复检测同时覆盖原入口和指向本轮研究目录的模块 worker；检测到进程时不再次启动。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES
mkdir -p /home/cfu/EVA-X-main/runs/partial_finetune_augmentation

partial_augmentation_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_partial_finetune_experiments\.py|foundation_sft[.]teacher_weight_training.*runs/partial_finetune_augmentation'; then
    echo "已有局部微调研究或本轮worker，未重复启动。"
elif [ -z "$partial_augmentation_gpu" ]; then
    echo "当前没有符合条件的空闲GPU，未启动。"
    nvidia-smi
else
    partial_augmentation_log="/home/cfu/EVA-X-main/runs/partial_finetune_augmentation/start_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_partial_finetune_experiments.py \
        --project /home/cfu/EVA-X-main \
        --config foundation_sft/configs/partial_finetune_augmentation.json \
        --devices "cuda:$partial_augmentation_gpu" \
        --max-parallel 1 \
        --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache \
        > "$partial_augmentation_log" 2>&1 < /dev/null &
    echo "GPU：cuda:$partial_augmentation_gpu；PID：$!；日志：$partial_augmentation_log"
    tail -n 30 -f "$partial_augmentation_log"
fi
```

`Ctrl+C` 只退出 `tail` 日志查看，后台任务继续。记录日志中打印的 `Study directory` 准确路径。

## 查看指标与曲线

下面代码只在增强专用目录内恰有一个研究时取得其准确路径，并显式通过 `--study` 报告。有多个研究时会打印所有目录并停止，不按时间猜测；此时应使用日志中记录的准确路径。

```bash
cd /home/cfu/EVA-X-main
if partial_augmentation_study=$(/opt/anaconda3/envs/evax_lora/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/partial_finetune_augmentation/studies')
studies = sorted(path.parent for path in root.glob('study_*/study.json'))
if len(studies) != 1:
    print('需要唯一增强研究；请从启动日志确定准确 --study 路径。当前目录：', file=sys.stderr)
    for path in studies:
        print(path, file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
); then
    /opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --config foundation_sft/configs/partial_finetune_augmentation.json --study "$partial_augmentation_study" --show-metrics
    /opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --config foundation_sft/configs/partial_finetune_augmentation.json --study "$partial_augmentation_study" --plot-curves
fi
```

报告有两组：`Last8 existing` 和 `Last8 + shared intensity jitter`。报告目录保存 `screening.csv`、`test_comparison.csv`、`overfitting.csv`、`matched_epochs.csv` 和 `selection.json`。候选没有测试分数，不能把历史对照测试成绩与未测候选组成测试优劣结论。两组曲线包含训练 eval／验证的同口径 Focal loss、患者 ROC-AUC、PR-AUC，并标记验证最优轮次；优化总损失单独绘制，不能直接与验证 Focal loss 相减。

图片为 `loss_curves.png`、`roc_auc_curves.png`、`pr_auc_curves.png` 和 `optimization_losses.png`。命令打印实际生成目录，重复绘图保留旧图片。

## 中断后恢复

确认原任务已退出后，复制下方整段。它同样要求增强专用目录中只有一个研究，并选择一张空闲卡。已有完成结果会复用，失败尝试保留；未完成候选可能重新从同一 ImageNet 初始状态训练，队列恢复不保证从中断轮次继续优化器。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES

partial_augmentation_study=$(/opt/anaconda3/envs/evax_lora/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/partial_finetune_augmentation/studies')
studies = sorted(path.parent for path in root.glob('study_*/study.json'))
if len(studies) != 1:
    print('需要唯一增强研究；请从启动日志确定准确 --resume-study 路径。当前目录：', file=sys.stderr)
    for path in studies:
        print(path, file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
)
partial_augmentation_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_partial_finetune_experiments\.py|foundation_sft[.]teacher_weight_training.*runs/partial_finetune_augmentation'; then
    echo "已有局部微调研究或本轮worker，未重复启动。"
elif [ -z "$partial_augmentation_study" ] || [ -z "$partial_augmentation_gpu" ]; then
    echo "未确定唯一研究或空闲GPU，未启动。"
else
    partial_augmentation_log="/home/cfu/EVA-X-main/runs/partial_finetune_augmentation/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_partial_finetune_experiments.py \
        --project /home/cfu/EVA-X-main \
        --config foundation_sft/configs/partial_finetune_augmentation.json \
        --resume-study "$partial_augmentation_study" \
        --devices "cuda:$partial_augmentation_gpu" \
        --max-parallel 1 \
        --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache \
        > "$partial_augmentation_log" 2>&1 < /dev/null &
    echo "恢复：$partial_augmentation_study；GPU：cuda:$partial_augmentation_gpu；PID：$!；日志：$partial_augmentation_log"
    tail -n 30 -f "$partial_augmentation_log"
fi
```

有多个研究时，不要重跑启动段创建另一批。把已记录的准确研究路径赋给 `partial_augmentation_study`，再执行对应命令中的 `--study "$partial_augmentation_study"` 或 `--resume-study "$partial_augmentation_study"`。更换GPU不会更改科学配方，也不会重做已完成候选。

如需撤销代码安装，使用安装输出的准确 `backup` 目录配合安装器 `--rollback` 先预览，再加 `--apply`。回滚会保留安装后产生的未知本地代码修改并停止；原研究与训练产物始终留在原目录。
