# ImageNet 学生局部解冻筛选

本轮先固定三个教师、等权三项蒸馏和 seed42，只检验学生可以更新的 Transformer 块数。已有 P-E 全参数训练结果来自：

```text
/home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f
```

| 方案 | 学生可更新部分 | 本轮训练 |
|---|---|---|
| 已有 P-E | 全部12个块、最终归一化层和分类头 | 严格核验后复用 |
| 最后8块 | 最后8个块、最终归一化层和分类头 | 新增1项 |
| 最后4块 | 最后4个块、最终归一化层和分类头 | 新增1项 |
| 最后2块 | 最后2个块、最终归一化层和分类头 | 新增1项 |

三项新训练都从同一份官方 ImageNet DeiT-Tiny 初始权重重新开始，分类头和三位教师的全局特征投影器使用与 P-E 相同的初始化。不能从已训练的 P-E `best.pt` 继续训练。图像嵌入层、位置参数、CLS参数与前4/8/10个块分别冻结；三个投影器保持可训练，最后几层的空间 token 仍能接受局部关系蒸馏梯度。四种方案的学生可训练参数分别为5,524,609、3,559,489、1,780,033、890,305；推理模型总参数量相同。

主干学习率固定为 `5e-5`，分类头／投影器为 `1e-4`，WD=`0.01`、DropPath=`0`。沿用原患者划分、三个冻结教师、共享几何增强、Focal真实标签损失和软标签／全局／局部蒸馏系数 `0.5/0.1/1.0`。沿用同一batch、累积、FP32、最多100轮及patience15。这是冻结范围的单因素筛选，不自动运行随机初始化、CA、RL或额外种子。

三项新候选先只使用训练集和验证集，全部完成后才按最佳验证患者 ROC-AUC 与已有 P-E 比较；精确相同优先选择学生可训练参数更少的方案。原 P-E 的历史测试指标保留为描述性对照，不参与筛选。仅当新候选胜出时，才将其已保存的验证最优检查点评估一次测试集；未选中候选不计算测试指标。如果 P-E 胜出，不追加新的测试评估。

差距变小同时验证性能下降，不能称为解决过拟合。主表同时报告验证ROC/PR、最佳与最终轮次的同口径损失/AUC差距；测试部分明确区分已有对照与选中候选，保留 ROC-AUC、PR-AUC、AP、Sensitivity、Specificity。

## 上传并安装

解压 `FU254-partial-finetune-study.zip`，将整个 `partial_finetune_update` 文件夹上传到 `/home/cfu/EVA-X-main/`。执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/partial_finetune_update/update_partial_finetune_study.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/partial_finetune_update --apply
```

安装器只替换已发布版本的学生模型、检查点恢复与教师权重训练三个代码文件，新增筛选入口、配置、报告和本说明。所有来源文件先校验SHA256并编译，之后备份、原子发布；新入口最后发布。未知本地修改会保留并停止安装。它不修改原研究、权重、缓存、已恢复的 P-R 结果或并行调度入口，也不停止其他GPU任务。相同版本重复安装会识别已有内容。

先打印计划核对原 P-E 来源及三个新候选，计划模式不训练：

```bash
cd /home/cfu/EVA-X-main
/opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --plan
```

## 选择空闲卡并后台启动

下面整段最多选择三张空闲显存至少20GB、利用率不超过5%的GPU。每张卡运行一个独立候选，不改变单项训练batch。所有训练都从项目目录启动，保证历史教师配置里的相对权重路径正确解析。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES
mkdir -p runs/partial_finetune_study

mapfile -t partial_finetune_gpu_indices < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1}' | head -n 3)
partial_finetune_devices=()
for partial_finetune_index in "${partial_finetune_gpu_indices[@]}"; do
    partial_finetune_devices+=("cuda:$partial_finetune_index")
done

if pgrep -u "$(id -u)" -af '[r]un_partial_finetune_experiments\.py|foundation_sft[.]teacher_weight_training.*runs/partial_finetune_study'; then
    echo "已有本轮筛选进程，未重复启动。"
elif [ "${#partial_finetune_devices[@]}" -eq 0 ]; then
    echo "当前没有符合条件的空闲GPU，未启动。"
    nvidia-smi
else
    partial_finetune_log="runs/partial_finetune_study/start_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_partial_finetune_experiments.py \
        --project /home/cfu/EVA-X-main \
        --devices "${partial_finetune_devices[@]}" \
        --max-parallel "${#partial_finetune_devices[@]}" \
        --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache \
        > "$partial_finetune_log" 2>&1 < /dev/null &
    echo "GPU：${partial_finetune_devices[*]}；PID：$!；日志：$partial_finetune_log"
    tail -n 30 -f "$partial_finetune_log"
fi
```

按 `Ctrl+C` 只退出日志查看，后台任务继续。记录打印的 `Study directory`，它是本批筛选的准确路径。已有六组实验留在原研究目录，不会被这个新入口覆盖。

如果原 P-E 来源位于其他准确目录，启动时通过 `--source-study` 指定；不要把另一批次的测试最高分结果当作本轮来源。默认来源已经固定到上文路径。

## 查看结果与曲线

```bash
cd /home/cfu/EVA-X-main
/opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --show-metrics
```

```bash
cd /home/cfu/EVA-X-main
/opt/anaconda3/envs/evax_lora/bin/python run_partial_finetune_experiments.py --project /home/cfu/EVA-X-main --plot-curves
```

有多次筛选时，以上报告命令可加 `--study` 和终端打印的准确研究路径，各批分别展示。曲线包括同口径的训练eval／验证Focal loss、患者ROC-AUC和PR-AUC，并标记验证最优轮次；含蒸馏的优化损失单独展示。不要把优化总损失直接与验证Focal比较。

报告目录保存 `screening.csv`、`test_comparison.csv`、`overfitting.csv`、`matched_epochs.csv` 和 `selection.json`。四方案的曲线为 `loss_curves.png`、`roc_auc_curves.png`、`pr_auc_curves.png`，另有 `optimization_losses.png`；命令会打印实际生成目录，重复绘图保留旧图片。

## 中断后恢复

先确认原进程已退出，避免新建第二批。下面整段仅在专用目录中存在唯一研究时恢复；若存在多个研究，会打印路径并停止，需使用先前记录的准确 `--resume-study` 路径。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES

partial_finetune_study=$(/opt/anaconda3/envs/evax_lora/bin/python - <<'PY'
from pathlib import Path
import sys
studies = sorted(path.parent for path in Path('/home/cfu/EVA-X-main/runs/partial_finetune_study/studies').glob('study_*/study.json'))
if len(studies) != 1:
    print('需要唯一筛选研究，当前路径：' + ', '.join(map(str, studies)), file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
)
mapfile -t partial_finetune_gpu_indices < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1}' | head -n 3)
partial_finetune_devices=()
for partial_finetune_index in "${partial_finetune_gpu_indices[@]}"; do
    partial_finetune_devices+=("cuda:$partial_finetune_index")
done

if pgrep -u "$(id -u)" -af '[r]un_partial_finetune_experiments\.py|foundation_sft[.]teacher_weight_training.*runs/partial_finetune_study'; then
    echo "已有本轮筛选进程，未重复启动。"
elif [ -z "$partial_finetune_study" ] || [ "${#partial_finetune_devices[@]}" -eq 0 ]; then
    echo "未确定唯一研究或空闲GPU，未启动。"
else
    partial_finetune_log="runs/partial_finetune_study/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_partial_finetune_experiments.py \
        --project /home/cfu/EVA-X-main \
        --resume-study "$partial_finetune_study" \
        --devices "${partial_finetune_devices[@]}" \
        --max-parallel "${#partial_finetune_devices[@]}" \
        --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache \
        > "$partial_finetune_log" 2>&1 < /dev/null &
    echo "恢复：$partial_finetune_study；PID：$!；日志：$partial_finetune_log"
    tail -n 30 -f "$partial_finetune_log"
fi
```

恢复会核验冻结配方和已完成产物，保留失败尝试。未完成的候选可能创建新尝试从同一ImageNet初始状态重新训练；队列恢复不等于保证从中断轮次恢复优化器。已完成候选和已完成的选中检查点评估不会因为换GPU而自动重做。

本地CPU训练、梯度和模拟调度验证不能替代服务器实际GPU检查。执行前的预检查会核验学生冻结范围、教师不变、分类与特征梯度以及检查点恢复。单个seed42只用于筛选，不能估计随机种子间SD或宣称稳定、显著的改善。
