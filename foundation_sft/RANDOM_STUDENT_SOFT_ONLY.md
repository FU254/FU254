# 随机学生：仅软标签蒸馏对照

本更新在原训练代码上增加显式的 `distillation_profile="soft_only"` 选项，沿用原数据、训练循环、优化器、进度日志、最佳验证检查点、保存重载和患者评估流程。缺省配置仍按原三项蒸馏规则运行。

本轮固定随机初始化 ViT-Tiny、全参数训练、seed42、RAD-DINO LoRA 教师、FP32、`weight_decay=0.01`、`DropPath=0`。学习率、增强、划分、学生初始参数、温度和软标签系数从已有对照核对；只关闭全局与局部蒸馏。

| 比较组 | 训练信号 | 本轮执行 |
|---|---|---|
| supervised | 真实标签 | 复用已完成的 A、seed42 |
| full_kd | 真实标签＋三项蒸馏 | 复用已完成的 B、seed42 |
| soft_only | 真实标签＋软标签蒸馏 | **只新增这1项训练** |

固定系数为 `temperature=2`、`soft_weight=0.5`、`global_weight=0`、`local_weight=0`。软标签组不需要全局特征投影器；原三项组的投影器保留在其历史检查点中。新组的学生初始参数仍必须与原组完全相同。

固定对照来源：

```text
/home/cfu/EVA-X-main/runs/random_student_weight_decay/studies/study_20261025T045150Z_b978d3c4
```

其中选用已完成的 `arms/wd_0p01/suite`。启动时会校验对照状态、配置、初始参数、教师指纹、数据划分和共享图像视图；对照文件或设置改变会直接报错。结果写入独立的 `runs/random_student_soft_only/studies/study_...`，原结果保留。

## 上传并安装

解压 `FU254-random-student-soft-only.zip`，将整个 `random_student_soft_only_update` 文件夹上传到 `/home/cfu/EVA-X-main/`，执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/random_student_soft_only_update/update_random_student_soft_only.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/random_student_soft_only_update --apply
```

安装器校验上传文件的 SHA256，只替换两个已知版本的原训练文件，新增本轮入口、配置和说明，自动备份。未知本地修改会保留，重复安装会识别已经更新的文件。本包不包含权重、患者数据或历史结果。

## 查看计划

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_random_student_soft_only.py --project /home/cfu/EVA-X-main --plan
```

计划只应包含1个随机学生 B、seed42 训练。正式启动会先做预检查，确认软标签分支反向传播、学生保存重载以及冻结教师，再进入训练。实际 CUDA 运行由服务器执行。

## 后台启动

下面整段自动选择剩余显存至少20GB、利用率不超过5%的 GPU，并防止重复启动相关实验：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
mkdir -p runs/random_student_soft_only

soft_only_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_(random_student_soft_only|random_student_distillation|random_student_experiments|weight_decay_experiments|droppath_experiments)\.py'; then
    echo "已有相关实验进程，未重复启动。"
elif [ -z "$soft_only_gpu" ]; then
    echo "当前没有符合条件的空闲 GPU，未启动。"
    nvidia-smi
else
    soft_only_log="runs/random_student_soft_only/run_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_random_student_soft_only.py \
        --devices "cuda:$soft_only_gpu" --max-parallel 1 \
        --cache-dir runs/foundation_sft/data_cache \
        > "$soft_only_log" 2>&1 < /dev/null &
    echo "GPU：$soft_only_gpu；PID：$!；日志：$soft_only_log"
    tail -n 30 -f "$soft_only_log"
fi
```

按 Ctrl+C 只退出日志查看，后台任务继续。父研究负责复用两项已完成对照，因此看到对照检查和结果打印不代表重新训练了 A 或原 B。

## SSH 断开后恢复

先确认相关进程是否还在：

```bash
pgrep -u "$(id -u)" -af '[r]un_(random_student_soft_only|random_student_distillation)\.py'
```

若进程已经退出，下面整段从本轮专用目录中选择唯一的研究并后台恢复。若目录中有多次独立研究，会打印路径并停止，避免混用不同批次：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main

soft_only_study=$(python - <<'PY'
import json
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/random_student_soft_only/studies')
studies = []
for path in sorted(root.glob('study_*/study.json')):
    if json.loads(path.read_text()).get('format_version') == 'random_student_soft_only_v1':
        studies.append(path.parent)
if len(studies) != 1:
    print('需要唯一研究，当前路径：' + ', '.join(map(str, studies)), file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
)
soft_only_gpu=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1; exit}')

if pgrep -u "$(id -u)" -af '[r]un_(random_student_soft_only|random_student_distillation|random_student_experiments|weight_decay_experiments|droppath_experiments)\.py'; then
    echo "已有相关实验进程，未重复启动。"
elif [ -z "$soft_only_study" ] || [ -z "$soft_only_gpu" ]; then
    echo "未确定唯一研究或空闲 GPU，未启动。"
else
    soft_only_log="runs/random_student_soft_only/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_random_student_soft_only.py \
        --resume-study "$soft_only_study" \
        --devices "cuda:$soft_only_gpu" --max-parallel 1 \
        --cache-dir runs/foundation_sft/data_cache \
        > "$soft_only_log" 2>&1 < /dev/null &
    echo "恢复：$soft_only_study；GPU：$soft_only_gpu；PID：$!；日志：$soft_only_log"
    tail -n 30 -f "$soft_only_log"
fi
```

恢复会核验已完成结果并保留；未完成训练会新建尝试，从同一初始学生重新训练。此入口的恢复指研究队列恢复，不声称从未完成轮次继续优化器状态。

## 查看结果和曲线

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_random_student_soft_only.py --project /home/cfu/EVA-X-main --show-metrics
```

每个研究分别打印监督、三项蒸馏、仅软标签三组结果，以及验证指标、患者测试指标和最佳/最终轮次的同口径差距。三组均完成后，运行下面命令生成对比曲线，无需重训：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_random_student_soft_only.py --project /home/cfu/EVA-X-main --plot-curves
```

曲线保存在该研究的 `report/curves_<唯一标识>/training_validation.png`，输出路径会打印到终端。图中三行依次是 supervised、full_kd、soft_only，左列是损失，右列是训练/验证ROC-AUC。曲线使用 `train_eval_loss`，保留原训练总损失和各项 KD 损失供追溯。

主要比较依据是 seed42 的验证患者 ROC-AUC，PR-AUC和完整曲线辅助解释。测试指标用于描述，不能用于挑选权重、设置或种子。单种子无法估计种子间方差；训练/验证差距减小且验证性能提高，才更支持泛化改善。
