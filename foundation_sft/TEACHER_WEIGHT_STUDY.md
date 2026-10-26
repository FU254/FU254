# 两种学生初始化 × 三种教师权重策略

本更新基于现有教师加载、共享图像视图、ViT-Tiny学生、损失、患者评估和检查点代码增加研究入口。新文件独立保存，已有训练入口、配置和历史结果保留。

本轮比较的是**各初始化分别选择学习率后的训练配方**。相同初始化内只改变教师权重策略；随机初始化与ImageNet初始化之间的差异不能全部归因于初始化本身，因为其最终主干学习率可能不同。

| 正式组 | 学生初始化 | 教师权重 |
|---|---|---|
| R-E | 随机 | 等权 |
| R-C | 随机 | CA |
| R-R | 随机 | RL |
| P-E | ImageNet | 等权 |
| P-C | ImageNet | CA |
| P-R | ImageNet | RL |

六组固定seed42、全参数ViT-Tiny、RAD-DINO/CheXFound/CheXzero三个冻结教师、原患者划分和共享几何增强。沿用真实标签Focal loss及三项蒸馏：温度2，软标签0.5、全局0.1、局部1.0；WD=0.01、DropPath=0、FP32、batch8、累积8、最多100轮、patience15。分类头和各教师特征投影器学习率固定为1e-4。本轮不自动增加监督或单教师训练。

教师来源与学生ImageNet文件沿用已有项目配置。ImageNet学生文件应位于：

```text
foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth
```

## 先选学习率，再比较权重

| 初始化 | 等权预实验的主干学习率候选 |
|---|---|
| 随机 | 5e-5、1e-4、3e-4 |
| ImageNet | 1e-5、3e-5、5e-5 |

流程固定为：

1. 每种初始化先训练3个等权候选，共6项。只访问训练与验证数据，不计算这些候选的测试指标。
2. 按每种初始化的最佳验证患者ROC-AUC选择主干学习率；精确相同时优先较小学习率。未完成候选不会被默默跳过。
3. 选中的等权检查点直接作为正式E组，不重新训练，只在选择冻结后评估一次测试集。未选中的候选不做测试评估。
4. 每种初始化在自己的选定学习率下，各训练CA和RL一项，共4项。

因此最多有**10次优化训练：6个预实验＋4个CA/RL任务**，最终主表包含6组。对照在预实验中完成并复用，不能把“正式6组”误读成另加6次训练。

每项先做GPU预检查，再训练。教师与学生来源、患者划分、共享图像视图、初始化、系数和有效训练设置均记录在配置、元数据和回执中。相同初始化的三种策略使用相同学生与投影器初始张量；两种初始化的分类头使用相同随机初始化规则。RL使用独立随机流，不改变学生初始化、数据顺序或图像增强。

## 方法名称与适配边界

三个教师权重非负且和为1，真实标签损失只加一次。每位教师单独计算软标签、全局和局部蒸馏，再用该样本的教师权重加权；不同教师的原始特征向量不会直接平均。

- 等权始终为1/3。
- CA使用教师对训练真实标签的交叉熵误差，按`(1-softmax(error))/(K-1)`得到权重；这里K=3。它不是预测熵加权或`softmax(-error)`。
- RL使用轻量Dirichlet策略与REINFORCE更新。16维状态包含各教师预测、真实标签误差、三项师生差异及轮次进度，均停止梯度；采样得到的一套三教师权重用于三项蒸馏。策略网络为32维隐藏层，独立Adam学习率1e-4，浓度缩放6、浓度下限0.5、熵系数0.001、奖励统计EMA衰减0.95。采样随机流与学生和数据增强隔离，并随检查点保存。

所有方法第一轮均使用归一化等权。RL后续奖励为每个训练样本的负真实标签Focal与加权KD总损失，策略和学生分开反向传播与更新。这是当期训练代理，可能偏向更容易匹配的教师，不能称为更新后的泛化收益；权重和验证曲线用于检验这一限制。正式验证集不用于策略更新，测试集不参与奖励、学习率选择或教师权重训练。

本轮是**CA权重策略与MTKD-RL思想在现有三项蒸馏框架上的任务适配**，不是两篇论文全流程复现。策略网络仅在训练时使用，最终推理只需要学生检查点。

## 上传并安装

解压`FU254-teacher-weight-study.zip`，将整个`teacher_weight_update`文件夹上传到`/home/cfu/EVA-X-main/`。在服务器执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/teacher_weight_update/update_teacher_weight_study.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/teacher_weight_update --apply
```

安装器校验清单、备份和原子安装流程与原项目更新方式一致。本包只新增代码、研究配置和本说明；已存在的未知同名文件会保留并报错，防止覆盖本地修改。重复安装相同版本会识别已安装内容。本包不含患者数据、权重或历史结果。

先打印研究计划，核对两个初始化、六个学习率候选及最终方法：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py --project /home/cfu/EVA-X-main --plan
```

## 自动选择空闲GPU并后台运行

下面整段使用当前空闲显存至少20GB、利用率不超过5%的GPU，并行调度独立任务。每张卡同时最多运行本研究的一个任务。GPU数量影响任务并行度，不会改变单项训练的batch或数据划分。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
mkdir -p runs/teacher_weight_study

mapfile -t teacher_weight_gpu_indices < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1}')
teacher_weight_devices=()
for teacher_weight_index in "${teacher_weight_gpu_indices[@]}"; do
    teacher_weight_devices+=("cuda:$teacher_weight_index")
done

if pgrep -u "$(id -u)" -af '[r]un_teacher_weight_(experiments|training)\.py'; then
    echo "已有本研究进程，未重复启动。"
elif [ "${#teacher_weight_devices[@]}" -eq 0 ]; then
    echo "当前没有符合条件的空闲GPU，未启动。"
    nvidia-smi
else
    teacher_weight_log="runs/teacher_weight_study/run_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_teacher_weight_experiments.py \
        --devices "${teacher_weight_devices[@]}" \
        --max-parallel "${#teacher_weight_devices[@]}" \
        --cache-dir runs/foundation_sft/data_cache \
        > "$teacher_weight_log" 2>&1 < /dev/null &
    echo "GPU：${teacher_weight_devices[*]}；PID：$!；日志：$teacher_weight_log"
    tail -n 30 -f "$teacher_weight_log"
fi
```

按Ctrl+C只退出日志查看，后台任务继续。记录终端打印的`Study directory`，它是后续恢复和查看本批次结果的准确路径。

## 断线或中断后恢复

先检查进程是否还在，避免重复启动：

```bash
pgrep -u "$(id -u)" -af '[r]un_teacher_weight_(experiments|training)\.py'
```

如果任务确实已经退出，下面整段会从本研究专用目录中选择唯一研究，并恢复队列。存在多次独立研究时会打印路径并停止；此时应使用原终端记录的准确`--resume-study`路径，避免混合批次。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main

teacher_weight_study=$(python - <<'PY'
import json
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/teacher_weight_study/studies')
studies = [path.parent for path in sorted(root.glob('study_*/study.json'))
           if json.loads(path.read_text()).get('format_version') == 'teacher_weight_study_v1']
if len(studies) != 1:
    print('需要唯一研究，当前路径：' + ', '.join(map(str, studies)), file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
)
mapfile -t teacher_weight_gpu_indices < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2>=20000 && $3<=5 {gsub(/[[:space:]]/,"",$1); print $1}')
teacher_weight_devices=()
for teacher_weight_index in "${teacher_weight_gpu_indices[@]}"; do
    teacher_weight_devices+=("cuda:$teacher_weight_index")
done

if pgrep -u "$(id -u)" -af '[r]un_teacher_weight_(experiments|training)\.py'; then
    echo "已有本研究进程，未重复启动。"
elif [ -z "$teacher_weight_study" ] || [ "${#teacher_weight_devices[@]}" -eq 0 ]; then
    echo "未确定唯一研究或空闲GPU，未启动。"
else
    teacher_weight_log="runs/teacher_weight_study/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_teacher_weight_experiments.py \
        --resume-study "$teacher_weight_study" \
        --devices "${teacher_weight_devices[@]}" \
        --max-parallel "${#teacher_weight_devices[@]}" \
        --cache-dir runs/foundation_sft/data_cache \
        > "$teacher_weight_log" 2>&1 < /dev/null &
    echo "恢复：$teacher_weight_study；GPU：${teacher_weight_devices[*]}；PID：$!；日志：$teacher_weight_log"
    tail -n 30 -f "$teacher_weight_log"
fi
```

恢复会核验已完成预实验、冻结的学习率和正式结果，保留失败尝试并继续未完成任务。未完成训练可能从相同初始状态创建新尝试；不要把队列恢复理解为必然从中断轮次恢复优化器。已完成结果不会因为重新调度GPU而自动重训。

## 查看结果与曲线

查看预实验状态、每种初始化的验证选择，以及正式六组结果：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py --project /home/cfu/EVA-X-main --show-metrics
```

读取已保存历史生成曲线，无需重训：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py --project /home/cfu/EVA-X-main --plot-curves
```

有多次独立研究时，可在上述任一完整命令末尾增加`--study`与该研究的准确绝对路径；各研究分别报告，不会按测试成绩挑选批次。

主要损失图、ROC-AUC图和PR-AUC图使用2行×3列：行是随机/ImageNet，列是等权/CA/RL，并标记最佳验证ROC轮次。`train_eval_loss`与`val_loss`只含同口径的真实标签Focal loss，均在eval模式、无随机增强下计算；包含蒸馏的优化总损失另存，不能拿它与验证Focal loss直接比较。另记录各教师权重、权重熵以及RL奖励与策略损失。

主要选择指标始终是患者验证ROC-AUC。患者测试ROC-AUC、PR-AUC、AP、敏感度、特异度和区间用于描述；PR-AUC与AP分开报告。训练/验证差距缩小且验证表现提高或保持，更支持泛化改善；两者同时下降不能据此说解决了过拟合。最佳与最终轮次分别报告。本轮只有seed42，不估计种子间SD，也不宣称稳定或显著优越。

本地CPU及模拟测试不能替代服务器实际GPU预检查。最终检查点只部署学生，不需要教师、投影器或策略网络参与推理。
