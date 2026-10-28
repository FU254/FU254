# 单教师 RAD-DINO → ViT-Tiny：过拟合诊断

本轮新增四项 seed42 训练，沿用原单教师实验的完整有效配方。来源固定为：

```text
/home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96
```

| 组别 | 初始化及更新范围 | 训练目标 |
|---|---|---|
| A supervised | 同一 ImageNet ViT-Tiny，全参数微调 | 标签监督 |
| B full KD | 同一 ImageNet ViT-Tiny，全参数微调 | 标签监督 + 原软预测／全局／局部蒸馏 |
| C soft only | 同一 ImageNet ViT-Tiny，全参数微调 | 标签监督 + 原软预测蒸馏；全局／局部系数为0 |
| D full KD + L2-SP | 同一 ImageNet ViT-Tiny，全参数微调 | B的目标 + ImageNet主干锚定 |

四组均重新从原始 ImageNet 权重开始，不载入旧学生的 `best.pt`。程序核对配对的学生初始化，使用同一患者划分、冻结 RAD-DINO 教师、seed42、标签损失、batch、梯度累积、精度、学习率、权重衰减、增强、最大轮数和早停设置。具体数值从来源批次保存的有效配置读取，不能由当前默认配置推测或用其他研究的配方替代。来源不存在、配置不完整、配方不匹配或权重校验失败时明确停止；`--source-suite` 只用于明确指定这个原始来源的实际位置。

D额外加入 `0.0001 * sum((student_backbone - initial_ImageNet_backbone)^2)`。这是所有主干参数差的原始平方和，包含最终归一化层，不除以参数数量；排除分类头、投影器和教师。锚点来自本次重新初始化的 ImageNet 主干，在整个训练期间固定。此项与优化器的原权重衰减分别计算，其他设置仍取来源配方。单独保存原始锚定损失、加权锚定损失和锚点元数据。`0.0001` 是本轮固定的探索性系数，未调优，不能据此称其为最优系数。

本轮仅对训练／验证影像执行前向与评估，**不运行任何新测试评估**。来源核验和患者划分准备会读取队列元数据，其中可以包含测试行，但不读取测试影像像素、不对测试集前向或生成测试指标。每轮另用无随机增强的 `train_eval` 计算与验证集同口径的标签损失、患者 ROC-AUC 和 PR-AUC；优化总损失及其各项另存，不能拿优化总损失直接减验证标签损失。模型选择、早停和分类阈值均只依据验证集，报告的敏感度／特异度也属于验证集。

四组自己的最佳轮次、最后轮次、最佳验证 ROC/PR、匹配轮次差异和完整曲线需要一起看。不同最佳轮次的 train-val gap 变小不能单独证明过拟合改善；若训练和验证同时下降，也可能是欠拟合。seed42 单次诊断不能估计跨种子 SD 或证明统计显著性。

## 安装增量包

下载 `FU254-single-teacher-overfit.zip`，解压后把整个 `single_teacher_overfit_update` 文件夹上传到 `/home/cfu/EVA-X-main/`。例如 Windows 本地解压到 `D:\99\EVA-X-main`，通过 PyCharm 上传该完整文件夹，服务器应出现：

```text
/home/cfu/EVA-X-main/single_teacher_overfit_update/update_single_teacher_overfit.py
```

使用项目现有环境的绝对 Python 路径，整段执行预览和安装：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/single_teacher_overfit_update/update_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --source-dir /home/cfu/EVA-X-main/single_teacher_overfit_update
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/single_teacher_overfit_update/update_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --source-dir /home/cfu/EVA-X-main/single_teacher_overfit_update --apply
```

本包适用于已安装当前研究代码的项目（仓库基线 `950c7c65ca55764b51262a2e4de789200c8cc754`）。只替换已知 SHA256 的 `foundation_sft/distillation.py`，新增本轮协调器、诊断模块、报告、配置、说明和入口。原单教师入口、模型定义、划分、旧配置、既有结果、权重及缓存保留。安装器直接复用包内未经修改的 `update_partial_finetune_study.py` 原子安装引擎，不安装依赖、不启动训练、不终止进程。先检查全部源文件 SHA256 和语法；遇到未知本地代码修改会整批拒绝并保留修改。相同内容重复安装不重复写入，备份在 `.single-teacher-overfit-backups/`。

## 查看计划与GPU

先查看来源和四组计划，此命令不需要GPU，不创建训练任务：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
/opt/anaconda3/envs/evax_lora/bin/python run_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --source-suite /home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96 \
    --plan
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

根据实际查询结果选择GPU。显式指定时使用如 `--devices cuda:2 cuda:5 --max-parallel 2`；编号只是格式示例，不能直接假定2、5空闲。多张卡按一张卡一个worker运行，四组超过可用卡数量时排队。配置默认单卡、并行1；下方启动段会根据实时空闲情况选择最多4张卡。

`--check-only` 对四个配置做前向、反向和保存／重载检查，不进行训练轮次。可在选择好空闲卡后执行，例如把 `cuda:2` 换成刚查询确认的实际空闲卡：

```bash
/opt/anaconda3/envs/evax_lora/bin/python run_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --source-suite /home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96 \
    --devices cuda:2 --max-parallel 1 \
    --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache --check-only
```

检查输出会打印准确研究目录；记录该路径，后续通过 `--resume-study` 在同一研究中继续。不要把本轮协调器的 `--check-only` 改成原单教师入口的检查命令。

## 自动选择空闲卡并后台启动

此段创建一个新的四组研究。它只挑选利用率不超过5%、已用显存不超过500MiB且空闲显存至少20000MiB的GPU，最多4张；有1至3张时自动排队，没有符合条件的卡则停止并打印查询结果。启动前检测本轮协调器和指向本轮目录的worker，避免重复启动；不会终止其他任务。若已有检查研究或中断研究，使用后文恢复段。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES
mkdir -p /home/cfu/EVA-X-main/runs/single_teacher_overfit

mapfile -t single_overfit_devices < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2<=500 && $3>=20000 && $4<=5 {gsub(/[[:space:]]/,"",$1); print "cuda:" $1; if (++count==4) exit}')

if pgrep -u "$(id -u)" -af '[r]un_single_teacher_overfit[.]py|foundation_sft[.]single_teacher_overfit|[r]un_single_teacher_distillation[.]py.*runs/single_teacher_overfit|foundation_sft[.]distillation.*runs/single_teacher_overfit'; then
    echo "已有本轮协调器或worker，未重复启动。"
elif [ "${#single_overfit_devices[@]}" -eq 0 ]; then
    echo "当前没有符合条件的空闲GPU，未启动。"
    nvidia-smi
else
    single_overfit_log="/home/cfu/EVA-X-main/runs/single_teacher_overfit/start_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_single_teacher_overfit.py \
        --project /home/cfu/EVA-X-main \
        --source-suite /home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/suite_20261019T110343Z_3b1acd96 \
        --devices "${single_overfit_devices[@]}" \
        --max-parallel "${#single_overfit_devices[@]}" \
        --cache-dir /home/cfu/EVA-X-main/runs/foundation_sft/data_cache \
        > "$single_overfit_log" 2>&1 < /dev/null &
    echo "GPU：${single_overfit_devices[*]}；PID：$!；日志：$single_overfit_log"
    tail -n 40 -f "$single_overfit_log"
fi
```

`Ctrl+C` 只退出 `tail`，后台任务继续。记录启动日志打印的准确 `Study directory`，其格式为 `runs/single_teacher_overfit/studies/study_时间_标识/`。该目录保存来源解析后的完整配置、来源文件哈希、四个任务配置、初始化核验和各组独立产物。旧原始单教师批次只读，四组均为新训练。

## 查看指标与绘图

下面整段仅在本轮目录中恰有一个研究时读取其准确路径；有多个研究时打印目录并停止，需从启动日志确定实际路径，不按最新时间猜测：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
if single_overfit_study=$(/opt/anaconda3/envs/evax_lora/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/single_teacher_overfit/studies')
studies = sorted(path.parent for path in root.glob('study_*/study.json'))
if len(studies) != 1:
    print('请从启动日志确定准确 --study 路径。当前研究：', file=sys.stderr)
    for path in studies:
        print(path, file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
); then
    /opt/anaconda3/envs/evax_lora/bin/python run_single_teacher_overfit.py \
        --project /home/cfu/EVA-X-main --study "$single_overfit_study" --show-metrics
    /opt/anaconda3/envs/evax_lora/bin/python run_single_teacher_overfit.py \
        --project /home/cfu/EVA-X-main --study "$single_overfit_study" --plot-curves
fi
```

`--show-metrics` 和 `--plot-curves` 只读取本轮保存的文件，不加载模型或重新扫描影像。历史记录缺失的指标保持未知，不补造训练曲线或测试成绩。绘图命令打印实际输出路径，使用新报告目录保留已有图片。

表格保存在本轮 `report/comparison.csv`、`report/matched_epoch.csv` 和 `report/report.md`。控制台标明验证 ROC-AUC、PR-AUC、AP、敏感度、特异度，以及最佳／最后轮次的 Focal 和 AUC 差距、最后相对最佳验证 ROC 的变化。曲线保存在独立的 `report/curves_*/`，包括 `focal_loss.png`、`roc_auc.png`、`pr_auc.png`、`optimization.png` 和D组的 `l2_sp.png`；前四张图每组各一个子图，L2-SP原始平方和与加权项分轴显示。

## 中断后继续同一研究

先确认本轮旧协调器和worker已退出，再复制下段。完成组核验本地结果后直接复用，不重新训练；未完成组保留旧尝试，从同一 ImageNet 初始化重新开始。这里恢复的是研究队列，**不是从中断轮次恢复优化器**。`--resume-study` 读取研究内保存的来源、缓存和配方，无需重新依赖当前默认来源路径；只允许更换调度GPU和并发数，不传 `--source-suite` 或 `--cache-dir` 覆盖研究设置。

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES

single_overfit_study=$(/opt/anaconda3/envs/evax_lora/bin/python - <<'PY'
from pathlib import Path
import sys
root = Path('/home/cfu/EVA-X-main/runs/single_teacher_overfit/studies')
studies = sorted(path.parent for path in root.glob('study_*/study.json'))
if len(studies) != 1:
    print('请从启动日志确定准确 --resume-study 路径。当前研究：', file=sys.stderr)
    for path in studies:
        print(path, file=sys.stderr)
    raise SystemExit(1)
print(studies[0])
PY
)
mapfile -t single_overfit_devices < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2<=500 && $3>=20000 && $4<=5 {gsub(/[[:space:]]/,"",$1); print "cuda:" $1; if (++count==4) exit}')

if pgrep -u "$(id -u)" -af '[r]un_single_teacher_overfit[.]py|foundation_sft[.]single_teacher_overfit|[r]un_single_teacher_distillation[.]py.*runs/single_teacher_overfit|foundation_sft[.]distillation.*runs/single_teacher_overfit'; then
    echo "已有本轮协调器或worker，未重复启动。"
elif [ -z "$single_overfit_study" ] || [ "${#single_overfit_devices[@]}" -eq 0 ]; then
    echo "未确定唯一研究或空闲GPU，未启动。"
else
    single_overfit_log="/home/cfu/EVA-X-main/runs/single_teacher_overfit/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup /opt/anaconda3/envs/evax_lora/bin/python -u run_single_teacher_overfit.py \
        --project /home/cfu/EVA-X-main \
        --resume-study "$single_overfit_study" \
        --devices "${single_overfit_devices[@]}" \
        --max-parallel "${#single_overfit_devices[@]}" \
        > "$single_overfit_log" 2>&1 < /dev/null &
    echo "恢复：$single_overfit_study；GPU：${single_overfit_devices[*]}；PID：$!；日志：$single_overfit_log"
    tail -n 40 -f "$single_overfit_log"
fi
```

有多个研究时，把日志里的准确绝对路径赋给 `single_overfit_study`，再执行相应的 `--study` 或 `--resume-study` 命令；不要重跑新建段。需要停止时先检查上面同一条 `pgrep` 输出，核对本轮研究路径和PID后再单独处理对应进程，不使用匹配所有Python、原始蒸馏或其他研究的批量终止命令。

## 撤销代码安装

使用安装输出的准确 `backup` 路径替换下面示例目录，先预览，再添加 `--apply`：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/single_teacher_overfit_update/update_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --rollback /home/cfu/EVA-X-main/.single-teacher-overfit-backups/实际安装备份目录
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/single_teacher_overfit_update/update_single_teacher_overfit.py \
    --project /home/cfu/EVA-X-main \
    --rollback /home/cfu/EVA-X-main/.single-teacher-overfit-backups/实际安装备份目录 --apply
```

回滚恢复被替换代码并移除本次新增代码；发现安装后用户修改时停止并保留修改。研究记录、权重、缓存和所有历史结果保留。

## 开发验证记录

本轮专项验证49项通过：诊断模块7项、协调器22项、报告10项、安装器9项，以及真实协调器调用原生worker的端到端检查1项。原单教师测试6项、原更新器测试15项也通过。端到端检查使用本地CPU、合成DICOM、微型Hugging Face教师和完整可训练的合成学生；没有在本地GPU或真实服务器影像上执行训练，服务器运行前仍需完成上述四组预检查。

完整仓库运行980项，结果为2项失败、23项跳过。两个失败均为既有VLM清单对 `data.py` 的哈希检查：`test_extra_vlm_tools.ExtraVLMPublishedManifestTests` 和 `test_vlm_update.VLMManifestTests`；在隔离的未修改基线 `950c7c65ca55764b51262a2e4de789200c8cc754` 中可复现相同失败。本包不修改 `data.py`，也不为消除这些既有失败改写历史发布清单。
