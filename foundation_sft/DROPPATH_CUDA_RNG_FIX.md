# DropPath CUDA 随机状态兼容性修复

本修复用于已安装 DropPath 实验、但 A/B 预检查均报错的项目：

```text
ValueError: Invalid DropPath CUDA RNG state length
```

原代码把 CUDA 随机状态固定为16字节并按字节推断种子；服务器的 PyTorch 使用了不同布局，导致训练前保存重载检查失败。本修复把 CUDA 随机状态视为不透明的 ByteTensor，由实际目标设备的独立 `torch.Generator.set_state()` 校验并恢复。CPU 加载保留 CUDA 状态，待 GPU 使用时再由对应后端校验，避免误用默认 GPU 0 或重新播种绕过检查。

本包更新两份训练代码，并新增此说明。`DropPath=0.1`、12层线性概率、隔离随机流的种子、学生初始参数、教师、`weight_decay=0.01`、seed42、学习率和蒸馏系数均保持原设置。已有结果和失败记录保留。当前两项失败发生在预检查，尚未正式训练。

安装器只允许替换已发布的原 DropPath 文件版本，校验来源与 SHA256，自动备份并原子更新；未知本地修改会保留。修复经过 CPU 和模拟后端回归测试，实际 CUDA 保存重载仍由服务器重新预检查验证。

## 安装小修复包

将 `FU254-droppath-cuda-rng-fix.zip` 解压，得到 `droppath_cuda_fix` 文件夹。通过 PyCharm 将整个文件夹上传到 `/home/cfu/EVA-X-main/`，在服务器执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/droppath_cuda_fix/update_droppath_cuda_rng.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/droppath_cuda_fix --apply
```

## 恢复原批次

修复后恢复原研究，继续使用已有对照：

```text
/home/cfu/EVA-X-main/runs/random_student_droppath/studies/study_20261025T091628Z_7db1bca9
```

下面整段会检查同类进程，并自动选择剩余显存至少20GB、利用率不超过5%的 GPU，后台串行重做预检查及训练：

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
    droppath_log="runs/random_student_droppath/resume_$(date +%Y%m%d_%H%M%S).log"
    nohup python -u run_droppath_experiments.py \
        --resume-study /home/cfu/EVA-X-main/runs/random_student_droppath/studies/study_20261025T091628Z_7db1bca9 \
        --devices "cuda:$droppath_gpu" --max-parallel 1 \
        --cache-dir runs/foundation_sft/data_cache \
        > "$droppath_log" 2>&1 < /dev/null &
    echo "使用 GPU $droppath_gpu；PID：$!；日志：$droppath_log"
    tail -n 30 -f "$droppath_log"
fi
```

按 Ctrl+C 退出日志查看，后台训练继续。恢复会重新检查来源与随机状态，原两项失败任务从头训练；后续若某项完成，恢复时会核验并保留完成结果。

查看这次原研究的总状态和结果：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_droppath_experiments.py --show-metrics --study /home/cfu/EVA-X-main/runs/random_student_droppath/studies/study_20261025T091628Z_7db1bca9
```

单项日志正常结束后可能仍以 `[evaluation]` 为最后一行；请依据总状态中的 `completed`、`training` 或 `failed` 判断进度。
