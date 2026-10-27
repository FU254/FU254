# 保留正在训练的 P-C，单独并行运行 P-R

本增量更新用于已经启动的 teacher-weight study：保留 GPU 0 上当前 P-C 的轮次和优化器状态，在空闲 GPU 3 上运行尚未开始的 P-R。它沿用本研究已经冻结的教师、学生 ImageNet 初始化、验证集选择的学习率、数据划分、损失系数和 seed42，不创建新的科学实验配方。

本批次路径是：

```text
/home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f
```

## 安装

解压 `FU254-teacher-weight-parallel.zip`，将整个 `teacher_weight_parallel_update` 目录上传到 `/home/cfu/EVA-X-main/`。不用停止 P-C，也不用重新运行旧调度命令。在服务器执行：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/teacher_weight_parallel_update/update_teacher_weight_parallel.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/teacher_weight_parallel_update --apply
```

安装器只新增并行辅助模块、入口和本说明，并以原子替换更新很小的 `run_teacher_weight_training.py` 入口。它不导入训练模块、不发送停止信号、不改当前配置、权重或历史数据。入口最后发布，保证未来进程可先找到已安装的辅助模块。正在运行的 P-C 已经加载入口和训练函数，因此会继续当前训练。

更新只接受已发布版本的旧入口或完全相同的新入口；未知的本地修改会保留并报错。预览、校验、备份、失败恢复和重复安装沿用原更新方式。安装器会打印备份路径。不要在独立任务或旧队列仍可能使用本更新时回滚入口。

## 在 GPU 3 后台启动 P-R

先看当前显存和利用率，确认 GPU 3 仍空闲。独立入口实际启动前还会自动检查目标 GPU 空闲显存至少 20GB、利用率不超过 5%；不满足时会退出，保留原任务继续运行。`--plan` 和 `--status` 只读状态，不做 GPU 训练检查：

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
```

然后直接执行下面整段。路径全部为绝对路径，当前终端是 `base` 也可以执行：

```bash
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES
teacher_weight_parallel_log="/home/cfu/EVA-X-main/runs/teacher_weight_study/parallel_PR_$(date +%Y%m%d_%H%M%S).log"

nohup /opt/anaconda3/envs/evax_lora/bin/python -u \
    /home/cfu/EVA-X-main/run_teacher_weight_parallel.py \
    --project /home/cfu/EVA-X-main \
    --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f \
    --group P-R --device cuda:3 \
    > "$teacher_weight_parallel_log" 2>&1 < /dev/null &

echo "P-R 辅助调度PID：$!；日志：$teacher_weight_parallel_log"
tail -n 30 -f "$teacher_weight_parallel_log"
```

按 `Ctrl+C` 只退出日志查看。`nohup` 后台训练继续；不要重复创建整批研究。GPU 0 和 GPU 3 各承担一个独立实验，GPU 4 可以保留给其他任务。

## 如何避免旧队列再次训练 P-R

独立任务使用当前研究的已冻结 P-R 配置做检查和训练，并保存独立的回执；它不与旧协调器并发改写 `study.json`。每组独立锁控制同时进入训练的进程。

旧队列完成 P-C 后，仍会走到自己的 P-R 工作项。新的小入口会识别本组已被独立任务接管；如果 P-R 正在训练，它会等待；如果 P-R 已完成，它会严格核验并复用同一个学生检查点，把结果交回旧协调器。随后由原协调器正常汇总六组结果和曲线所需历史。

因此，旧日志出现等待或复用提示是正常行为，不等于多启动一次优化训练。独立任务失败时按日志排查；不要删除组锁文件或手工拼接回执来绕过校验。

如果安装前旧队列已经开始训练 P-R，独立辅助入口应识别正在运行的本组任务并拒绝重复启动。此时以现有 P-R 的日志和 GPU 占用为准。

## 查看状态、结果和曲线

GPU 分配及当前进程：

```bash
nvidia-smi
pgrep -u "$(id -u)" -af '[r]un_teacher_weight_(experiments|training|parallel)\.py'
```

查看独立 P-R 的接管记录、运行目录和原协调器状态：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_parallel.py --project /home/cfu/EVA-X-main --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f --group P-R --device cuda:3 --status
```

本批次汇总：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py --project /home/cfu/EVA-X-main --show-metrics --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f
```

由本批次保存的历史生成曲线：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/run_teacher_weight_experiments.py --project /home/cfu/EVA-X-main --plot-curves --study /home/cfu/EVA-X-main/runs/teacher_weight_study/studies/study_20261026T134529Z_e3d7615f
```

独立 P-R 尚未被旧协调器汇总时，总表可能暂时仍显示待运行；以独立日志和它打印的运行目录为准。原协调器收到回执后会正常更新主表。没有旧协调器时，不应假定主表会由独立进程自动更新。

仍用同口径的 `train_eval_loss` 与 `val_loss`、训练与验证患者 ROC-AUC 判断拟合趋势；优化总损失另含蒸馏，不能直接拿来与验证 Focal loss 比较。本更新只改变任务调度，不改变最佳验证检查点选择或过拟合判定标准。

本地 CPU 与进程模拟验证不能替代服务器 GPU 检查，实际训练前由项目原有检查核对教师、学生、梯度和检查点兼容性。
