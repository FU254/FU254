# 三教师等权蒸馏

本轮固定已选的 RAD-DINO、CheXFound、CheXzero 三位 LoRA 教师，均使用 seed42 的验证集最优检查点，冻结参数。学生为 ImageNet 初始化的 ViT-Tiny，采用全参数微调。

| 条件 | 训练信号 | 学生种子 |
| --- | --- | --- |
| `supervised` | 真实标签 | 42、43、44 |
| `single_teacher` | 真实标签 + RAD-DINO 三项蒸馏 | 42、43、44 |
| `equal_teacher` | 真实标签 + 三教师等权三项蒸馏 | 42、43、44 |

每位教师权重为 1/3，Focal Loss 只计算一次：

```text
L = Focal + mean_j(0.5 × soft_j + 0.1 × global_j + 1.0 × local_j)
```

温度为 2，局部关系池化为 7×7；每位教师使用独立的学生全局投影头，不直接平均不同教师的特征。学生初始化、数据划分、标签、输入视野、几何增强、学习率、FP32、轮数和早停均与原全参实验保持一致。

所有输入来自同一份增强后的 RAD-DINO 图像视野。学生和 RAD-DINO 输入保留原流程；另两位教师对该视野做确定性缩放并使用各自的归一化，不额外裁剪或补边。教师原生预处理与本轮实际共享视野预处理分别记录在 `teacher_views.json`，包括 CheXzero 原生补边未应用这一差异。

本轮共 9 条预先指定的结果。默认仅从下面这个已完成的消融批次核对复用六条对照：

```text
runs/distillation_ablation/suites/suite_20261021T090957Z_5c9c9936
```

对应 `supervised` 和 `full_kd`。只有保存的配置、学生初始化、教师检查点、冻结数据、输入视野和评估记录一致时才复用；不匹配则说明原因并补训。全部匹配时只新增三次等权蒸馏训练。

## 安装

将 `FU254-equal-teacher-update.zip` 解压到 Windows 项目目录，得到：

```text
D:\99\EVA-X-main\equal_teacher_update\update_equal_teacher.py
```

通过 PyCharm 上传整个 `equal_teacher_update` 文件夹到服务器项目，再执行下面完整命令：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/equal_teacher_update/update_equal_teacher.py --project /home/cfu/EVA-X-main --source-dir /home/cfu/EVA-X-main/equal_teacher_update --apply
```

安装新增九个文件，不更改旧训练入口、权重、数据和历史实验。上传内容先通过 SHA256 和 Python/JSON 语法检查；同版本重复安装会跳过。发现目标文件已有不同内容时保留并停止整批安装。安装记录保存在 `.equal-teacher-backups`。

## 启动实验

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_equal_teacher_experiments.py --devices cuda:0 --max-parallel 1 --cache-dir runs/foundation_sft/data_cache
```

入口会自动进行训练前检查、对照复用核对和训练。无需先运行 `--check-only`。默认每张卡同时只运行一个任务，每轮训练完成都会在终端显示进度；任务状态也及时写入当前批次的 `suite.json`。

使用多张空闲 GPU 时只改命令参数，例如：

```bash
python run_equal_teacher_experiments.py --devices cuda:0 cuda:1 cuda:2 --max-parallel 3 --cache-dir runs/foundation_sft/data_cache
```

三教师训练需要额外的教师前向计算，耗时和显存会高于单教师。程序保留已确认的批量和 FP32 设置；设备不足时改用空闲卡，不自动改变实验设置。

只查看计划或只运行实际前后向与保存重载检查：

```bash
python run_equal_teacher_experiments.py --plan
python run_equal_teacher_experiments.py --devices cuda:0 --max-parallel 1 --check-only
```

这两条不会完成训练，也不会生成测试性能结论。

## 教师来源

配置文件是 `foundation_sft/configs/equal_teacher_distillation.json`。默认分别读取明确指定的训练日志或回执所记录的 `best.pt`：

| 教师 | 原训练批次 |
| --- | --- |
| RAD-DINO、CheXFound | `runs/foundation_sft/suites/suite_20261013T024024Z_92ac596d` |
| CheXzero | `runs/foundation_sft_vlm/suites/suite_20261015T045005Z_2162f5a1` |

若教师做过恢复训练，或者原日志已移动，在对应教师条目中将 `checkpoint` 从 `null` 改为实际微调结果的 `best.pt` 绝对路径。这里需要的是已训练的任务检查点，不是 `foundation_weights` 中下载的原始基础模型文件。程序验证模型、LoRA 模式、种子、标签和患者划分，不会按测试成绩或“最新目录”自行挑选教师。

## 查看结果和继续队列

```bash
python run_equal_teacher_experiments.py --show-metrics
```

查看某批次时，在下列命令中替换启动时打印的实际 `suite_...` 路径：

```bash
python run_equal_teacher_experiments.py --show-metrics --suite /home/cfu/EVA-X-main/runs/equal_teacher_distillation/suites/实际批次目录
python run_equal_teacher_experiments.py --resume-suite /home/cfu/EVA-X-main/runs/equal_teacher_distillation/suites/实际批次目录 --devices cuda:0 --max-parallel 1
```

继续队列会核对已完成结果并补齐缺项，未完成训练任务重新开始；已完成任务无需重复训练。继续队列时可以更换 GPU 和并发数，科学设置必须一致。希望全部重训对照时，在新批次启动命令中增加 `--no-reuse`。

报告输出到该批次的 `report` 文件夹，保留逐种子结果、跨种子均值及样本标准差、同种子配对差值。验证集 ROC-AUC 用于检查点选择，阈值由验证集确定；测试集只用于最终报告。PR-AUC 和 AP 分开输出。患者 bootstrap 置信区间和种子间标准差分别反映不同的不确定性，不筛掉不利种子。

本轮能够比较三教师等权方案与纯监督、RAD-DINO 单教师的差异。若要声称优于每一个组成教师，还需另补 CheXFound 和 CheXzero 单教师的三个种子对照。
