# 单教师 RAD-DINO → ViT-Tiny 蒸馏

本实验新增一个独立入口，比较同一 ViT-Tiny 学生的两组结果：

| 组别 | 学生 | 训练目标 |
| --- | --- | --- |
| 监督基线 | ImageNet 预训练 ViT-Tiny | 真实标签监督 |
| 单教师蒸馏 | 同一 ImageNet 预训练 ViT-Tiny | 标签监督 + 软预测 + 全局特征 + 局部特征蒸馏 |

教师为已完成本任务 LoRA 微调的 RAD-DINO。教师保持冻结并使用评估模式，训练阶段仅更新学生及特征对齐模块。监督基线与蒸馏组使用相同的数据划分、学生初始权重和实验设置。教师只在训练集上提供蒸馏信号；学生模型选择和阈值确定使用验证集，测试集在选择完成后进行最终评估。

两组依次运行，默认使用 `cuda:0`、种子 42、FP32，每组最多 100 轮，验证集连续 15 轮不改善时早停；每批 8 张，累积 8 批更新。蒸馏温度为 2，软预测、全局、局部损失权重分别为 0.5、0.1、1.0，局部特征池化为 7×7。这些是首次实验的预设参数，尚未通过本数据验证或调优。

此增量包只增加独立实验文件，不替换旧训练器、模型注册表、公共配置，不重新训练原有教师。原有影像、Excel、基础权重、数据划分、像素缓存和历史实验保留。

## 安装到现有服务器项目

在 Windows 把 `FU254-single-teacher-kd.zip` 解压到 `D:\99\EVA-X-main`，得到 `D:\99\EVA-X-main\kd_update`。在 PyCharm 中上传整个 `kd_update` 文件夹，服务器对应目录为 `/home/cfu/EVA-X-main/kd_update`。不要把该目录中的配置手动覆盖到旧公共配置中。

上传后安装器应在：

```text
/home/cfu/EVA-X-main/kd_update/update_single_teacher_distillation.py
```

先检查本次新增的文件清单：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/kd_update/update_single_teacher_distillation.py --project /home/cfu/EVA-X-main
```

确认输出仅列出此独立蒸馏流程的文件后，执行安装：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/kd_update/update_single_teacher_distillation.py --project /home/cfu/EVA-X-main --apply
```

安装器默认从自身所在目录读取新增代码，核对所有 SHA256 后才写入。没有 `--apply` 时只预览；目标已存在且内容相同会跳过，目标已存在且内容不同会整批拒绝，保留已有修改。安装不联网、不安装或升级 Python 依赖。

新文件的安装记录与代码副本保存在项目的 `.single-teacher-kd-backups/<时间和标识>/`。安装中途失败会自动移除本次已发布的新文件；原有文件不参与替换。成功后也可先预览回退，再实际回退：

```bash
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/kd_update/update_single_teacher_distillation.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.single-teacher-kd-backups/实际备份目录
/opt/anaconda3/envs/evax_lora/bin/python /home/cfu/EVA-X-main/kd_update/update_single_teacher_distillation.py --project /home/cfu/EVA-X-main --rollback /home/cfu/EVA-X-main/.single-teacher-kd-backups/实际备份目录 --apply
```

回退只移除该次安装新增且后来未改动的文件；发现用户修改会拒绝，不删除权重、缓存或实验结果。备份目录名以安装输出中的 `backup` 为准。

## 教师与学生权重

默认从已完成的 RAD-DINO LoRA 批次 `suite_20261013T024024Z_92ac596d` 的训练日志定位训练目录，再加载其验证集选出的 `best.pt`。不依据测试分数筛选教师，不自动选取其他新批次。也可以通过 `--teacher-checkpoint /完整训练目录/best.pt` 明确指定教师。

LoRA 的 `best.pt` 包含该任务的可训练参数，恢复教师时仍需要当时使用的基础预训练权重及相同架构、输入预处理、LoRA 结构。程序核对保存的模型元数据和冻结数据划分；缺少文件或不匹配时先修正，不能用随机初始化替代教师。默认日志位置为：

```text
/home/cfu/EVA-X-main/runs/foundation_sft/suites/suite_20261013T024024Z_92ac596d/logs/rad_dino_lora_seed42_train.log
```

原教师的数据划分文件必须仍然存在。数据匹配与灰度像素处理继续调用原有流程，默认复用 `runs/foundation_sft/data_cache`，不会重新划分患者。

学生默认权重路径：

```text
/home/cfu/EVA-X-main/foundation_weights/vit_tiny/deit_tiny_patch16_224-a1311bcf.pth
```

这是 ViT-Tiny/16 的 ImageNet 初始化权重。只下载权重文件不能替代本任务训练；监督基线和蒸馏组都从同一份初始化开始。该增量代码包不包含教师或学生的大权重。

若尚未下载，可以在能联网的电脑打开 [DeiT 官方 ViT-Tiny 权重](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)，保存为：

```text
D:\99\EVA-X-main\foundation_weights\vit_tiny\deit_tiny_patch16_224-a1311bcf.pth
```

然后通过 PyCharm 上传到上面的服务器对应路径，等待完整传输后再检查。若该文件已经完整存在，就直接复用。

## 先检查，再运行两组实验

在服务器终端执行：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_single_teacher_distillation.py --check-only
```

`--check-only` 完整恢复教师，核对冻结划分，并对一个训练小批次执行蒸馏前向、反向及完整学生参数的保存/重载校验；不执行优化器更新。它不运行全量像素审计或正式训练轮次，检查记录保存在 `runs/single_teacher_distillation/checks/`。

检查成功后启动默认的监督基线与单教师蒸馏两组：

```bash
python run_single_teacher_distillation.py
```

需要明确指定教师时：

```bash
python run_single_teacher_distillation.py --teacher-checkpoint /实际RAD-DINO训练目录/best.pt
```

配置文件为 `foundation_sft/configs/distillation_rad_dino_vit_tiny.json`。可以用 `--config` 指向另一份独立实验配置，用 `--student-checkpoint` 指定已下载的官方学生权重，用 `--cache-dir` 指向原实验缓存，用 `--device cuda:4` 指定运行 GPU。`--project /home/cfu/EVA-X-main` 可以显式指定项目；各相对路径均以该项目为基准。

查看本流程已完成的结果：

```bash
python run_single_teacher_distillation.py --show-metrics
```

每次正式运行创建一个新的 `runs/single_teacher_distillation/suites/suite_时间_标识/`，保留旧批次。查看某一次批次时使用：

```bash
python run_single_teacher_distillation.py --show-metrics --suite /home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/实际批次目录
```

结果查看只读取保存的指标，不加载教师或学生权重，不扫描影像。

PyCharm 使用 `/opt/anaconda3/envs/evax_lora/bin/python` 作为远程解释器，脚本设为 `/home/cfu/EVA-X-main/run_single_teacher_distillation.py`。检查时在参数栏填写 `--check-only`；正式运行时清空该参数。`--show-metrics` 用于查看结果。若终端通过清除 `LD_LIBRARY_PATH` 解决过 CUDA 库冲突，PyCharm 的运行环境也应保持同样设置。

## 结果如何解释

两组均按验证集患者 ROC-AUC 选取最佳学生参数，随后在验证集确定分类阈值；测试集固定使用该阈值，不参与训练、早停或阈值选择。报告患者 ROC-AUC、PR-AUC、敏感度、特异度等指标以及患者 bootstrap 置信区间。一个训练种子的差异不能替代多次重复实验或统计显著性检验。

总损失为标签监督的 Focal Loss，加上 0.5 倍软预测损失、0.1 倍全局损失和 1.0 倍局部损失。软预测采用二分类 KL（包含温度平方缩放），传递教师对阳性/阴性的相对判断；全局损失将学生特征线性投影到教师维度后计算余弦距离。

局部损失比较肺野位置之间的关系：去除 CLS 等特殊 token 后，将 patch 特征池化到 7×7 并逐位置做 L2 归一化，再计算位置之间的 Gram 相似度矩阵，以非对角元素的均方误差约束学生。它不要求学生逐通道复制教师的 patch 向量。两种模型使用同一张图像的同一次空间增强，学生由教师视野重采样到 224×224，分别归一化，避免不同裁剪位置直接配对。

每个正式批次包含：

| 文件或目录 | 内容 |
| --- | --- |
| `suite.json`、`comparison.csv` | 两组完成状态及患者指标对比 |
| `supervised/`、`distillation/` | 两组各自的独立训练结果 |
| 每组 `best.pt` | 验证集选择的完整学生参数、阈值与元数据，可在恢复学生时独立于教师及 ImageNet 初始化文件加载 |
| 每组 `last.pt`、`history.json` | 最近一轮训练状态与逐轮损失、验证指标记录 |
| 每组 `training_loss.png`、`validation_auc.png` | 损失与验证指标曲线 |
| 每组 `teacher.json`、`shared_view.json` | 教师指纹及共享视野、归一化设置 |
| 每组 `test/metrics.json`、`test/patient_predictions.csv` | 固定阈值的测试指标、置信区间和逐患者预测 |
| 每组 `test/roc_curve.png`、`test/precision_recall_curve.png`、`test/calibration_curve.png` | 测试 ROC、PR、校准曲线 |

## 使用保存的学生重新预测

部署或复算时应同时调用 `load_student_checkpoint` 和 `inference_dataset`。后者根据 checkpoint 的 `shared_view` 重建原视野：先执行保存的教师尺寸缩放/裁剪，再二次缩放到学生的 224×224 并归一化。这一步只重建图像处理，不加载教师模型、教师基础权重或学生的 ImageNet 初始化文件。

仅使用 `CXRDataset(student.preprocess)` 会遗漏第一次视野处理，改变输入，不能替代这个 helper。下面示例复算某次蒸馏组原有的测试患者，使用已保存的验证集阈值：

```python
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from foundation_sft.distillation import load_student_checkpoint, inference_dataset
from foundation_sft.metrics import aggregate_patient_predictions
from foundation_sft.train import predict

run = Path("/home/cfu/EVA-X-main/runs/single_teacher_distillation/suites/实际批次目录/distillation")
device = torch.device("cuda:0")
student, payload = load_student_checkpoint(run / "best.pt", device=device)
split = pd.read_csv(run / "split.csv", dtype={"sample_id": str, "patient_id": str},
                    keep_default_na=False)
test = split.loc[split["split"] == "test"].copy()
dataset = inference_dataset(test, payload, cache_dir=payload["config"].get("cache_dir"))
predictions = predict(student, DataLoader(dataset, batch_size=8, shuffle=False), device)
patients = aggregate_patient_predictions(predictions)
patients["predicted_label"] = (patients["probability"] >= payload["threshold"]).astype(int)
```

完整性检查用于证明训练流程可执行，不能证明蒸馏一定提高效果。模型效果以这次独立、冻结测试集的实际结果为准。
