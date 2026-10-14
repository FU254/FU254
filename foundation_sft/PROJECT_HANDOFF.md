# 项目交接：胸片模型实验与 CNN 基线

更新日期：2026-10-14。本文用于新对话或新任务接续工作；详细使用说明见 [README.md](README.md)，权重与源码来源见 [MODEL_SOURCES.md](MODEL_SOURCES.md)。

**当前任务：新增三个 CNN，仅训练分类头。代码和云端验证已完成；尚未上传用户服务器，三份权重需在联网电脑下载并上传，再进行服务器检查。**

## 0. 最新进度与下一步

- 用户现有模型范围为 EVA-X、RAD-DINO、CXformer、CheXFound、RadJEPA、XraySigLIP、BiomedCLIP、CheXzero；代码仍兼容 CXR-CLIP。不要把所有权重目录都解释成已完成实验。
- 用户已打印五个基础模型两模式的结果：`suite_20261013T024024Z_92ac596d` 中 9 项完成，CXformer LoRA 因非有限损失失败。已有 FP32 恢复工具，但未收到恢复后的完成证据。
- XraySigLIP 的线性探测和 LoRA 已由服务器日志确认完成：`runs/foundation_sft_vlm/suites/suite_20261013T114034Z_19e8e5d5`。其他 VLM 的完成状态仍需查看实际结果。
- 新增 `densenet121_imagenet`（torchvision ImageNet）、`densenet121_cxr`（TorchXRayVision 多胸片数据集监督预训练）、`resnet18_cxr`（Stanford MoCo-CXR/CheXpert 自监督）。用户确认三项只训练分类头；没有下载其权重。torchvision 自带的权重属于 ImageNet，不能替代医学预训练权重。
- CNN 套件为 `foundation_sft/configs/experiments_cnn.json`，只有 `linear_probe`，FP32、种子 42、默认 GPU 0/1/2。冻结编码器及 BatchNorm；DenseNet 分类头可训练参数 1025，ResNet 分类头 513。数据/标签/划分及训练设置继承用户当前 `common.json`，保留用户已有缓存和恢复后的清单路径。
- 使用 `download_cnn_weights.py` 在联网电脑下载，再用 PyCharm 上传到服务器三个新增权重目录；代码使用 `update_foundation_cnn.py` 做带备份的增量安装。指南见 [CNN_BASELINES.md](CNN_BASELINES.md)。代码分支：[`codex/cnn-linear-baselines`](https://github.com/FU254/FU254/tree/codex/cnn-linear-baselines)。
- 云端已核对两份 DenseNet 官方实物的 SHA-256，验证加载、仅更新分类头、编码器/BN 不变。MoCo-CXR 官方来源和架构已核实，但云端访问 Google Storage 返回 403，真实检查点未取得；其软件验证使用合成 MoCo 权重，不能称为真实医学权重验证。
- 两套 CPU 环境（Torch 2.9.1 与用户相同版本的 Torch 2.0.1）各运行 368 项 unittest，无失败，分别跳过 16/21 项可选依赖或上游模型检查。另用合成 DICOM、两份真实 DenseNet 权重与一份合成 MoCo 权重完成三项 prepare、全量审核、smoke、两轮训练、自动测试及比较报告。
- 下一步在服务器运行 CNN `--check-only`，全部通过后正式运行。新结果保存到 `runs/foundation_sft_cnn`；现有 `show_foundation_results.py` 默认扫描可包含 CNN，显式指定 `--models` 时添加三个新 ID。
- GitHub 提交不等于已上传服务器。不要声称已替用户连接校园网服务器、下载其电脑文件或启动正式 CNN 训练。

以下第 1–6 节保留 2026-10-12 的历史记录；训练状态与当前工作以本节为准。

## 1. 工作方式与目标

- 用户在 Windows PyCharm 编辑代码，通过 Deployment 上传到服务器，使用服务器的 SSH 解释器、数据和 GPU。
- 本地项目：`D:\99\EVA-X-main`；实际服务器项目：`/home/cfu/EVA-X-main`。早期提到的 `/home/cfu/DR` 不再作为操作目录。
- 服务器连接需要校园网或 VPN，当前 Codex 云环境无法直接访问。GitHub 提交不等于已经上传服务器；服务器上的执行结果需由用户反馈。
- 仓库：<https://github.com/FU254/FU254>，分支 `main`。本文对应的训练/数据修复代码版本：[`764b7c8`](https://github.com/FU254/FU254/commit/764b7c8994a3f9b1fbef495a3b531b944139cd3e)。
- 目标：五个基础模型分别运行线性探测（`linear_probe`）和 LoRA，共 10 组完整实验，包含数据检查、训练/验证、最佳参数选择、最终测试和结果汇总。

## 2. 模型、环境和入口

| 模型 | 配置 | Conda 环境 | 默认 GPU |
|---|---|---|---|
| EVA-X Base | `configs/eva_x.json` | `evax_lora` | `cuda:0` |
| RAD-DINO | `configs/rad_dino.json` | `evax_lora` | `cuda:1` |
| CXformer Base | `configs/cxformer.json` | `evax_lora` | `cuda:2` |
| CheXFound | `configs/chexfound.json` | `evax_lora` | `cuda:3` |
| RadJEPA | `configs/radjepa_imagenet.json` | `radjepa_sft` | `cuda:4` |

默认 GPU 映射来自 `configs/experiments.json`；启动前需根据服务器实际占用选择。不同卡并行，同一张卡上的两种模式顺序执行。RayDINO 已有适配器，但未确认取得权重，不在本次五模型实验内。

主入口为项目根目录的 `run_foundation_experiments.py`。单模型可以运行 `train_eva_x.py`、`train_rad_dino.py`、`train_cxformer.py`、`train_chexfound.py`、`train_radjepa.py`，也可以使用模块 `foundation_sft.train`。直接以文件方式运行 `foundation_sft/train.py` 曾触发相对导入错误，推荐模块或根目录入口。

已知兼容组合：Python 3.10、PyTorch `2.0.1+cu118`、xformers `0.0.21`；公共 Hugging Face 依赖固定为 Transformers `4.48.3`、huggingface-hub `0.29.3`、tokenizers `0.21.1`、safetensors `0.4.5`。RadJEPA 使用单独环境及 timm `1.0.24`，避免影响 EVA-X 的旧 timm 依赖。服务器诊断确认 pydicom 为 `2.4.4`。

旧 CUDA 路径 `/usr/local/cuda-11.0/lib64` 会通过 `LD_LIBRARY_PATH` 干扰 Torch 的 CUDA 库。终端运行前使用 `unset LD_LIBRARY_PATH`；PyCharm 运行配置需设置空 `LD_LIBRARY_PATH`。统一实验入口会为子进程清除这个变量。服务器已验证 xformers GPU 注意力前向、反向成功。

## 3. 数据与实验规范

- 当前影像目录：`/data/users/cfu/DR数据_按文件10份-筛选合成`。
- 候选完整原图目录：`/data/users/cfu/DR数据`。用户已经确认它存在，尚未确认其中的对应原图完整。
- Excel：`/home/cfu/肺功能万能表最终版1_fev1fvc差值标签(1).xlsx`；标签列 `阻塞性通气障碍`，检查号列 `放射编号`。
- Excel 没有已识别的患者列，当前使用 DICOM PatientID。用户已确认每位真实患者仅保留一次胸片，无需再次询问这个问题。
- 已有 prepare 结果：Excel 2282 行，DICOM 2006 个，11 个不支持的影像；纳入 1995 张。训练 1397 张/阳性 168，验证 199 张/阳性 24，测试 399 张/阳性 48。
- 共同配置为 `foundation_sft/configs/common.json`；旧清单为 `runs/foundation_sft/shared_patient_split.csv`。训练种子与划分种子均为 42，比例 0.7/0.1/0.2。
- 所有模型和模式使用同一份固定患者分组。最佳参数按验证集患者 ROC-AUC 选择，阈值按验证集确定；测试集仅用于最终评估，不用测试结果选择种子、阈值或排除影像。
- 线性探测只训练一个线性分类层。LoRA 训练注意力适配器、输出 LayerNorm 和分类头，基础编码器冻结。
- RadJEPA 当前采用明确记录的 ImageNet 研究归一化设置；作者预训练的归一化和 resize 规范尚未核实，报告必须保留这一说明。
- Excel 标签直接沿用，LLN 参考方程及支气管舒张状态仍未确认，不能声称代码重新建立了临床诊断标准。

## 4. 已完成与当前阻塞

此前已解决 CUDA 库冲突、Torch 2.0 与 Transformers 的兼容问题、受限 checkpoint 加载的 NumPy/set 元数据、GPU 重载验证的浮点容差，以及 EVA-X MIM checkpoint 的 `norm`/`fc_norm` 架构匹配。CheXFound 权重曾上传不完整，后已重新上传并通过检查。

用户随后提供了五模型、两模式共 10 项真实服务器 smoke 全部通过的日志。smoke 验证一个小批次，不代表全部 DICOM 都完整，也不代表正式训练完成。

正式实验 `suite_20261012T061516Z_1bb2a182` 中，RAD-DINO 和 CXformer 在训练分组的零基行号 624 读取同一张影像失败。实际诊断为：

| 技术项目 | 已确认结果 |
|---|---|
| 传输语法 | Explicit VR Little Endian，未压缩 |
| 像素布局 | 3032×3032，单通道，16 位 |
| PixelData 声明/预期字节数 | 18,386,048 |
| 实际 PixelData 字节数 | 5,419,848 |
| 压缩封装、JPEG、JPEG2000 标记 | 均无 |

因此这是物理截断，缺失真实像素。原图为 2000 多像素也可使用，各模型预处理会调整输入尺寸；模型不要求原始 DICOM 必须为 3200×3200。该故障无法通过改尺寸、补零或安装解码器恢复。

版本 `764b7c8` 新增全量像素审核：每个不同 Conda 环境先解码全部训练、验证和测试影像；任何失败都会在模型 smoke 和训练开始前阻止统一实验。审核只记录错误，不丢弃样本或修改分组。

新增恢复工具默认只生成计划；`--apply` 才写入。它按 SOPInstanceUID 查找副本，核对身份、像素参数、灰阶/窗口元数据、残存像素字节和完整解码，拒绝不匹配或歧义副本。应用时先备份损坏文件，再校验临时副本 SHA-256 并原子替换，复制期间源/目标变化会拒绝覆盖。

## 5. 下一步：在服务器执行

**最新代码是否已经上传服务器、全量审核有多少失败、候选原图是否能恢复，这三件事尚待用户实际输出确认。**

先将版本 `764b7c8` 的 `foundation_sft/audit_data.py`、`recover_data.py`、`train.py`、`workflow.py` 替换到本地项目，再通过 PyCharm 上传到 `/home/cfu/EVA-X-main/foundation_sft`。保留用户已修改的数据和权重配置。

在 SSH 终端逐行执行，避免把多条命令粘贴成一行：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
```

全量审核（目录名需是一个未使用的新名字）：

```bash
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action audit-data --audit-output runs/foundation_sft/audit_before_restore_01
```

有损坏时退出非零是预期结果，所有失败仍会保存。再生成只读恢复计划：

```bash
python -m foundation_sft.recover_data --audit-dir runs/foundation_sft/audit_before_restore_01 --source-dir /data/users/cfu/DR数据
```

下一次对话先查看用户反馈的 `failed_images` 和恢复 `counts`。`ready` 表示找到可恢复原图；`unresolved` 表示缺少唯一完整匹配副本，`not_eligible` 表示应先处理解码器或预处理问题。不要假设只有一张损坏文件，也不要凭目录存在就宣称原图完整。

确认计划后、结束仍在读取这批数据的旧实验，再添加 `--apply` 恢复已验证文件：

```bash
python -m foundation_sft.recover_data --audit-dir runs/foundation_sft/audit_before_restore_01 --source-dir /data/users/cfu/DR数据 --apply
```

备份保存在本次审核目录的 `recovery_<时间>_<标识>/backup/`。`report.json` 和 `summary.json` 是技术报告；带具体文件路径的 `failures_private.csv`、`recovery_private.csv` 仅供服务器本地定位，不需要上传或粘贴。

修复会改变文件大小或修改时间，旧清单拒绝新输入属于正常保护。保留旧清单及旧实验目录，把 `common.json` 的 `split_manifest` 改为一个新路径，例如 `runs/foundation_sft/shared_patient_split_restored.csv`；保持患者/样本成员、标签、划分种子及比例。确定性分组在这些条件相同时会保持原患者分配，并记录修复后的文件信息。

之后运行：

```bash
python run_foundation_experiments.py --check-only
python run_foundation_experiments.py
```

只有前一条全量数据和模型检查通过后，才执行后一条完整实验。统一入口自动测试和汇总，不需要再手动评估。新旧数据指纹不同的实验不混入同一比较报告。

## 6. 验证证据与后续接续方式

- 云端 Python 3.12/Torch 2.9.1 CPU 与 Python 3.10/Torch 2.0.1 CPU 两套环境，各通过全部 149 项 unittest。
- 与用户一致的 pydicom 2.4.4 下，全部 23 项数据审核/恢复测试通过。
- 实际 CLI 子进程使用 22 张合成影像验证：定位 1 张截断文件、只读计划、带备份恢复、旧清单不变、新清单患者分配相同，随后完成 LP/LoRA 训练、自动测试和比较报告。另验证损坏数据会阻止模型计算和训练。
- 这些云端检查使用合成数据/权重；真实医学权重的 10 项 GPU smoke 通过证据来自用户服务器日志。真实数据恢复及正式训练完成仍未确认。
- 结果目录为 `runs/foundation_sft/suites/suite_<时间>_<标识>/`。`suite.json` 记录状态与准确运行目录，`logs/` 保存报错，`data_audit/` 保存审核，`report/` 保存汇总 CSV 和患者 ROC/PR 图。RadJEPA 单次训练结果使用独立输出根目录。

新对话可以发送：“请先阅读仓库 `foundation_sft/PROJECT_HANDOFF.md`，继续检查 DICOM 恢复和五模型实验”，并附上最新服务器终端输出。文档保存进度，但不会自动清空或缩短当前对话；新开对话后可据此接续。
