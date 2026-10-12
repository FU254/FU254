# 六个胸片基础模型的监督训练

当前进度、服务器路径和下一步操作见 [项目交接文档](PROJECT_HANDOFF.md)。

提供 EVA-X、RAD-DINO、RayDINO、CheXFound、CXformer 和 RadJEPA 的统一入口。每个模型均支持真正的线性探测（`linear_probe`）以及注意力 LoRA 加分类头（`lora`），用于预测 Excel 中的“阻塞性通气障碍”。代码可以在本地 PyCharm 编辑，通过 SSH 解释器使用服务器的数据、权重和 GPU。

原 `LoRA-分类/base-1.py` 保留。新框架独立放在本目录，各模型通过适配器保留自己的输入尺寸、归一化和注意力结构。

## 先运行已有的 EVA-X

以下命令在 SSH 服务器终端中执行。先进入**包含 `foundation_sft` 文件夹的项目根目录**，然后：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r foundation_sft/requirements-common.txt
```

这里沿用你已经验证成功的 PyTorch `2.0.1+cu118` 和 xformers `0.0.21`。依赖文件不要求重装 Torch、CUDA 或 xformers；NumPy 限制在 1.x。不要直接用原仓库的整套旧 `requirements.txt` 更新这个环境。

打开两个配置文件：

- `configs/common.json`：核对 `data.excel_path`、`data.dicom_dir`、标签列和患者 ID 列。当前默认路径取自你的 `base` 代码，默认 GPU 是 `cuda:3`。
- `configs/eva_x.json`：把 `checkpoint` 改成服务器上 `eva_x_base_patch16_merged520k_mim.pt` 的实际路径。默认相对路径表示权重放在项目根目录。

按顺序执行：

```bash
# 只检查数据匹配并建立公共患者分组，不需要加载模型权重
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action prepare

# 解码全部训练/验证/测试影像；失败时先修复数据，不加载模型权重
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action audit-data

# 用真实数据和本地权重检查前向、反向、冻结参数及保存重载
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action smoke --device cuda:3

# 线性探测：只训练一个线性分类层
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --mode linear_probe --device cuda:3

# LoRA：训练注意力适配器、输出 LayerNorm 和分类头
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --mode lora --device cuda:3
```

前一项成功后再执行下一项。`prepare` 输出匹配数量、患者来源和各组数量，不打印患者 ID。`smoke` 输出 `smoke_passed: true` 才表示**该模型在当前服务器、权重和数据下**通过了计算检查。

所有路径相对于运行时的项目根目录；`base_config` 路径相对于它所在的 JSON 文件。也可以在 JSON 中填写服务器的绝对路径。

## PyCharm 怎么配置

1. 在 PyCharm 的项目根目录保留整个 `foundation_sft` 文件夹，并将新增代码同步到服务器。
2. 使用项目现有 SSH 解释器 `/opt/anaconda3/envs/evax_lora/bin/python`。
3. 打开 **Run → Edit Configurations → 新建 Python**，运行目标选择 **Module name**，填 `foundation_sft.train`。
4. **Parameters** 先填 `--config foundation_sft/configs/eva_x.json --action prepare`。
5. **Working directory** 选择服务器上包含 `foundation_sft` 的项目根目录。
6. **Environment variables** 设置 `LD_LIBRARY_PATH=`（空值）。终端执行的 `unset` 不会自动修改 PyCharm 的运行配置。
7. 点击运行；数据匹配成功后，把参数中的 `--action prepare` 改成 `--action audit-data`。全部像素检查通过后，再运行 `--action smoke`，之后运行 `--action train --mode linear_probe` 或 `--action train --mode lora`。

你仍然可以在 PyCharm 中修改代码。Git 用于同步和保存版本；数据和预训练权重继续留在服务器上。

## 五个可直接运行的训练入口

仓库根目录新增五个入口，均调用 `foundation_sft.train.main`，自动以项目根目录作为工作目录。可在 PyCharm 中直接运行这些根目录脚本；五个文件和整个 `foundation_sft` 一起上传到 `/home/cfu/EVA-X-main`。同步时保留自己的 `common.json` 数据路径配置，模型源码和已下载权重继续使用现有目录。

| 模型 | 根目录脚本 | 默认 GPU | Python 环境 |
|---|---|---|---|
| EVA-X Base | `train_eva_x.py` | `cuda:0` | `evax_lora` |
| RAD-DINO | `train_rad_dino.py` | `cuda:1` | `evax_lora` |
| CXformer Base | `train_cxformer.py` | `cuda:2` | `evax_lora` |
| CheXFound | `train_chexfound.py` | `cuda:3` | `evax_lora` |
| RadJEPA | `train_radjepa.py` | `cuda:4` | `radjepa_sft` |

入口默认训练 LoRA。先执行全量数据审核和 `--action smoke`，通过后再正式训练。用户已提供五个模型两种模式共 10 项真实权重/GPU smoke 全部通过的日志，但随后的正式训练遇到了被截断的 DICOM，仍需修复数据后重新检查。PyCharm 解释器和空 `LD_LIBRARY_PATH` 的设置同上。GPU 编号可用 `--device` 覆盖，启动前用 `nvidia-smi` 查看可用卡；示例编号不代表服务器当前空闲状态。

```bash
# 在对应环境中运行，可分别在独立终端选择不同模型。
python train_chexfound.py --action smoke
python train_chexfound.py
python train_chexfound.py --mode linear_probe

# 后置参数覆盖入口默认值，也可检查 CLI 用法。
python train_eva_x.py --device cuda:3 --action smoke
python train_radjepa.py --help
```

RadJEPA 入口选用 `radjepa_imagenet.json` 中明确记录的研究预处理，不宣称已核实为作者预训练规范；环境准备和官方规范获取方法见 [MODEL_SOURCES.md](MODEL_SOURCES.md)。RayDINO 尚未确认取得权重，仍可通过统一入口和 `configs/ray_dino.json` 使用，未纳入这五个便捷入口。

## 五个模型的完整实验流程

单个 `train_模型.py` 不只训练：它已经自动进行数据检查、训练/验证、验证集选最佳参数和阈值、最后测试，并保存预测、指标、置信区间及曲线。根目录 `run_foundation_experiments.py` 再将这些步骤串成五模型实验，自动汇总结果。

完整顺序是：读取同一份原始 DICOM/Excel → 建立或校验共同患者划分 → 每个环境解码全部影像 → 所有选中模型/模式完成 smoke → 不同 GPU 并行训练，同卡任务顺序执行 → 各实验自动测试 → 汇总全部预定实验。数据审核或某个 smoke 失败会阻止所有正式训练；正式训练某项失败，其他任务的结果仍会保存，汇总表注明失败项，命令退出非零。

先同步**完整 `foundation_sft` 和根目录 `run_foundation_experiments.py`**，保留已修改的数据路径、权重路径和外部源码。检查 `configs/experiments.json` 的环境名和 GPU 编号。默认前四个模型用 `evax_lora`，RadJEPA 用 `radjepa_sft`，依次分配 `cuda:0` 至 `cuda:4`。这些编号需根据 `nvidia-smi` 的空闲卡修改；同一编号的任务会自动排队。

默认运行五模型的线性探测和 LoRA，共 10 组，训练种子 42、固定划分种子 42。可通过参数改为仅 LoRA 或预先指定多个种子。RadJEPA 的研究归一化设置会在对比表和图中注明。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main

# 查看计划，不加载权重、不训练。
python run_foundation_experiments.py --plan

# 先检查所有模型的真实数据/权重/GPU，成功状态为 checked。
python run_foundation_experiments.py --check-only

# 完整实验：默认五模型 × 线性探测和 LoRA，共 10 组。
python run_foundation_experiments.py

# 也可以先完成当前五模型 LoRA 实验，共 5 组。
python run_foundation_experiments.py --modes lora

# 只运行某个模型的完整流程；仍使用共同数据划分。
python run_foundation_experiments.py --models chexfound --modes lora

# 预先指定重复训练的种子；划分种子和测试集保持相同。
python run_foundation_experiments.py --seeds 42 43 44
```

运行多种子或两种训练方式时，先确定方案再开始，不能根据测试指标选择较好的种子。每次调用都会产生新的实验目录；`--check-only` 后正式运行会再次 smoke。运行中不修改配置，程序已保存每项合并配置的快照。每个子进程会清除冲突的 `LD_LIBRARY_PATH`，使用本地权重离线运行，不安装依赖或下载模型。

程序打印 `Suite directory`，形式为 `runs/foundation_sft/suites/suite_<时间>_<标识>/`。每项的真实训练目录保留在 `suite.json`；RadJEPA 的训练文件继续使用其独立结果根目录。无需再手动执行 `evaluate`，训练已经自动测试。

| 文件或目录 | 内容 |
|---|---|
| `suite.json` | 每项模型/模式/种子、环境、GPU、准确运行目录及成功/失败状态 |
| `configs/` | 各任务的完整配置快照 |
| `data_audit/` | 每个环境的全量像素审核报告；失败文件的本机路径单独保存 |
| `logs/` | prepare、各环境 audit-data、smoke、train 的输出和报错，及报告日志 |
| `report/run_metrics.csv` | 逐次患者测试指标、固定阈值、95% bootstrap CI |
| `report/model_mode_summary.csv` | 按模型、模式和实验规范汇总；多种子为均值和样本标准差 |
| `report/patient_roc_<模式>.png`、`patient_pr_<模式>.png` | 同模式所有指定实验的患者 ROC/PR 对比图 |
| `report/excluded_jobs.csv`、`README.md`、`protocols.json` | 未完成项和原因、中文说明、权重/预处理/训练规范 |

终端打印各阶段状态；失败时会附上对应日志最后 12 行，显示具体异常。训练每轮进展写入日志。用实际打印的路径查看，例如 `tail -f <suite目录>/logs/chexfound_lora_seed42_train.log`。单模型训练目录还保存独立的 ROC/PR/校准图和 `history.json`。

## 避免重复读取：可选数据缓存

数据未改变时，可以复用文件头匹配、成功的完整像素审核和训练所需的确定性图像，减少重跑与每轮训练的 DICOM 读取。原命令默认不启用缓存；启动统一实验时指定一个缓存目录：

```bash
python run_foundation_experiments.py --cache-dir runs/foundation_sft/data_cache
```

首次在各环境中完整审核，并逐步缓存各模型尺寸的图像。后续同一环境的同一批文件复用成功审核，终端打印 `[audit-data] Reused successful pixel audit`；每个模型的线性探测和 LoRA 可以共享相同预处理的图像。缓存命中仍扫描文件清单、核对文件状态及 Excel 内容，但不重复解析全部 DICOM 头和像素。旧版未记录环境指纹的审核报告不会自动作为缓存使用。模型 smoke 仍逐项执行。

图像缓存为单通道 float32 NPY，保存原有解码、窗宽窗位、灰阶方向及模型缩放后的精确结果。随机旋转、翻转和各模型均值方差归一化仍在每次读取时执行；缓存不改变标签、患者分组或随机增强。仅缓存模型需要的尺寸，不保存巨大的原始分辨率 float32 数组。1995 张 224×224 图像约占 0.4 GB，518×518 约占 2.1 GB；不同缩放规范、解码环境分别占用缓存，实际容量随模型配置增加。建议选择有足够空间的本地 SSD 目录。

文件新增、删除、替换或状态变化，Excel/数据配置、预处理代码、Python 环境或解码依赖变化会使相关缓存失效。源文件读取期间变化仍会阻止使用。缓存是可删除的本地生成数据，目录和文件使用私有权限；缓存损坏会重新生成。修改标签或患者信息后，原有冻结划分校验仍可能拒绝新输入，需要按原有流程处理。

需要重新解码全部影像时执行：

```bash
python run_foundation_experiments.py --cache-dir runs/foundation_sft/data_cache --force-audit
```

单模型入口也支持 `--cache-dir` 和 `--force-audit`，或在本地 JSON 中设置顶层 `cache_dir`。保持实验过程中源文件、依赖和代码不变；当前已启动的进程不会自动启用缓存。

## 全量像素检查与原图恢复

`prepare` 检查 DICOM 头与标签，`smoke` 只检查一个小批次，二者无法证明所有影像完整。统一实验流程现在先在每个不同 Conda 环境里解码全部训练、验证、测试影像；任一失败都会在加载模型、正式训练之前阻止实验。手动检查也可以使用下面的命令，审核目录需使用一个尚未存在的新名字：

```bash
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action audit-data --audit-output runs/foundation_sft/audit_before_restore_01
```

数据有问题时退出非零属于预期结果，完整检查仍会记录所有失败。`report.json` 只包含分组内行号、标准传输语法、像素参数、实际/声明字节数及已清理的技术原因；`failures_private.csv` 另外保存具体文件路径，权限为仅当前用户读写，留在服务器本地，不需要粘贴或上传。

当前已确认的一张影像为 3032×3032、16 位未压缩数据，PixelData 声明 18,386,048 字节、实际仅 5,419,848 字节。这是文件被截断。原始胸片为 2000 多像素并不影响训练，预处理会变换为各模型的输入尺寸；修改图像头、补零、换解码器均不能恢复丢失的真实像素。

另一个目录 `/data/users/cfu/DR数据` 是候选原图来源，需要先验证。停止或等待当前旧实验结束后运行检查；恢复前保持源目录和目标文件不变。下面第一条只生成恢复计划：

```bash
python -m foundation_sft.recover_data --audit-dir runs/foundation_sft/audit_before_restore_01 --source-dir /data/users/cfu/DR数据
```

工具递归检查来源目录，按 SOPInstanceUID 匹配，文件名无需相同。它核对患者/检查标识、像素布局、灰阶和窗口元数据，验证完整解码；对于被截断的 PixelData，还核对原图与损坏文件中保留的像素字节一致。原图有歧义、不完整、无法解码或身份不匹配时，结果为 `unresolved`，不会替换。`counts.ready` 表示找到可恢复的影像，`already_valid` 表示已由其他方式修好。

核对计划后，可在同一条命令末尾添加 `--apply` 恢复已验证文件：

```bash
python -m foundation_sft.recover_data --audit-dir runs/foundation_sft/audit_before_restore_01 --source-dir /data/users/cfu/DR数据 --apply
```

每个旧文件都会先保留在本次 `recovery_<时间>_<标识>/backup/`，新文件在目标目录内通过临时副本核对 SHA-256、影像身份、残存像素及解码结果后原子替换。复制期间源/目标变化会拒绝替换。技术汇总为 `summary.json`，具体来源与备份路径仅保存在权限受限的 `recovery_private.csv`。不会删除影像、修改标签或改写旧患者划分。若有 `unresolved` 或 `not_eligible`，先处理对应原因。

恢复会改变文件大小或修改时间，旧划分清单将正确拒绝这些输入变化。保留旧清单和旧实验目录，把 `configs/common.json` 中的 `split_manifest` 改为一个新路径，例如 `runs/foundation_sft/shared_patient_split_restored.csv`；保持 `split_seed=42` 和分组比例。患者、标签和样本成员保持一致时，原有确定性分组算法会给出相同的患者分配，同时记录修复后的文件信息。所有模型都继承这一个共同清单。然后执行 `python run_foundation_experiments.py --check-only`，全量数据和 10 项 smoke 通过后，再执行 `python run_foundation_experiments.py`。新旧输入指纹不同的实验不混入同一比较报告。

已有实验不必重训。将其**事先确定的**完整运行目录显式列出，即可单独汇总；程序会核对共同划分和相同测试患者/标签，不会自动寻找最高测试分。输出目录必须是新目录。

```bash
python -m foundation_sft.report --runs <EVA-X运行目录> <CheXFound运行目录> --output runs/comparison_manual_01
```

中断训练沿用下文的 `--resume` 和准确 `last.pt` 路径。重新运行 suite 会建立新实验，不会自动继续旧任务或跳过已完成任务。

## 其他五个模型

权重和依赖准备方法见 [MODEL_SOURCES.md](MODEL_SOURCES.md)。目前你已有 EVA-X Base，已在 Windows 电脑下载 RAD-DINO、CXformer Base、RadJEPA 的官方完整目录；还需要上传服务器并逐个验证。准备完成后，替换命令中的配置文件即可：

| 模型 | 配置 | 必需的本地资产 |
|---|---|---|
| EVA-X Base | `configs/eva_x.json` | EVA-X Base `.pt` |
| RAD-DINO | `configs/rad_dino.json` | 官方完整 Hugging Face 模型目录 |
| RayDINO / XRay-DINO | `configs/ray_dino.json` | 官方 DINOv2 源码、RayDINO `.pth`、核实后的归一化参数 |
| CheXFound | `configs/chexfound.json` | 官方源码及 YAML、ViT-L/16 teacher 权重 |
| CXformer Base | `configs/cxformer.json` | 官方完整 Hugging Face 模型目录；也支持 Small |
| RadJEPA | `configs/radjepa.json` | 官方完整目录（`jepa_encoder.pth.tar` 和源码）；核实后的归一化均值、标准差 |

六个适配器都已经实现。缺少权重、预处理参数或不匹配的编码器参数会报错。训练入口不会自动下载模型，也不会用随机权重继续正式训练。

本地目录 `D:\99\EVA-X-main\foundation_weights` 通过 PyCharm 的 **Deployment → Upload** 上传到 `/home/cfu/EVA-X-main/foundation_weights`。同步代码时同时上传整个 `foundation_sft`，避免只更新 `train.py` 而漏掉适配器和公共预处理代码。RAD-DINO 与 CXformer 可沿用 `evax_lora`，先分别运行 `--action smoke`；CXformer 现在会按发布配置执行直方图均衡化。

RadJEPA 的官方自定义 `from_pretrained` 总会联网；新适配器直接读取已下载的 `.pth.tar`，不调用该方法。其输入尺寸已确认是 224，作者发布未给出归一化均值、标准差，默认仍留空并明确报错。若先采用明确的研究设置做计算验证和探索训练，可选择 `configs/radjepa_imagenet.json`，使用官方 I-JEPA 上游默认的 ImageNet 统计值，并记录为本研究选择，**不宣称已核实为 RadJEPA 作者的预训练规范**。该配置使用独立结果目录；详见来源说明中的依据及作者规范获取入口。RadJEPA 的 timm 依赖与 EVA-X 不同，按来源说明使用单独环境。

默认 `common.json` 设置微批次 8、累积 8，约相当于有效批次 64。大模型高分辨率输入可能需要更小微批次；显存不足时可改为 `batch_size: 1`、`accumulation_steps: 64`，其余设置保持一致。最后一个不完整累积窗口按实际样本数归一化。

代码已支持不同 GPU 上的独立进程，所有进程复用同一患者分组。可分别启动：

```bash
# 两条命令分别在两个终端运行，并先确认 GPU 没有被其他任务占用
python -m foundation_sft.train --config foundation_sft/configs/rad_dino.json --device cuda:0 --mode linear_probe
python -m foundation_sft.train --config foundation_sft/configs/cxformer.json --device cuda:1 --mode linear_probe
```

修改运行种子用 `--seed 43`。比较不同种子时保留 `split_seed` 和 `split_manifest`，确保数据分组一致。

## AMP 梯度累计与溢出处理

统一训练循环现在在反向传播前按窗口固定分母缩小 loss，窗口结束后按实际样本数校正梯度，因此不完整窗口和不等长微批次仍对应正确的样本平均损失。这样避免在半精度反向传播中先累计过大的样本损失，再于反向结束后归一化。

启用 AMP 时，若 `GradScaler.unscale_` 检测到梯度溢出，程序不裁剪该窗口的无效梯度，由 scaler 跳过更新、降低 loss scale，并清空窗口梯度。日志中的 `[amp] Skipped overflowing accumulation window` 记录这一过程。有限梯度仍使用严格裁剪；非有限 loss、非 AMP 的非有限梯度、有限梯度产生的非有限总范数仍会报错。若连续 16 个窗口溢出，或整轮没有一次成功更新，训练停止，需在独立诊断中检查完整精度计算。

原 `smoke` 验证的是 FP32 的前后向、冻结参数及保存重载，没有覆盖完整训练的 AMP 和梯度累计路径。五模型的 smoke 全部通过后，仍应查看正式训练日志确认 epoch 正常完成。不要通过关闭非有限梯度检查来继续训练。修复的半精度数值回归测试使用合成样本和真实 CPU GradScaler；医学权重的 CUDA 训练需在服务器验证。

### 单项训练的非有限 loss 与 FP32 恢复

`Non-finite training loss` 表示前向计算的 loss 已出现 NaN/Inf，降低 GradScaler 的反向缩放不能修复这种前向错误。若上一轮 FP32 验证正常，而 AMP 训练失败，应检查输入和检查点；半精度前向溢出是可能原因，不能仅凭日志断言。原严格检查仍保留，恢复后遇到非有限 loss 或梯度也会停止。

允许用 `--resume-full-precision` 明确将 **AMP 检查点转为 FP32 继续训练**。它只允许 `amp: true → false`，其他训练设置、种子、预训练权重及分组校验保持有效。优化器、调度器、随机状态、已完成轮数、历史和原最佳模型保留；原 AMP scaler 停用。新运行保存独立的 `config.json` 和 `resume.json`，记录源检查点 SHA256、精度变更和接续轮数。精度变更前的历史仍属于原 AMP 训练，这不是从第一轮开始的纯 FP32 实验。无需覆盖或改写原检查点。

如果是套件中只有一项失败，可在项目根目录执行：

```bash
python -m foundation_sft.recover_experiment \
  --suite runs/foundation_sft/suites/实际套件目录 \
  --job cxformer_lora_seed42 --resume-full-precision
```

查看已经完成的各模型患者指标和当前任务状态，使用只读入口（套件仍在运行时也可查看）：

```bash
python -m foundation_sft.recover_experiment \
  --suite runs/foundation_sft/suites/实际套件目录 --show-metrics
```

输出最佳验证 ROC-AUC、患者测试 ROC-AUC 及其 95% CI、梯形 PR-AUC、AP、敏感度和特异度。验证指标和测试指标分别注明；仅指定种子 42 时，患者 bootstrap CI 不代表跨种子训练波动。

辅助入口从失败任务日志定位准确的 `last.pt`，进入原 Conda 环境恢复，实时显示训练日志，保存独立恢复目录和成功回执，不修改原 `suite.json`。它拒绝恢复非失败任务和同 GPU 仍有训练任务的情形；同一失败项并发恢复由独立锁阻止。确认所用 GPU 的可用显存足以运行 FP32。

等原套件的其他任务结束，再使用恢复命令打印的准确恢复目录：

```bash
python -m foundation_sft.recover_experiment \
  --suite runs/foundation_sft/suites/实际套件目录 \
  --job cxformer_lora_seed42 --report-only \
  --recovery-dir runs/foundation_sft/suites/实际套件目录/实际恢复目录
```

报告仅使用原套件的其他完成项与明确指定的恢复项。原失败记录和原部分报告保留，新报告输出到独立目录；完整样本/标签来源/分组指纹需一致，且原计划的每项模型、模式、种子都必须有一个成功结果。恢复记录不应通过比较测试成绩挑选。

也可直接调用训练入口：

```bash
python -m foundation_sft.train --config 原配置快照.json \
  --action train --resume 失败运行目录/last.pt --resume-full-precision
```

普通 `--resume` 仍要求训练设置完全一致。FP32 恢复后若再次正常中断，应使用新运行的 `config.json` 和 `last.pt` 做普通 `--resume`。

## 修复服务器第一次统一检查的错误

用户的第一次五模型检查已完成数据准备并通过 CheXFound，另外四项失败。实际日志分别是 EVA-X 的 NumPy 标量元数据不被 Torch `weights_only` 支持，RAD-DINO/CXformer 的 Transformers 4.50.3 访问 Torch 2.0.1 不存在的 `torch.compiler`，以及尚未创建 `radjepa_sft`。这些是三项不同的问题。

先同步新版 `models/common.py` 和 `requirements-hf.txt` 到服务器对应的 `foundation_sft` 目录。保留自己的 JSON 配置、数据划分、模型源码和权重；EVA-X 加载兼容代码不需要改写权重文件。随后按顺序执行：

```bash
cd /home/cfu/EVA-X-main
conda activate evax_lora
unset LD_LIBRARY_PATH
python -m pip install -r foundation_sft/requirements-hf.txt
python -m pip check
```

新的 HF 固定组合为 Transformers 4.48.3 / hub 0.29.3 / tokenizers 0.21.1 / safetensors 0.4.5。只调整上述 HF 包，沿用已可用的 Torch 2.0.1、CUDA、torchvision 和 xformers；不要为此更新整套 Torch 环境。该组合已在独立 Python 3.10 / Torch 2.0.1 CPU 环境实际通过导入、LP/LoRA、本地权重加载和完整已有测试；远端 GPU 仍需下述检查。

如果 `radjepa_sft` 不存在，从修好依赖的 `evax_lora` 克隆后，只在新环境中更新 timm：

```bash
conda create --name radjepa_sft --clone evax_lora -y
conda activate radjepa_sft
unset LD_LIBRARY_PATH
python -m pip install -r foundation_sft/requirements-hf.txt
python -m pip install --no-deps "timm==1.0.24"
python -m pip check
```

若该环境已经创建，跳过 `conda create`，仍检查其 HF 依赖和 timm。然后回到主环境，执行已确定的五模型 × 两种方式方案：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python run_foundation_experiments.py --check-only
# 上一项全部检查通过后，运行正式 10 组实验；不加 --modes lora。
python run_foundation_experiments.py
```

重新检查会生成新 suite，不会覆盖旧失败记录或改动患者划分。仍有错误时查看新 suite 的对应 `logs/`，按实际异常处理。

后续服务器检查中，RAD-DINO、CXformer、CheXFound 的两种方式均已通过；EVA-X 在 NumPy 兼容路径中进一步遇到 `__builtin__.set`。加载代码现已补充普通 set 和协议 2/3/4 支持；未知应用类仍拒绝。RadJEPA 已加载权重并完成前后向，重载后 GPU 预测相差约 `4.47e-8`，被旧的零容差输出检查误判。更新 `models/common.py`、`train.py` 及 `workflow.py` 后，可先检查剩余两个模型，新版入口会直接显示失败日志的末尾异常：

```bash
python run_foundation_experiments.py --models eva_x radjepa --check-only
```

重载验证先严格比较保存前后每个可训练参数的键、形状、dtype 和数值，再比较有限的预测值。CPU 保留零容差；CUDA FP32 输出使用 `rtol=1e-5`、`atol=1e-6`，并在 `reload_check` 中记录实际最大绝对误差和使用的容差。参数不一致、非有限输出或超出容差的预测仍会失败，不改变模型计算或训练设置。全部检查通过后再运行默认 10 组完整实验。

最新服务器结果中，RadJEPA 两种方式也已通过，只剩 EVA-X。EVA-X 已读入真实 MIM 权重，但构造时采用分类 `fc_norm`，产生 `missing fc_norm` 与 `unexpected norm/mask_token/lm_head`。新版 `models/eva_x.py` 会按权重的 `norm` 或 `fc_norm` 参数对选择正确结构；保留预训练归一化，精确忽略不用的 MIM/旧分类头，继续严格核对全部编码器参数。同步该文件后执行：

```bash
python run_foundation_experiments.py --models eva_x --check-only
# EVA-X 两种方式均通过后，启动五模型×两种方式的完整实验。
python run_foundation_experiments.py
```

## 数据和实验约定

- Excel 的“放射编号”与 DICOM 的 `AccessionNumber` 匹配，“阻塞性通气障碍”必须为 0 或 1。未匹配到 Excel 的影像跳过，避免沿用上一张影像的标签。
- `patient_column: "auto"` 按“住院号、患者ID、病人ID、PatientID”查找；没有可用列时尝试 DICOM `PatientID`。同一批数据须使用同一个 ID 来源。若部分 Excel ID 缺失而混入 DICOM ID，程序会要求明确共同的患者映射；可以核实 DICOM ID 后设 `patient_id_source: "dicom"`。
- 必须确认所选 ID 能跨不同检查识别同一人；若“住院号”只表示一次住院，应在配置中明确填写稳定的患者 ID 列。程序不会把放射编号当成患者 ID。姓名默认不用作 ID。
- 首次生成 `runs/foundation_sft/shared_patient_split.csv`，按患者分层划分 70%/10%/20%，同一患者不会跨组。后续模型复用并校验这个清单；数据、标签、患者映射、文件信息或分组设置变化时会报错。确需新实验时填写新的清单路径，避免覆盖旧实验。
- 从原始 DICOM 按需读取，处理斜率、截距、灰阶方向、窗宽窗位或分位数；三个通道共享灰阶内容。不会复用旧脚本的 224 像素缓存。训练默认旋转 ±10°、不左右翻转，验证和测试不增强。
- 当前框架直接使用现有 Excel 标签。**尚未根据肺功能数值重新计算 LLN，也未自动核验胸片与肺功能间隔 ≤7 天或筛选每位患者一次合格检查。** 正式实验前，需按开题方案在原表和影像目录中完成这些纳入条件，并在 `label_definition` 记录参考方程、舒张试验状态及筛选方法。
- 同一患者若出现冲突标签，程序报错，须先按确定的检查配对规则整理。若保留同次检查的多幅影像，患者级评估取预测概率均值。

## 输出、继续训练和评估

训练打印独立的运行目录：`runs/foundation_sft/<model>/<mode>/seed<seed>_<时间>_<随机标识>/`。

| 文件 | 内容 |
|---|---|
| `config.json`、`model_metadata.json`、`environment.json` | 参数、编码器和权重指纹、输入规范、参数数量、设备版本 |
| `split.csv`、`split_metadata.json` | 此次实验使用的公共分组及指纹 |
| `history.json` | 每轮训练损失、验证集患者 ROC-AUC/PR-AUC、学习率 |
| `last.pt` | 最后一轮可训练参数、优化器、调度器、随机状态和历史，用于继续训练 |
| `best.pt`、`selection.json` | 按验证集患者 ROC-AUC 选择的参数及验证集 Youden 阈值 |
| `validation_predictions.csv` | 最佳模型的验证集预测 |
| `test/` | 影像和患者预测、ROC/PR/校准图、指标及患者重采样置信区间 |

编码器原始权重不重复写入 `best.pt`。重载需要原预训练权重、相同预处理和患者分组；程序校验这些信息。测试集在最佳模型和阈值确定后评估，不能据此反复选择训练参数。

```bash
# 将下列路径替换为程序实际打印的运行目录；继续训练要求原训练设置和种子相同
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --mode lora --resume runs/foundation_sft/eva_x/lora/实际运行目录/last.pt

# 用保存的验证集阈值独立重载评估，结果写入新目录
python -m foundation_sft.train --config foundation_sft/configs/eva_x.json --action evaluate --mode lora --checkpoint runs/foundation_sft/eva_x/lora/实际运行目录/best.pt
```

指标包括 ROC-AUC、PR-AUC（梯形积分）及 Average Precision、敏感度、特异度、PPV、NPV、准确率、F1 和 Brier，并保存校准图。默认进行 1000 次患者簇 bootstrap，测试阈值固定，不在重采样中重新优化。单一类别的 AUC 返回空值并说明原因。

`model(images, return_features=True)` 返回 `logits`、`global_features [B,D]`、`local_features [B,N,D]` 和 `grid_shape`，可用于后续蒸馏。本次实现仅包含图像监督训练；回归、临床融合、蒸馏训练和部署尚未加入。

新框架默认 EVA-X 取空间 patch 特征均值。MIM 权重保留逐 token 预训练 LayerNorm，再取 patch 均值；分类导出保留原来的均值池化后 `fc_norm`。空间特征与原生 `return_patch_tokens=True` 一致；MIM 分支原生默认全局输出取 CLS，本框架明确选择 patch 均值。旧 `base` 手工池化包含 CLS，并重新初始化 `fc_norm`；`pooling: "legacy_all_tokens_mean"` 仅用于对照包含 CLS 的池化，不等同于重现旧模型全部数值行为。新 LoRA 使用独立的可训练输出 LayerNorm；线性探测只含一个线性层，因此新实验不能当作旧实验的无差别继续训练。

## 已完成的验证

云端 Python 3.12 / PyTorch 2.9.1 CPU 环境通过 149 项 unittest：患者匹配和不可变分组、真实小型 HF 编码器的 LP/LoRA、训练到重载评估、继续训练与连续训练一致、不等微批次的梯度累积、各模型特征和注意力目标、患者 bootstrap、EVA 注意力回退、带非对称窗口的 DICOM 灰阶方向，以及实际 CXformer 预处理与 OpenCV＋BiT 的对照、RadJEPA 本地 `.pth.tar` 严格加载和联网方法不会被调用。覆盖统一流程的检查屏障、同卡排队与跨卡并行、子进程失败和成功回执、完整测试产物核对，以及比较报告的共同测试患者/标签校验、置信区间和跨种子统计。

为修复真实服务器日志中的兼容错误，还在独立 Python 3.10.22 / Torch 2.0.1+cpu / torchvision 0.15.2+cpu / timm 0.4.12 / Transformers 4.48.3 环境通过相同的全部 149 项测试，无跳过。15 项 checkpoint 测试覆盖真实 Tensor + NumPy 标量 + set 元数据、协议 2/3/4 加载，以及不支持的对象、嵌套加载和扩展 opcode 拒绝；不会修改 Torch 全局。工作流还验证失败日志末尾异常可见及读取日志失败不改变原任务状态。完整 CXformer-small 也以合成权重完成本地严格加载、LoRA 反向和冻结检查。云端不含用户的医学预训练文件或 A6000；用户服务器日志已确认五模型两模式的 smoke 通过，正式训练尚因数据损坏受阻。

新增 10 项全量像素审核和 13 项恢复测试覆盖真实物理截断、声明长度不足即使仍能解码也拒绝、各分组完整扫描、异常技术信息清理、不可变清单、只读恢复计划、改名的原图、身份和残存像素校验、歧义副本拒绝、原子替换与旧文件备份，以及复制过程中源/目标改变时拒绝覆盖。在与用户相同的 pydicom 2.4.4 下，这 23 项数据测试也全部通过。统一流程另有 3 项全量审核屏障测试，验证同环境只审核一次、失败时所有模型均停止、无效成功回执不会放行。

统一流程还用合成 DICOM、真实小型 DINOv2 编码器和实际子进程，跑通 prepare → 两种模式 smoke → LP/LoRA 训练及自动测试 → 汇总 CSV 和 ROC/PR 图。测试中将 Conda 启动器替换为隔离 CPU Python 路由；这验证了软件流程，没有验证远端 Conda 环境或五个模型的医学权重。

恢复流程另用实际 CLI 子进程验证：扫描 22 张合成影像，定位 1 张物理截断的文件，只读计划保留原文件；应用恢复保留损坏备份，旧划分 CSV 保持字节不变。使用新清单重新准备后，所有患者/样本/标签的分配与修复前相同，再完成两种模式的训练、自动测试和比较报告。另一个实际工作流检查确认未恢复的数据会阻止任何模型 smoke 或训练启动。

8 项重载校验测试覆盖近零 FP32 logits 的 `4.47e-8` 差异允许通过、显著偏差及 NaN/Inf 拒绝、可训练参数微小漂移和键/形状/dtype 不一致拒绝，以及 CPU 输出仍采用零容差。

10 项 EVA-X checkpoint 回归验证 MIM `norm` 与分类 `fc_norm` 的严格结构选择、原生空间特征、非平凡归一化参数、已知头精确移除、LP/LoRA 梯度和保存重载、错误归一化或编码器参数拒绝。另外，完整 85,850,880 参数 EVA-X Base / 224 输入架构已在 Torch 2.0.1 CPU 使用合成 MIM checkpoint（含 NumPy 和 set 元数据）完成实际构造与严格加载，验证 36 个 LoRA 投影、14×14 空间网格、冻结参数、两种模式前后向及精确重载。合成 checkpoint 随后删除；这些检查不代表用户医学权重或 GPU 已通过。

另外，RayDINO 官方 `xray_dino_vitl16` 完整 ViT-L/16 架构已用随机初始化和小尺寸合成输入完成 24 层 LoRA 前向、反向、冻结参数及重载一致性检查。这些检查不代表已加载医学预训练权重，也不代表已经在你的 SSH 服务器完成训练。

RadJEPA 已按用户提供的官方模型源码，在隔离的 `timm==1.0.24` 环境构造完整 85,704,960 参数 ViT-B/14、224 输入，并用合成的本地 encoder checkpoint 完成 LP 和 12 层 LoRA 前向、反向、冻结参数及重载一致性检查；检查期间禁止 Hub 和 HTTP 请求。此检查使用明确的测试归一化，不能作为作者预训练归一化参数的依据，也没有验证用户实际下载的医学权重。

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
```

真实权重、DICOM 解码依赖和远端 GPU 路径仍以每个配置的 `prepare`、`smoke` 结果为准。压缩 DICOM 若缺少解码器，应根据其传输语法安装相匹配的 pydicom 解码插件。

患者清单、预测、数据和权重只保存在服务器或本地实验目录，不提交 Git。本仓库 `.gitignore` 已排除默认实验和权重目录。
