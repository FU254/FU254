# 六个胸片基础模型的监督训练

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
7. 点击运行；数据检查成功后，把参数中的 `--action prepare` 改成 `--action smoke`，之后再改成 `--action train --mode linear_probe` 或 `--action train --mode lora`。

你仍然可以在 PyCharm 中修改代码。Git 用于同步和保存版本；数据和预训练权重继续留在服务器上。

## 其他五个模型

权重和依赖准备方法见 [MODEL_SOURCES.md](MODEL_SOURCES.md)。目前只确认了你已有 EVA-X Base，其他模型的权重需要逐一核对或下载。准备完成后，替换命令中的配置文件即可：

| 模型 | 配置 | 必需的本地资产 |
|---|---|---|
| EVA-X Base | `configs/eva_x.json` | EVA-X Base `.pt` |
| RAD-DINO | `configs/rad_dino.json` | 官方完整 Hugging Face 模型目录 |
| RayDINO / XRay-DINO | `configs/ray_dino.json` | 官方 DINOv2 源码、RayDINO `.pth`、核实后的归一化参数 |
| CheXFound | `configs/chexfound.json` | 官方源码及 YAML、ViT-L/16 teacher 权重 |
| CXformer Base | `configs/cxformer.json` | 官方完整 Hugging Face 模型目录；也支持 Small |
| RadJEPA | `configs/radjepa.json` | 官方完整模型目录和自定义源码；完整预处理配置 |

六个适配器都已经实现。缺少权重、预处理参数或不匹配的编码器参数会报错。训练入口不会自动下载模型，也不会用随机权重继续正式训练。

默认 `common.json` 设置微批次 8、累积 8，约相当于有效批次 64。大模型高分辨率输入可能需要更小微批次；显存不足时可改为 `batch_size: 1`、`accumulation_steps: 64`，其余设置保持一致。最后一个不完整累积窗口按实际样本数归一化。

代码已支持不同 GPU 上的独立进程，所有进程复用同一患者分组。可分别启动：

```bash
# 两条命令分别在两个终端运行，并先确认 GPU 没有被其他任务占用
python -m foundation_sft.train --config foundation_sft/configs/rad_dino.json --device cuda:0 --mode linear_probe
python -m foundation_sft.train --config foundation_sft/configs/cxformer.json --device cuda:1 --mode linear_probe
```

修改运行种子用 `--seed 43`。比较不同种子时保留 `split_seed` 和 `split_manifest`，确保数据分组一致。

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

新框架默认 EVA-X 取 patch token 均值，与编码器原生特征接口一致。旧 `base` 手工池化包含 CLS；如需对照该池化方式，可设置 `pooling: "legacy_all_tokens_mean"`。新 LoRA 使用独立的可训练输出 LayerNorm；线性探测只含一个线性层，因此新实验不能当作旧实验的无差别继续训练。

## 已完成的验证

云端 Python 3.12 / PyTorch 2.9.1 CPU 环境通过 54 项 unittest：患者匹配和不可变分组、真实小型 HF 编码器的 LP/LoRA、训练到重载评估、继续训练与连续训练一致、不等微批次的梯度累积、各模型特征和注意力目标、患者 bootstrap、EVA 注意力回退、带非对称窗口的 DICOM 灰阶方向。

另外，RayDINO 官方 `xray_dino_vitl16` 完整 ViT-L/16 架构已用随机初始化和小尺寸合成输入完成 24 层 LoRA 前向、反向、冻结参数及重载一致性检查。这些检查不代表已加载医学预训练权重，也不代表已经在你的 SSH 服务器完成训练。

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
```

真实权重、DICOM 解码依赖和远端 GPU 路径仍以每个配置的 `prepare`、`smoke` 结果为准。压缩 DICOM 若缺少解码器，应根据其传输语法安装相匹配的 pydicom 解码插件。

患者清单、预测、数据和权重只保存在服务器或本地实验目录，不提交 Git。本仓库 `.gitignore` 已排除默认实验和权重目录。
