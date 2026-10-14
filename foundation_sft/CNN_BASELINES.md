# 三个 CNN 的线性探测基线

新增三个实验：ImageNet DenseNet121、胸片监督预训练 DenseNet121，以及胸片自监督预训练 ResNet18。每个实验都冻结整个编码器，只训练一个线性二分类头（`linear_probe`），预测现有 Excel 的“阻塞性通气障碍”。DenseNet 的原病种分类头和 MoCo 的投影头不参与训练；冻结编码器保持 eval 状态，BatchNorm 统计量也不更新。三个配置仅支持 `linear_probe`，不执行全模型微调或 LoRA。

三个实验继承服务器现有 `configs/common.json` 的数据路径、标签定义、患者映射和共同划分。`common_cnn.json` 只补充 CNN 的线性探测模式、独立结果目录和 `amp: false`；沿用已有训练参数及随机增强，CNN 使用 FP32。既有基础模型和 VLM 的配置与结果继续保留。

## 模型来源与实验角色

权重来源、文件名、校验信息和预处理声明集中保存在 [cnn_sources.py](cnn_sources.py) 的 `CNN_SOURCES`，训练读取本地文件，不自动下载。

| 模型 ID | 预训练来源与对照作用 | 配置 | 本地权重目录 |
|---|---|---|---|
| `densenet121_imagenet` | torchvision `IMAGENET1K_V1`，ImageNet-1K 监督预训练；通用图像基线 | `configs/densenet121_imagenet.json` | `foundation_weights/densenet121_imagenet/` |
| `densenet121_cxr` | TorchXRayVision `densenet121-res224-all`，多个胸片数据集的 18 病种监督预训练；胸片领域基线 | `configs/densenet121_cxr.json` | `foundation_weights/densenet121_cxr/` |
| `resnet18_cxr` | Stanford MoCo-CXR，先 ImageNet 初始化，再在 CheXpert 胸片上自监督预训练；胸片自监督基线 | `configs/resnet18_cxr.json` | `foundation_weights/resnet18_cxr/` |

官方来源：

- ImageNet DenseNet121：[torchvision 权重说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html)，[官方权重](https://download.pytorch.org/models/densenet121-a639ec97.pth)。
- 胸片 DenseNet121：[TorchXRayVision 官方仓库](https://github.com/mlmed/torchxrayvision)，[官方 res224-all 权重](https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt)。
- 胸片 ResNet18：[MoCo-CXR 官方仓库](https://github.com/stanfordmlgroup/MoCo-CXR)，[作者发布的 `r8w-00001.pth.tar`](https://storage.googleapis.com/moco-cxr/r8w-00001.pth.tar)。适配器读取 MoCo 的 `encoder_q` 主干，严格核对主干参数；普通 ImageNet ResNet18 不满足这个实验的权重要求。

ImageNet 和胸片 DenseNet121 的实际官方文件已在云端下载并核对 SHA-256。MoCo-CXR 的来源及作者公开地址已核实，但本云端访问 Google Storage 返回 HTTP 403，未取得其实际文件，尚不能声称已用该官方医学权重验证计算。下载和服务器 smoke 均通过后才能启动包含它的完整三项实验。

## 输入与预处理

所有模型共用本项目的 DICOM 解码、窗宽窗位、灰阶方向及患者分组，再按来源声明执行几何变换和归一化。训练时沿用 `common.json` 的旋转等随机增强；验证与测试使用确定性处理。

| 模型 | 输入 | 几何变换 | 归一化 |
|---|---|---|---|
| ImageNet DenseNet121 | 灰阶复制为三通道，224×224 | 短边缩放到 256，再中心裁剪到 224 | mean `(0.485, 0.456, 0.406)`；std `(0.229, 0.224, 0.225)` |
| 胸片 DenseNet121 | 单通道，224×224 | 短边缩放到 224，再中心裁剪到 224 | 本项目显示强度 `x∈[0,1]` 转为 `2048x−1024`，即约 `[-1024,1024]`；适配器从重复灰阶中取第一通道 |
| 胸片 ResNet18 | 灰阶复制为三通道，320×320 | 短边缩放到 320，再中心裁剪到 320 | 每通道 mean `0.5020`；std `0.085585` |

三者均使用本项目的 bilinear 插值。上述来源参数保留了模型各自的输入规范，但本项目从 DICOM 经共享显示处理生成灰阶输入；作者原始 RGB/skimage 图像读取和缩放流程没有逐像素复现。这一差异会记入模型 metadata，分析结果时需保留该实验规范说明。

## 在 Windows 下载三份权重

在可以联网的电脑上使用新版项目中的 `download_cnn_weights.py`。脚本及来源表只依赖 Python 标准库，下载不需要 Torch、torchvision 或 TorchXRayVision。

在 PowerShell 或 PyCharm 终端执行：

```powershell
cd D:\99\EVA-X-main
python download_cnn_weights.py --output foundation_weights
```

成功时末尾显示 `Failed models: none`。每个目录保存实际权重和 `download_manifest.json`（来源地址、文件大小、SHA-256）；已有文件会重新核对，不覆盖成功下载。ImageNet 和胸片 DenseNet121 必须符合来源表中的固定 SHA-256；MoCo-CXR 作者来源未提供本项目可独立核实的固定校验值，脚本记录下载文件的 SHA-256，服务器继续核对其 MoCo 主干结构。

下载后的新增目录为：

```text
foundation_weights/
├── densenet121_imagenet/
│   └── densenet121-a639ec97.pth
├── densenet121_cxr/
│   └── nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
└── resnet18_cxr/
    └── r8w-00001.pth.tar
```

如果某项下载失败，脚本继续处理其余模型并退出非零。可单独重试该项，例如：

```powershell
python download_cnn_weights.py --output foundation_weights --models resnet18_cxr
```

若联网电脑也收到 MoCo-CXR 的 403，需从作者发布入口取得可用的同一官方检查点；缺失权重时该模型会明确停止。保留已经成功的两份 DenseNet 文件。

通过 PyCharm **Deployment → Upload**，只上传以上三个新增目录至服务器 `/home/cfu/EVA-X-main/foundation_weights/`。保留现有其他模型权重和服务器自己的配置。文件路径相对于运行时项目根目录；需要自定义存放位置时，只修改三个新增模型 JSON 的 `checkpoint`。

## 给服务器现有项目添加 CNN 文件

服务器继续使用 `/home/cfu/EVA-X-main` 和 `evax_lora` 环境，沿用现有 Torch 2.0.1 / torchvision，不新增依赖。胸片 DenseNet121 的旧式序列化文件由本项目的受限数据加载器读取，运行时不要求安装 TorchXRayVision，也不调用其下载方法。

先把新版代码包解压或上传到一个独立的临时目录。下例 `/home/cfu/EVA-X-main/cnn_update` 表示**新版代码包的根目录**，其中应能找到 `update_foundation_cnn.py`、`download_cnn_weights.py` 和 `foundation_sft/`；按实际解压位置替换此路径。

在服务器终端先查看更新计划：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
python /home/cfu/EVA-X-main/cnn_update/update_foundation_cnn.py \
  --source-dir /home/cfu/EVA-X-main/cnn_update \
  --project /home/cfu/EVA-X-main
```

默认只显示计划。核对来源和目标目录后，执行同一命令并添加 `--apply`：

```bash
python /home/cfu/EVA-X-main/cnn_update/update_foundation_cnn.py \
  --source-dir /home/cfu/EVA-X-main/cnn_update \
  --project /home/cfu/EVA-X-main --apply
```

此更新器限定于新增 CNN 配置、来源表、适配器、权重加载辅助、下载脚本和本说明，并为模型注册入口的修改保留备份。现有 `train.py`、数据处理、缓存逻辑、`common.json`、旧模型配置和用户权重保留。不要用新版整套 `configs/` 替换服务器目录；新增 CNN 配置通过 `base_config` 自动继承当前服务器的 `common.json`。

## 运行三个线性探测实验

检查 `configs/experiments_cnn.json`：默认三个模型都使用 `evax_lora`，按上表顺序分别分配 `cuda:0`、`cuda:1`、`cuda:2`，种子 42，只有 `linear_probe` 一种模式。启动前用 `nvidia-smi` 查看空闲卡，再按实际情况调整这个新增套件的 GPU；相同 GPU 的任务会顺序执行。

在服务器项目根目录执行以下命令。示例复用已有缓存目录；如果此前使用其他缓存路径，沿用其实际位置。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main

# 查看计划：应只有三个 linear_probe 任务，不加载模型权重。
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_cnn.json --plan

# 全量数据审核及三项真实权重/GPU smoke。
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_cnn.json \
  --cache-dir runs/foundation_sft/data_cache --check-only

# 上一项全部成功后，运行三项完整实验。
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_cnn.json \
  --cache-dir runs/foundation_sft/data_cache
```

缓存仍按环境、文件状态、数据配置和预处理规范校验；不同尺寸或几何规范分别生成缓存。已有基础模型进程和已启动实验不会自动改变。新套件复用共同患者清单；如现有清单因原图恢复而已更换，继续使用服务器 `common.json` 中的新路径。

`--check-only` 成功状态为 `checked`；正式运行会再次检查。全量数据审核或任一选中模型的 smoke 失败，会阻止选中任务的正式训练。训练自动执行验证集选模和阈值选择，再评估测试集，保存患者指标、置信区间、曲线和比较报告，无需手工追加 `evaluate`。

尚未取得 ResNet18 权重时，可以先检查已下载的两个 DenseNet：

```bash
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_cnn.json \
  --models densenet121_imagenet densenet121_cxr \
  --cache-dir runs/foundation_sft/data_cache --check-only
```

这只验证两项；完整三模型方案仍需取得 ResNet18 权重并通过三项检查。

## 查看结果与比较限制

CNN 单项运行和套件独立保存在 `runs/foundation_sft_cnn/`，原有实验目录保留。使用终端打印的准确 suite 目录检查日志和 `report/`。现有统一查看器的默认 `runs/foundation_sft*` 扫描已涵盖这个新目录：

```bash
python show_foundation_results.py --project /home/cfu/EVA-X-main
```

若此前使用 `--models` 只显示八个指定模型，需把三个新 ID 加入列表：

```bash
python show_foundation_results.py --project /home/cfu/EVA-X-main \
  --models eva_x rad_dino chexfound cxformer radjepa \
  xray_siglip biomedclip chexzero \
  densenet121_imagenet densenet121_cxr resnet18_cxr
```

比较时核对相同患者/标签与共同分组指纹，并注明输入尺寸、预训练来源、数据处理和 FP32 规范。ImageNet DenseNet 与胸片 DenseNet 的主干相同，但来源、通道和归一化不同；MoCo-CXR 同时改变了主干和预训练方法，三项不能单独证明某一种因素的因果作用。结果查看器保留每批实验及恢复记录，不挑选最高测试分数；单种子患者 bootstrap 区间也不表示跨训练种子的波动。

本说明中的来源文件核验与程序测试不能替代服务器上的真实 DICOM、权重和 GPU smoke。正式可运行性以当前服务器三项检查的实际结果为准。
