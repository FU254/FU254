# 官方模型来源和本地资产

这里记录适配器的实现依据及服务器需要准备的文件。配置里的路径只是预期位置，文件不会随代码提交。下载前按各作者发布页面查看模型使用条款，并保留下载来源和版本信息。

## EVA-X Base

- 官方仓库：[hustvl/EVA-X](https://github.com/hustvl/EVA-X)。本项目已有相应的 `classification/models/models_eva.py`。
- 官方权重：[MapleF/eva_x](https://huggingface.co/MapleF/eva_x)。你现有 `base` 使用 `eva_x_base_patch16_merged520k_mim.pt`。
- 架构 ViT-B/16，224 输入，12 层、768 通道、独立 Q/K/V；LoRA 注入 36 个投影。
- 在 `configs/eva_x.json` 填本地 `checkpoint`；分类预训练头可以忽略，编码器参数必须完整匹配。
- 先检查权重的完整归一化参数对。MIM 发布含 `norm.weight/bias`，据此构造逐 token LayerNorm（`use_mean_pooling=False`）；分类导出含 `fc_norm.weight/bias`，据此保留池化后的 LayerNorm。不会把两者互相重命名；`mean(LN(tokens))` 与 `LN(mean(tokens))` 不等价。两种布局同时出现、不完整或缺失会拒绝。
- 特征编码器没有 MIM 任务的 `mask_token`、`lm_head.weight/bias` 和旧分类 `head.weight/bias`，仅精确移除这些已知键；其他参数缺失、意外键或形状错误继续报错。选择的归一化布局和忽略键保存在模型指纹中。
- 保留已修正的原生注意力回退，不需要因安装 xformers 而回滚。
- 旧官方 checkpoint 可能同时保存 NumPy 数值标量和普通 Python `set` 元数据。加载器先使用 Torch 的 `weights_only=True`；遇到这些已知类型或协议 4 FRAME 限制时，以同一个局部受限加载方式兼容，仍只返回编码器张量，不需要改写权重文件。

## RAD-DINO

- 官方发布：[microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino)。使用该医学预训练发布的完整 Hugging Face 快照。
- 默认位置 `foundation_weights/rad_dino/`，包括 `config.json`、`preprocessor_config.json`、权重文件和所有分片。
- 使用内置 DINOv2 编码器，取归一化 CLS 和空间 patch 特征，LoRA 注入逐层 Q/K/V。输入尺寸、裁剪、归一化从本地 processor 读取。
- 安装 `requirements-hf.txt`，在 `configs/rad_dino.json` 填 `model_dir`。

服务器能够访问 Hugging Face 时，可执行下面的下载命令；它只下载模型文件，不执行模型源码：

```bash
python -m pip install -r foundation_sft/requirements-hf.txt
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="microsoft/rad-dino", local_dir="foundation_weights/rad_dino")
PY
```

无法联网时，在能联网的电脑下载整个快照目录，再传到服务器。仅复制一个 `.bin` 或 `.safetensors` 不够。

## RayDINO / XRay-DINO

- 论文：[Advancing human-centric AI for robust X-ray analysis through holistic self-supervised learning](https://arxiv.org/abs/2405.01469)。
- 原生代码：[facebookresearch/dinov2 的 XRay-DINO 发布说明](https://github.com/facebookresearch/dinov2#pretrained-backbone-xray-dino)。核对源码版本 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，该版本提供 `dinov2/hub/xray_dino/backbones.py`。
- 作者的权重获取入口：[RayDINO downloads](https://ai.meta.com/resources/models-and-libraries/raydino-downloads/)。按官方表单申请，收到下载链接后把权重下载到服务器；链接可能临时有效。
- 实际 factory 是 `xray_dino_vitl16`：ViT-L/16、512 预训练分辨率、24 层、1024 通道、无 register token、4 个 block chunks。与常见 DINOv2 ViT-L/14 的 patch 大小不同。
- `configs/ray_dino.json` 填 `repo_dir`、`checkpoint`；`preprocessing.mean` 和 `preprocessing.std` 先留空，必须按作者提供的模型说明填写。不要随意套用 EVA-X 的归一化。
- 适配器构建官方架构后严格加载本地权重，支持原生 chunked QKV 的 24 个 LoRA 投影，取 CLS 和 patch 特征。
- 核对 `LICENSE_XRAY_DINO_MODEL`，该发布采用 FAIR Noncommercial Research 条款。

```bash
mkdir -p foundation_sources foundation_weights/ray_dino
git clone https://github.com/facebookresearch/dinov2.git foundation_sources/dinov2
git -C foundation_sources/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

权重默认文件名 `xray_dino_vitl16.pth` 是本项目配置约定，可以改为实际下载的文件名。如果官方文件包含外层键，可在核对参数结构后配置 `state_key`。

已有经过核实的 Hugging Face 转换版本时，也可选择 `backend: "hf"`、`model_dir` 和明确的 `checkpoint_source`。需要完整配置和 processor，并核实其确由 RayDINO 医学权重转换而来。

## CheXFound

- 官方仓库：[RPIDIAL/CheXFound](https://github.com/RPIDIAL/CheXFound)，核对版本 `7e288d06532f24abedb7d00781185c36727d9d69`。
- 官方 README 的 [模型和配置目录](https://drive.google.com/drive/folders/1GX2BWbujuVABtVpSZ4PTBykGULzrw806?usp=sharing) 提供发布文件。
- ViT-L/16、512 输入、1024 通道、4 个 register token、SwiGLU、24 个 fused QKV 投影。适配器使用编码器，不加载 GLoRI 下游分类头。
- 用户实际下载的官方示例文件为 `teacher_checkpoint.pth`（1,610,888,895 字节）、`config.yaml`（2,740 字节）、`glori.pth` 和 `results_eval_linear.json`。文件清单与配套 YAML 已核对：`vit_large`、patch16、512 输入、4 个 register tokens、`swiglufused` 和 layerscale `1e-5` 与适配器匹配。真实权重张量和服务器计算仍需通过加载检查。GLoRI 文件是作者原有分类头，本项目的新分类任务只加载基础编码器。
- `configs/chexfound.json` 已按用户实际目录设置 `checkpoint: "foundation_weights/CheXFound/teacher_checkpoint.pth"` 和 `config_file: "foundation_weights/CheXFound/config.yaml"`，同时读取官方源码中 `chexfound/configs/ssl_default_config.yaml` 的默认值。目录大小写按上传后的实际名称填写，Linux 区分大小写。未指定 `config_file` 时，适配器仍可使用源码中 `chexfound/configs/train/vitl16_ibot333_highres512.yaml`。
- `configs/chexfound.json` 的 `repo_dir` 指向独立的官方源码目录。默认 `state_key: "teacher"`，对应作者示例读取的容器；实际 teacher 导出若不含这层键，应根据文件结构设置为空值。编码器参数必须完整匹配，FSDP chunk 包装会转换为等价的连续 block 名称。
- 输入按官方 evaluation 规则做 shortest-edge resize / center crop 和 ImageNet 归一化。

```bash
python -m pip install -r foundation_sft/requirements-chexfound.txt
mkdir -p foundation_sources foundation_weights/CheXFound
git clone https://github.com/RPIDIAL/CheXFound.git foundation_sources/CheXFound
git -C foundation_sources/CheXFound checkout 7e288d06532f24abedb7d00781185c36727d9d69
```

电脑上的权重位于 `D:\99\EVA-X-main\foundation_weights\CheXFound`。如果服务器不能访问 GitHub，也可在电脑下载 [官方源码 ZIP](https://github.com/RPIDIAL/CheXFound/archive/7e288d06532f24abedb7d00781185c36727d9d69.zip)，把解压后的仓库根目录放到 `D:\99\EVA-X-main\foundation_sources\CheXFound`，确保其中直接存在 `chexfound/models/vision_transformer.py`。通过 PyCharm 分别上传 `foundation_weights` 和 `foundation_sources` 到项目根目录。下载了权重不代表同时下载了这份模型源码。

作者原环境使用 Torch 2.0 和 xformers；你当前 2.0.1 / 0.0.21 需要用 `smoke` 验证这一组合。CPU 检查时若装有 xformers，启动 Python 前设置 `XFORMERS_DISABLED=1`；GPU 检查使用实际安装的加速路径。

## CXformer

- 官方仓库：[m42-health/CXformer](https://github.com/m42-health/CXformer)，核对版本 `da740db15a6d16cc06a26b82bd78527f2616dd7e`。
- 官方发布：[CXformer-base](https://huggingface.co/m42-health/CXformer-base)、[CXformer-small](https://huggingface.co/m42-health/CXformer-small)。
- 发布使用内置 `Dinov2WithRegistersModel`：patch14、4 个 register token、12 层；Base 为 768 通道、Small 为 384 通道，发布输入为 518。精确预处理读取本地 processor。
- 默认 `configs/cxformer.json` 使用 Base。切换 Small 时同时更改 `variant`、`checkpoint_source` 和 `model_dir`。
- LoRA 注入逐层 Q/K/V；空间特征排除 CLS 和 register token。
- 实际发布的 `preprocessor_config.json` 使用 `CustomBitImageProcessor`，并启用 `do_hist_equalization: true`。作者先对 uint8 图像逐通道做 OpenCV 等价的直方图均衡化，再缩放、裁剪和归一化。适配器直接解析本地配置并复现这个流程，不需要执行 `custom_processor.py`；均衡化开关记录在模型预处理指纹中。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="m42-health/CXformer-base", local_dir="foundation_weights/cxformer")
PY
```

先安装 `requirements-hf.txt`；必须下载完整目录。

## RadJEPA

- 官方仓库：[aidelab-iitbombay/RadJEPA](https://github.com/aidelab-iitbombay/RadJEPA)，核对版本 `5f490ab58295a403ae6f29aa304db43a2aded423`。
- 官方模型：[AIDELab-IITBombay/RadJEPA](https://huggingface.co/AIDELab-IITBombay/RadJEPA)。GitHub 仓库主要提供说明，实际实现随 Hugging Face 模型目录的自定义 Python 文件发布。
- 核对目标编码器 ViT-B/14、768 通道、12 层和 12 个 fused QKV 投影。取编码器空间 patch token，均值作为全局特征。
- 默认 `backend: "hf"`，`checkpoint` 填完整本地目录。用户从官方发布下载的文件是 `config.json`、`modeling_radjepa.py`、`jepa_encoder.pth.tar` 和模型说明，没有标准 `.bin` / `.safetensors` 权重，也没有 `preprocessor_config.json`。
- 已核对这次发布的源码：使用 `timm` ViT-B，替换为 14 像素 patch，去掉 CLS，输入 224；官方 `from_pretrained` 会再次调用 `hf_hub_download`，即使传入本地目录也不能满足离线要求。适配器只从本地加载配置和模型类、直接构造架构，再严格加载 `jepa_encoder.pth.tar` 中的 `encoder` 或 `state_dict.encoder`；不会调用这个联网方法。标准 HF 权重格式仍然支持，目录同时出现多种格式时会要求明确整理。
- `allow_remote_code: true` 表示允许构造已审阅的本地模型类；默认配置现已开启，用于上面核对过的发布。以后更新作者源码时应重新核对。外部模块引用会被拒绝；源码和权重指纹随训练记录。若模型同时含多个编码器，需要明确配置 `encoder_path`。
- 输入尺寸 `preprocessing.image_size` 已按这次 `config.json` 填为 **224**。**作者这次发布和 README 没有给出归一化均值、标准差，所以默认 `radjepa.json` 的 `mean` / `std` 仍留空。** 若目标是采用作者规范，应先核实这些参数；若采用本研究自己的下游预处理，则应明确选择并记录，下面提供了独立实验配置。若另一发布包含完整的 processor 元数据，也可自动读取；默认配置不会自动填入其他模型的归一化。
- 获取作者规范的入口是上面的官方模型卡、GitHub 仓库，以及论文 [arXiv:2601.15891](https://arxiv.org/abs/2601.15891)。若论文及补充材料仍未说明，可在 [官方 GitHub Issues](https://github.com/aidelab-iitbombay/RadJEPA/issues) 向作者询问输入像素范围、归一化均值/标准差、resize/crop、插值及是否采用直方图均衡化。它们是数值和处理规范，不是另一份权重文件。
- 如果先按明确的下游实验规范试运行，使用单独的 **`configs/radjepa_imagenet.json`**：224 输入、直接 bicubic resize、ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`。统计值取自 [官方 I-JEPA `make_transforms` 默认参数](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/transforms.py)，RadJEPA README 确认使用 I-JEPA 训练代码，**但未确认保留这些默认值**，resize 规则也属于本研究的显式选择。因此此配置不宣称复现作者预处理；选择它就能开展真实权重的计算验证和探索训练。配置中的 `preprocessing_reference` 与实际参数会随结果保存，结果另存于 `runs/foundation_sft_imagenet`，公共患者划分仍保持一致。默认 `radjepa.json` 继续留空，待取得作者规范后填写。
- 官方说明建议 `timm==1.0.24`。EVA-X 使用 `timm==0.4.12`，建议为 RadJEPA 单独建环境，保留现有可运行环境：

```bash
conda create --name radjepa_sft --clone evax_lora -y
conda activate radjepa_sft
unset LD_LIBRARY_PATH
python -m pip install -r foundation_sft/requirements-hf.txt
python -m pip install --no-deps timm==1.0.24
python -m pip check
# 在项目根目录执行，使用已经下载并上传的官方权重。
# 此处选用明确标注的 ImageNet 实验配置；计算验证不等同于复现论文指标。
python -m foundation_sft.train --config foundation_sft/configs/radjepa_imagenet.json --action smoke --device cuda:3
```

克隆环境需要磁盘空间和目录写权限。若 `pip check` 或官方源码提示缺少额外依赖，按实际错误补齐，随后重新检查 Torch 和 `smoke`。不要在这个新环境中运行依赖旧 timm 的 EVA-X 脚本。

另有 `backend: "ijepa"`，仅适用于**核实过的 RadJEPA 编码器导出**：使用 [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) 版本 `52c1ae95d05f743e000e8f10a1f3a79b10cff048`，填 `local_repo_dir`、医疗权重 `checkpoint`、`state_key: "encoder"` 或 `"target_encoder"`、预处理规范及 `allow_remote_code: true`。Hugging Face 文件不能仅凭改名当作此格式使用。

## 共同的检查顺序

每个模型填好路径后先运行 `--action smoke`，再启动监督训练。没有医学权重的云端测试只验证代码和架构，不验证医学任务表现。用户已在服务器上报告 RAD-DINO、CXformer、CheXFound、RadJEPA 的线性探测及 LoRA 检查均通过，包括真实权重加载、GPU 前向/反向和保存重载。助手仍没有连接该 SSH 服务器；EVA-X 最新失败是 MIM 归一化结构与分类构造不一致，代码已按上述权重结构适配，等待服务器重新检查。RayDINO 未确认取得权重。

HF 依赖固定为 Transformers **4.48.3**、huggingface-hub 0.29.3、tokenizers 0.21.1 和 safetensors 0.4.5。此前的 4.50.3 在用户 Torch 2.0.1 环境导入时访问不存在的 `torch.compiler`，已据真实日志修正。4.48.3 已在独立 Python 3.10.22 / Torch 2.0.1+cpu / torchvision 0.15.2+cpu / timm 0.4.12 环境验证 DINOv2 与带 register 的 DINOv2 导入、本地权重加载、特征和 LoRA 反向；这些 CPU 软件检查不代表服务器医学权重或 CUDA 计算已经通过。更旧的 4.44/4.46 缺少 CXformer 所需内置 register 架构，不作为替代版本。

每次训练保存权重和预处理指纹、LoRA 投影名称和输入网格，重新评估时核对。默认目录 `foundation_sources/`、`foundation_weights/` 和 `runs/` 已加入 Git 排除项。

## 新增 CXR-CLIP / XraySigLIP / BiomedCLIP / CheXzero

这四个模型的固定发布版本、官方权重下载地址、视觉特征和预处理定义、独立 OpenCLIP 环境及八项实验运行步骤见 [FOUNDATION_VLM_SETUP.md](FOUNDATION_VLM_SETUP.md)。CXR-CLIP 使用 Swin-Tiny MCC，XraySigLIP 使用 512 输入 ViT-B/16，BiomedCLIP 使用 224 输入 ViT-B/16，CheXzero 使用医学 CLIP ViT-B/32 的单个预定 checkpoint。实际发布权重需要下载后进行服务器 smoke；云端合成样本检查不代表医学性能。
