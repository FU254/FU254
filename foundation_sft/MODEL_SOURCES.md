# 官方模型来源和本地资产

这里记录适配器的实现依据及服务器需要准备的文件。配置里的路径只是预期位置，文件不会随代码提交。下载前按各作者发布页面查看模型使用条款，并保留下载来源和版本信息。

## EVA-X Base

- 官方仓库：[hustvl/EVA-X](https://github.com/hustvl/EVA-X)。本项目已有相应的 `classification/models/models_eva.py`。
- 官方权重：[MapleF/eva_x](https://huggingface.co/MapleF/eva_x)。你现有 `base` 使用 `eva_x_base_patch16_merged520k_mim.pt`。
- 架构 ViT-B/16，224 输入，12 层、768 通道、独立 Q/K/V；LoRA 注入 36 个投影。
- 在 `configs/eva_x.json` 填本地 `checkpoint`；分类预训练头可以忽略，编码器参数必须完整匹配。
- 保留已修正的原生注意力回退，不需要因安装 xformers 而回滚。

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
- 默认配置文件 `chexfound/configs/train/vitl16_ibot333_highres512.yaml`，同时读取 `chexfound/configs/ssl_default_config.yaml` 中的架构默认值。若下载的权重配有其他配置，在 JSON 里填写对应 `config_file`。
- `configs/chexfound.json` 填 `repo_dir` 和 `checkpoint`。默认 `state_key: "teacher"`；实际 teacher 导出若不含这层键，应根据文件结构设置为空值。编码器参数必须完整匹配，FSDP chunk 包装会转换为等价的连续 block 名称。
- 输入按官方 evaluation 规则做 shortest-edge resize / center crop 和 ImageNet 归一化。

```bash
python -m pip install -r foundation_sft/requirements-chexfound.txt
mkdir -p foundation_sources foundation_weights/chexfound
git clone https://github.com/RPIDIAL/CheXFound.git foundation_sources/CheXFound
git -C foundation_sources/CheXFound checkout 7e288d06532f24abedb7d00781185c36727d9d69
```

作者原环境使用 Torch 2.0 和 xformers；你当前 2.0.1 / 0.0.21 需要用 `smoke` 验证这一组合。CPU 检查时若装有 xformers，启动 Python 前设置 `XFORMERS_DISABLED=1`；GPU 检查使用实际安装的加速路径。

## CXformer

- 官方仓库：[m42-health/CXformer](https://github.com/m42-health/CXformer)，核对版本 `da740db15a6d16cc06a26b82bd78527f2616dd7e`。
- 官方发布：[CXformer-base](https://huggingface.co/m42-health/CXformer-base)、[CXformer-small](https://huggingface.co/m42-health/CXformer-small)。
- 发布使用内置 `Dinov2WithRegistersModel`：patch14、4 个 register token、12 层；Base 为 768 通道、Small 为 384 通道，发布输入为 518。精确预处理读取本地 processor。
- 默认 `configs/cxformer.json` 使用 Base。切换 Small 时同时更改 `variant`、`checkpoint_source` 和 `model_dir`。
- LoRA 注入逐层 Q/K/V；空间特征排除 CLS 和 register token。

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
- 默认 `backend: "hf"`，`checkpoint` 填完整本地目录，包括权重、`config.json`、自定义 `.py`、其依赖文件和 processor 元数据。
- 本地源码核对完成后，在 JSON 将 `allow_remote_code` 设为 `true`，以使用作者要求的自定义模型类。这只从本地加载；外部代码引用会被拒绝。若模型同时含多个编码器，需要明确配置 `encoder_path`。
- 如果完整 `preprocessor_config.json` 含尺寸、resize/crop、归一化、rescale 和插值规范，自动读取；否则根据官方模型说明明确填写 `preprocessing.image_size/mean/std`，图像尺寸须为 14 的倍数。当前没有猜测这些数值。
- 官方说明建议 `timm==1.0.24`。EVA-X 使用 `timm==0.4.12`，建议为 RadJEPA 单独建环境，保留现有可运行环境：

```bash
conda create --name radjepa_sft --clone evax_lora
conda activate radjepa_sft
unset LD_LIBRARY_PATH
python -m pip install -r foundation_sft/requirements-hf.txt
python -m pip install --no-deps timm==1.0.24
python -m pip check
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="AIDELab-IITBombay/RadJEPA", local_dir="foundation_weights/radjepa")
PY
```

克隆环境需要磁盘空间和目录写权限。若 `pip check` 或官方源码提示缺少额外依赖，按实际错误补齐，随后重新检查 Torch 和 `smoke`。不要在这个新环境中运行依赖旧 timm 的 EVA-X 脚本。

另有 `backend: "ijepa"`，仅适用于**核实过的 RadJEPA 编码器导出**：使用 [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) 版本 `52c1ae95d05f743e000e8f10a1f3a79b10cff048`，填 `local_repo_dir`、医疗权重 `checkpoint`、`state_key: "encoder"` 或 `"target_encoder"`、预处理规范及 `allow_remote_code: true`。Hugging Face 文件不能仅凭改名当作此格式使用。

## 共同的检查顺序

每个模型填好路径后先运行 `--action smoke`，再启动监督训练。没有医学权重的云端测试只验证代码和架构，不验证医学任务表现。目前也没有连接你的 SSH 服务器，五个新模型均未完成实际服务器权重加载与 GPU 训练验证。

HF 依赖固定为 Transformers 4.50.3、huggingface-hub 0.29.3、tokenizers 0.21.1 和 safetensors 0.4.5，以保留对现有 Torch 2.0 的接口支持；云端实际验证使用 Torch 2.9.1 CPU，服务器兼容性仍需实际检查。

每次训练保存权重和预处理指纹、LoRA 投影名称和输入网格，重新评估时核对。默认目录 `foundation_sources/`、`foundation_weights/` 和 `runs/` 已加入 Git 排除项。
