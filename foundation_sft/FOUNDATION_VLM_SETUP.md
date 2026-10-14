# 四个视觉语言基础模型的监督实验

新增 CXR-CLIP、XraySigLIP、BiomedCLIP、CheXzero 的图像编码器，用同一份胸片和患者划分进行线性探测与 LoRA。这里训练的是“阻塞性通气障碍”监督分类器，使用发布的医学视觉权重；不构造文本塔，不进行文本提示或零样本分类。

旧 `configs/common.json` 和 `configs/experiments.json` 保持原设置。新实验使用 `configs/experiments_vlm.json`，输出到 `runs/foundation_sft_vlm`，继承 `common.json` 中的 Excel、DICOM、标签定义、冻结患者划分路径、划分种子和比例。服务器上已经修复并验证过的共同划分应继续使用，不能为新模型重新挑测试患者。

| 模型 | 配置 | GPU | 环境 | 输入 | 微批次 × 累积 |
|---|---|---|---|---|---|
| CXR-CLIP Swin-Tiny MCC | `cxr_clip.json` | `cuda:0` | `evax_lora` | 224 | 8 × 8 |
| XraySigLIP ViT-B/16 | `xray_siglip.json` | `cuda:1` | `cxr_openclip` | 512 | 4 × 16 |
| BiomedCLIP ViT-B/16 | `biomedclip.json` | `cuda:2` | `cxr_openclip` | 224 | 8 × 8 |
| CheXzero CLIP ViT-B/32 | `chexzero.json` | `cuda:3` | `evax_lora` | 224 | 8 × 8 |

默认四模型 × 两种模式 × 种子 42，共 8 项。四模型均用 FP32（`amp: false`），名义有效批次为 64；最后不足一个累积窗口时按实际样本数计算。FP32 比混合精度慢，并占用更多显存；512 输入的 XraySigLIP 通常最耗时。300 轮为上限，连续 30 轮验证患者 ROC-AUC 未改善时提前停止。旧实验若使用 AMP，与这些 FP32 实验的训练规范不同，比较时需要报告这一点。

## 1. 在 Windows 下载官方权重

在本机 PowerShell 执行。下载工具使用独立虚拟环境，不修改服务器解释器。模型文件放入 `D:\99\EVA-X-main\foundation_weights`：

```powershell
py -m venv D:\99\hf_download
& 'D:\99\hf_download\Scripts\python.exe' -m pip install 'huggingface_hub[cli]==0.24.7'

& 'D:\99\hf_download\Scripts\huggingface-cli.exe' download `
  StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli `
  --local-dir 'D:\99\EVA-X-main\foundation_weights\xray_siglip' `
  --local-dir-use-symlinks False

& 'D:\99\hf_download\Scripts\huggingface-cli.exe' download `
  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 `
  --local-dir 'D:\99\EVA-X-main\foundation_weights\biomedclip' `
  --local-dir-use-symlinks False
```

对应官方目录：

- [StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli](https://huggingface.co/StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli)
- [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

两个 Hugging Face 发布使用不同格式，不能共用加载代码或结构配置：

- **XraySigLIP 是原生 Transformers SigLIP**：保留同一官方目录的 `model.safetensors`、`config.json` 和 `preprocessor_config.json`。默认 checkpoint 为 `model.safetensors`，实际权重键是 `vision_model.*`、`text_model.*`，不需要 `open_clip_config.json`。只加载视觉权重，保留原生注意力池化。
- **BiomedCLIP 是 OpenCLIP**：保留官方 `open_clip_config.json` 和 `open_clip_pytorch_model.bin`。如果实际发布的权重为 `model.safetensors`，显式修改 `biomedclip.json` 的 `checkpoint`；不会自动改选另一份权重。

checkpoint 文件名相对于 `model_dir`，两个目录不能交叉使用。不能用其他模型的配置、随机模型或补造 JSON 代替官方文件。

若已经下载 XraySigLIP 权重、只缺 `config.json`，在 **Windows PowerShell** 只补下载这一小文件，不必重新下载权重：

```powershell
& 'D:\99\hf_download\Scripts\huggingface-cli.exe' download StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli config.json --local-dir 'D:\99\EVA-X-main\foundation_weights\xray_siglip' --local-dir-use-symlinks False
scp 'D:\99\EVA-X-main\foundation_weights\xray_siglip\config.json' 'cfu@192.168.130.16:/home/cfu/EVA-X-main/foundation_weights/xray_siglip/'
```

也可用浏览器打开[同一模型的官方 config.json](https://huggingface.co/StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli/resolve/main/config.json?download=true)，保存到上述目录。若官方文件访问失败，保留错误信息；不要把 `preprocessor_config.json` 重命名为 `config.json`。

CXR-CLIP 下载官方 Swin-Tiny MCC 权重：

```powershell
New-Item -ItemType Directory -Force 'D:\99\EVA-X-main\foundation_weights\cxr_clip' | Out-Null
curl.exe --fail --location `
  'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar' `
  --output 'D:\99\EVA-X-main\foundation_weights\cxr_clip\swint_mcc.tar'
```

这个 `.tar` 是 PyTorch checkpoint，直接保留文件，**不用解压**。来源为 [Soombit-ai/cxr-clip](https://github.com/Soombit-ai/cxr-clip)，选择的是 MIMIC-CXR + CheXpert + ChestX-ray14 的 Swin-Tiny MCC 版本；不能替换为通用微软 Swin 权重。

CheXzero 从[官方仓库](https://github.com/rajpurkarlab/CheXzero)指向的[官方 Google Drive 权重目录](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)下载这一项：

```text
best_64_0.0001_original_17000_0.863.pt
```

保存到 `D:\99\EVA-X-main\foundation_weights\chexzero\`。这是本实验预先指定的单个医学 CLIP ViT-B/32 checkpoint，不是权重集成，也不按本实验测试成绩选择 checkpoint。适配器会核对 12 层、768 视觉通道和 512 维投影等结构；OpenAI 原始 CLIP 不能代替 CheXzero 医学权重。

下载完成后可记录文件校验值，服务器上传后再次核对，避免传输截断：

```powershell
Get-FileHash -Algorithm SHA256 'D:\99\EVA-X-main\foundation_weights\cxr_clip\swint_mcc.tar'
Get-FileHash -Algorithm SHA256 'D:\99\EVA-X-main\foundation_weights\chexzero\best_64_0.0001_original_17000_0.863.pt'
Get-FileHash -Algorithm SHA256 'D:\99\EVA-X-main\foundation_weights\xray_siglip\model.safetensors'
Get-FileHash -Algorithm SHA256 'D:\99\EVA-X-main\foundation_weights\biomedclip\open_clip_pytorch_model.bin'
```

这些值验证上传前后文件一致，不能单独证明下载来源；下载仍应使用上面的官方发布。若 Hugging Face 返回权限错误，在官方页面和本机下载工具完成所需授权，不要把令牌贴到对话或写进配置。

## 2. 离线更新代码，再上传权重目录

先下载本次代码：[四模型更新 ZIP](https://github.com/FU254/FU254/archive/refs/heads/codex/chest-vlm-experiments.zip)。解压后找到**直接包含 `update_foundation_vlm.py` 和 `foundation_sft/` 的仓库根目录**，将这个目录重命名为 `vlm_update`，放到 `D:\99\EVA-X-main\vlm_update`。保留其完整目录结构，不要把里面的代码直接覆盖到正在使用的 `foundation_sft`。

通过 PyCharm **Tools → Deployment → Upload**，将 `vlm_update` 上传到 `/home/cfu/EVA-X-main/vlm_update`。在服务器项目根目录执行：

```bash
cd /home/cfu/EVA-X-main
conda activate evax_lora
python --version

# 使用 evax_lora 的 Python 3.10 或更新版本，先检查上传文件和现有代码。
python vlm_update/update_foundation_vlm.py --project . --source-dir vlm_update

# 检查通过后，备份旧代码并应用本次更新。
python vlm_update/update_foundation_vlm.py --project . --source-dir vlm_update --apply
```

更新器核对固定源码校验值与已知旧版本，只更新本次运行所需的适配器、公共运行模块、新配置、依赖文件和本文；已有代码有未识别修改时会停止，检查失败不会覆盖任何代码。旧代码保存在 `runs/foundation_sft_vlm/code_backups/`。`configs/common.json`、原 `configs/experiments.json`、数据、权重和已有实验结果均不改动。更新器本身不安装依赖、不联网、不训练。

如果已经安装四模型代码，本次只修正 XraySigLIP，使用 `--xray-siglip-only`。它只检查并安装 5 个相关文件，自动备份；不会读取或改写其他模型的配置、训练器及数据模块，保留正在运行的其他实验：

```bash
/opt/anaconda3/envs/evax_lora/bin/python vlm_update/update_foundation_vlm.py --project . --source-dir vlm_update --xray-siglip-only
/opt/anaconda3/envs/evax_lora/bin/python vlm_update/update_foundation_vlm.py --project . --source-dir vlm_update --xray-siglip-only --apply
```

如果曾使用旧 `base` Python 运行旧版更新器，可能出现 `:=` 的 SyntaxError，以及 `unlink(missing_ok=True)` 的 TypeError。此错误发生在写入前的校验阶段。确认上一条更新命令已经退出后，在项目根目录用已激活的 `evax_lora` 清理残留锁，再重新检查：

```bash
python -c "from pathlib import Path; p=Path('.foundation-vlm-update.lock'); p.unlink() if p.exists() else None"
python vlm_update/update_foundation_vlm.py --project . --source-dir vlm_update
```

不要修改 `chexzero.py` 或关闭源码校验来迁就旧解释器。新版更新器会在创建锁前拒绝过旧的 Python。

然后上传四个新权重目录到项目的 `foundation_weights/`。代码应用完成后，核心文件位置如下：

```text
/home/cfu/EVA-X-main/
  foundation_sft/
    models/cxr_clip.py
    models/openclip_vlm.py
    models/hf_siglip.py
    models/chexzero.py
    training_history.py
    configs/common_vlm.json
    configs/experiments_vlm.json
    configs/cxr_clip.json
    configs/xray_siglip.json
    configs/biomedclip.json
    configs/chexzero.json
    requirements-openclip.txt
  foundation_weights/
    cxr_clip/swint_mcc.tar
    xray_siglip/config.json
    xray_siglip/preprocessor_config.json
    xray_siglip/model.safetensors
    biomedclip/open_clip_config.json
    biomedclip/open_clip_pytorch_model.bin
    chexzero/best_64_0.0001_original_17000_0.863.pt
```

树中只列出核心文件；公共 checkpoint 安全加载、DICOM 预处理、训练和缓存模块已由更新器一起更新。CXR checkpoint 包含 OmegaConf 配置容器，新版加载代码会安全处理官方配置格式；不要为解决加载错误全局关闭 checkpoint 安全限制。

服务器终端核对文件 SHA-256：

```bash
cd /home/cfu/EVA-X-main
sha256sum foundation_weights/cxr_clip/swint_mcc.tar \
  foundation_weights/chexzero/best_64_0.0001_original_17000_0.863.pt \
  foundation_weights/xray_siglip/model.safetensors \
  foundation_weights/biomedclip/open_clip_pytorch_model.bin
```

## 3. 在服务器创建 OpenCLIP 环境

CXR-CLIP 和 CheXzero 沿用已工作的 `evax_lora`。BiomedCLIP 要求较新的 OpenCLIP/timm，另建克隆环境，避免影响 EVA-X 的 timm。XraySigLIP 使用已安装的 `transformers==4.48.3`，仍沿用既有 `cxr_openclip` 任务环境分配：

```bash
cd /home/cfu/EVA-X-main
unset LD_LIBRARY_PATH
conda create --name cxr_openclip --clone evax_lora --yes
conda activate cxr_openclip

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda)"
python -m pip install --no-deps -r foundation_sft/requirements-openclip.txt
python -m pip check
python -c "import torch, torchvision, open_clip, timm; print(torch.__version__, torchvision.__version__, torch.version.cuda, timm.__version__); print('CUDA:', torch.cuda.is_available())"
```

若 `cxr_openclip` 已存在，先检查其版本再补安装，不必再次克隆。安装固定 `open_clip_torch==2.23.0`、`timm==0.9.12`、`ftfy==6.1.1`、`sentencepiece==0.2.0`；`--no-deps` 防止 pip 自动替换已经工作的 Torch/torchvision，所需非 Torch 依赖在文件中单列。科学计算、Transformers 和 DICOM 依赖继承原环境。前后打印的 Torch、torchvision 和 CUDA 构建版本应一致，原 `evax_lora` 不需要安装 OpenCLIP 或更新 timm。`pip check` 通过后再 smoke。

所有训练子进程只读本地权重、禁用 Hub 联网，BiomedCLIP 不会因为文本塔而下载 PubMedBERT。CXR-CLIP 通过 Transformers 自带 Swin 架构构造视觉塔，CheXzero 使用本地视觉实现，两者也不需要在线构造文本模型。

## 4. 检查方案并启动实验

先检查 `nvidia-smi`，确认 GPU 0–3 可用。若需换卡，改 `configs/experiments_vlm.json` 中各项 `device`；同一张卡上的任务自动顺序执行。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main

python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_vlm.json --plan

python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_vlm.json \
  --cache-dir runs/foundation_sft/data_cache --check-only
```

`--plan` 不加载权重、不读取真实影像、不启动训练。检查流程会建立或校验继承的公共划分，在每个不同环境审核完整影像，然后完成 8 项真实权重 smoke；其中任一失败都会阻止正式训练。缓存目录可复用既有成功审核和模型预处理缓存，依赖或输入不同则重新验证，不能直接绕过像素审核。

检查完成显示 `Workflow checked` 后再运行完整实验：

```bash
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_vlm.json \
  --cache-dir runs/foundation_sft/data_cache
```

只运行一个预先指定模型或模式，也可以使用：

```bash
python run_foundation_experiments.py \
  --config foundation_sft/configs/experiments_vlm.json \
  --models xray_siglip --modes lora \
  --cache-dir runs/foundation_sft/data_cache
```

检查和正式运行各自创建新的 suite；第二次启动仍进行必要校验和 smoke。重新运行不会自动继续旧训练。种子、模式、超参数应在实验前确定，不能用测试集结果决定重跑哪个种子。

## 5. 预处理和特征定义

所有模型从同一份 DICOM 解码、窗宽窗位和灰阶方向处理出发，再执行对应尺寸、几何变换及归一化。共享 float32 DICOM 缩放与官方先生成 uint8/PIL RGB 的流程存在量化和插值差别，**不能宣称像素级复现官方 RGB 输入**；这一差别会保存在模型预处理记录中。模型间可比较的是本研究明确的共享 DICOM 分类流程。

- CXR-CLIP 保留官方 Swin-Tiny 224 输入、短边缩放后中心裁剪及 0.5/0.5 归一化。Swin 没有 CLS token，官方全局特征选最终空间序列的第一个 token，本适配器保留此行为。
- XraySigLIP 使用 Transformers 4.48.3 原生 SigLIP 512 视觉塔和发布的多头注意力池化。模型结构来自本地官方 `config.json` 的 `vision_config`，预处理来自 `preprocessor_config.json`；用户实际快照为 512×512 双三次缩放、0.5/0.5 归一化。仅加载 `vision_model.*`，既不构造文本塔，也不读取文本张量；LoRA 注入 12 层视觉自注意力的 36 个独立 Q/K/V 投影。兼容已有 PyTorch 2.0.1，使用原生 eager 注意力，不插值位置编码。
- BiomedCLIP 使用官方 224 视觉塔与 512 维视觉投影。预处理遵循固定 OpenCLIP 2.23.0 的官方加载规则：先取该版本 `PreprocessCfg` 默认值，再合并本地官方 `preprocess_cfg`；发布字段优先。两模型继承的默认字段、库版本和官方 JSON SHA-256 均记录；不在模型配置中编造均值、标准差，不改写官方配置。
- CheXzero 保留医学 CLIP ViT-B/32 的 224 编码器，以及官方医疗图像长边缩放、320 黑边填充、224 张量缩放与医学归一化路径；单个发布 checkpoint 的投影也被保留。

局部特征用于本研究的监督分类扩展，不是作者发布的独立局部评分任务。BiomedCLIP 将空间 patch token 通过同一发布视觉投影；CheXzero 使用空间 patch token 经发布 `ln_post` 和视觉投影；XraySigLIP 提供视觉 patch 序列，CXR-CLIP 提供最终 Swin 空间序列。LoRA 注入各视觉块的 Q/K/V 投影，发布池化和视觉投影保持冻结；原始投影没有被新建随机层替换。全局及局部特征的定义、checkpoint SHA-256 和预处理记录保存在每个 run 的 `model_metadata.json`。

沿用原框架：线性探测使用线性分类头，LoRA 使用输出 LayerNorm 和 MLP 分类头。因此两种模式的差异包含分类头和训练状态变化，结果比较反映整套训练方案，不能单独归因为 LoRA。

## 6. 查看结果和过拟合迹象

终端打印 `Suite directory`，新目录位于 `runs/foundation_sft_vlm/suites/`。每项实际训练目录见 `suite.json` 的 `run_dir`；不同 GPU 并行，同卡的线性探测和 LoRA 顺序执行。训练日志在 `logs/*_train.log`，报告在 `report/`。正式训练会自动验证、选择最佳 checkpoint 和阈值、评估测试集，不需要再次手动测试。

新训练每轮记录 `train_loss`、`val_loss`、患者验证 ROC-AUC/PR-AUC 和学习率。完成后 run 目录还会生成：

| 文件 | 内容 |
|---|---|
| `history.json` | 实际逐轮记录，包含验证损失 |
| `training_loss.png` | 原始训练/验证损失曲线，标记最佳验证 ROC 轮次 |
| `validation_auc.png` | 原始患者验证 ROC-AUC/PR-AUC 曲线，标记最佳轮次 |
| `best.pt`、`last.pt` | 验证选出的最佳 checkpoint、最后完整轮次 checkpoint |
| `test/metrics.json` | 测试指标与患者 bootstrap 置信区间 |
| `report/run_metrics.csv`（suite 下） | 8 项预定实验的指标和运行来源，未完成项另列 |

训练损失持续降低、验证损失开始升高，同时验证 AUC 持续下降，是可能过拟合的信号。单轮波动不足以确定过拟合；本实现依然按验证患者 ROC-AUC 选最佳权重并提前停止。损失曲线用于诊断，不用测试集调整停止轮次或阈值。

对旧实验或中途失败的实验，直接从保存记录画图，不重新训练，也不读取 DICOM：

```bash
# 把路径换成 suite.json 中的确切 run_dir。
python -m foundation_sft.training_history --run /home/cfu/EVA-X-main/runs/foundation_sft/模型/模式/确切运行目录

# 整个旧 suite：已完成项，以及日志能唯一确定run且已有history的失败项。
python -m foundation_sft.training_history --suite /home/cfu/EVA-X-main/runs/foundation_sft/suites/确切suite目录

# 也可选择一个尚不存在的新输出目录。
python -m foundation_sft.training_history \
  --suite /home/cfu/EVA-X-main/runs/foundation_sft/suites/确切suite目录 \
  --output runs/training_curves_review_01
```

单 run 默认输出到 `run/training_curves`，suite 默认输出到 `suite/training_curves/<job.id>`；如果默认目录已存在，则新建带时间戳的目录。显式 `--output` 必须指向新目录。已有 history、日志和图均不覆盖，曲线不平滑。终端会列出第一轮、最佳验证 ROC 轮、最后一轮的 train loss、val loss、验证 ROC/PR。

旧 history 若未记录 `val_loss`，图注会明确说明，终端显示 `val_loss=unknown`，不会根据 ROC-AUC 反推验证损失。此时可以查看训练损失和验证 AUC，但不能从缺失的验证损失曲线作判断。失败 run 只展示最后已完成轮次，不代表训练或测试成功。

当前云端的软件和合成样本检查不代表这四个医学 checkpoint 已经在服务器 GPU、真实 DICOM 和任务标签上验证。下载并上传实际权重后，需要上述真实 smoke、完整训练和医学结果评估；本说明不报告尚未运行的医学性能。
