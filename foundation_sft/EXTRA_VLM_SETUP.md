# Additional chest X-ray vision-language encoders

本次新增 MGCA、BioViL-T、MedKLIP，并把已有 CXR-CLIP 纳入独立批次。保留用户现有 `common.json`、原始患者划分、数据缓存和所有历史实验；新增输出根目录为 `runs/foundation_sft_vlm_extra`。

| 模型 ID | 实际结构/特征 | 当前实验 |
|---|---|---|
| `cxr_clip` | 官方 Swin-Tiny MCC；保留原适配器 | 线性探测、注意力 Q/K/V LoRA |
| `mgca` | 官方 MGCA ViT-B/16 query image encoder；768 维 CLS/patch | 线性探测、12 层 fused-QKV LoRA |
| `biovil_t` | 官方 ResNet50 静态图像＋缺失历史图像向量＋128 维投影 | 线性探测 |
| `medklip` | ResNet50 至 layer3＋视觉投影＋4 层医学知识查询解码器 | 线性探测 |

BioViL-T 在当前每位患者仅一张胸片的设置下，使用作者 `previous_image=None` 分支。时序 Transformer 的权重仍完整加载，但该分支不执行时序注意力，所以不伪造一项没有有效梯度的时序 LoRA。MedKLIP 保留知识解码器；其 CNN/decoder 参数高效微调尚未列为已验证实验。默认先跑四组线性探测，再单独跑两个模型 LoRA，共六组。

## 1. 安装代码

增量 ZIP 解压后只有一层 `vlm_extra_update`，其中直接包含 `update_foundation_vlm_extra.py`、`download_extra_vlm_weights.py`、`foundation_sft/` 和四个根目录训练入口。**它不是之前的 `cnn_update` 或 `vlm_update` 包。**

把整个 `vlm_extra_update` 文件夹放入本地 `D:\99\EVA-X-main`，通过 PyCharm 上传到服务器对应路径。随后在 SSH 终端执行：

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python vlm_extra_update/update_foundation_vlm_extra.py --project /home/cfu/EVA-X-main --apply
```

安装器默认从自己所在目录读取 payload。它只接受已知版本的三个共享代码文件，先核对完整 payload，再备份和安装，最后启用模型注册。遇到未知本地修改会停止，保留原文件。现有数据路径和原实验配置不属于覆盖范围；已有 `common_vlm.json` 与 `cxr_clip.json` 继续使用，缺失时才补齐。

如果下载的是完整 GitHub 代码 ZIP，可将解压后的仓库目录上传为 `vlm_extra_update`，执行同一命令。不要把 ZIP 中整个项目直接覆盖正在使用的项目。

如需在本地 PyCharm 项目中看到新增的四个根目录入口，可先在 Windows PowerShell 安装同一包，然后将安装器列出的文件通过 Deployment 上传：

```powershell
& "D:\99\hf-download-env\Scripts\python.exe" "D:\99\EVA-X-main\vlm_extra_update\update_foundation_vlm_extra.py" --project "D:\99\EVA-X-main" --apply
```

直接更新 Git checkout 时无需再执行增量安装器；完整源码已经在相应位置。

## 2. 准备权重（在能联网的 Windows 电脑上）

先只查看路径和官方来源，不需要 Torch：

```powershell
& "D:\99\hf-download-env\Scripts\python.exe" "D:\99\EVA-X-main\vlm_extra_update\download_extra_vlm_weights.py" --output "D:\99\EVA-X-main\foundation_weights" --plan
```

如果已经在本地安装代码，下载器也会位于 `D:\99\EVA-X-main\download_extra_vlm_weights.py`。从增量文件夹直接运行同样有效。

可直接下载的 CXR-CLIP、BioViL-T，以及 MedKLIP 的公开知识文本和 ClinicalBERT：

```powershell
& "D:\99\hf-download-env\Scripts\python.exe" -u "D:\99\EVA-X-main\vlm_extra_update\download_extra_vlm_weights.py" --models cxr_clip biovil_t medklip --clinicalbert --output "D:\99\EVA-X-main\foundation_weights"
```

下载器沿用 HTTPS/TLS 验证。BioViL-T 使用官方 loader 公布的 MD5，MedKLIP 知识 JSON 使用固定源码版本的 SHA256。已有文件不会被悄悄覆盖。脚本会明确列出仍缺失的手动文件和知识向量，并返回 2；这表示准备尚未完成。

MGCA 和 MedKLIP 的主权重需要在作者网盘中选择正确文件：

| 模型 | 官方下载入口 | 保存到本地 |
|---|---|---|
| CXR-CLIP | [Swin-Tiny MCC 官方文件](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar) | `foundation_weights/cxr_clip/swint_mcc.tar`，不用解压 |
| MGCA | [作者 Google Drive](https://drive.google.com/drive/folders/15_mP9Lqq2H15R53qlKn3l_xzGVzi9jX9?usp=sharing) | 选择 **ViT-Base MGCA 完整 `.ckpt`**，保存为 `foundation_weights/mgca/mgca_vit_base.ckpt` |
| BioViL-T | [官方 v1.0](https://huggingface.co/microsoft/BiomedVLP-BioViL-T/tree/v1.0) | `foundation_weights/biovil_t/biovil_t_image_model_proj_size_128.pt` |
| MedKLIP | [作者百度网盘，提取码 2vuc](https://pan.baidu.com/s/1qea5jVtHcW81i-jI4IP7Cg?pwd=2vuc) / [Google Drive](https://drive.google.com/drive/folders/1HBShH7J_pO8onkzuweDgDq2QPqj6zjG_?usp=sharing) | 完整医学预训练模型保存为 `foundation_weights/medklip/checkpoint_final.pth` |

更改文件名只用于匹配配置，不转换权重。MGCA 的 ResNet checkpoint、普通 ImageNet/DeiT 权重，以及只保留 ResNet 的 MedKLIP 导出都不能代替上述文件。可在新增模型 JSON 中修改 `checkpoint` 为真实文件名。

完整 MedKLIP 辅助资产：

```text
foundation_weights/medklip/
  checkpoint_final.pth
  observation_explanation.json
  clinicalbert/
    config.json
    vocab.txt
    pytorch_model.bin 或 model.safetensors
    其他下载到的 tokenizer 配置
```

把对应权重目录通过 PyCharm 上传到 `/home/cfu/EVA-X-main/foundation_weights/`。对照电脑、服务器文件大小和 SHA256，确保传输完成。

## 3. 在服务器一次性生成 MedKLIP 的知识查询向量

这里编码的是作者公开的 75 条医学解释文本，不读取患者报告或标签，也不需要你准备临床文本。完全从本地 ClinicalBERT 加载，之后训练不构造文本塔。

```bash
conda activate evax_lora
unset LD_LIBRARY_PATH
cd /home/cfu/EVA-X-main
python -c "import torch, torchvision, timm, transformers, cv2; print(torch.__version__, torchvision.__version__, timm.__version__, transformers.__version__, cv2.__version__)"
python prepare_medklip_book.py
```

使用已验证的 Torch 2.0.1 / torchvision 0.15.2 / timm 0.4.12 / Transformers 4.48.3。只有缺少 `cv2` 时才需要安装 `foundation_sft/requirements-vlm-extra.txt`；无需升级 Torch/CUDA。不要在 base Python 3.7 环境运行训练。

生成文件为 `foundation_weights/medklip/disease_book_embeddings.pt`。它记录知识文本顺序、JSON SHA256、ClinicalBERT 文件 SHA256，以及大小写和 128-token 截断规范。保留原顺序，不能用随机向量代替。已经生成的文件不覆盖，重建时另选输出路径并同步修改新配置。

## 4. 检查，再运行实验

```bash
python run_foundation_experiments.py --config foundation_sft/configs/experiments_vlm_extra.json --cache-dir runs/foundation_sft/data_cache --check-only
```

四项都通过后，运行四个模型的线性探测完整流程（含自动测试和汇总）：

```bash
python run_foundation_experiments.py --config foundation_sft/configs/experiments_vlm_extra.json --cache-dir runs/foundation_sft/data_cache
```

随后运行 CXR-CLIP 与 MGCA 的 LoRA：

```bash
python run_foundation_experiments.py --config foundation_sft/configs/experiments_vlm_extra_lora.json --cache-dir runs/foundation_sft/data_cache
```

可用 `--models cxr_clip mgca` 等参数先检查权重齐全的模型。默认使用 `evax_lora` 和 GPU 0–3；按空闲卡调整这两个**新**实验配置，不改旧实验。BioViL-T 448 输入采用 batch4×累积16，其余为 batch8×累积8；均为 FP32，有效 batch64。

单模型 PyCharm 入口是 `train_cxr_clip.py`、`train_mgca.py`、`train_biovil_t.py`、`train_medklip.py`，默认只训练线性头。前两个可通过参数 `--mode lora` 启用对应注意力 LoRA；使用远端 SSH 的 `evax_lora` 解释器。

查看这四个模型的已保存结果：

```bash
python show_foundation_results.py --project /home/cfu/EVA-X-main --models cxr_clip mgca biovil_t medklip
```

## 特征接口和后续蒸馏

所有适配器实现 `forward_features(images)`，返回 `global_features [B,D]`、`local_features [B,N,D]`、`grid_shape`。`FoundationClassifier(..., return_features=True)` 同时返回二分类 logits。

- MGCA：全局 CLS 和排除 CLS 的 patch token。特征位于原作者 downstream encoder 的投影头之前；不混入随机重建的文本或图像投影。
- BioViL-T：官方联合空间的 128 维投影特征，全局为投影空间 patch 均值；缺失历史影像向量和完整时序池化权重均来自 checkpoint。
- MedKLIP：全局为最后一层 75 个医学知识查询 token 的均值，这是本研究的下游池化选择；空间局部特征来自解码前的 14×14 图像 memory。额外 `forward_query_features` 返回 `[B,75,256]` 语义 token，它们**没有空间网格**，不能伪装为 75 个图像 patch。

不同模型的维度、网格及预处理不同。后续蒸馏需要分别处理各教师输入，并使用明确的维度投影/空间对齐；这些新适配器没有自动实施多教师加权或学生训练。本次保留全局、空间和知识语义接口，为后续蒸馏使用。

所有医学权重只从本地读取，缺失/形状不匹配时立即失败。结构完整加载并不独立证明权重身份，MGCA/MedKLIP 应从上表官方来源获取；可额外设置 `checkpoint_sha256` 锁定本地已核实文件。训练、验证和测试继续使用原患者划分，最佳模型及分类阈值只由验证集确定。

## 来源和验证边界

- MGCA：`HKU-MedAI/MGCA@3a2b4f25d47efc62e7d01d9204632a23d453afae`。
- BioViL-T：`microsoft/hi-ml@b67c1d27c6b17d8e8ff01f8c507f3cabdb307388`；保留的时序源码附有 `models/BIOVIL_LICENSE.txt`。
- MedKLIP：`MediaBrain-SJTU/MedKLIP@8e60b035ddc160f44a389f476b596f1052f02d8f`。

云端使用合成权重和合成 DICOM 验证软件、官方源码数学对应和保存重载，不代表已下载并验证你的医学权重，也不代表服务器 GPU 已通过。真实权重与数据的计算验证由上面的服务器 `--check-only` 完成。

本次本地验证：Python 3.10 / Torch 2.0.1 CPU 共 393 项检查，其中 371 项通过、22 项因可选运行环境缺失跳过；新增 18 项检查在 Torch 2.9.1 CPU 环境全部通过。完整 MGCA ViT-B/16（85,798,656 参数）、BioViL-T 图像模型（27,349,928 参数）、MedKLIP 视觉/查询模型（14,464,322 参数）均与上述固定版本官方源码完成严格 state dict 和对应特征输出对照。另以合成的 12 层、768 维本地 BERT 完成知识向量生成和缓存重载，验证 tokenizer 大小写配置被保留；合成权重随后删除。实际增量 ZIP 在旧 checkout 上安装、再次安装及六组任务规划均通过，原自定义数据配置和历史文件保持不变。
