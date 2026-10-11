#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EVA-X Base LoRA training pipeline for pulmonary-function regression."""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def log(message):
    logger.info(message)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CLASSIFICATION_DIR = os.path.join(PROJECT_DIR, "classification")
if not os.path.exists(os.path.join(CLASSIFICATION_DIR, "models", "models_eva.py")):
    raise FileNotFoundError(
        f"找不到 EVA-X 模型定义: {os.path.join(CLASSIFICATION_DIR, 'models', 'models_eva.py')}"
    )
sys.path.insert(0, CLASSIFICATION_DIR)

from models.models_eva import eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE


IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 30
DEFAULT_SEED = 42
HEAD_LR = 1e-4
LORA_LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.6
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj')


DEFAULT_EXCEL_PATH = r"/home/cfu/肺功能万能表最终版1_fev1fvc差值标签(1).xlsx"
DEFAULT_DICOM_DIR = r"/data/users/cfu/DR数据_按文件10份-筛选合成"
DEFAULT_PRETRAINED_PATH = os.path.join(
    PROJECT_DIR, "eva_x_base_patch16_merged520k_mim.pt"
)
ACCESSION_COLUMN = "放射编号"
REFERENCE_CLASSIFICATION_LABEL = "阻塞性通气障碍"
REFERENCE_SPLIT_VERSION = "group_stratified_70_10_20_v1"
VALID_SPLITS = ("train", "val", "test")
PLOT_DISPLAY_VERSION = "pycharm_show"


def make_safe_name(value):
    value = re.sub(r'[\\/:*?"<>|\s]+', "_", str(value).strip())
    return value.strip("_") or "task"


def normalize_accession(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested_device=None):
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"请求了 {requested_device}，但当前环境没有可用 CUDA。")
            index = 0 if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"请求了 {requested_device}，但当前仅检测到 {torch.cuda.device_count()} 张 GPU。"
                )
        log(f"使用设备: {device}")
        return device

    if torch.cuda.is_available():
        index = 2 if torch.cuda.device_count() > 2 else 0
        device = torch.device(f"cuda:{index}")
        log(f"使用设备: {device} ({torch.cuda.get_device_name(index)})")
        return device

    log("使用设备: cpu")
    return torch.device("cpu")


def load_torch_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LoRALinear(nn.Module):
    """冻结原始线性层，仅训练低秩 LoRA 增量。"""

    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear 仅支持 nn.Linear，实际为 {type(base_layer).__name__}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self):
        return self.base_layer.weight + torch.matmul(self.lora_B, self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, features):
        return nn.functional.linear(features, self.weight, self.bias)


def inject_lora_into_eva(backbone, rank, alpha):
    injected_names = []
    for block_index, block in enumerate(backbone.blocks):
        for module_name in LORA_TARGET_MODULES:
            base_layer = getattr(block.attn, module_name, None)
            if not isinstance(base_layer, nn.Linear):
                raise RuntimeError(
                    f"无法在 backbone.blocks.{block_index}.attn.{module_name} 注入 LoRA；"
                    f"期望 nn.Linear，实际为 {type(base_layer).__name__}"
                )
            setattr(block.attn, module_name, LoRALinear(base_layer, rank, alpha))
            injected_names.append(f"backbone.blocks.{block_index}.attn.{module_name}")
    return injected_names


class EVAXBaseLoRARegressor(nn.Module):
    def __init__(self, pretrained_weight_path, dropout_rate=DROPOUT_RATE):
        super().__init__()
        log("创建 EVA-X Base LoRA 回归模型")
        self.backbone = eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(pretrained=False)
        self._load_pretrained_weights(pretrained_weight_path)

        self.backbone.fc_norm = nn.LayerNorm(self.backbone.embed_dim)
        hidden_dim = self.backbone.embed_dim // 2
        output_dim = self.backbone.embed_dim // 4
        self.bottleneck = nn.Linear(self.backbone.embed_dim, hidden_dim)
        self.bottleneck2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(output_dim, 1)
        self.lora_module_names = inject_lora_into_eva(
            self.backbone, rank=LORA_RANK, alpha=LORA_ALPHA
        )
        self._init_head()
        self.freeze_backbone()

    def _load_pretrained_weights(self, pretrained_weight_path):
        if not pretrained_weight_path or not os.path.isfile(pretrained_weight_path):
            raise FileNotFoundError(f"找不到 EVA-X Base 预训练权重: {pretrained_weight_path}")

        log(f"加载预训练权重: {pretrained_weight_path}")
        checkpoint = load_torch_checkpoint(pretrained_weight_path)
        if isinstance(checkpoint, dict):
            for key in ("module", "model", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        cleaned_state = {}
        for name, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name.startswith("module."):
                name = name[7:]
            cleaned_state[name] = value

        backbone_state = self.backbone.state_dict()
        compatible_state = {
            name: value
            for name, value in cleaned_state.items()
            if name in backbone_state and backbone_state[name].shape == value.shape
        }
        if not compatible_state:
            raise RuntimeError("预训练权重中没有与 EVA-X Base Backbone 匹配的参数。")

        message = self.backbone.load_state_dict(compatible_state, strict=False)
        log(
            f"预训练权重加载完成 | 匹配参数: {len(compatible_state)} | "
            f"缺失键: {len(message.missing_keys)} | 意外键: {len(message.unexpected_keys)}"
        )

    def _init_head(self):
        for layer in (self.bottleneck, self.bottleneck2, self.regressor):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        for parameter in self.backbone.fc_norm.parameters():
            parameter.requires_grad = True

    def set_backbone_train_mode(self):
        self.backbone.eval()
        self.backbone.fc_norm.train()

    def log_parameter_status(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log(f"LoRA 目标层: {LORA_TARGET_MODULES} | 注入层数: {len(self.lora_module_names)}")
        log(f"参数状态 | 可训练: {trainable:,} | 冻结: {frozen:,}")

    def forward(self, images):
        features = self.backbone.patch_embed(images)
        batch_size = features.shape[0]
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)

        if self.backbone.pos_embed is not None:
            features = features + self.backbone.pos_embed
        features = self.backbone.pos_drop(features)

        rel_pos_bias = (
            self.backbone.rel_pos_bias()
            if self.backbone.rel_pos_bias is not None
            else None
        )
        for block in self.backbone.blocks:
            features = block(features, rel_pos_bias)

        features = self.backbone.norm(features)
        features = self.backbone.fc_norm(features.mean(1))
        features = self.dropout(torch.relu(self.bottleneck(features)))
        features = self.dropout(torch.relu(self.bottleneck2(features)))
        return self.regressor(features)


def source_signature(path):
    stat = os.stat(path)
    signature_text = f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(signature_text.encode("utf-8")).hexdigest()


def build_label_lookup(excel_path, label_column):
    log(f"读取回归标签: {excel_path} -> {label_column}")
    dataframe = pd.read_excel(excel_path, sheet_name=0)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    required_columns = {ACCESSION_COLUMN, label_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise KeyError(
            f"Excel 缺少列 {sorted(missing_columns)}；现有列: {dataframe.columns.tolist()}"
        )

    label_table = dataframe[[ACCESSION_COLUMN, label_column]].copy()
    label_table[ACCESSION_COLUMN] = label_table[ACCESSION_COLUMN].map(
        normalize_accession
    )
    label_table["target"] = pd.to_numeric(
        label_table[label_column], errors="coerce"
    )
    finite_mask = label_table["target"].notna() & np.isfinite(label_table["target"])
    valid_mask = label_table[ACCESSION_COLUMN].ne("") & finite_mask
    invalid_count = int((~valid_mask).sum())
    label_table = label_table.loc[valid_mask, [ACCESSION_COLUMN, "target"]]
    if label_table.empty:
        raise RuntimeError(f"Excel 中没有可用的 {label_column} 连续标签。")

    conflicting = label_table.groupby(ACCESSION_COLUMN)["target"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        examples = conflicting.index.astype(str).tolist()[:5]
        raise RuntimeError(
            f"发现 {len(conflicting)} 个放射编号对应多个 {label_column} 值，示例: {examples}"
        )

    duplicate_count = int(label_table.duplicated(ACCESSION_COLUMN).sum())
    label_table = label_table.drop_duplicates(ACCESSION_COLUMN, keep="first")
    label_lookup = dict(
        zip(label_table[ACCESSION_COLUMN], label_table["target"].astype(float))
    )
    target_values = label_table["target"].to_numpy(dtype=np.float64)
    log(
        f"有效标签: {len(label_lookup)} | 无效/缺失: {invalid_count} | "
        f"同值重复: {duplicate_count}"
    )
    log(
        f"标签统计 | mean={target_values.mean():.4f} | std={target_values.std():.4f} | "
        f"min={target_values.min():.4f} | max={target_values.max():.4f}"
    )
    return label_lookup


def preprocess_dicom(dicom):
    image = dicom.pixel_array.astype(np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"期望二维 DICOM，实际 shape={image.shape}")

    slope = float(getattr(dicom, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0) or 0.0)
    image = image * slope + intercept

    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        raise ValueError("DICOM 像素全部为非有限值")

    if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
        valid_pixels = image[finite_mask]
        image = float(valid_pixels.max() + valid_pixels.min()) - image
        finite_mask = np.isfinite(image)

    valid_pixels = image[finite_mask]
    lower, upper = np.percentile(valid_pixels, [1.0, 99.0])
    if upper <= lower:
        lower, upper = float(valid_pixels.min()), float(valid_pixels.max())
    if upper <= lower:
        raise ValueError("DICOM 像素没有有效动态范围")

    image = np.nan_to_num(image, nan=lower, posinf=upper, neginf=lower)
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower + 1e-8)

    height, width = image.shape
    if height > width:
        start = (height - width) // 2
        image = image[start:start + width, :]
    elif width > height:
        start = (width - height) // 2
        image = image[:, start:start + height]

    image = cv2.resize(
        image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
    )
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def load_and_preprocess_data(
    excel_path,
    dicom_dir,
    label_column,
    cache_path,
    force_reload=False,
):
    current_signature = source_signature(excel_path)
    if not force_reload and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {"X", "y", "accessions", "filenames", "source_signature"}
            if required.issubset(cached.files):
                cached_signature = str(cached["source_signature"].item())
                if cached_signature == current_signature:
                    images = cached["X"]
                    targets = cached["y"]
                    metadata = pd.DataFrame(
                        {
                            ACCESSION_COLUMN: cached["accessions"].astype(str),
                            "filename": cached["filenames"].astype(str),
                        }
                    )
                    log(f"从缓存加载 {len(images)} 个样本: {cache_path}")
                    return images, targets, metadata
        log("缓存与当前 Excel 不一致，重新预处理数据。")

    label_lookup = build_label_lookup(excel_path, label_column)
    dicom_files = sorted(
        entry.name
        for entry in os.scandir(dicom_dir)
        if entry.is_file() and entry.name.lower().endswith(".dcm")
    )
    if not dicom_files:
        raise RuntimeError(f"DICOM 目录中没有 .dcm 文件: {dicom_dir}")

    images = []
    targets = []
    accessions = []
    filenames = []
    reasons = {
        "missing_pixel_data": 0,
        "missing_accession": 0,
        "missing_label": 0,
        "invalid_image": 0,
        "other_error": 0,
    }
    error_examples = []

    for filename in tqdm(dicom_files, desc=f"Processing {label_column}"):
        path = os.path.join(dicom_dir, filename)
        try:
            dicom = pydicom.dcmread(path)
            if "PixelData" not in dicom:
                reasons["missing_pixel_data"] += 1
                continue

            accession = normalize_accession(getattr(dicom, "AccessionNumber", ""))
            if not accession:
                reasons["missing_accession"] += 1
                continue
            if accession not in label_lookup:
                reasons["missing_label"] += 1
                continue

            try:
                image = preprocess_dicom(dicom)
            except (ValueError, TypeError) as exc:
                reasons["invalid_image"] += 1
                if len(error_examples) < 5:
                    error_examples.append(f"{filename}: {exc}")
                continue

            images.append(image)
            targets.append(label_lookup[accession])
            accessions.append(accession)
            filenames.append(filename)
        except Exception as exc:
            reasons["other_error"] += 1
            if len(error_examples) < 5:
                error_examples.append(f"{filename}: {exc}")

    if not images:
        raise RuntimeError(
            f"没有得到可训练样本。过滤统计: {reasons}；错误示例: {error_examples}"
        )

    images = np.stack(images).astype(np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    metadata = pd.DataFrame(
        {ACCESSION_COLUMN: accessions, "filename": filenames}
    )
    log(f"数据加载完成，共 {len(images)} 个样本 | 过滤统计: {reasons}")
    if error_examples:
        log(f"错误示例: {error_examples}")
    log(
        f"匹配后标签 | mean={targets.mean():.4f} | std={targets.std():.4f} | "
        f"min={targets.min():.4f} | max={targets.max():.4f}"
    )

    np.savez_compressed(
        cache_path,
        X=images,
        y=targets,
        accessions=np.asarray(accessions, dtype=str),
        filenames=np.asarray(filenames, dtype=str),
        source_signature=np.asarray(current_signature),
    )
    log(f"预处理缓存已保存: {cache_path}")
    return images, targets, metadata


def resolve_fixed_split_path(excel_path, seed, configured_path=None):
    if configured_path:
        return os.path.abspath(os.path.expanduser(configured_path))
    filename = (
        f"fixed_group_split_{make_safe_name(REFERENCE_CLASSIFICATION_LABEL)}_"
        f"70_10_20_seed{seed}.csv"
    )
    return os.path.join(os.path.dirname(os.path.abspath(excel_path)), filename)


def load_reference_group_split(metadata, targets, manifest_path, seed):
    if metadata is None or ACCESSION_COLUMN not in metadata.columns:
        raise RuntimeError("缺少放射编号元数据，无法进行组级数据划分。")
    if len(metadata) != len(targets):
        raise RuntimeError("元数据数量与标签数量不一致。")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "找不到分类 tiny 的固定划分清单: "
            f"{manifest_path}。请先运行 分类/tiny-new-阻塞.py 生成该 CSV，"
            "回归任务不会自行创建另一套划分。"
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={ACCESSION_COLUMN: str, "split": str},
        encoding="utf-8-sig",
    )
    required_columns = {
        ACCESSION_COLUMN,
        "label",
        "split",
        "seed",
        "split_version",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise RuntimeError(
            f"固定划分清单缺少字段 {sorted(missing_columns)}: {manifest_path}"
        )

    manifest[ACCESSION_COLUMN] = manifest[ACCESSION_COLUMN].map(normalize_accession)
    if manifest[ACCESSION_COLUMN].eq("").any():
        raise RuntimeError(f"固定划分清单包含空放射编号: {manifest_path}")
    if manifest[ACCESSION_COLUMN].duplicated().any():
        duplicates = manifest.loc[
            manifest[ACCESSION_COLUMN].duplicated(keep=False), ACCESSION_COLUMN
        ].unique().tolist()[:5]
        raise RuntimeError(f"固定划分清单包含重复放射编号，示例: {duplicates}")

    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    invalid_splits = set(manifest["split"]).difference(VALID_SPLITS)
    missing_splits = set(VALID_SPLITS).difference(manifest["split"])
    if invalid_splits or missing_splits:
        raise RuntimeError(
            f"固定划分清单 split 无效；非法={sorted(invalid_splits)}，"
            f"缺少={sorted(missing_splits)}"
        )

    manifest_labels = pd.to_numeric(manifest["label"], errors="coerce")
    if manifest_labels.isna().any() or not manifest_labels.isin([0, 1]).all():
        raise RuntimeError(f"固定划分清单包含非 0/1 分类标签: {manifest_path}")
    manifest_seeds = pd.to_numeric(manifest["seed"], errors="coerce")
    if manifest_seeds.isna().any() or not (manifest_seeds == int(seed)).all():
        values = sorted(manifest_seeds.dropna().unique().tolist())
        raise RuntimeError(
            f"固定划分清单 seed={values}，当前回归任务 seed={seed}。"
        )
    versions = set(manifest["split_version"].astype(str))
    if versions != {REFERENCE_SPLIT_VERSION}:
        raise RuntimeError(
            f"固定划分清单版本={sorted(versions)}，"
            f"要求版本={REFERENCE_SPLIT_VERSION}。"
        )

    accessions = metadata[ACCESSION_COLUMN].map(normalize_accession).to_numpy()
    if any(not accession for accession in accessions):
        raise RuntimeError("回归数据包含空放射编号，无法映射固定划分。")
    split_by_accession = manifest.set_index(ACCESSION_COLUMN)["split"]
    current_groups = set(accessions)
    reference_groups = set(split_by_accession.index)
    current_only = sorted(current_groups.difference(reference_groups))
    reference_only = sorted(reference_groups.difference(current_groups))
    if current_only or reference_only:
        raise RuntimeError(
            "回归数据与分类 tiny 固定清单的放射编号集合不一致；"
            f"回归新增={len(current_only)}（示例: {current_only[:5]}），"
            f"回归缺少={len(reference_only)}（示例: {reference_only[:5]}）。"
            "为保证 train/val/test 完全一致，训练已停止。"
        )

    sample_splits = pd.Series(accessions).map(split_by_accession).to_numpy()
    indices = {
        split: np.flatnonzero(sample_splits == split)
        for split in VALID_SPLITS
    }
    if any(len(values) == 0 for values in indices.values()):
        raise RuntimeError("映射分类 tiny 固定清单后至少有一个回归数据集为空。")

    split_groups = {
        split: set(accessions[indices[split]])
        for split in VALID_SPLITS
    }
    if any(
        split_groups[left].intersection(split_groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("固定划分映射后发现放射编号跨数据集泄漏。")

    log(f"已读取分类 tiny 固定划分清单: {manifest_path}")
    log(
        f"分类清单放射编号={len(manifest)} | "
        f"当前回归可用放射编号={len(set(accessions))}"
    )
    for split in VALID_SPLITS:
        values = np.asarray(targets)[indices[split]]
        log(
            f"{split} | 样本={len(values)} | 放射编号={len(split_groups[split])} | "
            f"target mean={values.mean():.4f} std={values.std():.4f}"
        )
    return indices["train"], indices["val"], indices["test"], sample_splits


class MedicalImageRegressionDataset(Dataset):
    def __init__(self, images, targets, indices, transform, image_mean, image_std):
        self.images = images
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.image_mean = float(image_mean)
        self.image_std = float(image_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = int(self.indices[index])
        image = self.transform(self.images[sample_index])
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = (image - self.image_mean) / (self.image_std + 1e-8)
        target = torch.tensor(self.targets[sample_index], dtype=torch.float32)
        return image, target


def calculate_image_mean_std(images, indices, chunk_size=64):
    total = 0.0
    total_squared = 0.0
    pixel_count = 0
    for start in range(0, len(indices), chunk_size):
        batch_indices = indices[start:start + chunk_size]
        batch = images[batch_indices].astype(np.float64, copy=False)
        total += float(batch.sum())
        total_squared += float(np.square(batch).sum())
        pixel_count += int(batch.size)
    mean = total / pixel_count
    variance = max(total_squared / pixel_count - mean ** 2, 0.0)
    return float(mean), float(np.sqrt(variance))


def calculate_regression_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    residuals = predictions - labels
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((labels - labels.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    if len(labels) > 1 and labels.std() > 1e-12 and predictions.std() > 1e-12:
        pearson_r = float(np.corrcoef(labels, predictions)[0, 1])
    else:
        pearson_r = 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "pearson_r": pearson_r,
    }


def evaluate(model, data_loader, criterion, target_mean, target_std, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    labels_standardized = []
    predictions_standardized = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * labels.shape[0]
            total_count += int(labels.shape[0])
            labels_standardized.append(labels.cpu().numpy())
            predictions_standardized.append(outputs.cpu().numpy())

    labels_standardized = np.concatenate(labels_standardized)
    predictions_standardized = np.concatenate(predictions_standardized)
    labels = labels_standardized * target_std + target_mean
    predictions = predictions_standardized * target_std + target_mean
    metrics = calculate_regression_metrics(labels, predictions)
    return total_loss / total_count, metrics, labels, predictions


def save_checkpoint(
    path,
    model,
    target_mean,
    target_std,
    image_mean,
    image_std,
    label_column,
    epoch,
    val_metrics,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "image_mean": float(image_mean),
            "image_std": float(image_std),
            "label_column": label_column,
            "epoch": int(epoch),
            "val_metrics": val_metrics,
        },
        path,
    )


def add_file_logger(log_path):
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(file_handler)


def save_training_plot(history, test_labels, test_predictions, output_path):
    figure = plt.figure(figsize=(14, 10), dpi=140)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    )

    axes[0].plot(
        history["train_loss"], label="Train Loss", color="blue", linewidth=2.0
    )
    axes[0].plot(
        history["val_loss"], label="Val Loss", color="orange", linewidth=2.0
    )
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("SmoothL1 Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history["train_rmse"], label="Train RMSE", color="blue", linewidth=2.0
    )
    axes[1].plot(
        history["val_rmse"], label="Val RMSE", color="orange", linewidth=2.0
    )
    axes[1].set_title("Training & Validation RMSE", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("RMSE (original unit)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(test_labels, test_predictions, alpha=0.7, s=30)
    lower = float(min(np.min(test_labels), np.min(test_predictions)))
    upper = float(max(np.max(test_labels), np.max(test_predictions)))
    if upper <= lower:
        upper = lower + 1.0
    axes[2].plot([lower, upper], [lower, upper], "r--", linewidth=1.5)
    axes[2].set_xlim(lower, upper)
    axes[2].set_ylim(lower, upper)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title("Test Predictions", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("True value", fontsize=12)
    axes[2].set_ylabel("Predicted value", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    for axis in axes:
        axis.tick_params(axis="both", labelsize=11)

    figure.tight_layout(pad=2.0)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")


def build_parser(label_column, task_name):
    parser = argparse.ArgumentParser(
        description=f"EVA-X Base LoRA regression for {label_column}."
    )
    parser.add_argument(
        "--excel-path",
        default=os.environ.get("PFT_EXCEL_PATH", DEFAULT_EXCEL_PATH),
    )
    parser.add_argument(
        "--dicom-dir",
        default=os.environ.get("PFT_DICOM_DIR", DEFAULT_DICOM_DIR),
    )
    parser.add_argument(
        "--pretrained-path",
        default=os.environ.get("EVA_X_PRETRAINED_PATH", DEFAULT_PRETRAINED_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "runs", task_name),
    )
    parser.add_argument(
        "--fixed-split-path",
        default=os.environ.get("FIXED_GROUP_SPLIT_PATH"),
        help="分类 tiny 生成的固定放射编号 70:10:20 split CSV。",
    )
    parser.add_argument("--device", default=os.environ.get("EVA_X_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-reload", action="store_true")
    return parser


def run_regression_task(label_column, task_name, argv=None):
    args = build_parser(label_column, task_name).parse_args(argv)
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs 和 --patience 必须为正整数。")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size 必须为正整数，--num-workers 不能为负数。")

    fixed_split_path = resolve_fixed_split_path(
        args.excel_path,
        args.seed,
        args.fixed_split_path,
    )

    for path, description in (
        (args.excel_path, "标签 Excel"),
        (args.pretrained_path, "预训练权重"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到{description}: {path}")
    if not os.path.isdir(args.dicom_dir):
        raise FileNotFoundError(f"找不到 DICOM 目录: {args.dicom_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    add_file_logger(os.path.join(args.output_dir, f"train_{run_stamp}.log"))
    set_seed(args.seed)
    device = resolve_device(args.device)
    label_tag = make_safe_name(label_column)
    cache_path = os.path.join(args.output_dir, f"cached_{label_tag}.npz")
    best_model_path = os.path.join(
        args.output_dir, f"best_evax_tiny_{label_tag}.pth"
    )

    log(f"任务: {task_name} | 标签列: {label_column}")
    log(f"公共训练脚本: {os.path.abspath(__file__)}")
    log(f"Excel: {args.excel_path}")
    log(f"DICOM: {args.dicom_dir}")
    log(f"输出目录: {args.output_dir}")
    log(f"分类 tiny 固定划分清单: {fixed_split_path}")
    log(f"绘图模式: {PLOT_DISPLAY_VERSION} | Matplotlib backend: {plt.get_backend()}")

    images, targets, metadata = load_and_preprocess_data(
        excel_path=args.excel_path,
        dicom_dir=args.dicom_dir,
        label_column=label_column,
        cache_path=cache_path,
        force_reload=args.force_reload,
    )
    train_idx, val_idx, test_idx, sample_splits = load_reference_group_split(
        metadata,
        targets,
        fixed_split_path,
        args.seed,
    )

    split_manifest = metadata.copy()
    split_manifest["target"] = targets
    split_manifest["split"] = sample_splits
    split_manifest_path = os.path.join(
        args.output_dir, f"split_{label_tag}_{run_stamp}.csv"
    )
    split_manifest.to_csv(split_manifest_path, index=False, encoding="utf-8-sig")

    image_mean, image_std = calculate_image_mean_std(images, train_idx)
    target_mean = float(targets[train_idx].mean())
    target_std = float(targets[train_idx].std())
    if image_std < 1e-8:
        raise RuntimeError("训练集图像标准差过小，无法进行标准化。")
    if target_std < 1e-8:
        raise RuntimeError("训练集目标标准差过小，无法进行回归训练。")
    standardized_targets = ((targets - target_mean) / target_std).astype(np.float32)
    log(
        f"训练集标准化参数 | image mean={image_mean:.6f} std={image_std:.6f} | "
        f"target mean={target_mean:.6f} std={target_std:.6f}"
    )

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.ToTensor()]
    )

    def make_dataset(indices, transform):
        return MedicalImageRegressionDataset(
            images,
            standardized_targets,
            indices,
            transform,
            image_mean,
            image_std,
        )

    train_dataset = make_dataset(train_idx, train_transform)
    train_evaluation_dataset = make_dataset(train_idx, evaluation_transform)
    val_dataset = make_dataset(val_idx, evaluation_transform)
    test_dataset = make_dataset(test_idx, evaluation_transform)
    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    train_evaluation_loader = DataLoader(
        train_evaluation_dataset, shuffle=False, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = EVAXBaseLoRARegressor(args.pretrained_path).to(device)
    model.log_parameter_status()
    criterion = nn.SmoothL1Loss(beta=1.0)

    lora_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A" in name or ".lora_B" in name:
            lora_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    parameter_groups = []
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": HEAD_LR})
    if lora_parameters:
        parameter_groups.append({"params": lora_parameters, "lr": LORA_LR, "weight_decay": 0.0})
    if not parameter_groups:
        raise RuntimeError("模型没有可训练参数。")

    optimizer = optim.AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-7
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
    }
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.set_backbone_train_mode()
        running_loss = 0.0
        sample_count = 0
        for batch_images, batch_targets in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            batch_images = batch_images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(batch_images).view(-1)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * batch_targets.shape[0]
            sample_count += int(batch_targets.shape[0])

        train_loss = running_loss / sample_count
        _, train_metrics, _, _ = evaluate(
            model,
            train_evaluation_loader,
            criterion,
            target_mean,
            target_std,
            device,
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, target_mean, target_std, device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path,
                model,
                target_mean,
                target_std,
                image_mean,
                image_std,
                label_column,
                epoch,
                val_metrics,
            )
            improvement_text = " | 已保存最佳模型"
        else:
            epochs_without_improvement += 1
            improvement_text = (
                f" | 未改进 {epochs_without_improvement}/{args.patience}"
            )

        scheduler.step()
        log(
            f"Epoch {epoch:03d} | train loss={train_loss:.6f} "
            f"RMSE={train_metrics['rmse']:.4f} | val loss={val_loss:.6f} "
            f"RMSE={val_metrics['rmse']:.4f} R2={val_metrics['r2']:.4f}"
            f"{improvement_text}"
        )
        if epochs_without_improvement >= args.patience:
            log("早停触发。")
            break

    checkpoint = load_torch_checkpoint(best_model_path)
    if checkpoint.get("label_column") != label_column:
        raise RuntimeError("最佳模型中的标签列与当前任务不一致。")
    model.load_state_dict(checkpoint["model_state_dict"])
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    train_loss, train_metrics, _, _ = evaluate(
        model,
        train_evaluation_loader,
        criterion,
        target_mean,
        target_std,
        device,
    )
    val_loss, val_metrics, _, _ = evaluate(
        model, val_loader, criterion, target_mean, target_std, device
    )
    test_loss, test_metrics, test_labels, test_predictions = evaluate(
        model, test_loader, criterion, target_mean, target_std, device
    )

    prediction_path = os.path.join(
        args.output_dir, f"test_predictions_{label_tag}_{run_stamp}.csv"
    )
    prediction_table = metadata.iloc[test_idx].reset_index(drop=True).copy()
    prediction_table["true_value"] = test_labels
    prediction_table["predicted_value"] = test_predictions
    prediction_table["residual"] = test_predictions - test_labels
    prediction_table.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    plot_path = os.path.join(
        args.output_dir, f"training_curves_{label_tag}_{run_stamp}.png"
    )
    save_training_plot(history, test_labels, test_predictions, plot_path)

    results = {
        "run_stamp": run_stamp,
        "task_name": task_name,
        "label_column": label_column,
        "excel_path": os.path.abspath(args.excel_path),
        "dicom_dir": os.path.abspath(args.dicom_dir),
        "pretrained_path": os.path.abspath(args.pretrained_path),
        "fixed_group_split_path": os.path.abspath(fixed_split_path),
        "seed": args.seed,
        "sample_counts": {
            "all": int(len(targets)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "target_mean_train": target_mean,
        "target_std_train": target_std,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "loss": {
            "train": float(train_loss),
            "val": float(val_loss),
            "test": float(test_loss),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "files": {
            "best_model": best_model_path,
            "fixed_group_split": fixed_split_path,
            "split_manifest": split_manifest_path,
            "test_predictions": prediction_path,
            "training_plot": plot_path,
        },
    }
    results_path = os.path.join(
        args.output_dir, f"results_{label_tag}_{run_stamp}.json"
    )
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, allow_nan=False)

    log("=== 最佳模型测试结果 ===")
    log(f"最佳 epoch: {best_epoch} | 最佳验证 RMSE: {best_val_rmse:.4f}")
    log(
        f"Test MAE={test_metrics['mae']:.4f} | RMSE={test_metrics['rmse']:.4f} | "
        f"R2={test_metrics['r2']:.4f} | Pearson r={test_metrics['pearson_r']:.4f} | "
        f"Bias={test_metrics['bias']:.4f}"
    )
    log(f"结果 JSON: {results_path}")
    log(f"测试集预测: {prediction_path}")
    log(f"训练曲线: {plot_path}")
    plt.show()
    log("训练曲线已显示")
    print("\n训练完成！")
    return results


if __name__ == "__main__":
    run_regression_task(
        label_column="fev1fvc_z",
        task_name="base_lora_fev1fvc_z_regression",
    )
