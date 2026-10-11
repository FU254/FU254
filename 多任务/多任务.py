#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA-X 医学影像分类训练脚本 - Base LoRA-分类 微调
冻结预训练 Backbone，仅训练注意力 LoRA-分类、最终归一化层和分类头
"""

import warnings

warnings.filterwarnings('ignore')

import os
import sys
import logging
import math
import random
import shutil
import hashlib
from functools import lru_cache
from datetime import datetime
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import pydicom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from cxr_shared import (
    PREPROCESS_VERSION,
    load_or_create_patient_split,
    normalize_accession,
    patient_key_from_sources,
    preprocess_cxr_dicom,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()


def log(message):
    logger.info(message)


sys.path.insert(0, os.path.join(PROJECT_DIR, 'classification'))
from models.models_eva import eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

IMAGE_SIZE = 224
img_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
TASK_NAME = 'base_lora_obstruction_multitask_fev1fvc_diff'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'runs', TASK_NAME)
EXCEL_PATH2 = r'/home/cfu/肺功能万能表最终版1_fev1fvc差值标签(1).xlsx'
DICOM_DIR = r'/data/users/cfu/DR数据_按文件10份-筛选合成'
PRETRAINED_PATH = os.path.join(PROJECT_DIR, 'eva_x_base_patch16_merged520k_mim.pt')

LABEL_COLUMN = '阻塞性通气障碍'
FEV1_FVC_MEASURED_COLUMN = 'fev1fvc实际值'
FEV1_FVC_LLN_COLUMN = 'fev1fvc_lln'
DIFF_LABEL_NAME = 'fev1fvc_diff'
LAMBDA_DIFF = 0.1
SEED = 42
LEARNING_RATE = 1e-4
LORA_LEARNING_RATE = 1e-4
CLINICAL_COLUMNS = ['age', 'sex', 'height', 'weight', 'smoking']
WEIGHT_DECAY = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_TARGET_MODULES = ('q_proj', 'v_proj')


def make_safe_name(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|\s]+', '_', value)
    return value.strip('_') or 'task'


LABEL_TAG = make_safe_name(LABEL_COLUMN)
PATIENT_SPLIT_MANIFEST_PATH = os.environ.get(
    'PATIENT_SPLIT_MANIFEST_PATH',
    os.path.join(os.path.dirname(EXCEL_PATH2), 'patient_split_manifest_fev1fvc_diff.csv'),
)
CACHE_FILE = os.path.join(OUTPUT_DIR, f'cached_data_direct_accession_{LABEL_TAG}.npz')
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, f'best_evax_base_lora_multitask_adapter_{LABEL_TAG}.pth')
RUN_INFO_FILE = os.path.join(OUTPUT_DIR, f'run_info_{LABEL_TAG}.txt')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_hardware():
    log("开始初始化硬件配置")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        requested_gpu = os.environ.get('CUDA_DEVICE_INDEX', '5').strip()
        try:
            preferred_gpu_index = int(requested_gpu)
        except ValueError:
            preferred_gpu_index = 5
        gpu_index = preferred_gpu_index if 0 <= preferred_gpu_index < gpu_count else 0
        device = torch.device(f'cuda:{gpu_index}')
        gpu_name = torch.cuda.get_device_name(gpu_index)
        if gpu_index != preferred_gpu_index:
            log(
                f"请求 cuda:{preferred_gpu_index}，但当前仅可见 {gpu_count} 张 GPU；"
                f"回退到 cuda:{gpu_index}"
            )
        log(f"使用设备: cuda:{gpu_index}")
        log(f"GPU型号: {gpu_name}")
    else:
        device = torch.device('cpu')
        log(f"使用设备: cpu")
    log("硬件配置初始化完成")
    return device


def parse_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def encode_binary_feature(value, positive_tokens):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return np.nan
        if text in positive_tokens:
            return 1.0
        if text in {'0', '1'}:
            return float(text)
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(numeric):
        return np.nan
    return numeric


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, scale=0.8):
        super().__init__()
        self.gamma = float(gamma)
        self.scale = float(scale)

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1).to(
            device=inputs.device,
            dtype=inputs.dtype,
        )

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none",
        )

        pt = torch.exp(-bce_loss)

        focal_loss = (
            self.scale
            * (1.0 - pt).pow(self.gamma)
            * bce_loss
        )

        return focal_loss.mean()


class LoRALinear(nn.Module):
    """为现有 Linear 权重添加低秩增量，同时兼容直接读取 layer.weight 的代码。"""

    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear 仅支持 nn.Linear，实际类型: {type(base_layer).__name__}")
        if rank <= 0:
            raise ValueError(f"LoRA-分类 rank 必须大于 0，实际值: {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for param in self.base_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        delta_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return self.base_layer.weight + delta_weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


def inject_lora_into_eva(backbone, rank, alpha, target_modules):
    injected_names = []
    for block_index, block in enumerate(backbone.blocks):
        for module_name in target_modules:
            base_layer = getattr(block.attn, module_name, None)
            if not isinstance(base_layer, nn.Linear):
                raise RuntimeError(
                    f"无法在 backbone.blocks.{block_index}.attn.{module_name} 注入 LoRA-分类；"
                    f"期望 nn.Linear，实际为 {type(base_layer).__name__}"
                )
            setattr(block.attn, module_name, LoRALinear(base_layer, rank=rank, alpha=alpha))
            injected_names.append(f"backbone.blocks.{block_index}.attn.{module_name}")
    return injected_names


class EVAXLoRAMultiTaskModel(nn.Module):
    def __init__(self, pretrained_weight_path=None, num_classes=1, lora_rank=LORA_RANK,
                 lora_alpha=LORA_ALPHA, clinical_dim=0):
        super(EVAXLoRAMultiTaskModel, self).__init__()

        log("创建EVA-X-base-patch16模型")
        self.backbone = eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(pretrained=False)

        if pretrained_weight_path is not None and os.path.exists(pretrained_weight_path):
            log(f"加载预训练权重: {pretrained_weight_path}")
            checkpoint = torch.load(pretrained_weight_path, map_location='cpu', weights_only=False)

            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    if k.startswith('module.'):
                        k = k[7:]
                    new_state_dict[k] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            log(f"权重加载完成")
            if msg.missing_keys:
                log(f"缺失的权重: {msg.missing_keys}")
            if msg.unexpected_keys:
                log(f"意外的权重: {msg.unexpected_keys}")
        else:
            log(f"警告: 未找到预训练权重或路径无效: {pretrained_weight_path}")

        self.backbone.fc_norm = nn.LayerNorm(self.backbone.embed_dim)
        self.bottleneck = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim // 2)
        self.bottleneck2 = nn.Linear(self.backbone.embed_dim // 2, self.backbone.embed_dim // 4)
        self.clinical_dim = clinical_dim
        shared_feature_dim = self.backbone.embed_dim // 4 + clinical_dim
        self.classifier = nn.Linear(shared_feature_dim, num_classes)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_module_names = inject_lora_into_eva(
            self.backbone,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
        )
        self._init_head()
        self._create_diff_regressor_without_advancing_rng(shared_feature_dim)
        self._configure_trainable_parameters()
        log(
            f"LoRA-分类 注入完成 | rank={self.lora_rank} | alpha={self.lora_alpha:g} | "
            f"目标层={LORA_TARGET_MODULES} | 注入线性层数={len(self.lora_module_names)}"
        )

    def _init_head(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def _create_diff_regressor_without_advancing_rng(self, feature_dim):
        # 保持分类模型的 LoRA、分类头和后续数据加载随机序列与单任务脚本一致。
        # 模型仍在 CPU 上构建，因此只需隔离 CPU RNG 状态。
        with torch.random.fork_rng(devices=[]):
            self.diff_regressor = nn.Linear(feature_dim, 1)
            nn.init.trunc_normal_(self.diff_regressor.weight, std=0.02)
            if self.diff_regressor.bias is not None:
                nn.init.constant_(self.diff_regressor.bias, 0)

    def _configure_trainable_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
        for param in self.backbone.fc_norm.parameters():
            param.requires_grad = True

    @staticmethod
    def _parameter_group_name(parameter_name):
        parts = parameter_name.split('.')
        if parameter_name.startswith('backbone.blocks.') and len(parts) >= 3:
            return '.'.join(parts[:3])
        if parameter_name.startswith('backbone.') and len(parts) >= 2:
            return '.'.join(parts[:2])
        return parts[0]

    def log_parameter_status(self):
        group_stats = {}
        for name, param in self.named_parameters():
            group_name = self._parameter_group_name(name)
            stats = group_stats.setdefault(group_name, {'trainable': 0, 'frozen': 0})
            state = 'trainable' if param.requires_grad else 'frozen'
            stats[state] += param.numel()

        frozen_groups = []
        trainable_groups = []
        partially_trainable_groups = []
        for name, stats in group_stats.items():
            if stats['trainable'] == 0:
                frozen_groups.append(name)
            elif stats['frozen'] == 0:
                trainable_groups.append(name)
            else:
                partially_trainable_groups.append(name)

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_count = trainable_count + frozen_count
        trainable_ratio = 100.0 * trainable_count / total_count if total_count else 0.0
        log(f"冻结 EVA-X 层: {', '.join(frozen_groups) if frozen_groups else '无'}")
        log(f"保留可训练层: {', '.join(trainable_groups) if trainable_groups else '无'}")
        if partially_trainable_groups:
            log(f"部分可训练层: {', '.join(partially_trainable_groups)}")
        log(
            f"参数状态 | 可训练: {trainable_count} | 冻结: {frozen_count} | "
            f"可训练占比: {trainable_ratio:.4f}%"
        )

    def trainable_state_dict(self):
        trainable_names = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        full_state_dict = self.state_dict()
        return {
            name: full_state_dict[name].detach().cpu()
            for name in trainable_names
        }

    def lora_config(self):
        return {
            'rank': self.lora_rank,
            'alpha': self.lora_alpha,
            'target_modules': list(LORA_TARGET_MODULES),
            'injected_module_count': len(self.lora_module_names),
        }

    def forward(self, x, clinical=None, verbose=False):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"模型前向传播日志")
            print(f"{'=' * 60}")
            print(f"[输入] 原始输入: shape={x.shape}, dtype={x.dtype}")
            print(f"       数值范围: [{x.min().item():.4f}, {x.max().item():.4f}]")

        # Patch Embedding
        x = self.backbone.patch_embed(x)
        if verbose:
            print(f"[1] Patch Embedding: shape={x.shape}")
            print(f"    处理: 将224x224图像切成14x14个patch，每个patch编码为向量")

        # 添加CLS Token
        batch_size, seq_len, embed_dim = x.size()
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if verbose:
            print(f"[2] 添加CLS Token: shape={x.shape}")
            print(f"    处理: 在序列开头添加[CLS]标记，用于分类任务")

        # 位置嵌入
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
            if verbose:
                print(f"[3] 添加位置嵌入: shape={x.shape}")
                print(f"    处理: 加上预训练的位置编码，提供位置信息")

        # 位置Dropout
        x = self.backbone.pos_drop(x)
        if verbose:
            print(f"[4] 位置Dropout: shape={x.shape}")
            print(f"    处理: 随机丢弃部分位置特征，防止过拟合")

        # Transformer Blocks
        rel_pos_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        if verbose:
            print(f"[5] Transformer Blocks: 共{len(self.backbone.blocks)}层")

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rel_pos_bias)
            if verbose and (i == 0 or i == len(self.backbone.blocks) - 1):
                print(f"    Block {i + 1}: shape={x.shape}")

        # LayerNorm
        x = self.backbone.norm(x)
        if verbose:
            print(f"[6] 层归一化: shape={x.shape}")
            print(f"    处理: 对Transformer输出进行层归一化")

        # 提取特征（取[CLS]标记或均值）
        features = self.backbone.fc_norm(x.mean(1))
        if verbose:
            print(f"[7] 特征提取: shape={features.shape}")
            print(f"    处理: 对所有token取均值作为图像特征表示")

        # 分类头 - 第一层
        x = self.bottleneck(features)
        if verbose:
            print(f"[8] Bottleneck1: shape={x.shape}")
            print(f"    处理: 全连接层，维度从{features.shape[1]}降至{x.shape[1]}")

        x = nn.ReLU()(x)
        if verbose:
            print(f"[9] ReLU激活: shape={x.shape}")
            print(f"    处理: 引入非线性，增加模型表达能力")

        # 分类头 - 第二层
        x = self.bottleneck2(x)
        if verbose:
            print(f"[10] Bottleneck2: shape={x.shape}")
            print(f"    处理: 全连接层，维度进一步降至{x.shape[1]}")

        x = nn.ReLU()(x)
        if verbose:
            print(f"[11] ReLU激活: shape={x.shape}")

        # 共享特征同时供分类主任务和回归辅助任务使用
        if clinical is not None:
            x = torch.cat([x, clinical], dim=1)
        class_logits = self.classifier(x)
        diff_prediction = self.diff_regressor(x)
        if verbose:
            print(f"[12] 分类输出: shape={class_logits.shape}")
            print(f"[13] FEV1/FVC-LLN 差值回归输出: shape={diff_prediction.shape}")
            print(f"{'=' * 60}")

        return class_logits, diff_prediction


def mixup_data(x_img, class_labels, diff_values, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_img.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_img = lam * x_img + (1 - lam) * x_img[index, :]
    class_a, class_b = class_labels, class_labels[index]
    diff_a, diff_b = diff_values, diff_values[index]
    return mixed_img, class_a, class_b, diff_a, diff_b, lam


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


def split_dataset(metadata, y, seed):
    train_idx, val_idx, test_idx, _ = load_or_create_patient_split(
        metadata=metadata,
        classification_labels=y,
        z_scores=metadata['diff_value'].to_numpy(dtype=np.float32),
        manifest_path=PATIENT_SPLIT_MANIFEST_PATH,
        seed=seed,
        log_fn=log,
    )
    return train_idx, val_idx, test_idx


def excel_source_signature(excel_path):
    stat = os.stat(excel_path)
    text = f"{os.path.abspath(excel_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def extract_clinical_features(row):
    return np.array([
        parse_float(row.get('age')),
        encode_binary_feature(row.get('sex'), {'1', 'm', 'male', '男'}),
        parse_float(row.get('height')),
        parse_float(row.get('weight')),
        encode_binary_feature(row.get('smoking'), {'1', 'yes', 'y', 'true', 'current', 'former', '吸烟', '已戒烟', '现在吸烟'}),
    ], dtype=np.float32)


def log_diff_data_check(diff_values, class_labels, source, missing_count=0):
    diff_values = np.asarray(diff_values, dtype=np.float64)
    class_labels = np.asarray(class_labels, dtype=np.int64)
    valid = np.isfinite(diff_values)
    missing_count += int((~valid).sum())
    if not valid.any():
        log(f"diff 数据检查 ({source}) | 缺失数量: {missing_count} | 无有效 diff 标签")
        return

    valid_diff = diff_values[valid]
    expected_labels = (valid_diff < 0).astype(np.int64)
    consistency = np.mean(expected_labels == class_labels[valid])
    log(
        f"diff 数据检查 ({source}) | min: {valid_diff.min():.6f} | "
        f"max: {valid_diff.max():.6f} | mean: {valid_diff.mean():.6f} | "
        f"std: {valid_diff.std():.6f} | 缺失数量: {missing_count} | "
        f"diff<0 与分类标签=1 一致率: {consistency:.6f}"
    )


def load_and_preprocess_data(force_reload=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = CACHE_FILE
    current_source_signature = excel_source_signature(EXCEL_PATH2)

    if not force_reload and os.path.exists(cache_file):
        log("从缓存加载数据...")
        cached = np.load(cache_file, allow_pickle=True)
        cached_version = (
            str(cached['preprocess_version'].item())
            if 'preprocess_version' in cached.files
            else 'missing'
        )
        log(f"缓存预处理版本: {cached_version} | 当前版本: {PREPROCESS_VERSION}")
        log(f"缓存字段: {', '.join(cached.files)}")
        required_cache_fields = {
            'X', 'y', 'accessions', 'filenames', 'patient_keys', 'diff_values',
            'view_positions', 'patient_positions', 'ap_flags', 'obvious_cropping',
            'low_inspiration', 'quality_flags',
            'preprocess_version', 'source_signature',
        }
        if (
            required_cache_fields.issubset(cached.files)
            and str(cached['preprocess_version'].item()) == PREPROCESS_VERSION
            and str(cached['source_signature'].item()) == current_source_signature
        ):
            X = cached['X']
            y = cached['y']
            metadata = pd.DataFrame({
                '放射编号': cached['accessions'].astype(str),
                'filename': cached['filenames'].astype(str),
                'patient_key': cached['patient_keys'].astype(str),
                'diff_value': cached['diff_values'].astype(np.float32),
                'view_position': cached['view_positions'].astype(str),
                'patient_position': cached['patient_positions'].astype(str),
                'ap_flag': cached['ap_flags'].astype(bool),
                'obvious_cropping': cached['obvious_cropping'].astype(bool),
                'low_inspiration': cached['low_inspiration'].astype(str),
                'quality_flag': cached['quality_flags'].astype(str),
            })
        else:
            missing_cache_fields = sorted(required_cache_fields.difference(cached.files))
            log(
                "缓存版本/字段不匹配，将重新读取 DICOM 并生成缓存"
                f" | 缺失字段: {missing_cache_fields}"
            )
            metadata = None
        if metadata is not None:
            log("缓存命中：跳过 DICOM 读取，直接使用已预处理图像")
            log(f"数据从缓存加载完成，共 {len(X)} 个样本")
            log(f"当前缓存路径: {cache_file}")
            class_counts = np.bincount(y.astype(int), minlength=2)
            log(f"缓存类别分布: 0类 {class_counts[0]}, 1类 {class_counts[1]}")
            log_diff_data_check(metadata['diff_value'].to_numpy(), y, '缓存')
            return X, y, metadata
    if force_reload and os.path.exists(cache_file):
        log("强制重新加载数据，删除旧缓存...")
        os.remove(cache_file)

    log("开始加载真实DICOM数据...")

    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)
    df2.columns = [str(column).strip() for column in df2.columns]

    required_columns = {
        '放射编号', LABEL_COLUMN, FEV1_FVC_MEASURED_COLUMN, FEV1_FVC_LLN_COLUMN,
    }
    missing_columns = required_columns.difference(df2.columns)
    if missing_columns:
        log(f"错误：Excel 缺少必要列: {sorted(missing_columns)}")
        log(f"错误：df2中不存在'放射编号'列！")
        log(f"请检查Excel文件，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([]), None
    df2['放射编号'] = df2['放射编号'].map(normalize_accession)

    log(f"df2中的列名：{df2.columns.tolist()}")

    X, y = [], []
    accessions, filenames, patient_keys, diff_values = [], [], [], []
    view_positions, patient_positions = [], []
    ap_flags, obvious_cropping = [], []
    low_inspiration, quality_flags = [], []
    filter_reasons = {
        'other_error': 0,
        'no_pixel_data': 0,
        'invalid_label': 0,
        'missing_or_invalid_diff': 0,
        'missing_patient_key': 0,
        'missing_label_column': 0
    }
    filtered_filenames = []

    for filename in tqdm(sorted(os.listdir(DICOM_DIR)), desc="Processing DICOM"):
        if filename.lower().endswith('.dcm'):
            try:
                dcm = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
                if 'PixelData' not in dcm:
                    print(f"跳过无像素数据文件：{filename}")
                    filter_reasons['no_pixel_data'] += 1
                    filtered_filenames.append(filename)
                    continue

                accession_str = normalize_accession(getattr(dcm, 'AccessionNumber', ''))
                if not accession_str:
                    filter_reasons['invalid_label'] += 1
                    filtered_filenames.append(filename)
                    continue
                label_df = df2[df2['放射编号'] == accession_str]

                if len(label_df) == 0:
                    print(f"警告：放射编号 {accession_str} 无标签，跳过该样本")
                    filter_reasons['invalid_label'] += 1
                    filtered_filenames.append(filename)
                    continue
                else:
                    try:
                        excel_row = label_df.iloc[0]
                        label_value = excel_row[LABEL_COLUMN]
                        label_numeric = pd.to_numeric(label_value, errors='coerce')
                        if (
                            pd.isna(label_numeric)
                            or not np.isfinite(float(label_numeric))
                            or float(label_numeric) not in {0.0, 1.0}
                        ):
                            print(f"警告：放射编号 {accession_str} 的标签无效（{label_value}），跳过该样本")
                            filter_reasons['invalid_label'] += 1
                            filtered_filenames.append(filename)
                            continue
                        else:
                            label = int(float(label_numeric))
                            print(f"放射编号 {accession_str} 的标签为: {label}")
                    except (ValueError, IndexError) as e:
                        print(f"警告：放射编号 {accession_str} 的标签无法转换为整数或数据缺失，错误: {str(e)}，跳过该样本")
                        filter_reasons['invalid_label'] += 1
                        filtered_filenames.append(filename)
                        continue
                    except KeyError as e:
                        print(f"警告：放射编号 {accession_str} 的标签列不存在: {str(e)}，跳过该样本")
                        filter_reasons['missing_label_column'] += 1
                        filtered_filenames.append(filename)
                        continue

                measured_fev1_fvc = pd.to_numeric(
                    excel_row[FEV1_FVC_MEASURED_COLUMN], errors='coerce'
                )
                fev1_fvc_lln = pd.to_numeric(
                    excel_row[FEV1_FVC_LLN_COLUMN], errors='coerce'
                )
                if (
                    pd.isna(measured_fev1_fvc)
                    or pd.isna(fev1_fvc_lln)
                    or not np.isfinite(float(measured_fev1_fvc))
                    or not np.isfinite(float(fev1_fvc_lln))
                ):
                    print(
                        f"警告：放射编号 {accession_str} 的 "
                        f"{FEV1_FVC_MEASURED_COLUMN} 或 {FEV1_FVC_LLN_COLUMN} 无效，跳过该样本"
                    )
                    filter_reasons['missing_or_invalid_diff'] += 1
                    filtered_filenames.append(filename)
                    continue
                diff_value = float(measured_fev1_fvc) - float(fev1_fvc_lln)

                patient_key = patient_key_from_sources(excel_row, dcm)
                if not patient_key:
                    print(f"警告：放射编号 {accession_str} 缺少患者标识，跳过该样本")
                    filter_reasons['missing_patient_key'] += 1
                    filtered_filenames.append(filename)
                    continue

                img, image_metadata = preprocess_cxr_dicom(
                    dcm, img_SIZE, return_metadata=True
                )

                X.append(img)
                y.append(label)
                accessions.append(accession_str)
                filenames.append(filename)
                patient_keys.append(patient_key)
                diff_values.append(diff_value)
                view_positions.append(image_metadata['view_position'])
                patient_positions.append(image_metadata['patient_position'])
                ap_flags.append(image_metadata['ap_flag'])
                obvious_cropping.append(image_metadata['obvious_cropping'])
                low_inspiration.append(image_metadata['low_inspiration'])
                quality_flags.append(image_metadata['quality_flag'])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                filter_reasons['other_error'] += 1
                filtered_filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    log(f"过滤原因统计: {filter_reasons}")
    log(f"被过滤文件数: {len(filtered_filenames)}")

    log(f"数据加载完成，共 {len(X)} 个样本")

    class_counts = np.bincount(y.astype(int))
    log(f"类别分布：0类: {class_counts[0]}, 1类: {class_counts[1]}")

    log("保存数据到缓存...")
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        accessions=np.array(accessions, dtype=str),
        filenames=np.array(filenames, dtype=str),
        patient_keys=np.array(patient_keys, dtype=str),
        diff_values=np.array(diff_values, dtype=np.float32),
        view_positions=np.array(view_positions, dtype=str),
        patient_positions=np.array(patient_positions, dtype=str),
        ap_flags=np.array(ap_flags, dtype=bool),
        obvious_cropping=np.array(obvious_cropping, dtype=bool),
        low_inspiration=np.array(low_inspiration, dtype=str),
        quality_flags=np.array(quality_flags, dtype=str),
        preprocess_version=np.asarray(PREPROCESS_VERSION),
        source_signature=np.asarray(current_source_signature),
    )
    log(f"缓存已保存到: {cache_file}")

    metadata = pd.DataFrame({
        '放射编号': accessions,
        'filename': filenames,
        'patient_key': patient_keys,
        'diff_value': diff_values,
        'view_position': view_positions,
        'patient_position': patient_positions,
        'ap_flag': ap_flags,
        'obvious_cropping': obvious_cropping,
        'low_inspiration': low_inspiration,
        'quality_flag': quality_flags,
    })
    log_diff_data_check(
        metadata['diff_value'].to_numpy(),
        y,
        '原始数据',
        missing_count=filter_reasons['missing_or_invalid_diff'],
    )
    return X, y, metadata


def save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp):
    if metadata is None:
        log("当前缓存不包含放射编号元数据，无法导出 split 清单")
        return None

    split_df = metadata.copy()
    split_df['split'] = ''
    split_df.loc[train_idx, 'split'] = 'train'
    split_df.loc[val_idx, 'split'] = 'val'
    split_df.loc[test_idx, 'split'] = 'test'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_path = os.path.join(OUTPUT_DIR, f'evax_base_lora_split_{LABEL_TAG}_{run_stamp}.csv')
    split_df.to_csv(split_path, index=False, encoding='utf-8-sig')
    log(f"已导出本次数据划分: {split_path}")
    return split_path


def save_label_manifest(metadata, y, run_stamp):
    if metadata is None or len(metadata) != len(y):
        log("当前数据缺少完整元数据，无法导出 label manifest")
        return None, None

    manifest_df = metadata.copy()
    manifest_df['label'] = y.astype(int)
    manifest_path = os.path.join(OUTPUT_DIR, f'label_manifest_{LABEL_TAG}_{run_stamp}.csv')
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8-sig')

    signature_source = (
        manifest_df['放射编号'].astype(str) + '|' +
        manifest_df['label'].astype(str)
    )
    label_signature = hashlib.sha1('\n'.join(signature_source.tolist()).encode('utf-8')).hexdigest()
    log(f"标签清单已导出: {manifest_path}")
    log(f"标签签名: {label_signature}")
    return manifest_path, label_signature


def write_run_info(run_stamp, cache_path, best_model_path, split_path, manifest_path, label_signature, y):
    class_counts = np.bincount(y.astype(int))
    neg_count = int(class_counts[0]) if len(class_counts) > 0 else 0
    pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
    lines = [
        f"run_stamp={run_stamp}",
        f"task_name={TASK_NAME}",
        f"label_column={LABEL_COLUMN}",
        f"fev1_fvc_measured_column={FEV1_FVC_MEASURED_COLUMN}",
        f"fev1_fvc_lln_column={FEV1_FVC_LLN_COLUMN}",
        f"diff_label_name={DIFF_LABEL_NAME}",
        f"lambda_diff={LAMBDA_DIFF}",
        f"label_tag={LABEL_TAG}",
        f"excel_path={EXCEL_PATH2}",
        f"cache_path={cache_path}",
        f"best_model_path={best_model_path}",
        f"patient_split_manifest_path={PATIENT_SPLIT_MANIFEST_PATH}",
        f"preprocess_version={PREPROCESS_VERSION}",
        f"split_path={split_path or ''}",
        f"label_manifest_path={manifest_path or ''}",
        f"label_signature={label_signature or ''}",
        f"sample_count={len(y)}",
        f"class_0_count={neg_count}",
        f"class_1_count={pos_count}",
        f"lora_rank={LORA_RANK}",
        f"lora_alpha={LORA_ALPHA}",
        f"lora_target_modules={','.join(LORA_TARGET_MODULES)}",
        f"head_learning_rate={LEARNING_RATE}",
        f"lora_learning_rate={LORA_LEARNING_RATE}",
    ]
    with open(RUN_INFO_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"任务信息已保存: {RUN_INFO_FILE}")


class MedicalImageDataset(Dataset):
    def __init__(self, images, class_labels, diff_values, transform=None,
                 normalize_mean=None, normalize_std=None):
        self.images = images
        self.class_labels = class_labels
        self.diff_values = diff_values
        self.transform = transform
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        if len(images) != len(class_labels) or len(images) != len(diff_values):
            raise ValueError("图像、分类标签和回归标签的样本数量必须一致。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        if self.normalize_mean is not None and self.normalize_std is not None:
            img = (img - self.normalize_mean) / self.normalize_std

        class_label = torch.tensor(self.class_labels[idx], dtype=torch.float)
        diff_value = torch.tensor(self.diff_values[idx], dtype=torch.float)
        return img, class_label, diff_value


def compute_multitask_loss(class_logits, diff_prediction, class_labels, diff_values,
                           classification_criterion, diff_criterion):
    classification_loss = classification_criterion(class_logits.view(-1), class_labels)
    diff_loss = diff_criterion(diff_prediction.view(-1), diff_values)
    total_loss = classification_loss + LAMBDA_DIFF * diff_loss
    return total_loss, classification_loss, diff_loss


def compute_diff_metrics(diff_predictions, diff_targets):
    diff_predictions = np.asarray(diff_predictions, dtype=np.float64)
    diff_targets = np.asarray(diff_targets, dtype=np.float64)
    diff_mae = np.mean(np.abs(diff_predictions - diff_targets))
    diff_rmse = np.sqrt(np.mean((diff_predictions - diff_targets) ** 2))
    if len(diff_targets) < 2 or np.std(diff_predictions) == 0 or np.std(diff_targets) == 0:
        diff_pearson = float('nan')
    else:
        diff_pearson = float(np.corrcoef(diff_predictions, diff_targets)[0, 1])
    return diff_mae, diff_rmse, diff_pearson


def validate(model, val_loader, classification_criterion, diff_criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_diff_predictions = []
    all_diff_targets = []

    with torch.no_grad():
        for img, class_labels, diff_values in val_loader:
            img = img.to(device)
            class_labels = class_labels.to(device).float()
            diff_values = diff_values.to(device).float()
            class_logits, diff_prediction = model(img)
            loss, _, _ = compute_multitask_loss(
                class_logits,
                diff_prediction,
                class_labels,
                diff_values,
                classification_criterion,
                diff_criterion,
            )
            val_loss += loss.item()

            probs = torch.sigmoid(class_logits.view(-1)).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(class_labels.cpu().numpy())
            all_diff_predictions.extend(diff_prediction.view(-1).cpu().numpy())
            all_diff_targets.extend(diff_values.cpu().numpy())

    val_loss /= len(val_loader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pred_labels = (all_preds > 0.5).astype(int)
    accuracy = np.mean(pred_labels == all_labels)
    f1 = 2 * np.sum((pred_labels == 1) & (all_labels == 1))
    f1_denominator = 2 * np.sum((pred_labels == 1) & (all_labels == 1)) + np.sum(
        (pred_labels == 1) & (all_labels == 0)
    ) + np.sum((pred_labels == 0) & (all_labels == 1))
    f1 = f1 / f1_denominator if f1_denominator else 0.0

    if np.unique(all_labels).size < 2:
        roc_auc = 0.5
        pr_auc = float(all_labels.mean())
    else:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

        roc_auc = roc_auc_score(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

    diff_mae, diff_rmse, diff_pearson = compute_diff_metrics(
        all_diff_predictions, all_diff_targets
    )

    return val_loss, accuracy, roc_auc, pr_auc, f1, diff_mae, diff_rmse, diff_pearson


def main():
    global device

    set_seed()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = setup_hardware()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"数据配置 - 图像尺寸: {IMAGE_SIZE}, 批次大小: {BATCH_SIZE}, 工作进程数: {NUM_WORKERS}")
    log(f"当前任务: {TASK_NAME}")
    log(f"标签来源: {EXCEL_PATH2}")
    log(f"标签列: {LABEL_COLUMN}")
    log(f"任务输出目录: {OUTPUT_DIR}")
    log(f"缓存文件: {CACHE_FILE}")
    log(f"最佳模型文件: {BEST_MODEL_FILE}")

    force_reload_cache = os.environ.get('FORCE_RELOAD_CACHE', '0').strip().lower() in {
        '1', 'true', 'yes', 'y'
    }
    if force_reload_cache:
        log("FORCE_RELOAD_CACHE 已启用：本次将强制重新读取 DICOM")
    X, y, metadata = load_and_preprocess_data(force_reload=force_reload_cache)

    if len(X) == 0:
        log("数据加载失败，训练终止")
        return

    train_idx, val_idx, test_idx = split_dataset(metadata, y, SEED)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    diff_values = metadata['diff_value'].to_numpy(dtype=np.float32)
    diff_train = diff_values[train_idx]
    diff_val = diff_values[val_idx]
    diff_test = diff_values[test_idx]

    log(f"数据集划分完成 | 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")
    split_path = save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp)
    manifest_path, label_signature = save_label_manifest(metadata, y, run_stamp)
    write_run_info(run_stamp, CACHE_FILE, BEST_MODEL_FILE, split_path, manifest_path, label_signature, y)

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    log(f"数据归一化参数 - 均值: {X_train_mean:.4f}, 标准差: {X_train_std:.4f}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=4,
            translate=(0.02, 0.02),
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = MedicalImageDataset(X_train, y_train, diff_train, transform=train_transform,
                                        normalize_mean=X_train_mean, normalize_std=X_train_std)
    val_dataset = MedicalImageDataset(X_val, y_val, diff_val, transform=val_test_transform,
                                      normalize_mean=X_train_mean, normalize_std=X_train_std)
    test_dataset = MedicalImageDataset(X_test, y_test, diff_test, transform=val_test_transform,
                                       normalize_mean=X_train_mean, normalize_std=X_train_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    log(f"创建数据集 - 样本数量: {len(train_dataset)}, 转换策略: 训练增强")
    log(f"创建数据集 - 样本数量: {len(val_dataset)}, 转换策略: 验证/测试")
    log(f"创建数据集 - 样本数量: {len(test_dataset)}, 转换策略: 验证/测试")

    log("初始化EVA-X分类模型 - Base LoRA版本")

    pretrained_path = PRETRAINED_PATH

    model = EVAXLoRAMultiTaskModel(
        pretrained_weight_path=pretrained_path,
        num_classes=1,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    )
    model = model.to(device)

    log("模型初始化完成")
    model.log_parameter_status()

    train_class_counts = np.bincount(y_train.astype(int), minlength=2)
    train_neg_count = int(train_class_counts[0])
    train_pos_count = int(train_class_counts[1])
    if train_pos_count == 0:
        raise RuntimeError("训练集没有正样本，无法进行二分类训练。")
    class_ratio = train_neg_count / train_pos_count
    classification_criterion = FocalLoss(
        gamma=2.0,
        scale=0.8,
    )
    diff_criterion = nn.SmoothL1Loss()
    log(
        f"训练集类别分布 | 0类={train_neg_count} | 1类={train_pos_count} | "
        f"负/正比例={class_ratio:.6f}"
    )
    log(
        f"损失函数: 分类 FocalLoss(scale=0.8, gamma=2.0) + "
        f"{LAMBDA_DIFF:g} × diff SmoothL1Loss | "
        f"回归标签: {DIFF_LABEL_NAME} = {FEV1_FVC_MEASURED_COLUMN} - {FEV1_FVC_LLN_COLUMN}"
    )

    patience = 30
    no_improvement_epochs = 0
    total_epochs = 300
    best_val_auprc = -float('inf')
    best_model_path = BEST_MODEL_FILE

    # 测试模型前向传播日志
    log("测试模型前向传播...")
    test_input = torch.randn(2, 3, 224, 224).to(device)  # batch_size=2, 3通道, 224x224
    with torch.no_grad():
        model.eval()
        _ = model(test_input, verbose=True)

    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.lora_A' in name or '.lora_B' in name:
            lora_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
        })
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': LORA_LEARNING_RATE,
            'weight_decay': 0.0,
        })
    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查 LoRA-分类 注入和分类头配置")
    if not lora_params:
        raise RuntimeError("未找到 LoRA-分类 可训练参数，LoRA-分类 注入失败")

    log(f"分类头可训练参数数量: {sum(p.numel() for p in head_params)} | 学习率: {LEARNING_RATE}")
    log(
        f"LoRA-分类 可训练参数数量: {sum(p.numel() for p in lora_params)} | "
        f"学习率: {LORA_LEARNING_RATE} | weight_decay: 0"
    )

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)

    log(f"开始训练，共 {total_epochs} 个 epoch")
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    train_pr_aucs = []
    val_pr_aucs = []

    for epoch in range(total_epochs):
        log(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")

        model.train()
        train_loss = 0

        for batch_idx, (img, class_labels, diff_values) in enumerate(tqdm(train_loader, desc="Training")):
            img = img.to(device)
            class_labels = class_labels.to(device).float()
            diff_values = diff_values.to(device).float()
            optimizer.zero_grad()

            if random.random() < 0.4:
                mixed_img, class_a, class_b, diff_a, diff_b, lam = mixup_data(
                    img, class_labels, diff_values
                )
                class_logits, diff_prediction = model(mixed_img)
                classification_loss = mixup_criterion(
                    classification_criterion, class_logits.view(-1), class_a, class_b, lam
                )
                diff_loss = mixup_criterion(
                    diff_criterion, diff_prediction.view(-1), diff_a, diff_b, lam
                )
                loss = classification_loss + LAMBDA_DIFF * diff_loss
            else:
                class_logits, diff_prediction = model(img)
                loss, _, _ = compute_multitask_loss(
                    class_logits,
                    diff_prediction,
                    class_labels,
                    diff_values,
                    classification_criterion,
                    diff_criterion,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 计算训练集 ROC-AUC 和 PR-AUC
        train_loss_eval, train_acc, train_auc, train_pr_auc, train_f1, train_diff_mae, train_diff_rmse, train_diff_pearson = validate(
            model, train_loader, classification_criterion, diff_criterion
        )
        train_aucs.append(train_auc)
        train_pr_aucs.append(train_pr_auc)

        val_loss, val_acc, val_auc, val_pr_auc, val_f1, val_diff_mae, val_diff_rmse, val_diff_pearson = validate(
            model, val_loader, classification_criterion, diff_criterion
        )
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_pr_aucs.append(val_pr_auc)

        if val_pr_auc > best_val_auprc:
            best_val_auprc = val_pr_auc
            no_improvement_epochs = 0
            torch.save(
                {
                    'model': model.trainable_state_dict(),
                    'lora_config': model.lora_config(),
                    'best_val_auprc': float(best_val_auprc),
                    'diff_label_name': DIFF_LABEL_NAME,
                    'lambda_diff': LAMBDA_DIFF,
                },
                best_model_path,
            )
            log(f"★ 最佳 LoRA-分类 adapter 已保存 (PR-AUC: {best_val_auprc:.4f})")
        else:
            no_improvement_epochs += 1
            log(f"未改进: {no_improvement_epochs}/{patience}")

        scheduler.step(epoch + 1)
        head_lr = optimizer.param_groups[0]['lr']
        lora_lr = optimizer.param_groups[1]['lr']
        log(f"当前学习率 | 分类头: {head_lr:.8f} | LoRA-分类: {lora_lr:.8f}")

        log(
            f"Classification | 训练 AUROC: {train_auc:.4f} | PR-AUC: {train_pr_auc:.4f} | "
            f"Accuracy: {train_acc:.4f} | F1: {train_f1:.4f} | 验证 AUROC: {val_auc:.4f} | "
            f"PR-AUC: {val_pr_auc:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} || "
            f"Regression | 训练 diff MAE: {train_diff_mae:.4f} | diff RMSE: {train_diff_rmse:.4f} | "
            f"Pearson: {train_diff_pearson:.4f} | 验证 diff MAE: {val_diff_mae:.4f} | "
            f"diff RMSE: {val_diff_rmse:.4f} | Pearson: {val_diff_pearson:.4f}"
        )

        if no_improvement_epochs >= patience:
            log(f"早停触发，训练结束")
            break

    log("加载最佳模型进行测试...")
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, weights_only=False, map_location='cpu')
            adapter_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
            load_result = model.load_state_dict(adapter_state, strict=False)
            if load_result.unexpected_keys:
                raise RuntimeError(f"adapter 包含意外参数: {load_result.unexpected_keys}")
            log(f"成功加载最佳 LoRA-分类 adapter: {best_model_path}")
        except Exception as e:
            log(f"警告: 无法加载模型文件 {best_model_path}, 错误: {str(e)}")
            log("使用当前训练的模型进行测试...")
    else:
        log(f"警告: 模型文件 {best_model_path} 不存在")
        log("使用当前训练的模型进行测试...")
    model.eval()

    if os.path.exists(best_model_path):
        archived_model_path = os.path.join(
            OUTPUT_DIR,
            f"best_evax_base_lora_multitask_adapter_prauc{best_val_auprc:.4f}_{run_stamp}.pth"
        )
        shutil.copy2(best_model_path, archived_model_path)
        log(f"已归档最佳 LoRA-分类 adapter: {archived_model_path}")

    def collect_predictions(data_loader, desc):
        all_probs = []
        all_labels = []
        all_diff_predictions = []
        all_diff_targets = []
        with torch.no_grad():
            for img, class_labels, diff_values in tqdm(data_loader, desc=desc):
                img = img.to(device)
                class_labels = class_labels.to(device).float()
                diff_values = diff_values.to(device).float()
                class_logits, diff_prediction = model(img)
                probs = torch.sigmoid(class_logits.view(-1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(class_labels.cpu().numpy())
                all_diff_predictions.extend(diff_prediction.view(-1).cpu().numpy())
                all_diff_targets.extend(diff_values.cpu().numpy())
        return (
            np.array(all_labels),
            np.array(all_probs),
            np.array(all_diff_targets),
            np.array(all_diff_predictions),
        )

    def compute_metrics(labels, probs, threshold):
        from sklearn.metrics import roc_auc_score, confusion_matrix

        pred_labels = (np.array(probs) > threshold).astype(int)
        accuracy = np.mean(pred_labels == np.array(labels))
        auc = roc_auc_score(labels, probs)
        cm = confusion_matrix(labels, pred_labels, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'f1': f1,
            'confusion_matrix': cm,
        }

    def plot_pr_curve(labels, probs, output_path):
        from sklearn.metrics import auc, precision_recall_curve

        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        positive_rate = np.mean(labels)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', linewidth=2,
                 label=f'Test PR curve (PR-AUC = {pr_auc:.4f})')
        plt.axhline(positive_rate, color='navy', linestyle='--', linewidth=2,
                    label=f'No skill (positive rate = {positive_rate:.4f})')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('EVA-X Base LoRA-分类 Test Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return pr_auc

    def find_optimal_threshold(labels, probs):
        best_threshold = 0.5
        best_score = -1.0
        best_metrics = compute_metrics(labels, probs, best_threshold)

        for threshold in np.arange(0.05, 0.96, 0.01):
            metrics = compute_metrics(labels, probs, threshold)
            youden = metrics['sensitivity'] + metrics['specificity'] - 1
            if youden > best_score or (
                np.isclose(youden, best_score) and metrics['f1'] > best_metrics['f1']
            ):
                best_score = youden
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics, best_score

    all_val_labels, all_val_probs, _, _ = collect_predictions(val_loader, "Selecting Threshold")
    best_threshold, val_threshold_metrics, best_youden = find_optimal_threshold(all_val_labels, all_val_probs)
    all_test_labels, all_test_probs, all_test_diff_targets, all_test_diff_predictions = collect_predictions(
        test_loader, "Testing"
    )
    best_metrics = compute_metrics(all_test_labels, all_test_probs, best_threshold)
    test_diff_mae, test_diff_rmse, test_diff_pearson = compute_diff_metrics(
        all_test_diff_predictions, all_test_diff_targets
    )
    pr_curve_path = os.path.join(
        OUTPUT_DIR, f"evax_base_lora_test_pr_curve_{LABEL_TAG}_{run_stamp}.png"
    )
    test_pr_auc = plot_pr_curve(all_test_labels, all_test_probs, pr_curve_path)

    log(f"\n=== 结果总结 ===")
    log(f"最佳验证PR-AUC: {best_val_auprc:.4f}")
    log(f"验证集最佳阈值: {best_threshold:.4f}")
    log(f"验证集Youden指数: {best_youden:.4f}")
    log(f"Classification | AUROC: {best_metrics['auc']:.4f} | PR-AUC: {test_pr_auc:.4f}")
    log(f"Classification | Accuracy: {best_metrics['accuracy']:.4f} | F1: {best_metrics['f1']:.4f}")
    log(f"测试灵敏度: {best_metrics['sensitivity']:.4f}")
    log(f"测试特异度: {best_metrics['specificity']:.4f}")
    log(f"测试Precision: {best_metrics['precision']:.4f}")
    log(f"测试NPV: {best_metrics['npv']:.4f}")
    log(
        f"Regression | diff MAE: {test_diff_mae:.4f} | "
        f"diff RMSE: {test_diff_rmse:.4f} | Pearson: {test_diff_pearson:.4f}"
    )
    log(f"测试混淆矩阵 [[TN, FP], [FN, TP]]: {best_metrics['confusion_matrix'].tolist()}")
    log(f"测试PR曲线已保存: {pr_curve_path}")

    log("\n绘制训练曲线...")
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_aucs, label='Train AUC', color='blue')
    plt.plot(val_aucs, label='Val AUC', color='orange')
    plt.title('Training & Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_pr_aucs, label='Train PR-AUC', color='blue')
    plt.plot(val_pr_aucs, label='Val PR-AUC', color='orange')
    plt.title('Training & Validation PR-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR-AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    log("训练曲线已显示")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
