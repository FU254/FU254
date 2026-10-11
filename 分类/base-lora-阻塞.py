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
TASK_NAME = 'base_lora_obstruction_final1'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'runs', TASK_NAME)
EXCEL_PATH2 = r"/home/cfu/肺功能万能表最终版1.xlsx"
DICOM_DIR = r'/data/users/cfu/DR数据'
PRETRAINED_PATH = os.path.join(PROJECT_DIR, 'eva_x_base_patch16_merged520k_mim.pt')

LABEL_COLUMN = '阻塞性通气障碍'
REGRESSION_SPLIT_COLUMN = 'fev1fvc_z'
SEED = 42
LEARNING_RATE = 1e-4

LORA_LEARNING_RATE = 1e-4
CLINICAL_COLUMNS = ['age', 'sex', 'height', 'weight', 'smoking']
WEIGHT_DECAY = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_TARGET_MODULES = ('q_proj', 'v_proj', 'k_proj')


def make_safe_name(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|\s]+', '_', value)
    return value.strip('_') or 'task'


LABEL_TAG = make_safe_name(LABEL_COLUMN)
PATIENT_SPLIT_MANIFEST_PATH = os.environ.get(
    "PATIENT_SPLIT_MANIFEST_PATH",
    os.path.join(
        os.path.dirname(EXCEL_PATH2),
        f"patient_split_manifest_seed{SEED}.csv",
    ),
)
CACHE_FILE = os.path.join(OUTPUT_DIR, f'cached_data_direct_accession_{LABEL_TAG}.npz')
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, f'best_evax_base_lora_adapter_{LABEL_TAG}.pth')
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
        device = torch.device('cuda:1')
        gpu_name = torch.cuda.get_device_name(1)
        log(f"使用设备: cuda:1")
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


class EVAXLoRAClassifier(nn.Module):
    def __init__(self, pretrained_weight_path=None, num_classes=1, lora_rank=LORA_RANK,
                 lora_alpha=LORA_ALPHA, clinical_dim=0):
        super(EVAXLoRAClassifier, self).__init__()

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
        self.classifier = nn.Linear(self.backbone.embed_dim // 4 + clinical_dim, num_classes)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_module_names = inject_lora_into_eva(
            self.backbone,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
        )
        self._init_head()
        self._configure_trainable_parameters()
        log(
            f"LoRA-分类 注入完成 | rank={self.lora_rank} | alpha={self.lora_alpha:g} | "
            f"目标层={LORA_TARGET_MODULES} | 注入线性层数={len(self.lora_module_names)}"
        )

    def _init_head(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

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

        # 分类输出
        if clinical is not None:
            x = torch.cat([x, clinical], dim=1)
        x = self.classifier(x)
        if verbose:
            print(f"[12] 分类输出: shape={x.shape}")
            print(f"    处理: 最终分类层，输出概率logit")
            print(f"{'=' * 60}")

        return x


def mixup_data(x_img, y, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_img.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_img = lam * x_img + (1 - lam) * x_img[index, :]
    y_a, y_b = y, y[index]
    return mixed_img, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def split_dataset(metadata, y, seed):
    train_idx, val_idx, test_idx, _ = load_or_create_patient_split(
        metadata=metadata,
        classification_labels=y,
        z_scores=metadata['z_score'].to_numpy(dtype=np.float32),
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


def load_and_preprocess_data(force_reload=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = CACHE_FILE
    current_source_signature = excel_source_signature(EXCEL_PATH2)

    if not force_reload and os.path.exists(cache_file):
        log("从缓存加载数据...")
        cached = np.load(cache_file, allow_pickle=True)
        required_cache_fields = {
            'X', 'y', 'accessions', 'filenames', 'patient_keys', 'z_scores',
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
                'z_score': cached['z_scores'].astype(np.float32),
            })
        else:
            log("旧缓存不包含统一预处理或患者级划分所需字段，将重新生成缓存")
            metadata = None
        if metadata is not None:
            log(f"数据从缓存加载完成，共 {len(X)} 个样本")
            log(f"当前缓存路径: {cache_file}")
            class_counts = np.bincount(y.astype(int), minlength=2)
            log(f"缓存类别分布: 0类 {class_counts[0]}, 1类 {class_counts[1]}")
            return X, y, metadata
    if force_reload and os.path.exists(cache_file):
        log("强制重新加载数据，删除旧缓存...")
        os.remove(cache_file)

    log("开始加载真实DICOM数据...")

    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)
    df2.columns = [str(column).strip() for column in df2.columns]

    required_columns = {'放射编号', LABEL_COLUMN, REGRESSION_SPLIT_COLUMN}
    missing_columns = required_columns.difference(df2.columns)
    if missing_columns:
        log(f"错误：Excel 缺少必要列: {sorted(missing_columns)}")
        log(f"错误：df2中不存在'放射编号'列！")
        log(f"请检查Excel文件，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([]), None
    df2['放射编号'] = df2['放射编号'].map(normalize_accession)

    log(f"df2中的列名：{df2.columns.tolist()}")

    X, y = [], []
    accessions, filenames, patient_keys, z_scores = [], [], [], []
    filter_reasons = {
        'other_error': 0,
        'no_pixel_data': 0,
        'invalid_label': 0,
        'invalid_z_score': 0,
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

                z_score = pd.to_numeric(excel_row[REGRESSION_SPLIT_COLUMN], errors='coerce')
                if pd.isna(z_score) or not np.isfinite(float(z_score)):
                    print(f"警告：放射编号 {accession_str} 的 {REGRESSION_SPLIT_COLUMN} 无效，跳过该样本")
                    filter_reasons['invalid_z_score'] += 1
                    filtered_filenames.append(filename)
                    continue

                patient_key = patient_key_from_sources(excel_row, dcm)
                if not patient_key:
                    print(f"警告：放射编号 {accession_str} 缺少患者标识，跳过该样本")
                    filter_reasons['missing_patient_key'] += 1
                    filtered_filenames.append(filename)
                    continue

                img = preprocess_cxr_dicom(dcm, img_SIZE)

                X.append(img)
                y.append(label)
                accessions.append(accession_str)
                filenames.append(filename)
                patient_keys.append(patient_key)
                z_scores.append(float(z_score))
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
        z_scores=np.array(z_scores, dtype=np.float32),
        preprocess_version=np.asarray(PREPROCESS_VERSION),
        source_signature=np.asarray(current_source_signature),
    )
    log(f"缓存已保存到: {cache_file}")

    metadata = pd.DataFrame({
        '放射编号': accessions,
        'filename': filenames,
        'patient_key': patient_keys,
        'z_score': z_scores,
    })
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
    def __init__(self, images, labels, transform=None, normalize_mean=None, normalize_std=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            to_pil = transforms.ToPILImage()
            img = to_pil(img)
            img = self.transform(img)

        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        if self.normalize_mean is not None and self.normalize_std is not None:
            img = (img - self.normalize_mean) / self.normalize_std

        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return img, label


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, labels in val_loader:
            img, labels = img.to(device), labels.to(device).float()
            outputs = model(img).view(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    pred_labels = (all_preds > 0.5).astype(int)
    accuracy = np.mean(pred_labels == all_labels)

    if np.unique(all_labels).size < 2:
        roc_auc = 0.5
        pr_auc = float(all_labels.mean())
    else:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

        roc_auc = roc_auc_score(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)

    return val_loss, accuracy, roc_auc, pr_auc


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

    X, y, metadata = load_and_preprocess_data(force_reload=False)

    if len(X) == 0:
        log("数据加载失败，训练终止")
        return

    train_idx, val_idx, test_idx = split_dataset(metadata, y, SEED)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    log(f"数据集划分完成 | 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")
    split_path = save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp)
    manifest_path, label_signature = save_label_manifest(metadata, y, run_stamp)
    write_run_info(run_stamp, CACHE_FILE, BEST_MODEL_FILE, split_path, manifest_path, label_signature, y)

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    log(f"数据归一化参数 - 均值: {X_train_mean:.4f}, 标准差: {X_train_std:.4f}")

    train_transform = transforms.Compose([
        transforms.Resize(img_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = MedicalImageDataset(X_train, y_train, transform=train_transform,
                                        normalize_mean=X_train_mean, normalize_std=X_train_std)
    val_dataset = MedicalImageDataset(X_val, y_val, transform=val_test_transform,
                                      normalize_mean=X_train_mean, normalize_std=X_train_std)
    test_dataset = MedicalImageDataset(X_test, y_test, transform=val_test_transform,
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

    model = EVAXLoRAClassifier(
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
    criterion = FocalLoss(
        gamma=2.0,
        scale=0.8,
    )
    log(
        f"训练集类别分布 | 0类={train_neg_count} | 1类={train_pos_count} | "
        f"负/正比例={class_ratio:.6f}"
    )
    log("损失函数: FocalLoss | scale=0.8 | gamma=2.0")

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

        for batch_idx, (img, labels) in enumerate(tqdm(train_loader, desc="Training")):
            img, labels = img.to(device), labels.to(device).float()
            optimizer.zero_grad()

            if random.random() < 0.4:
                mixed_img, labels_a, labels_b, lam = mixup_data(img, labels)
                outputs = model(mixed_img).view(-1)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(img).view(-1)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 计算训练集 ROC-AUC 和 PR-AUC
        train_loss_eval, train_acc, train_auc, train_pr_auc = validate(
            model, train_loader, criterion
        )
        train_aucs.append(train_auc)
        train_pr_aucs.append(train_pr_auc)

        val_loss, val_acc, val_auc, val_pr_auc = validate(model, val_loader, criterion)
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
            f"训练损失: {train_loss:.6f} | 训练ROC-AUC: {train_auc:.4f} | "
            f"训练PR-AUC: {train_pr_auc:.4f} | 验证损失: {val_loss:.6f} | "
            f"验证ROC-AUC: {val_auc:.4f} | 验证PR-AUC: {val_pr_auc:.4f}"
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
            f"best_evax_base_lora_adapter_prauc{best_val_auprc:.4f}_{run_stamp}.pth"
        )
        shutil.copy2(best_model_path, archived_model_path)
        log(f"已归档最佳 LoRA-分类 adapter: {archived_model_path}")

    def collect_probs(data_loader, desc):
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for img, labels in tqdm(data_loader, desc=desc):
                img, labels = img.to(device), labels.to(device).float()
                outputs = model(img).view(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        return np.array(all_labels), np.array(all_probs)

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

    all_val_labels, all_val_probs = collect_probs(val_loader, "Selecting Threshold")
    best_threshold, val_threshold_metrics, best_youden = find_optimal_threshold(all_val_labels, all_val_probs)
    all_test_labels, all_test_probs = collect_probs(test_loader, "Testing")
    best_metrics = compute_metrics(all_test_labels, all_test_probs, best_threshold)
    pr_curve_path = os.path.join(
        OUTPUT_DIR, f"evax_base_lora_test_pr_curve_{LABEL_TAG}_{run_stamp}.png"
    )
    test_pr_auc = plot_pr_curve(all_test_labels, all_test_probs, pr_curve_path)

    log(f"\n=== 结果总结 ===")
    log(f"最佳验证PR-AUC: {best_val_auprc:.4f}")
    log(f"验证集最佳阈值: {best_threshold:.4f}")
    log(f"验证集Youden指数: {best_youden:.4f}")
    log(f"测试AUC: {best_metrics['auc']:.4f}")
    log(f"测试PR-AUC: {test_pr_auc:.4f}")
    log(f"测试准确率: {best_metrics['accuracy']:.2f}")
    log(f"测试灵敏度: {best_metrics['sensitivity']:.4f}")
    log(f"测试特异度: {best_metrics['specificity']:.4f}")
    log(f"测试Precision: {best_metrics['precision']:.4f}")
    log(f"测试NPV: {best_metrics['npv']:.4f}")
    log(f"测试F1: {best_metrics['f1']:.4f}")
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
