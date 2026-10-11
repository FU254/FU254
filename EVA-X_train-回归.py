#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA-X 医学影像回归训练脚本 - base模型
仅训练回归层，Backbone全程冻结
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import logging
from pathlib import Path

import numpy as npco
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydicom
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()


def log(message):
    logger.info(message)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_CANDIDATES = [
    os.path.join(PROJECT_DIR, 'classification'),
    os.path.join(os.getcwd(), 'classification'),
    '/tmp/pycharm_project_601/EVA-X-main/classification',
]
CLASSIFICATION_DIR = next((p for p in CLASSIFICATION_CANDIDATES if os.path.exists(os.path.join(p, 'models', 'models_eva.py'))), None)
if CLASSIFICATION_DIR is None:
    raise FileNotFoundError(
        "找不到 EVA-X classification/models/models_eva.py，请确认 classification 文件夹已和脚本一起复制到运行目录。"
        f" 已检查路径: {CLASSIFICATION_CANDIDATES}"
    )
sys.path.insert(0, CLASSIFICATION_DIR)
from models.models_eva import eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

IMAGE_SIZE = 224
img_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
FREEZE_BACKBONE = True
UNFREEZE_LAST_N_BLOCKS = 0
DROPOUT_RATE = 0.6
HEAD_LR = 5e-5
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 300
PATIENCE = 30
BEST_MODEL_PATH = 'best_evax_base_regression_model.pth'
EXCEL_PATH1 = r'/home/cfu/.virtualenvs/DR/tmp/pycharm_project_689/tmp/pycharm_project_715/DR已筛选2019-2023.xlsx'
EXCEL_PATH2 = r'/tmp/pycharm_project_601/副本2025.7.14-基础信息+肺功能-郭屿老师.xlsx'
DICOM_DIR = r'/home/cfu/.virtualenvs/DR/TEST/NEW.DR'
PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), 'eva_x_base_patch16_merged520k_mim.pt')
EXPECTED_PRETRAINED_SIZE = 1210286919
LABEL_CANDIDATES = [
    'FEV1%F 实际值列',
    'FEV1%F 实际值',
    'FEV1%F实际值',
    'FEV1%F',
]


def resolve_pretrained_path(preferred_path=None):
    env_path = os.environ.get('EVA_X_PRETRAINED_PATH')
    filename = 'eva_x_base_patch16_merged520k_mim.pt'
    candidates = []
    for candidate in [
        env_path,
        preferred_path,
        os.path.join(PROJECT_DIR, filename),
        os.path.join(os.getcwd(), filename),
        '/tmp/pycharm_project_601/EVA-X-main/eva_x_base_patch16_merged520k_mim.pt',
    ]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        size = os.path.getsize(candidate)
        if size >= EXPECTED_PRETRAINED_SIZE:
            return candidate, size, candidates
        log(f"跳过疑似不完整的预训练权重: {candidate} ({size} bytes)")

    return None, None, candidates


def setup_hardware():
    log("开始初始化硬件配置")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        log("使用设备: cuda:0")
        log(f"GPU型号: {gpu_name}")
    else:
        device = torch.device('cpu')
        log("使用设备: cpu")
    log("硬件配置初始化完成")
    return device


class EVAXRegressor(nn.Module):
    def __init__(self, pretrained_weight_path=None, dropout_rate=0.6, model_type='base',
                 freeze_backbone=True, unfreeze_last_n_blocks=0):
        super(EVAXRegressor, self).__init__()

        log("创建EVA-X-base-patch16回归模型")
        self.backbone = eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(pretrained=False)

        resolved_weight_path, weight_size, checked_candidates = resolve_pretrained_path(pretrained_weight_path)
        if resolved_weight_path is not None:
            weight_size = weight_size
            log(f"加载预训练权重: {pretrained_weight_path}")
            log(f"预训练权重大小: {weight_size} bytes")
            try:
                checkpoint = torch.load(pretrained_weight_path, map_location='cpu', weights_only=False)
            except RuntimeError as e:
                raise RuntimeError(
                    f"预训练权重文件无法读取，通常是文件复制不完整或已损坏: {pretrained_weight_path}\n"
                    f"当前文件大小: {weight_size} bytes\n"
                    "请重新复制 eva_x_base_patch16_merged520k_mim.pt 到 Linux 运行目录，并确认大小约为 1210286919 bytes。"
                ) from e

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
            log("权重加载完成")
            if msg.missing_keys:
                log(f"缺失的权重: {msg.missing_keys}")
            if msg.unexpected_keys:
                log(f"意外的权重: {msg.unexpected_keys}")
        else:
            log(f"警告: 未找到预训练权重或路径无效: {pretrained_weight_path}")

        self.backbone.fc_norm = nn.LayerNorm(self.backbone.embed_dim)
        self.bottleneck = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim // 2)
        self.bottleneck2 = nn.Linear(self.backbone.embed_dim // 2, self.backbone.embed_dim // 4)
        self.regressor = nn.Linear(self.backbone.embed_dim // 4, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_head()

        if freeze_backbone:
            self.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    def _init_head(self):
        nn.init.trunc_normal_(self.regressor.weight, std=0.02)
        if self.regressor.bias is not None:
            nn.init.constant_(self.regressor.bias, 0)

    def freeze_backbone(self, unfreeze_last_n_blocks=0):
        log("冻结Backbone所有参数，仅训练回归层")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        total_blocks = len(self.backbone.blocks)
        if unfreeze_last_n_blocks > 0:
            selected_blocks = self.backbone.blocks[-unfreeze_last_n_blocks:]
            for blk in selected_blocks:
                for param in blk.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True
            log(f"已解冻最后 {min(unfreeze_last_n_blocks, total_blocks)} 个Transformer blocks")
        log(f"已冻结 {sum(1 for _ in self.backbone.parameters())} 个Backbone参数")

    def forward(self, x, verbose=False):
        if verbose:
            print(f"\n{'='*60}")
            print("模型前向传播日志")
            print(f"{'='*60}")
            print(f"[输入] 原始输入: shape={x.shape}, dtype={x.dtype}")
            print(f"       数值范围: [{x.min().item():.4f}, {x.max().item():.4f}]")

        x = self.backbone.patch_embed(x)
        if verbose:
            print(f"[1] Patch Embedding: shape={x.shape}")

        batch_size, seq_len, embed_dim = x.size()
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if verbose:
            print(f"[2] 添加CLS Token: shape={x.shape}")

        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
            if verbose:
                print(f"[3] 添加位置嵌入: shape={x.shape}")

        x = self.backbone.pos_drop(x)
        if verbose:
            print(f"[4] 位置Dropout: shape={x.shape}")

        rel_pos_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        if verbose:
            print(f"[5] Transformer Blocks: 共{len(self.backbone.blocks)}层")

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rel_pos_bias)
            if verbose and (i == 0 or i == len(self.backbone.blocks) - 1):
                print(f"    Block {i + 1}: shape={x.shape}")

        x = self.backbone.norm(x)
        if verbose:
            print(f"[6] 层归一化: shape={x.shape}")

        features = self.backbone.fc_norm(x.mean(1))
        if verbose:
            print(f"[7] 特征提取: shape={features.shape}")

        x = self.bottleneck(features)
        if verbose:
            print(f"[8] Bottleneck1: shape={x.shape}")

        x = nn.ReLU()(x)
        x = self.dropout(x)
        if verbose:
            print(f"[9] ReLU + Dropout: shape={x.shape}")

        x = self.bottleneck2(x)
        if verbose:
            print(f"[10] Bottleneck2: shape={x.shape}")

        x = nn.ReLU()(x)
        x = self.dropout(x)
        if verbose:
            print(f"[11] ReLU + Dropout: shape={x.shape}")

        x = self.regressor(x)
        if verbose:
            print(f"[12] 回归输出: shape={x.shape}")
            print(f"{'='*60}")

        return x


def load_and_preprocess_data(force_reload=False):
    cache_file = '/tmp/evax_base_regression_cached_data.npz'

    if not force_reload and os.path.exists(cache_file):
        log("从缓存加载数据...")
        cached = np.load(cache_file, allow_pickle=True)
        X = cached['X']
        y = cached['y']
        log(f"数据从缓存加载完成，共 {len(X)} 个样本")
        return X, y
    elif force_reload and os.path.exists(cache_file):
        log("强制重新加载数据，删除旧缓存...")
        os.remove(cache_file)

    log("开始加载真实DICOM数据...")

    df1 = pd.read_excel(EXCEL_PATH1, sheet_name=0)
    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)

    df1['放射编号'] = df1['放射编号'].astype(str).str.strip()
    df1['病人姓名'] = df1['病人姓名'].astype(str).str.strip()
    if '姓名' in df2.columns:
        df2['姓名'] = df2['姓名'].astype(str).str.strip()

    log(f"df1中的列名：{df1.columns.tolist()}")
    log(f"df2中的列名：{df2.columns.tolist()}")

    label_col = next((col for col in LABEL_CANDIDATES if col in df2.columns), None)
    if label_col is None:
        log(f"错误：df2中不存在回归标签列！候选列：{LABEL_CANDIDATES}")
        log(f"请检查列名是否正确，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([])

    X, y = [], []
    filter_reasons = {
        'other_error': 0,
        'no_pixel_data': 0,
        'no_accession': 0,
        'invalid_label': 0,
    }

    for filename in tqdm(os.listdir(DICOM_DIR), desc="Processing DICOM"):
        if filename.endswith('.dcm'):
            try:
                dcm = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
                if 'PixelData' not in dcm:
                    filter_reasons['no_pixel_data'] += 1
                    continue

                if not hasattr(dcm, 'AccessionNumber') or not dcm.AccessionNumber:
                    filter_reasons['no_accession'] += 1
                    continue

                accession = str(dcm.AccessionNumber).strip()

                df1_match = df1[df1['放射编号'] == accession]
                if len(df1_match) == 0:
                    filter_reasons['no_accession'] += 1
                    continue

                patient_name = df1_match['病人姓名'].values[0]
                label_df = df2[df2['姓名'] == patient_name]

                if len(label_df) == 0:
                    filter_reasons['invalid_label'] += 1
                    continue

                try:
                    label_value = float(label_df[label_col].values[0])
                    if np.isnan(label_value) or np.isinf(label_value):
                        filter_reasons['invalid_label'] += 1
                        continue

                    img = dcm.pixel_array.astype(np.float32)
                    if img.ndim > 2:
                        img = np.squeeze(img)
                    if img.ndim != 2:
                        filter_reasons['other_error'] += 1
                        continue

                    slope = float(getattr(dcm, 'RescaleSlope', 1.0) or 1.0)
                    intercept = float(getattr(dcm, 'RescaleIntercept', 0.0) or 0.0)
                    img = img * slope + intercept
                    if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                        img = np.max(img) - img

                    finite_mask = np.isfinite(img)
                    if not np.any(finite_mask):
                        filter_reasons['other_error'] += 1
                        continue

                    valid_pixels = img[finite_mask]
                    lo, hi = np.percentile(valid_pixels, [1.0, 99.0])
                    if hi <= lo:
                        lo = float(np.min(valid_pixels))
                        hi = float(np.max(valid_pixels))
                        if hi <= lo:
                            filter_reasons['other_error'] += 1
                            continue

                    img = np.clip(img, lo, hi)
                    img = (img - lo) / (hi - lo + 1e-8)

                    height, width = img.shape
                    if height > width:
                        start = (height - width) // 2
                        img = img[start:start + width, :]
                    else:
                        start = (width - height) // 2
                        img = img[:, start:start + height]

                    img = cv2.resize(img, (img_SIZE, img_SIZE), interpolation=cv2.INTER_CUBIC)
                    img = np.clip(img, 0.0, 1.0)

                    X.append(img.astype(np.float32))
                    y.append(label_value)
                except (ValueError, IndexError, TypeError):
                    filter_reasons['invalid_label'] += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                filter_reasons['other_error'] += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    log(f"数据加载完成，共 {len(X)} 个样本")
    if len(y) > 0:
        log(f"标签统计：mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    log(f"过滤统计：{filter_reasons}")

    log("保存数据到缓存...")
    np.savez_compressed(cache_file, X=X, y=y)
    log(f"缓存已保存到: {cache_file}")

    return X, y


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
            img = (img - self.normalize_mean) / (self.normalize_std + 1e-8)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def calculate_regression_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)

    mae = float(np.mean(np.abs(labels - preds)))
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    bias = float(np.mean(preds - labels))

    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

    label_std = float(np.std(labels))
    pred_std = float(np.std(preds))
    if len(labels) > 1 and label_std > 1e-8 and pred_std > 1e-8:
        pearson_r = float(np.corrcoef(labels, preds)[0, 1])
    else:
        pearson_r = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'bias': bias,
        'pearson_r': pearson_r,
    }


def validate(model, val_loader, criterion, target_mean, target_std):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, labels in val_loader:
            img, labels = img.to(device), labels.to(device)
            outputs = model(img).view(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy().reshape(-1))
            all_labels.extend(labels.cpu().numpy().reshape(-1))

    val_loss /= len(val_loader)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_preds_raw = all_preds * target_std + target_mean
    all_labels_raw = all_labels * target_std + target_mean
    metrics = calculate_regression_metrics(all_labels_raw, all_preds_raw)

    return val_loss, metrics


def main():
    global device

    device = setup_hardware()

    log(f"数据配置 - 图像尺寸: {IMAGE_SIZE}, 批次大小: {BATCH_SIZE}, 工作进程数: {NUM_WORKERS}")

    X, y = load_and_preprocess_data(force_reload=False)

    if len(X) == 0:
        log("数据加载失败，训练终止")
        return

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    train_ratio = 0.64
    val_ratio = 0.16

    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train_raw, y_val_raw, y_test_raw = y[:train_end], y[train_end:val_end], y[val_end:]

    log(f"数据集划分完成 | 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    log(f"数据归一化参数 - 均值: {X_train_mean:.4f}, 标准差: {X_train_std:.4f}")

    target_mean = y_train_raw.mean()
    target_std = y_train_raw.std()
    if target_std < 1e-8:
        log("错误：训练集标签标准差过小，无法标准化")
        return
    log(f"目标标准化参数 - 均值: {target_mean:.4f}, 标准差: {target_std:.4f}")

    y_train = ((y_train_raw - target_mean) / target_std).astype(np.float32)
    y_val = ((y_val_raw - target_mean) / target_std).astype(np.float32)
    y_test = ((y_test_raw - target_mean) / target_std).astype(np.float32)

    train_transform = transforms.Compose([
        transforms.Resize(img_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
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

    log("初始化EVA-X回归模型 - base版本")
    model = EVAXRegressor(
        pretrained_weight_path=PRETRAINED_PATH,
        dropout_rate=DROPOUT_RATE,
        model_type='base',
        freeze_backbone=FREEZE_BACKBONE,
        unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
    )
    model = model.to(device)

    if FREEZE_BACKBONE and UNFREEZE_LAST_N_BLOCKS == 0:
        log("模型初始化完成，Backbone全程冻结，仅训练回归层")
    elif FREEZE_BACKBONE:
        log(f"模型初始化完成，训练回归层 + Backbone最后 {UNFREEZE_LAST_N_BLOCKS} 个blocks")
    else:
        log("模型初始化完成，Backbone全量参与训练")

    criterion = nn.SmoothL1Loss(beta=1.0)

    patience = PATIENCE
    no_improvement_epochs = 0
    total_epochs = TOTAL_EPOCHS
    best_val_rmse = float('inf')
    best_model_path = BEST_MODEL_PATH

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    log("测试模型前向传播...")
    test_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        model.eval()
        _ = model(test_input, verbose=True)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({'params': head_params, 'lr': HEAD_LR})
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': BACKBONE_LR})
    if not param_groups:
        raise RuntimeError("没有可训练参数，请检查 FREEZE_BACKBONE 和 UNFREEZE_LAST_N_BLOCKS 配置")

    log(f"回归头可训练参数量: {sum(p.numel() for p in head_params)} | 学习率: {HEAD_LR}")
    log(f"Backbone可训练参数量: {sum(p.numel() for p in backbone_params)} | 学习率: {BACKBONE_LR}")

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)

    log(f"开始训练，共 {total_epochs} 个epoch")
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []

    for epoch in range(total_epochs):
        log(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")

        model.train()
        if FREEZE_BACKBONE and UNFREEZE_LAST_N_BLOCKS == 0:
            model.backbone.eval()
        train_loss = 0

        for batch_idx, (img, labels) in enumerate(tqdm(train_loader, desc="Training")):
            img, labels = img.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(img).view(-1)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        train_loss_eval, train_metrics = validate(model, train_loader, criterion, target_mean, target_std)
        train_rmses.append(train_metrics['rmse'])

        val_loss, val_metrics = validate(model, val_loader, criterion, target_mean, target_std)
        val_losses.append(val_loss)
        val_rmses.append(val_metrics['rmse'])

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            no_improvement_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            log(f"★ 最佳模型已保存 (Val RMSE: {best_val_rmse:.4f})")
        else:
            no_improvement_epochs += 1
            log(f"未改进: {no_improvement_epochs}/{patience}")

        scheduler.step(epoch + val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        log(f"当前学习率: {current_lr:.8f}")

        log(
            f"训练损失: {train_loss:.6f} | 训练RMSE: {train_metrics['rmse']:.4f} | "
            f"验证损失: {val_loss:.6f} | 验证RMSE: {val_metrics['rmse']:.4f}"
        )

        if no_improvement_epochs >= patience:
            log("早停触发，训练结束")
            break

    log("加载最佳模型进行测试...")
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, weights_only=False, map_location='cpu')
            model.load_state_dict(checkpoint)
            log(f"成功加载最佳模型: {best_model_path}")
        except Exception as e:
            log(f"警告: 无法加载模型文件 {best_model_path}, 错误: {str(e)}")
            log("使用当前训练的模型进行测试...")
    else:
        log(f"警告: 模型文件 {best_model_path} 不存在")
        log("使用当前训练的模型进行测试...")

    model.eval()
    train_loss_eval, train_metrics = validate(model, train_loader, criterion, target_mean, target_std)
    test_loss, test_metrics = validate(model, test_loader, criterion, target_mean, target_std)

    log("\n=== 结果总结 ===")
    log(f"最佳验证RMSE: {best_val_rmse:.4f}")
    log(f"训练集 MAE: {train_metrics['mae']:.4f}")
    log(f"训练集 RMSE: {train_metrics['rmse']:.4f}")
    log(f"训练集 R2: {train_metrics['r2']:.4f}")
    log(f"训练集 Bias: {train_metrics['bias']:.4f}")
    log(f"训练集 Pearson r: {train_metrics['pearson_r']:.4f}")
    log(f"测试集 MAE: {test_metrics['mae']:.4f}")
    log(f"测试集 RMSE: {test_metrics['rmse']:.4f}")
    log(f"测试集 R2: {test_metrics['r2']:.4f}")
    log(f"测试集 Bias: {test_metrics['bias']:.4f}")
    log(f"测试集 Pearson r: {test_metrics['pearson_r']:.4f}")

    log("\n绘制训练曲线...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_rmses, label='Train RMSE', color='blue')
    plt.plot(val_rmses, label='Val RMSE', color='orange')
    plt.title('Training & Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    log("训练曲线已显示")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
