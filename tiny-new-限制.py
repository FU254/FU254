#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA-X 医学影像分类训练脚本 - tiny模型
仅训练fc_norm与分类头，Transformer blocks全程冻结
"""

import warnings

warnings.filterwarnings('ignore')

import os
import sys
import logging
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import pydicom
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger()


def log(message):
    logger.info(message)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification'))
from models.models_eva import eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

IMAGE_SIZE = 224
img_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_METRIC_EVERY = 2
ENABLE_STARTUP_VERBOSE_FORWARD = False
SCRIPT_DIR = os.path.dirname(__file__)
TASK_NAME = 'tiny_new_obstruction_final1'
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'runs', TASK_NAME)
EXCEL_PATH1 = r'/home/cfu/.virtualenvs/DR/TEST/DR已筛选2019-2023.xlsx'
EXCEL_PATH2 = r'/tmp/pycharm_project_601/肺功能万能表最终版1.xlsx'
DICOM_DIR = r'/tmp/pycharm_project_601/DR数据'
PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), 'eva_x_tiny_patch16_merged520k_mim.pt')

LABEL_COLUMN = '限制性通气障碍'
SEED = 42
LEARNING_RATE = 5e-5
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True
UNFREEZE_LAST_N_BLOCKS = 0
HEAD_DROPOUT_RATE = 0.45


def make_safe_name(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|\s]+', '_', value)
    return value.strip('_') or 'task'


LABEL_TAG = make_safe_name(LABEL_COLUMN)
CACHE_FILE = os.path.join(OUTPUT_DIR, f'cached_data_{LABEL_TAG}.npz')
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, f'best_evax_tiny_model_{LABEL_TAG}.pth')
RUN_INFO_FILE = os.path.join(OUTPUT_DIR, f'run_info_{LABEL_TAG}.txt')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_unique_seq(accession, df1):
    seq_list = df1[df1['放射编号'] == accession]['序号'].tolist()
    if len(seq_list) > 1:
        print(f"警告：Accession {accession}对应多个序号，取第一个")
    return seq_list[0] if seq_list else None


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class EVAXClassifier(nn.Module):
    def __init__(self, pretrained_weight_path=None, num_classes=1, dropout_rate=0.7, model_type='tiny',
                 freeze_backbone=False, unfreeze_last_n_blocks=0):
        super(EVAXClassifier, self).__init__()

        log("创建EVA-X-tiny-patch16模型")
        self.backbone = eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)

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

        self.bottleneck = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim // 2)
        self.bottleneck2 = nn.Linear(self.backbone.embed_dim // 2, self.backbone.embed_dim // 4)
        self.head_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.backbone.embed_dim // 4, num_classes)
        self.freeze_backbone_enabled = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self._init_head()

        if freeze_backbone:
            self.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    def _init_head(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def freeze_backbone(self, unfreeze_last_n_blocks=0):
        log("冻结Backbone，仅解冻fc_norm与分类头")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc_norm.parameters():
            param.requires_grad = True
        for module in (self.bottleneck, self.bottleneck2, self.classifier):
            for param in module.parameters():
                param.requires_grad = True

        total_blocks = len(self.backbone.blocks)
        if unfreeze_last_n_blocks > 0:
            selected_blocks = self.backbone.blocks[-unfreeze_last_n_blocks:]
            for blk in selected_blocks:
                for param in blk.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            log(f"已额外解冻最后 {min(unfreeze_last_n_blocks, total_blocks)} 个Transformer blocks")
        log(f"Backbone冻结参数数量: {sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)}")

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_enabled:
            self.backbone.eval()
            self.backbone.fc_norm.train()
            if self.unfreeze_last_n_blocks > 0:
                for blk in self.backbone.blocks[-self.unfreeze_last_n_blocks:]:
                    blk.train()
                self.backbone.norm.train()
        return self

    def forward(self, x, verbose=False):
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

        # 与EVA-X原始forward_features一致：仅对patch tokens取均值，不包含CLS token
        patch_tokens = x[:, 1:, :]
        features = self.backbone.fc_norm(patch_tokens.mean(1))
        if verbose:
            print(f"[7] 特征提取: shape={features.shape}")
            print(f"    处理: 对patch tokens取均值作为图像特征表示")

        # 分类头 - 第一层
        x = self.head_dropout(features)
        x = self.bottleneck(x)
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
        x = self.head_dropout(x)
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


def split_dataset(X, y, seed):
    try:
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(X))
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            indices, y, test_size=0.36, random_state=seed, stratify=y
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )
        return train_idx, val_idx, test_idx
    except ValueError as e:
        log(f"鍒嗗眰鍒掑垎澶辫触锛屽洖閫€鍒伴殢鏈哄垝鍒? {e}")
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        train_ratio = 0.64
        val_ratio = 0.16
        train_end = int(len(indices) * train_ratio)
        val_end = int(len(indices) * (train_ratio + val_ratio))
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def load_and_preprocess_data(force_reload=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = CACHE_FILE

    if not force_reload and os.path.exists(cache_file):
        log("从缓存加载数据...")
        cached = np.load(cache_file, allow_pickle=True)
        X = cached['X']
        y = cached['y']
        metadata = None
        if {'accessions', 'seqs', 'filenames'}.issubset(cached.files):
            metadata = pd.DataFrame({
                '放射编号': cached['accessions'].astype(str),
                '序号': cached['seqs'].astype(str),
                'filename': cached['filenames'].astype(str),
            })
        log(f"数据从缓存加载完成，共 {len(X)} 个样本")
        log(f"当前缓存路径: {cache_file}")
        class_counts = np.bincount(y.astype(int))
        if len(class_counts) == 1:
            log(f"缓存类别分布: 0类 {class_counts[0]}, 1类 0")
        else:
            log(f"缓存类别分布: 0类 {class_counts[0]}, 1类 {class_counts[1]}")
        return X, y, metadata
    elif force_reload and os.path.exists(cache_file):
        log("强制重新加载数据，删除旧缓存...")
        os.remove(cache_file)

    log("开始加载真实DICOM数据...")

    df1 = pd.read_excel(EXCEL_PATH1, sheet_name=0)
    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)

    df1['序号'] = df1['序号'].astype(str).str.strip()
    if '放射编号' not in df2.columns:
        log(f"错误：df2中不存在'放射编号'列！")
        log(f"请检查Excel文件，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([]), None
    df2['放射编号'] = df2['放射编号'].astype(str).str.strip()

    log(f"df1中的列名：{df1.columns.tolist()}")
    log(f"df2中的列名：{df2.columns.tolist()}")

    if LABEL_COLUMN not in df2.columns:
        log(f"错误：df2中不存在'{LABEL_COLUMN}'列！")
        log(f"请检查列名是否正确，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([]), None

    X, y = [], []
    accessions, seqs, filenames = [], [], []
    filter_reasons = {
        'other_error': 0,
        'no_pixel_data': 0,
        'no_sequence': 0,
        'invalid_label': 0,
        'missing_label_column': 0
    }
    filtered_filenames = []
    missing_seq = []

    for filename in tqdm(os.listdir(DICOM_DIR), desc="Processing DICOM"):
        if filename.endswith('.dcm'):
            try:
                dcm = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
                if 'PixelData' not in dcm:
                    print(f"跳过无像素数据文件：{filename}")
                    filter_reasons['no_pixel_data'] += 1
                    filtered_filenames.append(filename)
                    continue

                accession = dcm.AccessionNumber

                seq = get_unique_seq(accession, df1)
                if seq is None:
                    print(f"跳过无序号文件：{accession}")
                    filter_reasons['no_sequence'] += 1
                    filtered_filenames.append(filename)
                    continue

                accession_str = str(accession).strip()
                label_df = df2[df2['放射编号'] == accession_str]

                if len(label_df) == 0:
                    print(f"警告：放射编号 {accession_str} 无标签，跳过该样本")
                    filter_reasons['invalid_label'] += 1
                    filtered_filenames.append(filename)
                else:
                    try:
                        label_value = label_df[LABEL_COLUMN].values[0]
                        label = int(label_value)
                        if label not in {0, 1}:
                            print(f"警告：放射编号 {accession_str} 的标签无效（{label_value}），跳过该样本")
                            filter_reasons['invalid_label'] += 1
                            filtered_filenames.append(filename)
                            continue
                        else:
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

                img = dcm.pixel_array.astype(float)

                height, width = img.shape
                if height > width:
                    start = (height - width) // 2
                    img = img[start:start + width, :]
                else:
                    start = (width - height) // 2
                    img = img[:, start:start + height]

                img = cv2.resize(img, (img_SIZE, img_SIZE), interpolation=cv2.INTER_CUBIC)
                img = img / (img.max() + 1e-8)

                X.append(img)
                y.append(label)
                accessions.append(str(accession).strip())
                seqs.append(str(seq).strip())
                filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                filter_reasons['other_error'] += 1
                filtered_filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    log(f"过滤原因统计: {filter_reasons}")
    log(f"被过滤文件数: {len(filtered_filenames)}")
    log(f"缺少序号的放射编号数: {len(missing_seq)}")

    log(f"数据加载完成，共 {len(X)} 个样本")

    class_counts = np.bincount(y.astype(int))
    log(f"类别分布：0类: {class_counts[0]}, 1类: {class_counts[1]}")

    log("保存数据到缓存...")
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        accessions=np.array(accessions, dtype=str),
        seqs=np.array(seqs, dtype=str),
        filenames=np.array(filenames, dtype=str),
    )
    log(f"缓存已保存到: {cache_file}")

    metadata = pd.DataFrame({
        '放射编号': accessions,
        '序号': seqs,
        'filename': filenames,
    })
    return X, y, metadata


def save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp):
    if metadata is None:
        log("当前缓存不包含放射编号/序号元数据，无法导出 split 清单")
        return None

    split_df = metadata.copy()
    split_df['split'] = ''
    split_df.loc[train_idx, 'split'] = 'train'
    split_df.loc[val_idx, 'split'] = 'val'
    split_df.loc[test_idx, 'split'] = 'test'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_path = os.path.join(OUTPUT_DIR, f'evax_tiny_split_{LABEL_TAG}_{run_stamp}.csv')
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
        manifest_df['序号'].astype(str) + '|' +
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
        f"split_path={split_path or ''}",
        f"label_manifest_path={manifest_path or ''}",
        f"label_signature={label_signature or ''}",
        f"sample_count={len(y)}",
        f"class_0_count={neg_count}",
        f"class_1_count={pos_count}",
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

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5

    return val_loss, accuracy, auc


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

    train_idx, val_idx, test_idx = split_dataset(X, y, SEED)
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
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = MedicalImageDataset(X_train, y_train, transform=train_transform,
                                        normalize_mean=X_train_mean, normalize_std=X_train_std)
    train_eval_dataset = MedicalImageDataset(X_train, y_train, transform=val_test_transform,
                                             normalize_mean=X_train_mean, normalize_std=X_train_std)
    val_dataset = MedicalImageDataset(X_val, y_val, transform=val_test_transform,
                                      normalize_mean=X_train_mean, normalize_std=X_train_std)
    test_dataset = MedicalImageDataset(X_test, y_test, transform=val_test_transform,
                                       normalize_mean=X_train_mean, normalize_std=X_train_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                   pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    log(f"创建数据集 - 样本数量: {len(train_dataset)}, 转换策略: 训练增强")
    log(f"创建数据集 - 样本数量: {len(val_dataset)}, 转换策略: 验证/测试")
    log(f"创建数据集 - 样本数量: {len(test_dataset)}, 转换策略: 验证/测试")

    log("初始化EVA-X分类模型 - tiny版本")

    pretrained_path = PRETRAINED_PATH

    model = EVAXClassifier(
        pretrained_weight_path=pretrained_path,
        num_classes=1,
        dropout_rate=HEAD_DROPOUT_RATE,
        model_type='tiny',
        freeze_backbone=FREEZE_BACKBONE,
        unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
    )
    model = model.to(device)


    if FREEZE_BACKBONE:
        if UNFREEZE_LAST_N_BLOCKS > 0:
            log(f"模型初始化完成，解冻最后 {UNFREEZE_LAST_N_BLOCKS} 个blocks、fc_norm与分类头")
        else:
            log("模型初始化完成，所有Transformer blocks冻结，仅训练fc_norm、bottleneck、bottleneck2与classifier")
    else:
        log("模型初始化完成，backbone全量参与微调")

    criterion = FocalLoss(alpha=0.8, gamma=2.5)

    patience = 12
    no_improvement_epochs = 0
    total_epochs = 300
    best_val_auc = 0.0
    best_model_path = BEST_MODEL_FILE

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")

    # 测试模型前向传播日志
    log("测试模型前向传播...")
    test_input = torch.randn(2, 3, 224, 224).to(device)  # batch_size=2, 3通道, 224x224
    with torch.no_grad():
        model.eval()
        _ = model(test_input, verbose=ENABLE_STARTUP_VERBOSE_FORWARD)

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
        param_groups.append({'params': head_params, 'lr': LEARNING_RATE})
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': BACKBONE_LR})
    if not param_groups:
        raise RuntimeError("娌℃湁鍙缁冨弬鏁帮紝璇锋鏌?FREEZE_BACKBONE 鍜?UNFREEZE_LAST_N_BLOCKS 閰嶇疆")

    log(f"鍒嗙被澶村彲璁粌鍙傛暟閲? {sum(p.numel() for p in head_params)} | 瀛︿範鐜? {LEARNING_RATE}")
    log(f"Backbone鍙缁冨弬鏁伴噺: {sum(p.numel() for p in backbone_params)} | 瀛︿範鐜? {BACKBONE_LR}")

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)

    log(f"开始训练，共 {total_epochs} 个epoch")
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

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

        # 计算训练集AUC
        if epoch % TRAIN_METRIC_EVERY == 0:
            train_loss_eval, train_acc, train_auc = validate(model, train_eval_loader, criterion)
        else:
            train_loss_eval, train_acc = float('nan'), float('nan')
            train_auc = train_aucs[-1] if train_aucs else float('nan')
        train_aucs.append(train_auc)

        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improvement_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            log(f"★ 最佳模型已保存 (AUC: {best_val_auc:.4f})")
        else:
            no_improvement_epochs += 1
            log(f"未改进: {no_improvement_epochs}/{patience}")

        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        log(f"当前学习率: {current_lr:.8f}")

        log(f"训练损失: {train_loss:.6f} | 训练AUC: {train_auc:.4f} | 验证损失: {val_loss:.6f} | 验证AUC: {val_auc:.4f}")

        if no_improvement_epochs >= patience:
            log(f"早停触发，训练结束")
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

    if os.path.exists(best_model_path):
        archived_model_path = os.path.join(
            OUTPUT_DIR,
            f"best_evax_tiny_model_auc{best_val_auc:.4f}_{run_stamp}.pth"
        )
        shutil.copy2(best_model_path, archived_model_path)
        log(f"已归档最佳模型: {archived_model_path}")

    criterion = FocalLoss(alpha=0.8, gamma=2.5)

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
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        return {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1
        }

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

    log(f"\n=== 结果总结 ===")
    log(f"最佳验证AUC: {best_val_auc:.4f}")
    log(f"验证集最佳阈值: {best_threshold:.4f}")
    log(f"验证集Youden指数: {best_youden:.4f}")
    log(f"测试AUC: {best_metrics['auc']:.4f}")
    log(f"测试准确率: {best_metrics['accuracy']:.2f}")
    log(f"测试灵敏度: {best_metrics['sensitivity']:.4f}")
    log(f"测试特异度: {best_metrics['specificity']:.4f}")
    log(f"测试F1: {best_metrics['f1']:.4f}")

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
    plt.plot(train_aucs, label='Train AUC', color='blue')
    plt.plot(val_aucs, label='Val AUC', color='orange')
    plt.title('Training & Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, f'training_curves_{LABEL_TAG}_{run_stamp}.png')
    plt.savefig(curve_path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f"训练曲线已保存: {curve_path}")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
