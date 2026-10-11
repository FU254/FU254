#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA-X 医学影像分类训练脚本
支持 EVA-X-small 和 EVA-X-tiny 模型
"""

import os
import sys
import warnings
import logging
import random
from pathlib import Path
from functools import lru_cache

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
import cv2

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

def log(message):
    logger.info(message)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification'))
from models.models_eva import eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE, eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False

IMAGE_SIZE = 224
img_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
LABEL_COLUMN = '限制性通气功能正常0与异常1'
EXCEL_PATH1 = r'/home/cfu/.virtualenvs/DR/TEST/DR已筛选2019-2023.xlsx'
EXCEL_PATH2 = r'/home/cfu/.virtualenvs/DR/TEST/肺功能万能表.xlsx'
DICOM_DIR = r'/home/cfu/.virtualenvs/DR/TEST/NEW.DR'

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
    def __init__(self, pretrained_weight_path=None, num_classes=1, dropout_rate=0.7, model_type='small'):
        super(EVAXClassifier, self).__init__()

        if model_type == 'tiny':
            log("创建EVA-X-tiny-patch16模型")
            self.backbone = eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)
        else:
            log("创建EVA-X-small-patch16模型")
            self.backbone = eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)

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
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.5)
        self.bottleneck = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim // 2)
        self.bottleneck2 = nn.Linear(self.backbone.embed_dim // 2, self.backbone.embed_dim // 4)
        self.classifier = nn.Linear(self.backbone.embed_dim // 4, num_classes)
        self._init_head()

    def _init_head(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        rel_pos_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        for blk in self.backbone.blocks:
            x = blk(x, rel_pos_bias)

        x = self.backbone.norm(x)
        features = self.backbone.fc_norm(x.mean(1))

        x = self.dropout1(features)
        x = self.bottleneck(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.bottleneck2(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.classifier(x)

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

def get_unique_seq(accession, df1):
    seq_list = df1[df1['放射编号'] == accession]['序号'].tolist()
    if len(seq_list) > 1:
        print(f"警告：Accession {accession}对应多个序号，取第一个")
    return seq_list[0] if seq_list else None

def load_and_preprocess_data(force_reload=False):
    cache_file = '/tmp/evax_cached_data.npz'

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
    df2_sheets = pd.read_excel(EXCEL_PATH2, sheet_name=[1])
    df2 = pd.concat(df2_sheets.values(), ignore_index=True)

    df1['序号'] = df1['序号'].astype(str).str.strip()
    df2['序号'] = df2['序号'].astype(str).str.strip()

    log(f"df2中的列名：{df2.columns.tolist()}")

    LABEL_COLUMN = '限制性通气功能正常0与异常1'

    if LABEL_COLUMN not in df2.columns:
        log(f"错误：df2中不存在'{LABEL_COLUMN}'列！")
        log(f"请检查列名是否正确，现有列名为：{df2.columns.tolist()}")
        return np.array([]), np.array([])

    X, y = [], []
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
                else:
                    label_df = df2[df2['序号'] == seq]
                    if len(label_df) == 0:
                        print(f"警告：序号 {seq} 无标签，跳过该样本")
                        filter_reasons['invalid_label'] += 1
                        filtered_filenames.append(filename)
                    else:
                        try:
                            label_value = label_df[LABEL_COLUMN].values[0]
                            label = int(label_value)
                            if label not in {0, 1}:
                                print(f"警告：序号 {seq} 的标签无效（{label_value}），跳过该样本")
                                filter_reasons['invalid_label'] += 1
                                filtered_filenames.append(filename)
                            else:
                                print(f"序号 {seq} 的标签为: {label}")
                                
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
                        except (ValueError, IndexError) as e:
                            print(f"警告：序号 {seq} 的标签无法转换为整数或数据缺失，错误: {str(e)}，跳过该样本")
                            filter_reasons['invalid_label'] += 1
                            filtered_filenames.append(filename)
                        except KeyError as e:
                            print(f"警告：序号 {seq} 的标签列不存在: {str(e)}，跳过该样本")
                            filter_reasons['missing_label_column'] += 1
                            filtered_filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                filter_reasons['other_error'] += 1
                filtered_filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    log(f"数据加载完成，共 {len(X)} 个样本")

    class_counts = np.bincount(y.astype(int))
    log(f"类别分布：0类: {class_counts[0]}, 1类: {class_counts[1]}")

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
            outputs = model(img).squeeze()
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

    device = setup_hardware()

    log(f"数据配置 - 图像尺寸: {IMAGE_SIZE}, 批次大小: {BATCH_SIZE}, 工作进程数: {NUM_WORKERS}")

    X, y = load_and_preprocess_data(force_reload=True)

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
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    log(f"数据集划分完成 | 训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    log(f"数据归一化参数 - 均值: {X_train_mean:.4f}, 标准差: {X_train_std:.4f}")

    train_transform = transforms.Compose([
        transforms.Resize(img_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    log(f"创建数据集 - 样本数量: {len(train_dataset)}, 转换策略: 训练增强")
    log(f"创建数据集 - 样本数量: {len(val_dataset)}, 转换策略: 验证/测试")
    log(f"创建数据集 - 样本数量: {len(test_dataset)}, 转换策略: 验证/测试")

    log("初始化EVA-X分类模型")

    pretrained_path = '/tmp/pycharm_project_601/EVA-X-main/eva_x_tiny_patch16_merged520k_mim.pt'

    model = EVAXClassifier(pretrained_weight_path=pretrained_path, num_classes=1, dropout_rate=0.5, model_type='tiny')
    model = model.to(device)
    log("模型初始化完成")

    criterion = FocalLoss(alpha=0.8, gamma=2.5)

    patience = 15
    no_improvement_epochs = 0
    total_epochs = 100
    best_val_auc = 0.0
    best_model_path = 'best_evax_model.pth'
    best_val_loss = float('inf')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-8)

    log(f"开始训练，共 {total_epochs} 个epoch")
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(total_epochs):
        log(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")
        model.train()
        train_loss = 0

        for batch_idx, (img, labels) in enumerate(tqdm(train_loader, desc="Training")):
            img, labels = img.to(device), labels.to(device).float()
            optimizer.zero_grad()

            if random.random() < 0.3:
                mixed_img, labels_a, labels_b, lam = mixup_data(img, labels)
                outputs = model(mixed_img).squeeze()
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(img).squeeze()
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

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

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        log(f"当前学习率: {current_lr:.8f}")

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

    criterion = FocalLoss(alpha=0.8, gamma=2.5)

    all_test_probs = []
    all_test_labels = []

    with torch.no_grad():
        for img, labels in tqdm(test_loader, desc="Testing"):
            img, labels = img.to(device), labels.to(device).float()
            outputs = model(img).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_test_probs.extend(probs)
            all_test_labels.extend(labels.cpu().numpy())

    def find_optimal_threshold(labels, probs):
        best_threshold = 0.5
        best_auc = 0

        for threshold in np.arange(0.3, 0.7, 0.01):
            pred_labels = (np.array(probs) > threshold).astype(int)
            accuracy = np.mean(pred_labels == np.array(labels))

            try:
                from sklearn.metrics import roc_auc_score, confusion_matrix
                auc = roc_auc_score(labels, probs)
                cm = confusion_matrix(labels, pred_labels)

                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

                metrics = {
                    'auc': auc,
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1': f1
                }

                if auc > best_auc:
                    best_auc = auc
                    best_threshold = threshold
                    best_metrics = metrics
            except:
                pass

        return best_threshold, best_metrics

    best_threshold, best_metrics = find_optimal_threshold(all_test_labels, all_test_probs)

    log("\n=== 结果总结 ===")
    log(f"最佳验证AUC: {best_val_auc:.4f}")
    log(f"测试最佳阈值: {best_threshold:.4f}")
    log(f"测试AUC: {best_metrics['auc']:.4f}")
    log(f"测试准确率: {best_metrics['accuracy'] * 100:.2f}%")
    log(f"测试灵敏度: {best_metrics['sensitivity']:.4f}")
    log(f"测试特异度: {best_metrics['specificity']:.4f}")
    log(f"测试F1: {best_metrics['f1']:.4f}")

    log("\n绘制训练曲线...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='orange')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='验证AUC', color='green')
    plt.title('验证AUC曲线')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    log("训练曲线已显示")

    print("\n训练完成！")

if __name__ == "__main__":
    main()
