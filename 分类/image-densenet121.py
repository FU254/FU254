#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿照 tiny-new-阻塞.py 的完整训练流程，使用 torchvision ImageNet 预训练
DenseNet121 训练二分类模型，并支持可配置的分层解冻策略。
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import logging
import random
import shutil
import hashlib
import re
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from fixed_group_split import load_or_create_fixed_group_split


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def log(message):
    logger.info(message)


IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 41
EXCEL_PATH1 = r'/home/cfu/.virtualenvs/DR/TEST/DR已筛选2019-2023.xlsx'
EXCEL_PATH2 = r'/tmp/pycharm_project_601/肺功能万能表最终版1.xlsx'
DICOM_DIR = r'/tmp/pycharm_project_601/DR数据'
LABEL_COLUMN = "阻塞性通气障碍"
SEED = 42
LEARNING_RATE = 5e-5
NORM5_LEARNING_RATE = 2e-5
DENSEBLOCK4_LEARNING_RATE = 1e-5
TRANSITION3_LEARNING_RATE = 2e-6
DENSEBLOCK3_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
TOTAL_EPOCHS = 400
PATIENCE = 30
MIXUP_PROBABILITY = 0.15
IMAGENET_WEIGHTS_NAME = "IMAGENET1K_V2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_FINE_TUNE_STRATEGIES = {
    "classifier",
    "norm5_classifier",
    "denseblock4_norm5_classifier",
    "transition3_denseblock4_norm5_classifier",
    "denseblock3_transition3_denseblock4_norm5_classifier",
}
# Linux 示例：
# DENSENET_FINE_TUNE_STRATEGY=denseblock4_norm5_classifier python image-densenet121.py
FINE_TUNE_STRATEGY = os.environ.get(
    "DENSENET_FINE_TUNE_STRATEGY",
    "denseblock4_norm5_classifier",
).strip()
if FINE_TUNE_STRATEGY not in VALID_FINE_TUNE_STRATEGIES:
    raise ValueError(
        "DENSENET_FINE_TUNE_STRATEGY 不受支持: "
        f"{FINE_TUNE_STRATEGY}; 可选值: {sorted(VALID_FINE_TUNE_STRATEGIES)}"
    )
RUN_MODEL_NAME = "torchvision_densenet121_imagenet"
PRETRAIN_SOURCE = f"torchvision:{IMAGENET_WEIGHTS_NAME}"
TASK_NAME = f"densenet121_imagenet_{FINE_TUNE_STRATEGY}"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs", TASK_NAME)


def make_safe_name(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|\s]+', "_", value)
    return value.strip("_") or "task"


LABEL_TAG = make_safe_name(LABEL_COLUMN)
FIXED_GROUP_SPLIT_PATH = os.environ.get(
    "FIXED_GROUP_SPLIT_PATH",
    os.path.join(
        os.path.dirname(EXCEL_PATH2),
        f"fixed_group_split_{LABEL_TAG}_70_10_20_seed{SEED}.csv",
    ),
)
CACHE_FILE = os.path.join(OUTPUT_DIR, f"cached_data_{LABEL_TAG}.npz")
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, f"best_imagenet_densenet121_model_{LABEL_TAG}.pth")
RUN_INFO_FILE = os.path.join(OUTPUT_DIR, f"run_info_{LABEL_TAG}.txt")

device = torch.device("cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_unique_seq(accession, df1):
    seq_list = df1[df1["放射编号"] == accession]["序号"].tolist()
    if len(seq_list) > 1:
        log(f"警告: Accession {accession} 对应多个序号，默认取第一个")
    return seq_list[0] if seq_list else None


def setup_hardware():
    log("开始初始化硬件配置")
    if torch.cuda.is_available():
        preferred_index = 1 if torch.cuda.device_count() > 1 else 0
        current_device = torch.device(f"cuda:{preferred_index}")
        gpu_name = torch.cuda.get_device_name(preferred_index)
        log(f"使用设备: cuda:{preferred_index}")
        log(f"GPU型号: {gpu_name}")
    else:
        current_device = torch.device("cpu")
        log("使用设备: cpu")
    log("硬件配置初始化完成")
    return current_device


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce_loss
        return focal_loss.mean()


class DenseNet121BinaryClassifier(nn.Module):
    def __init__(self, fine_tune_strategy=FINE_TUNE_STRATEGY):
        super().__init__()
        if fine_tune_strategy not in VALID_FINE_TUNE_STRATEGIES:
            raise ValueError(
                f"Unsupported fine_tune_strategy: {fine_tune_strategy}. "
                f"Available: {sorted(VALID_FINE_TUNE_STRATEGIES)}"
            )
        self.fine_tune_strategy = fine_tune_strategy
        self.backbone = self._build_backbone()
        self._frozen_feature_modules = []
        self._freeze_front_layers()

    @staticmethod
    def _build_backbone():
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V2
            model = models.densenet121(weights=weights)
            log(
                "已加载 torchvision ImageNet 预训练 DenseNet121: "
                f"{IMAGENET_WEIGHTS_NAME}"
            )
        except AttributeError:
            try:
                model = models.densenet121(pretrained=True)
                log("已加载 torchvision ImageNet 预训练 DenseNet121（旧版 API）")
            except Exception as exc:
                raise RuntimeError("加载 ImageNet 预训练 DenseNet121 失败") from exc
        except Exception as exc:
            raise RuntimeError("加载 ImageNet 预训练 DenseNet121 失败") from exc

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.constant_(model.classifier.bias, 0)
        return model

    def _freeze_front_layers(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        strategy_to_feature_names = {
            "classifier": set(),
            "norm5_classifier": {"norm5"},
            "denseblock4_norm5_classifier": {"denseblock4", "norm5"},
            "transition3_denseblock4_norm5_classifier": {
                "transition3",
                "denseblock4",
                "norm5",
            },
            "denseblock3_transition3_denseblock4_norm5_classifier": {
                "denseblock3",
                "transition3",
                "denseblock4",
                "norm5",
            },
        }
        frozen_feature_names = []
        unfrozen_feature_names = []

        for name, module in self.backbone.features.named_children():
            if name in strategy_to_feature_names[self.fine_tune_strategy]:
                for param in module.parameters():
                    param.requires_grad = True
                unfrozen_feature_names.append(f"features.{name}")
            else:
                self._frozen_feature_modules.append(module)
                frozen_feature_names.append(f"features.{name}")

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        unfrozen_feature_names.append("classifier")
        log(f"fine_tune_strategy: {self.fine_tune_strategy}")

        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_count = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log(f"冻结 DenseNet 前段层: {', '.join(frozen_feature_names)}")
        log(f"保留可训练层: {', '.join(unfrozen_feature_names)}")
        log(f"参数状态 | 可训练: {trainable_count} | 冻结: {frozen_count}")

    def forward(self, x):
        return self.backbone(x)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self._frozen_feature_modules:
                module.eval()
        return self


def mixup_data(x_img, y, alpha=0.5):
    if alpha <= 0:
        return x_img, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x_img.size(0)
    index = torch.randperm(batch_size, device=x_img.device)
    mixed_x = lam * x_img + (1 - lam) * x_img[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def split_dataset(metadata, y, seed):
    return load_or_create_fixed_group_split(
        metadata=metadata,
        labels=y,
        manifest_path=FIXED_GROUP_SPLIT_PATH,
        seed=seed,
        log_fn=log,
    )


def load_and_preprocess_data(force_reload=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force_reload and os.path.exists(CACHE_FILE):
        log("从缓存加载数据...")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        metadata = None
        if {"accessions", "seqs", "filenames"}.issubset(cached.files):
            metadata = pd.DataFrame(
                {
                    "放射编号": cached["accessions"].astype(str),
                    "序号": cached["seqs"].astype(str),
                    "filename": cached["filenames"].astype(str),
                }
            )
        log(f"数据从缓存加载完成，共 {len(X)} 个样本")
        log(f"当前缓存路径: {CACHE_FILE}")
        class_counts = np.bincount(y.astype(int))
        neg_count = int(class_counts[0]) if len(class_counts) > 0 else 0
        pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
        log(f"缓存类别分布: 0类 {neg_count}, 1类 {pos_count}")
        return X, y, metadata

    if force_reload and os.path.exists(CACHE_FILE):
        log("强制重新加载数据，删除旧缓存...")
        os.remove(CACHE_FILE)

    log("开始加载真实DICOM数据...")

    df1 = pd.read_excel(EXCEL_PATH1, sheet_name=0)
    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)

    df1["序号"] = df1["序号"].astype(str).str.strip()
    label_match_mode = None
    if "序号" not in df2.columns and "Unnamed: 0" in df2.columns:
        df2 = df2.rename(columns={"Unnamed: 0": "序号"})
    if "序号" in df2.columns:
        df2["序号"] = df2["序号"].astype(str).str.strip()
        label_match_mode = "seq"
    elif "放射编号" in df2.columns:
        df2["放射编号"] = df2["放射编号"].astype(str).str.strip()
        label_match_mode = "accession"
    else:
        log("错误: df2 中既不存在 '序号' 列，也不存在 '放射编号' 列")
        log(f"请检查 Excel 文件，现有列名为: {df2.columns.tolist()}")
        return np.array([]), np.array([]), None

    log(f"df1 中的列名: {df1.columns.tolist()}")
    log(f"df2 中的列名: {df2.columns.tolist()}")
    log(f"df2 标签匹配方式: {label_match_mode}")

    if LABEL_COLUMN not in df2.columns:
        log(f"错误: df2 中不存在 '{LABEL_COLUMN}' 列")
        log(f"请检查列名是否正确，现有列名为: {df2.columns.tolist()}")
        return np.array([]), np.array([]), None

    X, y = [], []
    accessions, seqs, filenames = [], [], []
    filter_reasons = {
        "other_error": 0,
        "no_pixel_data": 0,
        "no_sequence": 0,
        "invalid_label": 0,
        "missing_label_column": 0,
    }
    filtered_filenames = []

    for filename in tqdm(os.listdir(DICOM_DIR), desc="Processing DICOM"):
        if not filename.endswith(".dcm"):
            continue

        try:
            dcm = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
            if "PixelData" not in dcm:
                filter_reasons["no_pixel_data"] += 1
                filtered_filenames.append(filename)
                continue

            accession = dcm.AccessionNumber
            accession_str = str(accession).strip()
            seq = get_unique_seq(accession, df1)

            if label_match_mode == "seq":
                if seq is None:
                    filter_reasons["no_sequence"] += 1
                    filtered_filenames.append(filename)
                    continue
                label_df = df2[df2["序号"] == str(seq).strip()]
            else:
                label_df = df2[df2["放射编号"] == accession_str]

            if len(label_df) == 0:
                filter_reasons["invalid_label"] += 1
                filtered_filenames.append(filename)
                continue

            try:
                label_value = label_df[LABEL_COLUMN].values[0]
                label = int(label_value)
            except (ValueError, IndexError):
                filter_reasons["invalid_label"] += 1
                filtered_filenames.append(filename)
                continue
            except KeyError:
                filter_reasons["missing_label_column"] += 1
                filtered_filenames.append(filename)
                continue

            if label not in {0, 1}:
                filter_reasons["invalid_label"] += 1
                filtered_filenames.append(filename)
                continue

            img = dcm.pixel_array.astype(np.float32)
            height, width = img.shape
            if height > width:
                start = (height - width) // 2
                img = img[start:start + width, :]
            else:
                start = (width - height) // 2
                img = img[:, start:start + height]

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            img = img / (img.max() + 1e-8)

            X.append(img)
            y.append(label)
            accessions.append(accession_str)
            seqs.append("" if seq is None else str(seq).strip())
            filenames.append(filename)
        except Exception as exc:
            log(f"处理文件 {filename} 失败: {exc}")
            filter_reasons["other_error"] += 1
            filtered_filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    log(f"过滤原因统计: {filter_reasons}")
    log(f"被过滤文件数: {len(filtered_filenames)}")
    log(f"数据加载完成，共 {len(X)} 个样本")

    class_counts = np.bincount(y.astype(int))
    neg_count = int(class_counts[0]) if len(class_counts) > 0 else 0
    pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
    log(f"类别分布: 0类 {neg_count}, 1类 {pos_count}")

    log("保存数据到缓存...")
    np.savez_compressed(
        CACHE_FILE,
        X=X,
        y=y,
        accessions=np.array(accessions, dtype=str),
        seqs=np.array(seqs, dtype=str),
        filenames=np.array(filenames, dtype=str),
    )
    log(f"缓存已保存到: {CACHE_FILE}")

    metadata = pd.DataFrame(
        {
            "放射编号": accessions,
            "序号": seqs,
            "filename": filenames,
        }
    )
    return X, y, metadata


def save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp):
    if metadata is None:
        log("当前缓存不包含放射编号/序号元数据，无法导出 split 清单")
        return None

    split_df = metadata.copy()
    split_df["split"] = ""
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"

    split_path = os.path.join(
        OUTPUT_DIR, f"imagenet_densenet121_split_{LABEL_TAG}_{run_stamp}.csv"
    )
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    log(f"已导出本次数据划分: {split_path}")
    return split_path


def save_label_manifest(metadata, y, run_stamp):
    if metadata is None or len(metadata) != len(y):
        log("当前数据缺少完整元数据，无法导出 label manifest")
        return None, None

    manifest_df = metadata.copy()
    manifest_df["label"] = y.astype(int)
    manifest_path = os.path.join(OUTPUT_DIR, f"label_manifest_{LABEL_TAG}_{run_stamp}.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    signature_source = (
        manifest_df["放射编号"].astype(str) + "|"
        + manifest_df["序号"].astype(str) + "|"
        + manifest_df["label"].astype(str)
    )
    label_signature = hashlib.sha1("\n".join(signature_source.tolist()).encode("utf-8")).hexdigest()
    log(f"标签清单已导出: {manifest_path}")
    log(f"标签签名: {label_signature}")
    return manifest_path, label_signature


def write_run_info(run_stamp, cache_path, best_model_path, split_path, manifest_path, label_signature, y):
    class_counts = np.bincount(y.astype(int))
    neg_count = int(class_counts[0]) if len(class_counts) > 0 else 0
    pos_count = int(class_counts[1]) if len(class_counts) > 1 else 0
    lines = [
        f"run_stamp={run_stamp}",
        f"model_name={RUN_MODEL_NAME}",
        f"pretrained_source={PRETRAIN_SOURCE}",
        f"imagenet_weights={IMAGENET_WEIGHTS_NAME}",
        f"fine_tune_strategy={FINE_TUNE_STRATEGY}",
        f"task_name={TASK_NAME}",
        f"label_column={LABEL_COLUMN}",
        f"label_tag={LABEL_TAG}",
        f"excel_path={EXCEL_PATH2}",
        f"cache_path={cache_path}",
        f"best_model_path={best_model_path}",
        f"fixed_group_split_path={FIXED_GROUP_SPLIT_PATH}",
        f"split_path={split_path or ''}",
        f"label_manifest_path={manifest_path or ''}",
        f"label_signature={label_signature or ''}",
        f"sample_count={len(y)}",
        f"class_0_count={neg_count}",
        f"class_1_count={pos_count}",
    ]
    with open(RUN_INFO_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    log(f"任务信息已保存: {RUN_INFO_FILE}")


class MedicalImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for img, labels in data_loader:
            img = img.to(device)
            labels = labels.to(device).float()
            outputs = model(img).view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    total_loss /= len(data_loader)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    pred_labels = (all_probs > 0.5).astype(int)
    accuracy = np.mean(pred_labels == all_labels)

    if np.unique(all_labels).size < 2:
        roc_auc = 0.5
        pr_auc = float(all_labels.mean())
    else:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

        roc_auc = roc_auc_score(all_labels, all_probs)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)

    return total_loss, accuracy, roc_auc, pr_auc


def collect_probs(model, data_loader, desc):
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for img, labels in tqdm(data_loader, desc=desc):
            img = img.to(device)
            labels = labels.to(device).float()
            outputs = model(img).view(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def compute_metrics(labels, probs, threshold):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    pred_labels = (np.array(probs) > threshold).astype(int)
    accuracy = np.mean(pred_labels == np.array(labels))
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_pr_curve(labels, probs, output_path):
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    positive_rate = np.mean(labels)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", linewidth=2,
             label=f"Test PR curve (PR-AUC = {pr_auc:.4f})")
    plt.axhline(positive_rate, color="navy", linestyle="--", linewidth=2,
                label=f"No skill (positive rate = {positive_rate:.4f})")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ImageNet DenseNet121 Test Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return pr_auc


def find_optimal_threshold(labels, probs):
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = compute_metrics(labels, probs, best_threshold)

    for threshold in np.arange(0.05, 0.96, 0.01):
        metrics = compute_metrics(labels, probs, threshold)
        youden = metrics["sensitivity"] + metrics["specificity"] - 1
        if youden > best_score or (np.isclose(youden, best_score) and metrics["f1"] > best_metrics["f1"]):
            best_score = youden
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics, best_score


def show_training_curves(
    train_losses,
    val_losses,
    train_aucs,
    val_aucs,
    train_pr_aucs,
    val_pr_aucs,
):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_aucs, label="Train AUC", color="blue")
    plt.plot(val_aucs, label="Val AUC", color="orange")
    plt.title("Training & Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_pr_aucs, label="Train PR-AUC", color="blue")
    plt.plot(val_pr_aucs, label="Val PR-AUC", color="orange")
    plt.title("Training & Validation PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    log("训练曲线已显示")


def show_confusion_matrix(confusion_matrix_array, threshold):
    plt.figure(figsize=(5, 4))
    plt.imshow(confusion_matrix_array, interpolation="nearest", cmap="Blues")
    plt.title(f"Test Confusion Matrix (threshold={threshold:.2f})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])

    max_value = confusion_matrix_array.max() if confusion_matrix_array.size > 0 else 0
    for i in range(confusion_matrix_array.shape[0]):
        for j in range(confusion_matrix_array.shape[1]):
            value = int(confusion_matrix_array[i, j])
            color = "white" if value > max_value / 2 else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=color)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
    log("混淆矩阵已显示")


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
    log("模型架构: torchvision DenseNet121")
    log(f"预训练来源: ImageNet-1K ({IMAGENET_WEIGHTS_NAME})")
    log(f"任务输出目录: {OUTPUT_DIR}")
    log(f"缓存文件: {CACHE_FILE}")
    log(f"最佳模型文件: {BEST_MODEL_FILE}")
    log(f"当前解冻策略: {FINE_TUNE_STRATEGY}")
    log(
        "可选解冻策略（环境变量 DENSENET_FINE_TUNE_STRATEGY）: "
        f"{', '.join(sorted(VALID_FINE_TUNE_STRATEGIES))}"
    )
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

    log(
        "ImageNet 输入归一化 | "
        f"mean={IMAGENET_MEAN} | std={IMAGENET_STD} | 灰度图复制为3通道"
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.92, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = MedicalImageDataset(X_train, y_train, transform=train_transform)
    train_eval_dataset = MedicalImageDataset(X_train, y_train, transform=val_test_transform)
    val_dataset = MedicalImageDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = MedicalImageDataset(X_test, y_test, transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    log(f"创建数据集 - 样本数量: {len(train_dataset)}, 转换策略: 训练增强")
    log(f"创建数据集 - 样本数量: {len(val_dataset)}, 转换策略: 验证/测试")
    log(f"创建数据集 - 样本数量: {len(test_dataset)}, 转换策略: 验证/测试")

    log(f"初始化 ImageNet 预训练 DenseNet121 | 解冻策略: {FINE_TUNE_STRATEGY}")
    log(f"train_samples: {len(train_eval_dataset)}")
    model = DenseNet121BinaryClassifier(fine_tune_strategy=FINE_TUNE_STRATEGY).to(device)

    train_class_counts = np.bincount(y_train.astype(int), minlength=2)
    train_neg_count = int(train_class_counts[0])
    train_pos_count = int(train_class_counts[1])
    if train_neg_count == 0 or train_pos_count == 0:
        raise RuntimeError("训练集必须同时包含正负样本，无法进行二分类训练。")
    class_ratio = train_neg_count / train_pos_count
    criterion = FocalLoss(alpha=0.8, gamma=2.5)
    log(
        f"训练集类别分布 | 0类={train_neg_count} | 1类={train_pos_count} | "
        f"负/正比例={class_ratio:.6f}"
    )
    log("损失函数: FocalLoss | alpha=0.8 | gamma=2.5")

    classifier_params = []
    norm5_params = []
    denseblock4_params = []
    transition3_params = []
    denseblock3_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.classifier"):
            classifier_params.append(param)
        elif name.startswith("backbone.features.norm5"):
            norm5_params.append(param)
        elif name.startswith("backbone.features.denseblock4"):
            denseblock4_params.append(param)
        elif name.startswith("backbone.features.transition3"):
            transition3_params.append(param)
        elif name.startswith("backbone.features.denseblock3"):
            denseblock3_params.append(param)

    trainable_params = (
        classifier_params
        + norm5_params
        + denseblock4_params
        + transition3_params
        + denseblock3_params
    )
    optimizer_param_groups = []
    if classifier_params:
        optimizer_param_groups.append(
            {"params": classifier_params, "lr": LEARNING_RATE, "name": "classifier"}
        )
    if norm5_params:
        optimizer_param_groups.append(
            {"params": norm5_params, "lr": NORM5_LEARNING_RATE, "name": "norm5"}
        )
    if denseblock4_params:
        optimizer_param_groups.append(
            {
                "params": denseblock4_params,
                "lr": DENSEBLOCK4_LEARNING_RATE,
                "name": "denseblock4",
            }
        )
    if transition3_params:
        optimizer_param_groups.append(
            {
                "params": transition3_params,
                "lr": TRANSITION3_LEARNING_RATE,
                "name": "transition3",
            }
        )
    if denseblock3_params:
        optimizer_param_groups.append(
            {
                "params": denseblock3_params,
                "lr": DENSEBLOCK3_LEARNING_RATE,
                "name": "denseblock3",
            }
        )
    if not optimizer_param_groups:
        raise RuntimeError("No trainable parameters were selected by fine_tune_strategy.")
    optimizer = optim.AdamW(
        optimizer_param_groups,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        threshold=1e-4,
        min_lr=1e-7,
    )

    best_val_auc = 0.0
    no_improvement_epochs = 0
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    train_pr_aucs = []
    val_pr_aucs = []

    log(f"开始训练，共 {TOTAL_EPOCHS} 个 epoch")
    log(
        f"总参数数量: {sum(p.numel() for p in model.parameters())} | "
        f"可训练参数数量: {sum(p.numel() for p in trainable_params)}"
    )
    log(
        "参数组学习率 | "
        f"classifier={LEARNING_RATE:.6f} | "
        f"norm5={NORM5_LEARNING_RATE:.6f} | "
        f"denseblock4={DENSEBLOCK4_LEARNING_RATE:.6f} | "
        f"transition3={TRANSITION3_LEARNING_RATE:.6f} | "
        f"denseblock3={DENSEBLOCK3_LEARNING_RATE:.6f}"
    )

    log(
        "criterion=FocalLoss(alpha=0.8, gamma=2.5) | "
        f"mixup_probability={MIXUP_PROBABILITY:.2f}"
    )
    log("scheduler=ReduceLROnPlateau(mode=max, factor=0.5, patience=10, min_lr=1e-7)")
    initial_lrs = ", ".join(
        f"{group['name']}={group['lr']:.6f}" for group in optimizer.param_groups
    )
    log(f"active_param_group_lrs: {initial_lrs}")

    for epoch in range(TOTAL_EPOCHS):
        log(f"\n=== Epoch {epoch + 1}/{TOTAL_EPOCHS} ===")
        model.train()
        for img, labels in tqdm(train_loader, desc="Training"):
            img = img.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()

            if random.random() < MIXUP_PROBABILITY:
                mixed_img, labels_a, labels_b, lam = mixup_data(img, labels)
                outputs = model(mixed_img).view(-1)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(img).view(-1)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss, train_acc, train_auc, train_pr_auc = validate(model, train_eval_loader, criterion)
        val_loss, val_acc, val_auc, val_pr_auc = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_pr_aucs.append(train_pr_auc)
        val_pr_aucs.append(val_pr_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improvement_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            log(f"最佳模型已保存 (AUC: {best_val_auc:.4f})")
        else:
            no_improvement_epochs += 1
            log(f"未改进: {no_improvement_epochs}/{PATIENCE}")

        scheduler.step(val_auc)
        current_lrs = ", ".join(
            f"{group.get('name', f'group{idx}')}={group['lr']:.8f}"
            for idx, group in enumerate(optimizer.param_groups)
        )
        log(f"当前学习率: {current_lrs}")
        log(
            f"训练损失: {train_loss:.6f} | 训练Acc: {train_acc:.4f} | "
            f"训练ROC-AUC: {train_auc:.4f} | 训练PR-AUC: {train_pr_auc:.4f} | "
            f"验证损失: {val_loss:.6f} | 验证Acc: {val_acc:.4f} | "
            f"验证ROC-AUC: {val_auc:.4f} | 验证PR-AUC: {val_pr_auc:.4f}"
        )

        if no_improvement_epochs >= PATIENCE:
            log("早停触发，训练结束")
            break

    log("加载最佳模型进行测试...")
    if os.path.exists(BEST_MODEL_FILE):
        checkpoint = torch.load(BEST_MODEL_FILE, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint)
        log(f"成功加载最佳模型: {BEST_MODEL_FILE}")
    else:
        log("警告: 未找到最佳模型文件，将使用当前模型继续评估")
    model.eval()

    if os.path.exists(BEST_MODEL_FILE):
        archived_model_path = os.path.join(
            OUTPUT_DIR,
            f"best_imagenet_densenet121_model_{LABEL_TAG}_auc{best_val_auc:.4f}_{run_stamp}.pth",
        )
        shutil.copy2(BEST_MODEL_FILE, archived_model_path)
        log(f"已归档最佳模型: {archived_model_path}")

    all_val_labels, all_val_probs = collect_probs(model, val_loader, "Selecting Threshold")
    best_threshold, val_threshold_metrics, best_youden = find_optimal_threshold(
        all_val_labels, all_val_probs
    )
    all_test_labels, all_test_probs = collect_probs(model, test_loader, "Testing")
    best_metrics = compute_metrics(all_test_labels, all_test_probs, best_threshold)
    pr_curve_path = os.path.join(
        OUTPUT_DIR, f"imagenet_densenet121_test_pr_curve_{LABEL_TAG}_{run_stamp}.png"
    )
    test_pr_auc = plot_pr_curve(all_test_labels, all_test_probs, pr_curve_path)

    log("\n=== 结果总结 ===")
    log(f"最佳验证AUC: {best_val_auc:.4f}")
    log(f"验证集最佳阈值: {best_threshold:.4f}")
    log(f"验证集Youden指数: {best_youden:.4f}")
    log(f"验证集指标: {val_threshold_metrics}")
    log(f"测试AUC: {best_metrics['auc']:.4f}")
    log(f"测试PR-AUC: {test_pr_auc:.4f}")
    log(f"测试准确率: {best_metrics['accuracy']:.4f}")
    log(f"测试灵敏度: {best_metrics['sensitivity']:.4f}")
    log(f"测试特异度: {best_metrics['specificity']:.4f}")
    log(f"测试F1: {best_metrics['f1']:.4f}")
    log(
        f"测试混淆矩阵: [[TN={best_metrics['tn']}, FP={best_metrics['fp']}], "
        f"[FN={best_metrics['fn']}, TP={best_metrics['tp']}]]"
    )
    log(f"测试PR曲线已保存: {pr_curve_path}")

    log("\n绘制训练曲线...")
    show_training_curves(
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        train_pr_aucs,
        val_pr_aucs,
    )
    log("\n绘制测试集混淆矩阵...")
    show_confusion_matrix(best_metrics["confusion_matrix"], best_threshold)
    log("训练完成")


if __name__ == "__main__":
    main()
