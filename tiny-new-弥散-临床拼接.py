#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA-X tiny training script with gated fusion for image and clinical features.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import logging
import random
import shutil
import hashlib
from datetime import datetime
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def log(message):
    logger.info(message)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classification"))
from models.models_eva import eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

IMAGE_SIZE = 224
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_METRIC_EVERY = 2
ENABLE_STARTUP_VERBOSE_FORWARD = False
SCRIPT_DIR = os.path.dirname(__file__)
TASK_NAME = "tiny_new_diffusion_clinical_gate"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs", TASK_NAME)
EXCEL_PATH1 = "/home/cfu/.virtualenvs/DR/TEST/DR\u5df2\u7b5b\u90092019-2023.xlsx"
EXCEL_PATH2 = "/tmp/pycharm_project_601/\u80ba\u529f\u80fd\u4e07\u80fd\u8868\u6700\u7ec8\u72481.xlsx"
DICOM_DIR = "/home/cfu/.virtualenvs/DR/TEST/NEW.DR"
PRETRAINED_PATH = os.path.join(SCRIPT_DIR, "eva_x_tiny_patch16_merged520k_mim.pt")

ACCESSION_COL = "\u653e\u5c04\u7f16\u53f7"
SEQ_COL = "\u5e8f\u53f7"
LABEL_COLUMN = "\u5f25\u6563\u529f\u80fd\u4e0b\u964d"
CLINICAL_COLUMNS = ["age", "sex", "height", "weight", "smoking"]

SEED = 42
LEARNING_RATE = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-3
FREEZE_BACKBONE = True
UNFREEZE_LAST_N_BLOCKS = 3


def make_safe_name(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|\s]+', "_", value)
    return value.strip("_") or "task"


LABEL_TAG = make_safe_name(LABEL_COLUMN)
CACHE_FILE = os.path.join(OUTPUT_DIR, f"cached_data_{LABEL_TAG}_clinical.npz")
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, f"best_evax_tiny_model_{LABEL_TAG}.pth")
RUN_INFO_FILE = os.path.join(OUTPUT_DIR, f"run_info_{LABEL_TAG}.txt")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_unique_seq(accession, df1):
    seq_list = df1[df1[ACCESSION_COL] == accession][SEQ_COL].tolist()
    if len(seq_list) > 1:
        log(f"multiple seq values for accession={accession}, using first one")
    return seq_list[0] if seq_list else None


def setup_hardware():
    log("initializing hardware")
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        log(f"device={device}")
        log(f"gpu={torch.cuda.get_device_name(1)}")
    else:
        device = torch.device("cpu")
        log("device=cpu")
    return device


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


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
        if text in {"0", "1"}:
            return float(text)
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(numeric):
        return np.nan
    return numeric


def extract_clinical_features(row):
    return np.array(
        [
            parse_float(row.get("age")),
            encode_binary_feature(row.get("sex"), {"1", "m", "male"}),
            parse_float(row.get("height")),
            parse_float(row.get("weight")),
            encode_binary_feature(row.get("smoking"), {"1", "yes", "y", "true", "current", "former"}),
        ],
        dtype=np.float32,
    )


class EVAXClassifier(nn.Module):
    def __init__(
        self,
        pretrained_weight_path=None,
        num_classes=1,
        dropout_rate=0.7,
        model_type="tiny",
        freeze_backbone=False,
        unfreeze_last_n_blocks=0,
        clinical_dim=0,
    ):
        super().__init__()

        log("creating EVA-X tiny model")
        self.backbone = eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE(pretrained=False)

        if pretrained_weight_path is not None and os.path.exists(pretrained_weight_path):
            log(f"loading pretrained weights: {pretrained_weight_path}")
            checkpoint = torch.load(pretrained_weight_path, map_location="cpu", weights_only=False)

            if "module" in checkpoint:
                state_dict = checkpoint["module"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    if k.startswith("module."):
                        k = k[7:]
                    new_state_dict[k] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            log("weights loaded")
            if msg.missing_keys:
                log(f"missing keys: {msg.missing_keys}")
            if msg.unexpected_keys:
                log(f"unexpected keys: {msg.unexpected_keys}")
        else:
            log(f"warning: pretrained weights not found: {pretrained_weight_path}")

        self.backbone.fc_norm = nn.LayerNorm(self.backbone.embed_dim)
        self.bottleneck = nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim // 2)
        self.bottleneck2 = nn.Linear(self.backbone.embed_dim // 2, self.backbone.embed_dim // 4)
        self.dropout = nn.Dropout(dropout_rate)

        fusion_dim = self.backbone.embed_dim // 4
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self._init_head()

        if freeze_backbone:
            self.freeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    def _init_head(self):
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def freeze_backbone(self, unfreeze_last_n_blocks=0):
        log("freezing backbone")
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
            log(f"unfroze last {min(unfreeze_last_n_blocks, total_blocks)} transformer blocks")

    def forward(self, x, clinical=None, verbose=False):
        if verbose:
            print(f"\n{'=' * 60}")
            print("forward debug")
            print(f"{'=' * 60}")
            print(f"input shape={x.shape} dtype={x.dtype}")

        x = self.backbone.patch_embed(x)
        batch_size = x.size(0)
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

        h_img = torch.relu(self.bottleneck(features))
        h_img = self.dropout(h_img)
        h_img = torch.relu(self.bottleneck2(h_img))
        h_img = self.dropout(h_img)

        if clinical is None:
            raise ValueError("clinical input is required")
        h_cli = self.clinical_encoder(clinical)
        gate = self.gate(torch.cat([h_img, h_cli], dim=1))
        fused = gate * h_img + (1.0 - gate) * h_cli
        out = self.classifier(fused)

        if verbose:
            print(f"h_img shape={h_img.shape}")
            print(f"h_cli shape={h_cli.shape}")
            print(f"gate shape={gate.shape}")
            print(f"fused shape={fused.shape}")
            print(f"output shape={out.shape}")
            print(f"{'=' * 60}")

        return out


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
        log(f"stratified split failed, fallback to random split: {e}")
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        train_ratio = 0.64
        val_ratio = 0.16
        train_end = int(len(indices) * train_ratio)
        val_end = int(len(indices) * (train_ratio + val_ratio))
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def load_and_preprocess_data(force_reload=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force_reload and os.path.exists(CACHE_FILE):
        log("loading cached data")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        clinical = cached["clinical"].astype(np.float32) if "clinical" in cached.files else None
        metadata = None
        if {"accessions", "seqs", "filenames"}.issubset(cached.files):
            metadata = pd.DataFrame(
                {
                    ACCESSION_COL: cached["accessions"].astype(str),
                    SEQ_COL: cached["seqs"].astype(str),
                    "filename": cached["filenames"].astype(str),
                }
            )
        if clinical is not None:
            log(f"loaded {len(X)} samples from cache")
            return X, y, clinical, metadata
        log("cache missing clinical features, rebuilding")

    elif force_reload and os.path.exists(CACHE_FILE):
        log("force reload enabled, deleting old cache")
        os.remove(CACHE_FILE)

    log("loading DICOM data")

    df1 = pd.read_excel(EXCEL_PATH1, sheet_name=0)
    df2 = pd.read_excel(EXCEL_PATH2, sheet_name=0)

    df1[SEQ_COL] = df1[SEQ_COL].astype(str).str.strip()
    if ACCESSION_COL not in df2.columns:
        log(f"missing required column in df2: {ACCESSION_COL}")
        return np.array([]), np.array([]), np.array([]), None
    df2[ACCESSION_COL] = df2[ACCESSION_COL].astype(str).str.strip()

    if LABEL_COLUMN not in df2.columns:
        log(f"missing label column in df2: {LABEL_COLUMN}")
        return np.array([]), np.array([]), np.array([]), None

    X, y, clinical = [], [], []
    accessions, seqs, filenames = [], [], []

    for filename in tqdm(os.listdir(DICOM_DIR), desc="Processing DICOM"):
        if not filename.endswith(".dcm"):
            continue
        try:
            dcm = pydicom.dcmread(os.path.join(DICOM_DIR, filename))
            if "PixelData" not in dcm:
                continue

            accession = str(dcm.AccessionNumber).strip()
            seq = get_unique_seq(accession, df1)
            if seq is None:
                continue

            label_df = df2[df2[ACCESSION_COL] == accession]
            if len(label_df) == 0:
                continue

            try:
                label = int(label_df[LABEL_COLUMN].values[0])
            except (ValueError, IndexError, KeyError):
                continue
            if label not in {0, 1}:
                continue

            clinical_feature = extract_clinical_features(label_df.iloc[0])
            img = dcm.pixel_array.astype(float)

            height, width = img.shape
            if height > width:
                start = (height - width) // 2
                img = img[start:start + width, :]
            else:
                start = (width - height) // 2
                img = img[:, start:start + height]

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            img = img / (img.max() + 1e-8)

            X.append(img)
            y.append(label)
            clinical.append(clinical_feature)
            accessions.append(accession)
            seqs.append(str(seq).strip())
            filenames.append(filename)
        except Exception as e:
            log(f"failed to process {filename}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    clinical = np.array(clinical, dtype=np.float32)

    np.savez_compressed(
        CACHE_FILE,
        X=X,
        y=y,
        clinical=clinical,
        accessions=np.array(accessions, dtype=str),
        seqs=np.array(seqs, dtype=str),
        filenames=np.array(filenames, dtype=str),
    )
    log(f"saved cache to {CACHE_FILE}")

    metadata = pd.DataFrame({ACCESSION_COL: accessions, SEQ_COL: seqs, "filename": filenames})
    return X, y, clinical, metadata


def normalize_clinical(train_data, val_data, test_data):
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()
    means = np.nanmean(train_data, axis=0)
    stds = np.nanstd(train_data, axis=0)
    stds[stds == 0] = 1.0

    for data in (train_data, val_data, test_data):
        missing = np.isnan(data)
        data[missing] = np.take(means, np.where(missing)[1])
        data[:] = (data - means) / stds

    return train_data, val_data, test_data


def save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp):
    if metadata is None:
        log("metadata missing, skip split manifest")
        return None

    split_df = metadata.copy()
    split_df["split"] = ""
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"

    split_path = os.path.join(OUTPUT_DIR, f"evax_tiny_split_{LABEL_TAG}_{run_stamp}.csv")
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
    log(f"saved split manifest: {split_path}")
    return split_path


def save_label_manifest(metadata, y, run_stamp):
    if metadata is None or len(metadata) != len(y):
        log("metadata incomplete, skip label manifest")
        return None, None

    manifest_df = metadata.copy()
    manifest_df["label"] = y.astype(int)
    manifest_path = os.path.join(OUTPUT_DIR, f"label_manifest_{LABEL_TAG}_{run_stamp}.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    signature_source = (
        manifest_df[ACCESSION_COL].astype(str)
        + "|"
        + manifest_df[SEQ_COL].astype(str)
        + "|"
        + manifest_df["label"].astype(str)
    )
    label_signature = hashlib.sha1("\n".join(signature_source.tolist()).encode("utf-8")).hexdigest()
    log(f"saved label manifest: {manifest_path}")
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
    with open(RUN_INFO_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"saved run info: {RUN_INFO_FILE}")


class MedicalImageDataset(Dataset):
    def __init__(self, images, clinical, labels, transform=None, normalize_mean=None, normalize_std=None):
        self.images = images
        self.clinical = clinical
        self.labels = labels
        self.transform = transform
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)

        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)

        if self.normalize_mean is not None and self.normalize_std is not None:
            img = (img - self.normalize_mean) / self.normalize_std

        clinical = torch.tensor(self.clinical[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, clinical, label


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, clinical, labels in val_loader:
            img = img.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(img, clinical).view(-1)
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
    except Exception:
        auc = 0.5

    return val_loss, accuracy, auc


def main():
    global device

    set_seed()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = setup_hardware()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log(f"task={TASK_NAME}")
    log(f"label_source={EXCEL_PATH2}")
    log(f"label_column={LABEL_COLUMN}")
    log(f"output_dir={OUTPUT_DIR}")
    log(f"cache_file={CACHE_FILE}")
    log(f"best_model_file={BEST_MODEL_FILE}")

    X, y, clinical, metadata = load_and_preprocess_data(force_reload=False)
    if len(X) == 0:
        log("data loading failed, stop training")
        return

    train_idx, val_idx, test_idx = split_dataset(X, y, SEED)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    c_train, c_val, c_test = clinical[train_idx], clinical[val_idx], clinical[test_idx]
    c_train, c_val, c_test = normalize_clinical(c_train, c_val, c_test)

    split_path = save_split_manifest(metadata, train_idx, val_idx, test_idx, run_stamp)
    manifest_path, label_signature = save_label_manifest(metadata, y, run_stamp)
    write_run_info(run_stamp, CACHE_FILE, BEST_MODEL_FILE, split_path, manifest_path, label_signature, y)

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    log(f"image_norm mean={X_train_mean:.4f} std={X_train_std:.4f}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )

    train_dataset = MedicalImageDataset(
        X_train, c_train, y_train, transform=train_transform,
        normalize_mean=X_train_mean, normalize_std=X_train_std
    )
    train_eval_dataset = MedicalImageDataset(
        X_train, c_train, y_train, transform=val_test_transform,
        normalize_mean=X_train_mean, normalize_std=X_train_std
    )
    val_dataset = MedicalImageDataset(
        X_val, c_val, y_val, transform=val_test_transform,
        normalize_mean=X_train_mean, normalize_std=X_train_std
    )
    test_dataset = MedicalImageDataset(
        X_test, c_test, y_test, transform=val_test_transform,
        normalize_mean=X_train_mean, normalize_std=X_train_std
    )

    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "pin_memory": True}
    if NUM_WORKERS > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = EVAXClassifier(
        pretrained_weight_path=PRETRAINED_PATH,
        num_classes=1,
        dropout_rate=0.6,
        model_type="tiny",
        freeze_backbone=FREEZE_BACKBONE,
        unfreeze_last_n_blocks=UNFREEZE_LAST_N_BLOCKS,
        clinical_dim=len(CLINICAL_COLUMNS),
    ).to(device)

    criterion = FocalLoss(alpha=0.8, gamma=2.5)

    patience = 25
    no_improvement_epochs = 0
    total_epochs = 300
    best_val_auc = 0.0
    best_model_path = BEST_MODEL_FILE

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable_params={sum(p.numel() for p in trainable_params)}")

    test_input = torch.randn(2, 3, 224, 224).to(device)
    test_clinical = torch.zeros(2, len(CLINICAL_COLUMNS)).to(device)
    with torch.no_grad():
        model.eval()
        _ = model(test_input, test_clinical, verbose=ENABLE_STARTUP_VERBOSE_FORWARD)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": LEARNING_RATE})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    if not param_groups:
        raise RuntimeError("no trainable parameters found")

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    for epoch in range(total_epochs):
        log(f"epoch {epoch + 1}/{total_epochs}")

        model.train()
        if FREEZE_BACKBONE and UNFREEZE_LAST_N_BLOCKS == 0:
            model.backbone.eval()
        train_loss = 0.0

        for img, clinical_batch, labels in tqdm(train_loader, desc="Training"):
            img = img.to(device, non_blocking=True)
            clinical_batch = clinical_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if random.random() < 0.4:
                    mixed_img, labels_a, labels_b, lam = mixup_data(img, labels)
                    outputs = model(mixed_img, clinical_batch).view(-1)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(img, clinical_batch).view(-1)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        if epoch % TRAIN_METRIC_EVERY == 0:
            _, _, train_auc = validate(model, train_eval_loader, criterion)
        else:
            train_auc = train_aucs[-1] if train_aucs else float("nan")
        val_loss, _, val_auc = validate(model, val_loader, criterion)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improvement_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            log(f"saved best model auc={best_val_auc:.4f}")
        else:
            no_improvement_epochs += 1

        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"train_loss={train_loss:.6f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.6f} val_auc={val_auc:.4f} lr={current_lr:.8f}"
        )

        if no_improvement_epochs >= patience:
            log("early stopping triggered")
            break

    log("loading best model for test")
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, weights_only=False, map_location="cpu")
            model.load_state_dict(checkpoint)
            log(f"loaded best model: {best_model_path}")
        except Exception as e:
            log(f"failed to load best model, using current weights: {e}")
    model.eval()

    if os.path.exists(best_model_path):
        archived_model_path = os.path.join(OUTPUT_DIR, f"best_evax_tiny_model_auc{best_val_auc:.4f}_{run_stamp}.pth")
        shutil.copy2(best_model_path, archived_model_path)
        log(f"archived best model: {archived_model_path}")

    def collect_probs(data_loader, desc):
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for img, clinical_batch, labels in tqdm(data_loader, desc=desc):
                img = img.to(device, non_blocking=True)
                clinical_batch = clinical_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(img, clinical_batch).view(-1)
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
            "auc": auc,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
        }

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

    all_val_labels, all_val_probs = collect_probs(val_loader, "Selecting Threshold")
    best_threshold, _, best_youden = find_optimal_threshold(all_val_labels, all_val_probs)
    all_test_labels, all_test_probs = collect_probs(test_loader, "Testing")
    best_metrics = compute_metrics(all_test_labels, all_test_probs, best_threshold)
    fixed_05_metrics = compute_metrics(all_test_labels, all_test_probs, 0.5)

    log(f"best_val_auc={best_val_auc:.4f}")
    log(f"best_threshold={best_threshold:.4f}")
    log(f"val_youden={best_youden:.4f}")
    log(f"test_auc={best_metrics['auc']:.4f}")
    log(f"test_accuracy={best_metrics['accuracy']:.4f}")
    log(f"test_sensitivity={best_metrics['sensitivity']:.4f}")
    log(f"test_specificity={best_metrics['specificity']:.4f}")
    log(f"test_f1={best_metrics['f1']:.4f}")
    log(
        "test@0.50 "
        f"auc={fixed_05_metrics['auc']:.4f} "
        f"accuracy={fixed_05_metrics['accuracy']:.4f} "
        f"sensitivity={fixed_05_metrics['sensitivity']:.4f} "
        f"specificity={fixed_05_metrics['specificity']:.4f} "
        f"f1={fixed_05_metrics['f1']:.4f}"
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC", color="blue")
    plt.plot(val_aucs, label="Val AUC", color="orange")
    plt.title("Training & Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
