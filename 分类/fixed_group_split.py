
# -*- coding: utf-8 -*-
"""Create once and reuse a strict accession-level 70:10:20 split manifest."""

import os
import tempfile

import numpy as np
import pandas as pd


ACCESSION_COLUMN = "放射编号"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"
SPLIT_VERSION = "group_stratified_70_10_20_v1"
VALID_SPLITS = ("train", "val", "test")
MISSING_ACCESSION_VALUES = {"", "nan", "none", "null", "<na>"}


def _preview(values, limit=5):
    return ", ".join(sorted(str(value) for value in values)[:limit])


def _prepare_groups(metadata, labels):
    if metadata is None or ACCESSION_COLUMN not in metadata.columns:
        raise RuntimeError(
            "固定分组划分需要完整的放射编号元数据；请删除旧缓存并重新生成包含 accessions 的缓存。"
        )

    labels = np.asarray(labels).reshape(-1)
    if len(metadata) != len(labels):
        raise RuntimeError(
            f"元数据与标签数量不一致: metadata={len(metadata)}, labels={len(labels)}"
        )

    raw_accessions = metadata[ACCESSION_COLUMN]
    accessions = raw_accessions.astype(str).str.strip()
    missing_mask = raw_accessions.isna() | accessions.str.lower().isin(MISSING_ACCESSION_VALUES)
    if missing_mask.any():
        examples = metadata.index[missing_mask].tolist()[:5]
        raise RuntimeError(
            f"发现 {int(missing_mask.sum())} 个缺失放射编号（示例行: {examples}），"
            "无法安全地进行患者级固定划分。"
        )

    numeric_labels = pd.to_numeric(pd.Series(labels), errors="coerce")
    invalid_label_mask = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    if invalid_label_mask.any():
        invalid_values = pd.Series(labels)[invalid_label_mask].astype(str).unique().tolist()
        raise RuntimeError(
            f"固定二分类划分只接受标签 0/1，发现无效标签: {_preview(invalid_values)}"
        )

    sample_table = pd.DataFrame(
        {
            ACCESSION_COLUMN: accessions.to_numpy(),
            LABEL_COLUMN: numeric_labels.astype(int).to_numpy(),
        }
    )
    label_counts = sample_table.groupby(ACCESSION_COLUMN)[LABEL_COLUMN].nunique()
    conflicting_groups = label_counts[label_counts > 1].index.tolist()
    if conflicting_groups:
        raise RuntimeError(
            f"有 {len(conflicting_groups)} 个放射编号对应多个标签"
            f"（示例: {_preview(conflicting_groups)}），请先修正标签冲突。"
        )

    group_table = (
        sample_table.groupby(ACCESSION_COLUMN, as_index=False)[LABEL_COLUMN]
        .first()
        .sort_values(ACCESSION_COLUMN, kind="stable")
        .reset_index(drop=True)
    )
    if group_table[LABEL_COLUMN].nunique() < 2:
        raise RuntimeError("当前放射编号组只有一个类别，无法进行分层划分。")

    return accessions.to_numpy(), numeric_labels.astype(int).to_numpy(), group_table


def _build_manifest(group_table, seed):
    from sklearn.model_selection import train_test_split

    groups = group_table[ACCESSION_COLUMN].to_numpy()
    group_labels = group_table[LABEL_COLUMN].to_numpy()

    try:
        train_groups, temp_groups, _, temp_labels = train_test_split(
            groups,
            group_labels,
            test_size=0.30,
            random_state=seed,
            stratify=group_labels,
        )
        val_groups, test_groups, _, _ = train_test_split(
            temp_groups,
            temp_labels,
            test_size=2 / 3,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError as exc:
        class_counts = group_table[LABEL_COLUMN].value_counts().sort_index().to_dict()
        raise RuntimeError(
            f"无法按放射编号分层生成 70:10:20 固定清单；组级类别数={class_counts}: {exc}"
        ) from exc

    split_by_group = {
        **{str(group): "train" for group in train_groups},
        **{str(group): "val" for group in val_groups},
        **{str(group): "test" for group in test_groups},
    }
    manifest = group_table.copy()
    manifest[SPLIT_COLUMN] = manifest[ACCESSION_COLUMN].map(split_by_group)
    manifest["seed"] = int(seed)
    manifest["split_version"] = SPLIT_VERSION

    if manifest[SPLIT_COLUMN].isna().any():
        raise RuntimeError("生成固定清单时有放射编号未被分配。")
    return manifest


def _write_manifest_once(manifest, manifest_path):
    parent_dir = os.path.dirname(os.path.abspath(manifest_path))
    os.makedirs(parent_dir, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(manifest_path) + ".",
        suffix=".tmp",
        dir=parent_dir,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8-sig", newline="") as file:
            manifest.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temp_path, manifest_path)
            return True
        except FileExistsError:
            return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _load_manifest(manifest_path, seed):
    try:
        manifest = pd.read_csv(
            manifest_path,
            dtype={ACCESSION_COLUMN: str, SPLIT_COLUMN: str},
            encoding="utf-8-sig",
        )
    except Exception as exc:
        raise RuntimeError(f"无法读取固定划分清单 {manifest_path}: {exc}") from exc

    required_columns = {
        ACCESSION_COLUMN,
        LABEL_COLUMN,
        SPLIT_COLUMN,
        "seed",
        "split_version",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise RuntimeError(
            f"固定划分清单缺少字段: {sorted(missing_columns)}；文件: {manifest_path}"
        )

    manifest[ACCESSION_COLUMN] = manifest[ACCESSION_COLUMN].astype(str).str.strip()
    invalid_accession_mask = manifest[ACCESSION_COLUMN].str.lower().isin(MISSING_ACCESSION_VALUES)
    if invalid_accession_mask.any():
        raise RuntimeError(f"固定划分清单包含缺失放射编号: {manifest_path}")
    if manifest[ACCESSION_COLUMN].duplicated().any():
        duplicates = manifest.loc[
            manifest[ACCESSION_COLUMN].duplicated(keep=False), ACCESSION_COLUMN
        ].unique().tolist()
        raise RuntimeError(
            f"固定划分清单包含重复放射编号: {_preview(duplicates)}"
        )

    manifest[LABEL_COLUMN] = pd.to_numeric(manifest[LABEL_COLUMN], errors="coerce")
    if manifest[LABEL_COLUMN].isna().any() or not manifest[LABEL_COLUMN].isin([0, 1]).all():
        raise RuntimeError(f"固定划分清单包含非 0/1 标签: {manifest_path}")
    manifest[LABEL_COLUMN] = manifest[LABEL_COLUMN].astype(int)

    invalid_splits = set(manifest[SPLIT_COLUMN].dropna().unique()).difference(VALID_SPLITS)
    if invalid_splits or manifest[SPLIT_COLUMN].isna().any():
        raise RuntimeError(
            f"固定划分清单包含无效 split: {sorted(invalid_splits)}"
        )
    missing_splits = set(VALID_SPLITS).difference(manifest[SPLIT_COLUMN].unique())
    if missing_splits:
        raise RuntimeError(
            f"固定划分清单缺少 split: {sorted(missing_splits)}"
        )

    manifest_seed_values = pd.to_numeric(manifest["seed"], errors="coerce")
    if manifest_seed_values.isna().any() or not (manifest_seed_values == int(seed)).all():
        manifest_seeds = sorted(manifest_seed_values.dropna().unique().tolist())
        raise RuntimeError(
            f"固定划分清单 seed={manifest_seeds}，当前脚本 seed={seed}。"
        )
    versions = set(manifest["split_version"].astype(str))
    if versions != {SPLIT_VERSION}:
        raise RuntimeError(
            f"固定划分清单版本={sorted(versions)}，要求版本={SPLIT_VERSION}。"
        )

    return manifest


def load_or_create_fixed_group_split(metadata, labels, manifest_path, seed, log_fn=None):
    """Return sample indices mapped from one immutable accession-level manifest."""

    accessions, sample_labels, group_table = _prepare_groups(metadata, labels)
    created = False
    if not os.path.exists(manifest_path):
        candidate_manifest = _build_manifest(group_table, seed)
        created = _write_manifest_once(candidate_manifest, manifest_path)

    manifest = _load_manifest(manifest_path, seed)

    current_groups = set(group_table[ACCESSION_COLUMN])
    fixed_groups = set(manifest[ACCESSION_COLUMN])
    current_only = current_groups.difference(fixed_groups)
    fixed_only = fixed_groups.difference(current_groups)
    if current_only or fixed_only:
        raise RuntimeError(
            "当前数据与固定划分清单的放射编号集合不一致；"
            f"当前新增={len(current_only)}（示例: {_preview(current_only)}），"
            f"当前缺少={len(fixed_only)}（示例: {_preview(fixed_only)}）。"
            "若确实更换了数据集，请人工归档旧清单后再生成新清单，程序不会自动覆盖。"
        )

    current_labels = group_table.set_index(ACCESSION_COLUMN)[LABEL_COLUMN]
    fixed_labels = manifest.set_index(ACCESSION_COLUMN)[LABEL_COLUMN]
    mismatched_groups = [
        group
        for group in current_groups
        if int(current_labels.loc[group]) != int(fixed_labels.loc[group])
    ]
    if mismatched_groups:
        raise RuntimeError(
            f"有 {len(mismatched_groups)} 个放射编号的标签与固定清单不一致"
            f"（示例: {_preview(mismatched_groups)}）。"
        )

    split_by_group = manifest.set_index(ACCESSION_COLUMN)[SPLIT_COLUMN]
    sample_splits = pd.Series(accessions).map(split_by_group).to_numpy()
    if pd.isna(sample_splits).any():
        raise RuntimeError("当前样本中有放射编号未在固定清单中找到。")

    indices = {
        split: np.flatnonzero(sample_splits == split)
        for split in VALID_SPLITS
    }
    if any(len(indices[split]) == 0 for split in VALID_SPLITS):
        raise RuntimeError("固定划分映射后至少有一个数据集为空。")
    if sum(len(indices[split]) for split in VALID_SPLITS) != len(sample_labels):
        raise RuntimeError("固定划分映射未完整覆盖全部样本。")

    if log_fn is not None:
        action = "已创建" if created else "已加载"
        group_counts = manifest[SPLIT_COLUMN].value_counts()
        sample_counts = {split: len(indices[split]) for split in VALID_SPLITS}
        positive_rates = {
            split: float(sample_labels[indices[split]].mean())
            for split in VALID_SPLITS
        }
        log_fn(f"固定放射编号划分清单{action}: {manifest_path}")
        log_fn(
            "放射编号组数 train/val/test: "
            f"{int(group_counts.get('train', 0))}/"
            f"{int(group_counts.get('val', 0))}/"
            f"{int(group_counts.get('test', 0))}"
        )
        log_fn(
            "样本数 train/val/test: "
            f"{sample_counts['train']}/{sample_counts['val']}/{sample_counts['test']} | "
            "阳性率: "
            f"{positive_rates['train']:.4f}/"
            f"{positive_rates['val']:.4f}/"
            f"{positive_rates['test']:.4f}"
        )

    return indices["train"], indices["val"], indices["test"]
