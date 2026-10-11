#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect EVA-X split/label manifests for leakage, imbalance, and label conflicts."
    )
    parser.add_argument("--run-info", type=str, help="Path to run_info_*.txt")
    parser.add_argument("--split-csv", type=str, help="Path to evax_tiny_split_*.csv")
    parser.add_argument("--label-csv", type=str, help="Path to label_manifest_*.csv")
    parser.add_argument(
        "--search-root",
        type=str,
        default=".",
        help="Directory used to auto-discover the latest run_info/split/label files.",
    )
    return parser.parse_args()


def read_run_info(path: Path):
    info = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def discover_latest(pattern: str, root: Path):
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def resolve_inputs(args):
    run_info_path = Path(args.run_info).expanduser() if args.run_info else None
    split_csv = Path(args.split_csv).expanduser() if args.split_csv else None
    label_csv = Path(args.label_csv).expanduser() if args.label_csv else None
    search_root = Path(args.search_root).expanduser()

    if run_info_path is None and (split_csv is None or label_csv is None):
        run_info_path = discover_latest("run_info_*.txt", search_root)

    if run_info_path and run_info_path.exists():
        info = read_run_info(run_info_path)
        if split_csv is None and info.get("split_path"):
            split_csv = Path(info["split_path"])
        if label_csv is None and info.get("label_manifest_path"):
            label_csv = Path(info["label_manifest_path"])

    if split_csv is None:
        split_csv = discover_latest("evax_tiny_split_*.csv", search_root)
    if label_csv is None:
        label_csv = discover_latest("label_manifest_*.csv", search_root)

    return run_info_path, split_csv, label_csv


def read_csv(path: Path):
    return pd.read_csv(path, encoding="utf-8-sig")


def find_column(df: pd.DataFrame, candidates, contains=None, required=True):
    for name in candidates:
        if name in df.columns:
            return name
    if contains:
        for col in df.columns:
            if any(token in str(col) for token in contains):
                return col
    if required:
        raise KeyError(f"Could not find column from candidates={candidates}, contains={contains}")
    return None


def summarize_labels(df: pd.DataFrame, split_col: str, label_col: str):
    print("\n[Label Distribution]")
    overall = df[label_col].value_counts(dropna=False).sort_index()
    print("overall:")
    for label, count in overall.items():
        print(f"  label={label}: {count}")

    split_stats = (
        df.groupby(split_col)[label_col]
        .agg(total="count", positives="sum", negative=lambda s: (s == 0).sum(), positive_rate="mean")
        .reset_index()
    )
    print("by split:")
    for row in split_stats.to_dict("records"):
        print(
            f"  {row[split_col]}: total={row['total']}, "
            f"neg={row['negative']}, pos={row['positives']}, pos_rate={row['positive_rate']:.4f}"
        )


def summarize_cross_split_overlap(df: pd.DataFrame, id_col: str, split_col: str, title: str):
    overlap = (
        df.groupby(id_col)[split_col]
        .nunique()
        .rename("split_count")
        .reset_index()
    )
    overlap = overlap[overlap["split_count"] > 1]
    print(f"\n[{title}]")
    print(f"count={len(overlap)}")
    if not overlap.empty:
        preview = overlap.head(10)
        for row in preview.to_dict("records"):
            print(f"  {id_col}={row[id_col]} split_count={row['split_count']}")
        if len(overlap) > len(preview):
            print(f"  ... {len(overlap) - len(preview)} more")


def summarize_conflicts(df: pd.DataFrame, id_col: str, label_col: str, title: str):
    conflicts = (
        df.groupby(id_col)[label_col]
        .nunique()
        .rename("label_count")
        .reset_index()
    )
    conflicts = conflicts[conflicts["label_count"] > 1]
    print(f"\n[{title}]")
    print(f"count={len(conflicts)}")
    if not conflicts.empty:
        preview = conflicts.head(10)
        for row in preview.to_dict("records"):
            print(f"  {id_col}={row[id_col]} label_count={row['label_count']}")
        if len(conflicts) > len(preview):
            print(f"  ... {len(conflicts) - len(preview)} more")


def main():
    args = parse_args()
    run_info_path, split_csv, label_csv = resolve_inputs(args)

    if split_csv is None or not split_csv.exists():
        print("split csv not found", file=sys.stderr)
        sys.exit(1)
    if label_csv is None or not label_csv.exists():
        print("label manifest csv not found", file=sys.stderr)
        sys.exit(1)

    split_df = read_csv(split_csv)
    label_df = read_csv(label_csv)

    accession_candidates = ["放射编号", "AccessionNumber", "accession", "accessions"]
    seq_candidates = ["序号", "seq", "seqs"]
    filename_candidates = ["filename", "file_name"]
    label_candidates = ["label", "标签"]
    split_candidates = ["split"]

    split_accession_col = find_column(split_df, accession_candidates, contains=["放射", "accession"])
    split_seq_col = find_column(split_df, seq_candidates, contains=["序号", "seq"], required=False)
    split_file_col = find_column(split_df, filename_candidates, contains=["file"], required=False)
    split_col = find_column(split_df, split_candidates)

    label_accession_col = find_column(label_df, accession_candidates, contains=["放射", "accession"])
    label_seq_col = find_column(label_df, seq_candidates, contains=["序号", "seq"], required=False)
    label_file_col = find_column(label_df, filename_candidates, contains=["file"], required=False)
    label_col = find_column(label_df, label_candidates, contains=["label", "标签"])

    merge_keys = []
    for split_name, label_name in (
        (split_accession_col, label_accession_col),
        (split_seq_col, label_seq_col),
        (split_file_col, label_file_col),
    ):
        if split_name and label_name:
            merge_keys.append((split_name, label_name))

    if not merge_keys:
        print("No shared keys found between split csv and label manifest.", file=sys.stderr)
        sys.exit(1)

    left_on = [pair[0] for pair in merge_keys]
    right_on = [pair[1] for pair in merge_keys]
    merged = split_df.merge(
        label_df,
        left_on=left_on,
        right_on=right_on,
        how="inner",
        suffixes=("_split", "_label"),
    )

    print("[Files]")
    if run_info_path:
        print(f"run_info={run_info_path}")
    print(f"split_csv={split_csv}")
    print(f"label_csv={label_csv}")
    print(f"merged_rows={len(merged)}")
    print(f"split_rows={len(split_df)}")
    print(f"label_rows={len(label_df)}")

    summarize_labels(merged, split_col=split_col, label_col=label_col)
    summarize_cross_split_overlap(merged, id_col=split_accession_col, split_col=split_col, title="Accession In Multiple Splits")
    if split_seq_col:
        summarize_cross_split_overlap(merged, id_col=split_seq_col, split_col=split_col, title="Sequence In Multiple Splits")

    summarize_conflicts(label_df, id_col=label_accession_col, label_col=label_col, title="Accession With Conflicting Labels")
    if label_seq_col:
        summarize_conflicts(label_df, id_col=label_seq_col, label_col=label_col, title="Sequence With Conflicting Labels")


if __name__ == "__main__":
    main()
