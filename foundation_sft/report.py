"""Compare explicitly listed, completed experiments on the same held-out patients.

No directory discovery or test-score selection is performed. The report reads
small saved artifacts only and keeps preprocessing and training protocols apart
when averaging repeated seeds.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import binary_metrics


RATE_METRICS = (
    "roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity",
    "ppv", "npv", "f1", "accuracy", "brier_score",
)
COUNT_METRICS = ("n_samples", "n_positive", "n_negative", "tp", "fp", "tn", "fn")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _protocol(config, metadata):
    # Including training settings avoids averaging different hyperparameters as
    # if they were simple repetitions. Device, run path and seed are not protocol.
    payload = {
        "training": config.get("training", {}),
        "preprocessing": metadata.get("preprocessing", {}),
        "preprocessing_reference": config.get("preprocessing_reference", {}),
        "pretrained": metadata.get("pretrained", {}),
        "rank": metadata.get("rank"), "alpha": metadata.get("alpha"),
        "lora_targets": metadata.get("lora_targets", []),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode()).hexdigest(), payload


def _load_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    config = _read_json(run_dir / "config.json")
    environment = _read_json(run_dir / "environment.json")
    metadata = _read_json(run_dir / "model_metadata.json")
    selection = _read_json(run_dir / "selection.json")
    metrics = _read_json(run_dir / "test" / "metrics.json")
    history = _read_json(run_dir / "history.json")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Completed training history is missing: {run_dir}")
    for name in ("model", "mode"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Run config has no valid {name}: {run_dir}")
    if config["mode"] not in {"linear_probe", "lora"}:
        raise ValueError(f"Unsupported experiment mode: {config['mode']}")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Run config has no integer seed: {run_dir}")
    if metadata.get("mode") != config["mode"]:
        raise ValueError(f"Saved training mode disagrees with config: {run_dir}")
    cohort_hash = environment.get("cohort_hash")
    if not isinstance(cohort_hash, str) or not cohort_hash:
        raise ValueError(f"Run has no cohort_hash: {run_dir}")
    if selection.get("source") != "validation patient predictions":
        raise ValueError(f"Threshold must be selected on validation patients: {run_dir}")
    threshold = selection.get("threshold")
    patients = pd.read_csv(
        run_dir / "test" / "patient_predictions.csv",
        dtype={"patient_id": str}, keep_default_na=False,
    )
    required = {"patient_id", "label", "probability"}
    if not required.issubset(patients):
        raise ValueError(f"Patient prediction columns are missing: {run_dir}")
    if patients.patient_id.str.strip().eq("").any() or patients.patient_id.duplicated().any():
        raise ValueError(f"Test patient IDs must be nonempty and unique: {run_dir}")
    actual = binary_metrics(patients.label.to_numpy(), patients.probability.to_numpy(), threshold)
    saved = metrics.get("patient", {})
    for name in (*COUNT_METRICS, *RATE_METRICS, "threshold"):
        value = saved.get(name)
        expected = actual[name]
        if ((value is None) != (expected is None)
                or (expected is not None and not np.isclose(value, expected, rtol=1e-8, atol=1e-10))):
            raise ValueError(f"Saved patient {name} disagrees with predictions/validation threshold: {run_dir}")
    if metrics.get("patient_aggregation") != "mean image probability":
        raise ValueError(f"Unsupported patient aggregation protocol: {run_dir}")
    if "n_images" in patients:
        values = pd.to_numeric(patients.n_images, errors="coerce").to_numpy()
        if not np.all(np.isfinite(values)) or np.any(values < 1) or np.any(values != values.astype(int)):
            raise ValueError(f"Invalid patient image counts: {run_dir}")
    protocol_id, protocol = _protocol(config, metadata)
    normalization = config.get("preprocessing_reference", {}).get("status", "model_adapter_default")
    row = {
        "model": config["model"], "mode": config["mode"], "seed": seed,
        "protocol_id": protocol_id, "normalization_status": normalization,
        "run_dir": str(run_dir), "cohort_hash": cohort_hash,
        "threshold": threshold,
        "best_val_patient_roc_auc": selection.get("best_val_patient_roc_auc"),
        "epochs_completed": len(history),
        "total_parameters": metadata.get("total_parameters"),
        "trainable_parameters": metadata.get("trainable_parameters"),
        "image_size": metadata.get("preprocessing", {}).get("image_size"),
        "mean": json.dumps(metadata.get("preprocessing", {}).get("mean")),
        "std": json.dumps(metadata.get("preprocessing", {}).get("std")),
        "bootstrap_samples": metrics.get("bootstrap", {}).get("samples", 0),
        "bootstrap_confidence_level": metrics.get("bootstrap", {}).get("confidence_level"),
        **{name: saved[name] for name in (*COUNT_METRICS, *RATE_METRICS)},
    }
    intervals = metrics.get("bootstrap", {}).get("patient", {})
    for name in RATE_METRICS:
        interval = intervals.get(name, {})
        low, high = interval.get("lower"), interval.get("upper")
        if ((low is None) != (high is None)
                or (low is not None and (not np.isfinite([low, high]).all() or low > high))):
            raise ValueError(f"Invalid patient confidence interval for {name}: {run_dir}")
        if low is not None and metrics.get("bootstrap", {}).get("confidence_level") != 0.95:
            raise ValueError(f"The comparison expects saved 95% bootstrap intervals: {run_dir}")
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
        row[f"{name}_ci_valid_samples"] = interval.get("valid_samples")
    return {"row": row, "patients": patients, "protocol": protocol}


def _validate_cohort(runs):
    if not runs:
        return
    first = runs[0]
    reference = first["patients"].set_index("patient_id").sort_index()
    seen = set()
    for run in runs:
        row = run["row"]
        key = row["model"], row["mode"], row["seed"]
        if key in seen:
            raise ValueError(f"Duplicate model/mode/seed: {key}; list one predeclared run per experiment")
        seen.add(key)
        if row["cohort_hash"] != first["row"]["cohort_hash"]:
            raise ValueError("Cannot compare runs with different cohort_hash values")
        patients = run["patients"].set_index("patient_id").sort_index()
        if not patients.index.equals(reference.index):
            raise ValueError("Cannot compare runs evaluated on different test patient IDs")
        if not np.array_equal(patients.label.to_numpy(), reference.label.to_numpy()):
            raise ValueError("Cannot compare runs with different test patient labels")
        if "n_images" in reference and "n_images" in patients:
            if not np.array_equal(patients.n_images.to_numpy(), reference.n_images.to_numpy()):
                raise ValueError("Cannot compare runs with different test image counts per patient")


def _summary(frame):
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=["model", "mode", "protocol_id", "normalization_status", "n_seeds", "seeds"])
    for (model, mode, protocol), group in frame.groupby(["model", "mode", "protocol_id"], sort=False):
        row = {
            "model": model, "mode": mode, "protocol_id": protocol,
            "normalization_status": group.iloc[0].normalization_status,
            "n_seeds": len(group), "seeds": ",".join(map(str, group.seed)),
        }
        for name in RATE_METRICS:
            values = pd.to_numeric(group[name], errors="coerce")
            row[f"{name}_mean"] = values.mean()
            row[f"{name}_std"] = values.std(ddof=1)
            row[f"{name}_valid_seeds"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _plots(runs, output_dir):
    if not runs:
        return []
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    outputs = []
    modes = dict.fromkeys(run["row"]["mode"] for run in runs)
    for mode in modes:
        # The training CLI permits these two mode strings; validation also makes
        # filenames safe when handling manually assembled artifact directories.
        if mode not in {"linear_probe", "lora"}:
            raise ValueError(f"Unsupported experiment mode: {mode}")
        selected = [run for run in runs if run["row"]["mode"] == mode]
        for kind in ("roc", "pr"):
            figure, axis = plt.subplots(figsize=(8, 6))
            try:
                drawn = False
                for run in selected:
                    row, patients = run["row"], run["patients"]
                    if patients.label.nunique() < 2:
                        continue
                    if kind == "roc":
                        x, y, _ = roc_curve(patients.label, patients.probability)
                        value = row["roc_auc"]
                    else:
                        y, x, _ = precision_recall_curve(patients.label, patients.probability)
                        value = row["pr_auc"]
                    research = " [research normalization]" if row["normalization_status"] == "research_choice_not_author_confirmed" else ""
                    label = f"{row['model']}{research}, seed={row['seed']}, protocol={row['protocol_id'][:8]}, AUC={value:.3f}"
                    axis.plot(x, y, label=label)
                    drawn = True
                if kind == "roc":
                    axis.plot([0, 1], [0, 1], "--", color="gray")
                    axis.set(xlabel="False positive rate", ylabel="Sensitivity", title=f"Patient ROC: {mode}")
                else:
                    prevalence = selected[0]["patients"].label.mean()
                    axis.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence={prevalence:.3f}")
                    axis.set(xlabel="Recall", ylabel="Precision", title=f"Patient precision-recall: {mode}")
                if drawn:
                    axis.legend(loc="best", fontsize=7)
                else:
                    axis.text(0.5, 0.5, "Undefined: test patients have one label class", ha="center", va="center", transform=axis.transAxes)
                axis.set(xlim=(0, 1), ylim=(0, 1))
                axis.grid(alpha=0.2)
                figure.tight_layout()
                filename = f"patient_{kind}_{mode}.png"
                figure.savefig(output_dir / filename, dpi=160)
                outputs.append(filename)
            finally:
                plt.close(figure)
    return outputs


def _markdown(frame, summary, exclusions, plots):
    lines = [
        "# 基础模型监督实验比较", "",
        f"纳入明确列举的 {len(frame)} 次已完成训练，排除 {len(exclusions)} 个未完成任务。",
        "仅比较已保存的测试结果；不扫描结果目录、不按测试分数挑选种子或模型。",
        "模型参数按验证集患者 ROC-AUC 选择，分类阈值由验证集患者预测确定，测试集使用固定阈值。", "",
        "所有纳入结果已核对同一 cohort_hash、测试患者 ID 和标签；逐次结果记录权重身份、预处理协议和研究归一化状态。",
        "同一模型、模式、协议才计算跨种子均值和样本标准差；单一种子的标准差留空。",
        "逐次患者 95% CI 是固定阈值的患者 bootstrap；跨种子标准差表示重复训练波动，两者含义不同。",
        "PR-AUC 采用梯形积分，Average Precision 单独列出。缺失 CI 或未定义指标留空。", "",
        "| 模型 | 模式 | 协议 | 归一化依据状态 | 种子数 | 患者 ROC-AUC 均值 | 种子间标准差 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        def display(value):
            return "" if pd.isna(value) else f"{value:.4f}"
        lines.append(f"| {row.model} | {row['mode']} | {row.protocol_id[:8]} | {row.normalization_status} | {row.n_seeds} | {display(row.get('roc_auc_mean'))} | {display(row.get('roc_auc_std'))} |")
    if frame.empty:
        lines.extend(["", "尚无完成训练的任务，因此本报告没有测试指标或比较曲线。"])
    else:
        reference = frame.iloc[0]
        lines.extend(["", f"测试患者 {int(reference.n_samples)} 人，其中阳性 {int(reference.n_positive)} 人、阴性 {int(reference.n_negative)} 人。", f"队列指纹：`{reference.cohort_hash}`。"])
    if not frame.empty and frame.normalization_status.eq("research_choice_not_author_confirmed").any():
        lines.extend(["", "RadJEPA 等标为 research_choice_not_author_confirmed 的归一化为本研究选择，尚未核实为作者预训练规范；该状态已写入表格和曲线，不应描述为作者确认的设置。"])
    lines.extend(["", "文件：", "", "- `run_metrics.csv`：逐次测试患者指标、固定阈值及 bootstrap CI。", "- `model_mode_summary.csv`：每个模型/模式/协议跨种子的均值、样本标准差及有效种子数。", "- `protocols.json`：每个协议的预处理、权重身份和训练设置。", "- `excluded_jobs.csv`：未完成任务及状态，保留失败原因。"])
    lines.extend(f"- `{filename}`：同模式下每个预先指定种子的患者曲线。" for filename in plots)
    if exclusions:
        lines.extend(["", "排除任务：", ""])
        for job in exclusions:
            lines.append(f"- {job.get('id', '')}：{job.get('status', '')}；{job.get('error', job.get('reason', '未完成训练'))}")
    lines.extend(["", "此比较不执行测试集调参、最佳种子筛选或显著性检验。", ""])
    return "\n".join(lines)


def _write_report(runs, output_dir, exclusions=()):
    _validate_cohort(runs)
    output_dir = Path(output_dir).expanduser().resolve()
    # A fresh directory prevents an old success report surviving a partial rerun.
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame([run["row"] for run in runs])
    if frame.empty:
        frame = pd.DataFrame(columns=["model", "mode", "seed", "protocol_id", "normalization_status", "run_dir", "cohort_hash"])
    summary = _summary(frame)
    frame.to_csv(output_dir / "run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "model_mode_summary.csv", index=False, encoding="utf-8-sig")
    excluded_frame = pd.DataFrame(exclusions)
    if excluded_frame.empty:
        excluded_frame = pd.DataFrame(columns=["id", "model", "mode", "seed", "status", "run_dir", "error"])
    excluded_frame.to_csv(output_dir / "excluded_jobs.csv", index=False, encoding="utf-8-sig")
    protocols = {run["row"]["protocol_id"]: run["protocol"] for run in runs}
    (output_dir / "protocols.json").write_text(json.dumps(protocols, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plots = _plots(runs, output_dir)
    (output_dir / "README.md").write_text(_markdown(frame, summary, exclusions, plots), encoding="utf-8")
    return output_dir


def summarize_runs(run_dirs, output_dir):
    """Report explicitly enumerated completed run directories; never discover runs.

    Validation finishes before a new output directory is created. An invalid or
    incomplete listed run is an error, rather than silently removed from analysis.
    """
    run_dirs = list(run_dirs)
    if not run_dirs:
        raise ValueError("Explicit run list must not be empty")
    return _write_report([_load_run(path) for path in run_dirs], output_dir)


def summarize_suite(suite_dir, output_dir=None):
    """Report manifest jobs marked completed and explicitly list other statuses."""
    suite_dir = Path(suite_dir).expanduser().resolve()
    manifest = _read_json(suite_dir / "suite.json")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("suite.json must contain a jobs list")
    runs, exclusions, ids = [], [], set()
    for job in jobs:
        if not isinstance(job, dict) or not job.get("id") or job["id"] in ids:
            raise ValueError("Suite jobs must have unique, nonempty IDs")
        ids.add(job["id"])
        if job.get("status") != "completed":
            exclusions.append(job)
            continue
        path = job.get("run_dir")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Completed suite job has no run_dir: {job['id']}")
        path = Path(path)
        run = _load_run(path if path.is_absolute() else suite_dir / path)
        for name in ("model", "mode", "seed"):
            if job.get(name) != run["row"][name]:
                raise ValueError(f"Suite job {job['id']} has inconsistent {name}")
        run["row"]["job_id"] = job["id"]
        runs.append(run)
    return _write_report(runs, output_dir or suite_dir / "report", exclusions)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare explicitly listed completed supervised experiments.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite", help="Suite directory containing suite.json")
    source.add_argument("--runs", nargs="+", help="Explicit completed training run directories")
    parser.add_argument("--output", help="New report directory (default: <suite>/report)")
    args = parser.parse_args(argv)
    if args.suite:
        output = summarize_suite(args.suite, args.output)
    else:
        if not args.output:
            parser.error("--runs requires --output")
        output = summarize_runs(args.runs, args.output)
    print(f"Comparison report: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
