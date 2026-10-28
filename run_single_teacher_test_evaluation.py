#!/usr/bin/env python3
"""Evaluate the four saved seed42 students on the frozen test split, once.

Upload this file beside foundation_sft/, or pass --project to that project.
No teacher, pretrained weight file, training, or threshold selection is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

FORMAT = "single_teacher_overfit_test_v1"
METRICS = ("roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity")
ARTIFACTS = ("metrics.json", "predictions.csv", "patient_predictions.csv",
             "roc_curve.png", "precision_recall_curve.png", "calibration_curve.png")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _same(name, actual, expected):
    def normalized(value):
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value
    if normalized(actual) != normalized(expected):
        raise ValueError("Provenance mismatch: " + name)


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_payload(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _preflight(study, checkpoints=True):
    """Check every group and split before constructing any test image dataset."""
    import pandas as pd
    from foundation_sft.single_teacher_overfit import FORMAT as SOURCE_FORMAT, protocol_hash, _scientific

    study = Path(study).expanduser().resolve()
    state = _read(study / "study.json")
    _same("study format", state.get("format_version"), SOURCE_FORMAT)
    _same("completed study", state.get("status"), "completed")
    plan = state["plan"]
    _same("frozen protocol", state.get("protocol_sha256"), protocol_hash(plan))
    jobs = {job["id"]: job for job in plan["jobs"]}
    expected_jobs = {group + "_seed42" for group in "ABCD"}
    _same("planned jobs", set(jobs), expected_jobs)
    _same("completed jobs", set(state["jobs"]), expected_jobs)
    source = plan["source"]
    frozen_splits = {sha for name, sha in source.get("files", {}).items()
                     if Path(name).name == "split.csv"
                     and Path(name).is_relative_to(Path(plan["source_suite"]))}
    entries, reference = [], None
    for group in "ABCD":
        job, record = jobs[group + "_seed42"], state["jobs"][group + "_seed42"]
        if record.get("status") not in {"completed", "resumed"}:
            raise ValueError("Incomplete group " + group)
        for name in ("id", "group", "seed", "condition"):
            _same("saved job " + name, record["job"].get(name), job.get(name))
        _same("saved job recipe", _scientific(record["job"]["config"]), _scientific(job["config"]))
        _same("group", job.get("group"), group)
        _same("seed", job.get("seed"), 42)
        receipt = record["receipt"]
        run = Path(receipt["run"]).expanduser()
        run = (run if run.is_absolute() else study / run).resolve()
        config, result = _read(run / "config.json"), _read(run / "result.json")
        metadata, geometry = _read(run / "model_metadata.json"), _read(run / "shared_view.json")
        _same("saved recipe", _scientific(config), _scientific(job["config"]))
        _same("student mode", config.get("student", {}).get("mode", "full_finetune"), "full_finetune")
        _same("ImageNet initialization", metadata.get("initialization", {}).get("pretraining"), "ImageNet-1k")
        for name, value in (("model_metadata", metadata), ("shared_view", geometry)):
            _same(name, value, source[name])
        for name, expected in (("status", "completed"), ("seed", 42),
                ("condition", job["condition"]), ("evaluation_phase", "validation_only"),
                ("test_evaluated", False), ("cohort_hash", source["cohort_hash"]),
                ("initial_state_sha256", source["initial_state_sha256"])):
            _same("receipt " + name, receipt.get(name), expected)
            _same("result " + name, result.get(name), expected)
        if "test_patient" in result or (run / "test").exists():
            raise ValueError("Source run already contains a test evaluation: " + str(run))
        threshold = receipt["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid saved validation threshold")
        for name in ("threshold", "best_epoch", "best_val_patient_roc_auc", "diagnostics"):
            _same("result " + name, result.get(name), receipt.get(name))
        files = {name: _sha(run / name) for name in ("best.pt", "split.csv", "config.json",
                 "result.json", "model_metadata.json", "shared_view.json", "split_metadata.json")}
        _same("best checkpoint SHA256", files["best.pt"], receipt["checkpoint_sha256"])
        if receipt.get("history_sha256"):
            _same("history SHA256", _sha(run / "history.json"), receipt["history_sha256"])
        if frozen_splits and files["split.csv"] not in frozen_splits:
            raise ValueError("Saved split differs from the frozen source split")
        _same("split cohort", _read(run / "split_metadata.json").get("_manifest_hash"), source["cohort_hash"])
        split = pd.read_csv(run / "split.csv", dtype={"sample_id": str, "patient_id": str}, keep_default_na=False)
        required = {"sample_id", "patient_id", "label", "split"}
        if not required <= set(split) or split.empty or split["sample_id"].duplicated().any():
            raise ValueError("Missing split columns, empty split, or duplicate sample IDs")
        if any(split[name].str.strip().eq("").any() for name in ("sample_id", "patient_id")):
            raise ValueError("Empty sample or patient identifier")
        if set(split["split"]) != {"train", "val", "test"} or not split["label"].isin([0, 1]).all():
            raise ValueError("Invalid frozen split assignments or labels")
        if (split.groupby("patient_id")["split"].nunique() > 1).any():
            raise ValueError("Patient overlap across frozen splits")
        if (split.groupby("patient_id")["label"].nunique() > 1).any():
            raise ValueError("Conflicting labels for a patient")
        normalized = split.sort_values("sample_id").reindex(sorted(split.columns), axis=1).to_dict("records")
        if reference is None:
            reference = normalized
        _same("identical A-D frozen split rows", normalized, reference)
        test = split.loc[split["split"] == "test"].copy()
        if set(test["label"]) != {0, 1}:
            raise ValueError("The frozen test cohort must contain both label classes")
        manifest = test.sort_values("sample_id")[["sample_id", "patient_id", "label"]].to_dict("records")
        provenance = {"protocol_sha256": state["protocol_sha256"],
            "group": group, "seed": 42, "run": str(run), "threshold": threshold,
            "cohort_hash": source["cohort_hash"], "test_manifest_sha256": _digest(manifest), "files": files}
        if checkpoints:
            payload = _load_payload(run / "best.pt")
            _same("checkpoint format", payload.get("format_version"), "single_teacher_kd_v1")
            _same("checkpoint recipe", _scientific(payload["config"]), _scientific(config))
            for name in ("condition", "cohort_hash", "initial_state_sha256", "threshold", "best_epoch",
                         "best_val_patient_roc_auc", "evaluation_phase", "test_evaluated", "diagnostics"):
                _same("checkpoint " + name, payload.get(name), receipt.get(name))
            _same("checkpoint shared view", payload.get("shared_view"), geometry)
            _same("checkpoint model metadata", payload.get("model_metadata"), metadata)
            del payload
        entries.append({"group": group, "run": run, "checkpoint": run / "best.pt", "config": config,
                        "test": test, "threshold": threshold, "provenance": provenance})
    return {"study": study, "output": study / "report" / "test_evaluation_seed42", "entries": entries}


def _resolve_device(device):
    if device != "auto":
        return device
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        for line in result.stdout.splitlines():
            index, used, free, utilization = map(int, line.split(","))
            if used <= 500 and free >= 20000 and utilization <= 5:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible is not None:
                    devices = [value.strip() for value in visible.split(",")]
                    if str(index) not in devices:
                        continue
                    index = devices.index(str(index))
                import torch
                if torch.cuda.is_available() and index < torch.cuda.device_count():
                    return "cuda:" + str(index)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "cpu"


def _predict(entry, device, cache_dir):
    import torch
    from torch.utils.data import DataLoader
    from foundation_sft.distillation import inference_dataset, load_student_checkpoint
    from foundation_sft.train import predict
    _same("checkpoint before inference", _sha(entry["checkpoint"]), entry["provenance"]["files"]["best.pt"])
    student, payload = load_student_checkpoint(entry["checkpoint"], device)
    _same("loaded checkpoint", _sha(entry["checkpoint"]), entry["provenance"]["files"]["best.pt"])
    config = entry["config"]
    loader = DataLoader(inference_dataset(entry["test"], payload,
                        cache_dir if cache_dir is not None else config.get("cache_dir")),
                        batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=config["training"]["num_workers"], pin_memory=torch.device(device).type == "cuda")
    return predict(student, loader, torch.device(device))


def _cached(entry, output):
    directory = output / entry["group"]
    if not (directory / "receipt.json").exists():
        return None
    receipt = _read(directory / "receipt.json")
    _same("evaluation format", receipt.get("format_version"), FORMAT)
    _same("evaluation completed", receipt.get("status"), "completed")
    _same("evaluation provenance", receipt.get("provenance"), entry["provenance"])
    _same("evaluation artifacts", set(receipt["artifacts"]), set(ARTIFACTS))
    for filename, expected in receipt["artifacts"].items():
        _same("cached " + filename, _sha(directory / filename), expected)
    return _read(directory / "metrics.json")


def _table(context, results):
    import pandas as pd
    rows = []
    for entry, result in zip(context["entries"], results):
        row = {"group": entry["group"], "seed": 42, "split": "TEST", "level": "patient",
               "threshold": entry["threshold"], "n_patients": result["patient"]["n_samples"]}
        for name in METRICS:
            row[name] = result["patient"][name]
            interval = result["bootstrap"]["patient"].get(name, {})
            row[name + "_ci_lower"], row[name + "_ci_upper"] = interval.get("lower"), interval.get("upper")
        rows.append(row)
    table = pd.DataFrame(rows)
    comparison = context["output"] / "comparison.csv"
    if comparison.exists():
        _same("cached comparison.csv", comparison.read_text(encoding="utf-8"), table.to_csv(index=False))
    return table


def show_metrics(study):
    """Read verified saved test results without loading Torch or any image."""
    context = _preflight(study, checkpoints=False)
    results = [_cached(entry, context["output"]) for entry in context["entries"]]
    if any(result is None for result in results):
        raise ValueError("No complete A-D test evaluation exists; run without --show-metrics first")
    return _table(context, results)


def evaluate(study, device="auto", cache_dir=None):
    context = _preflight(study, checkpoints=False)
    output = context["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = [_cached(entry, output) for entry in context["entries"]]
        if any(result is None for result in results):
            context = _preflight(study, checkpoints=True)
            device = _resolve_device(device)
            from foundation_sft.metrics import save_evaluation
            for index, entry in enumerate(context["entries"]):
                if results[index] is not None:
                    continue
                directory = output / entry["group"]
                if directory.exists():
                    raise FileExistsError("Incomplete evaluation artifacts require inspection: " + str(directory))
                print(f"[{entry['group']}] TEST inference on {device}; saved threshold={entry['threshold']}", flush=True)
                frame = _predict(entry, device, cache_dir)
                expected = entry["test"][["sample_id", "patient_id", "label"]].sort_values("sample_id").reset_index(drop=True)
                actual = frame[["sample_id", "patient_id", "label"]].sort_values("sample_id").reset_index(drop=True)
                _same("test prediction membership", actual.to_dict("records"), expected.to_dict("records"))
                staging = output / ("." + entry["group"] + "_pending_" + uuid.uuid4().hex)
                results[index] = save_evaluation(frame, staging, threshold=entry["threshold"],
                    bootstrap_samples=entry["config"]["evaluation"]["bootstrap_samples"], seed=42)
                _write(staging / "receipt.json", {"format_version": FORMAT, "status": "completed",
                    "evaluated_at": datetime.now(timezone.utc).isoformat(), "evaluation_phase": "test",
                    "threshold_source": "saved validation threshold", "device": str(device),
                    "provenance": entry["provenance"],
                    "artifacts": {name: _sha(staging / name) for name in ARTIFACTS}})
                staging.rename(directory)
        table = _table(context, results)
        if not (output / "comparison.csv").exists():
            temporary = output / (".comparison_" + uuid.uuid4().hex + ".csv")
            try:
                table.to_csv(temporary, index=False, mode="x")
                temporary.replace(output / "comparison.csv")
            finally:
                temporary.unlink(missing_ok=True)
        return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="Completed A-D diagnostic study directory")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--device", default="auto", help="auto selects an idle GPU with >=20 GB free, else CPU")
    parser.add_argument("--cache-dir", default=None, help="Optional pixel cache location; preprocessing stays fixed")
    parser.add_argument("--show-metrics", action="store_true", help="Read verified cached test results only")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve()
    sys.path.insert(0, str(project))
    study = Path(args.study).expanduser()
    study = (study if study.is_absolute() else project / study).resolve()
    table = show_metrics(study) if args.show_metrics else evaluate(study, args.device, args.cache_dir)
    print("\nTEST patient results (95% patient-bootstrap confidence intervals):")
    display = table[["group", "n_patients", "threshold"]].copy()
    def number(value):
        return "undefined" if value is None or not math.isfinite(value) else f"{value:.4f}"
    for name in METRICS:
        display[name] = [f"{number(row[name])} [{number(row[name + '_ci_lower'])}, "
                         f"{number(row[name + '_ci_upper'])}]" for _, row in table.iterrows()]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nArtifacts: " + str(study / "report" / "test_evaluation_seed42"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
