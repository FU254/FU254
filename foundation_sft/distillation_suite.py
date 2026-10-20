"""Standard-library coordinator for matched student update / KD experiments.

Torch is imported only while validating a checkpoint for reuse. Planning and
reading reports therefore work without the training environment or a GPU.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import uuid


FORMAT = "distillation_abcd_v1"
GROUPS = {
    "A": ("full_finetune", "supervised"),
    "B": ("full_finetune", "distillation"),
    "C": ("lora", "supervised"),
    "D": ("lora", "distillation"),
}
METRICS = ("val_auc", "roc_auc", "pr_auc", "ap", "sensitivity", "specificity")


def _read(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value, project):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project / path).resolve()


def _student_mode(config):
    return config.get("student", {}).get("mode", "full_finetune")


def make_plan(config_path, project, *, groups=None, seeds=None, devices=None,
              max_parallel=None, cache_dir=None):
    """Resolve raw JSON only; do not create a directory or load a tensor."""
    project = Path(project).resolve()
    path = _path(config_path, project)
    settings = _read(path)
    selected = list(groups if groups is not None else settings["groups"])
    if not selected or len(set(selected)) != len(selected) or any(group not in GROUPS for group in selected):
        raise ValueError("groups must be unique members of A B C D")
    selected_seeds = list(seeds if seeds is not None else settings.get("seeds", [42, 43, 44]))
    if (not selected_seeds or len(set(selected_seeds)) != len(selected_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in selected_seeds)):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    chosen_devices = list(devices if devices is not None else settings.get("devices", ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]))
    if not chosen_devices or any(not isinstance(device, str) or not device for device in chosen_devices):
        raise ValueError("devices must contain at least one nonempty device name")
    maximum = max_parallel if max_parallel is not None else settings.get("max_parallel", 4)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_parallel must be a positive integer")
    jobs = []
    for group in selected:
        definition = settings["groups"].get(group)
        if not definition:
            raise ValueError(f"Group {group} is not configured")
        mode, condition = GROUPS[group]
        if (definition.get("student_mode"), definition.get("condition")) != (mode, condition):
            raise ValueError(f"Group {group} must use {mode} / {condition}")
        template = _read(_path(definition["config"], project))
        if _student_mode(template) != mode:
            raise ValueError(f"Group {group} student config does not use {mode}")
        # The controls are fixed before test evaluation; no group-specific LR search.
        training = template["training"]
        encoder_lr = training.get("lora_lr") if mode == "lora" else training.get("backbone_lr")
        if encoder_lr != 5e-5 or training.get("head_lr") != 1e-4 or training.get("projection_lr") != 1e-4:
            raise ValueError("ABCD requires encoder/LoRA LR=5e-5, head/projection LR=1e-4")
        # Default mapping is stable even when only B or D is requested.
        index = list(GROUPS).index(group) if devices is None else selected.index(group)
        device = chosen_devices[index % len(chosen_devices)]
        for seed in selected_seeds:
            resolved = deepcopy(template)
            resolved.update(seed=seed, device=device)
            if cache_dir is not None:
                resolved["cache_dir"] = str(_path(cache_dir, project))
            jobs.append({"id": f"{group}_seed{seed}", "group": group, "seed": seed,
                         "student_mode": mode, "condition": condition, "device": device,
                         "config": resolved})
    # Head, weight decay, loss, selection, and augment settings must be matched.
    # Different unused inherited fields in the full model are kept for legacy reuse.
    normalized = []
    for job in jobs:
        train = deepcopy(job["config"]["training"])
        for key in ("backbone_lr", "lora_lr"):
            train.pop(key, None)
        normalized.append((train, job["config"].get("distillation", {})))
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError("ABCD group training settings and distillation settings must match")
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
            "output_root": str(_path(settings.get("output_root", "runs/distillation_abcd"), project)),
            "max_parallel": maximum, "groups": selected, "seeds": selected_seeds,
            "reuse_suites": [str(_path(value, project)) for value in settings.get("reuse_suites", [])],
            "jobs": jobs}


def protocol_hash(plan):
    # Scheduling concurrency can be adjusted on resume; scientific settings cannot.
    return _hash({key: value for key, value in plan.items() if key not in {"max_parallel", "config_path"}})


def _launch(job, action, project, directory):
    """One child process owns one GPU until it exits; its log is always retained."""
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    config_file, receipt_file = directory / "config.json", directory / "result.json"
    _json(config_file, config)
    command = [sys.executable, "-u", str(project / "run_single_teacher_distillation.py"),
               "--project", str(project), "--config", str(config_file), "--condition", job["condition"],
               "--result-json", str(receipt_file)]
    if action == "check":
        command.append("--check-only")
    log = directory / f"{action}.log"
    print(f"[{action}] {job['id']} {job['device']} log={log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Process exited {completed.returncode}; see {log}")
    if not receipt_file.is_file():
        raise RuntimeError(f"Child did not write {receipt_file}; update the distillation trainer")
    return _read(receipt_file)


def _parallel(jobs, maximum, function, callback):
    """Submit independent devices concurrently, with no oversubscription per GPU."""
    pending, active, busy = list(jobs), {}, set()
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        while pending or active:
            for job in list(pending):
                if len(active) >= maximum:
                    break
                if job["device"] in busy:
                    continue
                pending.remove(job)
                busy.add(job["device"])
                active[executor.submit(function, job)] = job
            if not active:
                raise RuntimeError("No runnable jobs")
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                busy.remove(job["device"])
                try:
                    result = future.result()
                except Exception as error:
                    callback(job, None, error)
                else:
                    callback(job, result, None)


def _check_receipt(value):
    if "check" in value and isinstance(value["check"], dict):
        value = value["check"]
    if value.get("check_passed") is not True:
        raise ValueError("Preflight did not pass")
    required = ("config", "model_metadata", "initial_state_sha256", "cohort_hash", "teacher", "shared_view")
    if any(key not in value for key in required):
        raise ValueError("Preflight lacks provenance; update the distillation trainer")
    return value


def _train_receipt(value, condition):
    if isinstance(value.get("conditions"), dict):
        value = value["conditions"][condition]
    elif value.get("suite"):
        value = _read(Path(value["suite"]) / "suite.json")["conditions"][condition]
    if value.get("status") != "completed" or value.get("condition") != condition or not value.get("run"):
        raise ValueError("Training receipt is incomplete or has the wrong condition")
    return value


def _default_checkpoint_loader(path):
    # This is intentionally lazy: --plan and --show-metrics use no Torch.
    from .distillation import load_student_checkpoint
    model, payload = load_student_checkpoint(path, "cpu")
    del model
    return payload


def _same(name, first, second):
    # torch.save preserves tuples while the adjacent JSON records turn them
    # into lists. Compare metadata by JSON meaning, retaining numeric equality.
    def normalized(value):
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value

    if normalized(first) != normalized(second):
        raise ValueError(f"Reuse rejected: {name} differs")


def _validate_protocol(saved, current, condition):
    """Only scientific settings are compared; output paths and GPU may differ."""
    _same("seed", saved.get("seed"), current.get("seed"))
    _same("student mode", _student_mode(saved), _student_mode(current))
    _same("effective training", saved.get("training"), current.get("training"))
    _same("evaluation", saved.get("evaluation"), current.get("evaluation"))
    _same("student options", saved.get("student", {"mode": "full_finetune"}),
          current.get("student", {"mode": "full_finetune"}))
    if condition == "distillation":
        _same("distillation settings", saved.get("distillation"), current.get("distillation"))


def verify_run(run, check, job, *, checkpoint_loader=None):
    """Check a complete recorded run against this job's current GPU preflight.

    No score is consulted to choose a run. Even the legacy full model must match
    the actual initialized tensors, strict student metadata and frozen cohort.
    """
    run = Path(run).resolve()
    result, saved = _read(run / "result.json"), _read(run / "config.json")
    metadata, teacher = _read(run / "model_metadata.json"), _read(run / "teacher.json")
    geometry = _read(run / "shared_view.json")
    _same("run status", result.get("status"), "completed")
    _same("condition", result.get("condition"), job["condition"])
    _same("seed", result.get("seed"), job["seed"])
    _same("job student mode", _student_mode(saved), job["student_mode"])
    _validate_protocol(saved, check["config"], job["condition"])
    _same("student metadata", metadata, check["model_metadata"])
    _same("initial student state", result.get("initial_state_sha256"), check["initial_state_sha256"])
    _same("cohort hash", result.get("cohort_hash"), check["cohort_hash"])
    _same("teacher checkpoint", teacher.get("checkpoint_sha256"), check["teacher"].get("checkpoint_sha256"))
    if not teacher.get("checkpoint_sha256"):
        raise ValueError("Reuse rejected: missing teacher fingerprint")
    _same("shared field of view", geometry, check["shared_view"])
    _same("frozen teacher", result.get("teacher_unchanged"), True)
    loader = checkpoint_loader or _default_checkpoint_loader
    payload = loader(run / "best.pt")
    _same("checkpoint condition", payload.get("condition"), job["condition"])
    _validate_protocol(payload.get("config", {}), saved, job["condition"])
    _same("checkpoint student metadata", payload.get("model_metadata"), metadata)
    _same("checkpoint initial state", payload.get("initial_state_sha256"), result["initial_state_sha256"])
    _same("checkpoint cohort", payload.get("cohort_hash"), result["cohort_hash"])
    _same("checkpoint teacher", payload.get("teacher"), teacher)
    _same("checkpoint shared view", payload.get("shared_view"), geometry)
    _same("checkpoint threshold", payload.get("threshold"), result.get("threshold"))
    _same("checkpoint validation selection", payload.get("best_val_patient_roc_auc"), result.get("best_val_patient_roc_auc"))
    # Reuse only complete evaluations, not a checkpoint produced before test finished.
    metrics = _read(run / "test" / "metrics.json")
    _same("saved test metrics", metrics.get("patient"), result.get("test_patient"))
    _same("saved bootstrap interval", metrics.get("bootstrap", {}).get("patient", {}).get("roc_auc"),
          result.get("roc_auc_ci"))
    for name in ("validation_predictions.csv", "test/predictions.csv", "test/patient_predictions.csv"):
        if not (run / name).is_file() or (run / name).stat().st_size == 0:
            raise ValueError(f"Reuse rejected: missing {name}")
    result = deepcopy(result)
    result.update(run=str(run), checkpoint_sha256=_file_hash(run / "best.pt"),
                  student_mode=job["student_mode"], model_metadata=metadata)
    if job["condition"] == "supervised":
        result["distillation_trainable_parameters"] = 0
    elif result.get("distillation_trainable_parameters") is None:
        state = payload.get("distillation_state")
        if isinstance(state, dict) and all(callable(getattr(value, "numel", None)) for value in state.values()):
            result["distillation_trainable_parameters"] = sum(value.numel() for value in state.values())
    return result


def _reuse_candidates(suites, condition, seed=None, student_mode=None):
    for suite in suites:
        path = Path(suite)
        if not path.is_dir():
            continue
        # Only explicitly listed suites are scanned, in configured order.
        manifest = path / "suite.json"
        if manifest.is_file():
            try:
                data = _read(manifest)
                if isinstance(data.get("jobs"), dict):
                    # Explicit coordinator batches may contain many seeds and
                    # conditions. Preserve manifest order; never rank test scores.
                    for record in data["jobs"].values():
                        if record.get("status") not in {"completed", "reused", "resumed"}:
                            continue
                        job, receipt = record.get("job", {}), record.get("result", {})
                        if (job.get("condition") != condition
                                or (seed is not None and job.get("seed") != seed)
                                or (student_mode is not None and job.get("student_mode") != student_mode)):
                            continue
                        if isinstance(receipt.get("run"), str) and receipt["run"]:
                            run = Path(receipt["run"])
                            yield run if run.is_absolute() else path / run
                    continue
                receipt = data.get("conditions", {}).get(condition, {})
            except (OSError, ValueError, TypeError, AttributeError):
                # An interrupted historical manifest must not prevent a new
                # experiment. The condition directory is verified independently.
                receipt = {}
            if isinstance(receipt, dict) and isinstance(receipt.get("run"), str) and receipt["run"]:
                if seed is not None and receipt.get("seed", seed) != seed:
                    continue
                if student_mode is not None and receipt.get("student_mode", "full_finetune") != student_mode:
                    continue
                run = Path(receipt["run"])
                yield run if run.is_absolute() else path / run
                continue
        yield path / condition


def run_suite(plan, *, check_only=False, resume_suite=None, launcher=None, verifier=None):
    """Run every preflight before training; keep successful groups after a failure."""
    launcher, verifier = launcher or _launch, verifier or verify_run
    project = Path(plan["project"])
    signature = protocol_hash(plan)
    workflow_format = plan.get("format_version", FORMAT)
    if resume_suite:
        suite = _path(resume_suite, project)
        state = _read(suite / "suite.json")
        if state.get("format_version") != workflow_format or state.get("protocol_sha256") != signature:
            raise ValueError("Resume request differs from the saved experiment protocol; start a new suite")
        if set(state.get("jobs", {})) != {job["id"] for job in plan["jobs"]}:
            raise ValueError("Resume manifest job set differs from this request")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suite = Path(plan["output_root"]) / "suites" / f"suite_{stamp}_{uuid.uuid4().hex[:8]}"
        suite.mkdir(parents=True)
        state = {"format_version": workflow_format, "protocol_sha256": signature, "plan": plan,
                 "jobs": {job["id"]: {"job": job, "status": "pending", "attempts": []} for job in plan["jobs"]}}
    print(f"Suite directory: {suite}", flush=True)
    for record in state["jobs"].values():
        record["resume_eligible"] = bool(resume_suite and record["status"] in {"completed", "reused", "resumed"})
    state.update(status="running", check_only=bool(check_only))
    _json(suite / "suite.json", state)

    def invoke(job, action):
        directory = suite / "jobs" / job["id"] / f"{action}_{uuid.uuid4().hex[:8]}"
        try:
            return launcher(job, action, project, directory), str(directory)
        except Exception as error:
            raise RuntimeError(f"{error} (attempt: {directory})") from error

    def checked(job, value, error):
        record = state["jobs"][job["id"]]
        if error is not None:
            record.update(status="failed", phase="check", error=str(error))
        else:
            raw, directory = value
            record["attempts"].append({"phase": "check", "directory": directory})
            try:
                record["check"] = _check_receipt(raw)
                _same("preflight seed", record["check"]["config"].get("seed"), job["seed"])
                _same("preflight student mode", _student_mode(record["check"]["config"]), job["student_mode"])
                _same("preflight condition", record["check"].get("condition"), job["condition"])
                for key, expected in job["config"]["training"].items():
                    _same(f"preflight training.{key}", record["check"]["config"]["training"].get(key), expected)
                _same("preflight distillation settings", record["check"]["config"].get("distillation"), job["config"].get("distillation"))
                record.update(status="checked", phase="check")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", phase="check", error=str(invalid))
        print(f"[check {record['status']}] {job['id']}" + (f": {record['error']}" if record.get("error") else ""), flush=True)
        _json(suite / "suite.json", state)

    _parallel(plan["jobs"], plan["max_parallel"], lambda job: invoke(job, "check"), checked)
    # A valid individual run is not sufficient for an A/B/C/D comparison. Check
    # that every successful preflight uses the same frozen task and view.
    reference, initial_states = None, {}
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "checked":
            continue
        check = record["check"]
        train = deepcopy(check["config"]["training"])
        train.pop("backbone_lr", None)
        train.pop("lora_lr", None)
        protocol = {"cohort": check["cohort_hash"], "teacher": check["teacher"].get("checkpoint_sha256"),
                    "shared_view": check["shared_view"], "training": train,
                    "evaluation": check["config"].get("evaluation"),
                    "student_initialization": check["model_metadata"].get("initialization")}
        try:
            if reference is not None:
                _same("cross-group frozen data/training protocol", protocol, reference)
            reference = protocol
            pair = (job["student_mode"], job["seed"])
            if pair in initial_states:
                _same("paired initial student state", check["initial_state_sha256"], initial_states[pair])
            initial_states[pair] = check["initial_state_sha256"]
        except ValueError as error:
            record.update(status="failed", phase="check", error=str(error))
            print(f"[check failed] {job['id']}: {error}", flush=True)
    train_jobs = []
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "checked":
            continue
        if check_only:
            # A diagnostic rerun must not erase a previously completed job's
            # eligibility for resume. Revalidate its artifact before retaining it.
            if record.get("resume_eligible") and record.get("result", {}).get("run"):
                try:
                    receipt = verifier(record["result"]["run"], record["check"], job)
                    if record["result"].get("checkpoint_sha256"):
                        _same("saved checkpoint fingerprint", receipt.get("checkpoint_sha256"),
                              record["result"]["checkpoint_sha256"])
                    record.update(status="resumed", result=receipt, source="resumed")
                except Exception as error:
                    record["reuse_rejections"] = [{"run": record["result"]["run"], "reason": str(error)}]
            continue
        candidates = []
        if record.get("resume_eligible") and record.get("result", {}).get("run"):
            candidates.append((record["result"]["run"], "resumed"))
        candidates.extend((str(run), "reused") for run in _reuse_candidates(
            plan["reuse_suites"], job["condition"], seed=job["seed"], student_mode=job["student_mode"]))
        rejections = []
        for candidate, source in candidates:
            try:
                receipt = verifier(candidate, record["check"], job)
                if source == "resumed" and record["result"].get("checkpoint_sha256"):
                    _same("saved checkpoint fingerprint", receipt.get("checkpoint_sha256"), record["result"]["checkpoint_sha256"])
                record.update(status=source, result=receipt, source=source)
                print(f"[{source}] {job['id']}: {receipt['run']}", flush=True)
                break
            except Exception as error:
                rejections.append({"run": candidate, "reason": str(error)})
        else:
            train_jobs.append(job)
        if rejections:
            record["reuse_rejections"] = rejections
            print(f"[reuse unavailable] {job['id']}: {rejections[-1]['reason']}", flush=True)
    _json(suite / "suite.json", state)

    def trained(job, value, error):
        record = state["jobs"][job["id"]]
        if error is not None:
            record.update(status="failed", phase="train", error=str(error))
        else:
            raw, directory = value
            record["attempts"].append({"phase": "train", "directory": directory})
            try:
                receipt = _train_receipt(raw, job["condition"])
                receipt = verifier(receipt["run"], record["check"], job)
                record.update(status="completed", phase="train", result=receipt, source="trained")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", phase="train", error=str(invalid))
        print(f"[train {record['status']}] {job['id']}" + (f": {record['error']}" if record.get("error") else ""), flush=True)
        _json(suite / "suite.json", state)

    if train_jobs:
        _parallel(train_jobs, plan["max_parallel"], lambda job: invoke(job, "train"), trained)
    failed = any(value["status"] == "failed" for value in state["jobs"].values())
    good = any(value["status"] in {"completed", "reused", "resumed", "checked"} for value in state["jobs"].values())
    state["status"] = ("partial" if good else "failed") if failed else ("checked" if check_only else "completed")
    _json(suite / "suite.json", state)
    write_report(suite, state)
    print_metrics([(suite, state)])
    print(f"Workflow {state['status']}: {suite}\nComparison report: {suite / 'report'}", flush=True)
    return suite


def metric_rows(state):
    rows = []
    for record in state.get("jobs", {}).values():
        job, result = record["job"], record.get("result", {})
        usable = record["status"] in {"completed", "reused", "resumed"}
        test = result.get("test_patient", {}) if usable else {}
        ci = result.get("roc_auc_ci") if usable else None
        metadata = result.get("model_metadata", record.get("check", {}).get("model_metadata", {}))
        row = {"group": job["group"], "student_mode": job["student_mode"], "condition": job["condition"],
               "seed": job["seed"], "status": record["status"], "source": record.get("source", ""),
               "val_auc": result.get("best_val_patient_roc_auc") if usable else None,
               **{key: test.get(key) for key in METRICS if key != "val_auc"},
               "roc_ci_lower": ci.get("lower") if isinstance(ci, dict) else (ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None),
               "roc_ci_upper": ci.get("upper") if isinstance(ci, dict) else (ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None),
               "total_parameters": metadata.get("total_parameters"),
               "trainable_parameters": metadata.get("trainable_parameters"),
               "distillation_trainable_parameters": result.get("distillation_trainable_parameters", record.get("check", {}).get("distillation_trainable_parameters")),
               "run": result.get("run", ""), "error": record.get("error", "")}
        row["ap"] = test.get("average_precision", test.get("ap"))
        for key in ("soft_weight", "global_weight", "local_weight"):
            row[key] = job["config"].get("distillation", {}).get(key) if job["condition"] == "distillation" else 0.0
        rows.append(row)
    return rows


def summary_rows(rows):
    summaries = []
    ordered_groups = list(GROUPS) + list(dict.fromkeys(row["group"] for row in rows if row["group"] not in GROUPS))
    for group in ordered_groups:
        selected = [row for row in rows if row["group"] == group]
        if not selected:
            continue
        completed = [row for row in selected if row["status"] in {"completed", "reused", "resumed"}]
        summary = {"group": group, "student_mode": selected[0]["student_mode"], "condition": selected[0]["condition"],
                   "n_completed": len(completed), "n_requested": len(selected),
                   "seeds": ",".join(str(row["seed"]) for row in completed)}
        for key in ("soft_weight", "global_weight", "local_weight"):
            if key in selected[0]:
                summary[key] = selected[0][key]
        for metric in METRICS:
            values = [float(row[metric]) for row in completed
                      if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])]
            summary[metric + "_n"] = len(values)
            summary[metric + "_mean"] = statistics.mean(values) if values else None
            summary[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    return summaries


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["group", "status"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(suite, state):
    rows = metric_rows(state)
    _csv(Path(suite) / "report" / "comparison.csv", rows)
    _csv(Path(suite) / "report" / "summary.csv", summary_rows(rows))


def _number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "-"


def print_metrics(batches):
    for suite, state in batches:
        print(f"\nBatch: {suite}")
        columns = ("group", "student_mode", "condition", "seed", "status", "val_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper", "pr_auc", "ap", "sensitivity", "specificity")
        print("  ".join(columns))
        rows = metric_rows(state)
        for row in rows:
            print("  ".join(_number(row[key]) if key in METRICS or key.startswith("roc_ci_") else str(row.get(key, "-")) for key in columns))
            if row.get("error"):
                print(f"  {row['group']} seed{row['seed']}: {row['error']}")
        print("Across seeds (sample SD; one seed has no estimable SD):")
        for summary in summary_rows(rows):
            values = []
            for metric in METRICS:
                mean, sd = summary[metric + "_mean"], summary[metric + "_sd"]
                values.append(f"{metric}={_number(mean)} ± {_number(sd)}")
            print(f"{summary['group']} n={summary['n_completed']}/{summary['n_requested']} seeds=[{summary['seeds']}] " + "; ".join(values))
    print("Patient test metrics; Val-AUC uses validation. Batches remain separate; no run or seed is selected by test scores.")


def saved_batches(output_root, suite=None):
    paths = [Path(suite)] if suite else sorted((Path(output_root) / "suites").glob("suite_*"))
    return [(path, _read(path / "suite.json")) for path in paths if (path / "suite.json").is_file()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default="foundation_sft/configs/distillation_abcd.json")
    parser.add_argument("--groups", choices=list(GROUPS), nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--plan", action="store_true", help="Print resolved jobs without executing or writing files")
    parser.add_argument("--check-only", action="store_true", help="Preflight all selected jobs without training")
    parser.add_argument("--resume-suite")
    parser.add_argument("--show-metrics", action="store_true", help="Read saved results without Torch or file changes")
    parser.add_argument("--suite", help="Print one saved suite; otherwise print all ABCD batches")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics:
            settings = _read(_path(args.config, project))
            batches = saved_batches(_path(settings.get("output_root", "runs/distillation_abcd"), project),
                                    _path(args.suite, project) if args.suite else None)
            if not batches:
                print("No saved ABCD batches found.")
            print_metrics(batches)
            return 0
        if args.suite:
            raise ValueError("--suite requires --show-metrics; use --resume-suite to continue training")
        plan = make_plan(args.config, project, groups=args.groups, seeds=args.seeds,
                         devices=args.devices, max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        suite = run_suite(plan, check_only=args.check_only, resume_suite=args.resume_suite)
        return 1 if _read(suite / "suite.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ABCD workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
