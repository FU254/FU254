"""Matched supervised, RAD-DINO, and equal-weight three-teacher experiments.

Planning and reading reports use only the standard library. Checkpoint loading
is imported lazily while verifying an existing or newly completed training run.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from . import distillation_suite as legacy


FORMAT = "equal_teacher_suite_v1"
CONDITIONS = ("supervised", "single_teacher", "equal_teacher")
TEACHERS = ("rad_dino", "chexfound", "chexzero")
SUCCESS = {"completed", "reused", "resumed"}
SOURCE_GROUPS = {"supervised": "supervised", "single_teacher": "full_kd"}
DEFAULT_CONFIG = "foundation_sft/configs/equal_teacher_distillation.json"
METRICS = legacy.METRICS


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(config_path, project, *, conditions=None, seeds=None, devices=None,
              max_parallel=None, cache_dir=None, no_reuse=False):
    """Resolve the fixed protocol without importing Torch or writing files."""
    project = Path(project).resolve()
    path = legacy._path(config_path, project)
    settings = legacy._read(path)
    selected = list(conditions if conditions is not None else settings.get("conditions", CONDITIONS))
    if not selected or len(set(selected)) != len(selected) or any(name not in CONDITIONS for name in selected):
        raise ValueError("conditions must be unique supervised, single_teacher, or equal_teacher")
    chosen_seeds = list(seeds if seeds is not None else settings.get("seeds", [42, 43, 44]))
    if (not chosen_seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in chosen_seeds)
            or len(set(chosen_seeds)) != len(chosen_seeds)):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    chosen_devices = list(devices if devices is not None else settings.get("devices", ["cuda:0"]))
    if (not chosen_devices or any(not isinstance(device, str) or not device for device in chosen_devices)
            or len(set(chosen_devices)) != len(chosen_devices)):
        raise ValueError("devices must contain unique nonempty names")
    maximum = max_parallel if max_parallel is not None else settings.get("max_parallel", 1)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_parallel must be a positive integer")
    teachers = deepcopy(settings["teachers"])
    if not isinstance(teachers, list) or len(teachers) != 3 or [item.get("model") for item in teachers] != list(TEACHERS):
        raise ValueError("teachers must be RAD-DINO, CheXFound, CheXzero in that order")
    for teacher in teachers:
        if teacher.get("mode") != "lora" or teacher.get("seed") != 42:
            raise ValueError("Use the fixed LoRA seed42 teachers for every student seed")
        weight = teacher.get("weight")
        if type(weight) not in (int, float) or not math.isfinite(weight) or not math.isclose(weight, 1 / 3, rel_tol=0, abs_tol=1e-15):
            raise ValueError("Each selected teacher must have weight 1/3")
        if not teacher.get("suite") and not teacher.get("checkpoint"):
            raise ValueError("Each teacher needs an explicit suite or checkpoint")
    template = legacy._read(legacy._path(settings["template"], project))
    if legacy._student_mode(template) != "full_finetune":
        raise ValueError("This matched experiment uses full_finetune ViT-Tiny")
    kd = template.get("distillation", {})
    expected = {"temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
                "local_weight": 1.0, "pool_grid": [7, 7]}
    legacy._same("confirmed distillation settings", kd, expected)
    train = template["training"]
    if (train.get("backbone_lr"), train.get("head_lr"), train.get("projection_lr")) != (5e-5, 1e-4, 1e-4):
        raise ValueError("Use matched backbone/head/projection learning rates 5e-5/1e-4/1e-4")
    jobs = []
    for condition in selected:
        for seed in chosen_seeds:
            config = deepcopy(template)
            device = chosen_devices[len(jobs) % len(chosen_devices)]
            config.update(seed=seed, device=device, teachers=deepcopy(teachers))
            # The anchor retains the old single-teacher inference/view contract.
            config.update(teacher_suite=teachers[0].get("suite"),
                          teacher_checkpoint=teachers[0].get("checkpoint"))
            if cache_dir is not None:
                config["cache_dir"] = str(legacy._path(cache_dir, project))
            jobs.append({"id": f"{condition}_seed{seed}", "group": condition,
                         "condition": condition, "student_mode": "full_finetune",
                         "seed": seed, "device": device, "config": config})
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
            "output_root": str(legacy._path(settings.get("output_root", "runs/equal_teacher_distillation"), project)),
            "conditions": selected, "seeds": chosen_seeds, "devices": chosen_devices,
            "max_parallel": maximum, "teachers": teachers, "no_reuse": bool(no_reuse),
            "reuse_suites": [str(legacy._path(value, project)) for value in settings.get("reuse_suites", [])],
            "jobs": jobs}


def protocol_hash(plan):
    """GPU assignment, concurrency, output/cache paths and reuse policy are execution choices."""
    value = deepcopy(plan)
    for key in ("config_path", "output_root", "max_parallel", "devices", "no_reuse", "reuse_suites"):
        value.pop(key, None)
    for job in value["jobs"]:
        job.pop("device", None)
        for key in ("device", "output_root", "cache_dir"):
            job["config"].pop(key, None)
    return legacy._hash(value)


def _launch(job, action, project, directory):
    """Keep the complete child log and stream each epoch to the parent terminal."""
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    config_file, receipt_file = directory / "config.json", directory / "result.json"
    legacy._json(config_file, config)
    command = [sys.executable, "-u", str(project / "run_equal_teacher_distillation.py"),
               "--project", str(project), "--config", str(config_file),
               "--condition", job["condition"], "--result-json", str(receipt_file)]
    if action == "check":
        command.append("--check-only")
    log = directory / f"{action}.log"
    print(f"[{action}] {job['id']} {job['device']} log={log}", flush=True)
    last_lines = deque(maxlen=12)
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line)
                last_lines.append(line.rstrip())
                try:
                    payload = json.loads(line)
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict) and "epoch" in payload:
                    print(f"[{job['condition']} seed{job['seed']}] {line.rstrip()}", flush=True)
                elif line.startswith(("[epoch]", "[train]", "[evaluation]", "Run directory:", "[prepare]")):
                    print(f"[{job['condition']} seed{job['seed']}] {line.rstrip()}", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        for line in last_lines:
            print(f"[{job['id']} log] {line}", flush=True)
        raise RuntimeError(f"Process exited {code}; see {log}")
    if not receipt_file.is_file():
        raise RuntimeError(f"Child did not write {receipt_file}")
    return legacy._read(receipt_file)


def verify_run(run, check, job):
    """Validate artifacts; never select a baseline or checkpoint using its scores."""
    if job["condition"] == "equal_teacher":
        from .equal_teacher_distillation import verify_completed_run
        result = verify_completed_run(run, check, job)
    else:
        mapped = deepcopy(job)
        mapped["condition"] = "supervised" if job["condition"] == "supervised" else "distillation"
        result = legacy.verify_run(run, check, mapped)
        result["source_condition"] = result["condition"]
        result["condition"] = job["condition"]
    return result


def _reuse_candidates(plan, job):
    """Only the configured completed ablation groups qualify as control candidates."""
    if plan.get("no_reuse") or job["condition"] not in SOURCE_GROUPS:
        return
    wanted = SOURCE_GROUPS[job["condition"]]
    for source in plan.get("reuse_suites", []):
        suite = Path(source)
        try:
            data = legacy._read(suite / "suite.json")
            records = data.get("jobs", {})
            if not isinstance(records, dict):
                continue
            for record in records.values():
                saved = record.get("job", {})
                expected_condition = "supervised" if wanted == "supervised" else "distillation"
                if (record.get("status") not in SUCCESS or saved.get("group") != wanted
                        or saved.get("condition") != expected_condition
                        or saved.get("seed") != job["seed"]
                        or saved.get("student_mode") != "full_finetune"):
                    continue
                run = record.get("result", {}).get("run")
                if isinstance(run, str) and run:
                    yield str(legacy._path(run, suite))
        except (OSError, ValueError, TypeError, AttributeError):
            continue


def _checked(value, job):
    check = legacy._check_receipt(value)
    legacy._same("preflight condition", check.get("condition"), job["condition"])
    legacy._same("preflight seed", check["config"].get("seed"), job["seed"])
    legacy._same("preflight student mode", legacy._student_mode(check["config"]), "full_finetune")
    for key, expected in job["config"]["training"].items():
        legacy._same(f"preflight training.{key}", check["config"]["training"].get(key), expected)
    if job["condition"] != "supervised":
        legacy._same("preflight distillation settings", check["config"].get("distillation"), job["config"]["distillation"])
    teachers = check.get("teachers")
    if isinstance(teachers, dict):
        teachers = [dict(value, model=name) for name, value in teachers.items()]
    if (not isinstance(teachers, list) or [item.get("model") for item in teachers] != list(TEACHERS)
            or any(not item.get("checkpoint_sha256") for item in teachers)):
        raise ValueError("Preflight lacks all three teacher checkpoint fingerprints")
    for teacher in teachers:
        if not math.isclose(teacher.get("weight", 0), 1 / 3, rel_tol=0, abs_tol=1e-15):
            raise ValueError("Preflight teacher weights differ from 1/3")
    return check


def _cross_protocol(check):
    return {"cohort_hash": check["cohort_hash"], "shared_view": check["shared_view"],
            "teacher": check["teacher"], "teachers": check["teachers"],
            "teacher_views": check.get("teacher_views"),
            "training": check["config"]["training"], "evaluation": check["config"].get("evaluation"),
            "student_initialization": check["model_metadata"].get("initialization")}


def run_suite(plan, *, check_only=False, resume_suite=None, launcher=None, verifier=None):
    """Persist each task start/end and preserve prior artifacts across partial failures."""
    launcher, verifier = launcher or _launch, verifier or verify_run
    project, signature = Path(plan["project"]), protocol_hash(plan)
    if resume_suite:
        suite = legacy._path(resume_suite, project)
        state = legacy._read(suite / "suite.json")
        if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
            raise ValueError("Resume differs from the saved scientific protocol; start a new suite")
        if set(state.get("jobs", {})) != {job["id"] for job in plan["jobs"]}:
            raise ValueError("Resume job set differs")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suite = Path(plan["output_root"]) / "suites" / f"suite_{stamp}_{uuid.uuid4().hex[:8]}"
        suite.mkdir(parents=True)
        state = {"format_version": FORMAT, "protocol_sha256": signature, "jobs": {
            job["id"]: {"job": deepcopy(job), "status": "pending", "attempts": []} for job in plan["jobs"]}}
    state.update(plan=deepcopy(plan), status="running", check_only=bool(check_only))
    state.setdefault("execution_history", []).append({"started_at": _now(), "devices": plan["devices"],
        "max_parallel": plan["max_parallel"], "check_only": bool(check_only), "no_reuse": plan.get("no_reuse", False)})
    previous = {}
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        record["job"] = deepcopy(job)
        if resume_suite and record.get("result", {}).get("run"):
            previous[job["id"]] = deepcopy(record["result"])
    lock = threading.RLock()

    def persist():
        with lock:
            state["updated_at"] = _now()
            legacy._json(suite / "suite.json", state)

    persist()
    print(f"Suite directory: {suite}", flush=True)

    def invoke(job, action):
        directory = suite / "jobs" / job["id"] / f"{action}_{uuid.uuid4().hex[:8]}"
        attempt = {"phase": action, "directory": str(directory), "status": "running", "started_at": _now(), "device": job["device"]}
        with lock:
            record = state["jobs"][job["id"]]
            record.update(status="checking" if action == "check" else "training", phase=action)
            record["attempts"].append(attempt)
            persist()
        try:
            value = launcher(job, action, project, directory)
        except Exception as error:
            with lock:
                attempt.update(status="failed", finished_at=_now(), error=str(error))
                persist()
            raise RuntimeError(f"{error} (attempt: {directory})") from error
        with lock:
            attempt.update(status="completed", finished_at=_now())
            persist()
        return value

    def checked(job, value, error):
        with lock:
            record = state["jobs"][job["id"]]
            try:
                if error is not None:
                    raise error
                record["check"] = _checked(value, job)
                record.update(status="checked", phase="check")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", phase="check", error=str(invalid))
            print(f"[check {record['status']}] {job['id']}" + (f": {record['error']}" if record.get("error") else ""), flush=True)
            persist()

    legacy._parallel(plan["jobs"], plan["max_parallel"], lambda job: invoke(job, "check"), checked)
    reference, initial_states = None, {}
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "checked":
            continue
        check = record["check"]
        try:
            protocol = _cross_protocol(check)
            if reference is not None:
                legacy._same("cross-condition frozen data/teachers/view/training", protocol, reference)
            reference = protocol
            if job["seed"] in initial_states:
                legacy._same("paired initial student state", check["initial_state_sha256"], initial_states[job["seed"]])
            initial_states[job["seed"]] = check["initial_state_sha256"]
        except ValueError as invalid:
            record.update(status="failed", phase="check", error=str(invalid))
            print(f"[check failed] {job['id']}: {invalid}", flush=True)
        persist()

    train_jobs = []
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "checked":
            continue
        candidates = []
        old = previous.get(job["id"])
        if old:
            candidates.append((old["run"], "resumed"))
        if not check_only:
            candidates.extend((candidate, "reused") for candidate in _reuse_candidates(plan, job))
        accepted = False
        for candidate, source in candidates:
            try:
                result = verifier(candidate, record["check"], job)
                if source == "resumed" and old.get("checkpoint_sha256"):
                    legacy._same("saved checkpoint fingerprint", result.get("checkpoint_sha256"), old["checkpoint_sha256"])
                record.update(status=source, result=result, source=source)
                record.pop("error", None)
                print(f"[{source}] {job['id']}: {result['run']}", flush=True)
                accepted = True
                break
            except Exception as invalid:
                record.setdefault("reuse_rejections", []).append({"run": candidate, "reason": str(invalid)})
                print(f"[reuse unavailable] {job['id']}: {invalid}", flush=True)
        if not accepted and not check_only:
            train_jobs.append(job)
        persist()

    def trained(job, value, error):
        with lock:
            record = state["jobs"][job["id"]]
            try:
                if error is not None:
                    raise error
                if "check" in value and not value.get("run"):
                    raise ValueError("Training returned only a check receipt")
                if value.get("status") != "completed" or value.get("condition") != job["condition"] or not value.get("run"):
                    raise ValueError("Training receipt is incomplete or has the wrong condition")
                receipt = verifier(value["run"], record["check"], job)
                record.update(status="completed", phase="train", result=receipt, source="trained")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", phase="train", error=str(invalid))
            print(f"[train {record['status']}] {job['id']}" + (f": {record['error']}" if record.get("error") else ""), flush=True)
            persist()

    if train_jobs:
        legacy._parallel(train_jobs, plan["max_parallel"], lambda job: invoke(job, "train"), trained)
    failed = any(record["status"] == "failed" for record in state["jobs"].values())
    good = any(record["status"] in SUCCESS | {"checked"} for record in state["jobs"].values())
    state["status"] = ("partial" if good else "failed") if failed else ("checked" if check_only else "completed")
    state["execution_history"][-1].update(finished_at=_now(), status=state["status"])
    persist()
    write_report(suite, state)
    print_metrics([(suite, state)])
    print(f"Workflow {state['status']}: {suite}\nComparison report: {suite / 'report'}", flush=True)
    return suite


def metric_rows(state):
    rows = legacy.metric_rows(state)
    for row in rows:
        distilling = row["condition"] != "supervised"
        job = state["jobs"][f"{row['condition']}_seed{row['seed']}"]["job"]
        for key in ("soft_weight", "global_weight", "local_weight"):
            row[key] = job["config"]["distillation"][key] if distilling else 0.0
        row["teacher_count"] = 3 if row["condition"] == "equal_teacher" else (1 if distilling else 0)
    return rows


def paired_rows(rows):
    pairs = (("single_teacher_gain", "single_teacher", "supervised"),
             ("equal_teacher_gain", "equal_teacher", "supervised"),
             ("equal_minus_single", "equal_teacher", "single_teacher"))
    result = []
    for name, candidate, reference in pairs:
        first = {row["seed"]: row for row in rows if row["condition"] == candidate and row["status"] in SUCCESS}
        second = {row["seed"]: row for row in rows if row["condition"] == reference and row["status"] in SUCCESS}
        for seed in sorted(first.keys() & second.keys()):
            value = {"comparison": name, "candidate": candidate, "reference": reference, "seed": seed}
            for metric in METRICS:
                left, right = first[seed].get(metric), second[seed].get(metric)
                value["delta_" + metric] = left - right if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
            result.append(value)
    return result


def write_report(suite, state):
    root = Path(suite) / "report"
    rows = metric_rows(state)
    legacy._csv(root / "comparison.csv", rows)
    legacy._csv(root / "summary.csv", legacy.summary_rows(rows))
    legacy._csv(root / "matched_seed_differences.csv", paired_rows(rows))


def print_metrics(batches):
    for suite, state in batches:
        rows = metric_rows(state)
        print(f"\nBatch: {suite}")
        columns = ("condition", "seed", "status", "val_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper", "pr_auc", "ap", "sensitivity", "specificity")
        print("  ".join(columns))
        for row in rows:
            print("  ".join(legacy._number(row.get(key)) if key in METRICS or key.startswith("roc_ci_") else str(row.get(key, "-")) for key in columns))
            if row.get("error"):
                print(f"  {row['condition']} seed{row['seed']}: {row['error']}")
        print("Across seeds (sample SD; one seed has no estimable SD):")
        for summary in legacy.summary_rows(rows):
            values = [f"{metric}={legacy._number(summary[metric + '_mean'])} ± {legacy._number(summary[metric + '_sd'])}" for metric in METRICS]
            print(f"{summary['group']} n={summary['n_completed']}/{summary['n_requested']} seeds=[{summary['seeds']}] " + "; ".join(values))
        for value in paired_rows(rows):
            print(f"{value['comparison']} seed{value['seed']} delta_roc_auc={legacy._number(value['delta_roc_auc'])}; delta_pr_auc={legacy._number(value['delta_pr_auc'])}")
    print("Patient test metrics; Val-AUC uses validation. PR-AUC and AP are distinct. No run or seed is selected by test scores.")


def saved_batches(output_root, suite=None):
    return [(path, state) for path, state in legacy.saved_batches(output_root, suite) if state.get("format_version") == FORMAT]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--no-reuse", action="store_true", help="Train controls again instead of importing historical controls")
    parser.add_argument("--plan", action="store_true", help="Print jobs without importing Torch or writing files")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume-suite")
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--suite", help="Read only this saved suite with --show-metrics")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics:
            settings = legacy._read(legacy._path(args.config, project))
            batches = saved_batches(legacy._path(settings.get("output_root", "runs/equal_teacher_distillation"), project),
                                    legacy._path(args.suite, project) if args.suite else None)
            if not batches:
                print("No saved equal-teacher experiment batches found.")
            print_metrics(batches)
            return 0
        if args.suite:
            raise ValueError("--suite requires --show-metrics; use --resume-suite to continue")
        plan = make_plan(args.config, project, conditions=args.conditions, seeds=args.seeds,
                         devices=args.devices, max_parallel=args.max_parallel, cache_dir=args.cache_dir, no_reuse=args.no_reuse)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        suite = run_suite(plan, check_only=args.check_only, resume_suite=args.resume_suite)
        return 1 if legacy._read(suite / "suite.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Equal-teacher workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
