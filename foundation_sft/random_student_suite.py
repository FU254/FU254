"""Plan and run matched randomly initialized ViT-Tiny student experiments.

Planning and reporting use the standard library. Historical ImageNet student
runs are never imported. Only complete runs from the explicitly resumed random
student batch may be retained after strict verification.
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


FORMAT = "random_student_suite_v1"
GROUPS = {"A": ("full_finetune", "supervised"),
          "B": ("full_finetune", "distillation")}
DEFAULT_CONFIG = "foundation_sft/configs/random_student_distillation.json"
SUCCESS = {"completed", "resumed"}
METRICS = legacy.METRICS


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(config_path, project, *, groups=None, seeds=None, devices=None,
              max_parallel=None, cache_dir=None):
    """Resolve explicit random initialization without reading weights or data."""
    project = Path(project).resolve()
    path = legacy._path(config_path, project)
    settings = legacy._read(path)
    if settings.get("student_initialization") != "random":
        raise ValueError("This workflow requires student_initialization=random")
    if settings.get("student_checkpoint") is not None:
        raise ValueError("Random students must not specify an ImageNet checkpoint")
    if settings.get("reuse_suites"):
        raise ValueError("Random-student batches do not import historical experiments")
    selected = list(groups if groups is not None else settings.get("groups", ["A", "B"]))
    if (not selected or len(set(selected)) != len(selected)
            or any(group not in GROUPS for group in selected)):
        raise ValueError("groups must be unique members of A B")
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
    if not settings.get("teacher_suite") and not settings.get("teacher_checkpoint"):
        raise ValueError("Declare the fixed RAD-DINO teacher suite or checkpoint")
    train = settings.get("training", {})
    if train.get("amp") is not False:
        raise ValueError("The matched random-student protocol uses FP32 training.amp=false")
    for name in ("epochs", "patience", "batch_size", "accumulation_steps"):
        if type(train.get(name)) is not int or train[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if type(train.get("num_workers")) is not int or train["num_workers"] < 0:
        raise ValueError("training.num_workers must be nonnegative")
    for name in ("backbone_lr", "head_lr", "projection_lr"):
        if type(train.get(name)) not in (int, float) or not math.isfinite(train[name]) or train[name] <= 0:
            raise ValueError(f"training.{name} must be positive and finite")
    expected_loss = {"temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
                     "local_weight": 1.0, "pool_grid": [7, 7]}
    legacy._same("matched distillation coefficients", settings.get("distillation"), expected_loss)
    jobs = []
    for group in selected:
        mode, condition = GROUPS[group]
        for seed in chosen_seeds:
            device = chosen_devices[len(jobs) % len(chosen_devices)]
            config = {"teacher_suite": settings.get("teacher_suite"),
                "teacher_checkpoint": settings.get("teacher_checkpoint"),
                "student_checkpoint": None, "student_initialization": "random",
                "student": {"mode": mode}, "seed": seed, "device": device,
                "training": deepcopy(train), "distillation": deepcopy(settings["distillation"]),
                "output_root": settings.get("output_root", "runs/random_student_distillation"),
                "cache_dir": settings.get("cache_dir")}
            if cache_dir is not None:
                config["cache_dir"] = str(legacy._path(cache_dir, project))
            jobs.append({"id": f"{group}_seed{seed}", "group": group, "seed": seed,
                         "student_mode": mode, "condition": condition, "device": device,
                         "initialization": "random", "config": config})
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
            "output_root": str(legacy._path(settings.get("output_root", "runs/random_student_distillation"), project)),
            "max_parallel": maximum, "groups": selected, "seeds": chosen_seeds,
            "devices": chosen_devices, "initialization": "random", "jobs": jobs}


def protocol_hash(plan):
    """Runtime GPU assignment and storage location may change during queue resume."""
    value = deepcopy(plan)
    for key in ("config_path", "output_root", "max_parallel", "devices"):
        value.pop(key, None)
    for job in value["jobs"]:
        job.pop("device", None)
        for key in ("device", "output_root", "cache_dir"):
            job["config"].pop(key, None)
    return legacy._hash(value)


def _launch(job, action, project, directory):
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    config_file, receipt_file = directory / "config.json", directory / "result.json"
    legacy._json(config_file, config)
    command = [sys.executable, "-u", str(project / "run_random_student_distillation.py"),
               "--project", str(project), "--config", str(config_file),
               "--condition", job["condition"], "--result-json", str(receipt_file)]
    if action == "check":
        command.append("--check-only")
    log = directory / f"{action}.log"
    print(f"[{action}] {job['id']} {job['device']} initialization=random log={log}", flush=True)
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
                    print(f"[{job['id']}] {line.rstrip()}", flush=True)
                elif line.startswith(("[epoch]", "[train]", "[evaluation]", "Run directory:", "[prepare]")):
                    print(f"[{job['id']}] {line.rstrip()}", flush=True)
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
    from .random_student_distillation import verify_completed_run
    return verify_completed_run(run, check, job)


def _checked(value, job):
    check = legacy._check_receipt(value)
    config = check["config"]
    for name, actual, expected in (
        ("condition", check.get("condition"), job["condition"]),
        ("seed", config.get("seed"), job["seed"]),
        ("student mode", legacy._student_mode(config), job["student_mode"]),
        ("student initialization", config.get("student_initialization"), "random"),
        ("student checkpoint", config.get("student_checkpoint"), None),
        ("frozen teacher", check.get("teacher_unchanged"), True)):
        legacy._same("preflight " + name, actual, expected)
    initialization = check["model_metadata"].get("initialization", {})
    if (initialization.get("pretraining") != "none"
            or initialization.get("initialization_mode") != "random"
            or initialization.get("seed") != job["seed"]):
        raise ValueError("Preflight metadata must identify this seed's random student initialization")
    if any(name in initialization for name in ("checkpoint", "checkpoint_sha256", "official_checkpoint_url")):
        raise ValueError("Random initialization must not claim an ImageNet source checkpoint")
    for key, expected in job["config"]["training"].items():
        legacy._same(f"preflight training.{key}", config["training"].get(key), expected)
    legacy._same("preflight distillation coefficients", config.get("distillation"), job["config"]["distillation"])
    return check


def _cross_protocol(check):
    training = deepcopy(check["config"]["training"])
    initialization = deepcopy(check["model_metadata"]["initialization"])
    initialization.pop("seed", None)
    return {"cohort_hash": check["cohort_hash"], "teacher": check["teacher"],
            "shared_view": check["shared_view"], "training": training,
            "evaluation": check["config"].get("evaluation"), "initialization": initialization}


def run_suite(plan, *, check_only=False, resume_suite=None, launcher=None, verifier=None):
    """Preflight every job, retain only verified completed jobs of this batch."""
    launcher, verifier = launcher or _launch, verifier or verify_run
    project, signature = Path(plan["project"]), protocol_hash(plan)
    if resume_suite:
        suite = legacy._path(resume_suite, project)
        state = legacy._read(suite / "suite.json")
        if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
            raise ValueError("Resume differs from the saved random-student scientific protocol")
        if set(state.get("jobs", {})) != {job["id"] for job in plan["jobs"]}:
            raise ValueError("Resume job set differs")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suite = Path(plan["output_root"]) / "suites" / f"suite_{stamp}_{uuid.uuid4().hex[:8]}"
        suite.mkdir(parents=True)
        state = {"format_version": FORMAT, "protocol_sha256": signature, "jobs": {
            job["id"]: {"job": deepcopy(job), "status": "pending", "attempts": []} for job in plan["jobs"]}}
    previous = {name: deepcopy(record["result"]) for name, record in state["jobs"].items()
                if resume_suite and record.get("result", {}).get("run")}
    for job in plan["jobs"]:
        state["jobs"][job["id"]]["job"] = deepcopy(job)
    state.update(plan=deepcopy(plan), status="running", check_only=bool(check_only), initialization="random")
    state.setdefault("execution_history", []).append({"started_at": _now(), "devices": plan["devices"],
        "max_parallel": plan["max_parallel"], "check_only": bool(check_only)})
    lock = threading.RLock()

    def persist():
        with lock:
            state["updated_at"] = _now()
            legacy._json(suite / "suite.json", state)

    persist()
    print(f"Suite directory: {suite}", flush=True)
    print("Student initialization: random; no historical ImageNet results will be imported.", flush=True)

    def invoke(job, action):
        directory = suite / "jobs" / job["id"] / f"{action}_{uuid.uuid4().hex[:8]}"
        attempt = {"phase": action, "directory": str(directory), "status": "running", "started_at": _now(), "device": job["device"]}
        with lock:
            record = state["jobs"][job["id"]]
            record.update(status="checking" if action == "check" else "training", phase=action)
            record["attempts"].append(attempt)
            persist()
        try:
            result = launcher(job, action, project, directory)
        except Exception as error:
            with lock:
                attempt.update(status="failed", finished_at=_now(), error=str(error))
                persist()
            raise RuntimeError(f"{error} (attempt: {directory})") from error
        with lock:
            attempt.update(status="completed", finished_at=_now())
            persist()
        return result

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
                legacy._same("cross-group frozen teacher/data/view/training", protocol, reference)
            reference = protocol
            pair = (job["student_mode"], job["seed"])
            if pair in initial_states:
                legacy._same("paired initialized student state", check["initial_state_sha256"], initial_states[pair])
            initial_states[pair] = check["initial_state_sha256"]
        except ValueError as invalid:
            record.update(status="failed", phase="check", error=str(invalid))
            print(f"[check failed] {job['id']}: {invalid}", flush=True)
        persist()

    train_jobs = []
    for job in plan["jobs"]:
        record = state["jobs"][job["id"]]
        if record["status"] != "checked":
            continue
        old = previous.get(job["id"])
        if old:
            try:
                result = verifier(old["run"], record["check"], job)
                if old.get("checkpoint_sha256"):
                    legacy._same("saved checkpoint fingerprint", result.get("checkpoint_sha256"), old["checkpoint_sha256"])
                record.update(status="resumed", result=result, source="resumed")
                record.pop("error", None)
                print(f"[resumed] {job['id']}: {result['run']}", flush=True)
            except Exception as invalid:
                record.setdefault("resume_rejections", []).append({"run": old["run"], "reason": str(invalid)})
                print(f"[resume unavailable] {job['id']}: {invalid}", flush=True)
        if record["status"] == "checked" and not check_only:
            train_jobs.append(job)
        persist()

    def trained(job, value, error):
        with lock:
            record = state["jobs"][job["id"]]
            try:
                if error is not None:
                    raise error
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
    return [{"initialization": "random", **row} for row in legacy.metric_rows(state)]


def summary_rows(rows):
    return [{"initialization": "random", **row} for row in legacy.summary_rows(rows)]


def paired_rows(rows):
    output = []
    for name, candidate, reference in (("full_kd_gain", "B", "A"),):
        first = {row["seed"]: row for row in rows if row["group"] == candidate and row["status"] in SUCCESS}
        second = {row["seed"]: row for row in rows if row["group"] == reference and row["status"] in SUCCESS}
        for seed in sorted(first.keys() & second.keys()):
            value = {"initialization": "random", "comparison": name, "candidate": candidate,
                     "reference": reference, "seed": seed}
            for metric in METRICS:
                left, right = first[seed].get(metric), second[seed].get(metric)
                value["delta_" + metric] = left - right if type(left) in (int, float) and type(right) in (int, float) else None
            output.append(value)
    return output


def write_report(suite, state):
    root = Path(suite) / "report"
    rows = metric_rows(state)
    legacy._csv(root / "comparison.csv", rows)
    legacy._csv(root / "summary.csv", summary_rows(rows))
    legacy._csv(root / "matched_seed_differences.csv", paired_rows(rows))


def print_metrics(batches):
    for suite, state in batches:
        rows = metric_rows(state)
        print(f"\nBatch: {suite}\nStudent initialization: random (no ImageNet checkpoint)")
        columns = ("initialization", "group", "student_mode", "condition", "seed", "status", "val_auc", "roc_auc", "roc_ci_lower", "roc_ci_upper", "pr_auc", "ap", "sensitivity", "specificity")
        print("  ".join(columns))
        for row in rows:
            print("  ".join(legacy._number(row.get(key)) if key in METRICS or key.startswith("roc_ci_") else str(row.get(key, "-")) for key in columns))
            if row.get("error"):
                print(f"  {row['group']} seed{row['seed']}: {row['error']}")
        print("Across seeds (sample SD; one seed has no estimable SD):")
        for row in summary_rows(rows):
            values = [f"{metric}={legacy._number(row[metric + '_mean'])} ± {legacy._number(row[metric + '_sd'])}" for metric in METRICS]
            print(f"{row['group']} n={row['n_completed']}/{row['n_requested']} seeds=[{row['seeds']}] " + "; ".join(values))
        for row in paired_rows(rows):
            print(f"{row['comparison']} seed{row['seed']} delta_roc_auc={legacy._number(row['delta_roc_auc'])}; delta_pr_auc={legacy._number(row['delta_pr_auc'])}")
    print("Patient test metrics; Val-AUC uses validation. PR-AUC and AP are distinct. Batches remain separate; no run or seed is selected by test scores.")


def saved_batches(output_root, suite=None):
    return [(path, state) for path, state in legacy.saved_batches(output_root, suite) if state.get("format_version") == FORMAT]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--groups", nargs="+", choices=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume-suite")
    parser.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--suite", help="Read only this saved random-student batch with --show-metrics")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics:
            settings = legacy._read(legacy._path(args.config, project))
            batches = saved_batches(legacy._path(settings.get("output_root", "runs/random_student_distillation"), project),
                                    legacy._path(args.suite, project) if args.suite else None)
            if not batches:
                print("No saved random-student experiment batches found.")
            print_metrics(batches)
            return 0
        if args.suite:
            raise ValueError("--suite requires --show-metrics; use --resume-suite to continue")
        plan = make_plan(args.config, project, groups=args.groups, seeds=args.seeds,
            devices=args.devices, max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        suite = run_suite(plan, check_only=args.check_only, resume_suite=args.resume_suite)
        return 1 if legacy._read(suite / "suite.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Random-student workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
