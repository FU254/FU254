"""Coordinate frozen-data checks, GPU experiments and comparison reports.

The coordinator uses only Python's standard library. Each model executes in
its configured Conda environment; no Torch installation is needed to plan jobs.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import uuid

from .config import load_config


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_plan(path, models=None, modes=None, seeds=None):
    """Resolve every inherited config before any child process starts."""
    with Path(path).open(encoding="utf-8") as handle:
        settings = json.load(handle)
    modes = modes or settings.get("modes", ["linear_probe", "lora"])
    seeds = seeds or settings.get("seeds", [42])
    if len(set(modes)) != len(modes) or set(modes) - {"linear_probe", "lora"}:
        raise ValueError("Choose distinct linear_probe/lora modes")
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Choose distinct nonnegative integer seeds")
    selected = [row for row in settings["jobs"] if models is None or row["model"] in models]
    if models and set(models) != {row["model"] for row in selected}:
        raise ValueError("A requested model is absent from the experiment configuration")
    jobs = []
    for row in selected:
        for mode in modes:
            for seed in seeds:
                config = load_config(row["config"], {"mode": mode, "seed": seed, "device": row["device"]})
                if config["model"] != row["model"]:
                    raise ValueError("Experiment model name differs from its model configuration")
                jobs.append({"id": f"{row['model']}_{mode}_seed{seed}", "model": row["model"],
                             "mode": mode, "seed": seed, "env": row["env"],
                             "device": row["device"], "config": config})
    plan = {"jobs": jobs, "suite_root": settings.get("suite_root", "runs/foundation_sft/suites"),
            "conda_executable": settings.get("conda_executable", "conda")}
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    if not plan.get("jobs"):
        raise ValueError("The experiment plan contains no jobs")
    ids, experiments, shared = set(), set(), None
    for job in plan["jobs"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]) or job["id"] in ids:
            raise ValueError("Each job needs a distinct safe id")
        ids.add(job["id"])
        identity = (job["model"], job["mode"], job["seed"])
        if identity in experiments:
            raise ValueError("Duplicate model/mode/seed experiment")
        experiments.add(identity)
        if not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", job["device"]):
            raise ValueError("Use cpu or an explicit cuda:N device to prevent GPU aliases")
        config = job["config"]
        if any(config[key] != job[key] for key in ("model", "mode", "seed", "device")):
            raise ValueError("Job identity must match the resolved configuration")
        signature = {key: config[key] for key in ("data", "split_seed", "split_ratios")}
        signature["split_manifest"] = str(Path(config["split_manifest"]).resolve())
        if shared is not None and signature != shared:
            raise ValueError("All models must share data, split_manifest, split_seed and split_ratios")
        shared = signature


def _command(command, log):
    # The CUDA 11.0 path in the user's shell shadows Torch's bundled CUDA 11.8.
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    with Path(log).open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=Path.cwd(), env=environment,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Process exited {result.returncode}; see {log}")


def _last_log_lines(log, limit=12):
    """Read a bounded failure excerpt without replacing the original failure."""
    try:
        with Path(log).open(encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=limit)).rstrip("\n")
    except OSError as error:
        return f"Log unavailable: {error}"


def _parallel_by_device(jobs, operation, max_parallel):
    groups = {}
    for job in jobs:
        groups.setdefault(job["device"], []).append(job)

    def serial_queue(queue):
        for job in queue:
            operation(job)

    with ThreadPoolExecutor(max_workers=min(max_parallel, len(groups))) as executor:
        futures = [executor.submit(serial_queue, queue) for queue in groups.values()]
        for future in futures:
            future.result()


def run_suite(plan, *, check_only=False, max_parallel=5):
    """Run every prespecified job, returning a manifest even for failed stages.

    Success receipts identify each exact run. A smoke failure blocks all formal
    training. Training failures are reported alongside all successful jobs;
    they never cause a different seed to be chosen by its test performance.
    """
    _validate_plan(plan)
    if type(max_parallel) is not int or max_parallel < 1:
        raise ValueError("max_parallel must be a positive integer")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite = Path(plan["suite_root"]).resolve() / f"suite_{stamp}_{uuid.uuid4().hex[:8]}"
    suite.mkdir(parents=True, exist_ok=False)
    for name in ("configs", "logs", "receipts"):
        (suite / name).mkdir()
    manifest = {"format_version": 1, "status": "preparing", "check_only": check_only,
                "project_dir": str(Path.cwd()), "jobs": [], "report_status": "pending"}
    for planned in plan["jobs"]:
        job = {key: value for key, value in planned.items() if key != "config"}
        job.update(status="pending", config_file=str(suite / "configs" / f"{job['id']}.json"))
        _write_json(job["config_file"], planned["config"])
        manifest["jobs"].append(job)
    lock = threading.Lock()
    console_lock = threading.Lock()

    def announce(message):
        # Keep each failure and its excerpt together when several GPUs fail.
        with console_lock:
            print(message, flush=True)

    def save():
        _write_json(suite / "suite.json", manifest)

    def update(job, **values):
        with lock:
            job.update(values)
            save()

    def stage(job, action):
        log = suite / "logs" / f"{job['id']}_{action}.log"
        receipt = suite / "receipts" / f"{job['id']}_{action}.json"
        command = [plan["conda_executable"], "run", "--no-capture-output", "-n", job["env"],
                   "python", "-m", "foundation_sft.train", "--config", job["config_file"],
                   "--action", action, "--result-json", str(receipt)]
        update(job, **{f"{action}_status": "running", f"{action}_log": str(log)})
        announce(f"[{action}] {job['id']}  {job['device']}  log={log}")
        try:
            _command(command, log)
            with receipt.open(encoding="utf-8") as handle:
                result = json.load(handle)
            if result.get("action") != action:
                raise ValueError("Missing or incorrect success receipt")
            for key in ("model", "mode", "seed"):
                if key in result and result[key] != job[key]:
                    raise ValueError("Receipt belongs to another experiment")
            if action == "prepare" and not result.get("cohort_hash"):
                raise ValueError("Preparation did not record the frozen split fingerprint")
            if action == "smoke" and result.get("smoke_passed") is not True:
                raise ValueError("Smoke test did not pass")
            if action == "train":
                run = Path(result["run_dir"])
                required = ("best.pt", "last.pt", "config.json", "environment.json", "selection.json",
                            "test/metrics.json", "test/patient_predictions.csv")
                if not run.is_absolute() or any(not (run / name).is_file() for name in required):
                    raise ValueError("The reported run is missing training/evaluation artifacts")
                saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
                if any(saved.get(key) != job[key] for key in ("model", "mode", "seed")):
                    raise ValueError("The reported run belongs to another experiment")
                provenance = json.loads((run / "environment.json").read_text(encoding="utf-8"))
                if provenance.get("cohort_hash") != manifest["cohort_hash"]:
                    raise ValueError("The trained run uses a different frozen patient split")
                update(job, status="completed", run_dir=str(run))
            update(job, **{f"{action}_status": "passed"})
            announce(f"[{action} passed] {job['id']}")
            return result
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            update(job, status="failed", error=str(error), **{f"{action}_status": "failed"})
            excerpt = _last_log_lines(log)
            announce(f"[{action} failed] {job['id']}: {error}\n"
                     f"[Last log lines] {job['id']}  model={job['model']}  "
                     f"mode={job['mode']}  action={action}\n{excerpt}")
            return None

    save()
    print(f"Suite directory: {suite}", flush=True)
    prepared = stage(manifest["jobs"][0], "prepare")
    if prepared is None:
        manifest["status"] = "failed"
        save()
        return suite
    manifest.update(cohort_hash=prepared["cohort_hash"], status="checking")
    save()
    # A new random seed does not change the architecture or checkpoint; check
    # one seed per model/mode, then share its receipt with that mode's jobs.
    smoke_jobs, references = [], {}
    for job in manifest["jobs"]:
        key = (job["model"], job["mode"])
        if key not in references:
            references[key] = job
            smoke_jobs.append(job)
    _parallel_by_device(smoke_jobs, lambda job: stage(job, "smoke"), max_parallel)
    for job in manifest["jobs"]:
        reference = references[(job["model"], job["mode"])]
        if job is not reference:
            job.update(smoke_status=reference["smoke_status"], smoke_log=reference["smoke_log"],
                       smoke_reference=reference["id"])
    if any(job["smoke_status"] != "passed" for job in smoke_jobs):
        for job in manifest["jobs"]:
            if job["status"] == "pending":
                job.update(status="blocked", error="A planned model/mode failed preflight")
        manifest["status"] = "failed"
        save()
        return suite
    if check_only:
        for job in manifest["jobs"]:
            job["status"] = "checked"
        manifest.update(status="checked", report_status="not_requested")
        save()
        return suite
    manifest["status"] = "training"
    save()
    _parallel_by_device(manifest["jobs"], lambda job: stage(job, "train"), max_parallel)
    completed = sum(job["status"] == "completed" for job in manifest["jobs"])
    manifest["status"] = "completed" if completed == len(manifest["jobs"]) else ("partial" if completed else "failed")
    save()
    if completed:
        first = manifest["jobs"][0]
        command = [plan["conda_executable"], "run", "--no-capture-output", "-n", first["env"],
                   "python", "-m", "foundation_sft.report", "--suite", str(suite)]
        try:
            _command(command, suite / "logs" / "report.log")
            if not (suite / "report" / "run_metrics.csv").is_file():
                raise ValueError("Comparison report was not produced")
            manifest.update(report_status="completed", report_dir=str(suite / "report"))
        except (OSError, ValueError, RuntimeError) as error:
            manifest.update(status="failed", report_status="failed", report_error=str(error))
        save()
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="foundation_sft/configs/experiments.json")
    parser.add_argument("--models", nargs="+", help="Run only these configured models")
    parser.add_argument("--modes", nargs="+", choices=("linear_probe", "lora"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--plan", action="store_true", help="Show resolved jobs without executing or writing files")
    parser.add_argument("--check-only", action="store_true", help="Prepare data and smoke all models; do not train")
    args = parser.parse_args(argv)
    plan = build_plan(args.config, args.models, args.modes, args.seeds)
    if args.plan:
        for job in plan["jobs"]:
            print(f"{job['id']}  environment={job['env']}  device={job['device']}")
        return 0
    suite = run_suite(plan, check_only=args.check_only, max_parallel=args.max_parallel)
    state = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    print(f"Workflow {state['status']}: {suite}", flush=True)
    if state.get("report_dir"):
        print(f"Comparison report: {state['report_dir']}", flush=True)
    return 0 if state["status"] in {"checked", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
