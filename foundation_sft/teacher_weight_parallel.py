"""Run one pending formal group beside a live, older study coordinator.

The study coordinator remains the only writer of study.json. A per-group lease
serializes this helper and future workers spawned by that coordinator. Results
are handed back through the original worker receipt, after full verification.
Planning, status and waiting for a lease use only the standard library.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import uuid

from . import distillation_suite as records
from . import teacher_weight_study as control


FORMAT = "teacher_weight_parallel_v1"
GROUPS = ("R-C", "R-R", "P-C", "P-R")
REGISTRY = "parallel_registry.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _signature(pid):
    try:
        path = Path("/proc") / str(pid)
        if path.stat().st_uid != os.getuid():
            return None
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def make_plan(study, group, device="cuda:3", project=None, *, allow_running=False):
    """Read the frozen protocol and one saved preflight without writing files."""
    study = Path(study).expanduser().resolve()
    state = records._read(study / "study.json")
    if state.get("format_version") != control.FORMAT:
        raise ValueError("Not a teacher-weight study")
    saved_plan = state.get("plan", {})
    project = Path(project or saved_plan["project"]).expanduser().resolve()
    records._same("parallel project", str(project), saved_plan.get("project"))
    records._same("parallel frozen protocol", state.get("protocol_sha256"), control.protocol_hash(saved_plan))
    if group not in GROUPS:
        raise ValueError("Only formal CA/RL groups may be launched independently")
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Declare one CUDA device, for example cuda:3")
    selected = control.select_learning_rates(state)
    records._same("parallel frozen LR selection", selected, state.get("selection"))
    initialization = "random" if group.startswith("R-") else "imagenet"
    if selected.get(initialization, {}).get("status") != "selected":
        raise ValueError("All three train/validation pilots must be complete before launching a formal group")
    record = state.get("main", {}).get(group)
    if not isinstance(record, dict) or not isinstance(record.get("job"), dict):
        raise ValueError("The coordinator has not materialized this formal group")
    if record.get("status") in {"checking", "training"} and not allow_running:
        raise ValueError("This group is already running; keep its current worker")
    expected = next((job for job in control._formal_jobs(state, saved_plan) if job["id"] == group), None)
    if expected is None:
        raise ValueError("Formal group is absent from the frozen LR selection")
    job = deepcopy(record["job"])
    records._same("parallel exact scientific recipe", control._scientific_config(job["config"]),
                  control._scientific_config(expected["config"]))
    for key in ("id", "group", "student_initialization", "method", "phase", "seed", "action"):
        records._same("parallel job " + key, job.get(key), expected.get(key))
    if not isinstance(record.get("check"), dict):
        raise ValueError("The coordinator must finish this group's GPU preflight first")
    check = control._checked(deepcopy(record["check"]), job)
    checks = [deepcopy(item["check"]) for section in (state["pilots"], state["main"])
              for item in section.values() if isinstance(item.get("check"), dict)]
    control._audit_checks(check, checks)
    job["device"] = device
    job["config"]["device"] = device
    return {"format_version": FORMAT, "study": str(study), "project": str(project),
            "group": group, "job": job, "check": check, "all_checks": checks,
            "protocol_sha256": state["protocol_sha256"],
            "saved_status": record.get("status"), "saved_result": deepcopy(record.get("result")),
            "legacy_attempts": deepcopy(record.get("attempts", []))}


def _job_hash(plan):
    return records._hash({"protocol_sha256": plan["protocol_sha256"], "group": plan["group"],
                          "config": control._scientific_config(plan["job"]["config"])})


def _job_directory(plan):
    return Path(plan["study"]) / "jobs" / plan["group"]


@contextmanager
def _process_marker(plan):
    directory = _job_directory(plan)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (".parallel_guard_" + str(os.getpid()) + ".json")
    records._json(path, {"format_version": FORMAT, "pid": os.getpid(),
        "starttime": _signature(os.getpid()), "group": plan["group"],
        "study": plan["study"], "job_sha256": _job_hash(plan)})
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _job_lock(plan):
    path = _job_directory(plan) / ".parallel_execution.lock"
    with path.open("a+b") as handle:
        os.set_inheritable(handle.fileno(), False)
        # DataLoader fork children must not keep the parent's lease alive after
        # it exits. Closing the fork child's file object leaves the parent lock.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=lambda: handle.close())
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[train] parallel waiting: {plan['group']} already has a guarded worker; no second training starts", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def active_unguarded_workers(study, group):
    """Find older entry-point workers that cannot participate in the lease."""
    study = Path(study).resolve()
    prefix = str(study / "jobs" / group) + "/"
    state = records._read(study / "study.json")
    expected_hash = records._hash({"protocol_sha256": state["protocol_sha256"], "group": group,
        "config": control._scientific_config(state["main"][group]["job"]["config"])})
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        pid = int(directory.name)
        signature = _signature(pid)
        if signature is None:
            continue
        try:
            args = [part.decode() for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        except (OSError, UnicodeError):
            continue
        named = {Path(arg).name for arg in args}
        matching = False
        if "run_teacher_weight_training.py" in named and "--config" in args:
            index = args.index("--config") + 1
            matching = index < len(args) and args[index].startswith(prefix)
            if "--action" in args:
                index = args.index("--action") + 1
                matching = matching and index < len(args) and args[index] == "train"
        if "run_teacher_weight_parallel.py" in named and "--study" in args and "--group" in args:
            left, right = args.index("--study") + 1, args.index("--group") + 1
            matching = (left < len(args) and right < len(args)
                        and str(Path(args[left]).resolve()) == str(study) and args[right] == group)
        if not matching:
            continue
        marker = study / "jobs" / group / (".parallel_guard_" + str(pid) + ".json")
        try:
            guarded = records._read(marker)
            if (guarded.get("format_version") == FORMAT and guarded.get("starttime") == signature
                    and guarded.get("pid") == pid and guarded.get("study") == str(study)
                    and guarded.get("group") == group and guarded.get("job_sha256") == expected_hash):
                continue
        except (OSError, ValueError):
            pass
        result.append({"pid": pid, "args": args})
    return result


def _default_check(config, project):
    from .teacher_weight_training import check
    directory = check(config, project)
    value = records._read(directory / "check.json")
    value.update(action="check", directory=str(directory))
    return value


def _default_train(config, project):
    from .teacher_weight_training import run_training
    return run_training(config, project)


def _default_verify(run, check, job):
    from .teacher_weight_training import verify_completed_run
    return verify_completed_run(run, check, job)


def validate_available_device(device, min_free_mb=20000, max_util=5):
    """Recheck the requested physical card before the independent GPU check."""
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Declare one CUDA device, for example cuda:3")
    try:
        output = subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        cards = {int(row.split(",")[0]): tuple(map(int, row.split(",")[1:]))
                 for row in output.splitlines() if row.strip()}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("Cannot verify the requested free GPU; the current study was left running") from error
    index = int(device[5:])
    if index not in cards or len(cards[index]) != 2:
        raise ValueError("The requested GPU is not present; no independent training was started")
    free_mb, utilization = cards[index]
    if free_mb < min_free_mb or utilization > max_util:
        raise ValueError(f"{device} is busy (free={free_mb} MiB, utilization={utilization}%); keep the current study running")
    return {"device": device, "free_mb": free_mb, "utilization_percent": utilization}


def _contained_run(run, plan):
    path = Path(run).expanduser().resolve()
    try:
        path.relative_to(_job_directory(plan).resolve())
    except ValueError as error:
        raise ValueError("Parallel receipt run is outside its exact study group") from error
    return path


def _verify_receipt(receipt, check, plan, verifier):
    if not isinstance(receipt, dict) or receipt.get("status") != "completed" or not receipt.get("run"):
        raise ValueError("Parallel receipt does not identify a completed run")
    run = _contained_run(receipt["run"], plan)
    verified = verifier(str(run), check, plan["job"])
    records._same("parallel complete receipt", verified, receipt)
    fingerprint = verified.get("checkpoint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Parallel result needs its checkpoint fingerprint")
    return verified


def _load_completed(plan, verifier):
    pointer = _job_directory(plan) / REGISTRY
    if not pointer.exists():
        return None
    registry = records._read(pointer)
    records._same("parallel registry version", registry.get("format_version"), FORMAT)
    records._same("parallel registry recipe", registry.get("job_sha256"), _job_hash(plan))
    attempt = Path(registry.get("attempt", "")).resolve()
    if attempt.parent != _job_directory(plan).resolve() or not attempt.name.startswith("parallel_"):
        raise ValueError("Parallel registry attempt is outside its study group")
    receipt_path = attempt / "receipt.json"
    if registry.get("status") != "completed" and not receipt_path.exists():
        return None
    check = records._read(attempt / "check.json")
    check = control._checked(check, plan["job"])
    control._audit_checks(check, plan["all_checks"])
    receipt = records._read(receipt_path)
    if registry.get("status") == "completed":
        records._same("parallel receipt bytes", registry.get("receipt_sha256"), records._file_hash(receipt_path))
        records._same("parallel pinned checkpoint", registry.get("checkpoint_sha256"), receipt.get("checkpoint_sha256"))
    result = _verify_receipt(receipt, plan["check"], plan, verifier)
    if registry.get("status") != "completed":
        registry.update(status="completed", checkpoint_sha256=result["checkpoint_sha256"],
                        receipt_sha256=records._file_hash(receipt_path), recovered_at=_now())
        records._json(pointer, registry)
    return result


def _legacy_completed(plan, verifier):
    """Recover a receipt written just before its old coordinator records it."""
    for attempt in plan.get("legacy_attempts", []):
        if attempt.get("action") != "train":
            continue
        directory = Path(attempt.get("directory", "")).expanduser().resolve()
        if directory.parent != _job_directory(plan).resolve() or not directory.name.startswith("train_"):
            raise ValueError("Legacy train attempt is outside its exact study group")
        receipt_path = directory / "result.json"
        if receipt_path.is_file():
            return _verify_receipt(records._read(receipt_path), plan["check"], plan, verifier)
    return None


def _publish_adoption(plan, result, origin):
    attempt = _job_directory(plan) / ("parallel_" + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    records._json(attempt / "config.json", plan["job"]["config"])
    records._json(attempt / "check.json", plan["check"])
    records._json(attempt / "receipt.json", result)
    registry = {"format_version": FORMAT, "group": plan["group"], "job_sha256": _job_hash(plan),
        "protocol_sha256": plan["protocol_sha256"], "attempt": str(attempt), "status": "completed",
        "origin": origin, "optimization_performed": False, "adopted_at": _now(), "run": result["run"],
        "checkpoint_sha256": result["checkpoint_sha256"], "receipt_sha256": records._file_hash(attempt / "receipt.json")}
    records._json(attempt / "state.json", registry)
    records._json(_job_directory(plan) / REGISTRY, registry)


def run_parallel(plan, *, origin="parallel", check_runner=None, train_runner=None,
                 verifier=None, process_scanner=None, result_file=None):
    """Train or adopt one group, serializing helper and coordinator workers.

    Injectable callbacks are used by CPU and real-process lease tests. This
    function never updates study.json, nor pauses or signals another process.
    """
    if origin not in {"parallel", "coordinator"}:
        raise ValueError("Declare origin=parallel or coordinator")
    fresh = make_plan(plan["study"], plan["group"], plan["job"]["config"]["device"],
                      plan["project"], allow_running=True)
    records._same("parallel launch protocol", fresh["protocol_sha256"], plan["protocol_sha256"])
    records._same("parallel launch recipe", _job_hash(fresh), _job_hash(plan))
    check_runner, train_runner, verifier = (check_runner or _default_check,
        train_runner or _default_train, verifier or _default_verify)
    scanner = process_scanner or active_unguarded_workers
    with _process_marker(plan), _job_lock(plan):
        # The live parent can finish its last step while we wait for the lease.
        # Read its new atomic snapshot before looking for an unrecorded result.
        fresh = make_plan(plan["study"], plan["group"], plan["job"]["config"]["device"],
                          plan["project"], allow_running=True)
        records._same("locked parallel launch recipe", _job_hash(fresh), _job_hash(plan))
        control._audit_checks(plan["check"], fresh["all_checks"])
        active = scanner(plan["study"], plan["group"])
        if active:
            raise ValueError(f"This group already has an unguarded live training worker; keep it running: {active}")
        result = _load_completed(plan, verifier)
        if result is None and fresh.get("saved_result"):
            result = _verify_receipt(fresh["saved_result"], plan["check"], plan, verifier)
            _publish_adoption(plan, result, "recorded_legacy_result")
        if result is None:
            result = _legacy_completed(fresh, verifier)
            if result is not None:
                _publish_adoption(plan, result, "unrecorded_legacy_receipt")
        if result is not None:
            print(f"[train] parallel adopted: {plan['group']} {result['run']}; optimization is not repeated", flush=True)
            if result_file:
                records._json(result_file, result)
                records._json(Path(result_file).parent / "parallel_adoption.json", {
                    "format_version": FORMAT, "group": plan["group"], "origin": origin,
                    "checkpoint_sha256": result["checkpoint_sha256"], "run": result["run"], "adopted_at": _now()})
            return deepcopy(result)
        attempt = _job_directory(plan) / ("parallel_" + uuid.uuid4().hex[:12])
        attempt.mkdir(parents=True, exist_ok=False)
        config = deepcopy(plan["job"]["config"])
        config["output_root"] = str(attempt / "outputs")
        records._json(attempt / "config.json", config)
        registry = {"format_version": FORMAT, "group": plan["group"], "job_sha256": _job_hash(plan),
            "protocol_sha256": plan["protocol_sha256"], "attempt": str(attempt), "status": "running",
            "origin": origin, "pid": os.getpid(), "device": config["device"], "started_at": _now()}
        records._json(attempt / "state.json", registry)
        records._json(_job_directory(plan) / REGISTRY, registry)
        try:
            print(f"[train] parallel preflight: {plan['group']} {config['device']} attempt={attempt}", flush=True)
            checked = control._checked(check_runner(config, Path(plan["project"])), plan["job"])
            control._audit_checks(checked, plan["all_checks"])
            records._json(attempt / "check.json", checked)
            receipt = train_runner(config, Path(plan["project"]))
            result = _verify_receipt(receipt, checked, plan, verifier)
            records._json(attempt / "receipt.json", result)
            registry.update(status="completed", finished_at=_now(), run=result["run"],
                checkpoint_sha256=result["checkpoint_sha256"], receipt_sha256=records._file_hash(attempt / "receipt.json"))
            records._json(attempt / "state.json", registry)
            records._json(_job_directory(plan) / REGISTRY, registry)
            if result_file:
                records._json(result_file, result)
            print(f"[train] parallel completed: {plan['group']} {result['run']}", flush=True)
            return deepcopy(result)
        except BaseException as error:
            registry.update(status="failed", finished_at=_now(), error=str(error))
            records._json(attempt / "state.json", registry)
            records._json(_job_directory(plan) / REGISTRY, registry)
            raise


def _guarded_location(config_path):
    path = Path(config_path).expanduser().resolve()
    if len(path.parents) < 4 or path.name != "config.json" or path.parents[2].name != "jobs":
        return None
    group, study = path.parents[1].name, path.parents[3]
    if group not in GROUPS or not (study / "study.json").is_file():
        return None
    return study, group


def worker_main(argv=None):
    """Compatibility entry used by future workers of an already-live parent."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config")
    parser.add_argument("--action")
    parser.add_argument("--result-json")
    options, _ = parser.parse_known_args(argv)
    project = Path(options.project).expanduser().resolve()
    path = records._path(options.config, project) if options.config else None
    location = _guarded_location(path) if path is not None else None
    if options.action != "train" or location is None:
        from .teacher_weight_training import main as original_main
        return original_main(argv)
    config = records._read(path)
    if config.get("phase") != "main" or config.get("method") not in {"ca", "rl"}:
        from .teacher_weight_training import main as original_main
        return original_main(argv)
    study, group = location
    plan = make_plan(study, group, config.get("device", "cuda:0"), project, allow_running=True)
    records._same("coordinator worker recipe", control._scientific_config(config),
                  control._scientific_config(plan["job"]["config"]))
    plan["job"]["config"] = deepcopy(config)
    result_file = records._path(options.result_json, project) if options.result_json else None
    return run_parallel(plan, origin="coordinator", result_file=result_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project")
    parser.add_argument("--study", required=True)
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument("--device", default="cuda:3")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.study, args.group, args.device, args.project, allow_running=args.status)
        if args.plan:
            import json
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if args.status:
            pointer = _job_directory(plan) / REGISTRY
            import json
            print(json.dumps({"study": plan["study"], "group": args.group,
                "coordinator_status": plan["saved_status"], "registry": records._read(pointer) if pointer.exists() else None,
                "active_unguarded_workers": active_unguarded_workers(plan["study"], args.group)}, ensure_ascii=False, indent=2))
            return 0
        validate_available_device(args.device)
        result = run_parallel(plan)
        print(f"Standalone result: {result['run']}", flush=True)
        print("The original coordinator will verify and adopt this result when it reaches this group.", flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Parallel teacher-weight worker failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
