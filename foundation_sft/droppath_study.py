"""Controlled two-job DropPath screening at fixed WD and random initialization.

The completed weight_decay=0.01 A/B seed42 runs are the frozen control.
Only training.drop_path_rate changes; native stochastic depth is train-only,
with a linear 12-block schedule and a separate saved Torch RNG stream.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import uuid

from . import random_student_suite as random
from . import droppath_baseline as baseline_api
from .weight_decay_study import _inner_plan, _runtime


FORMAT = "random_student_droppath_v1"
DEFAULT_CONFIG = "foundation_sft/configs/random_student_droppath.json"
SEEDS = [42]
DROP_PATH_RATE = 0.1
WEIGHT_DECAY = 0.01
ARM_ID = "drop_path_0p1"
SELECTION_RULE = {
    "candidate_sets": "frozen DropPath=0 control and DropPath=0.1 candidate",
    "condition": "B",
    "metric": "best validation patient ROC-AUC for seed42",
    "requires_complete_study": True,
    "ties": "baseline first",
    "test_scores_used": False,
    "paired_A_uses_same_drop_path_rate": True,
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _confirmed_number(value, expected, label):
    if type(value) not in (float, int) or not math.isfinite(value) or value != expected:
        raise ValueError(f"This confirmed study requires {label}={expected}")


def make_plan(config_path, project, *, devices=None, max_parallel=None,
              cache_dir=None, baseline_loader=None):
    """Resolve the two paired jobs without importing Torch or creating files."""
    project = Path(project).resolve()
    config_path = random.legacy._path(config_path, project)
    settings = random.legacy._read(config_path)
    seeds = settings.get("seeds", list(SEEDS))
    if not isinstance(seeds, list) or seeds != SEEDS or any(type(seed) is not int for seed in seeds):
        raise ValueError("This confirmed DropPath screening requires seeds=[42]")
    _confirmed_number(settings.get("drop_path_rate"), DROP_PATH_RATE, "drop_path_rate")
    _confirmed_number(settings.get("weight_decay"), WEIGHT_DECAY, "weight_decay")
    loader = baseline_loader or baseline_api.load_baseline
    baseline = loader(random.legacy._path(settings["baseline_study"], project), project)
    _confirmed_number(baseline["weight_decay"], WEIGHT_DECAY, "baseline weight_decay")
    _confirmed_number(baseline["drop_path_rate"], 0.0, "baseline drop_path_rate")
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    first = baseline["jobs"]["A_seed42"]
    source = first["config"]
    teacher = source.get("teacher_checkpoint") or first["check"]["teacher"].get("checkpoint")
    if not isinstance(teacher, str) or not teacher:
        raise ValueError("The baseline must name the exact local teacher checkpoint")
    if cache is None and source.get("cache_dir") is not None:
        cache = str(random.legacy._path(source["cache_dir"], project))
    output = random.legacy._path(settings.get("output_root", "runs/random_student_droppath"), project)
    for origin in (Path(baseline["study"]), Path(baseline["suite"])):
        if output == origin or output.is_relative_to(origin):
            raise ValueError("The DropPath study must not write inside its frozen source study")
    train = deepcopy(source["training"])
    _confirmed_number(train.get("weight_decay"), WEIGHT_DECAY, "effective source weight_decay")
    _confirmed_number(train.get("drop_path_rate", 0.0), 0.0, "effective source drop_path_rate")
    train["drop_path_rate"] = DROP_PATH_RATE
    config = {"teacher_suite": source.get("teacher_suite"),
        "teacher_checkpoint": str(random.legacy._path(teacher, project)),
        "student_initialization": "random", "student_checkpoint": None,
        "groups": ["A", "B"], "seeds": list(SEEDS),
        "devices": chosen, "max_parallel": maximum, "cache_dir": cache,
        "output_root": str(output / "unmaterialized" / ARM_ID),
        "training": train, "distillation": deepcopy(source["distillation"])}
    inner = _inner_plan(config, project, output / "unmaterialized" / (ARM_ID + ".json"))
    arms = {ARM_ID: {"drop_path_rate": DROP_PATH_RATE, "settings": config, "plan": inner}}
    return {"format_version": FORMAT, "project": str(project), "config_path": str(config_path),
        "output_root": str(output), "devices": chosen, "max_parallel": maximum, "cache_dir": cache,
        "baseline": baseline, "seeds": list(SEEDS), "drop_path_rate": DROP_PATH_RATE,
        "weight_decay": WEIGHT_DECAY, "selection_rule": deepcopy(SELECTION_RULE), "arms": arms,
        "jobs": [{"arm": ARM_ID, "drop_path_rate": DROP_PATH_RATE,
                  "weight_decay": WEIGHT_DECAY, **deepcopy(job)} for job in inner["jobs"]]}


def protocol_hash(plan):
    return random.legacy._hash({"format_version": FORMAT, "project": plan["project"],
        "baseline": plan["baseline"], "seeds": plan["seeds"],
        "drop_path_rate": plan["drop_path_rate"], "weight_decay": plan["weight_decay"],
        "selection_rule": plan["selection_rule"],
        "arms": {name: {"drop_path_rate": arm["drop_path_rate"],
            "protocol_sha256": random.protocol_hash(arm["plan"])} for name, arm in plan["arms"].items()}})


@contextmanager
def _study_lock(study):
    import fcntl
    with (Path(study) / ".execution.lock").open("a+b") as handle:
        os.set_inheritable(handle.fileno(), False)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This DropPath study is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _materialize(study, plan, state):
    for arm_id, planned in plan["arms"].items():
        settings = deepcopy(planned["settings"])
        settings["output_root"] = str(study / "arms" / arm_id / "outputs")
        config_path = study / "configs" / (arm_id + ".json")
        suite = study / "arms" / arm_id / "suite"
        if config_path.exists():
            saved = random.make_plan(config_path, Path(plan["project"]))
            random.legacy._same("saved DropPath arm scientific protocol", random.protocol_hash(saved),
                                random.protocol_hash(planned["plan"]))
        random.legacy._json(config_path, settings)
        actual = random.make_plan(config_path, Path(plan["project"]))
        random.legacy._same("materialized DropPath protocol", random.protocol_hash(actual),
                            random.protocol_hash(planned["plan"]))
        record = state["arms"].setdefault(arm_id, {"drop_path_rate": planned["drop_path_rate"], "status": "pending"})
        random.legacy._same("pinned arm drop_path_rate", record.get("drop_path_rate"), planned["drop_path_rate"])
        for field, value in (("config_path", str(config_path)), ("suite", str(suite))):
            if field in record:
                random.legacy._same("pinned arm " + field, record[field], value)
            record[field] = value
        if not (suite / "suite.json").exists():
            if record["status"] != "pending":
                raise ValueError("A previously started arm is missing its pinned child manifest")
            random.legacy._json(suite / "suite.json", {"format_version": random.FORMAT,
                "protocol_sha256": random.protocol_hash(actual), "status": "pending", "initialization": "random",
                "jobs": {job["id"]: {"job": deepcopy(job), "status": "pending", "attempts": []}
                         for job in actual["jobs"]}})
        record["plan"] = actual


def _report(study, state):
    from .droppath_report import print_metrics, write_report
    write_report(study, state)
    print_metrics(study, state)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None,
              verifier=None, baseline_verifier=None, reporter=None):
    project, signature = Path(plan["project"]), protocol_hash(plan)
    if resume_study:
        study = random.legacy._path(resume_study, project)
        if not study.is_dir():
            raise ValueError("The requested DropPath study directory does not exist")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        study = Path(plan["output_root"]) / "studies" / f"study_{stamp}_{uuid.uuid4().hex[:8]}"
        study.mkdir(parents=True)
    with _study_lock(study):
        if resume_study:
            state = random.legacy._read(study / "study.json")
            if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
                raise ValueError("Resume differs from the frozen DropPath scientific protocol")
            if set(state.get("arms", {})) != set(plan["arms"]):
                raise ValueError("Resume DropPath arm set differs")
            for name in ("baseline", "seeds", "drop_path_rate", "weight_decay", "selection_rule"):
                random.legacy._same("saved DropPath " + name, state.get(name), plan[name])
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature, "arms": {}, "status": "pending",
                     **{name: deepcopy(plan[name]) for name in
                        ("baseline", "seeds", "drop_path_rate", "weight_decay", "selection_rule")}}
        baseline_api.validate_snapshot(plan["baseline"], verify_runs=True, verifier=baseline_verifier)
        _materialize(study, plan, state)
        state.update(status="running", plan=deepcopy(plan), check_only=bool(check_only))
        execution = {"started_at": _now(), "devices": plan["devices"],
                     "max_parallel": plan["max_parallel"], "check_only": bool(check_only)}
        state.setdefault("execution_history", []).append(execution)

        def persist():
            state["updated_at"] = _now()
            random.legacy._json(study / "study.json", state)

        persist()
        print(f"Study directory: {study}", flush=True)
        print("2 planned new trainings; A/B seed42; weight_decay=0.01; "
              "DropPath max=0.1, linear 12-block schedule, isolated RNG", flush=True)
        launch = launcher or random._launch

        def audited_launch(job, action, child_project, directory):
            value = launch(job, action, child_project, directory)
            if action == "check":
                baseline_api.audit_check(value, job, plan["baseline"])
            return value

        try:
            for arm_id, record in state["arms"].items():
                baseline_api.validate_snapshot(plan["baseline"])
                record.update(status="running", started_at=_now())
                record.pop("error", None)
                persist()
                print(f"[DropPath arm] {arm_id} max_drop_path_rate={record['drop_path_rate']}", flush=True)
                try:
                    child = random.run_suite(record["plan"], check_only=check_only,
                        resume_suite=record["suite"], launcher=audited_launch, verifier=verifier)
                    random.legacy._same("DropPath child suite path", str(child), record["suite"])
                    record["status"] = random.legacy._read(child / "suite.json")["status"]
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                    record.update(status="failed", error=str(error))
                    print(f"[DropPath arm failed] {arm_id}: {error}", flush=True)
                record["finished_at"] = _now()
                persist()
            statuses = [record["status"] for record in state["arms"].values()]
            if all(value == "completed" for value in statuses):
                state["status"] = "completed"
            elif check_only and all(value in {"checked", "completed"} for value in statuses):
                state["status"] = "checked"
            else:
                state["status"] = "partial" if any(value in {"completed", "checked", "partial"} for value in statuses) else "failed"
            baseline_api.validate_snapshot(plan["baseline"])
            execution.update(finished_at=_now(), status=state["status"])
            persist()
        except BaseException as error:
            state.update(status="interrupted", error=str(error))
            execution.update(finished_at=_now(), status="interrupted")
            persist()
            raise
        (reporter or _report)(study, state)
        print(f"DropPath study {state['status']}: {study}", flush=True)
    return study


def saved_studies(output_root, study=None):
    paths = [Path(study)] if study else sorted((Path(output_root) / "studies").glob("study_*"))
    return [(path, state) for path in paths if (path / "study.json").is_file()
            and (state := random.legacy._read(path / "study.json")).get("format_version") == FORMAT]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--show-metrics", action="store_true")
    parser.add_argument("--resume-study")
    parser.add_argument("--study", help="Read only this saved DropPath study with --show-metrics")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics:
            if args.resume_study:
                raise ValueError("--resume-study cannot be used with --show-metrics")
            settings = random.legacy._read(random.legacy._path(args.config, project))
            batches = saved_studies(random.legacy._path(settings.get("output_root", "runs/random_student_droppath"), project),
                                   random.legacy._path(args.study, project) if args.study else None)
            if not batches:
                print("No saved DropPath studies found.")
                return 0
            from .droppath_report import print_metrics
            for study, state in batches:
                print_metrics(study, state)
            return 0
        if args.study:
            raise ValueError("--study requires --show-metrics; use --resume-study to continue")
        plan = make_plan(args.config, project, devices=args.devices,
            max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        study = run_study(plan, resume_study=args.resume_study, check_only=args.check_only)
        return 1 if random.legacy._read(study / "study.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"DropPath study failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
