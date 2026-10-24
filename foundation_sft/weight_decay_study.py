"""Controlled weight-decay study for the completed random-student A/B batch.

Each arm delegates to the existing random-student coordinator. Planning and
saved-result reporting use only the standard library; the training recipe is
read from the verified baseline, with weight_decay as its only changed setting.
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
from . import weight_decay_baseline as baseline_api


FORMAT = "random_student_weight_decay_v1"
DEFAULT_CONFIG = "foundation_sft/configs/random_student_weight_decay.json"
SEEDS = [42]
WEIGHT_DECAYS = [0.001, 0.01]
SELECTION_RULE = {
    "candidate_sets": "baseline and both declared weight-decay arms",
    "condition": "B",
    "metric": "best validation patient ROC-AUC for seed42",
    "requires_complete_study": True,
    "ties": "baseline first, then smaller weight_decay",
    "test_scores_used": False,
    "paired_A_uses_same_weight_decay": True,
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _runtime(settings, project, devices=None, max_parallel=None, cache_dir=None):
    chosen = list(devices if devices is not None else settings.get("devices", ["cuda:0"]))
    if (not chosen or any(not isinstance(value, str) or not value for value in chosen)
            or len(set(chosen)) != len(chosen)):
        raise ValueError("devices must contain unique nonempty names")
    maximum = max_parallel if max_parallel is not None else settings.get("max_parallel", 1)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_parallel must be a positive integer")
    cache = cache_dir if cache_dir is not None else settings.get("cache_dir")
    if cache is not None:
        cache = str(random.legacy._path(cache, project))
    return chosen, maximum, cache


def _arm_id(weight_decay):
    return "wd_" + format(weight_decay, ".12g").replace(".", "p")


def _inner_plan(settings, project, config_path):
    """Mirror the two-job plan without writing a temporary JSON during --plan."""
    jobs = []
    for group, (mode, condition) in random.GROUPS.items():
        for seed in SEEDS:
            device = settings["devices"][len(jobs) % len(settings["devices"])]
            config = {"teacher_suite": settings.get("teacher_suite"),
                "teacher_checkpoint": settings["teacher_checkpoint"], "student_checkpoint": None,
                "student_initialization": "random", "student": {"mode": mode},
                "seed": seed, "device": device, "training": deepcopy(settings["training"]),
                "distillation": deepcopy(settings["distillation"]),
                "output_root": settings["output_root"], "cache_dir": settings.get("cache_dir")}
            jobs.append({"id": f"{group}_seed{seed}", "group": group, "seed": seed,
                "student_mode": mode, "condition": condition, "device": device,
                "initialization": "random", "config": config})
    return {"format_version": random.FORMAT, "project": str(project),
        "config_path": str(config_path), "output_root": settings["output_root"],
        "max_parallel": settings["max_parallel"], "groups": ["A", "B"],
        "seeds": list(SEEDS), "devices": list(settings["devices"]),
        "initialization": "random", "jobs": jobs}


def make_plan(config_path, project, *, devices=None, max_parallel=None,
              cache_dir=None, baseline_loader=None):
    project = Path(project).resolve()
    config_path = random.legacy._path(config_path, project)
    settings = random.legacy._read(config_path)
    seeds = settings.get("seeds", list(SEEDS))
    if (not isinstance(seeds, list) or seeds != SEEDS
            or any(type(seed) is not int for seed in seeds)):
        raise ValueError("This confirmed screening study requires seeds=[42]")
    declared = settings.get("weight_decays")
    if (not isinstance(declared, list) or len(declared) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in declared)
            or declared != WEIGHT_DECAYS):
        raise ValueError("This confirmed study requires weight_decays=[0.001, 0.01]")
    loader = baseline_loader or baseline_api.load_baseline
    baseline = loader(random.legacy._path(settings["baseline_suite"], project), project)
    if baseline["weight_decay"] in declared:
        raise ValueError("A declared weight decay duplicates the actual baseline; revise the study design")
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    first = baseline["jobs"]["A_seed42"]
    source = first["config"]
    teacher = source.get("teacher_checkpoint") or first["check"]["teacher"].get("checkpoint")
    if not isinstance(teacher, str) or not teacher:
        raise ValueError("The baseline must name the exact local teacher checkpoint")
    if cache is None and source.get("cache_dir") is not None:
        cache = str(random.legacy._path(source["cache_dir"], project))
    output = random.legacy._path(settings.get("output_root", "runs/random_student_weight_decay"), project)
    if output == Path(baseline["suite"]) or output.is_relative_to(Path(baseline["suite"])):
        raise ValueError("The weight-decay study must not write inside the baseline suite")
    arms, jobs = {}, []
    for weight_decay in declared:
        arm_id = _arm_id(weight_decay)
        train = deepcopy(source["training"])
        train["weight_decay"] = weight_decay
        config = {"teacher_suite": source.get("teacher_suite"),
            "teacher_checkpoint": str(random.legacy._path(teacher, project)),
            "student_initialization": "random", "student_checkpoint": None,
            "groups": ["A", "B"], "seeds": list(SEEDS),
            "devices": chosen, "max_parallel": maximum, "cache_dir": cache,
            "output_root": str(output / "unmaterialized" / arm_id),
            "training": train, "distillation": deepcopy(source["distillation"])}
        inner = _inner_plan(config, project, output / "unmaterialized" / f"{arm_id}.json")
        arms[arm_id] = {"weight_decay": weight_decay, "settings": config, "plan": inner}
        jobs.extend({"arm": arm_id, "weight_decay": weight_decay, **deepcopy(job)} for job in inner["jobs"])
    return {"format_version": FORMAT, "project": str(project), "config_path": str(config_path),
        "output_root": str(output), "devices": chosen, "max_parallel": maximum,
        "cache_dir": cache, "baseline": baseline, "weight_decays": list(declared), "seeds": list(SEEDS),
        "selection_rule": deepcopy(SELECTION_RULE), "arms": arms, "jobs": jobs}


def protocol_hash(plan):
    return random.legacy._hash({"format_version": FORMAT, "project": plan["project"],
        "baseline": plan["baseline"], "weight_decays": plan["weight_decays"], "seeds": plan["seeds"],
        "selection_rule": plan["selection_rule"],
        "arms": {name: {"weight_decay": arm["weight_decay"],
            "protocol_sha256": random.protocol_hash(arm["plan"])} for name, arm in plan["arms"].items()}})


@contextmanager
def _study_lock(study):
    # POSIX flock releases automatically after a process exits or SSH disconnects.
    import fcntl

    with (Path(study) / ".execution.lock").open("a+b") as handle:
        os.set_inheritable(handle.fileno(), False)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This weight-decay study is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _materialize(study, plan, state):
    """Pin every child-suite path before starting any subprocess."""
    for arm_id, planned in plan["arms"].items():
        settings = deepcopy(planned["settings"])
        settings["output_root"] = str(study / "arms" / arm_id / "outputs")
        config_path = study / "configs" / f"{arm_id}.json"
        suite = study / "arms" / arm_id / "suite"
        if config_path.exists():
            before = random.make_plan(config_path, Path(plan["project"]))
            random.legacy._same("saved arm scientific protocol", random.protocol_hash(before),
                                random.protocol_hash(planned["plan"]))
        random.legacy._json(config_path, settings)
        actual = random.make_plan(config_path, Path(plan["project"]))
        random.legacy._same("materialized scientific protocol", random.protocol_hash(actual),
                            random.protocol_hash(planned["plan"]))
        record = state["arms"].setdefault(arm_id, {"weight_decay": planned["weight_decay"], "status": "pending"})
        random.legacy._same("pinned arm weight_decay", record.get("weight_decay"), planned["weight_decay"])
        for field, value in (("config_path", str(config_path)), ("suite", str(suite))):
            if field in record:
                random.legacy._same("pinned arm " + field, record[field], value)
            record[field] = value
        if not (suite / "suite.json").exists():
            if record["status"] != "pending":
                raise ValueError("A previously started arm is missing its pinned child manifest")
            random.legacy._json(suite / "suite.json", {"format_version": random.FORMAT,
                "protocol_sha256": random.protocol_hash(actual), "status": "pending",
                "initialization": "random", "jobs": {job["id"]: {
                    "job": deepcopy(job), "status": "pending", "attempts": []} for job in actual["jobs"]}})
        record["plan"] = actual


def _report(study, state):
    from .weight_decay_report import print_metrics, write_report
    write_report(study, state)
    print_metrics(study, state)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None,
              verifier=None, baseline_verifier=None, reporter=None):
    project = Path(plan["project"])
    signature = protocol_hash(plan)
    if resume_study:
        study = random.legacy._path(resume_study, project)
        if not study.is_dir():
            raise ValueError("The requested study directory does not exist")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        study = Path(plan["output_root"]) / "studies" / f"study_{stamp}_{uuid.uuid4().hex[:8]}"
        study.mkdir(parents=True)
    with _study_lock(study):
        if resume_study:
            state = random.legacy._read(study / "study.json")
            if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
                raise ValueError("Resume differs from the frozen weight-decay study protocol")
            if set(state.get("arms", {})) != set(plan["arms"]):
                raise ValueError("Resume arm set differs")
            random.legacy._same("baseline snapshot", state.get("baseline"), plan["baseline"])
            random.legacy._same("saved screening seeds", state.get("seeds"), plan["seeds"])
            random.legacy._same("saved weight-decay candidates", state.get("weight_decays"), plan["weight_decays"])
            random.legacy._same("saved validation selection rule", state.get("selection_rule"), plan["selection_rule"])
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature,
                "baseline": deepcopy(plan["baseline"]), "weight_decays": list(plan["weight_decays"]),
                "seeds": list(plan["seeds"]),
                "selection_rule": deepcopy(plan["selection_rule"]), "arms": {}, "status": "pending"}
        # Verify source artifacts before creating child plans or performing GPU work.
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
        print(f"Actual baseline weight_decay: {plan['baseline']['weight_decay']}; "
              "4 planned new trainings; random initialization; A/B seed42", flush=True)
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
                print(f"[weight-decay arm] {arm_id} weight_decay={record['weight_decay']}", flush=True)
                try:
                    child = random.run_suite(record["plan"], check_only=check_only,
                        resume_suite=record["suite"], launcher=audited_launch, verifier=verifier)
                    random.legacy._same("child suite path", str(child), record["suite"])
                    record["status"] = random.legacy._read(child / "suite.json")["status"]
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                    record.update(status="failed", error=str(error))
                    print(f"[weight-decay arm failed] {arm_id}: {error}", flush=True)
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
        print(f"Weight-decay study {state['status']}: {study}", flush=True)
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
    parser.add_argument("--study", help="Read only this saved study with --show-metrics")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics:
            if args.resume_study:
                raise ValueError("--resume-study cannot be used with --show-metrics")
            settings = random.legacy._read(random.legacy._path(args.config, project))
            batches = saved_studies(random.legacy._path(settings.get("output_root", "runs/random_student_weight_decay"), project),
                                   random.legacy._path(args.study, project) if args.study else None)
            if not batches:
                print("No saved weight-decay studies found.")
            from .weight_decay_report import print_metrics
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
        print(f"Weight-decay study failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
