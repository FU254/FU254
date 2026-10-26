"""One soft-label-only random-student training against frozen WD=0.01 controls.

The existing random-student suite owns preflight, training, checkpoint loading
and evaluation. This coordinator pins the completed A/B controls and permits
only the removal of global and local distillation from the new B run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import json
import sys
import uuid

from . import random_student_suite as random
from . import droppath_baseline as baseline_api
from .droppath_study import _study_lock
from .weight_decay_study import _inner_plan, _runtime
from .random_student_profiles import SOFT_ONLY, SOFT_ONLY_LOSS_SETTINGS


FORMAT = "random_student_soft_only_v1"
DEFAULT_CONFIG = "foundation_sft/configs/random_student_soft_only.json"
ARM_ID = "soft_only"
SELECTION_RULE = {
    "candidate_sets": "frozen full three-term KD and soft-label-only KD",
    "condition": "B", "metric": "best validation patient ROC-AUC for seed42",
    "requires_complete_study": True, "ties": "existing full KD first",
    "test_scores_used": False,
}
EMPTY_STATE_SHA256 = hashlib.sha256().hexdigest()


def _same(name, actual, expected):
    random.legacy._same("soft-only study " + name, actual, expected)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _inner_settings_plan(settings, project, path):
    # Reuse the original plan constructor, then retain just the confirmed B.
    plan = _inner_plan(settings, project, path)
    plan["groups"] = ["B"]
    plan["jobs"] = [job for job in plan["jobs"] if job["group"] == "B"]
    job = plan["jobs"][0]
    job["device"] = settings["devices"][0]
    job["config"]["device"] = job["device"]
    job["config"]["distillation_profile"] = SOFT_ONLY
    return plan


def make_plan(config_path, project, *, devices=None, max_parallel=None,
              cache_dir=None, baseline_loader=None):
    """Plan exactly one new B/seed42 job; planning never creates files."""
    project = Path(project).resolve()
    config_path = random.legacy._path(config_path, project)
    settings = random.legacy._read(config_path)
    if settings.get("seeds", [42]) != [42] or any(type(s) is not int for s in settings.get("seeds", [42])):
        raise ValueError("This confirmed soft-only screening requires seeds=[42]")
    for key, expected in (("weight_decay", .01), ("drop_path_rate", 0.0)):
        value = settings.get(key)
        if type(value) not in (float, int) or value != expected:
            raise ValueError(f"This confirmed soft-only screening requires {key}={expected}")
    _same("declared distillation profile", settings.get("distillation_profile", SOFT_ONLY), SOFT_ONLY)
    loader = baseline_loader or baseline_api.load_baseline
    baseline = loader(random.legacy._path(settings["baseline_study"], project), project)
    _same("source weight decay", baseline["weight_decay"], .01)
    _same("source DropPath", baseline["drop_path_rate"], 0.)
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    source = baseline["jobs"]["B_seed42"]["config"]
    teacher = source.get("teacher_checkpoint") or baseline["jobs"]["B_seed42"]["check"]["teacher"].get("checkpoint")
    if not isinstance(teacher, str) or not teacher:
        raise ValueError("The baseline must name the exact local teacher checkpoint")
    if cache is None and source.get("cache_dir") is not None:
        cache = str(random.legacy._path(source["cache_dir"], project))
    output = random.legacy._path(settings.get("output_root", "runs/random_student_soft_only"), project)
    for origin in (Path(baseline["study"]), Path(baseline["suite"])):
        if output == origin or output.is_relative_to(origin):
            raise ValueError("The soft-only study must not write inside its frozen source study")
    config = {"teacher_suite": source.get("teacher_suite"),
        "teacher_checkpoint": str(random.legacy._path(teacher, project)),
        "student_initialization": "random", "student_checkpoint": None,
        "distillation_profile": SOFT_ONLY, "groups": ["B"], "seeds": [42],
        "devices": chosen, "max_parallel": maximum, "cache_dir": cache,
        "output_root": str(output / "unmaterialized" / ARM_ID),
        "training": deepcopy(source["training"]),
        "distillation": deepcopy(SOFT_ONLY_LOSS_SETTINGS)}
    inner = _inner_settings_plan(config, project, output / "unmaterialized" / (ARM_ID + ".json"))
    return {"format_version": FORMAT, "project": str(project), "config_path": str(config_path),
        "output_root": str(output), "devices": chosen, "max_parallel": maximum,
        "cache_dir": cache, "baseline": baseline, "seeds": [42], "weight_decay": .01,
        "drop_path_rate": 0., "distillation_profile": SOFT_ONLY,
        "selection_rule": deepcopy(SELECTION_RULE),
        "candidate": {"settings": config, "plan": inner}, "jobs": deepcopy(inner["jobs"])}


def protocol_hash(plan):
    return random.legacy._hash({"format_version": FORMAT, "project": plan["project"],
        "baseline": plan["baseline"], "seeds": plan["seeds"],
        "weight_decay": plan["weight_decay"], "drop_path_rate": plan["drop_path_rate"],
        "distillation_profile": plan["distillation_profile"],
        "selection_rule": plan["selection_rule"],
        "candidate": random.protocol_hash(plan["candidate"]["plan"])})


def audit_check(value, job, baseline):
    check = random._checked(value, job)
    source = baseline["jobs"]["B_seed42"]
    for key in ("id", "group", "seed", "student_mode", "condition"):
        _same("candidate " + key, job.get(key), source["job"].get(key))
    _same("candidate profile", check["config"].get("distillation_profile"), SOFT_ONLY)
    _same("candidate loss", check["config"]["distillation"], SOFT_ONLY_LOSS_SETTINGS)
    _same("unchanged teacher/data/view/training/evaluation", random._cross_protocol(check),
          random._cross_protocol(source["check"]))
    _same("unchanged student architecture and initialization", check["model_metadata"], source["check"]["model_metadata"])
    _same("same initial student parameters", check["initial_state_sha256"], source["check"]["initial_state_sha256"])
    _same("same resolved teacher checkpoint", check["config"].get("teacher_checkpoint"),
          source["config"].get("teacher_checkpoint"))
    _same("no global projector state", check.get("distillation_initial_state_sha256"), EMPTY_STATE_SHA256)
    losses = check.get("losses", {})
    for key in ("global", "local"):
        _same("disabled " + key + " branch", losses.get(key), 0.)
    return check


def _materialize(study, plan, state):
    settings = deepcopy(plan["candidate"]["settings"])
    settings["output_root"] = str(study / "outputs")
    config_path, suite = study / "config.json", study / "suite"
    expected = random.protocol_hash(plan["candidate"]["plan"])
    if config_path.exists():
        _same("saved candidate scientific protocol", random.protocol_hash(random.make_plan(config_path, Path(plan["project"]))), expected)
    random.legacy._json(config_path, settings)
    actual = random.make_plan(config_path, Path(plan["project"]))
    _same("materialized candidate scientific protocol", random.protocol_hash(actual), expected)
    record = state.setdefault("candidate", {"status": "pending"})
    for key, value in (("config_path", str(config_path)), ("suite", str(suite))):
        if key in record:
            _same("pinned candidate " + key, record[key], value)
        record[key] = value
    if not (suite / "suite.json").exists():
        if record["status"] != "pending":
            raise ValueError("A previously started candidate is missing its pinned child manifest")
        random.legacy._json(suite / "suite.json", {"format_version": random.FORMAT,
            "protocol_sha256": random.protocol_hash(actual), "status": "pending", "initialization": "random",
            "jobs": {job["id"]: {"job": deepcopy(job), "status": "pending", "attempts": []} for job in actual["jobs"]}})
    record["plan"] = actual


def _report(study, state):
    from .random_student_soft_report import print_metrics, write_report
    write_report(study, state)
    print_metrics(study, state)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None,
              verifier=None, baseline_verifier=None, reporter=None):
    project, signature = Path(plan["project"]), protocol_hash(plan)
    if resume_study:
        study = random.legacy._path(resume_study, project)
        if not study.is_dir():
            raise ValueError("The requested soft-only study directory does not exist")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        study = Path(plan["output_root"]) / "studies" / f"study_{stamp}_{uuid.uuid4().hex[:8]}"
        study.mkdir(parents=True)
    with _study_lock(study):
        fields = ("baseline", "seeds", "weight_decay", "drop_path_rate", "distillation_profile", "selection_rule")
        if resume_study:
            state = random.legacy._read(study / "study.json")
            if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
                raise ValueError("Resume differs from the frozen soft-only scientific protocol")
            for name in fields:
                _same("saved " + name, state.get(name), plan[name])
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature, "status": "pending",
                     **{name: deepcopy(plan[name]) for name in fields}}
        baseline_api.validate_snapshot(plan["baseline"], verify_runs=True, verifier=baseline_verifier)
        _materialize(study, plan, state)
        state.update(status="running", plan=deepcopy(plan), check_only=bool(check_only))
        execution = {"started_at": _now(), "devices": plan["devices"], "check_only": bool(check_only)}
        state.setdefault("execution_history", []).append(execution)
        def persist():
            state["updated_at"] = _now()
            random.legacy._json(study / "study.json", state)
        persist()
        print(f"Study directory: {study}", flush=True)
        print("1 planned new training: B seed42, soft labels only; frozen A/full-KD controls; WD=0.01; DropPath=0", flush=True)
        launch = launcher or random._launch
        def audited_launch(job, action, child_project, directory):
            baseline_api.validate_snapshot(plan["baseline"])
            value = launch(job, action, child_project, directory)
            if action == "check":
                audit_check(value, job, plan["baseline"])
            return value
        try:
            baseline_api.validate_snapshot(plan["baseline"])
            record = state["candidate"]
            record.update(status="running", started_at=_now())
            record.pop("error", None)
            persist()
            try:
                child = random.run_suite(record["plan"], check_only=check_only,
                    resume_suite=record["suite"], launcher=audited_launch, verifier=verifier)
                _same("candidate child suite path", str(child), record["suite"])
                record["status"] = random.legacy._read(child / "suite.json")["status"]
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                record.update(status="failed", error=str(error))
                print(f"[soft-only candidate failed] {error}", flush=True)
            record["finished_at"] = _now()
            state["status"] = record["status"]
            baseline_api.validate_snapshot(plan["baseline"])
            execution.update(finished_at=_now(), status=state["status"])
            persist()
        except BaseException as error:
            state.update(status="interrupted", error=str(error))
            execution.update(finished_at=_now(), status="interrupted")
            persist()
            raise
        (reporter or _report)(study, state)
        print(f"Soft-only study {state['status']}: {study}", flush=True)
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
    mode.add_argument("--plot-curves", action="store_true")
    parser.add_argument("--resume-study")
    parser.add_argument("--study", help="Read only this study with --show-metrics or --plot-curves")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics or args.plot_curves:
            if args.resume_study:
                raise ValueError("--resume-study cannot be used with reporting")
            settings = random.legacy._read(random.legacy._path(args.config, project))
            batches = saved_studies(random.legacy._path(settings.get("output_root", "runs/random_student_soft_only"), project),
                random.legacy._path(args.study, project) if args.study else None)
            if not batches:
                print("No saved soft-only studies found.")
                return 0
            from .random_student_soft_report import plot_curves, print_metrics
            for study, state in batches:
                print(plot_curves(study, state)) if args.plot_curves else print_metrics(study, state)
            return 0
        if args.study:
            raise ValueError("--study requires --show-metrics or --plot-curves; use --resume-study to continue")
        plan = make_plan(args.config, project, devices=args.devices,
            max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        study = run_study(plan, resume_study=args.resume_study, check_only=args.check_only)
        return 1 if random.legacy._read(study / "study.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Soft-only study failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
