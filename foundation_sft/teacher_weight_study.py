"""Validation-only LR search, then six matched teacher-weight experiments.

Each initialization has three equal-weight train/validation pilots. Only the
selected pilot is evaluated on test, without retraining. CA and RL then use that
initialization's frozen LR. Planning and reporting do not import Torch.
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
from .droppath_study import _study_lock
from .weight_decay_study import _runtime


FORMAT = "teacher_weight_study_v1"
DEFAULT_CONFIG = "foundation_sft/configs/teacher_weight_study.json"
INITIALIZATIONS = ("random", "imagenet")
METHODS = ("equal", "ca", "rl")
TEACHERS = ("rad_dino", "chexfound", "chexzero")
SUCCESS = {"completed", "resumed"}
LOSS_SETTINGS = {"temperature": 2.0, "soft_weight": .5, "global_weight": .1,
                 "local_weight": 1., "pool_grid": [7, 7]}
LR_GRID = {"random": [5e-5, 1e-4, 3e-4], "imagenet": [1e-5, 3e-5, 5e-5]}
SELECTION_RULE = {
    "reference_method": "equal", "scope": "independently for each student initialization",
    "metric": "best validation patient ROC-AUC", "ties": "smaller backbone learning rate",
    "requires_all_declared_candidates": True, "test_metrics_used": False,
    "head_lr": 1e-4, "projection_lr": 1e-4,
    "selected_equal_pilot_is_reused_without_retraining": True,
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _same(name, actual, expected):
    legacy._same("teacher-weight study " + name, actual, expected)


def _rate(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return float(value)


def group_name(initialization, method):
    return ("R" if initialization == "random" else "P") + "-" + {"equal": "E", "ca": "C", "rl": "R"}[method]


def _job_id(initialization, rate):
    return ("R" if initialization == "random" else "P") + "_pilot_" + format(rate, ".12g").replace(".", "p").replace("-", "m")


def _scientific_config(config):
    result = deepcopy(config)
    for key in ("device", "output_root", "cache_dir"):
        result.pop(key, None)
    return result


def _job(config, name, *, action="train"):
    return {"id": name, "group": group_name(config["student_initialization"], config["method"]),
        "student_initialization": config["student_initialization"], "method": config["method"],
        "phase": config["phase"], "seed": 42, "student_mode": "full_finetune",
        "condition": "teacher_weight", "action": action, "device": config["device"], "config": config}


def make_plan(config_path, project, *, devices=None, max_parallel=None, cache_dir=None):
    """Resolve six pilots and six eventual rows without loading weights/data."""
    project = Path(project).expanduser().resolve()
    path = legacy._path(config_path, project)
    settings = legacy._read(path)
    if settings.get("seeds", [42]) != [42] or any(type(s) is not int for s in settings.get("seeds", [42])):
        raise ValueError("This approved screening uses seeds=[42]")
    template = legacy._read(legacy._path(settings.get("template", "foundation_sft/configs/distillation_rad_dino_vit_tiny.json"), project))
    teacher_settings = legacy._read(legacy._path(settings.get("teacher_sources", "foundation_sft/configs/equal_teacher_distillation.json"), project))
    teachers = deepcopy(settings.get("teachers", teacher_settings["teachers"]))
    if (not isinstance(teachers, list) or len(teachers) != 3
            or [item.get("model") for item in teachers] != list(TEACHERS)):
        raise ValueError("Use RAD-DINO, CheXFound and CheXzero in that order")
    for source in teachers:
        if source.get("mode") != "lora" or source.get("seed") != 42:
            raise ValueError("All student groups use the same frozen LoRA seed42 teachers")
        if not source.get("suite") and not source.get("checkpoint"):
            raise ValueError("Every teacher requires its fixed suite or checkpoint")
        for key in ("suite", "checkpoint"):
            if source.get(key) is not None:
                source[key] = str(legacy._path(source[key], project))
    _same("three KD terms", template.get("distillation"), LOSS_SETTINGS)
    if legacy._student_mode(template) != "full_finetune":
        raise ValueError("The approved students use full_finetune")
    train = deepcopy(template["training"])
    train.update(settings.get("training", {}))
    for key, default in (("epochs", 100), ("patience", 15), ("batch_size", 8), ("accumulation_steps", 8)):
        if type(train.get(key)) is not int or train[key] < 1:
            raise ValueError("training." + key + " must be a positive integer")
    for key, expected in (("head_lr", 1e-4), ("projection_lr", 1e-4), ("weight_decay", .01), ("drop_path_rate", 0.)):
        value = train.get(key, 0. if key == "drop_path_rate" else None)
        if type(value) not in (float, int) or not math.isfinite(value) or value != expected:
            raise ValueError(f"This study fixes training.{key}={expected}")
    if train.get("amp") is not False:
        raise ValueError("The approved study uses FP32 training.amp=false")
    grids = deepcopy(settings.get("learning_rates", LR_GRID))
    if not isinstance(grids, dict) or set(grids) != set(INITIALIZATIONS):
        raise ValueError("Declare separate random and imagenet learning-rate grids")
    for initialization, values in grids.items():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("Declare exactly three LR candidates per initialization")
        grids[initialization] = [_rate(value, initialization + " LR") for value in values]
        if len(set(grids[initialization])) != 3:
            raise ValueError("LR candidates must be unique")
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    if cache is None and template.get("cache_dir") is not None:
        cache = str(legacy._path(template["cache_dir"], project))
    checkpoint = settings.get("student_checkpoint", template.get("student_checkpoint"))
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("Declare the local ImageNet ViT-Tiny checkpoint")
    checkpoint = str(legacy._path(checkpoint, project))
    output = legacy._path(settings.get("output_root", "runs/teacher_weight_study"), project)
    policy = deepcopy(settings.get("policy", {}))
    if not isinstance(policy, dict):
        raise ValueError("policy must be an object")
    common = {"student": {"mode": "full_finetune"}, "seed": 42,
        "teachers": teachers, "teacher_suite": teachers[0].get("suite"),
        "teacher_checkpoint": teachers[0].get("checkpoint"), "policy": policy,
        "distillation": deepcopy(LOSS_SETTINGS), "cache_dir": cache,
        "output_root": str(output / "unmaterialized")}
    pilots = []
    for initialization in INITIALIZATIONS:
        for rate in grids[initialization]:
            cfg = deepcopy(common)
            cfg.update(student_initialization=initialization, student_checkpoint=None if initialization == "random" else checkpoint,
                method="equal", phase="pilot", device=chosen[len(pilots) % len(chosen)],
                training={**deepcopy(train), "backbone_lr": rate})
            pilots.append(_job(cfg, _job_id(initialization, rate)))
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
        "output_root": str(output), "devices": chosen, "max_parallel": maximum,
        "cache_dir": cache, "seeds": [42], "learning_rates": grids,
        "selection_rule": deepcopy(SELECTION_RULE), "pilots": pilots,
        "formal_groups": [group_name(i, m) for i in INITIALIZATIONS for m in METHODS],
        "new_optimization_trainings_maximum": 10}


def protocol_hash(plan):
    return legacy._hash({"format_version": FORMAT, "project": plan["project"],
        "learning_rates": plan["learning_rates"], "selection_rule": plan["selection_rule"],
        "seeds": plan["seeds"], "formal_groups": plan["formal_groups"],
        "pilots": [{"id": job["id"], "config": _scientific_config(job["config"])} for job in plan["pilots"]]})


def _launch(job, action, project, directory):
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    cfg, result_file = directory / "config.json", directory / "result.json"
    legacy._json(cfg, config)
    command = [sys.executable, "-u", str(project / "run_teacher_weight_training.py"),
        "--project", str(project), "--config", str(cfg), "--action", action, "--result-json", str(result_file)]
    if action == "evaluate":
        command.extend(["--run", job["evaluation_run"]])
    log, last_lines = directory / (action + ".log"), deque(maxlen=12)
    print(f"[{action}] {job['id']} {job['device']} log={log}", flush=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                last_lines.append(line.rstrip())
                if line.startswith(("[epoch]", "[train]", "[evaluation]", "Run directory:", "[prepare]", "{")):
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
    if not result_file.is_file():
        raise RuntimeError("Worker did not write " + str(result_file))
    return legacy._read(result_file)


def verify_run(run, check, job):
    from .teacher_weight_training import verify_completed_run
    return verify_completed_run(run, check, job)


def _checked(value, job):
    value = legacy._check_receipt(value)
    config = value["config"]
    for key in ("student_initialization", "method", "phase", "seed"):
        _same("preflight " + key, config.get(key), job["config"].get(key))
    _same("frozen teachers", value.get("teacher_unchanged"), True)
    _same("student backbone gradient proof", value.get("student_backbone_gradient_verified"), True)
    _same("projector gradient proof", value.get("projector_gradients_verified"), True)
    if job["method"] == "rl":
        _same("policy gradient proof", value.get("policy_gradient_verified"), True)
    _same("KD settings", config.get("distillation"), LOSS_SETTINGS)
    for key, expected in job["config"]["training"].items():
        _same("preflight training." + key, config["training"].get(key), expected)
    for key in ("initial_state_sha256", "head_initial_state_sha256", "distillation_initial_state_sha256"):
        text = value.get(key)
        if not isinstance(text, str) or len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError("Preflight requires exact initialized tensor fingerprint: " + key)
    teachers = value.get("teachers")
    if not isinstance(teachers, list) or [item.get("model") for item in teachers] != list(TEACHERS):
        raise ValueError("Preflight requires all three teachers in fixed order")
    if any(not item.get("checkpoint_sha256") for item in teachers):
        raise ValueError("Every teacher needs a checkpoint fingerprint")
    return value


def _audit_checks(check, all_checks):
    """Changing LR/method never changes the data, initial tensors or teachers."""
    for previous in all_checks:
        for key in ("cohort_hash", "teacher", "teachers", "shared_view", "teacher_views",
                    "head_initial_state_sha256", "distillation_initial_state_sha256"):
            _same("matched " + key, check.get(key), previous.get(key))
        _same("matched evaluation", check["config"].get("evaluation"), previous["config"].get("evaluation"))
        left, right = deepcopy(check["config"]["training"]), deepcopy(previous["config"]["training"])
        left.pop("backbone_lr", None)
        right.pop("backbone_lr", None)
        _same("only backbone LR changes in search", left, right)
        if check["config"]["student_initialization"] == previous["config"]["student_initialization"]:
            for key in ("initial_state_sha256", "model_metadata"):
                _same("same-initialization " + key, check[key], previous[key])
        else:
            left, right = deepcopy(check["model_metadata"]), deepcopy(previous["model_metadata"])
            left.pop("initialization", None)
            right.pop("initialization", None)
            _same("same student architecture", left, right)


def select_learning_rates(state):
    result = {}
    for initialization in INITIALIZATIONS:
        records = [r for r in state["pilots"].values() if r["job"]["student_initialization"] == initialization]
        if len(records) != 3 or any(r.get("status") not in SUCCESS for r in records):
            result[initialization] = {"status": "incomplete"}
            continue
        candidates = []
        for record in records:
            receipt = record["result"]
            if receipt.get("phase") != "pilot" or "test_patient" in receipt:
                raise ValueError("LR selection accepts train/validation pilot receipts only")
            score = receipt.get("best_val_patient_roc_auc")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Pilot validation ROC-AUC must be finite in [0,1]")
            candidates.append({"job_id": record["job"]["id"],
                "backbone_lr": record["job"]["config"]["training"]["backbone_lr"],
                "best_val_patient_roc_auc": score})
        winner = min(candidates, key=lambda c: (-c["best_val_patient_roc_auc"], c["backbone_lr"]))
        result[initialization] = {"status": "selected", **winner, "candidates": candidates,
                                  "test_metrics_used": False}
    return result


def _formal_jobs(state, plan):
    jobs = []
    for initialization in INITIALIZATIONS:
        selection = state["selection"].get(initialization, {})
        if selection.get("status") != "selected":
            continue
        pilot = state["pilots"][selection["job_id"]]
        for method in METHODS:
            name = group_name(initialization, method)
            config = deepcopy(pilot["job"]["config"])
            config.update(method=method, phase="main")
            config["device"] = plan["devices"][len(jobs) % len(plan["devices"])]
            config["cache_dir"] = plan["cache_dir"]
            config["output_root"] = str(Path(plan["output_root"]) / "unmaterialized")
            job = _job(config, name, action="evaluate" if method == "equal" else "train")
            if method == "equal":
                config["selected_checkpoint_sha256"] = pilot["result"]["checkpoint_sha256"]
                job["evaluation_run"] = pilot["result"]["run"]
            jobs.append(job)
    return jobs


def _record_jobs(records, jobs):
    for job in jobs:
        record = records.setdefault(job["id"], {"status": "pending", "attempts": []})
        if "job" in record:
            _same("saved job protocol", _scientific_config(record["job"]["config"]), _scientific_config(job["config"]))
            _same("saved selected pilot path", record["job"].get("evaluation_run"), job.get("evaluation_run"))
        record["job"] = deepcopy(job)


def _execute_jobs(study, state, records, jobs, plan, *, check_only, launcher, verifier, persist):
    lock = threading.RLock()
    previous = {name: deepcopy(record.get("result")) for name, record in records.items() if record.get("result")}
    checks = []
    for section in (state["pilots"], state["main"]):
        checks.extend(r["check"] for r in section.values() if isinstance(r.get("check"), dict))

    def invoke(job, action):
        directory = study / "jobs" / job["id"] / (action + "_" + uuid.uuid4().hex[:8])
        attempt = {"action": action, "directory": str(directory), "status": "running",
                   "started_at": _now(), "device": job["device"]}
        with lock:
            record = records[job["id"]]
            record.update(status="checking" if action == "check" else "training", phase=action)
            record["attempts"].append(attempt)
            persist()
        try:
            result = launcher(job, action, Path(plan["project"]), directory)
        except BaseException as error:
            with lock:
                attempt.update(status="failed", finished_at=_now(), error=str(error))
                persist()
            raise
        with lock:
            attempt.update(status="completed", finished_at=_now())
            persist()
        return result

    def checked(job, value, error):
        with lock:
            record = records[job["id"]]
            try:
                if error is not None:
                    raise error
                check = _checked(value, job)
                _audit_checks(check, checks)
                checks.append(check)
                record.update(status="checked", check=check)
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", error=str(invalid))
            print(f"[check {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
            persist()

    legacy._parallel(jobs, plan["max_parallel"], lambda job: invoke(job, "check"), checked)
    waiting = []
    for job in jobs:
        record = records[job["id"]]
        if record["status"] != "checked":
            continue
        old = previous.get(job["id"])
        if old:
            try:
                result = verifier(old["run"], record["check"], job)
                _same("saved checkpoint fingerprint", result.get("checkpoint_sha256"), old.get("checkpoint_sha256"))
                record.update(status="resumed", result=result)
                print(f"[resumed] {job['id']}: {result['run']}", flush=True)
            except Exception as invalid:
                record.setdefault("resume_rejections", []).append({"run": old["run"], "reason": str(invalid)})
                print(f"[resume unavailable] {job['id']}: {invalid}", flush=True)
        if record["status"] == "checked" and not check_only:
            waiting.append(job)
        persist()

    def completed(job, value, error):
        with lock:
            record = records[job["id"]]
            try:
                if error is not None:
                    raise error
                if value.get("status") != "completed" or not value.get("run"):
                    raise ValueError("Worker receipt does not identify a completed run")
                result = verifier(value["run"], record["check"], job)
                if job["phase"] == "pilot" and (result.get("phase") != "pilot" or "test_patient" in result):
                    raise ValueError("LR pilots must not evaluate or report test metrics")
                if job["action"] == "evaluate":
                    _same("selected equal pilot reused", str(Path(result["run"]).resolve()), str(Path(job["evaluation_run"]).resolve()))
                    _same("selected equal checkpoint reused", result.get("checkpoint_sha256"), job["config"]["selected_checkpoint_sha256"])
                record.update(status="completed", result=result)
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", error=str(invalid))
            print(f"[{job['action']} {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
            persist()

    if waiting:
        legacy._parallel(waiting, plan["max_parallel"], lambda job: invoke(job, job["action"]), completed)


def _report(study, state):
    from .teacher_weight_report import write_report, print_metrics
    write_report(study, state)
    print_metrics(study, state)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None, verifier=None, reporter=None):
    signature = protocol_hash(plan)
    if resume_study:
        study = legacy._path(resume_study, Path(plan["project"]))
        if not study.is_dir():
            raise ValueError("The requested teacher-weight study directory does not exist")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        study = Path(plan["output_root"]) / "studies" / f"study_{stamp}_{uuid.uuid4().hex[:8]}"
        study.mkdir(parents=True)
    with _study_lock(study):
        if resume_study:
            state = legacy._read(study / "study.json")
            if state.get("format_version") != FORMAT or state.get("protocol_sha256") != signature:
                raise ValueError("Resume differs from the frozen teacher-weight study protocol")
            if set(state.get("pilots", {})) != {job["id"] for job in plan["pilots"]}:
                raise ValueError("Resume pilot set differs")
            if set(state.get("main", {})) - set(plan["formal_groups"]):
                raise ValueError("Resume formal group set differs")
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature,
                "pilots": {}, "selection": {}, "main": {}, "status": "pending"}
        _record_jobs(state["pilots"], plan["pilots"])
        state.update(plan=deepcopy(plan), status="running", check_only=bool(check_only))
        state.setdefault("execution_history", []).append({"started_at": _now(), "devices": plan["devices"],
            "max_parallel": plan["max_parallel"], "check_only": bool(check_only)})
        def persist():
            state["updated_at"] = _now()
            legacy._json(study / "study.json", state)
        persist()
        print(f"Study directory: {study}", flush=True)
        print("Six validation-only equal pilots; two selected equal models reused; four CA/RL trainings. Maximum 10 optimization trainings.", flush=True)
        launch, verify = launcher or _launch, verifier or verify_run
        try:
            _execute_jobs(study, state, state["pilots"], plan["pilots"], plan,
                check_only=check_only, launcher=launch, verifier=verify, persist=persist)
            selected = select_learning_rates(state)
            for initialization, old in state.get("selection", {}).items():
                if old.get("status") == "selected":
                    _same("frozen selected LR " + initialization, selected.get(initialization), old)
            state["selection"] = selected
            persist()
            formal = _formal_jobs(state, plan)
            _record_jobs(state["main"], formal)
            persist()
            if formal:
                _execute_jobs(study, state, state["main"], formal, plan,
                    check_only=check_only, launcher=launch, verifier=verify, persist=persist)
            statuses = [record["status"] for record in state["main"].values()]
            if len(statuses) == 6 and all(s in SUCCESS for s in statuses):
                status = "completed"
            elif check_only and all(record["status"] in SUCCESS | {"checked"} for record in state["pilots"].values()) and all(s in SUCCESS | {"checked"} for s in statuses):
                status = "checked"
            else:
                status = "partial" if any(r["status"] in SUCCESS | {"checked"} for r in state["pilots"].values()) else "failed"
            state["status"] = status
            state["execution_history"][-1].update(finished_at=_now(), status=status)
            persist()
        except BaseException as error:
            state.update(status="interrupted", error=str(error))
            state["execution_history"][-1].update(finished_at=_now(), status="interrupted")
            persist()
            raise
        (reporter or _report)(study, state)
        print(f"Teacher-weight study {state['status']}: {study}", flush=True)
    return study


def saved_studies(output_root, study=None):
    paths = [Path(study)] if study else sorted((Path(output_root) / "studies").glob("study_*"))
    return [(path, state) for path in paths if (path / "study.json").is_file()
            and (state := legacy._read(path / "study.json")).get("format_version") == FORMAT]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume-study")
    parser.add_argument("--study", help="Read only this saved study with reporting")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--show-metrics", action="store_true")
    mode.add_argument("--plot-curves", action="store_true")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).resolve()
        if args.show_metrics or args.plot_curves:
            if args.resume_study:
                raise ValueError("Reporting cannot use --resume-study")
            settings = legacy._read(legacy._path(args.config, project))
            batches = saved_studies(legacy._path(settings.get("output_root", "runs/teacher_weight_study"), project),
                legacy._path(args.study, project) if args.study else None)
            if not batches:
                print("No saved teacher-weight studies found.")
            from .teacher_weight_report import print_metrics, plot_curves
            for study, state in batches:
                print(plot_curves(study, state)) if args.plot_curves else print_metrics(study, state)
            return 0
        if args.study:
            raise ValueError("--study requires --show-metrics or --plot-curves; use --resume-study to continue")
        plan = make_plan(args.config, project, devices=args.devices, max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        study = run_study(plan, resume_study=args.resume_study, check_only=args.check_only)
        return 1 if legacy._read(study / "study.json")["status"] in {"failed", "partial", "interrupted"} else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print("Teacher-weight study failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
