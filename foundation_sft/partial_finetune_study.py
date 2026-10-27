"""Three validation-only partial-finetuning candidates beside the fixed P-E baseline."""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from . import distillation_suite as records
from . import teacher_weight_study as original
from .weight_decay_study import _runtime, _study_lock


FORMAT = "partial_finetune_study_v1"
DEFAULT_CONFIG = "foundation_sft/configs/partial_finetune_study.json"
SUCCESS = {"completed", "resumed", "reused"}
SELECTION_RULE = {"metric": "best validation patient ROC-AUC",
    "ties": "fewer trainable student parameters", "test_metrics_used": False,
    "requires_all_candidates": True, "candidate_last_blocks": [8, 4, 2]}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _same(name, actual, expected):
    records._same("partial-finetuning " + name, actual, expected)


def _job(config, arm):
    return {"id": arm, "group": arm, "condition": "teacher_weight", "phase": config["phase"],
        "method": "equal", "student_initialization": "imagenet", "seed": 42,
        "student_mode": config["student"]["mode"], "device": config["device"],
        "action": "train", "config": config}


def make_plan(config_path, project, *, devices=None, max_parallel=None, cache_dir=None, source_study=None):
    """Read the exact source job, without loading Torch, models or test pixels."""
    project = Path(project).expanduser().resolve()
    path = records._path(config_path, project)
    settings = records._read(path)
    allowed = {"source_study", "seeds", "train_last_blocks", "devices", "max_parallel", "cache_dir", "output_root"}
    if set(settings) - allowed:
        raise ValueError("Only the freeze range changes in this study; unsupported settings: " + str(sorted(set(settings)-allowed)))
    if settings.get("seeds", [42]) != [42] or any(type(s) is not int for s in settings.get("seeds", [42])):
        raise ValueError("The approved screening uses only seed42")
    if settings.get("train_last_blocks", [8, 4, 2]) != [8, 4, 2] or any(type(n) is not int for n in settings.get("train_last_blocks", [8, 4, 2])):
        raise ValueError("Declare train_last_blocks=[8,4,2]")
    source = records._path(source_study or settings["source_study"], project)
    saved = records._read(source / "study.json")
    if saved.get("format_version") != original.FORMAT:
        raise ValueError("Source must be the existing six-group teacher-weight study")
    _same("source project", saved["plan"]["project"], str(project))
    _same("source protocol", original.protocol_hash(saved["plan"]), saved.get("protocol_sha256"))
    selected = original.select_learning_rates(saved).get("imagenet")
    _same("source ImageNet LR selection", selected, saved.get("selection", {}).get("imagenet"))
    if not selected or selected.get("status") != "selected":
        raise ValueError("All three ImageNet learning-rate pilots must be complete")
    source_record = deepcopy(saved.get("main", {}).get("P-E"))
    if not isinstance(source_record, dict) or source_record.get("status") not in SUCCESS:
        raise ValueError("The exact source P-E baseline must be complete")
    expected = next(job for job in original._formal_jobs(saved, saved["plan"]) if job["id"] == "P-E")
    _same("baseline recipe", original._scientific_config(source_record["job"]["config"]), original._scientific_config(expected["config"]))
    config = deepcopy(source_record["job"]["config"])
    for key, value in (("student_initialization", "imagenet"), ("method", "equal"), ("seed", 42), ("phase", "main")):
        _same("baseline " + key, config.get(key), value)
    _same("baseline student", config.get("student"), {"mode": "full_finetune"})
    _same("three KD settings", config.get("distillation"), original.LOSS_SETTINGS)
    train = config["training"]
    for key, value in (("backbone_lr", 5e-5), ("head_lr", 1e-4), ("projection_lr", 1e-4),
            ("weight_decay", .01), ("drop_path_rate", 0.), ("amp", False)):
        _same("fixed training." + key, train.get(key, 0. if key == "drop_path_rate" else None), value)
    result = source_record.get("result")
    if not isinstance(result, dict) or result.get("status") != "completed" or result.get("phase") != "main":
        raise ValueError("Source P-E must have its completed main evaluation receipt")
    checksum = result.get("checkpoint_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("Source P-E requires a pinned checkpoint fingerprint")
    _same("selected equal checkpoint", config.get("selected_checkpoint_sha256"), checksum)
    source_check = original._checked(deepcopy(source_record.get("check")), source_record["job"])
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    cache = cache or config.get("cache_dir")
    output = records._path(settings.get("output_root", "runs/partial_finetune_study"), project)
    if output == source or output.is_relative_to(source):
        raise ValueError("The new study cannot write inside the source study")
    config.update(device=chosen[0], cache_dir=cache, output_root=str(output / "unmaterialized"))
    baseline = {"job": _job(config, "baseline"), "check": source_check,
        "result": deepcopy(result), "source_study": str(source)}
    jobs = []
    for index, count in enumerate((8, 4, 2)):
        candidate = deepcopy(config)
        candidate.pop("selected_checkpoint_sha256", None)
        candidate.update(phase="pilot", device=chosen[index % len(chosen)],
            student={"mode": "partial_finetune", "train_last_blocks": count})
        jobs.append(_job(candidate, "last" + str(count)))
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
        "source_study": str(source), "source_protocol_sha256": saved["protocol_sha256"],
        "output_root": str(output), "devices": chosen, "max_parallel": maximum,
        "cache_dir": cache, "seeds": [42], "selection_rule": deepcopy(SELECTION_RULE),
        "baseline": baseline, "jobs": jobs, "new_optimization_trainings": 3}


def protocol_hash(plan):
    return records._hash({"format_version": FORMAT, "project": plan["project"],
        "source_study": plan["source_study"], "source_protocol_sha256": plan["source_protocol_sha256"],
        "baseline": {"run": plan["baseline"]["result"]["run"],
            "checkpoint_sha256": plan["baseline"]["result"]["checkpoint_sha256"],
            "config": original._scientific_config(plan["baseline"]["job"]["config"])},
        "selection_rule": plan["selection_rule"], "jobs": [{"id": job["id"],
            "config": original._scientific_config(job["config"])} for job in plan["jobs"]]})


def _checked(value, job):
    check = original._checked(value, job)
    _same("student update scope", check["config"].get("student"), job["config"]["student"])
    if job["id"] != "baseline":
        for key in ("partial_frozen_state_verified", "partial_optimizer_step_verified"):
            _same(key, check.get(key), True)
        _same("three feature KD gradient proofs", check.get("feature_kd_gradients_verified"),
            {"soft": True, "global": True, "local": True})
        frozen = check.get("frozen_parameter_sha256")
        if not isinstance(frozen, str) or len(frozen) != 64:
            raise ValueError("Partial preflight requires frozen tensor fingerprints")
    return check


def _audit(check, baseline):
    for key in ("initial_state_sha256", "head_initial_state_sha256", "distillation_initial_state_sha256",
                "head_initialization", "teacher", "teachers", "shared_view", "teacher_views", "cohort_hash", "weighting_metadata"):
        _same("same initialized tensors and sources: " + key, check.get(key), baseline.get(key))
    for key in ("training", "evaluation", "distillation", "student_initialization", "seed", "method"):
        _same("same effective " + key, check["config"].get(key), baseline["config"].get(key))
    left, right = deepcopy(check["model_metadata"]), deepcopy(baseline["model_metadata"])
    for value in (left, right):
        for key in ("mode", "trainable_parameters", "partial_finetune"):
            value.pop(key, None)
    _same("same encoder architecture/source/preprocessing", left, right)


def _launch(job, action, project, directory):
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    cfg, receipt = directory / "config.json", directory / "result.json"
    records._json(cfg, config)
    command = [sys.executable, "-u", "-m", "foundation_sft.teacher_weight_training",
        "--project", str(project), "--config", str(cfg), "--action", action, "--result-json", str(receipt)]
    if action == "evaluate":
        command.extend(["--run", job["evaluation_run"]])
    log, tail = directory / (action + ".log"), deque(maxlen=12)
    print(f"[{action}] {job['id']} {job['device']} log={log}", flush=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                tail.append(line.rstrip())
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
        raise RuntimeError(f"Process exited {code}; see {log}\n" + "\n".join(tail))
    return records._read(receipt)


def _verify(run, check, job):
    from .teacher_weight_training import verify_completed_run
    return verify_completed_run(run, check, job)


def select_setting(state):
    entries = {"baseline": state["baseline"], **state["candidates"]}
    if set(entries) != {"baseline", "last8", "last4", "last2"} or any(r.get("status") not in SUCCESS for r in entries.values()):
        return {"status": "incomplete", "test_metrics_used": False}
    choices = []
    for arm, record in entries.items():
        receipt = record["result"]
        if arm != "baseline" and (receipt.get("phase") != "pilot" or "test_patient" in receipt or "roc_auc_ci" in receipt):
            raise ValueError("Screening candidates must remain validation-only")
        score = receipt.get("best_val_patient_roc_auc")
        count = record["check"]["model_metadata"].get("trainable_parameters")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Selection requires a finite validation ROC-AUC")
        if type(count) is not int or count <= 0:
            raise ValueError("Selection requires the actual trainable student parameter count")
        choices.append({"arm": arm, "best_val_patient_roc_auc": score, "trainable_student_parameters": count,
                        "checkpoint_sha256": receipt["checkpoint_sha256"], "run": receipt["run"]})
    selected = min(choices, key=lambda r: (-r["best_val_patient_roc_auc"], r["trainable_student_parameters"]))
    return {"status": "selected", **selected, "candidates": choices, "test_metrics_used": False}


def _active_workers(study):
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = [part.decode() for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except (OSError, UnicodeError):
            continue
        if "--config" in args:
            index = args.index("--config") + 1
            if index < len(args) and args[index].startswith(str(study) + "/"):
                active.append(int(entry.name))
    return active


def _report(study, state):
    from .partial_finetune_report import write_report, print_metrics
    write_report(study, state)
    print_metrics(study, state)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None, verifier=None, reporter=None):
    signature = protocol_hash(plan)
    if resume_study:
        study = records._path(resume_study, Path(plan["project"]))
        if not study.is_dir():
            raise ValueError("The exact partial-finetuning study is missing")
    else:
        study = Path(plan["output_root"]) / "studies" / ("study_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
        study.mkdir(parents=True)
    with _study_lock(study):
        active = _active_workers(study)
        if active:
            raise ValueError("This study still has worker processes; preserve them before resuming: " + str(active))
        if resume_study:
            state = records._read(study / "study.json")
            _same("resume format", state.get("format_version"), FORMAT)
            _same("resume protocol", state.get("protocol_sha256"), signature)
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature, "baseline": {},
                "candidates": {}, "selection": {"status": "incomplete"}, "selected_evaluation": {"status": "pending"}}
        for job in [plan["baseline"]["job"], *plan["jobs"]]:
            section = state["baseline"] if job["id"] == "baseline" else state["candidates"].setdefault(job["id"], {})
            if section.get("job"):
                _same("resume job", original._scientific_config(section["job"]["config"]), original._scientific_config(job["config"]))
            section["job"] = deepcopy(job)
            section.setdefault("attempts", [])
        state.update(plan=deepcopy(plan), source_study=plan["source_study"], status="running")
        state.setdefault("execution_history", []).append({"started_at": _now(), "devices": plan["devices"], "max_parallel": plan["max_parallel"]})
        mutex = threading.RLock()
        def persist():
            state["updated_at"] = _now()
            records._json(study / "study.json", state)
        def entry(job):
            return state["baseline"] if job["id"] == "baseline" else state["candidates"][job["id"]]
        launch, verify = launcher or _launch, verifier or _verify
        def invoke(job, action, section=None):
            section = section if section is not None else entry(job)
            directory = study / "jobs" / job["id"] / (action + "_" + uuid.uuid4().hex[:8])
            attempt = {"action": action, "directory": str(directory), "status": "running", "device": job["device"], "started_at": _now()}
            with mutex:
                section.update(status="checking" if action == "check" else "training", phase=action)
                section.setdefault("attempts", []).append(attempt)
                persist()
            try:
                value = launch(job, action, Path(plan["project"]), directory)
            except BaseException as error:
                with mutex:
                    attempt.update(status="failed", error=str(error), finished_at=_now())
                    persist()
                raise
            with mutex:
                attempt.update(status="completed", finished_at=_now())
                persist()
            return value
        persist()
        print(f"Study directory: {study}", flush=True)
        print("Reuse exact P-E; three validation-only candidates (last8,last4,last2); evaluate at most one selected new model.", flush=True)
        previous = {job["id"]: deepcopy(entry(job).get("result")) for job in plan["jobs"]}
        try:
            jobs = [plan["baseline"]["job"], *plan["jobs"]]
            def checked(job, value, error):
                with mutex:
                    record = entry(job)
                    try:
                        if error is not None:
                            raise error
                        record.update(check=_checked(value, job), status="checked")
                        record.pop("error", None)
                    except Exception as invalid:
                        record.update(status="failed", error=str(invalid))
                    print(f"[check {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
                    persist()
            records._parallel(jobs, plan["max_parallel"], lambda job: invoke(job, "check"), checked)
            base = state["baseline"]
            if base["status"] != "checked":
                raise ValueError("Baseline preflight failed; no candidate optimization may start")
            _audit(base["check"], plan["baseline"]["check"])
            base_result = verify(plan["baseline"]["result"]["run"], base["check"], base["job"])
            _same("pinned baseline checkpoint", base_result["checkpoint_sha256"], plan["baseline"]["result"]["checkpoint_sha256"])
            base.update(status="reused", result=base_result)
            waiting = []
            for job in plan["jobs"]:
                record = entry(job)
                if record["status"] != "checked":
                    continue
                try:
                    _audit(record["check"], base["check"])
                    if record["check"]["model_metadata"]["trainable_parameters"] >= base["check"]["model_metadata"]["trainable_parameters"]:
                        raise ValueError("A partial candidate must train fewer student parameters than baseline")
                    if previous[job["id"]]:
                        receipt = verify(previous[job["id"]]["run"], record["check"], job)
                        _same("resume checkpoint", receipt["checkpoint_sha256"], previous[job["id"]]["checkpoint_sha256"])
                        record.update(status="resumed", result=receipt)
                    elif not check_only:
                        waiting.append(job)
                except Exception as error:
                    record.update(status="failed", error=str(error))
                persist()
            def completed(job, value, error):
                with mutex:
                    record = entry(job)
                    try:
                        if error is not None:
                            raise error
                        if not Path(value["run"]).resolve().is_relative_to((study / "jobs" / job["id"]).resolve()):
                            raise ValueError("Candidate receipt is outside its exact study arm")
                        receipt = verify(value["run"], record["check"], job)
                        if receipt.get("phase") != "pilot" or "test_patient" in receipt or "roc_auc_ci" in receipt:
                            raise ValueError("Candidate optimization must never evaluate test")
                        record.update(status="completed", result=receipt)
                        record.pop("error", None)
                    except Exception as invalid:
                        record.update(status="failed", error=str(invalid))
                    print(f"[train {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
                    persist()
            if waiting:
                records._parallel(waiting, plan["max_parallel"], lambda job: invoke(job, "train"), completed)
            selected = select_setting(state)
            if state["selection"].get("status") == "selected":
                _same("frozen validation selection", selected, state["selection"])
            state["selection"] = selected
            persist()
            if selected.get("status") == "selected" and not check_only:
                evaluation = state["selected_evaluation"]
                if selected["arm"] == "baseline":
                    evaluation.update(status="reused", arm="baseline", result=deepcopy(base["result"]))
                else:
                    candidate = state["candidates"][selected["arm"]]
                    job = deepcopy(candidate["job"])
                    job.update(id=selected["arm"] + "_selected", action="evaluate", phase="main", evaluation_run=selected["run"])
                    job["config"].update(phase="main", selected_checkpoint_sha256=selected["checkpoint_sha256"])
                    check = deepcopy(candidate["check"])
                    check["config"].update(phase="main", selected_checkpoint_sha256=selected["checkpoint_sha256"])
                    evaluation.update(arm=selected["arm"], job=job, check=check)
                    try:
                        if evaluation.get("result"):
                            result = verify(selected["run"], check, job)
                            _same("saved selected evaluation", result, evaluation["result"])
                            evaluation.update(status="resumed", result=result)
                        else:
                            value = invoke(job, "evaluate", evaluation)
                            _same("selected candidate evaluation run", str(Path(value["run"]).resolve()), str(Path(selected["run"]).resolve()))
                            result = verify(selected["run"], check, job)
                            _same("selected candidate checkpoint", result["checkpoint_sha256"], selected["checkpoint_sha256"])
                            evaluation.update(status="completed", result=result)
                        evaluation.pop("error", None)
                    except Exception as error:
                        evaluation.update(status="failed", error=str(error))
                persist()
            if selected.get("status") == "selected" and state["selected_evaluation"].get("status") in SUCCESS:
                state["status"] = "completed"
            elif check_only and all(entry(j)["status"] in SUCCESS | {"checked"} for j in jobs):
                state["status"] = "checked"
            else:
                state["status"] = "partial"
            state["execution_history"][-1].update(finished_at=_now(), status=state["status"])
            persist()
        except BaseException as error:
            state.update(status="interrupted", error=str(error))
            state["execution_history"][-1].update(finished_at=_now(), status="interrupted")
            persist()
            raise
        (reporter or _report)(study, state)
        print(f"Partial-finetuning study {state['status']}: {study}", flush=True)
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-study")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume-study")
    parser.add_argument("--study")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--show-metrics", action="store_true")
    mode.add_argument("--plot-curves", action="store_true")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).expanduser().resolve()
        if args.show_metrics or args.plot_curves:
            if args.resume_study:
                raise ValueError("Use --study for read-only reporting")
            settings = records._read(records._path(args.config, project))
            paths = ([records._path(args.study, project)] if args.study else
                sorted((records._path(settings.get("output_root", "runs/partial_finetune_study"), project) / "studies").glob("study_*")))
            from .partial_finetune_report import print_metrics, plot_curves
            for path in paths:
                if (path / "study.json").is_file():
                    state = records._read(path / "study.json")
                    if state.get("format_version") == FORMAT:
                        print(plot_curves(path, state)) if args.plot_curves else print_metrics(path, state)
            return 0
        if args.study:
            raise ValueError("--study is read-only; use --resume-study to continue")
        plan = make_plan(args.config, project, devices=args.devices, max_parallel=args.max_parallel,
            cache_dir=args.cache_dir, source_study=args.source_study)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        study = run_study(plan, resume_study=args.resume_study, check_only=args.check_only)
        return 0 if records._read(study / "study.json")["status"] in {"completed", "checked"} else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Partial-finetuning study failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
