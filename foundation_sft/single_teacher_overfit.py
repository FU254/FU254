"""Fresh single-teacher diagnostics, reusing the original trainer and GPU queue.

Copy the effective historical ImageNet recipe; never import a trained student or
new test scores. A/B/C vary supervision, and D adds one declared L2-SP strength.
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
import subprocess
import sys
import threading
import uuid

from . import distillation_suite as records


FORMAT = "single_teacher_overfit_study_v1"
DEFAULT_CONFIG = "foundation_sft/configs/single_teacher_overfit.json"
DIAGNOSTICS = {"protocol": "single_teacher_overfit_v1", "evaluation": "validation_only"}
GROUPS = {"A": ("supervised", (0., 0., 0.)),
          "B": ("distillation", (.5, .1, 1.)),
          "C": ("distillation", (.5, 0., 0.)),
          "D": ("distillation", (.5, .1, 1.))}
WEIGHTS = ("soft_weight", "global_weight", "local_weight")
SUCCESS = {"completed", "resumed"}


def _scientific(config):
    return {key: value for key, value in config.items()
            if key not in {"device", "cache_dir", "output_root", "teacher_config"}}


def protocol_hash(plan):
    return records._hash({"format_version": FORMAT, "project": plan["project"],
        "source_suite": plan["source_suite"], "source": plan["source"],
        "jobs": [{"id": job["id"], "config": _scientific(job["config"])} for job in plan["jobs"]]})


def _runtime(settings, project, devices=None, max_parallel=None, cache_dir=None):
    chosen = list(devices if devices is not None else settings.get("devices", ["cuda:0"]))
    if (not chosen or len(set(chosen)) != len(chosen)
            or any(not isinstance(value, str) or not value for value in chosen)):
        raise ValueError("devices must contain unique nonempty device names")
    maximum = max_parallel if max_parallel is not None else settings.get("max_parallel", 1)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_parallel must be a positive integer")
    cache = cache_dir if cache_dir is not None else settings.get("cache_dir")
    return chosen, maximum, str(records._path(cache, project)) if cache is not None else None


def validate_source(plan):
    """Pin source files and both pretrained checkpoints before starting workers."""
    for filename, expected in plan["source"]["files"].items():
        if records._file_hash(filename) != expected:
            raise ValueError("The frozen source artifact changed: " + filename)


def make_plan(config_path, project, *, source_suite=None, devices=None,
              max_parallel=None, cache_dir=None):
    project = Path(project).expanduser().resolve()
    path = records._path(config_path, project)
    settings = records._read(path)
    allowed = {"source_suite", "devices", "max_parallel", "cache_dir", "output_root", "seed", "l2_sp_coefficient"}
    if set(settings) - allowed:
        raise ValueError("This matched diagnostic study does not accept extra recipe overrides")
    if type(settings.get("seed", 42)) is not int or settings.get("seed", 42) != 42:
        raise ValueError("This exploratory study uses seed42 only")
    if type(settings.get("l2_sp_coefficient", 1e-4)) not in (float, int) or settings.get("l2_sp_coefficient", 1e-4) != 1e-4:
        raise ValueError("Declare the single exploratory L2-SP coefficient 0.0001 (sum reduction)")
    source = records._path(source_suite or settings["source_suite"], project)
    manifest = records._read(source / "suite.json")
    source_result = manifest.get("conditions", {}).get("distillation", {})
    if source_result.get("status") != "completed":
        raise ValueError("Source must be a completed original single-teacher distillation suite")
    run = records._path(source_result.get("run", "distillation"), source)
    config = records._read(run / "config.json")
    metadata = records._read(run / "model_metadata.json")
    teacher = records._read(run / "teacher.json")
    geometry = records._read(run / "shared_view.json")
    if (config.get("seed") != 42 or config.get("diagnostics") is not None
            or records._student_mode(config) != "full_finetune"
            or metadata.get("initialization", {}).get("pretraining") != "ImageNet-1k"):
        raise ValueError("Source must use the original ImageNet full-finetune seed42 student")
    train = config.get("training", {})
    required = ("epochs", "patience", "batch_size", "accumulation_steps", "num_workers",
        "backbone_lr", "head_lr", "projection_lr", "amp", "weight_decay", "minimum_lr",
        "focal_gamma", "focal_scale", "gradient_clip", "rotation_degrees", "horizontal_flip_probability")
    if any(key not in train for key in required):
        raise ValueError("Source config must contain the complete resolved effective training recipe")
    for key, expected in (("backbone_lr", 5e-5), ("head_lr", 1e-4), ("projection_lr", 1e-4),
                          ("epochs", 100), ("patience", 15), ("amp", False)):
        records._same("original training." + key, train[key], expected)
    kd = config.get("distillation", {})
    records._same("original KD weights", [kd.get(key) for key in WEIGHTS], list(GROUPS["B"][1]))
    records._same("original temperature", kd.get("temperature"), 2.)
    records._same("original KD grid", kd.get("pool_grid"), [7, 7])
    teacher_path = records._path(config["teacher_checkpoint"], project)
    student_path = records._path(config["student_checkpoint"], project)
    records._same("teacher checkpoint checksum", records._file_hash(teacher_path), teacher.get("checkpoint_sha256"))
    records._same("ImageNet checkpoint checksum", records._file_hash(student_path),
                  metadata["initialization"].get("checkpoint_sha256"))
    split_path = records._path(config["teacher_config"]["split_manifest"], project)
    paths = [source / "suite.json", run / "config.json", run / "model_metadata.json",
        run / "teacher.json", run / "shared_view.json", run / "split.csv", run / "split_metadata.json",
        teacher_path, student_path, split_path]
    snapshot = {"files": {str(file): records._file_hash(file) for file in paths},
        "training": deepcopy(train), "evaluation": deepcopy(config["evaluation"]),
        "teacher": teacher, "model_metadata": metadata, "shared_view": geometry,
        "cohort_hash": source_result["cohort_hash"], "initial_state_sha256": source_result["initial_state_sha256"]}
    chosen, maximum, cache = _runtime(settings, project, devices, max_parallel, cache_dir)
    root = records._path(settings.get("output_root", "runs/single_teacher_overfit"), project)
    if root == source or root.is_relative_to(source):
        raise ValueError("New diagnostics cannot write inside the historical source suite")
    jobs = []
    for index, (group, (condition, weights)) in enumerate(GROUPS.items()):
        current = deepcopy(config)
        current.update(device=chosen[index % len(chosen)], output_root=str(root / "unmaterialized"),
            student_checkpoint=str(student_path), teacher_checkpoint=str(teacher_path),
            diagnostics=deepcopy(DIAGNOSTICS))
        if cache is not None:
            current["cache_dir"] = cache
        current["distillation"].update(dict(zip(WEIGHTS, weights)))
        if group == "D":
            current["diagnostics"]["l2_sp"] = {"coefficient": 1e-4, "reduction": "sum"}
        jobs.append({"id": group + "_seed42", "group": group, "condition": condition,
            "seed": 42, "student_mode": "full_finetune", "device": current["device"], "config": current})
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
        "source_suite": str(source), "source": snapshot, "output_root": str(root),
        "devices": chosen, "max_parallel": maximum, "jobs": jobs,
        "evaluation_policy": "validation_only", "new_optimization_trainings": 4}


def _checked(raw, job, source):
    check = records._check_receipt(raw)
    config = check["config"]
    for key in ("training", "distillation", "diagnostics", "seed"):
        records._same("checked " + key, config.get(key), job["config"].get(key))
    records._same("checked full student", records._student_mode(config), "full_finetune")
    records._same("checked condition", check.get("condition"), job["condition"])
    records._same("checked evaluation", config.get("evaluation"), source["evaluation"])
    for key in ("cohort_hash", "initial_state_sha256", "model_metadata", "shared_view"):
        records._same("source matched " + key, check.get(key), source[key])
    records._same("source teacher", check["teacher"]["checkpoint_sha256"], source["teacher"]["checkpoint_sha256"])
    if check.get("teacher_unchanged") is not True:
        raise ValueError("Teacher must be frozen during preflight")
    if not isinstance(check.get("diagnostics"), dict):
        raise ValueError("Preflight lacks the explicit clean-evaluation diagnostic proof")
    proof = check["diagnostics"]
    for key, expected in DIAGNOSTICS.items():
        records._same("checked diagnostic " + key, proof.get(key), expected)
    records._same("clean evaluation RNG isolation", proof.get("training_rng_preserved"), True)
    clean = proof.get("clean_train_eval", {})
    for key, expected in (("split", "train"), ("training", False), ("shuffle", False),
                          ("include_teacher", False), ("private_generator", True)):
        records._same("clean train evaluation " + key, clean.get(key), expected)
    records._same("clean train shared view", clean.get("shared_view"), check["shared_view"])
    if not clean.get("augmentation") or any(value != 0 for value in clean["augmentation"].values()):
        raise ValueError("Clean train evaluation must disable every random augmentation")
    records._same("checked phase", check.get("evaluation_phase"), "validation_only")
    records._same("checked test isolation", check.get("test_evaluated"), False)
    for key in ("initial_backbone_state_sha256", "initial_head_state_sha256"):
        fingerprint = check.get(key)
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Preflight lacks a valid initial component fingerprint: " + key)
    if job["condition"] == "distillation":
        fingerprint = check.get("distillation_initial_state_sha256")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("Preflight lacks a valid distillation initialization fingerprint")
    return check


def verify_run(run, check, job, *, checkpoint_loader=None):
    """Verify validation-only receipts against the actual current preflight."""
    run = Path(run).resolve()
    result, config = records._read(run / "result.json"), records._read(run / "config.json")
    records._same("completed run", result.get("status"), "completed")
    records._same("diagnostic phase", result.get("evaluation_phase"), "validation_only")
    records._same("test isolation", result.get("test_evaluated"), False)
    if "test_patient" in result or (run / "test").exists():
        raise ValueError("Diagnostic runs must not contain test evaluations")
    records._same("worker recipe", _scientific(config), _scientific(check["config"]))
    for key in ("condition", "seed", "initial_state_sha256", "cohort_hash"):
        expected = job.get(key, check.get(key))
        records._same("run " + key, result.get(key), expected)
    records._same("frozen teacher", result.get("teacher_unchanged"), True)
    for key in ("initial_backbone_state_sha256", "initial_head_state_sha256", "distillation_initial_state_sha256"):
        records._same("initial component " + key, result.get(key), check.get(key))
    if not isinstance(result.get("diagnostics"), dict):
        raise ValueError("Missing clean evaluation proof in diagnostic receipt")
    for key in (*DIAGNOSTICS, "clean_train_eval", "training_rng_preserved", "l2_sp"):
        records._same("diagnostic proof " + key, result["diagnostics"].get(key), check["diagnostics"].get(key))
    if job["group"] == "D":
        anchor = result["diagnostics"].get("l2_sp", {})
        records._same("ImageNet backbone anchor", anchor.get("anchor_sha256"), result["initial_backbone_state_sha256"])
        records._same("fixed anchor", anchor.get("anchor_unchanged"), True)
    for file, key in (("teacher.json", "teacher"), ("model_metadata.json", "model_metadata"),
                       ("shared_view.json", "shared_view")):
        records._same("run " + key, records._read(run / file), check[key])
    history = records._read(run / "history.json")
    if not isinstance(history, list) or not history:
        raise ValueError("Missing clean training and validation history")
    required = {"epoch", "train_eval_loss", "train_eval_patient_roc_auc", "train_eval_patient_pr_auc",
                "val_loss", "val_patient_roc_auc", "val_patient_pr_auc"}
    if any(not isinstance(row, dict) or not required <= set(row) for row in history):
        raise ValueError("History lacks same-definition train/validation diagnostics")
    if (any(type(row[key]) not in (float, int) or not math.isfinite(row[key])
            for row in history for key in required)
            or [row["epoch"] for row in history] != list(range(1, len(history) + 1))):
        raise ValueError("Diagnostic history epochs/metrics are invalid")
    if job["group"] == "D":
        for row in history:
            raw, weighted = row.get("train_l2_sp_loss"), row.get("train_l2_sp_penalty")
            if (type(raw) not in (int, float) or type(weighted) not in (int, float)
                    or not math.isfinite(raw) or not math.isfinite(weighted)
                    or raw < 0 or weighted < 0
                    or not math.isclose(weighted, 1e-4 * raw, rel_tol=1e-5, abs_tol=1e-10)):
                raise ValueError("D must record the declared raw and weighted L2-SP penalty")
    best = max(history, key=lambda row: row["val_patient_roc_auc"])
    records._same("best epoch", result.get("best_epoch"), best["epoch"])
    records._same("last epoch", result.get("last_epoch"), history[-1]["epoch"])
    records._same("best validation selection", result.get("best_val_patient_roc_auc"), best["val_patient_roc_auc"])
    if len(history) > config["training"]["epochs"] or (len(history) < config["training"]["epochs"]
            and history[-1]["epoch"] - best["epoch"] != config["training"]["patience"]):
        raise ValueError("Completed history does not follow the saved early-stopping budget")
    for prefix, row in (("best", best), ("last", history[-1])):
        expected = {"auc_gap": row["train_eval_patient_roc_auc"] - row["val_patient_roc_auc"],
                    "loss_gap": row["val_loss"] - row["train_eval_loss"]}
        actual = result.get(prefix + "_gap", {})
        if any(type(actual.get(key)) not in (int, float) or not math.isclose(
                actual[key], value, rel_tol=1e-7, abs_tol=1e-9) for key, value in expected.items()):
            raise ValueError("Recorded clean train/validation gaps do not match history")
    for directory, key in (("validation", "validation_patient"), ("train_eval", "best_train_patient")):
        metrics = records._read(run / directory / "metrics.json")
        records._same("recorded " + directory + " metrics", metrics.get("patient"), result.get(key))
        if any(type(metrics.get("patient", {}).get(metric)) not in (float, int)
               or not math.isfinite(metrics["patient"][metric])
               or not 0 <= metrics["patient"][metric] <= 1
               for metric in ("roc_auc", "pr_auc", "average_precision", "sensitivity", "specificity")):
            raise ValueError("Diagnostic patient metrics are missing or invalid")
        for filename in ("predictions.csv", "patient_predictions.csv"):
            if not (run / directory / filename).is_file():
                raise ValueError("Missing diagnostic prediction artifact")
    records._same("selected validation ROC-AUC", result["validation_patient"]["roc_auc"], best["val_patient_roc_auc"])
    records._same("selected validation PR-AUC", result["validation_patient"]["pr_auc"], best["val_patient_pr_auc"])
    loader = checkpoint_loader or records._default_checkpoint_loader
    payload = loader(run / "best.pt")
    records._same("checkpoint recipe", _scientific(payload.get("config", {})), _scientific(config))
    for key in ("condition", "initial_state_sha256", "cohort_hash", "teacher", "shared_view", "model_metadata"):
        records._same("checkpoint " + key, payload.get(key), result.get(key, check.get(key)))
    records._same("checkpoint selected validation", payload.get("best_val_patient_roc_auc"), result["best_val_patient_roc_auc"])
    records._same("checkpoint threshold", payload.get("threshold"), result.get("threshold"))
    records._same("checkpoint diagnostic proof", payload.get("diagnostics"), result["diagnostics"])
    records._same("checkpoint diagnostic phase", payload.get("evaluation_phase"), "validation_only")
    records._same("checkpoint test isolation", payload.get("test_evaluated"), False)
    for key in ("initial_backbone_state_sha256", "initial_head_state_sha256", "distillation_initial_state_sha256"):
        records._same("checkpoint initial component " + key, payload.get(key), check.get(key))
    if not (run / "last.pt").is_file():
        raise ValueError("Missing final checkpoint")
    return {**result, "run": str(run), "checkpoint_sha256": records._file_hash(run / "best.pt"),
        "history_sha256": records._file_hash(run / "history.json")}


def _launch(job, action, project, directory):
    """Keep the original worker/config protocol and relay its live progress."""
    directory.mkdir(parents=True, exist_ok=False)
    config = deepcopy(job["config"])
    config["output_root"] = str(directory / "outputs")
    config_file, receipt_file = directory / "config.json", directory / "result.json"
    records._json(config_file, config)
    command = [sys.executable, "-u", str(project / "run_single_teacher_distillation.py"),
        "--project", str(project), "--config", str(config_file), "--condition", job["condition"],
        "--result-json", str(receipt_file)]
    if action == "check":
        command.append("--check-only")
    log = directory / f"{action}.log"
    print(f"[{action}] {job['id']} {job['device']} log={log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f"[{job['id']}] " + line.rstrip("\n"), flush=True)
            code = process.wait()
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Process exited {code}; see {log}")
    if not receipt_file.is_file():
        raise RuntimeError(f"Worker did not write {receipt_file}")
    return records._read(receipt_file)


def _live_workers(study):
    found = []
    for process in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = process.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            if "run_single_teacher_distillation.py" in args and str(study / "jobs") in args:
                found.append(int(process.parent.name))
        except (OSError, ValueError):
            continue
    return found


@contextmanager
def _lock(study):
    import fcntl
    with (study / ".execution.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This single-teacher diagnostic study is already running") from error
        try:
            active = _live_workers(study)
            if active:
                raise ValueError("Existing workers still own this study; keep them running: " + str(active))
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_study(plan, *, resume_study=None, check_only=False, launcher=None, verifier=None, reporter=None):
    project = Path(plan["project"])
    signature = protocol_hash(plan)
    if resume_study:
        study = records._path(resume_study, project)
        if not study.is_dir():
            raise ValueError("The requested study directory does not exist")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        study = Path(plan["output_root"]) / "studies" / f"study_{stamp}_{uuid.uuid4().hex[:8]}"
        study.mkdir(parents=True)
    with _lock(study):
        validate_source(plan)
        if resume_study:
            state = records._read(study / "study.json")
            records._same("saved format", state.get("format_version"), FORMAT)
            records._same("frozen scientific protocol", state.get("protocol_sha256"), signature)
            records._same("saved job set", sorted(state["jobs"]), sorted(job["id"] for job in plan["jobs"]))
        else:
            state = {"format_version": FORMAT, "protocol_sha256": signature,
                "jobs": {job["id"]: {"job": deepcopy(job), "status": "pending", "attempts": []} for job in plan["jobs"]}}
        previous = {key: deepcopy(record.get("receipt")) for key, record in state["jobs"].items()
                    if record.get("status") in SUCCESS and record.get("receipt")}
        state.update(plan=deepcopy(plan), status="running", check_only=check_only, study=str(study))
        state.setdefault("executions", []).append({"started_at": datetime.now(timezone.utc).isoformat(),
            "devices": plan["devices"], "max_parallel": plan["max_parallel"], "check_only": check_only})

        mutation_lock = threading.RLock()

        def persist():
            with mutation_lock:
                records._json(study / "study.json", state)

        def invoke(job, action):
            directory = study / "jobs" / job["id"] / f"{action}_{uuid.uuid4().hex[:8]}"
            with mutation_lock:
                record = state["jobs"][job["id"]]
                record.update(status="checking" if action == "check" else "training")
                record["attempts"].append({"phase": action, "directory": str(directory)})
                persist()
            return (launcher or _launch)(job, action, project, directory)

        def checked(job, value, error):
            record = state["jobs"][job["id"]]
            try:
                if error is not None:
                    raise error
                record["check"] = _checked(value, job, plan["source"])
                record.update(status="checked")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", error=str(invalid))
            print(f"[check {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
            persist()

        def trained(job, value, error):
            record = state["jobs"][job["id"]]
            try:
                if error is not None:
                    raise error
                raw = records._train_receipt(value, job["condition"])
                record["receipt"] = (verifier or verify_run)(raw["run"], record["check"], job)
                record.update(status="completed")
                record.pop("error", None)
            except Exception as invalid:
                record.update(status="failed", error=str(invalid))
            print(f"[train {record['status']}] {job['id']}" + (": " + record["error"] if record.get("error") else ""), flush=True)
            persist()

        def serialized(callback):
            def wrapped(*args):
                with mutation_lock:
                    return callback(*args)
            return wrapped

        print(f"Study directory: {study}", flush=True)
        print("Four fresh ImageNet students; single frozen RAD-DINO; validation only; no test evaluation.", flush=True)
        persist()
        try:
            records._parallel(plan["jobs"], plan["max_parallel"], lambda job: invoke(job, "check"), serialized(checked))
            # The full-KD and L2-SP pair must also share their random projector.
            b, d = (state["jobs"][group + "_seed42"] for group in ("B", "D"))
            if b["status"] == d["status"] == "checked":
                if not b["check"].get("distillation_initial_state_sha256"):
                    raise ValueError("B/D preflights must fingerprint the initial KD projector")
                records._same("paired initial KD projector", b["check"].get("distillation_initial_state_sha256"),
                              d["check"].get("distillation_initial_state_sha256"))
            pending = []
            for job in plan["jobs"]:
                record = state["jobs"][job["id"]]
                if record["status"] != "checked":
                    continue
                if job["id"] in previous:
                    saved = previous[job["id"]]
                    receipt = (verifier or verify_run)(saved["run"], record["check"], job)
                    for key in ("checkpoint_sha256", "history_sha256"):
                        records._same("completed artifact " + key, receipt.get(key), saved.get(key))
                    record.update(status="resumed", receipt=receipt)
                    print("[resumed] " + job["id"], flush=True)
                else:
                    # Recover a completed child if its coordinator exited before
                    # accepting the receipt. Only declared attempts are scanned.
                    for attempt in reversed(record["attempts"]):
                        file = Path(attempt["directory"]) / "result.json"
                        if attempt["phase"] != "train" or not file.is_file():
                            continue
                        try:
                            raw = records._train_receipt(records._read(file), job["condition"])
                            receipt = (verifier or verify_run)(raw["run"], record["check"], job)
                        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                            record.setdefault("recovery_rejections", []).append({"file": str(file), "reason": str(error)})
                            continue
                        record.update(status="resumed", receipt=receipt)
                        print("[adopted completed attempt] " + job["id"], flush=True)
                        break
                    else:
                        if not check_only:
                            pending.append(job)
            persist()
            if pending:
                records._parallel(pending, plan["max_parallel"], lambda job: invoke(job, "train"), serialized(trained))
            validate_source(plan)
            statuses = [record["status"] for record in state["jobs"].values()]
            if all(status in SUCCESS for status in statuses):
                state["status"] = "completed"
            elif check_only and all(status in SUCCESS | {"checked"} for status in statuses):
                state["status"] = "checked"
            else:
                state["status"] = "partial" if any(status in SUCCESS | {"checked"} for status in statuses) else "failed"
        except BaseException as error:
            state.update(status="interrupted", error=str(error))
            raise
        finally:
            state["executions"][-1].update(finished_at=datetime.now(timezone.utc).isoformat(), status=state["status"])
            persist()
        if reporter:
            reporter(study, state)
        else:
            from .single_teacher_overfit_report import print_metrics, write_reports, plot_curves
            write_reports(study, state)
            print_metrics(state)
            if any(record.get("receipt") for record in state["jobs"].values()):
                print(plot_curves(study, state), flush=True)
        print(f"Single-teacher diagnostics {state['status']}: {study}", flush=True)
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-suite")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    actions = parser.add_mutually_exclusive_group()
    for name in ("plan", "check-only", "show-metrics", "plot-curves"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--study")
    parser.add_argument("--resume-study")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).expanduser().resolve()
        if args.show_metrics or args.plot_curves:
            if not args.study or args.resume_study:
                raise ValueError("Read-only reporting requires --study PATH without --resume-study")
            study = records._path(args.study, project)
            state = records._read(study / "study.json")
            records._same("saved format", state.get("format_version"), FORMAT)
            from .single_teacher_overfit_report import print_metrics, plot_curves
            print(plot_curves(study, state)) if args.plot_curves else print_metrics(state)
            return 0
        if args.study:
            raise ValueError("--study is read-only; continue a study using --resume-study")
        if args.resume_study:
            study = records._path(args.resume_study, project)
            saved = records._read(study / "study.json")
            plan = deepcopy(saved["plan"])
            records._same("saved project", plan["project"], str(project))
            if args.source_suite or args.cache_dir:
                raise ValueError("Resume uses its saved source and cache; override only devices/concurrency")
            devices, maximum, _ = _runtime(plan, project, args.devices, args.max_parallel)
            plan.update(devices=devices, max_parallel=maximum)
            for index, job in enumerate(plan["jobs"]):
                job["device"] = job["config"]["device"] = devices[index % len(devices)]
        else:
            plan = make_plan(args.config, project, source_suite=args.source_suite, devices=args.devices,
                max_parallel=args.max_parallel, cache_dir=args.cache_dir)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        study = run_study(plan, resume_study=args.resume_study, check_only=args.check_only)
        return int(records._read(study / "study.json")["status"] in {"partial", "failed", "interrupted"})
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print("Single-teacher diagnostics failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
