"""Read and audit the frozen random-student baseline for a WD-only study.

Planning uses JSON and fingerprints only. Complete tensor/checkpoint validation
is deferred until execution, keeping ``--plan`` and reports usable without Torch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from . import random_student_suite as random


SEEDS = (42, 43, 44)
JOB_IDS = tuple(f"{group}_seed{seed}" for group in ("A", "B") for seed in SEEDS)
LOSS_SETTINGS = {"temperature": 2.0, "soft_weight": 0.5, "global_weight": 0.1,
                 "local_weight": 1.0, "pool_grid": [7, 7]}
FILES = ("config.json", "best.pt", "result.json", "history.json",
         "model_metadata.json", "teacher.json", "shared_view.json",
         "test/metrics.json", "validation_predictions.csv",
         "test/predictions.csv", "test/patient_predictions.csv")


def _json(path):
    data = Path(path).read_bytes()
    if not data:
        raise ValueError(f"Baseline file is empty: {path}")
    return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()


def _fingerprint(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty baseline file: {path}")
    return random.legacy._file_hash(path)


def _same(name, actual, expected):
    random.legacy._same("weight-decay study " + name, actual, expected)


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Baseline {name} must be a complete SHA256")
    return value


def _weight_decay(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Baseline effective weight_decay must be finite and nonnegative")
    return value


def _contract(config, job):
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError("Baseline requires a resolved student training configuration")
    _same("random initialization", config.get("student_initialization"), "random")
    _same("no student source checkpoint", config.get("student_checkpoint"), None)
    _same("full student training", random.legacy._student_mode(config), "full_finetune")
    if config["training"].get("amp") is not False:
        raise ValueError("Baseline requires explicit FP32 training.amp=false")
    _same("student seed", config.get("seed"), job["seed"])
    _same("distillation settings", config.get("distillation"), LOSS_SETTINGS)
    _weight_decay(config.get("training", {}).get("weight_decay"))


def _history(history, result):
    if not isinstance(history, list) or not history:
        raise ValueError("Baseline training history must contain completed epochs")
    previous = 0
    for row in history:
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] <= previous:
            raise ValueError("Baseline history epochs must be strictly increasing")
        previous = row["epoch"]
        for key in ("train_eval_loss", "val_loss", "train_eval_patient_roc_auc", "val_patient_roc_auc"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Baseline history has an invalid " + key)
            if key.endswith("roc_auc") and value > 1:
                raise ValueError("Baseline history AUC must be in [0, 1]")
    best = max(row["val_patient_roc_auc"] for row in history)
    _same("history validation selection", best, result.get("best_val_patient_roc_auc"))


def load_baseline(suite, project):
    """Snapshot exactly six completed, matched random A/B experiments, read-only."""
    project = Path(project).expanduser().resolve()
    suite = random.legacy._path(suite, project)
    state, suite_sha = _json(suite / "suite.json")
    if not isinstance(state, dict) or state.get("format_version") != random.FORMAT:
        raise ValueError("Baseline must be a random_student_suite_v1 batch")
    records = state.get("jobs")
    if not isinstance(records, dict) or set(records) != set(JOB_IDS):
        raise ValueError("Baseline must contain exactly A/B x seeds42,43,44")
    snapshot = {"suite": str(suite), "project": str(project), "weight_decay": None,
                "jobs": {}, "fingerprints": {"suite.json": suite_sha}}
    reference = None
    initialized = {}
    for name in JOB_IDS:
        record = records[name]
        if not isinstance(record, dict) or record.get("status") not in random.SUCCESS:
            raise ValueError("Baseline task is incomplete: " + name)
        job = record.get("job", {})
        group, seed_text = name.split("_seed")
        expected = {"id": name, "group": group, "seed": int(seed_text),
                    "student_mode": "full_finetune", "condition": random.GROUPS[group][1]}
        for key, value in expected.items():
            _same("baseline job " + key, job.get(key), value)
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("run"):
            raise ValueError("Baseline task has no completed run: " + name)
        run = random.legacy._path(result["run"], suite)
        payload, fingerprints = {}, {}
        for relative in FILES:
            if relative.endswith(".json"):
                payload[relative], fingerprints[relative] = _json(run / relative)
            else:
                fingerprints[relative] = _fingerprint(run / relative)
        config, receipt = payload["config.json"], payload["result.json"]
        if not isinstance(receipt, dict):
            raise ValueError("Baseline completed receipt must be a JSON object")
        _contract(config, job)
        check = random._checked(record.get("check", {}), job)
        random.legacy._validate_protocol(config, check["config"], job["condition"])
        _contract(check["config"], job)
        _same("saved random experiment", receipt.get("experiment"), "random_student_kd_v1")
        _same("saved result initialization", receipt.get("student_initialization"), "random")
        _same("saved result mode", receipt.get("student_mode"), "full_finetune")
        _same("saved result status", receipt.get("status"), "completed")
        _same("saved result condition", receipt.get("condition"), job["condition"])
        _same("saved result seed", receipt.get("seed"), job["seed"])
        _same("frozen teacher", receipt.get("teacher_unchanged"), True)
        for key, value in receipt.items():
            if key == "run":
                _same("saved run path", str(random.legacy._path(value, suite)), str(run))
            else:
                _same("manifest result " + key, result.get(key), value)
        _same("student metadata", payload["model_metadata.json"], check["model_metadata"])
        _same("saved student update mode", check["model_metadata"].get("mode"), "full_finetune")
        _same("teacher metadata", payload["teacher.json"], check["teacher"])
        _same("shared view", payload["shared_view.json"], check["shared_view"])
        _same("cohort", receipt.get("cohort_hash"), check["cohort_hash"])
        _same("initial student", receipt.get("initial_state_sha256"), check["initial_state_sha256"])
        _sha(check["initial_state_sha256"], "initial student fingerprint")
        _same("saved checkpoint fingerprint", result.get("checkpoint_sha256"), fingerprints["best.pt"])
        _sha(check["teacher"].get("checkpoint_sha256"), "teacher checkpoint fingerprint")
        if not check["teacher"].get("checkpoint"):
            raise ValueError("Baseline lacks the teacher checkpoint path")
        _same("resolved teacher checkpoint", str(random.legacy._path(config.get("teacher_checkpoint"), project)),
              str(random.legacy._path(check["teacher"]["checkpoint"], project)))
        projection = check.get("distillation_initial_state_sha256")
        _same("initial projector", receipt.get("distillation_initial_state_sha256"), projection)
        if job["condition"] == "distillation":
            _sha(projection, "initial projector fingerprint")
        elif projection is not None:
            raise ValueError("A supervised baseline cannot contain a KD projector")
        metrics = payload["test/metrics.json"]
        _same("saved test metrics", metrics.get("patient"), receipt.get("test_patient"))
        _same("saved confidence interval", metrics.get("bootstrap", {}).get("patient", {}).get("roc_auc"), receipt.get("roc_auc_ci"))
        _history(payload["history.json"], receipt)
        protocol = random._cross_protocol(check)
        if reference is not None:
            _same("shared baseline effective protocol", protocol, reference)
        reference = protocol
        if job["seed"] in initialized:
            _same("paired initial student", check["initial_state_sha256"], initialized[job["seed"]])
        initialized[job["seed"]] = check["initial_state_sha256"]
        snapshot["jobs"][name] = {"job": deepcopy(job), "check": deepcopy(check),
            "result": deepcopy(result), "config": config, "run": str(run), "fingerprints": fingerprints}
    snapshot["weight_decay"] = _weight_decay(reference["training"]["weight_decay"])
    return snapshot


def validate_snapshot(snapshot, *, verify_runs=False, verifier=None):
    """Reject changed source artifacts; optionally restore every complete student."""
    fresh = load_baseline(snapshot["suite"], snapshot["project"])
    _same("immutable baseline snapshot", fresh, snapshot)
    if not verify_runs:
        return
    verifier = verifier or random.verify_run
    teachers = {}
    for entry in snapshot["jobs"].values():
        verified = verifier(entry["run"], entry["check"], entry["job"])
        if not isinstance(verified, dict):
            raise ValueError("Baseline checkpoint verifier returned no complete receipt")
        _same("complete baseline run", verified, entry["result"])
        teacher = entry["check"]["teacher"]
        path = random.legacy._path(teacher["checkpoint"], Path(snapshot["project"]))
        expected = teacher["checkpoint_sha256"]
        if str(path) not in teachers:
            teachers[str(path)] = _fingerprint(path)
        _same("actual teacher checkpoint fingerprint", teachers[str(path)], expected)


def audit_check(value, job, baseline):
    """Require a new preflight to differ from its source only in weight decay."""
    check = random._checked(value, job)
    if job.get("id") not in baseline["jobs"]:
        raise ValueError("Weight-decay candidate is not a confirmed baseline job")
    source = baseline["jobs"][job["id"]]
    for key in ("id", "group", "seed", "student_mode", "condition"):
        _same("candidate job " + key, job.get(key), source["job"].get(key))
    _contract(check["config"], job)
    desired = job["config"]["training"].get("weight_decay")
    _weight_decay(desired)
    _same("explicit candidate weight_decay", check["config"]["training"]["weight_decay"], desired)
    left, right = random._cross_protocol(check), random._cross_protocol(source["check"])
    left["training"].pop("weight_decay")
    right["training"].pop("weight_decay")
    _same("only weight decay may differ", left, right)
    _same("unchanged student model", check["model_metadata"], source["check"]["model_metadata"])
    _same("same initial student", check["initial_state_sha256"], source["check"]["initial_state_sha256"])
    _same("same initial KD projector", check.get("distillation_initial_state_sha256"),
          source["check"].get("distillation_initial_state_sha256"))
    return check
