"""Pin the completed WD=0.01 A/B seed42 control for a DropPath-only study.

Planning reads JSON and file hashes. Full student reconstruction and teacher
file verification are deliberately deferred until execution.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import random_student_suite as random
from . import weight_decay_baseline as wd
from . import weight_decay_study as wd_study


JOB_IDS = ("A_seed42", "B_seed42")
WEIGHT_DECAY = 0.01
DROP_PATH_RATE = 0.1
REGULARIZATION = {"drop_path_rate": DROP_PATH_RATE, "schedule": "linear_12_blocks",
                  "rng": "isolated_torch", "rng_seed": 42 ^ 0x44504154}


def _same(name, actual, expected):
    random.legacy._same("DropPath study " + name, actual, expected)


def _zero_drop_path(config, metadata):
    value = config["training"].get("drop_path_rate", 0.0)
    if type(value) not in (int, float) or value != 0:
        raise ValueError("The DropPath control must have drop_path_rate=0")
    if "regularization" in metadata:
        raise ValueError("The control must use the legacy zero-DropPath student metadata")


def _parent(study, project):
    state, digest = wd._json(study / "study.json")
    if not isinstance(state, dict) or state.get("format_version") != wd_study.FORMAT:
        raise ValueError("DropPath control must come from a completed weight-decay study")
    _same("completed source study", state.get("status"), "completed")
    _same("source study seeds", state.get("seeds"), [42])
    _same("source weight-decay candidates", state.get("weight_decays"), [0.001, 0.01])
    _same("source selection rule", state.get("selection_rule"), wd_study.SELECTION_RULE)
    plan = state.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("The source study lacks its frozen scientific plan")
    _same("source project", str(Path(plan["project"]).resolve()), str(project))
    _same("source study protocol", state.get("protocol_sha256"), wd_study.protocol_hash(plan))
    for name in ("baseline", "weight_decays", "seeds", "selection_rule"):
        _same("source frozen " + name, state.get(name), plan.get(name))
    arms = state.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"wd_0p001", "wd_0p01"}:
        raise ValueError("The source study must retain both confirmed weight-decay arms")
    for name, expected in (("wd_0p001", 0.001), ("wd_0p01", 0.01)):
        arm = arms[name]
        _same("completed source arm " + name, arm.get("status"), "completed")
        _same("source arm weight decay " + name, arm.get("weight_decay"), expected)
        _same("source arm plan " + name, random.protocol_hash(arm["plan"]),
              random.protocol_hash(plan["arms"][name]["plan"]))
    arm = arms["wd_0p01"]
    suite = random.legacy._path(arm.get("suite"), study)
    expected_suite = study / "arms" / "wd_0p01" / "suite"
    _same("pinned WD=0.01 source suite", str(suite), str(expected_suite.resolve()))
    return state, arm, suite, digest


def load_baseline(study, project):
    """Snapshot the fixed WD=0.01, DropPath=0 A/B seed42 control read-only."""
    project = Path(project).expanduser().resolve()
    study = random.legacy._path(study, project)
    parent, arm, suite, parent_sha = _parent(study, project)
    state, suite_sha = wd._json(suite / "suite.json")
    if not isinstance(state, dict) or state.get("format_version") != random.FORMAT:
        raise ValueError("The source arm must be a random_student_suite_v1 batch")
    _same("completed source batch", state.get("status"), "completed")
    expected_protocol = random.protocol_hash(arm["plan"])
    _same("source child protocol", state.get("protocol_sha256"), expected_protocol)
    _same("source recorded child plan", random.protocol_hash(state["plan"]), expected_protocol)
    records = state.get("jobs")
    if not isinstance(records, dict) or set(records) != set(JOB_IDS):
        raise ValueError("The source control must contain exactly A/B seed42")
    planned = {job["id"]: job for job in arm["plan"]["jobs"]}
    if set(planned) != set(JOB_IDS):
        raise ValueError("The pinned source arm must plan exactly A/B seed42")
    snapshot = {"study": str(study), "suite": str(suite), "project": str(project),
        "weight_decay": WEIGHT_DECAY, "drop_path_rate": 0.0, "jobs": {},
        "fingerprints": {"study.json": parent_sha, "suite.json": suite_sha}}
    reference, initial = None, None
    for name in JOB_IDS:
        record = records[name]
        if not isinstance(record, dict) or record.get("status") not in random.SUCCESS:
            raise ValueError("The source DropPath control is incomplete: " + name)
        job = record.get("job", {})
        group = name[0]
        expected = {"id": name, "group": group, "seed": 42, "student_mode": "full_finetune",
                    "condition": random.GROUPS[group][1]}
        for key, value in expected.items():
            _same("control job " + key, job.get(key), value)
        random.legacy._validate_protocol(job["config"], planned[name]["config"], job["condition"])
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("run"):
            raise ValueError("The source control lacks a completed run: " + name)
        run = random.legacy._path(result["run"], suite)
        if not run.is_relative_to(suite):
            raise ValueError("The DropPath control run must belong to its pinned source suite")
        payload, fingerprints = {}, {}
        for relative in wd.FILES:
            if relative.endswith(".json"):
                payload[relative], fingerprints[relative] = wd._json(run / relative)
            else:
                fingerprints[relative] = wd._fingerprint(run / relative)
        config, receipt = payload["config.json"], payload["result.json"]
        if not isinstance(receipt, dict):
            raise ValueError("The source completed receipt must be a JSON object")
        wd._contract(config, job)
        _same("fixed weight decay", config["training"]["weight_decay"], WEIGHT_DECAY)
        check = random._checked(record.get("check", {}), job)
        wd._contract(check["config"], job)
        random.legacy._validate_protocol(config, check["config"], job["condition"])
        _zero_drop_path(config, check["model_metadata"])
        _zero_drop_path(check["config"], check["model_metadata"])
        for key, expected_value in (("experiment", "random_student_kd_v1"),
                ("student_initialization", "random"), ("student_mode", "full_finetune"),
                ("status", "completed"), ("condition", job["condition"]),
                ("seed", 42), ("teacher_unchanged", True)):
            _same("saved control result " + key, receipt.get(key), expected_value)
        for key, value in receipt.items():
            if key == "run":
                _same("saved control run", str(random.legacy._path(value, suite)), str(run))
            else:
                _same("control manifest result " + key, result.get(key), value)
        _same("control checkpoint fingerprint", result.get("checkpoint_sha256"), fingerprints["best.pt"])
        _same("control student metadata", payload["model_metadata.json"], check["model_metadata"])
        _same("control student mode", check["model_metadata"].get("mode"), "full_finetune")
        _same("control teacher", payload["teacher.json"], check["teacher"])
        _same("control shared view", payload["shared_view.json"], check["shared_view"])
        _same("control cohort", receipt.get("cohort_hash"), check["cohort_hash"])
        _same("control initial student", receipt.get("initial_state_sha256"), check["initial_state_sha256"])
        wd._sha(check["initial_state_sha256"], "initial student")
        wd._sha(check["teacher"].get("checkpoint_sha256"), "teacher checkpoint")
        _same("control resolved teacher checkpoint",
              str(random.legacy._path(config.get("teacher_checkpoint"), project)),
              str(random.legacy._path(check["teacher"].get("checkpoint"), project)))
        projection = check.get("distillation_initial_state_sha256")
        _same("control initial projector", receipt.get("distillation_initial_state_sha256"), projection)
        if job["condition"] == "distillation":
            wd._sha(projection, "initial projector")
        elif projection is not None:
            raise ValueError("Supervised DropPath control must not have a KD projector")
        metrics = payload["test/metrics.json"]
        _same("control test metrics", metrics.get("patient"), receipt.get("test_patient"))
        _same("control interval", metrics.get("bootstrap", {}).get("patient", {}).get("roc_auc"), receipt.get("roc_auc_ci"))
        wd._history(payload["history.json"], receipt)
        protocol = random._cross_protocol(check)
        if reference is not None:
            _same("paired control scientific protocol", protocol, reference)
            _same("paired control initial student", check["initial_state_sha256"], initial)
        reference, initial = protocol, check["initial_state_sha256"]
        snapshot["jobs"][name] = {"job": deepcopy(job), "check": deepcopy(check),
            "result": deepcopy(result), "config": config, "run": str(run), "fingerprints": fingerprints}
    return snapshot


def validate_snapshot(snapshot, *, verify_runs=False, verifier=None):
    """Reject changed controls and optionally reconstruct their complete students."""
    fresh = load_baseline(snapshot["study"], snapshot["project"])
    _same("immutable DropPath baseline snapshot", fresh, snapshot)
    if not verify_runs:
        return
    verifier = verifier or random.verify_run
    teachers = {}
    for entry in snapshot["jobs"].values():
        verified = verifier(entry["run"], entry["check"], entry["job"])
        _same("complete control checkpoint", verified, entry["result"])
        teacher = entry["check"]["teacher"]
        path = random.legacy._path(teacher["checkpoint"], Path(snapshot["project"]))
        if str(path) not in teachers:
            teachers[str(path)] = wd._fingerprint(path)
        _same("actual frozen teacher fingerprint", teachers[str(path)], teacher["checkpoint_sha256"])


def audit_check(value, job, baseline):
    """Accept only DP=0.1 with unchanged source weights, data and training recipe."""
    check = random._checked(value, job)
    if job.get("id") not in baseline["jobs"]:
        raise ValueError("Candidate is not one of the confirmed A/B seed42 jobs")
    source = baseline["jobs"][job["id"]]
    for key in ("id", "group", "seed", "student_mode", "condition"):
        _same("candidate job " + key, job.get(key), source["job"].get(key))
    wd._contract(check["config"], job)
    _same("candidate fixed weight decay", check["config"]["training"]["weight_decay"], WEIGHT_DECAY)
    for config in (job["config"], check["config"]):
        rate = config.get("training", {}).get("drop_path_rate")
        if type(rate) not in (int, float) or rate != DROP_PATH_RATE:
            raise ValueError("This controlled candidate requires training.drop_path_rate=0.1")
    left, right = random._cross_protocol(check), random._cross_protocol(source["check"])
    left["training"].pop("drop_path_rate", None)
    right["training"].pop("drop_path_rate", None)
    _same("only DropPath may differ", left, right)
    metadata = deepcopy(check["model_metadata"])
    _same("candidate DropPath semantics", metadata.pop("regularization", None), REGULARIZATION)
    _same("unchanged student architecture and initialization", metadata, source["check"]["model_metadata"])
    _same("same initial student parameters", check["initial_state_sha256"], source["check"]["initial_state_sha256"])
    _same("same initial KD projector", check.get("distillation_initial_state_sha256"),
          source["check"].get("distillation_initial_state_sha256"))
    return check
