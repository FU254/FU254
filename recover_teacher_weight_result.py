#!/usr/bin/env python3
"""Verify and register an already-completed teacher-weight result; never train."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid


def recover(study, *, project=None, group="P-R", expected_checkpoint_sha256=None,
            verifier=None, reporter=None):
    # Import the installed implementation. This utility also works with the
    # originally released parallel module: it never calls run_parallel(), the
    # process scanner, GPU preflight, model training or test evaluation.
    project = Path(project or Path(__file__).resolve().parent).expanduser().resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from foundation_sft import teacher_weight_parallel as parallel
    from foundation_sft import teacher_weight_study as control

    study = Path(study).expanduser().resolve()
    if not (study / "study.json").is_file():
        raise ValueError("The exact study.json is missing")
    if expected_checkpoint_sha256 is not None and (
            len(expected_checkpoint_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_checkpoint_sha256)):
        raise ValueError("Expected checkpoint SHA256 must be 64 lowercase hexadecimal characters")
    with control._study_lock(study):
        plan = parallel.make_plan(study, group, "cuda:0", project, allow_running=True)
        group_path = study / "jobs" / group
        with (group_path / ".parallel_execution.lock").open("a+b") as handle:
            os.set_inheritable(handle.fileno(), False)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("The group is still executing; its current worker was left running") from error
            try:
                registry = control.legacy._read(group_path / parallel.REGISTRY)
                control.legacy._same("recovery requires a completed registry", registry.get("status"), "completed")
                result = parallel._load_completed(plan, verifier or parallel._default_verify)
                if result is None:
                    raise ValueError("No verified completed result exists; no training was started")
                if expected_checkpoint_sha256 is not None:
                    control.legacy._same("explicitly pinned recovery checkpoint",
                        result.get("checkpoint_sha256"), expected_checkpoint_sha256)
                state_path = study / "study.json"
                original = state_path.read_bytes()
                state = json.loads(original)
                record = state["main"][group]
                if record.get("status") in control.SUCCESS:
                    control.legacy._same("already registered result", record.get("result"), result)
                    print(f"[recovery] {group} was already registered; no state changes", flush=True)
                else:
                    backup = study / ("study.json.before_result_recovery_" + uuid.uuid4().hex + ".bak")
                    with backup.open("xb") as stream:
                        stream.write(original)
                    audit = {"action": "adopt_verified_completed_result", "group": group,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "previous_status": record.get("status"), "previous_error": record.get("error"),
                        "run": result["run"], "checkpoint_sha256": result["checkpoint_sha256"],
                        "receipt_sha256": registry["receipt_sha256"], "registry": str(group_path / parallel.REGISTRY),
                        "backup": str(backup), "training_performed": False,
                        "test_evaluation_performed": False}
                    record.setdefault("recovery_history", []).append(deepcopy(audit))
                    record.update(status="resumed", phase="verify_completed_result", result=deepcopy(result))
                    record.pop("error", None)
                    state.setdefault("recovery_history", []).append(deepcopy(audit))
                    state["status"] = ("completed" if len(state["main"]) == 6 and all(
                        item.get("status") in control.SUCCESS for item in state["main"].values()) else "partial")
                    state["updated_at"] = audit["time"]
                    # The original study lock excludes the coordinator. The
                    # normal atomic writer keeps a partially-written JSON out
                    # of readers, and the exact prior bytes remain in backup.
                    if state_path.read_bytes() != original:
                        raise ValueError("Study changed during verification; the new state was not published")
                    control.legacy._json(state_path, state)
                    print(f"[recovery] {group} registered without training or test reevaluation", flush=True)
                    print(f"[recovery] Original state backup: {backup}", flush=True)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        (reporter or control._report)(study, state)
        print(f"[recovery] Study status: {state['status']}", flush=True)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--group", choices=("R-C", "R-R", "P-C", "P-R"), default="P-R")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args(argv)
    try:
        recover(args.study, project=args.project, group=args.group,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Completed-result recovery failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
