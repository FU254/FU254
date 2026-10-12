"""Resume one failed experiment and report its explicit replacement separately.

The original suite, logs and runs remain untouched. Child commands execute in
the job's Conda environment, so this coordinator only needs the standard library.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid


_IDENTITY = ("model", "mode", "seed")
_FINGERPRINTS = ("_cohort_hash", "_source_hash", "_manifest_hash")
_ARTIFACTS = (
    "best.pt", "last.pt", "config.json", "environment.json",
    "model_metadata.json", "history.json", "selection.json",
    "split_metadata.json", "test/metrics.json", "test/predictions.csv",
    "test/patient_predictions.csv",
)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing required file or directory path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _suite_job(suite, job_id):
    manifest = _read_json(suite / "suite.json")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("suite.json must contain a nonempty jobs list")
    ids, identities = set(), set()
    for job in jobs:
        if not isinstance(job, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(job.get("id", ""))):
            raise ValueError("Suite jobs must have safe, nonempty IDs")
        identity = tuple(job.get(key) for key in _IDENTITY)
        if job["id"] in ids or identity in identities:
            raise ValueError("Suite has duplicate job IDs or model/mode/seed experiments")
        ids.add(job["id"])
        identities.add(identity)
    matches = [job for job in jobs if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Job not found in suite: {job_id}")
    project = _path(manifest.get("project_dir"), suite)
    if not project.is_dir():
        raise ValueError(f"Original project directory is unavailable: {project}")
    return manifest, matches[0], project


def _check_identity(value, job, description):
    if any(value.get(key) != job.get(key) for key in _IDENTITY):
        raise ValueError(f"{description} belongs to another model/mode/seed experiment")


def _source(suite, job):
    config_file = _path(job.get("config_file"), suite)
    config = _read_json(config_file)
    _check_identity(config, job, "Source configuration")
    if config.get("device") != job.get("device"):
        raise ValueError("Source configuration device differs from the job")
    if config.get("training", {}).get("amp") is not True:
        raise ValueError("Full-precision recovery requires an AMP-enabled source configuration")
    log = _path(job.get("train_log"), suite)
    directories = {
        _path(match.group(1).strip(), suite)
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if (match := re.fullmatch(r"Run directory:\s*(.+)", line.strip()))
    }
    if len(directories) != 1:
        raise ValueError("Training log must identify one unambiguous Run directory")
    run = directories.pop()
    checkpoint = run / "last.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Last complete-epoch checkpoint is missing: {checkpoint}")
    saved_config = _read_json(run / "config.json")
    if saved_config != config:
        raise ValueError("Failed run configuration differs from its suite snapshot")
    return config_file, config, run, checkpoint.resolve()


def _fingerprint(run):
    metadata = _read_json(run / "split_metadata.json")
    values = tuple(metadata.get(key) for key in _FINGERPRINTS)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"Full cohort/source/split fingerprints are missing: {run}")
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run(run, job):
    if not run.is_absolute() or any(not (run / name).is_file() for name in _ARTIFACTS):
        raise ValueError(f"Completed run is missing required training/evaluation artifacts: {run}")
    _check_identity(_read_json(run / "config.json"), job, "Saved run")
    return _fingerprint(run)


def _validate_recovery(receipt_file, job, source_config, checkpoint, source_run):
    receipt = _read_json(receipt_file)
    if receipt.get("action") != "train":
        raise ValueError("Recovery receipt must record successful training")
    _check_identity(receipt, job, "Recovery receipt")
    value = receipt.get("run_dir")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Recovery receipt must name an absolute run_dir")
    run = Path(value).resolve()
    if run == source_run:
        raise ValueError("Recovery must use a new run directory")
    _validate_run(run, job)
    expected = deepcopy(source_config)
    expected["training"]["amp"] = False
    if _read_json(run / "config.json") != expected:
        raise ValueError("Recovery configuration must change only AMP to full precision")
    resume = _read_json(run / "resume.json")
    if (resume.get("checkpoint") != str(checkpoint)
            or resume.get("checkpoint_sha256") != _sha256(checkpoint)
            or resume.get("precision_changed") is not True
            or resume.get("source_amp") is not True or resume.get("target_amp") is not False):
        raise ValueError("resume.json must verify the source checkpoint SHA256 and AMP-to-FP32 precision change")
    if _fingerprint(run) != _fingerprint(source_run):
        raise ValueError("Recovery changed the full cohort/source/split fingerprints")
    return run


def _command(command, log, project):
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=project, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Process exited {code}; see {log}")


def recover(suite, job_id, conda):
    manifest, job, project = _suite_job(suite, job_id)
    if job.get("status") != "failed" or job.get("train_status") != "failed":
        raise ValueError("Recovery requires a job whose training has failed")
    lock = suite / f".recovery_{job_id}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"Another recovery holds this job lock: {lock}") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "job_id": job_id, "device": job.get("device")}, handle)
        # Re-read after taking the lock: other original jobs may still progress.
        manifest, job, project = _suite_job(suite, job_id)
        if job.get("status") != "failed" or job.get("train_status") != "failed":
            raise ValueError("Job is no longer a failed training experiment")
        if any(other["id"] != job_id and other.get("device") == job.get("device")
               and other.get("train_status") == "running" for other in manifest["jobs"]):
            raise ValueError("Another training job is running on the same GPU")
        config_file, config, source_run, checkpoint = _source(suite, job)
        _fingerprint(source_run)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        recovery_dir = suite / f"recovery_{stamp}_{uuid.uuid4().hex[:8]}"
        recovery_dir.mkdir(exist_ok=False)
        receipt = recovery_dir / "train_receipt.json"
        record = {"format_version": 1, "suite": str(suite), "job_id": job_id,
                  "status": "running", "source_run_dir": str(source_run),
                  "source_checkpoint": str(checkpoint), "config_file": str(config_file),
                  "env": job["env"], "device": job["device"], "receipt_file": str(receipt)}
        _write_json(recovery_dir / "recovery_manifest.json", record)
        print(f"Recovery directory: {recovery_dir}", flush=True)
        command = [conda, "run", "--no-capture-output", "-n", job["env"], "python", "-m",
                   "foundation_sft.train", "--config", str(config_file), "--action", "train",
                   "--resume", str(checkpoint), "--resume-full-precision", "--result-json", str(receipt)]
        try:
            _command(command, recovery_dir / "train.log", project)
            run = _validate_recovery(receipt, job, config, checkpoint, source_run)
        except BaseException as error:
            record.update(status="failed", error=str(error))
            _write_json(recovery_dir / "recovery_manifest.json", record)
            raise
        record.update(status="completed", recovered_run_dir=str(run))
        _write_json(recovery_dir / "recovery_manifest.json", record)
        print(f"Recovery completed: {run}", flush=True)
        print(f"After the original suite finishes, use --report-only --recovery-dir {recovery_dir}", flush=True)
        return recovery_dir
    finally:
        lock.unlink(missing_ok=True)


def report_recovery(suite, job_id, recovery_dir, conda):
    manifest, job, project = _suite_job(suite, job_id)
    if manifest.get("status") not in {"completed", "partial", "failed"}:
        raise ValueError("Wait until the original suite reaches a terminal state before reporting")
    if any(original.get("train_status") == "running" for original in manifest["jobs"]):
        raise ValueError("Original suite still has running training jobs; cannot report yet")
    if (suite / f".recovery_{job_id}.lock").exists():
        raise ValueError("Recovery is still running; cannot report yet")
    record_file = recovery_dir / "recovery_manifest.json"
    record = _read_json(record_file)
    if (record.get("suite") != str(suite) or record.get("job_id") != job_id
            or record.get("status") != "completed"):
        raise ValueError("Explicit recovery directory does not contain this job's successful recovery")
    config_file, config, source_run, checkpoint = _source(suite, job)
    if (record.get("source_checkpoint") != str(checkpoint)
            or record.get("source_run_dir") != str(source_run)
            or record.get("config_file") != str(config_file)):
        raise ValueError("Recovery mapping differs from the original failed experiment")
    run = _validate_recovery(recovery_dir / "train_receipt.json", job, config, checkpoint, source_run)
    if record.get("recovered_run_dir") != str(run):
        raise ValueError("Recovery mapping disagrees with its training receipt")
    reference = _fingerprint(source_run)
    runs, mapping = [], []
    for original in manifest["jobs"]:
        if original["id"] == job_id:
            selected = run
        else:
            if original.get("status") != "completed" or original.get("train_status") == "running":
                raise ValueError("All other original suite jobs must have completed successfully")
            selected = _path(original.get("run_dir"), suite)
        if _validate_run(selected, original) != reference:
            raise ValueError("Cannot report different full cohort/source/split fingerprints")
        runs.append(str(selected))
        mapping.append({"job_id": original["id"], "original_status": original.get("status"),
                        "run_dir": str(selected), "recovered": original["id"] == job_id})
    if len(set(runs)) != len(manifest["jobs"]):
        raise ValueError("Each planned experiment must have its own unique completed run")
    output = recovery_dir / "report"
    if output.exists():
        raise ValueError(f"Report directory already exists; preserve it: {output}")
    # Persist the explicit selection before reporting; no test-score selection.
    record.update(report_runs=mapping, report_status="running")
    _write_json(record_file, record)
    try:
        _command([conda, "run", "--no-capture-output", "-n", job["env"], "python", "-m",
                  "foundation_sft.report", "--runs", *runs, "--output", str(output)],
                 recovery_dir / "report.log", project)
        if not (output / "run_metrics.csv").is_file():
            raise ValueError("Report command did not create run_metrics.csv")
    except BaseException as error:
        record.update(report_status="failed", report_error=str(error))
        _write_json(record_file, record)
        raise
    record.update(report_status="completed", report_dir=str(output))
    _write_json(record_file, record)
    print(f"Complete experiment report: {output}", flush=True)
    return output


def show_metrics(suite, recovery_dir=None):
    """Read completed metrics and current job statuses without launching anything."""
    manifest = _read_json(suite / "suite.json")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("suite.json must contain a jobs list")
    selected = [(job, job.get("status", "unknown"), job.get("run_dir")) for job in jobs]
    if recovery_dir is not None:
        recovery = _read_json(recovery_dir / "recovery_manifest.json")
        matches = [job for job in jobs if job.get("id") == recovery.get("job_id")]
        if recovery.get("suite") != str(suite) or len(matches) != 1:
            raise ValueError("Explicit recovery directory belongs to another suite or job")
        if recovery.get("status") == "completed":
            selected.append((matches[0], "recovered", recovery.get("recovered_run_dir")))
        else:
            print(f"Explicit recovery status: {recovery.get('status', 'unknown')}; no completed metrics")

    def number(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "-"

    headers = ("Model", "Mode", "Seed", "Status", "Val-AUC", "ROC-AUC", "ROC95%CI",
               "PR-AUC", "AP", "Sensitivity", "Specificity")
    rows = []
    for job, status, run_dir in selected:
        patient, interval, selection = {}, {}, {}
        if status in {"completed", "recovered"}:
            run = _path(run_dir, suite)
            metrics = _read_json(run / "test" / "metrics.json")
            patient = metrics.get("patient", {})
            interval = metrics.get("bootstrap", {}).get("patient", {}).get("roc_auc", {})
            selection = _read_json(run / "selection.json")
        elif job.get("train_status") == "running":
            status = "running"
        low, high = interval.get("lower"), interval.get("upper")
        ci = f"[{number(low)},{number(high)}]" if low is not None and high is not None else "-"
        rows.append((str(job.get("model", "?")), str(job.get("mode", "?")),
                     str(job.get("seed", "?")), str(status),
                     number(selection.get("best_val_patient_roc_auc")), number(patient.get("roc_auc")), ci,
                     number(patient.get("pr_auc")), number(patient.get("average_precision")),
                     number(patient.get("sensitivity")), number(patient.get("specificity"))))
    widths = [max([len(header)] + [len(row[index]) for row in rows]) for index, header in enumerate(headers)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    print("PR-AUC: trapezoidal precision-recall area; AP: average precision; ROC95%CI: patient bootstrap.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Original suite directory")
    parser.add_argument("--job", help="Exact failed suite job ID; unnecessary with --show-metrics")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--resume-full-precision", action="store_true", help="Resume last.pt in FP32")
    actions.add_argument("--report-only", action="store_true", help="Report all original jobs with this replacement")
    actions.add_argument("--show-metrics", action="store_true", help="Read current statuses and completed patient metrics")
    parser.add_argument("--recovery-dir", help="Explicit recovery directory; required for --report-only, optional with --show-metrics")
    parser.add_argument("--conda-executable", default="conda", help="Conda executable (default: conda)")
    args = parser.parse_args(argv)
    if args.report_only and not args.recovery_dir:
        parser.error("--report-only requires --recovery-dir")
    if args.resume_full_precision and args.recovery_dir:
        parser.error("--recovery-dir cannot be used with --resume-full-precision")
    if not args.show_metrics and not args.job:
        parser.error("--job is required for recovery and reporting")
    suite = Path(args.suite).expanduser().resolve()
    if args.show_metrics:
        show_metrics(suite, Path(args.recovery_dir).expanduser().resolve() if args.recovery_dir else None)
        return 0
    conda = shutil.which(args.conda_executable)
    if conda is None:
        raise ValueError("Conda executable is unavailable; supply --conda-executable /path/to/conda")
    if args.report_only:
        report_recovery(suite, args.job, Path(args.recovery_dir).expanduser().resolve(), conda)
    else:
        recover(suite, args.job, conda)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Recovery failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
