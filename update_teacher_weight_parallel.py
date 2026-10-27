#!/usr/bin/env python3
"""Install safe independent teacher-weight jobs while an existing job continues."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid


PAYLOAD_FILES = (
    "foundation_sft/teacher_weight_parallel.py",
    "run_teacher_weight_parallel.py",
    "foundation_sft/TEACHER_WEIGHT_PARALLEL.md",
    "run_teacher_weight_training.py",
)
# The thin worker entry is the only existing file that this update changes.
# It is published last, after the new module and independent entry are present.
PREVIOUS_SHA256 = {
    "run_teacher_weight_training.py": "e40b97b736fb2641ae744e4ceea8f22a3087e1fc098fc4876da177f834f09ace",
}
# The package builder freezes the reviewed payload before distribution.
PAYLOAD_SHA256 = {
    "foundation_sft/teacher_weight_parallel.py": "af79ae03d499d70ffccc51fd49ab3a502bab7cca7d093f794882eee52488af3f",
    "run_teacher_weight_parallel.py": "f99685830ee6fe1bafef5227e4e4de22eb76ad6f58ac6ce926dfa7217d7190ad",
    "foundation_sft/TEACHER_WEIGHT_PARALLEL.md": "c50d2bca8abd4ab7275954f995cf91cd6a6049f7b3280110fced281ee608d244",
    "run_teacher_weight_training.py": "400c5cd212248c348b923ec9661fc4e83481c3fae1e50a0751f79dfd4c35372f"
}

PREREQUISITES = (
    "foundation_sft/teacher_weight_training.py",
    "foundation_sft/teacher_weight_study.py",
    "foundation_sft/teacher_weight_report.py",
    "foundation_sft/teacher_weight_losses.py",
    "foundation_sft/configs/teacher_weight_study.json",
    "run_teacher_weight_experiments.py",
)

BACKUP_DIRECTORY = ".teacher-weight-parallel-backups"
LOCK_NAME = ".teacher-weight-parallel-update.lock"
MAX_BYTES = 1024 * 1024


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _require_python():
    if sys.version_info < (3, 10):
        raise ValueError("Use Python 3.10+, for example conda activate evax_lora")


def _path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe relative path")
    path = root / relative
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise ValueError("Refusing a symbolic-link path: " + str(relative))
    return path


def _current(path):
    if not path.exists():
        return None, None
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Expected a small regular code file: " + str(path))
    return path.read_bytes(), path.stat().st_mode & 0o777


def _prepare(project, source):
    _require_python()
    missing = [name for name in PREREQUISITES if not _path(project, name).is_file()]
    if missing:
        raise ValueError("Install into the existing teacher-weight study project; missing: " + ", ".join(missing))
    if set(PAYLOAD_SHA256) != set(PAYLOAD_FILES):
        raise ValueError("The Teacher-weight parallel update checksum manifest has not been frozen")
    payload = {}
    # Validate every uploaded source before considering any destination change.
    for relative in PAYLOAD_FILES:
        origin = _path(source, relative)
        if not origin.is_file() or origin.stat().st_size > MAX_BYTES:
            raise ValueError("Missing or oversized Teacher-weight parallel update payload: " + relative)
        data = origin.read_bytes()
        if _digest(data) != PAYLOAD_SHA256[relative]:
            raise ValueError("Teacher-weight parallel update source checksum mismatch: " + relative)
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json"):
            if not isinstance(json.loads(data), dict):
                raise ValueError("Configuration payload must be an object: " + relative)
        else:
            data.decode("utf-8")
        payload[relative] = data
    changes = {}
    for relative, data in payload.items():
        previous, mode = _current(_path(project, relative))
        if previous == data:
            continue
        expected_old = PREVIOUS_SHA256.get(relative)
        if expected_old is not None:
            if previous is None or _digest(previous) != expected_old:
                raise ValueError("Unknown local changes preserved; no files were updated: " + relative)
        elif previous is not None:
            raise ValueError("Existing file or local changes preserved; no files were updated: " + relative)
        changes[relative] = {"before": previous, "before_mode": mode,
                             "after": data, "after_mode": mode if mode is not None else 0o644}
    return changes


def _change(project, relative, before, after, before_mode, after_mode):
    """Check the expected bytes/mode immediately before an atomic publication."""
    target = _path(project, relative)
    if _current(target) != (before, before_mode):
        raise ValueError("File changed during update; local changes preserved: " + relative)
    if after is None:
        target.unlink()
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".teacher-weight-parallel-update-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, after_mode)
        target = _path(project, relative)
        if _current(target) != (before, before_mode):
            raise ValueError("File changed during update; local changes preserved: " + relative)
        if before is None:
            # Do not overwrite a new user file created after the check above.
            os.link(name, target)
        else:
            os.replace(name, target)
    finally:
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass


def _transaction(project, changes):
    published = []
    try:
        for relative, entry in changes.items():
            _change(project, relative, **entry)
            published.append(relative)
    except BaseException as original:
        conflicts = []
        for relative in reversed(published):
            entry = changes[relative]
            try:
                if _current(_path(project, relative)) != (entry["after"], entry["after_mode"]):
                    conflicts.append(relative)
                    continue
                _change(project, relative, entry["after"], entry["before"],
                        entry["after_mode"], entry["before_mode"])
            except (OSError, ValueError):
                conflicts.append(relative)
        if conflicts:
            raise RuntimeError("Update failed; later local changes preserved or restoration blocked: "
                               + ", ".join(conflicts) + ". Original error: " + str(original)) from original
        raise


def _acquire_lock(project):
    lock = _path(project, LOCK_NAME)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    return lock


def _backup(project, changes):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = _path(project, BACKUP_DIRECTORY) / (stamp + "_" + uuid.uuid4().hex[:8])
    backup.mkdir(parents=True, mode=0o700)
    records = {}
    for relative, entry in changes.items():
        for kind, data in (("previous", entry["before"]), ("updated", entry["after"])):
            if data is not None:
                saved = backup / kind / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(data)
        records[relative] = {
            "previous_sha256": _digest(entry["before"]) if entry["before"] is not None else None,
            "updated_sha256": _digest(entry["after"]),
            "previous_mode": entry["before_mode"], "updated_mode": entry["after_mode"],
        }
    journal = {"format_version": 1, "kind": "teacher_weight_parallel_update",
               "project": str(project), "files": records}
    (backup / "installation.json").write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    return backup


def install(project, source_dir, apply=False):
    project, source = Path(project).expanduser().resolve(), Path(source_dir).expanduser().resolve()
    changes = _prepare(project, source)
    if not apply or not changes:
        return {"apply": apply, "files": list(changes), "backup": None}
    if project == source:
        raise ValueError("Unpack the update into its own teacher_weight_parallel_update directory")
    lock = _acquire_lock(project)
    try:
        changes = _prepare(project, source)
        if not changes:
            return {"apply": True, "files": [], "backup": None}
        backup = _backup(project, changes)
        _transaction(project, changes)
        return {"apply": True, "files": list(changes), "backup": str(backup)}
    finally:
        lock.unlink()


def _rollback_plan(project, backup_dir):
    backup = Path(backup_dir).expanduser().absolute()
    try:
        relative = backup.relative_to(project)
    except ValueError as error:
        raise ValueError("Backup must belong to this project") from error
    if len(relative.parts) != 2 or relative.parts[0] != BACKUP_DIRECTORY:
        raise ValueError("Choose an installation directory under " + BACKUP_DIRECTORY)
    backup = _path(project, relative)
    journal = json.loads((backup / "installation.json").read_text(encoding="utf-8"))
    files = journal.get("files")
    if (journal.get("format_version") != 1 or journal.get("kind") != "teacher_weight_parallel_update"
            or journal.get("project") != str(project) or not isinstance(files, dict)
            or not files or not set(files).issubset(PAYLOAD_FILES)):
        raise ValueError("Invalid Teacher-weight parallel update installation journal")
    changes = {}
    for relative, record in files.items():
        previous_sha = record.get("previous_sha256")
        if previous_sha != PREVIOUS_SHA256.get(relative):
            raise ValueError("Backup is not a recognized previous code version: " + relative)
        before = _path(backup, "updated/" + relative).read_bytes()
        if _digest(before) != record.get("updated_sha256"):
            raise ValueError("Updated backup checksum mismatch: " + relative)
        after = _path(backup, "previous/" + relative).read_bytes() if previous_sha else None
        if after is not None and _digest(after) != previous_sha:
            raise ValueError("Previous backup checksum mismatch: " + relative)
        before_mode, after_mode = record.get("updated_mode"), record.get("previous_mode")
        if (type(before_mode) is not int or not 0 <= before_mode <= 0o777
                or (after is not None and (type(after_mode) is not int or not 0 <= after_mode <= 0o777))
                or (after is None and after_mode is not None)):
            raise ValueError("Invalid backup permission mode: " + relative)
        current = _current(_path(project, relative))
        if current == (after, after_mode):
            continue  # Already restored, including absent newly added files.
        if current != (before, before_mode):
            raise ValueError("Local changes preserved; rollback refused: " + relative)
        changes[relative] = {"before": before, "after": after,
                             "before_mode": before_mode, "after_mode": after_mode}
    return backup, changes


def rollback(project, backup_dir, apply=False):
    _require_python()
    project = Path(project).expanduser().resolve()
    backup, changes = _rollback_plan(project, backup_dir)
    if not apply or not changes:
        return {"apply": apply, "files": list(changes), "rollback": str(backup)}
    lock = _acquire_lock(project)
    try:
        backup, changes = _rollback_plan(project, backup_dir)
        _transaction(project, changes)
        return {"apply": True, "files": list(changes), "rollback": str(backup)}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", help="Installation backup to preview/restore")
    args = parser.parse_args(argv)
    result = (rollback(args.project, args.rollback, args.apply) if args.rollback else
              install(args.project, args.source_dir, args.apply))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        print("Teacher-weight parallel update installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
