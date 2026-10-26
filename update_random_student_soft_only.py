#!/usr/bin/env python3
"""Add soft-label-only screening to the existing random-student KD workflow."""

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
    "foundation_sft/random_student_distillation.py",
    "foundation_sft/random_student_suite.py",
    "foundation_sft/random_student_profiles.py",
    "foundation_sft/random_student_soft_only.py",
    "foundation_sft/random_student_soft_report.py",
    "run_random_student_soft_only.py",
    "foundation_sft/configs/random_student_soft_only.json",
    "foundation_sft/RANDOM_STUDENT_SOFT_ONLY.md",
)
# Replace only these reviewed source versions; preserve unknown local edits.
PREVIOUS_SHA256 = {
    "foundation_sft/random_student_distillation.py": "6cc060695bd0896631660512dd8d839944e714c6d83f754cb621b1f0c87618ab",
    "foundation_sft/random_student_suite.py": "57b28038c961ad7f63903999b20937afb2e6388e3b92f959db668fce300b4c22"
}
# The package builder freezes the reviewed payload before distribution.
PAYLOAD_SHA256 = {
    "foundation_sft/random_student_distillation.py": "05ba615fc6f8325b5b634c41038b73e21c25f67cd50b8b2526e96336e050c2bd",
    "foundation_sft/random_student_suite.py": "be704066b347addb40ba7eed0534291f332bfd678ac716b38c21e1515f0516ca",
    "foundation_sft/random_student_profiles.py": "1afd9d7003aa5f85df8f16ea8d62e7a9caa03563bb461c3ab35d19e10d2ce531",
    "foundation_sft/random_student_soft_only.py": "ee7a272c629967df55a4e412e8f92aab02053403b54079bd2ad6c734bcac302b",
    "foundation_sft/random_student_soft_report.py": "42586e04926ff596b7239aea441756cac12bea1c3f2ff030ce47576563d34ba1",
    "run_random_student_soft_only.py": "f54aebe0e1aaca9a72596519fe9c8bed64fb28b5f9812ef04fafa746bdcf199e",
    "foundation_sft/configs/random_student_soft_only.json": "87b6938a728f5457b801dfda30e64c5c5801c42f147220e33420cb72295a1c49",
    "foundation_sft/RANDOM_STUDENT_SOFT_ONLY.md": "2e65de4eacee0979dfdec705c5626c0d3b98694689207334e973be2cee66b2c1"
}

PREREQUISITES = (
    "foundation_sft/train.py", "foundation_sft/distillation.py",
    "foundation_sft/distillation_data.py", "foundation_sft/distillation_losses.py",
    "foundation_sft/distillation_suite.py", "foundation_sft/training_history.py",
    "foundation_sft/models/vit_tiny_student.py",
    "foundation_sft/random_student_distillation.py",
    "foundation_sft/random_student_suite.py",
    "foundation_sft/weight_decay_baseline.py",
    "foundation_sft/weight_decay_study.py",
    "foundation_sft/weight_decay_report.py",
    "run_random_student_distillation.py",
    "run_random_student_experiments.py",
    "foundation_sft/droppath_baseline.py",
    "foundation_sft/droppath_study.py",
    "foundation_sft/droppath_report.py",
    "run_droppath_experiments.py",
)

BACKUP_DIRECTORY = ".random-student-soft-only-backups"
LOCK_NAME = ".random-student-soft-only-update.lock"
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
        raise ValueError("Install into the existing random-student KD project; missing: " + ", ".join(missing))
    if set(PAYLOAD_SHA256) != set(PAYLOAD_FILES):
        raise ValueError("The Random-student soft-only update checksum manifest has not been frozen")
    payload = {}
    # Validate every uploaded source before considering any destination change.
    for relative in PAYLOAD_FILES:
        origin = _path(source, relative)
        if not origin.is_file() or origin.stat().st_size > MAX_BYTES:
            raise ValueError("Missing or oversized Random-student soft-only update payload: " + relative)
        data = origin.read_bytes()
        if _digest(data) != PAYLOAD_SHA256[relative]:
            raise ValueError("Random-student soft-only update source checksum mismatch: " + relative)
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
    descriptor, name = tempfile.mkstemp(prefix=".random-student-soft-only-update-", suffix=".tmp", dir=target.parent)
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
    journal = {"format_version": 1, "kind": "random_student_soft_only_update",
               "project": str(project), "files": records}
    (backup / "installation.json").write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
    return backup


def install(project, source_dir, apply=False):
    project, source = Path(project).expanduser().resolve(), Path(source_dir).expanduser().resolve()
    changes = _prepare(project, source)
    if not apply or not changes:
        return {"apply": apply, "files": list(changes), "backup": None}
    if project == source:
        raise ValueError("Unpack the update into its own random_student_soft_only_update directory")
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
    if (journal.get("format_version") != 1 or journal.get("kind") != "random_student_soft_only_update"
            or journal.get("project") != str(project) or not isinstance(files, dict)
            or not files or not set(files).issubset(PAYLOAD_FILES)):
        raise ValueError("Invalid Random-student soft-only update installation journal")
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
        print("Random-student soft-only update installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
