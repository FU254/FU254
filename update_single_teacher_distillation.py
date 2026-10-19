#!/usr/bin/env python3
"""Install an independent single-teacher experiment without replacing old code."""

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
    "run_single_teacher_distillation.py",
    "foundation_sft/distillation.py",
    "foundation_sft/distillation_losses.py",
    "foundation_sft/distillation_data.py",
    "foundation_sft/models/vit_tiny_student.py",
    "foundation_sft/configs/distillation_rad_dino_vit_tiny.json",
    "foundation_sft/SINGLE_TEACHER_DISTILLATION.md",
)
# Frozen by package_single_teacher_distillation.freeze_manifest() after review.
PAYLOAD_SHA256 = {
    "run_single_teacher_distillation.py": "4c07485645b13338403d86321f06ef30079ad1799a3004702318da82311e7a61",
    "foundation_sft/distillation.py": "722c4b113fa90cbce6471ef7c046a4a5e37d6af0fb591773a30693d6b5553f0d",
    "foundation_sft/distillation_losses.py": "156dd2a23d0af3dda65289b1defe49456ac0b1e426c2da1e5f63224cacf75a9e",
    "foundation_sft/distillation_data.py": "0f716cf5ae77b1d7845ac60b8ba0cb6fb15d0c39da9ebfd859765488d672bbb3",
    "foundation_sft/models/vit_tiny_student.py": "9cc8ce95374b1206fb8ce1e3dcce30f4682fda32710be7131fae9a3946777527",
    "foundation_sft/configs/distillation_rad_dino_vit_tiny.json": "c8fcb4868a5385bcf70e6a2517e513ed7ad3bb35127d9bd2ae2796b1a38a4a3a",
    "foundation_sft/SINGLE_TEACHER_DISTILLATION.md": "32a718499cee878aebbfa56dd3bcc6e9bf0f734e661265e6e7675547ee430c9d"
}

BACKUP_DIRECTORY = ".single-teacher-kd-backups"
LOCK_NAME = ".single-teacher-kd-update.lock"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


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


def _prepare(project, source):
    if sys.version_info < (3, 10):
        raise ValueError("Use Python 3.10+, for example conda activate evax_lora")
    if not _path(project, "foundation_sft/train.py").is_file():
        raise ValueError("--project must be the existing EVA-X project")
    if set(PAYLOAD_SHA256) != set(PAYLOAD_FILES):
        raise ValueError("The KD payload checksum manifest has not been frozen")
    pending = {}
    for relative in PAYLOAD_FILES:
        origin = _path(source, relative)
        if not origin.is_file() or origin.stat().st_size > 1024 * 1024:
            raise ValueError("Missing or oversized KD payload: " + relative)
        data = origin.read_bytes()
        if _digest(data) != PAYLOAD_SHA256[relative]:
            raise ValueError("KD source checksum mismatch: " + relative)
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json"):
            if not isinstance(json.loads(data), dict):
                raise ValueError("Configuration payload must be an object: " + relative)
        else:
            data.decode("utf-8")
        target = _path(project, relative)
        if target.exists():
            if not target.is_file():
                raise ValueError("Destination is not a regular file: " + relative)
            if target.read_bytes() != data:
                raise ValueError("Existing file or local changes preserved; no files were updated: " + relative)
        else:
            pending[relative] = data
    return pending


def _publish_new(path, data):
    """Publish a complete new file atomically, refusing even a raced-in target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".single-teacher-kd-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o644)
        # os.replace would overwrite a user file created after our preflight.
        os.link(name, path)
    finally:
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass


def _acquire_lock(project):
    lock = _path(project, LOCK_NAME)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    return lock


def install(project, source_dir, apply=False):
    project = Path(project).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    pending = _prepare(project, source)
    if not apply or not pending:
        return {"apply": apply, "files": list(pending), "backup": None}
    if project == source:
        raise ValueError("Unpack the KD add-on separately from the existing project")
    lock = _acquire_lock(project)
    try:
        pending = _prepare(project, source)
        if not pending:
            return {"apply": True, "files": [], "backup": None}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = _path(project, BACKUP_DIRECTORY) / (stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(parents=True, mode=0o700)
        # This add-on only adds files. Record their previous absence; no old
        # project file is copied or replaced. Keep reviewed payload copies too.
        for relative, data in pending.items():
            saved = backup / "payload" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(data)
        journal = {"format_version": 1, "kind": "single_teacher_kd_addition",
                   "project": str(project), "previous": {name: None for name in pending},
                   "added": {name: _digest(data) for name, data in pending.items()}}
        (backup / "installation.json").write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for relative, data in pending.items():
                target = _path(project, relative)
                _publish_new(target, data)
                published.append(relative)
        except BaseException:
            # Do not erase a user's edit even during an interrupted installation.
            for relative in reversed(published):
                target = _path(project, relative)
                if target.is_file() and _digest(target.read_bytes()) == _digest(pending[relative]):
                    target.unlink()
            raise
        return {"apply": True, "files": list(pending), "backup": str(backup)}
    finally:
        lock.unlink()


def rollback(project, backup_dir, apply=False):
    """Remove this installation's new files only if their bytes are unchanged."""
    if sys.version_info < (3, 10):
        raise ValueError("Use Python 3.10+, for example conda activate evax_lora")
    project = Path(project).expanduser().resolve()
    backup = Path(backup_dir).expanduser().absolute()
    try:
        relative_backup = backup.relative_to(project)
    except ValueError as error:
        raise ValueError("Backup must belong to this project") from error
    if len(relative_backup.parts) != 2 or relative_backup.parts[0] != BACKUP_DIRECTORY:
        raise ValueError("Choose an installation directory under " + BACKUP_DIRECTORY)
    backup = _path(project, relative_backup)
    journal = json.loads((backup / "installation.json").read_text(encoding="utf-8"))
    added = journal.get("added")
    if (journal.get("format_version") != 1 or journal.get("kind") != "single_teacher_kd_addition"
            or journal.get("project") != str(project) or not isinstance(added, dict)
            or not added or not set(added).issubset(PAYLOAD_FILES)
            or journal.get("previous") != {name: None for name in added}):
        raise ValueError("Invalid KD installation journal")

    def candidates():
        files = []
        for relative, checksum in added.items():
            saved = _path(backup, "payload/" + relative)
            if not saved.is_file() or _digest(saved.read_bytes()) != checksum:
                raise ValueError("Backup payload checksum mismatch: " + relative)
            target = _path(project, relative)
            if target.exists():
                if not target.is_file() or _digest(target.read_bytes()) != checksum:
                    raise ValueError("Local changes preserved; rollback refused: " + relative)
                files.append(relative)
        return files

    files = candidates()
    if not apply or not files:
        return {"apply": apply, "rollback": str(backup), "files": files}
    lock = _acquire_lock(project)
    try:
        files = candidates()
        removed = []
        try:
            for relative in files:
                target = _path(project, relative)
                if _digest(target.read_bytes()) != added[relative]:
                    raise ValueError("File changed during rollback: " + relative)
                target.unlink()
                removed.append(relative)
        except BaseException:
            # Restore an interrupted rollback from its verified saved payload.
            for relative in removed:
                saved = _path(backup, "payload/" + relative)
                data = saved.read_bytes()
                if _digest(data) != added[relative]:
                    raise ValueError("Backup payload checksum mismatch: " + relative)
                _publish_new(_path(project, relative), data)
            raise
        return {"apply": True, "rollback": str(backup), "files": files}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent), help="Existing project directory")
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise preview only")
    parser.add_argument("--rollback", help="Preview/remove unchanged files from this installation backup")
    args = parser.parse_args(argv)
    result = (rollback(args.project, args.rollback, args.apply) if args.rollback
              else install(args.project, args.source_dir, args.apply))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError) as error:
        print("Single-teacher KD installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
