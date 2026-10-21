#!/usr/bin/env python3
"""Repair the published GPU save/reload smoke check without changing training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid


CORE = "foundation_sft/distillation.py"
SOURCE_SHA256 = "3318844711c090d076f9f42070625632449387fa2abb7e912b380848f24c9787"
# Frozen from the reviewed core diff after the smoke-only fix is finalized.
TARGET_SHA256 = 'dceb270c7af766d2316968327c46c1c67d58223f8b023ceee1e853f5250879c8'
PATCHES = ((b'        digest.update(tensor.numpy().tobytes())\n    return digest.hexdigest()\n\n\n',
  b'        digest.update(tensor.numpy().tobytes())\n    return digest.hexdigest()\n\n\ndef _ver'
  b'ify_roundtrip_state(student, restored):\n    """Check exact serialization before the restored'
  b' model runs inference."""\n    original_hash = _state_hash(student.state_dict())\n    restored'
  b'_hash = _state_hash(restored.state_dict())\n    if original_hash != restored_hash:\n        ra'
  b'ise ValueError("Complete student state changed during save/reload")\n    return {"student_sta'
  b'te_exact": True, "student_state_sha256": original_hash,\n            "restored_state_sha256":'
  b' restored_hash}\n\n\ndef _verify_roundtrip_outputs(expected, actual, batch_size):\n    """Al'
  b'low bounded FP32 inference roundoff after an exact state comparison."""\n    if type(batch_si'
  b'ze) is not int or batch_size < 1:\n        raise ValueError("Roundtrip batch size must be a p'
  b'ositive integer")\n    for name, value in (("expected", expected), ("actual", actual)):\n     '
  b'   if (not isinstance(value, torch.Tensor) or not value.is_floating_point()\n                '
  b'or tuple(value.shape) != (batch_size,)):\n            raise ValueError(f"Roundtrip {name} log'
  b'its must have shape [batch_size]")\n        if not torch.isfinite(value).all().item():\n      '
  b'      raise ValueError(f"Roundtrip {name} logits contain non-finite values")\n    # The weigh'
  b'ts are checked separately and exactly. CUDA FP32 inference may\n    # vary slightly near zero'
  b' even with identical restored tensors.\n    rtol, atol = 1e-4, 1e-5\n    torch.testing.assert_'
  b'close(actual, expected, rtol=rtol, atol=atol)\n    return {"roundtrip_max_abs_error": (actual'
  b' - expected).abs().max().item(),\n            "roundtrip_rtol": rtol, "roundtrip_atol": atol}'
  b'\n\n\n'),
 (b'                            condition, initial_hash, info, loader.dataset.metadata()))\n    r'
  b'estored, _ = load_student_checkpoint(snapshot, device)\n    with torch.no_grad():\n        act'
  b'ual = restored(batch["image"].to(device)).detach().cpu()\n    torch.testing.assert_close(actu'
  b'al, expected, rtol=1e-5, atol=1e-6)\n    snapshot.unlink()\n    result = {"check_passed": True'
  b', "training_started": False,\n              "teacher_unchanged": True, "student_full_checkpoi'
  b'nt_roundtrip": True,\n              "teacher_features": teacher.adapter.feature_dim, "student'
  b'_features": student.feature_dim,\n',
  b'                            condition, initial_hash, info, loader.dataset.metadata()))\n    r'
  b'estored, _ = load_student_checkpoint(snapshot, device)\n    roundtrip_state = _verify_roundtr'
  b'ip_state(student, restored)\n    with torch.no_grad():\n        actual = restored(batch["image'
  b'"].to(device)).detach().cpu()\n    roundtrip_outputs = _verify_roundtrip_outputs(expected, ac'
  b'tual, len(batch["label"]))\n    snapshot.unlink()\n    result = {"check_passed": True, "traini'
  b'ng_started": False,\n              **roundtrip_state, **roundtrip_outputs,\n              "tea'
  b'cher_unchanged": True, "student_full_checkpoint_roundtrip": True,\n              "teacher_fea'
  b'tures": teacher.adapter.feature_dim, "student_features": student.feature_dim,\n'))

BACKUP_DIRECTORY = ".distillation-roundtrip-backups"
LOCK_NAME = ".distillation-roundtrip-repair.lock"


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


def _read(path):
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Expected the existing small distillation core file")
    return path.read_bytes(), path.stat().st_mode & 0o777


def _build(source):
    if not PATCHES or len(TARGET_SHA256) != 64:
        raise ValueError("The reviewed repair constants have not been frozen")
    if _digest(source) != SOURCE_SHA256:
        raise ValueError("Unknown local changes preserved; core SHA256 is not the reviewed official version")
    updated = source
    for previous, replacement in PATCHES:
        if not previous or not replacement or updated.count(previous) != 1:
            raise ValueError("Each reviewed repair fragment must occur exactly once")
        updated = updated.replace(previous, replacement, 1)
    if _digest(updated) != TARGET_SHA256:
        raise ValueError("Repaired core checksum does not match the reviewed target")
    compile(updated, CORE, "exec")
    return updated


def _replace(project, source, updated, mode):
    target = _path(project, CORE)
    descriptor, name = tempfile.mkstemp(prefix=".roundtrip-repair-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, mode)
        target = _path(project, CORE)
        if _read(target) != (source, mode):
            raise ValueError("Core changed during repair; later local changes preserved")
        os.replace(name, target)
    finally:
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass


def repair(project, apply=False):
    if sys.version_info < (3, 10):
        raise ValueError("Use Python 3.10+, for example conda activate evax_lora")
    project = Path(project).expanduser().resolve()
    target = _path(project, CORE)
    source, mode = _read(target)
    digest = _digest(source)
    result = {"apply": apply, "file": str(target), "source_sha256": digest,
              "target_sha256": TARGET_SHA256, "backup": None}
    if TARGET_SHA256 and digest == TARGET_SHA256:
        return {**result, "status": "already_repaired"}
    updated = _build(source)
    if not apply:
        return {**result, "status": "would_repair"}
    lock = _path(project, LOCK_NAME)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        if _read(_path(project, CORE)) != (source, mode):
            raise ValueError("Core changed before repair; local changes preserved")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = _path(project, BACKUP_DIRECTORY) / (stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(parents=True, mode=0o700)
        saved = backup / "distillation.py"
        saved.write_bytes(source)
        saved.chmod(mode)
        journal = {"format_version": 1, "kind": "distillation_roundtrip_smoke_repair",
                   "file": CORE, "project": str(project), "previous_mode": mode,
                   "source_sha256": SOURCE_SHA256, "target_sha256": TARGET_SHA256}
        (backup / "repair.json").write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
        _replace(project, source, updated, mode)
        if _read(_path(project, CORE)) != (updated, mode):
            raise RuntimeError("Core changed after publication; later local changes preserved. Backup: " + str(backup))
        return {**result, "status": "repaired", "backup": str(backup)}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--apply", action="store_true", help="Apply the reviewed repair; default previews only")
    args = parser.parse_args(argv)
    result = repair(args.project, args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, SyntaxError) as error:
        print("Roundtrip repair failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
