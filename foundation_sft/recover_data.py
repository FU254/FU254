"""Recover audit failures from verified original DICOMs, keeping backups.

The default is a read-only plan. --apply restores eligible files atomically,
after checking image identity, pixel layout and successful full decoding. It
does not edit the frozen patient manifest or drop any samples.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import pydicom

from .audit_data import _native_pixel_shortfall, _pixel_metadata, _quiet_decode, _transfer_syntax
from .data import _dicom_pixels


def _read(path, pixels=False, include_pixels=False):
    with _quiet_decode():
        dataset = pydicom.dcmread(path, stop_before_pixels=not (pixels or include_pixels))
        if pixels:
            syntax, _ = _transfer_syntax(dataset)
            if _native_pixel_shortfall(syntax, _pixel_metadata(dataset)):
                raise ValueError("Truncated native PixelData")
            _dicom_pixels(dataset)
    return dataset


def _sop(dataset):
    instance = str(getattr(dataset, "SOPInstanceUID", ""))
    storage = str(getattr(getattr(dataset, "file_meta", None), "MediaStorageSOPInstanceUID", ""))
    if instance and storage and instance != storage:
        return ""
    return instance or storage


def _same_image(original, damaged):
    if not _sop(damaged) or _sop(original) != _sop(damaged):
        return False
    fields = ("PatientID", "AccessionNumber", "Rows", "Columns", "SamplesPerPixel",
              "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
              "PhotometricInterpretation", "RescaleSlope", "RescaleIntercept",
              "WindowCenter", "WindowWidth")
    if any(str(getattr(original, key, "")) != str(getattr(damaged, key, "")) for key in fields):
        return False
    return int(getattr(original, "NumberOfFrames", 1)) == int(getattr(damaged, "NumberOfFrames", 1))


def _matches_surviving_pixels(original, damaged, category):
    if category != "truncated_pixel_data":
        return True
    surviving = getattr(damaged, "PixelData", None)
    complete = getattr(original, "PixelData", None)
    return (isinstance(surviving, bytes) and isinstance(complete, bytes)
            and len(surviving) < len(complete) and complete.startswith(surviving))


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore(target, source, damaged, category, source_digest, target_digest, backup):
    """Keep the old bytes and replace only with a verified, stable original."""
    if _digest(target) != target_digest:
        raise ValueError("Current file changed during recovery")
    original_mode = target.stat().st_mode & 0o777
    backup.parent.mkdir(mode=0o700, exist_ok=True)
    shutil.copy2(target, backup)
    os.chmod(backup, 0o600)
    if _digest(backup) != target_digest:
        raise ValueError("Current file changed while making backup")
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".dicom_recovery_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary)
        if _digest(temporary) != source_digest:
            raise ValueError("Original changed during recovery")
        copied = _read(temporary, pixels=True)
        if not (_same_image(copied, damaged) and _matches_surviving_pixels(copied, damaged, category)):
            raise ValueError("Copied original no longer matches image identity")
        if _digest(target) != target_digest:
            raise ValueError("Current file changed before replacement")
        os.chmod(temporary, original_mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def recover_from_backup(audit_dir, source_dir, *, apply=False):
    """Return a safe summary; exact paths are written only to a private CSV."""
    audit_dir, source_dir = Path(audit_dir).resolve(), Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError("The original-data source must be an existing directory")
    with (audit_dir / "failures_private.csv").open(encoding="utf-8", newline="") as stream:
        failures = list(csv.DictReader(stream))
    with (audit_dir / "report.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    keys = ("split", "row", "error_category")
    if len(failures) != report["failed_images"]:
        raise ValueError("Audit technical report and private failure list disagree")
    if [tuple(str(item[key]) for key in keys) for item in failures] != [
            tuple(str(item[key]) for key in keys) for item in report["failures"]]:
        raise ValueError("Audit technical report and private failure list disagree")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = audit_dir / f"recovery_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(mode=0o700)
    files = sorted(path for path in source_dir.rglob("*") if path.is_file() and not path.is_symlink())
    uid_index = None
    results, private = [], []
    for failure in failures:
        item = {"split": failure["split"], "row": int(failure["row"])}
        details = {**item, "path": failure["path"], "source": "", "backup": ""}
        target = Path(failure["path"]).resolve()
        try:
            _read(target, pixels=True)
            item.update(status="already_valid", reason="Current file already decodes successfully")
        except Exception:
            if failure["error_category"] in {"missing_pixel_decoder", "pixel_preprocessing_failed"}:
                item.update(status="not_eligible", reason="Fix the decoder or preprocessing issue before recovery")
            else:
                try:
                    target_digest = _digest(target)
                    damaged = _read(target, include_pixels=True)
                    if _digest(target) != target_digest:
                        raise ValueError("Current file changed while reading its metadata")
                    if uid_index is None:
                        uid_index = {}
                        print(f"[recovery] Checking image identity in {len(files)} original files.", flush=True)
                        for path in files:
                            try:
                                uid = _sop(_read(path))
                                if uid:
                                    uid_index.setdefault(uid, []).append(path)
                            except Exception:
                                continue
                    candidates = uid_index.get(_sop(damaged), [])
                    valid = {}
                    for candidate in candidates:
                        if candidate.resolve() == target:
                            continue
                        try:
                            original = _read(candidate, pixels=True)
                            if (_same_image(original, damaged)
                                    and _matches_surviving_pixels(original, damaged, failure["error_category"])):
                                valid.setdefault(_digest(candidate), candidate)
                        except Exception:
                            continue
                    if len(valid) != 1:
                        item.update(status="unresolved", reason="No unique complete original with matching identity and pixel metadata")
                    else:
                        digest, source = next(iter(valid.items()))
                        details["source"] = str(source)
                        item.update(status="ready", reason="Verified complete original available")
                        if apply:
                            backup = output / "backup" / f"{len(results):06d}.dcm"
                            details["backup"] = str(backup)
                            _restore(target, source, damaged, failure["error_category"], digest, target_digest, backup)
                            item.update(status="restored", reason="Original restored; previous file retained in backup")
                except Exception:
                    item.update(status="unresolved", reason="Recovery could not verify or restore this file")
        results.append(item)
        private.append({**details, **item})
    summary = {"apply": apply, "output": str(output), "files": results,
               "counts": {name: sum(item["status"] == name for item in results)
                          for name in ("ready", "restored", "already_valid", "unresolved", "not_eligible")}}
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    descriptor = os.open(output / "recovery_private.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "row", "path", "source", "backup", "status", "reason"))
        writer.writeheader()
        writer.writerows(private)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, help="Directory printed by audit-data")
    parser.add_argument("--source-dir", required=True, help="Complete original DICOM directory")
    parser.add_argument("--apply", action="store_true", help="Restore verified files and keep previous copies")
    args = parser.parse_args(argv)
    result = recover_from_backup(args.audit_dir, args.source_dir, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result["counts"]["unresolved"] or result["counts"]["not_eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
