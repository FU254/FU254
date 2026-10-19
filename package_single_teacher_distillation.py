#!/usr/bin/env python3
"""Freeze reviewed KD checksums and build the independent offline code ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import update_single_teacher_distillation as installer


def freeze_manifest(root=None):
    """Call only after all seven payload files and documentation are reviewed."""
    root = Path(root or Path(__file__).resolve().parent)
    checksums = {}
    for relative in installer.PAYLOAD_FILES:
        data = (root / relative).read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError("Oversized KD payload: " + relative)
        checksums[relative] = hashlib.sha256(data).hexdigest()
    path = root / "update_single_teacher_distillation.py"
    text = path.read_text(encoding="utf-8")
    block = "PAYLOAD_SHA256 = " + json.dumps(checksums, indent=4)
    updated, count = re.subn(r"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", lambda match: block, text, count=1)
    if count != 1:
        raise ValueError("Could not locate the KD checksum manifest")
    path.write_text(updated, encoding="utf-8")
    installer.PAYLOAD_SHA256 = checksums
    return checksums


def package(output, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    if set(installer.PAYLOAD_SHA256) != set(installer.PAYLOAD_FILES):
        raise ValueError("Freeze the reviewed KD manifest before packaging")
    payload = {}
    for relative in installer.PAYLOAD_FILES:
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != installer.PAYLOAD_SHA256[relative]:
            raise ValueError("Code differs from the reviewed KD manifest: " + relative)
        payload[relative] = data
    payload["update_single_teacher_distillation.py"] = (root / "update_single_teacher_distillation.py").read_bytes()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payload.items():
            archive.writestr("kd_update/" + relative, data)
        archive.writestr("kd_update/SHA256SUMS.txt", "\n".join(
            hashlib.sha256(data).hexdigest() + "  " + relative
            for relative, data in payload.items()) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("KD ZIP integrity check failed")
    print(output)
    print("Bytes:", output.stat().st_size, "SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", action="store_true", help="Explicitly freeze hashes after code review")
    parser.add_argument("--output", help="Create a new ZIP at this path")
    args = parser.parse_args(argv)
    if not args.freeze_manifest and not args.output:
        parser.error("Specify --freeze-manifest and/or --output")
    if args.freeze_manifest:
        freeze_manifest()
    if args.output:
        package(args.output)


if __name__ == "__main__":
    main()
