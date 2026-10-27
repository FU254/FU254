#!/usr/bin/env python3
"""Freeze and package the independent teacher-weight worker incremental update."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import update_teacher_weight_parallel as installer


def freeze_manifest(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    checksums = {}
    for relative in installer.PAYLOAD_FILES:
        origin = installer._path(root, relative)
        if not origin.is_file():
            raise ValueError("Missing teacher-weight-parallel payload: " + relative)
        data = origin.read_bytes()
        if len(data) > installer.MAX_BYTES:
            raise ValueError("Oversized teacher-weight-parallel payload: " + relative)
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json"):
            if not isinstance(json.loads(data), dict):
                raise ValueError("Configuration payload must be an object: " + relative)
        else:
            data.decode("utf-8")
        checksums[relative] = hashlib.sha256(data).hexdigest()
    path = root / "update_teacher_weight_parallel.py"
    source = path.read_text(encoding="utf-8")
    replacement = "PAYLOAD_SHA256 = " + json.dumps(checksums, indent=4)
    updated, count = re.subn(r"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", lambda match: replacement, source, count=1)
    if count != 1:
        raise ValueError("teacher-weight-parallel checksum declaration not found")
    path.write_text(updated, encoding="utf-8")
    installer.PAYLOAD_SHA256 = checksums
    return checksums


def package(output, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    if set(installer.PAYLOAD_SHA256) != set(installer.PAYLOAD_FILES):
        raise ValueError("Freeze the reviewed teacher-weight-parallel manifest before packaging")
    payload = {}
    for relative in installer.PAYLOAD_FILES:
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != installer.PAYLOAD_SHA256[relative]:
            raise ValueError("Code differs from the reviewed teacher-weight-parallel manifest: " + relative)
        payload[relative] = data
    payload["update_teacher_weight_parallel.py"] = (root / "update_teacher_weight_parallel.py").read_bytes()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payload.items():
            archive.writestr("teacher_weight_parallel_update/" + relative, data)
        archive.writestr("teacher_weight_parallel_update/SHA256SUMS.txt", "\n".join(
            hashlib.sha256(data).hexdigest() + "  " + relative for relative, data in payload.items()) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("teacher-weight-parallel ZIP integrity check failed")
    print(output)
    print("Bytes:", output.stat().st_size, "SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.freeze_manifest and not args.output:
        parser.error("Specify --freeze-manifest and/or --output")
    if args.freeze_manifest:
        freeze_manifest()
    if args.output:
        package(args.output)


if __name__ == "__main__":
    main()
