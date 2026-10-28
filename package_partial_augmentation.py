#!/usr/bin/env python3
"""Freeze and package the small augmentation update using the existing installer engine."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

import update_partial_augmentation as installer


def freeze_manifest(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    engine = installer._engine
    checksums = {}
    for relative in installer.PAYLOAD_FILES:
        origin = engine._path(root, relative)
        if not origin.is_file() or origin.stat().st_size > engine.MAX_BYTES:
            raise ValueError("Missing or oversized augmentation payload: " + relative)
        data = origin.read_bytes()
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json"):
            if not isinstance(json.loads(data), dict):
                raise ValueError("Configuration payload must be an object: " + relative)
        else:
            data.decode("utf-8")
        checksums[relative] = hashlib.sha256(data).hexdigest()
    path = root / "update_partial_augmentation.py"
    replacement = "PAYLOAD_SHA256 = " + json.dumps(checksums, indent=4)
    updated, count = re.subn(r"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", lambda match: replacement,
                             path.read_text(encoding="utf-8"), count=1)
    if count != 1:
        raise ValueError("Augmentation checksum declaration not found")
    path.write_text(updated, encoding="utf-8")
    return checksums


def package(output, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    spec = importlib.util.spec_from_file_location("_packaged_augmentation_installer", root / "update_partial_augmentation.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if set(frozen.PAYLOAD_SHA256) != set(frozen.PAYLOAD_FILES):
        raise ValueError("Freeze the reviewed augmentation manifest before packaging")
    payload = {}
    for relative in frozen.PAYLOAD_FILES:
        data = frozen._engine._path(root, relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen.PAYLOAD_SHA256[relative]:
            raise ValueError("Code differs from the reviewed augmentation manifest: " + relative)
        payload[relative] = data
    for relative in ("update_partial_augmentation.py", frozen.ENGINE_FILE):
        payload[relative] = frozen._engine._path(root, relative).read_bytes()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payload.items():
            archive.writestr("partial_augmentation_update/" + relative, data)
        archive.writestr("partial_augmentation_update/SHA256SUMS.txt", "\n".join(
            hashlib.sha256(data).hexdigest() + "  " + relative for relative, data in payload.items()) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Augmentation ZIP integrity check failed")
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
