#!/usr/bin/env python3
"""Freeze reviewed ABCD payloads and package only the incremental code files."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

import update_distillation_abcd as installer


def _verify_previous_metadata(root):
    # Read the old official manifest without executing or changing the old tool.
    source = (root / "update_single_teacher_distillation.py").read_text(encoding="utf-8")
    parsed = ast.parse(source)
    matches = [ast.literal_eval(node.value) for node in parsed.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "PAYLOAD_SHA256" for target in node.targets)]
    if len(matches) != 1 or any(matches[0].get(name) != old for name, old in installer.PREVIOUS_SHA256.items()):
        raise ValueError("Old official checksum metadata differs from the reviewed upgrade identities")


def freeze_manifest(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    _verify_previous_metadata(root)
    checksums = {}
    for relative in installer.PAYLOAD_FILES:
        data = (root / relative).read_bytes()
        if len(data) > installer.MAX_BYTES:
            raise ValueError("Oversized ABCD payload: " + relative)
        checksums[relative] = hashlib.sha256(data).hexdigest()
    path = root / "update_distillation_abcd.py"
    source = path.read_text(encoding="utf-8")
    replacement = "PAYLOAD_SHA256 = " + json.dumps(checksums, indent=4)
    updated, count = re.subn(r"(?m)^PAYLOAD_SHA256 = \{[^}]*\}", lambda match: replacement, source, count=1)
    if count != 1:
        raise ValueError("ABCD checksum declaration not found")
    path.write_text(updated, encoding="utf-8")
    installer.PAYLOAD_SHA256 = checksums
    return checksums


def package(output, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    _verify_previous_metadata(root)
    if set(installer.PAYLOAD_SHA256) != set(installer.PAYLOAD_FILES):
        raise ValueError("Freeze the reviewed ABCD manifest before packaging")
    payload = {}
    for relative in installer.PAYLOAD_FILES:
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != installer.PAYLOAD_SHA256[relative]:
            raise ValueError("Code differs from the reviewed ABCD manifest: " + relative)
        payload[relative] = data
    payload["update_distillation_abcd.py"] = (root / "update_distillation_abcd.py").read_bytes()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, data in payload.items():
            archive.writestr("distillation_abcd_update/" + relative, data)
        archive.writestr("distillation_abcd_update/SHA256SUMS.txt", "\n".join(
            hashlib.sha256(data).hexdigest() + "  " + relative for relative, data in payload.items()) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ABCD ZIP integrity check failed")
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
