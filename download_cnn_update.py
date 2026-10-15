#!/usr/bin/env python3
"""Fetch the CNN add-on into a separate folder using Python's standard library."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request


SOURCE_REVISION = "3a6c43b8582ab4c024d06feaa4f6f8a03d5d1050"
SOURCE_URL = "https://raw.githubusercontent.com/FU254/FU254/" + SOURCE_REVISION
INSTALLER = "update_foundation_cnn.py"
INSTALLER_SHA256 = "1e97cbd129e8f543ae134982e68a4640a861ac8b1a7f40971d9d3b3fd1d7c886"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _fetch(relative, expected):
    with urllib.request.urlopen(SOURCE_URL + "/" + relative, timeout=30) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or _digest(data) != expected:
        raise ValueError("Downloaded file checksum mismatch: " + relative)
    return data


def _manifest(installer):
    # Read the verified installer's literal manifest without executing its code.
    tree = ast.parse(installer.decode("utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "PAYLOAD_SHA256"
                              for target in node.targets))
    manifest = ast.literal_eval(assignment.value)
    if not isinstance(manifest, dict) or len(manifest) != 11:
        raise ValueError("Unexpected CNN installation manifest")
    for relative, checksum in manifest.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError("Invalid add-on path")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError("Invalid add-on checksum")
    return manifest


def download_update(output):
    root = Path(output).expanduser().resolve()
    print("Fetching the verified CNN installer...", flush=True)
    installer = _fetch(INSTALLER, INSTALLER_SHA256)
    manifest = _manifest(installer)
    print("Downloading CNN installation files...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = dict(zip(manifest, executor.map(lambda name: _fetch(name, manifest[name]), manifest)))
    downloaded[INSTALLER] = installer
    # Verify every download and every existing destination before writing files.
    for relative, data in downloaded.items():
        target = root / relative
        try:
            target.resolve().relative_to(root)
        except ValueError:
            raise ValueError("Destination leaves the output folder: " + relative) from None
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise ValueError("Existing local file differs; use a new --output folder: " + str(target))
    for relative, data in downloaded.items():
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".cnn-fetch-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print("Completed: " + str(root), flush=True)
    print("Verified 12 files. Upload this folder through PyCharm Deployment.", flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cnn_update", help="Separate folder for the installation files")
    args = parser.parse_args(argv)
    download_update(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SyntaxError, StopIteration) as error:
        print("CNN code download failed: " + str(error), flush=True)
        raise SystemExit(1)
