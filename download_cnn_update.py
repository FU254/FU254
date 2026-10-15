#!/usr/bin/env python3
"""Fetch the CNN add-on into a separate folder using Python's standard library."""

import argparse
import ast
import hashlib
import http.client
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


SOURCE_REVISION = "3a6c43b8582ab4c024d06feaa4f6f8a03d5d1050"
SOURCE_URL = "https://raw.githubusercontent.com/FU254/FU254/" + SOURCE_REVISION
INSTALLER = "update_foundation_cnn.py"
INSTALLER_SHA256 = "1e97cbd129e8f543ae134982e68a4640a861ac8b1a7f40971d9d3b3fd1d7c886"
MAX_ATTEMPTS = 3
POWERSHELL_DOWNLOAD = (
    "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
    "Invoke-WebRequest -UseBasicParsing -Uri $env:CNN_CODE_FETCH_URL "
    "-OutFile $env:CNN_CODE_FETCH_PATH -TimeoutSec 45"
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _powershell_download(url):
    executable = shutil.which("powershell.exe")
    if executable is None:
        raise ValueError("The powershell transport requires Windows PowerShell (powershell.exe)")
    with tempfile.TemporaryDirectory(prefix="cnn-https-") as temporary:
        target = Path(temporary) / "payload"
        environment = os.environ.copy()
        environment.update(CNN_CODE_FETCH_URL=url, CNN_CODE_FETCH_PATH=str(target))
        result = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_DOWNLOAD],
            env=environment, capture_output=True, text=True, errors="replace", timeout=60,
        )
        if result.returncode:
            raise OSError("PowerShell HTTPS request failed: " + result.stderr.strip()[-1000:])
        with target.open("rb") as stream:
            return stream.read(1024 * 1024 + 1)


def _fetch(relative, expected, transport="urllib"):
    url = SOURCE_URL + "/" + relative
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if transport == "powershell":
                data = _powershell_download(url)
            else:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read(1024 * 1024 + 1)
            break
        except (OSError, http.client.IncompleteRead, subprocess.TimeoutExpired) as error:
            if (isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500
                    and error.code not in (408, 429)):
                raise
            if attempt == MAX_ATTEMPTS:
                raise OSError("Download failed after {} attempts: {} ({})".format(
                    MAX_ATTEMPTS, relative, error)) from error
            print("Connection interrupted; retrying {}/{}: {}".format(
                attempt + 1, MAX_ATTEMPTS, relative), flush=True)
            time.sleep(attempt)
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


def download_update(output, transport="urllib"):
    root = Path(output).expanduser().resolve()
    print("Fetching the verified CNN installer...", flush=True)
    installer = _fetch(INSTALLER, INSTALLER_SHA256, transport=transport)
    manifest = _manifest(installer)
    print("Downloading CNN installation files...", flush=True)
    downloaded = {}
    for relative, expected in manifest.items():
        print("Downloading: " + relative, flush=True)
        downloaded[relative] = _fetch(relative, expected, transport=transport)
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
    parser.add_argument("--transport", choices=("urllib", "powershell"), default="urllib",
                        help="Use Windows PowerShell HTTPS if Python's TLS connection fails")
    args = parser.parse_args(argv)
    download_update(args.output, transport=args.transport)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SyntaxError, StopIteration) as error:
        print("CNN code download failed: " + str(error), flush=True)
        raise SystemExit(1)
