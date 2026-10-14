#!/usr/bin/env python3
"""Download verified CNN assets on a connected computer, before offline training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from foundation_sft.cnn_sources import CNN_SOURCES


def _checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _verify(source, checksum):
    if source.get("sha256") and checksum != source["sha256"]:
        raise ValueError("Official checkpoint SHA-256 mismatch")
    if source.get("sha256_prefix") and not checksum.startswith(source["sha256_prefix"]):
        raise ValueError("Official checkpoint SHA-256 prefix mismatch")


def download_one(name, output_root):
    source = CNN_SOURCES[name]
    directory = Path(output_root).expanduser() / name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source["checkpoint_name"]
    if target.exists():
        checksum = _checksum(target)
        _verify(source, checksum)
        print("Already downloaded: " + str(target), flush=True)
    else:
        print("Downloading {} from {}".format(name, source["url"]), flush=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".cnn-download-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream, urllib.request.urlopen(source["url"], timeout=60) as response:
                total = response.headers.get("Content-Length")
                count = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    count += len(block)
                    if count % (8 * 1024 * 1024) == 0:
                        print("  {} MiB".format(count // (1024 * 1024)), flush=True)
                if count == 0 or (total and count != int(total)):
                    raise ValueError("Checkpoint download is empty or truncated")
            checksum = _checksum(Path(temporary))
            _verify(source, checksum)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    manifest = {"model": name, "source_url": source["url"], "pretraining": source["pretraining"],
                "bytes": target.stat().st_size, "sha256": checksum}
    with (directory / "download_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="foundation_weights")
    parser.add_argument("--models", nargs="+", choices=tuple(CNN_SOURCES), default=list(CNN_SOURCES))
    args = parser.parse_args(argv)
    failed = []
    for name in args.models:
        try:
            download_one(name, args.output)
        except (OSError, ValueError) as error:
            failed.append(name)
            print("Download failed for {}: {}".format(name, error), flush=True)
    print("Failed models: " + (", ".join(failed) if failed else "none"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
