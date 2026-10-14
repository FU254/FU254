#!/usr/bin/env python3
"""Install the reviewed recovery update from an immutable repository commit.

Only two Python sources are updated. All payloads are checked before writing;
the known previous trainer is backed up, and unexpected local edits are refused.
This uses the standard library so it can run in the server's base environment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
import urllib.request
import uuid


REPOSITORY = "https://raw.githubusercontent.com/FU254/FU254"
FILES = {
    "foundation_sft/train.py": {
        "previous": "eaea0a743beddb291969d28c62c53997a368669bd1a617484cb95e7c753baef7",
        "updated": "db5e192b2028b224d83ef4ae7a5e88cdf8ad119a99fa0c224322bda295a72b8b",
    },
    "foundation_sft/recover_experiment.py": {
        "previous": None,
        "updated": "2246a488130f2ff473e508b53e6f320318a9c3e2de6688c4923aaed184c6a847",
    },
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _download(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("The source download is unexpectedly large")
    return data


def _atomic_write(path, data, mode):
    descriptor, name = tempfile.mkstemp(prefix=".recovery-update-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install(project, revision, *, download=_download):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be the complete immutable commit SHA")
    project = Path(project).expanduser().resolve()
    if not (project / "foundation_sft" / "train.py").is_file():
        raise ValueError("--project must point to the existing EVA-X project")
    lock = project / ".foundation-recovery-update.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("Another recovery update holds the installation lock") from error
    os.close(descriptor)
    try:
        originals, modes, pending = {}, {}, {}
        for relative, fingerprints in FILES.items():
            destination = project / relative
            if destination.is_symlink():
                raise ValueError(f"Refusing to replace a symbolic link: {relative}")
            data = destination.read_bytes() if destination.exists() else None
            if data is not None and _sha256(data) == fingerprints["updated"]:
                continue
            if ((data is None and fingerprints["previous"] is not None)
                    or (data is not None and _sha256(data) != fingerprints["previous"])):
                raise ValueError(f"Unexpected local changes in {relative}; no sources were updated")
            originals[relative] = data
            modes[relative] = stat.S_IMODE(destination.stat().st_mode) if data is not None else 0o644
            pending[relative] = None
        if not pending:
            print("Recovery sources are already up to date.", flush=True)
            return None
        # Fetch, checksum and syntax-check every source before modifying either.
        for relative in pending:
            data = download(f"{REPOSITORY}/{revision}/{relative}")
            if _sha256(data) != FILES[relative]["updated"]:
                raise ValueError(f"Source checksum mismatch for {relative}; no sources were updated")
            compile(data, relative, "exec")
            pending[relative] = data
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project / "runs" / "foundation_sft" / "code_backups" / f"fp32_{stamp}_{uuid.uuid4().hex[:8]}"
        backup.mkdir(parents=True, mode=0o700, exist_ok=False)
        for relative, data in originals.items():
            if data is not None:
                (backup / Path(relative).name).write_bytes(data)
        replaced = []
        try:
            for relative, data in pending.items():
                _atomic_write(project / relative, data, modes[relative])
                replaced.append(relative)
        except BaseException:
            for relative in reversed(replaced):
                data = originals[relative]
                if data is None:
                    (project / relative).unlink(missing_ok=True)
                else:
                    _atomic_write(project / relative, data, modes[relative])
            raise
        print(f"Updated {', '.join(pending)}", flush=True)
        print(f"Previous source backup: {backup}", flush=True)
        print("Configurations, checkpoints, caches and suite records were retained.", flush=True)
        return backup
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args(argv)
    try:
        install(arguments.project, arguments.revision)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"Recovery update failed: {error}\n")


if __name__ == "__main__":
    main()
