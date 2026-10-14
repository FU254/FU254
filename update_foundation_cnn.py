#!/usr/bin/env python3
"""Install the reviewed CNN add-on locally, preserving existing experiment files."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


PAYLOAD_SHA256 = {
    "foundation_sft/cnn_sources.py": "e4f741d720b34e1245f89b17819494eee8bd016a0c9a85d3cff20923b5257baa",
    "foundation_sft/models/cnn.py": "9c179a228c48946e0dbf592a69498bb1974e81befbbea9a9df4e13666c1e9b5d",
    "foundation_sft/models/cnn_checkpoint.py": "8e72f875e623939b8597dbede80dc6877c67a069541f5d18158d7b55435a743e",
    "foundation_sft/configs/common_cnn.json": "1a6fd56f1655dc62fad06e371e53014b59e4b3b88ebfef6827dde2168d03055f",
    "foundation_sft/configs/densenet121_imagenet.json": "08d3dd8daa2785f7cbf20e2e8377e2db16b62665b9cee031dee41d54446993a0",
    "foundation_sft/configs/densenet121_cxr.json": "b440e5fa9b68594a86d21137c3ee6842626347390c503ee36a26d605faa09987",
    "foundation_sft/configs/resnet18_cxr.json": "496de075e79b2d2cfffae9c3ef8198ac0161e67b8d51cade446fce4e74561dd5",
    "foundation_sft/configs/experiments_cnn.json": "713372c22a29f093d991bdeda4c0f262260b96eb798bbf56b5c19458d5824f7f",
    "foundation_sft/CNN_BASELINES.md": "915fb2b5d3f29a860edcac877a877e577f6ec1a26bca6962c6d718902b84185a",
    "download_cnn_weights.py": "79f838018b2fd794e5f07462dcc7e4d7f145c1ff15152757ce86a61cbdc61eb6",
    "foundation_sft/models/__init__.py": "0ac0c81fe27240eec7bebab955f7d26eba58917c3ec182671a845967f5b683a9"
}
# Earlier releases before the restricted numeric checkpoint loader are not
# compatible with this add-on. Accept the verified current main/VLM registries.
PREVIOUS_REGISTRIES = {'7a4ee990697a1ca79eb7b1b7dcd7fced971092cec72b00b48d21a9cee45e9034', 'c7bdc7014861514bdbdaf41b6711007668511f902b380730ae81da47e5e40d29'}
REGISTRY = "foundation_sft/models/__init__.py"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _prepare(project, source):
    if not (project / REGISTRY).is_file():
        raise ValueError("Project must be an existing EVA-X checkout")
    pending, originals, modes = {}, {}, {}
    for relative, expected in PAYLOAD_SHA256.items():
        data = (source / relative).read_bytes()
        if _digest(data) != expected:
            raise ValueError("Add-on source checksum mismatch: " + relative)
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json"):
            json.loads(data.decode("utf-8"))
        destination = project / relative
        if destination.is_symlink():
            raise ValueError("Refusing to replace a symbolic link: " + relative)
        previous = destination.read_bytes() if destination.exists() else None
        if previous is not None and _digest(previous) == expected:
            continue
        if previous is not None and (relative != REGISTRY or _digest(previous) not in PREVIOUS_REGISTRIES):
            raise ValueError("Unexpected local changes; no files updated: " + relative)
        pending[relative] = data
        originals[relative] = previous
        modes[relative] = destination.stat().st_mode & 0o777 if previous is not None else 0o644
    return pending, originals, modes


def _atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".cnn-update-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def install(project, source_dir, apply=False):
    project, source = Path(project).expanduser().resolve(), Path(source_dir).expanduser().resolve()
    pending, originals, modes = _prepare(project, source)
    if not apply or not pending:
        return {"apply": apply, "files": list(pending), "backup": None}
    lock = project / ".foundation-cnn-update.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Another CNN update holds the installation lock") from None
    os.close(descriptor)
    try:
        pending, originals, modes = _prepare(project, source)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project / "runs/foundation_sft_cnn/code_backups" / ("cnn_" + stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(parents=True, mode=0o700)
        for relative, data in originals.items():
            if data is not None:
                destination = backup / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        replaced = []
        # Activate the registry last, after all new modules/configurations exist.
        order = [name for name in pending if name != REGISTRY] + ([REGISTRY] if REGISTRY in pending else [])
        try:
            for relative in order:
                target = project / relative
                current = target.read_bytes() if target.exists() else None
                if current != originals[relative]:
                    raise ValueError("Project source changed during update: " + relative)
                _atomic_write(target, pending[relative], modes[relative])
                replaced.append(relative)
        except BaseException:
            for relative in reversed(replaced):
                if originals[relative] is None:
                    (project / relative).unlink()
                else:
                    _atomic_write(project / relative, originals[relative], modes[relative])
            raise
        return {"apply": True, "files": order, "backup": str(backup)}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument("--source-dir", required=True, help="Unpacked add-on repository directory")
    parser.add_argument("--apply", action="store_true", help="Apply the verified add-on and keep prior sources")
    args = parser.parse_args(argv)
    result = install(args.project, args.source_dir, args.apply)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, SyntaxError) as error:
        print("CNN update failed: " + str(error))
        raise SystemExit(1)
