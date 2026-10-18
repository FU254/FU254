#!/usr/bin/env python3
"""Install the additional chest VLMs, preserving prior experiments and inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

PAYLOAD_SHA256 = {
    "foundation_sft/__init__.py": "7a49ab04965be0b2cdf65fbb61f28c5e487f4dafb92acfe0b3f5b8bd87d91ba2",
    "foundation_sft/models/__init__.py": "0d6fb7511c2b0f71c35225871513484c2b8a93200f77f81ff43daf67193e4008",
    "foundation_sft/models/extra_vlm_registry.py": "0a3ec54f53bc7b01c3c5798ae063ac8d6f9849d104f2de031e8255d7ebe25bdf",
    "foundation_sft/models/extra_vlm_common.py": "9797773f2d85a7a6c5e37325bbfb9f2afbb0e8e56e501bfc8cbbf4ebc2e9807a",
    "foundation_sft/models/mgca.py": "5ce917d82cad479824414f8b3cf1d6e9c8f528dee8591afa35a56c7ce52911b7",
    "foundation_sft/models/biovil_t.py": "9b85d34d76f427ce1cebcd2ad9fe066e16ffc954ae1b68cd661829b48e8a9bd6",
    "foundation_sft/models/biovil_temporal.py": "f6d3a824fd910293301b88ada9ca4d6e4d89decf8a08be1cd842237f3f39f751",
    "foundation_sft/models/BIOVIL_LICENSE.txt": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
    "foundation_sft/models/medklip.py": "c5a43a9463b9ce719014d630c879744b456afc3736f7b94d346caf27e9db8095",
    "foundation_sft/models/cxr_clip.py": "659de6e93bd580f56fdc5ab595d959b22883826b02d7765441b48d0b4f7375c2",
    "foundation_sft/data.py": "985a8b083928b27313c04cf7c916f9899d8fd6e4db4870f5786c5c20d748afc6",
    "foundation_sft/pixel_cache.py": "30be5c3ef00c23d0817def2c36ac84fecb6589332260e4bcbf9ac3fb9453363e",
    "foundation_sft/extra_vlm_preprocessing.py": "62d74803c63b67e2d3289866cfe94a1ca8426fda648005451e81aa4bf8b4b773",
    "foundation_sft/extra_vlm_assets.py": "bee2242ef70553a4f1b295dea054583ba23ea9d4f99120812cfd90255628e060",
    "foundation_sft/configs/common_vlm.json": "1903dc33e9810462cc82cc09eb1216db6b3c493196e32225360a27880874be8c",
    "foundation_sft/configs/cxr_clip.json": "b286a3fe731356e542affdde20267222e82524eabd66a105076196dcb0a82677",
    "foundation_sft/configs/common_vlm_extra.json": "d727c94880b3097590d8504838446ab3353ae38d95ecbb42b3d578ddc0a26db3",
    "foundation_sft/configs/cxr_clip_extra.json": "6b7f9121ff29789b119e56e5b145883d0c72dbda063c846baa164fb8d7ce7c5f",
    "foundation_sft/configs/mgca.json": "7af7d1f32dbb68d5d5fd6f5ffd3c96e85925dca14d0372ee9974da76f5c7859b",
    "foundation_sft/configs/biovil_t.json": "c458a20bd5c37d00f2b38c5139cc5afc0dc614474567a771c03cb802566ce828",
    "foundation_sft/configs/medklip.json": "1efede5de5e58dea314df5e299e8471ba0059cffcc0c44afe8808c46feeb2191",
    "foundation_sft/configs/experiments_vlm_extra.json": "4987116e41410d5b211909eb1efa49fc0ff293a1c8f4b2d0527fd60aef215ae2",
    "foundation_sft/configs/experiments_vlm_extra_lora.json": "3395b634f404023cde590a76d1417dfee86c09f4c702aad3d7b8f00a9eac3105",
    "foundation_sft/requirements-vlm-extra.txt": "ca7abb60f7059831c92d62ecc0975c51deb0963e6e512179404a4fc090eaa699",
    "foundation_sft/EXTRA_VLM_SETUP.md": "3fb52e1d03b32677a83ebc547e8f30873ab4d55921ff03001d8c28113139ba1e",
    "download_extra_vlm_weights.py": "aea3581db3b64300334c2b7ea41c56413a953b1fcd98b611363bcca8f919ef08",
    "prepare_medklip_book.py": "5e36fbc57ea5424406b0ca092f99846c3dc1a0941d46893b0fe9ab2044246f7d",
    "show_foundation_results.py": "117d773bf67ff99e5151dfcf6a12ba37b1f5ebf5481a8a96b8033d79d4d4a693",
    "train_cxr_clip.py": "885a7ecc5454f14ba290ea7fc9698e847957262a96374ac3181b37ce464bf1a5",
    "train_mgca.py": "0cd0f9beeba7ac4ae59a7330391f234b46b387232bc41b9109de35227fb97901",
    "train_biovil_t.py": "6368c12936e0f32aa2707ffb480313d57764909e30d90d4b660ff3c31e060fff",
    "train_medklip.py": "23533393172451886e642c8713cd0a75aae8b9b2aee9b29994538c4c922de530"
}
PREVIOUS_SHA256 = {
    "foundation_sft/__init__.py": [
        "7a49ab04965be0b2cdf65fbb61f28c5e487f4dafb92acfe0b3f5b8bd87d91ba2"
    ],
    "foundation_sft/models/__init__.py": [
        "0ac0c81fe27240eec7bebab955f7d26eba58917c3ec182671a845967f5b683a9"
    ],
    "foundation_sft/models/cxr_clip.py": [
        "659de6e93bd580f56fdc5ab595d959b22883826b02d7765441b48d0b4f7375c2"
    ],
    "foundation_sft/data.py": [
        "0a3e9253dd6a65209222ecb9f111a7259aee91e85a4df8431c57f7b74bb996e3"
    ],
    "foundation_sft/pixel_cache.py": [
        "fc8f293c27e6d662388b4fc28423887e85d5130f073ed1682d45e6ca38e8e7da"
    ],
    "foundation_sft/configs/common_vlm.json": [
        "1903dc33e9810462cc82cc09eb1216db6b3c493196e32225360a27880874be8c"
    ],
    "foundation_sft/configs/cxr_clip.json": [
        "b286a3fe731356e542affdde20267222e82524eabd66a105076196dcb0a82677"
    ],
    "show_foundation_results.py": [
        "117d773bf67ff99e5151dfcf6a12ba37b1f5ebf5481a8a96b8033d79d4d4a693"
    ]
}
REGISTRY = "foundation_sft/models/__init__.py"
PRESERVE_EXISTING_CONFIGS = {"foundation_sft/configs/common_vlm.json", "foundation_sft/configs/cxr_clip.json"}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, relative):
    path = root / relative
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise ValueError("Refusing a symbolic-link path: " + relative)
    return path


def _prepare(project, source):
    if sys.version_info < (3, 10):
        raise ValueError("Use Python 3.10+, for example conda activate evax_lora")
    if not (project / "foundation_sft/train.py").is_file():
        raise ValueError("--project must be the existing EVA-X project")
    pending, originals, modes = {}, {}, {}
    for relative, checksum in PAYLOAD_SHA256.items():
        origin = _path(source, relative)
        if not origin.is_file() or origin.stat().st_size > 1024 * 1024:
            raise ValueError("Missing or oversized add-on payload: " + relative)
        data = origin.read_bytes()
        if _digest(data) != checksum:
            raise ValueError("Add-on source checksum mismatch: " + relative)
        if relative.endswith(".py"):
            compile(data, relative, "exec")
        elif relative.endswith(".json") and not isinstance(json.loads(data), dict):
            raise ValueError("Configuration payload must be an object: " + relative)
        target = _path(project, relative)
        if target.exists() and not target.is_file():
            raise ValueError("Destination is not a regular file: " + relative)
        previous = target.read_bytes() if target.exists() else None
        if previous is not None and relative in PRESERVE_EXISTING_CONFIGS:
            if not isinstance(json.loads(previous), dict):
                raise ValueError("Existing configuration is invalid: " + relative)
            continue
        if previous == data:
            continue
        if previous is not None and _digest(previous) not in PREVIOUS_SHA256.get(relative, []):
            raise ValueError("Unknown local changes; no files were updated: " + relative)
        pending[relative], originals[relative] = data, previous
        modes[relative] = target.stat().st_mode & 0o777 if previous is not None else 0o644
    return pending, originals, modes


def _atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".vlm-extra-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def install(project, source_dir, apply=False):
    project, source = Path(project).expanduser().resolve(), Path(source_dir).expanduser().resolve()
    pending, originals, modes = _prepare(project, source)
    if not apply or not pending:
        return {"apply": apply, "files": list(pending), "backup": None}
    if project == source:
        raise ValueError("Unpack the add-on separately from the existing project")
    lock = project / ".foundation-vlm-extra-update.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        pending, originals, modes = _prepare(project, source)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project / "runs/foundation_sft_vlm_extra/code_backups" / (stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(parents=True, mode=0o700)
        for relative, previous in originals.items():
            if previous is not None:
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(previous)
        (backup / "installation.json").write_text(json.dumps({
            "previous": {key: _digest(value) if value is not None else None for key, value in originals.items()},
            "updated": {key: _digest(value) for key, value in pending.items()},
        }, indent=2), encoding="utf-8")
        order = [key for key in pending if key != REGISTRY] + ([REGISTRY] if REGISTRY in pending else [])
        replaced = []
        try:
            for relative in order:
                target = _path(project, relative)
                current = target.read_bytes() if target.exists() else None
                if current != originals[relative]:
                    raise ValueError("File changed during installation: " + relative)
                _atomic_write(target, pending[relative], modes[relative])
                replaced.append(relative)
        except BaseException:
            for relative in reversed(replaced):
                target = _path(project, relative)
                if originals[relative] is None:
                    target.unlink()
                else:
                    _atomic_write(target, originals[relative], modes[relative])
            raise
        return {"apply": True, "files": order, "backup": str(backup)}
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(install(args.project, args.source_dir, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError) as error:
        print("Additional VLM installation failed:", str(error), file=sys.stderr)
        raise SystemExit(1)
