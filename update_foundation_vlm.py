#!/usr/bin/env python3
"""Install the four-model update from a locally uploaded, reviewed source tree.

No network, dependency installation or training occurs here. Every selected
payload and existing source is checked before any project file is replaced.
Common dataset configuration, existing experiments and weights are excluded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import uuid


# Filled from the reviewed sources before publication; hashes are immutable.
FILES = {
    "foundation_sft/data.py": {
        "previous": [
            "fcf0bcfa9ff0bf9564ef491c32fd5ba65fc65ca271dd65328f03f238b12e2ec1"
        ],
        "updated": "0a3e9253dd6a65209222ecb9f111a7259aee91e85a4df8431c57f7b74bb996e3"
    },
    "foundation_sft/models/__init__.py": {
        "previous": [
            "92039b30fb2f2a68d617656f62a28ee179eed027a384edc2ea4fbbebae9cd48a"
        ],
        "updated": "7a4ee990697a1ca79eb7b1b7dcd7fced971092cec72b00b48d21a9cee45e9034"
    },
    "foundation_sft/models/common.py": {
        "previous": [
            "a266639641c39441f7bdec76482779c1f134dc51741d43cc26a31bf6e02a5e92"
        ],
        "updated": "4b980fedad66acca07503714584a07d12b6503eb7d700772f61abb8d3f7fd20f"
    },
    "foundation_sft/pixel_cache.py": {
        "previous": [
            "80cdb2328eeb11dc11126283c60bd8dc85afb91e9c71ce90eaa36aed01d17afa"
        ],
        "updated": "fc8f293c27e6d662388b4fc28423887e85d5130f073ed1682d45e6ca38e8e7da"
    },
    "foundation_sft/train.py": {
        "previous": [
            "db5e192b2028b224d83ef4ae7a5e88cdf8ad119a99fa0c224322bda295a72b8b",
            "eaea0a743beddb291969d28c62c53997a368669bd1a617484cb95e7c753baef7"
        ],
        "updated": "b8132ba69ba05ca3eba2d1241ed22803a8b62bd48347d8d75191242ca9f10865"
    },
    "foundation_sft/models/cxr_clip.py": {
        "previous": [
            None
        ],
        "updated": "659de6e93bd580f56fdc5ab595d959b22883826b02d7765441b48d0b4f7375c2"
    },
    "foundation_sft/models/openclip_vlm.py": {
        "previous": [
            None
        ],
        "updated": "df0b5519182318c39c8e8ba2dff778013060f77b1f3ca4cff5742caf6125fca6"
    },
    "foundation_sft/models/chexzero.py": {
        "previous": [
            None
        ],
        "updated": "d8347a1763b657948ed103e842f9df75b809b816da3e0e5df0f1ab2ff09de560"
    },
    "foundation_sft/training_history.py": {
        "previous": [
            None
        ],
        "updated": "d567ae73b1131c1f368eb2034abd8fdf554e2f7d52d84892f8f74d524a869bc0"
    },
    "foundation_sft/configs/common_vlm.json": {
        "previous": [
            None
        ],
        "updated": "1903dc33e9810462cc82cc09eb1216db6b3c493196e32225360a27880874be8c"
    },
    "foundation_sft/configs/experiments_vlm.json": {
        "previous": [
            None
        ],
        "updated": "163e67bbe222620ba5fa56feefe1cff402f7c4629911184afee2bf47939ea3a4"
    },
    "foundation_sft/configs/cxr_clip.json": {
        "previous": [
            None
        ],
        "updated": "b286a3fe731356e542affdde20267222e82524eabd66a105076196dcb0a82677"
    },
    "foundation_sft/configs/xray_siglip.json": {
        "previous": [
            None
        ],
        "updated": "99fbf5424c107728c505e380890dd10ebbc078eb2c572e64f6b6d087d1b67bfc"
    },
    "foundation_sft/configs/biomedclip.json": {
        "previous": [
            None
        ],
        "updated": "6b53042ddd993756ac2b9135bd54bd79cdff351f6b174f3f59d2689c4727a14c"
    },
    "foundation_sft/configs/chexzero.json": {
        "previous": [
            None
        ],
        "updated": "dfd9272a1eefe7fe3654466fa9397da4753e223d5a982f64bacb0be7a1d79490"
    },
    "foundation_sft/requirements-openclip.txt": {
        "previous": [
            None
        ],
        "updated": "c7ce5d6dd969deea6a417e199464ac0846ec05aacfe6e4a5d87e4dae3ae1c3f3"
    },
    "foundation_sft/FOUNDATION_VLM_SETUP.md": {
        "previous": [
            None
        ],
        "updated": "84115fddbdf4905302b9a812690d665581072bf1a0483add4bfc2f4dcda0b512"
    }
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _path(root, relative):
    path = root / relative
    for part in (path, *path.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise ValueError(f"Refusing a symbolic-link path: {relative}")
    return path


def _atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".vlm-update-", suffix=".tmp", dir=path.parent)
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


def install(project, source_dir, *, apply=False):
    project = Path(project).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    if project == source:
        raise ValueError("The uploaded source tree must differ from the existing project")
    if not (project / "foundation_sft" / "train.py").is_file():
        raise ValueError("--project must point to the existing EVA-X project")
    if not FILES:
        raise ValueError("This installer has no reviewed payload manifest")
    lock = project / ".foundation-vlm-update.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        pending, originals, modes = {}, {}, {}
        for relative, fingerprints in FILES.items():
            payload_path = _path(source, relative)
            if not payload_path.is_file() or payload_path.stat().st_size > 1024 * 1024:
                raise ValueError(f"Missing or oversized uploaded payload: {relative}")
            payload = payload_path.read_bytes()
            if _digest(payload) != fingerprints["updated"]:
                raise ValueError(f"Uploaded payload checksum differs: {relative}; no files were changed")
            if relative.endswith(".py"):
                compile(payload, relative, "exec")
            elif relative.endswith(".json"):
                if not isinstance(json.loads(payload), dict):
                    raise ValueError(f"Invalid configuration payload: {relative}")
            destination = _path(project, relative)
            if destination.exists() and not destination.is_file():
                raise ValueError(f"Destination is not a file: {relative}")
            original = destination.read_bytes() if destination.exists() else None
            digest = _digest(original) if original is not None else None
            if digest == fingerprints["updated"]:
                continue
            if digest not in fingerprints["previous"]:
                raise ValueError(f"Unexpected local changes in {relative}; no files were changed")
            pending[relative] = payload
            originals[relative] = original
            modes[relative] = stat.S_IMODE(destination.stat().st_mode) if original is not None else 0o644
        if not pending:
            print("Four-model sources are already up to date.", flush=True)
            return None
        print(json.dumps({"apply": apply, "files": list(pending)}, ensure_ascii=False, indent=2), flush=True)
        if not apply:
            print("Checks passed. Add --apply to install these reviewed files.", flush=True)
            return None
        # Confirm all destinations again, after all sources have been read.
        for relative, original in originals.items():
            destination = _path(project, relative)
            current = destination.read_bytes() if destination.exists() else None
            if current != original:
                raise ValueError(f"Destination changed during checks: {relative}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project / "runs" / "foundation_sft_vlm" / "code_backups" / f"vlm_{stamp}_{uuid.uuid4().hex[:8]}"
        _path(project, str(backup.relative_to(project)))
        backup.mkdir(parents=True, mode=0o700, exist_ok=False)
        for relative, data in originals.items():
            if data is not None:
                target = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        (backup / "installation.json").write_text(json.dumps({
            "files": list(pending), "new_files": [name for name, data in originals.items() if data is None],
        }, indent=2) + "\n", encoding="utf-8")
        replaced = []
        try:
            for relative, payload in pending.items():
                _atomic_write(project / relative, payload, modes[relative])
                replaced.append(relative)
        except BaseException:
            for relative in reversed(replaced):
                if originals[relative] is None:
                    (project / relative).unlink(missing_ok=True)
                else:
                    _atomic_write(project / relative, originals[relative], modes[relative])
            raise
        print(f"Updated {len(pending)} files. Previous sources: {backup}", flush=True)
        return backup
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    parser.add_argument("--source-dir", required=True, help="Uploaded extracted ZIP root containing foundation_sft/")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        install(args.project, args.source_dir, apply=args.apply)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"Four-model update failed: {error}\n")


if __name__ == "__main__":
    main()
