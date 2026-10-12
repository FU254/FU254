"""Private, disposable cache artifacts bound to source files and runtime."""

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_signature(path):
    try:
        path = Path(path).expanduser().resolve()
    except RuntimeError:
        raise OSError("Cache source cannot be resolved") from None
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise OSError("Cache source is not a regular file")
    return {"path": str(path), "size": status.st_size, "mtime_ns": status.st_mtime_ns,
            "ctime_ns": status.st_ctime_ns, "ino": status.st_ino, "dev": status.st_dev}


@lru_cache(maxsize=1)
def _runtime_base():
    """Separate interpreter/decoder environments and invalidate changed code."""
    packages = {}
    for name in ("pydicom", "numpy", "pandas", "openpyxl", "pillow", "torch", "pylibjpeg",
                 "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pylibjpeg-rle",
                 "python-gdcm", "gdcm", "pyjpegls"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in ("data.py", "audit_data.py", "cache.py", "pixel_cache.py",
                 "cohort_cache.py", "audit_cache.py"):
        path = Path(__file__).parent / name
        if path.is_file():
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"version": 1, "python": list(sys.version_info[:3]),
            "executable": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).resolve()), "packages": packages,
            "sources": sources}


def runtime_signature():
    # A caller can change handler order/validation flags inside one process.
    # Keep these live even though interpreter/code/package metadata is stable.
    import pydicom
    handlers = []
    for handler in pydicom.config.pixel_data_handlers:
        try:
            available = bool(handler.is_available())
        except Exception:
            available = False
        handlers.append({"name": handler.__name__, "available": available})
    return {**_runtime_base(), "pixel_handlers": handlers,
            "enforce_valid_values": bool(pydicom.config.enforce_valid_values),
            "apply_j2k_corrections": bool(getattr(pydicom.config, "APPLY_J2K_CORRECTIONS", True))}


def private_directory(path):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def load_json(path):
    """A corrupt cache is a miss; never deserialize executable objects."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            envelope = json.load(handle)
        payload = envelope["payload"]
        if envelope.get("sha256") != digest(payload):
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return None


def save_json(path, payload):
    path = Path(path)
    private_directory(path.parent)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".cache-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"payload": payload, "sha256": digest(payload)}, handle,
                      ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
