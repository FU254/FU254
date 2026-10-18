"""Lossless cache of deterministic display intensities before augmentation.

The cache is optional. Its NPY arrays contain the same float32 grayscale image
as the normal decoder/resizer, before random geometry or channel normalization.
The source file and decoder/preprocessing runtime are part of every cache key.
"""

import hashlib
import io
import os
from pathlib import Path
import tempfile

import numpy as np

from .cache import digest, file_signature, load_json, private_directory, runtime_signature, save_json


PIXEL_CACHE_VERSION = "deterministic_grayscale_float32_v1"


class PixelCacheError(ValueError):
    """Cache failures that do not expose source paths or patient identifiers."""


def _spec_value(spec, name, default=None):
    return spec.get(name, default) if isinstance(spec, dict) else getattr(spec, name, default)


def _resize_signature(preprocess):
    """Keep only operations before augmentation and channel normalization."""
    size = int(_spec_value(preprocess, "image_size"))
    mode = _spec_value(preprocess, "resize_mode", "resize")
    equalize = _spec_value(preprocess, "do_hist_equalization", False)
    interpolation = _spec_value(preprocess, "interpolation", "bicubic")
    if size < 1 or mode not in {"resize", "letterbox", "shortest_edge_center_crop", "chexzero_pad320",
                               "mgca_pad256_center_crop", "biovil_center_crop"}:
        raise PixelCacheError("Invalid deterministic image resize configuration.")
    if interpolation not in {"nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"}:
        raise PixelCacheError("Invalid deterministic image interpolation configuration.")
    if not isinstance(equalize, bool):
        raise PixelCacheError("Histogram equalization must be a boolean.")
    if mode == "chexzero_pad320" and (size != 224 or interpolation != "bicubic" or equalize):
        raise PixelCacheError("CheXzero preprocessing requires 224 pixels, bicubic interpolation, and no equalization.")
    edge = int(_spec_value(preprocess, "resize_shortest_edge") or size)
    if mode == "shortest_edge_center_crop" and edge < 1:
        raise PixelCacheError("Invalid deterministic image shortest edge configuration.")
    extra = {}
    if mode in {"mgca_pad256_center_crop", "biovil_center_crop"}:
        if interpolation != "bilinear" or equalize:
            raise PixelCacheError("Extra VLM preprocessing requires bilinear metadata and no histogram equalization.")
        if mode == "mgca_pad256_center_crop":
            if size != 224:
                raise PixelCacheError("MGCA requires pad256/crop224 geometry.")
            import cv2
            extra["opencv_version"] = cv2.__version__
        elif size != 448 or edge != 512:
            raise PixelCacheError("BioViL-T requires resize512/crop448 geometry.")
    return {
        "image_size": size,
        "resize_mode": mode,
        # An unused edge must not prevent otherwise identical models sharing.
        "resize_shortest_edge": edge if mode in {"shortest_edge_center_crop", "biovil_center_crop"} else None,
        "interpolation": interpolation,
        "do_hist_equalization": equalize,
        **extra,
    }


def _source_signature(path, expected_size, expected_mtime_ns):
    try:
        signature = file_signature(path)
    except (OSError, ValueError):
        raise PixelCacheError("The source image is unavailable; cached pixels cannot replace it.") from None
    if ((expected_size is not None and signature["size"] != int(expected_size))
            or (expected_mtime_ns is not None and signature["mtime_ns"] != int(expected_mtime_ns))):
        raise PixelCacheError("The source image changed since the frozen cohort was prepared.")
    return signature


def _valid_array(array, size):
    return (isinstance(array, np.ndarray) and array.dtype == np.dtype("float32")
            and array.shape == (size, size) and np.isfinite(array).all()
            and (array >= 0).all() and (array <= 1).all())


class PixelCache:
    """Reuse validated deterministic pixels across workers, epochs, and modes."""

    def __init__(self, cache_dir):
        self.directory = private_directory(Path(cache_dir).expanduser() / "pixels")

    def get(self, path, preprocess, build_callback, expected_size=None, expected_mtime_ns=None):
        """Return exact float32 intensities, rebuilding missing/corrupt entries.

        ``build_callback`` must decode and deterministically resize this source,
        without random augmentation, labels, or channel mean/std normalization.
        A source changed or removed after cohort preparation fails even on hits.
        """
        source = _source_signature(path, expected_size, expected_mtime_ns)
        spec = _resize_signature(preprocess)
        identity = {
            "version": PIXEL_CACHE_VERSION,
            "source": source,
            "resize": spec,
            "runtime": runtime_signature(),
        }
        key = digest(identity)
        metadata_path = self.directory / (key + ".json")
        array_path = self.directory / (key + ".npy")
        image = self._read(metadata_path, array_path, identity, spec["image_size"])
        if image is None:
            image = build_callback()
            if not _valid_array(image, spec["image_size"]):
                raise PixelCacheError("Deterministic image preprocessing returned invalid pixels.")
            self._unchanged(path, source, expected_size, expected_mtime_ns)
            # np.save preserves every float32 bit; no quantization or lossy codec.
            stream = io.BytesIO()
            np.save(stream, image, allow_pickle=False)
            payload = stream.getvalue()
            descriptor, temporary = tempfile.mkstemp(prefix=".pixels-", suffix=".tmp", dir=self.directory)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, array_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            save_json(metadata_path, {
                "identity": identity,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [spec["image_size"], spec["image_size"]],
                "dtype": "float32",
            })
        self._unchanged(path, source, expected_size, expected_mtime_ns)
        return image

    @staticmethod
    def _unchanged(path, previous, expected_size, expected_mtime_ns):
        current = _source_signature(path, expected_size, expected_mtime_ns)
        if current != previous:
            raise PixelCacheError("The source image changed while reading or caching its pixels.")

    @staticmethod
    def _read(metadata_path, array_path, identity, size):
        metadata = load_json(metadata_path)
        if (not isinstance(metadata, dict) or metadata.get("identity") != identity
                or metadata.get("shape") != [size, size] or metadata.get("dtype") != "float32"):
            return None
        try:
            payload = array_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
                return None
            image = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        return image if _valid_array(image, size) else None
