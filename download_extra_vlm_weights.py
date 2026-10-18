#!/usr/bin/env python3
"""Get additional VLM assets on a computer with Internet access; never install Torch.

MGCA/MedKLIP checkpoints require selecting the author's file in Drive/Baidu.
Their official pages and destination paths are printed explicitly. A missing
manual file is reported as incomplete, never as a successful model download.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from foundation_sft.extra_vlm_assets import EXTRA_VLM_ASSETS, MEDKLIP_BOOK


def fingerprint(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def _validate(path, asset):
    result = fingerprint(path)
    if not result["bytes"]:
        raise ValueError("Empty downloaded asset: " + str(path))
    for algorithm in ("sha256", "md5"):
        if asset.get(algorithm) and result[algorithm] != asset[algorithm]:
            raise ValueError("Published checksum mismatch; existing files were preserved: " + str(path))
    return result


def download(asset, root):
    destination = root / asset["filename"]
    if destination.exists():
        print("Using existing:", destination, flush=True)
        return _validate(destination, asset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".vlm-download-", suffix=".part", dir=destination.parent)
    os.close(descriptor)
    try:
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "foundation-vlm-assets/1.0"})
        print("Downloading:", asset["url"], "\nSaving:", destination, flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as handle:
            expected = response.headers.get("Content-Length")
            size, last_progress = 0, 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                size += len(chunk)
                if size - last_progress >= 32 * 1024 * 1024:
                    print("Received:", size, "bytes", flush=True)
                    last_progress = size
            if expected is not None and size != int(expected):
                raise ValueError("Download was interrupted: byte count differs from Content-Length")
        result = _validate(temporary, asset)
        # A concurrently created destination is never overwritten.
        os.link(temporary, destination)
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def clinicalbert(root):
    from huggingface_hub import HfApi, snapshot_download

    source = "emilyalsentzer/Bio_ClinicalBERT"
    info = HfApi().model_info(source)
    available = {item.rfilename for item in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
    if weight not in available:
        raise ValueError("Official ClinicalBERT model file was not found")
    names = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", weight]
    destination = root / "medklip/clinicalbert"
    snapshot_download(repo_id=source, revision=info.sha, local_dir=str(destination),
                      allow_patterns=names, max_workers=2, etag_timeout=30)
    for name in ("config.json", "vocab.txt", weight):
        if not (destination / name).is_file():
            raise ValueError("ClinicalBERT snapshot is incomplete: " + name)
    return {"repo_id": source, "revision": info.sha,
            "files": {name: fingerprint(destination/name) for name in names if (destination/name).is_file()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(EXTRA_VLM_ASSETS), default=list(EXTRA_VLM_ASSETS))
    parser.add_argument("--output", default="foundation_weights")
    parser.add_argument("--plan", action="store_true", help="Print official sources/paths, with no writes or downloads")
    parser.add_argument("--check", action="store_true", help="Only inspect existing model/book files and print fingerprints")
    parser.add_argument("--clinicalbert", action="store_true", help="Also download the local BERT needed to prepare MedKLIP queries")
    args = parser.parse_args(argv)
    root = Path(args.output).expanduser().resolve()
    if args.plan:
        print(json.dumps({name: {**EXTRA_VLM_ASSETS[name], "destination": str(root/EXTRA_VLM_ASSETS[name]["filename"])}
                          for name in args.models}, ensure_ascii=False, indent=2))
        if "medklip" in args.models:
            print("Also required:", root/MEDKLIP_BOOK["filename"], "and disease_book_embeddings.pt")
        return 0
    records, incomplete = {}, []
    for name in args.models:
        asset = EXTRA_VLM_ASSETS[name]
        path = root / asset["filename"]
        try:
            if not path.exists() and (args.check or "manual_url" in asset):
                incomplete.append(name)
                print("MISSING:", path, flush=True)
                if "manual_url" in asset:
                    print(asset["instruction"], "\nOfficial weights:", asset["manual_url"], flush=True)
                continue
            records[name] = _validate(path, asset) if args.check else download(asset, root)
            print("Present:", name, records[name]["bytes"], "bytes; SHA256:", records[name]["sha256"], flush=True)
        except (OSError, ValueError) as error:
            incomplete.append(name)
            print("FAILED:", name, str(error), flush=True)
    if "medklip" in args.models:
        try:
            book = root / MEDKLIP_BOOK["filename"]
            records["medklip_book"] = _validate(book, MEDKLIP_BOOK) if args.check else download(MEDKLIP_BOOK, root)
            if args.clinicalbert and not args.check:
                records["clinicalbert"] = clinicalbert(root)
            embeddings = root / "medklip/disease_book_embeddings.pt"
            if not embeddings.is_file():
                incomplete.append("medklip_book_embeddings")
                print("NEXT: on the server run python prepare_medklip_book.py with the local ClinicalBERT files", flush=True)
        except (OSError, ValueError, ImportError) as error:
            incomplete.append("medklip_book")
            print("FAILED: MedKLIP auxiliary assets:", str(error), flush=True)
    print(json.dumps({"file_fingerprints": records, "incomplete": sorted(set(incomplete)),
                      "note": "File presence/fingerprints only. The GPU smoke test verifies checkpoint architecture."},
                     ensure_ascii=False, indent=2), flush=True)
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
