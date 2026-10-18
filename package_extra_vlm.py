#!/usr/bin/env python3
"""Build a self-contained, checksum-verified incremental VLM code ZIP."""

import argparse
import hashlib
from pathlib import Path
import zipfile

import update_foundation_vlm_extra as installer


def package(output):
    root = Path(__file__).resolve().parent
    payload = {relative:(root/relative).read_bytes() for relative in installer.PAYLOAD_SHA256}
    for relative,data in payload.items():
        if hashlib.sha256(data).hexdigest()!=installer.PAYLOAD_SHA256[relative]:
            raise ValueError("Code package differs from reviewed installer manifest: "+relative)
    payload["update_foundation_vlm_extra.py"]=(root/"update_foundation_vlm_extra.py").read_bytes()
    output=Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",zipfile.ZIP_DEFLATED) as archive:
        for relative,data in payload.items():
            archive.writestr("vlm_extra_update/"+relative,data)
        archive.writestr("vlm_extra_update/SHA256SUMS.txt","\n".join(
            hashlib.sha256(data).hexdigest()+"  "+relative for relative,data in payload.items())+"\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Code ZIP integrity check failed")
    print(output)
    print("Bytes:",output.stat().st_size,"SHA256:",hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    package(parser.parse_args().output)
