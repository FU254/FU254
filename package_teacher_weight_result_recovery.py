#!/usr/bin/env python3
"""Package the standalone result-registration tool without training changes."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def package(output, root=None):
    root = Path(root or Path(__file__).resolve().parent)
    payload = {
        "recover_teacher_weight_result.py": (root / "recover_teacher_weight_result.py").read_bytes(),
        "README.md": (root / "foundation_sft/TEACHER_WEIGHT_RESULT_RECOVERY.md").read_bytes(),
    }
    compile(payload["recover_teacher_weight_result.py"], "recover_teacher_weight_result.py", "exec")
    payload["README.md"].decode("utf-8")
    checksums = "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n"
                        for name, data in payload.items())
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "teacher_weight_result_recovery/"
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(prefix + name, data)
        archive.writestr(prefix + "SHA256SUMS.txt", checksums)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Recovery archive integrity verification failed")
    print(output)
    print("Bytes:", output.stat().st_size)
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    package(parser.parse_args().output)
