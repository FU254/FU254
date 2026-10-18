"""Additional chest VLM training entry; flags can override these defaults."""

import os
from pathlib import Path
import sys

from foundation_sft.train import main


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    main([
        "--config", str(project_dir / "foundation_sft/configs/cxr_clip_extra.json"),
        "--action", "train", "--mode", "linear_probe", "--device", "cuda:0",
        *sys.argv[1:],
    ])
