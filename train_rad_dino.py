"""RAD-DINO LoRA training entry; CLI flags override the model defaults."""

import os
from pathlib import Path
import sys

from foundation_sft.train import main


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    main([
        "--config", str(project_dir / "foundation_sft/configs/rad_dino.json"),
        "--action", "train", "--mode", "lora", "--device", "cuda:1",
        *sys.argv[1:],
    ])
