"""Run the complete five-model supervised experiment workflow."""

import os
from pathlib import Path

from foundation_sft.workflow import main


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(main())
