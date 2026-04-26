from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.ranking import train_deepfm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-warm-start", action="store_true")
    args = parser.parse_args()
    train_deepfm(warm_start=not args.no_warm_start)
