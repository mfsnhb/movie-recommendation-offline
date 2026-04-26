from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.retrieval import train_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-warm-start", action="store_true")
    args = parser.parse_args()
    train_sequence(warm_start=not args.no_warm_start)
