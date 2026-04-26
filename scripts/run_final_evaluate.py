from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.ranking import evaluate_final_ranking


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deepfm", "din"], required=True)
    args = parser.parse_args()
    evaluate_final_ranking(args.model)
