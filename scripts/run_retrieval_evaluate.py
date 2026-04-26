from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.retrieval import evaluate_retrieval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()
    evaluate_retrieval(topk=args.topk)
