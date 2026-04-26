from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.retrieval import run_retrieval_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-build-multi-recall", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()
    run_retrieval_training(
        routes=args.routes,
        warm_start=not args.no_warm_start,
        build_multi_recall=not args.no_build_multi_recall,
        evaluate=not args.no_evaluate,
    )
