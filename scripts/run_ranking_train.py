from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.ranking import run_ranking_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deepfm", "din"], default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-final-evaluate", action="store_true")
    args = parser.parse_args()
    run_ranking_training(
        model_name=args.model,
        models=args.models,
        warm_start=not args.no_warm_start,
        evaluate=not args.no_evaluate,
        final_evaluate=not args.no_final_evaluate,
    )
