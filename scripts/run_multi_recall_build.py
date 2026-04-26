from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.evaluate.multi_recall import run_multi_recall_build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default=None)
    parser.add_argument("--topk", type=int, default=None)
    args = parser.parse_args()
    run_multi_recall_build(routes=args.routes, topk=args.topk)
