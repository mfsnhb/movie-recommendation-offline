from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.utils.io import (
    MULTI_RECALL_METRICS_PATH,
    RETRIEVAL_METRICS_PATH,
    get_final_metrics_path,
    get_ranking_metrics_path,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-models", default="deepfm,din")
    args = parser.parse_args()

    for path in (RETRIEVAL_METRICS_PATH, MULTI_RECALL_METRICS_PATH):
        if not path.exists():
            continue
        print(f"===== {path.name} =====")
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))

    for model_name in [name.strip().lower() for name in args.ranking_models.split(",") if name.strip()]:
        ranking_metrics_path = get_ranking_metrics_path(model_name)
        final_metrics_path = get_final_metrics_path(model_name)
        for path in (ranking_metrics_path, final_metrics_path):
            if not path.exists():
                continue
            print(f"===== {model_name} / {path.name} =====")
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
