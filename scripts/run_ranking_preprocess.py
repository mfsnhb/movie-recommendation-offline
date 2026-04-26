from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.ranking.preprocess import run_ranking_preprocessing


if __name__ == "__main__":
    run_ranking_preprocessing()
