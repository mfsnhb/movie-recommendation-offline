from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.features.retrieval import run_retrieval_preprocessing


if __name__ == "__main__":
    run_retrieval_preprocessing()
