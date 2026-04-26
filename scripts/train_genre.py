from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.training.retrieval import train_genre


if __name__ == "__main__":
    train_genre()
