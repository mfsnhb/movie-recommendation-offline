from __future__ import annotations

from pathlib import Path

import pandas as pd

from offline.utils.io import RAW_DATA_DIR



def load_raw_data(raw_data_dir: Path | None = None):
    dataset_dir = raw_data_dir or RAW_DATA_DIR
    df_movies = pd.read_pickle(dataset_dir / "movies.pkl")
    df_ratings = pd.read_pickle(dataset_dir / "ratings.pkl")
    df_users = pd.read_pickle(dataset_dir / "users.pkl")
    return df_movies, df_ratings, df_users
