from __future__ import annotations

from pathlib import Path

import pandas as pd

from offline.utils.io import RAW_DATA_DIR


def _normalize_user_ratings(df_ratings: pd.DataFrame) -> pd.DataFrame:
    df_ratings = df_ratings.copy()
    ratings = pd.to_numeric(df_ratings["rating"], errors="coerce").astype("float32")
    if float(ratings.max()) > 5.0:
        ratings = ratings / 2.0
    df_ratings["rating"] = ratings.clip(1.0, 5.0)
    return df_ratings



def load_raw_data(raw_data_dir: Path | None = None):
    dataset_dir = raw_data_dir or RAW_DATA_DIR
    df_movies = pd.read_pickle(dataset_dir / "movies.pkl")
    df_ratings = pd.read_pickle(dataset_dir / "ratings.pkl")
    df_users = pd.read_pickle(dataset_dir / "users.pkl")
    df_ratings = _normalize_user_ratings(df_ratings)
    return df_movies, df_ratings, df_users
