from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from offline.data.loaders import load_raw_data
from offline.features.multimodal import build_or_load_multimodal_embeddings, resolve_multimodal_settings
from offline.utils.io import (
    CONFIG_DIR,
    ITEM_CATALOG_PATH,
    RANKING_FEATURE_DICT_PATH,
    RANKING_SAMPLE_PATH,
    RANKING_VOCAB_DICT_PATH,
    save_pickle,
)
from offline.utils.logging import get_logger


logger = get_logger("offline.features.ranking")
_SAMPLE_PROTOCOL = "prefix_positive_targets_v9_all_history"
_ID_DTYPE = np.uint16
_POPULARITY_BUCKET_COUNT = 10
_AVERAGE_RATING_BUCKET_COUNT = 5
_RECENCY_BUCKET_BOUNDARIES_DAYS = (1, 3, 7, 14, 30, 90, 365)
_RECENCY_BUCKET_BOUNDARIES_SECONDS = np.asarray(_RECENCY_BUCKET_BOUNDARIES_DAYS, dtype=np.int64) * 24 * 60 * 60
_RECENCY_BUCKET_COUNT = len(_RECENCY_BUCKET_BOUNDARIES_DAYS) + 2


def process_features_for_ranking(df_movies, df_ratings, df_users):
    user_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]

    df_users = df_users[user_columns].copy()
    df_movies = df_movies[["movie_id", "genres", "isAdult", "startYear", "averageRating"]].copy()
    df_movies["genres_raw"] = df_movies["genres"].fillna("unknown").astype(str)
    df_movies["genre_tokens"] = df_movies["genres_raw"].str.split("|")
    df_movies["isAdult"] = df_movies["isAdult"].fillna(False)
    df_movies["startYear_raw"] = pd.to_numeric(df_movies["startYear"], errors="coerce").fillna(0).astype(np.float32)
    df_movies["startYear"] = df_movies["startYear_raw"].astype(np.int32)
    df_movies["averageRating_raw"] = pd.to_numeric(df_movies["averageRating"], errors="coerce").fillna(0).clip(0, 10).astype(np.float32)
    df_movies["averageRating"] = np.ceil(df_movies["averageRating_raw"] / 2.0).astype(np.int32)
    df_ratings = df_ratings[ratings_columns].copy()
    df_ratings["timestamp"] = df_ratings["timestamp"].astype(np.int64)

    user_vocab = {}
    for feat_name in ["user_id", "gender", "age", "occupation", "zip_code"]:
        label_encoder = LabelEncoder()
        df_users[f"{feat_name}_encoded"] = label_encoder.fit_transform(df_users[feat_name]) + 1
        user_vocab[feat_name] = label_encoder.classes_

    movie_vocab = {}
    for feat_name in ["movie_id", "isAdult", "startYear"]:
        label_encoder = LabelEncoder()
        df_movies[feat_name] = df_movies[feat_name].fillna(0)
        df_movies[f"{feat_name}_encoded"] = label_encoder.fit_transform(df_movies[feat_name].astype(str)) + 1
        movie_vocab[feat_name] = label_encoder.classes_

    genre_token_encoder = LabelEncoder()
    exploded_genres = df_movies[["movie_id", "genre_tokens"]].explode("genre_tokens")
    exploded_genres["genre_tokens"] = exploded_genres["genre_tokens"].fillna("unknown").astype(str)
    exploded_genres["genre_token_encoded"] = genre_token_encoder.fit_transform(exploded_genres["genre_tokens"]) + 1
    genre_token_map = exploded_genres.groupby("movie_id", sort=False)["genre_token_encoded"].apply(list).to_dict()
    df_movies["genre_tokens_encoded"] = df_movies["movie_id"].map(lambda movie_id: genre_token_map.get(movie_id, [])).apply(lambda values: [int(v) for v in values])
    movie_vocab["genres"] = genre_token_encoder.classes_
    df_movies["averageRating_encoded"] = df_movies["averageRating"].astype(np.int32)
    movie_vocab["averageRating"] = np.arange(1, _AVERAGE_RATING_BUCKET_COUNT + 1, dtype=np.int32)

    df_merged = df_ratings.merge(
        df_users[["user_id", "user_id_encoded", "gender_encoded", "age_encoded", "occupation_encoded", "zip_code_encoded"]],
        on="user_id",
        how="left",
    )
    df_merged = df_merged.merge(
        df_movies[["movie_id", "movie_id_encoded", "isAdult_encoded", "startYear_encoded", "averageRating_encoded", "startYear_raw", "averageRating_raw", "genre_tokens_encoded"]],
        on="movie_id",
        how="left",
    )
    df_merged = df_merged.rename(
        columns={
            "user_id_encoded": "user_id_enc",
            "gender_encoded": "gender",
            "age_encoded": "age",
            "occupation_encoded": "occupation",
            "zip_code_encoded": "zip_code",
            "movie_id_encoded": "movie_id_enc",
            "isAdult_encoded": "isAdult",
            "startYear_encoded": "startYear",
            "averageRating_encoded": "averageRating",
        }
    )
    df_merged["user_id_original"] = df_merged["user_id"].astype(np.int32)
    df_merged["user_id"] = df_merged["user_id_enc"].astype(np.int32)
    df_merged["movie_id"] = df_merged["movie_id_enc"].astype(np.int32)
    df_merged["startYear_raw"] = df_merged["startYear_raw"].astype(np.float32)
    df_merged["averageRating_raw"] = df_merged["averageRating_raw"].astype(np.float32)
    return df_merged, user_vocab, movie_vocab


def _allocate_split_arrays(sample_count: int, max_seq_len: int) -> dict[str, np.ndarray]:
    return {
        "user_id": np.zeros(sample_count, dtype=_ID_DTYPE),
        "gender": np.zeros(sample_count, dtype=_ID_DTYPE),
        "age": np.zeros(sample_count, dtype=_ID_DTYPE),
        "occupation": np.zeros(sample_count, dtype=_ID_DTYPE),
        "zip_code": np.zeros(sample_count, dtype=_ID_DTYPE),
        "user_id_raw": np.zeros(sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((sample_count, max_seq_len), dtype=_ID_DTYPE),
        "hist_rating": np.zeros((sample_count, max_seq_len), dtype=np.float32),
        "hist_recency_bucket": np.zeros((sample_count, max_seq_len), dtype=_ID_DTYPE),
        "hist_length": np.zeros(sample_count, dtype=_ID_DTYPE),
        "low_rating_movie_id": np.zeros((sample_count, max_seq_len), dtype=_ID_DTYPE),
        "target_movie_id": np.zeros(sample_count, dtype=_ID_DTYPE),
        "target_rating": np.zeros(sample_count, dtype=np.float32),
    }


def _fill_padded_row(target: np.ndarray, values: list[int]) -> int:
    clipped = values[-len(target) :]
    length = len(clipped)
    if length > 0:
        target[-length:] = np.asarray(clipped, dtype=target.dtype)
    return length



def _fill_padded_float_row(target: np.ndarray, values: list[float]) -> None:
    clipped = values[-len(target) :]
    if clipped:
        target[-len(clipped) :] = np.asarray(clipped, dtype=target.dtype)



def _build_recency_buckets(history_timestamps: list[int], target_timestamp: int, width: int) -> np.ndarray:
    timestamps = np.asarray(history_timestamps, dtype=np.int64).reshape(-1)[-width:]
    buckets = np.zeros(width, dtype=_ID_DTYPE)
    if timestamps.size > 0:
        deltas = np.maximum(np.int64(target_timestamp) - timestamps, 0)
        bucket_ids = np.searchsorted(_RECENCY_BUCKET_BOUNDARIES_SECONDS, deltas, side="left") + 1
        buckets[-bucket_ids.size :] = bucket_ids.astype(_ID_DTYPE, copy=False)
    return buckets


def _split_history_by_rating(
    movie_sequence: list[int],
    rating_sequence: list[float],
    history_timestamps: list[int],
    target_idx: int,
    negative_rating_max: float,
) -> tuple[list[int], list[float], list[int], list[int]]:
    hist_movies: list[int] = []
    hist_ratings: list[float] = []
    hist_timestamps: list[int] = []
    low_rating_movies: list[int] = []
    for movie_id, rating, timestamp in zip(movie_sequence[:target_idx], rating_sequence[:target_idx], history_timestamps[:target_idx], strict=True):
        rating_value = float(rating)
        hist_movies.append(int(movie_id))
        hist_ratings.append(rating_value)
        hist_timestamps.append(int(timestamp))
        if rating_value <= float(negative_rating_max):
            low_rating_movies.append(int(movie_id))
    return hist_movies, hist_ratings, hist_timestamps, low_rating_movies


def _write_sample_row(
    store: dict[str, np.ndarray],
    row_idx: int,
    user_values: tuple[int, int, int, int, int],
    user_id_raw: int,
    hist_movies: list[int],
    hist_ratings: list[float],
    hist_timestamps: list[int],
    low_rating_movies: list[int],
    target_movie_id: int,
    target_rating: float,
    target_timestamp: int,
) -> None:
    user_id, gender, age, occupation, zip_code = user_values
    store["user_id"][row_idx] = user_id
    store["gender"][row_idx] = gender
    store["age"][row_idx] = age
    store["occupation"][row_idx] = occupation
    store["zip_code"][row_idx] = zip_code
    store["user_id_raw"][row_idx] = user_id_raw
    store["hist_movie_id"][row_idx].fill(0)
    store["hist_rating"][row_idx].fill(0.0)
    store["hist_recency_bucket"][row_idx].fill(0)
    store["low_rating_movie_id"][row_idx].fill(0)
    hist_length = _fill_padded_row(store["hist_movie_id"][row_idx], hist_movies)
    _fill_padded_float_row(store["hist_rating"][row_idx], hist_ratings)
    store["hist_recency_bucket"][row_idx] = _build_recency_buckets(hist_timestamps, target_timestamp, store["hist_recency_bucket"][row_idx].shape[0])
    store["hist_length"][row_idx] = hist_length
    _fill_padded_row(store["low_rating_movie_id"][row_idx], low_rating_movies)
    store["target_movie_id"][row_idx] = target_movie_id
    store["target_rating"][row_idx] = np.float32(target_rating)


def _write_target_sample(
    store: dict[str, np.ndarray],
    row_idx: int,
    user_values: tuple[int, int, int, int, int],
    user_id_raw: int,
    movie_sequence: list[int],
    rating_sequence: list[float],
    timestamp_sequence: list[int],
    target_idx: int,
    negative_rating_max: float,
) -> None:
    hist_movies, hist_ratings, hist_timestamps, low_rating_movies = _split_history_by_rating(
        movie_sequence,
        rating_sequence,
        timestamp_sequence,
        target_idx,
        negative_rating_max,
    )
    _write_sample_row(
        store,
        row_idx=row_idx,
        user_values=user_values,
        user_id_raw=user_id_raw,
        hist_movies=hist_movies,
        hist_ratings=hist_ratings,
        hist_timestamps=hist_timestamps,
        low_rating_movies=low_rating_movies,
        target_movie_id=int(movie_sequence[target_idx]),
        target_rating=float(rating_sequence[target_idx]),
        target_timestamp=int(timestamp_sequence[target_idx]),
    )


def _popularity_buckets(item_popularity: np.ndarray) -> np.ndarray:
    counts = np.asarray(item_popularity, dtype=np.float32)
    buckets = np.zeros(counts.shape[0], dtype=np.int32)
    nonzero = counts > 0
    if np.any(nonzero):
        scaled = np.log1p(counts[nonzero])
        max_value = float(scaled.max())
        if max_value > 0:
            buckets[nonzero] = np.ceil(scaled / max_value * _POPULARITY_BUCKET_COUNT).astype(np.int32)
        else:
            buckets[nonzero] = 1
    return buckets



def _build_item_feature_arrays(df_merged: pd.DataFrame, total_item_count: int, item_popularity: np.ndarray | None = None) -> dict[str, np.ndarray]:
    genre_count = max(
        (
            int(np.asarray(tokens, dtype=np.int32).max())
            for tokens in df_merged["genre_tokens_encoded"].tolist()
            if np.asarray(tokens, dtype=np.int32).size > 0
        ),
        default=0,
    )
    item_features = {
        "genres": np.zeros((total_item_count + 1, genre_count), dtype=np.int32),
        "isAdult": np.zeros(total_item_count + 1, dtype=np.int32),
        "startYear": np.zeros(total_item_count + 1, dtype=np.int32),
        "startYear_raw": np.zeros(total_item_count + 1, dtype=np.float32),
        "averageRating": np.zeros(total_item_count + 1, dtype=np.int32),
        "averageRating_raw": np.zeros(total_item_count + 1, dtype=np.float32),
        "popularity": _popularity_buckets(np.zeros(total_item_count + 1, dtype=np.int64) if item_popularity is None else item_popularity),
    }
    deduped = df_merged[["movie_id", "isAdult", "startYear", "startYear_raw", "averageRating", "averageRating_raw", "genre_tokens_encoded"]].drop_duplicates(subset=["movie_id"])
    for row in deduped.itertuples(index=False):
        movie_id = int(row.movie_id)
        genre_tokens = np.asarray(list(row.genre_tokens_encoded) if row.genre_tokens_encoded is not None else [], dtype=np.int32)
        valid_genres = genre_tokens[genre_tokens > 0] - 1
        item_features["genres"][movie_id, valid_genres] = 1
        item_features["isAdult"][movie_id] = int(row.isAdult)
        item_features["startYear"][movie_id] = int(row.startYear)
        item_features["startYear_raw"][movie_id] = np.float32(row.startYear_raw)
        item_features["averageRating"][movie_id] = int(row.averageRating)
        item_features["averageRating_raw"][movie_id] = np.float32(row.averageRating_raw)
    return item_features


def build_prefix_train_eval_samples(
    df_merged: pd.DataFrame,
    total_item_count: int,
    max_seq_len: int,
    negative_rating_max: float,
    positive_rating_min: float,
    neutral_rating: float = 3.0,
    history_policy: str = "all_ratings",
) -> dict:
    df_sorted = df_merged.sort_values(["user_id", "timestamp"], kind="stable")

    train_sample_count = 0
    validation_sample_count = 0
    test_sample_count = 0
    positive_threshold = float(positive_rating_min)
    for _, user_rows in df_sorted.groupby("user_id", sort=False):
        history_length = len(user_rows)
        if history_length < 2:
            continue
        rating_sequence = [float(rating) for rating in user_rows["rating"].tolist()]
        positive_target_indices = [idx for idx in range(1, history_length) if rating_sequence[idx] >= positive_threshold]
        if len(positive_target_indices) < 2:
            continue
        validation_sample_count += 1
        test_sample_count += 1
        train_sample_count += max(len(positive_target_indices) - 2, 0)

    logger.info(
        "Ranking sample allocation | protocol=%s | train_samples=%s | validation_samples=%s | test_samples=%s | max_seq_len=%s | positive_rating_min=%s | negative_rating_max=%s | neutral_rating=%s | history_policy=%s",
        _SAMPLE_PROTOCOL,
        train_sample_count,
        validation_sample_count,
        test_sample_count,
        max_seq_len,
        positive_threshold,
        negative_rating_max,
        neutral_rating,
        history_policy,
    )

    train_store = _allocate_split_arrays(train_sample_count, max_seq_len)
    validation_store = _allocate_split_arrays(validation_sample_count, max_seq_len)
    test_store = _allocate_split_arrays(test_sample_count, max_seq_len)
    item_popularity = np.zeros(total_item_count + 1, dtype=np.int64)

    train_row = 0
    validation_row = 0
    test_row = 0
    for _, user_rows in df_sorted.groupby("user_id", sort=False):
        movie_sequence = [int(movie_id) for movie_id in user_rows["movie_id"].tolist()]
        rating_sequence = [float(rating) for rating in user_rows["rating"].tolist()]
        timestamp_sequence = [int(timestamp) for timestamp in user_rows["timestamp"].tolist()]
        history_length = len(movie_sequence)
        if history_length < 2:
            continue
        positive_target_indices = [idx for idx in range(1, history_length) if rating_sequence[idx] >= positive_threshold]
        if len(positive_target_indices) < 2:
            continue

        user_id_raw = int(user_rows["user_id_original"].iloc[0])
        user_values = (
            int(user_rows["user_id"].iloc[0]),
            int(user_rows["gender"].iloc[0]),
            int(user_rows["age"].iloc[0]),
            int(user_rows["occupation"].iloc[0]),
            int(user_rows["zip_code"].iloc[0]),
        )

        validation_target_idx = positive_target_indices[-2]
        test_target_idx = positive_target_indices[-1]
        _write_target_sample(
            validation_store,
            row_idx=validation_row,
            user_values=user_values,
            user_id_raw=user_id_raw,
            movie_sequence=movie_sequence,
            rating_sequence=rating_sequence,
            timestamp_sequence=timestamp_sequence,
            target_idx=validation_target_idx,
            negative_rating_max=negative_rating_max,
        )
        validation_row += 1

        _write_target_sample(
            test_store,
            row_idx=test_row,
            user_values=user_values,
            user_id_raw=user_id_raw,
            movie_sequence=movie_sequence,
            rating_sequence=rating_sequence,
            timestamp_sequence=timestamp_sequence,
            target_idx=test_target_idx,
            negative_rating_max=negative_rating_max,
        )
        test_row += 1

        for target_idx in positive_target_indices[:-2]:
            target_movie_id = int(movie_sequence[target_idx])
            _write_target_sample(
                train_store,
                row_idx=train_row,
                user_values=user_values,
                user_id_raw=user_id_raw,
                movie_sequence=movie_sequence,
                rating_sequence=rating_sequence,
                timestamp_sequence=timestamp_sequence,
                target_idx=target_idx,
                negative_rating_max=negative_rating_max,
            )
            item_popularity[target_movie_id] += 1
            train_row += 1

    return {
        "protocol": _SAMPLE_PROTOCOL,
        "train": train_store,
        "validation": validation_store,
        "test": test_store,
        "all_item_ids": np.arange(1, total_item_count + 1, dtype=np.int32),
        "item_popularity": item_popularity,
        "rating_semantics": {
            "positive_rating_min": float(positive_rating_min),
            "negative_rating_max": float(negative_rating_max),
            "neutral_rating": float(neutral_rating),
            "history_policy": str(history_policy),
        },
    }


def build_item_catalog(
    df_merged: pd.DataFrame,
    total_item_count: int,
    item_popularity: np.ndarray,
    multimodal_embedding: np.ndarray,
) -> dict:
    item_features = _build_item_feature_arrays(df_merged, total_item_count=total_item_count, item_popularity=item_popularity)
    item_features["multimodal_embedding"] = np.asarray(multimodal_embedding, dtype=np.float32)
    return {
        "protocol": "item_catalog_v4",
        "all_item_ids": np.arange(1, total_item_count + 1, dtype=np.int32),
        "item_features": item_features,
    }


def run_ranking_preprocessing():
    settings = yaml.safe_load((CONFIG_DIR / "preprocess.yaml").read_text(encoding="utf-8")) or {}
    ranking_settings = settings.get("ranking", {})
    multimodal_settings = resolve_multimodal_settings(settings.get("multimodal", {}))
    max_seq_len = int(ranking_settings.get("max_seq_len", settings.get("retrieval", {}).get("max_seq_len", 10)))
    negative_rating_max = float(ranking_settings.get("negative_rating_max", settings.get("retrieval", {}).get("negative_rating_max", 2.0)))
    positive_rating_min = float(ranking_settings.get("positive_rating_min", 4.0))
    neutral_rating = float(ranking_settings.get("neutral_rating", settings.get("retrieval", {}).get("neutral_rating", 3.0)))
    history_policy = str(ranking_settings.get("history_policy", settings.get("retrieval", {}).get("history_policy", "all_ratings")))

    logger.info(
        "Ranking preprocessing start | protocol=%s | max_seq_len=%s | positive_rating_min=%s | negative_rating_max=%s | neutral_rating=%s | history_policy=%s",
        _SAMPLE_PROTOCOL,
        max_seq_len,
        positive_rating_min,
        negative_rating_max,
        neutral_rating,
        history_policy,
    )
    df_movies, df_ratings, df_users = load_raw_data()
    logger.info("Raw data loaded | movies=%s | ratings=%s | users=%s", len(df_movies), len(df_ratings), len(df_users))
    df_merged, user_vocab, movie_vocab = process_features_for_ranking(df_movies, df_ratings, df_users)
    logger.info("Ranking feature processing done | merged_rows=%s", len(df_merged))

    total_item_count = len(movie_vocab["movie_id"])
    multimodal_artifacts = build_or_load_multimodal_embeddings(df_movies, movie_vocab, multimodal_settings)
    samples = build_prefix_train_eval_samples(
        df_merged,
        total_item_count=total_item_count,
        max_seq_len=max_seq_len,
        negative_rating_max=negative_rating_max,
        positive_rating_min=positive_rating_min,
        neutral_rating=neutral_rating,
        history_policy=history_policy,
    )
    item_catalog = build_item_catalog(
        df_merged,
        total_item_count=total_item_count,
        item_popularity=samples["item_popularity"],
        multimodal_embedding=multimodal_artifacts.embeddings,
    )

    vocab_dict = {**user_vocab, **movie_vocab}
    vocab_dict["popularity"] = np.arange(_POPULARITY_BUCKET_COUNT, dtype=np.int32)
    vocab_dict["recency_bucket"] = np.arange(1, _RECENCY_BUCKET_COUNT + 1, dtype=np.int32)
    feature_dict = {key: len(values) + 1 for key, values in vocab_dict.items()}
    feature_dict["multimodal_embedding_dim"] = int(multimodal_artifacts.embeddings.shape[1])

    save_pickle(samples, RANKING_SAMPLE_PATH)
    save_pickle(item_catalog, ITEM_CATALOG_PATH)
    save_pickle(feature_dict, RANKING_FEATURE_DICT_PATH)
    save_pickle(vocab_dict, RANKING_VOCAB_DICT_PATH)
    logger.info(
        "Ranking preprocessing done | protocol=%s | train_samples=%s | validation_samples=%s | test_samples=%s | global_item_catalog=%s",
        _SAMPLE_PROTOCOL,
        len(samples["train"]["user_id"]),
        len(samples["validation"]["user_id"]),
        len(samples["test"]["user_id"]),
        ITEM_CATALOG_PATH.name,
    )
    return samples, feature_dict
