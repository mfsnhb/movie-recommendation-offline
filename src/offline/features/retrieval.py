from __future__ import annotations

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

from offline.data.loaders import load_raw_data
from offline.utils.io import (
    CONFIG_DIR,
    MOVIE_RAW_IDS_PATH,
    OPENCLIP_PREPROCESS_META_PATH,
    RETRIEVAL_FEATURE_DICT_PATH,
    RETRIEVAL_PREPROCESS_META_PATH,
    RETRIEVAL_SAMPLE_PATH,
    RETRIEVAL_VOCAB_DICT_PATH,
    load_pickle,
    save_json,
    save_numpy,
    save_pickle,
)
from offline.utils.logging import get_logger


logger = get_logger("offline.features.retrieval")
_PREPROCESS_VERSION = 21
_TIME_GAP_BUCKET_BOUNDARIES_DAYS = (1, 3, 7, 14, 30, 90, 365)
_TIME_GAP_BUCKET_BOUNDARIES_SECONDS = np.asarray(_TIME_GAP_BUCKET_BOUNDARIES_DAYS, dtype=np.int64) * 24 * 60 * 60
_TIME_GAP_BUCKET_COUNT = len(_TIME_GAP_BUCKET_BOUNDARIES_DAYS) + 2
_POPULARITY_BUCKET_COUNT = 10
_AVERAGE_RATING_BUCKET_COUNT = 5


def process_features(df_movies, df_ratings, df_users):
    user_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    movie_columns = ["movie_id", "genres", "isAdult", "startYear", "averageRating"]
    ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]

    df_users = df_users[user_columns]
    df_movies = df_movies[movie_columns].copy()
    df_movies["genres"] = df_movies["genres"].str.split("|")
    df_movies["isAdult"] = df_movies["isAdult"].fillna(False).astype(str)
    df_movies["startYear"] = df_movies["startYear"].replace("\\N", 0).fillna(0).astype(str)
    average_rating_raw = df_movies["averageRating"].astype(float).fillna(0).clip(0, 10)
    df_movies["averageRating"] = np.ceil(average_rating_raw / 2.0).astype(np.int32)
    df_movies["averageRating"] = df_movies["averageRating"].astype(str)
    df_ratings = df_ratings[ratings_columns].copy()
    df_ratings["rating"] = df_ratings["rating"].astype(np.float32)

    new_user_feature_df = df_users.copy()
    user_vocab = {}
    for feat_name in user_columns:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(new_user_feature_df[feat_name]) + 1
        if feat_name == "user_id":
            new_user_feature_df[feat_name + "_encode"] = encoded.astype(np.int32)
        else:
            new_user_feature_df[feat_name] = encoded.astype(np.int32)
        user_vocab[feat_name] = label_encoder.classes_

    new_movie_feature_df = df_movies.copy()
    movie_vocab = {}
    for feat_name in movie_columns:
        label_encoder = LabelEncoder()
        if feat_name == "genres":
            label_encoder.fit(new_movie_feature_df[feat_name].explode())
            new_movie_feature_df[feat_name] = new_movie_feature_df[feat_name].apply(
                lambda values: (label_encoder.transform(values) + 1).astype(np.int32)
            )
            movie_vocab[feat_name] = label_encoder.classes_
        elif feat_name == "averageRating":
            new_movie_feature_df[feat_name + "_encode"] = new_movie_feature_df[feat_name].astype(np.int32)
            movie_vocab[feat_name] = np.arange(1, _AVERAGE_RATING_BUCKET_COUNT + 1, dtype=np.int32)
        else:
            encoded = label_encoder.fit_transform(new_movie_feature_df[feat_name]) + 1
            new_movie_feature_df[feat_name + "_encode"] = encoded.astype(np.int32)
            movie_vocab[feat_name] = label_encoder.classes_

    df_merged = df_ratings.merge(new_user_feature_df, on="user_id", how="left")
    df_merged = df_merged.merge(new_movie_feature_df, on="movie_id", how="left")
    df_merged["user_id"] = df_merged["user_id_encode"].astype(np.int32)
    df_merged["movie_id"] = df_merged["movie_id_encode"].astype(np.int32)
    df_merged["gender"] = df_merged["gender"].astype(np.int32)
    df_merged["age"] = df_merged["age"].astype(np.int32)
    df_merged["occupation"] = df_merged["occupation"].astype(np.int32)
    df_merged["zip_code"] = df_merged["zip_code"].astype(np.int32)
    df_merged["timestamp"] = df_merged["timestamp"].astype(np.int64)
    df_merged["rating"] = df_merged["rating"].astype(np.float32)
    return df_merged[["user_id", "gender", "age", "occupation", "zip_code", "movie_id", "genres", "rating", "timestamp"]], user_vocab, movie_vocab


def _pad_int_sequence(values, max_seq_len: int, padding_value: int = 0):
    arr = np.asarray(values, dtype=np.int32).reshape(-1)[-max_seq_len:]
    padded = np.full(max_seq_len, padding_value, dtype=np.int32)
    if arr.size > 0:
        padded[-arr.size:] = arr
    return padded


def _pad_float_sequence(values, max_seq_len: int, padding_value: float = 0.0):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)[-max_seq_len:]
    padded = np.full(max_seq_len, padding_value, dtype=np.float32)
    if arr.size > 0:
        padded[-arr.size:] = arr
    return padded


def _time_gap_buckets(history_timestamps, target_timestamp, max_seq_len: int, padding_value: int = 0) -> np.ndarray:
    history = np.asarray(history_timestamps, dtype=np.int64).reshape(-1)
    if history.size == 0:
        return np.full(max_seq_len, padding_value, dtype=np.int32)
    clipped_history = history[-max_seq_len:]
    deltas = np.maximum(np.int64(target_timestamp) - clipped_history, 0)
    bucket_ids = np.searchsorted(_TIME_GAP_BUCKET_BOUNDARIES_SECONDS, deltas, side="left") + 1
    padded = np.full(max_seq_len, padding_value, dtype=np.int32)
    if bucket_ids.size > 0:
        padded[-bucket_ids.size :] = bucket_ids.astype(np.int32, copy=False)
    return padded


def _positive_target_indices(rating_sequence: np.ndarray, positive_rating_min: float) -> list[int]:
    return [idx for idx in range(1, len(rating_sequence)) if float(rating_sequence[idx]) >= float(positive_rating_min)]


def _split_eval_targets(rating_sequence: np.ndarray, eval_positive_rating_min: float) -> tuple[int | None, int | None]:
    eval_target_indices = _positive_target_indices(rating_sequence, eval_positive_rating_min)
    if len(eval_target_indices) < 2:
        return None, None
    return eval_target_indices[-2], eval_target_indices[-1]


def _empty_retrieval_samples(sample_count: int, max_hist_seq_len: int) -> dict[str, np.ndarray]:
    return {
        "user_id": np.zeros(sample_count, dtype=np.int32),
        "gender": np.zeros(sample_count, dtype=np.int32),
        "age": np.zeros(sample_count, dtype=np.int32),
        "occupation": np.zeros(sample_count, dtype=np.int32),
        "zip_code": np.zeros(sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_time_gap_bucket": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_rating": np.zeros((sample_count, max_hist_seq_len), dtype=np.float32),
        "movie_id": np.zeros(sample_count, dtype=np.int32),
        "rating": np.zeros(sample_count, dtype=np.float32),
    }


def _fill_user_fields(store: dict[str, np.ndarray], row_idx: int, user_id_value: int, user_features: dict[str, np.int32], user_columns: list[str]) -> None:
    store["user_id"][row_idx] = np.int32(user_id_value)
    for col in user_columns:
        store[col][row_idx] = user_features[col]


def _fill_history(
    store: dict[str, np.ndarray],
    row_idx: int,
    movie_sequence: np.ndarray,
    rating_sequence: np.ndarray,
    timestamp_sequence: np.ndarray,
    target_idx: int,
    max_hist_seq_len: int,
    padding_value: int,
) -> None:
    store["hist_movie_id"][row_idx] = _pad_int_sequence(movie_sequence[:target_idx], max_hist_seq_len, padding_value)
    store["hist_time_gap_bucket"][row_idx] = _time_gap_buckets(timestamp_sequence[:target_idx], timestamp_sequence[target_idx], max_hist_seq_len, padding_value)
    store["hist_rating"][row_idx] = _pad_float_sequence(rating_sequence[:target_idx], max_hist_seq_len, 0.0)


def generate_train_eval_samples(
    data_df,
    user_columns,
    item_columns,
    max_hist_seq_len=10,
    max_feat_seq_len=10,
    padding_value=0,
    positive_rating_min: float = 3.0,
    eval_positive_rating_min: float | None = None,
):
    del item_columns, max_feat_seq_len
    eval_positive_rating_min = float(positive_rating_min if eval_positive_rating_min is None else eval_positive_rating_min)
    data_df = data_df.sort_values(["user_id", "timestamp"], kind="stable")
    grouped_users = list(data_df.groupby("user_id", sort=False))
    train_sample_count = 0
    validation_sample_count = 0
    test_sample_count = 0
    max_user_history = 0
    for _, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        max_user_history = max(max_user_history, history_length)
        if history_length < 2:
            continue
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        validation_target_idx, test_target_idx = _split_eval_targets(rating_sequence, eval_positive_rating_min)
        if validation_target_idx is None or test_target_idx is None:
            continue
        train_target_indices = [
            idx for idx in range(1, validation_target_idx) if float(rating_sequence[idx]) >= float(positive_rating_min)
        ]
        train_sample_count += len(train_target_indices)
        validation_sample_count += 1
        test_sample_count += 1

    logger.info(
        "Retrieval sample allocation | users=%s | train=%s | validation=%s | test=%s | max_hist_seq_len=%s",
        len(grouped_users),
        train_sample_count,
        validation_sample_count,
        test_sample_count,
        max_hist_seq_len,
    )

    train_data_dict = _empty_retrieval_samples(train_sample_count, max_hist_seq_len)
    validation_data_dict = _empty_retrieval_samples(validation_sample_count, max_hist_seq_len)
    test_data_dict = _empty_retrieval_samples(test_sample_count, max_hist_seq_len)

    train_row = 0
    validation_row = 0
    test_row = 0
    for user_id_value, grouped_feats in grouped_users:
        if len(grouped_feats) < 2:
            continue

        user_features = {col: np.int32(grouped_feats[col].iloc[0]) for col in user_columns}
        movie_sequence = grouped_feats["movie_id"].to_numpy(dtype=np.int32, copy=False)
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        timestamp_sequence = grouped_feats["timestamp"].to_numpy(dtype=np.int64, copy=False)
        validation_target_idx, test_target_idx = _split_eval_targets(rating_sequence, eval_positive_rating_min)
        if validation_target_idx is None or test_target_idx is None:
            continue

        for train_target_idx in range(1, validation_target_idx):
            if float(rating_sequence[train_target_idx]) < float(positive_rating_min):
                continue
            _fill_user_fields(train_data_dict, train_row, user_id_value, user_features, user_columns)
            _fill_history(train_data_dict, train_row, movie_sequence, rating_sequence, timestamp_sequence, train_target_idx, max_hist_seq_len, padding_value)
            train_data_dict["movie_id"][train_row] = np.int32(movie_sequence[train_target_idx])
            train_data_dict["rating"][train_row] = np.float32(rating_sequence[train_target_idx])
            train_row += 1

        _fill_user_fields(validation_data_dict, validation_row, user_id_value, user_features, user_columns)
        _fill_history(validation_data_dict, validation_row, movie_sequence, rating_sequence, timestamp_sequence, validation_target_idx, max_hist_seq_len, padding_value)
        validation_data_dict["movie_id"][validation_row] = np.int32(movie_sequence[validation_target_idx])
        validation_data_dict["rating"][validation_row] = np.float32(rating_sequence[validation_target_idx])
        validation_row += 1

        _fill_user_fields(test_data_dict, test_row, user_id_value, user_features, user_columns)
        _fill_history(test_data_dict, test_row, movie_sequence, rating_sequence, timestamp_sequence, test_target_idx, max_hist_seq_len, padding_value)
        test_data_dict["movie_id"][test_row] = np.int32(movie_sequence[test_target_idx])
        test_data_dict["rating"][test_row] = np.float32(rating_sequence[test_target_idx])
        test_row += 1

    return {
        "train": train_data_dict,
        "validation": validation_data_dict,
        "test": test_data_dict,
        "rating_semantics": {
            "positive_rating_min": float(positive_rating_min),
            "eval_positive_rating_min": float(eval_positive_rating_min),
            "neutral_rating": 3.0,
            "history_policy": "all_interactions",
            "max_user_history": int(max_user_history),
        },
    }


def _multimodal_embedding_dim() -> int:
    meta = yaml.safe_load(OPENCLIP_PREPROCESS_META_PATH.read_text(encoding="utf-8")) or {}
    return int(meta["embedding_dim"])


def _build_preprocess_meta(
    max_seq_len: int,
    positive_rating_min: float,
    eval_positive_rating_min: float,
    multimodal_embedding_dim: int,
) -> dict:
    return {
        "version": _PREPROCESS_VERSION,
        "retrieval_max_seq_len": int(max_seq_len),
        "positive_rating_min": float(positive_rating_min),
        "eval_positive_rating_min": float(eval_positive_rating_min),
        "neutral_rating": 3.0,
        "history_policy": "all_interactions",
        "train_target_policy": f"rating>={positive_rating_min:g}",
        "eval_target_policy": f"rating>={eval_positive_rating_min:g}",
        "rating_scale": "five_point",
        "time_gap_bucket_boundaries_days": list(_TIME_GAP_BUCKET_BOUNDARIES_DAYS),
        "time_gap_bucket_count": int(_TIME_GAP_BUCKET_COUNT),
        "multimodal_embedding_dim": int(multimodal_embedding_dim),
    }


def _can_reuse_cache(
    max_seq_len: int,
    positive_rating_min: float,
    eval_positive_rating_min: float,
    multimodal_embedding_dim: int,
) -> bool:
    if not RETRIEVAL_PREPROCESS_META_PATH.exists():
        return False
    if not RETRIEVAL_SAMPLE_PATH.exists() or not RETRIEVAL_FEATURE_DICT_PATH.exists() or not RETRIEVAL_VOCAB_DICT_PATH.exists():
        return False
    try:
        meta = yaml.safe_load(RETRIEVAL_PREPROCESS_META_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return meta == _build_preprocess_meta(
        max_seq_len,
        positive_rating_min,
        eval_positive_rating_min,
        multimodal_embedding_dim,
    )


def run_retrieval_preprocessing():
    settings = yaml.safe_load((CONFIG_DIR / "preprocess.yaml").read_text(encoding="utf-8")) or {}
    retrieval_settings = settings.get("retrieval", {})
    max_seq_len = int(retrieval_settings.get("max_seq_len", 10))
    positive_rating_min = float(retrieval_settings.get("positive_rating_min", 3.0))
    eval_positive_rating_min = float(
        retrieval_settings.get("eval_positive_rating_min", settings.get("ranking", {}).get("positive_rating_min", 4.0))
    )
    multimodal_embedding_dim = _multimodal_embedding_dim()
    if _can_reuse_cache(max_seq_len, positive_rating_min, eval_positive_rating_min, multimodal_embedding_dim):
        samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
        feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
        logger.info(
            "Retrieval preprocessing cache hit | max_seq_len=%s | train=%s | validation=%s | test=%s",
            max_seq_len,
            len(samples["train"]["user_id"]),
            len(samples["validation"]["user_id"]),
            len(samples["test"]["user_id"]),
        )
        return samples, feature_dict

    logger.info(
        "Retrieval preprocessing start | max_seq_len=%s | version=%s | positive_rating_min=%s | eval_positive_rating_min=%s | history_policy=all_interactions",
        max_seq_len,
        _PREPROCESS_VERSION,
        positive_rating_min,
        eval_positive_rating_min,
    )
    df_movies, df_ratings, df_users = load_raw_data()
    logger.info("Raw data loaded | movies=%s | ratings=%s | users=%s", len(df_movies), len(df_ratings), len(df_users))
    df_merged, user_vocab, movie_vocab = process_features(df_movies, df_ratings, df_users)
    logger.info("Feature processing done | merged_rows=%s", len(df_merged))
    user_columns = ["gender", "age", "occupation", "zip_code"]
    item_columns = ["movie_id", "genres"]
    samples = generate_train_eval_samples(
        df_merged,
        user_columns,
        item_columns,
        max_hist_seq_len=max_seq_len,
        max_feat_seq_len=max_seq_len,
        positive_rating_min=positive_rating_min,
        eval_positive_rating_min=eval_positive_rating_min,
    )
    vocab_dict = {**user_vocab, **movie_vocab}
    feature_dict = {k: len(v) + 1 for k, v in vocab_dict.items()}
    feature_dict["hist_time_gap_bucket"] = _TIME_GAP_BUCKET_COUNT
    feature_dict["popularity"] = _POPULARITY_BUCKET_COUNT + 1
    feature_dict["multimodal_embedding_dim"] = multimodal_embedding_dim
    save_pickle(samples, RETRIEVAL_SAMPLE_PATH)
    save_pickle(vocab_dict, RETRIEVAL_VOCAB_DICT_PATH)
    save_pickle(feature_dict, RETRIEVAL_FEATURE_DICT_PATH)
    save_json(
        _build_preprocess_meta(max_seq_len, positive_rating_min, eval_positive_rating_min, multimodal_embedding_dim),
        RETRIEVAL_PREPROCESS_META_PATH,
    )
    save_numpy(np.array(vocab_dict["movie_id"]), MOVIE_RAW_IDS_PATH)
    logger.info(
        "Retrieval preprocessing done | train=%s | validation=%s | test=%s",
        len(samples["train"]["user_id"]),
        len(samples["validation"]["user_id"]),
        len(samples["test"]["user_id"]),
    )
    return samples, feature_dict
