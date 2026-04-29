from __future__ import annotations

from collections import defaultdict

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
_PREPROCESS_VERSION = 14
_SEQUENCE_TRAIN_SCHEMA = "prefix_expanded_v1"
_RECENCY_BUCKET_BOUNDARIES_DAYS = (1, 3, 7, 14, 30, 90, 365)
_RECENCY_BUCKET_BOUNDARIES_SECONDS = np.asarray(_RECENCY_BUCKET_BOUNDARIES_DAYS, dtype=np.int64) * 24 * 60 * 60
_RECENCY_BUCKET_COUNT = len(_RECENCY_BUCKET_BOUNDARIES_DAYS) + 2
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



def _pad_sequence(values, max_seq_len: int, padding_value: int = 0):
    arr = np.asarray(values, dtype=np.int32).reshape(-1)[-max_seq_len:]
    padded = np.full(max_seq_len, padding_value, dtype=np.int32)
    if arr.size > 0:
        padded[-arr.size:] = arr
    return padded




def _recency_buckets(history_timestamps, target_timestamp, max_seq_len: int, padding_value: int = 0) -> np.ndarray:
    history = np.asarray(history_timestamps, dtype=np.int64).reshape(-1)
    if history.size == 0:
        return np.full(max_seq_len, padding_value, dtype=np.int32)
    clipped_history = history[-max_seq_len:]
    deltas = np.maximum(np.int64(target_timestamp) - clipped_history, 0)
    bucket_ids = np.searchsorted(_RECENCY_BUCKET_BOUNDARIES_SECONDS, deltas, side="left") + 1
    padded = np.full(max_seq_len, padding_value, dtype=np.int32)
    if bucket_ids.size > 0:
        padded[-bucket_ids.size :] = bucket_ids.astype(np.int32, copy=False)
    return padded



def _feedback_tokens(rating_sequence, max_seq_len: int, positive_rating_min: float, negative_rating_max: float) -> np.ndarray:
    ratings = np.asarray(rating_sequence, dtype=np.float32).reshape(-1)
    padded = np.zeros(max_seq_len, dtype=np.int32)
    clipped = ratings[-max_seq_len:]
    if clipped.size == 0:
        return padded
    feedback = np.full(clipped.shape[0], 2, dtype=np.int32)
    feedback[clipped >= float(positive_rating_min)] = 3
    feedback[clipped <= float(negative_rating_max)] = 1
    padded[-clipped.size :] = feedback
    return padded



def _user_negative_pool(movie_sequence, rating_sequence, max_pool_size: int, negative_rating_max: float) -> np.ndarray:
    movies = np.asarray(movie_sequence, dtype=np.int32).reshape(-1)
    ratings = np.asarray(rating_sequence, dtype=np.float32).reshape(-1)
    negatives = movies[ratings <= float(negative_rating_max)] if movies.size > 0 else np.zeros(0, dtype=np.int32)
    if negatives.size == 0:
        return np.zeros(max_pool_size, dtype=np.int32)
    unique_negatives = np.unique(negatives.astype(np.int32, copy=False))[-max_pool_size:]
    padded = np.zeros(max_pool_size, dtype=np.int32)
    padded[-unique_negatives.size :] = unique_negatives
    return padded



def _count_sequence_prefix_samples(grouped_users, positive_rating_min: float) -> int:
    sample_count = 0
    threshold = float(positive_rating_min)
    for _, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        if history_length < 3:
            continue
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        positive_targets = rating_sequence[1:-1] >= threshold
        sample_count += int(np.count_nonzero(positive_targets))
    return sample_count



def _build_sequence_train_split(grouped_users, user_columns, max_hist_seq_len: int, padding_value: int, positive_rating_min: float, negative_rating_max: float, negative_pool_size: int) -> dict[str, np.ndarray]:
    sample_count = _count_sequence_prefix_samples(grouped_users, positive_rating_min)
    sequence_train = {
        "user_id": np.zeros(sample_count, dtype=np.int32),
        "gender": np.zeros(sample_count, dtype=np.int32),
        "age": np.zeros(sample_count, dtype=np.int32),
        "occupation": np.zeros(sample_count, dtype=np.int32),
        "zip_code": np.zeros(sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_recency_bucket": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_rating": np.zeros((sample_count, max_hist_seq_len), dtype=np.float32),
        "hist_feedback": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "movie_id": np.zeros(sample_count, dtype=np.int32),
        "rating": np.zeros(sample_count, dtype=np.float32),
        "user_negative_movie_id": np.zeros((sample_count, negative_pool_size), dtype=np.int32),
    }
    row_idx = 0
    threshold = float(positive_rating_min)
    for user_id_value, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        if history_length < 3:
            continue
        user_features = {col: np.int32(grouped_feats[col].iloc[0]) for col in user_columns}
        movie_sequence = grouped_feats["movie_id"].to_numpy(dtype=np.int32, copy=False)
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        timestamp_sequence = grouped_feats["timestamp"].to_numpy(dtype=np.int64, copy=False)

        for target_idx in range(1, history_length - 1):
            target_rating = float(rating_sequence[target_idx])
            if target_rating < threshold:
                continue
            prefix_movie_sequence = movie_sequence[:target_idx]
            prefix_rating_sequence = rating_sequence[:target_idx]
            prefix_timestamp_sequence = timestamp_sequence[:target_idx]

            sequence_train["user_id"][row_idx] = np.int32(user_id_value)
            for col in user_columns:
                sequence_train[col][row_idx] = user_features[col]
            sequence_train["hist_movie_id"][row_idx] = _pad_sequence(prefix_movie_sequence, max_hist_seq_len, padding_value)
            sequence_train["hist_recency_bucket"][row_idx] = _recency_buckets(
                prefix_timestamp_sequence,
                timestamp_sequence[target_idx],
                max_hist_seq_len,
                padding_value,
            )
            sequence_train["hist_rating"][row_idx] = _pad_sequence(prefix_rating_sequence, max_hist_seq_len, 0.0)
            sequence_train["hist_feedback"][row_idx] = _feedback_tokens(
                prefix_rating_sequence,
                max_hist_seq_len,
                positive_rating_min,
                negative_rating_max,
            )
            sequence_train["movie_id"][row_idx] = np.int32(movie_sequence[target_idx])
            sequence_train["rating"][row_idx] = np.float32(target_rating)
            sequence_train["user_negative_movie_id"][row_idx] = _user_negative_pool(
                prefix_movie_sequence,
                prefix_rating_sequence,
                negative_pool_size,
                negative_rating_max,
            )
            row_idx += 1
    return sequence_train



def generate_train_eval_samples(data_df, user_columns, item_columns, max_hist_seq_len=10, max_feat_seq_len=10, padding_value=0, positive_rating_min: float = 4.0, negative_rating_max: float = 2.0, sequence_negative_pool_size: int | None = None):
    del item_columns, max_feat_seq_len
    data_df = data_df.sort_values(["user_id", "timestamp"], kind="stable")
    grouped_users = list(data_df.groupby("user_id", sort=False))
    train_sample_count = 0
    test_sample_count = 0
    max_user_history = 0
    threshold = float(positive_rating_min)
    for _, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        max_user_history = max(max_user_history, history_length)
        if history_length < 2:
            continue
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        positive_target_indices = [idx for idx in range(1, history_length) if float(rating_sequence[idx]) >= threshold]
        if not positive_target_indices:
            continue
        train_sample_count += max(len(positive_target_indices) - 1, 0)
        test_sample_count += 1
    negative_pool_size = int(sequence_negative_pool_size or max_hist_seq_len)
    sequence_train_sample_count = _count_sequence_prefix_samples(grouped_users, positive_rating_min)

    logger.info(
        "Retrieval sample allocation | grouped_users=%s | train_samples=%s | test_samples=%s | sequence_train_samples=%s | max_hist_seq_len=%s | negative_pool_size=%s",
        len(grouped_users),
        train_sample_count,
        test_sample_count,
        sequence_train_sample_count,
        max_hist_seq_len,
        negative_pool_size,
    )

    train_data_dict = {
        "user_id": np.zeros(train_sample_count, dtype=np.int32),
        "gender": np.zeros(train_sample_count, dtype=np.int32),
        "age": np.zeros(train_sample_count, dtype=np.int32),
        "occupation": np.zeros(train_sample_count, dtype=np.int32),
        "zip_code": np.zeros(train_sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((train_sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_recency_bucket": np.zeros((train_sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_rating": np.zeros((train_sample_count, max_hist_seq_len), dtype=np.float32),
        "movie_id": np.zeros(train_sample_count, dtype=np.int32),
        "rating": np.zeros(train_sample_count, dtype=np.float32),
    }
    test_data_dict = {
        "user_id": np.zeros(test_sample_count, dtype=np.int32),
        "gender": np.zeros(test_sample_count, dtype=np.int32),
        "age": np.zeros(test_sample_count, dtype=np.int32),
        "occupation": np.zeros(test_sample_count, dtype=np.int32),
        "zip_code": np.zeros(test_sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((test_sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_recency_bucket": np.zeros((test_sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_rating": np.zeros((test_sample_count, max_hist_seq_len), dtype=np.float32),
        "hist_feedback": np.zeros((test_sample_count, max_hist_seq_len), dtype=np.int32),
        "movie_id": np.zeros(test_sample_count, dtype=np.int32),
        "rating": np.zeros(test_sample_count, dtype=np.float32),
    }

    train_row = 0
    test_row = 0
    for user_id_value, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        if history_length < 2:
            continue

        user_features = {col: np.int32(grouped_feats[col].iloc[0]) for col in user_columns}
        movie_sequence = grouped_feats["movie_id"].to_numpy(dtype=np.int32, copy=False)
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        timestamp_sequence = grouped_feats["timestamp"].to_numpy(dtype=np.int64, copy=False)
        positive_target_indices = [idx for idx in range(1, history_length) if float(rating_sequence[idx]) >= threshold]
        if not positive_target_indices:
            continue

        test_target_idx = positive_target_indices[-1]
        test_data_dict["user_id"][test_row] = np.int32(user_id_value)
        for col in user_columns:
            test_data_dict[col][test_row] = user_features[col]
        test_data_dict["hist_movie_id"][test_row] = _pad_sequence(movie_sequence[:test_target_idx], max_hist_seq_len, padding_value)
        test_data_dict["hist_recency_bucket"][test_row] = _recency_buckets(timestamp_sequence[:test_target_idx], timestamp_sequence[test_target_idx], max_hist_seq_len, padding_value)
        test_data_dict["hist_rating"][test_row] = _pad_sequence(rating_sequence[:test_target_idx], max_hist_seq_len, 0.0)
        test_data_dict["hist_feedback"][test_row] = _feedback_tokens(
            rating_sequence[:test_target_idx],
            max_hist_seq_len,
            positive_rating_min,
            negative_rating_max,
        )
        test_data_dict["movie_id"][test_row] = np.int32(movie_sequence[test_target_idx])
        test_data_dict["rating"][test_row] = np.float32(rating_sequence[test_target_idx])
        test_row += 1

        for i in positive_target_indices[:-1]:
            train_data_dict["user_id"][train_row] = np.int32(user_id_value)
            for col in user_columns:
                train_data_dict[col][train_row] = user_features[col]
            train_data_dict["hist_movie_id"][train_row] = _pad_sequence(movie_sequence[:i], max_hist_seq_len, padding_value)
            train_data_dict["hist_recency_bucket"][train_row] = _recency_buckets(timestamp_sequence[:i], timestamp_sequence[i], max_hist_seq_len, padding_value)
            train_data_dict["hist_rating"][train_row] = _pad_sequence(rating_sequence[:i], max_hist_seq_len, 0.0)
            train_data_dict["movie_id"][train_row] = np.int32(movie_sequence[i])
            train_data_dict["rating"][train_row] = np.float32(rating_sequence[i])
            train_row += 1

    sequence_train = _build_sequence_train_split(
        grouped_users,
        user_columns,
        max_hist_seq_len=max_hist_seq_len,
        padding_value=padding_value,
        positive_rating_min=positive_rating_min,
        negative_rating_max=negative_rating_max,
        negative_pool_size=negative_pool_size,
    )

    return {
        "train": train_data_dict,
        "test": test_data_dict,
        "sequence_train": sequence_train,
        "rating_semantics": {
            "positive_rating_min": float(positive_rating_min),
            "negative_rating_max": float(negative_rating_max),
            "sequence_negative_pool_size": int(negative_pool_size),
            "max_user_history": int(max_user_history),
        },
    }



def _multimodal_embedding_dim() -> int:
    meta = yaml.safe_load(OPENCLIP_PREPROCESS_META_PATH.read_text(encoding="utf-8")) or {}
    return int(meta["embedding_dim"])


def _build_preprocess_meta(max_seq_len: int, positive_rating_min: float, negative_rating_max: float, sequence_negative_pool_size: int, multimodal_embedding_dim: int) -> dict:
    return {
        "version": _PREPROCESS_VERSION,
        "sequence_train_schema": _SEQUENCE_TRAIN_SCHEMA,
        "retrieval_max_seq_len": int(max_seq_len),
        "positive_rating_min": float(positive_rating_min),
        "negative_rating_max": float(negative_rating_max),
        "sequence_negative_pool_size": int(sequence_negative_pool_size),
        "rating_scale": "five_point",
        "recency_bucket_boundaries_days": list(_RECENCY_BUCKET_BOUNDARIES_DAYS),
        "recency_bucket_count": int(_RECENCY_BUCKET_COUNT),
        "multimodal_embedding_dim": int(multimodal_embedding_dim),
    }



def _can_reuse_cache(max_seq_len: int, positive_rating_min: float, negative_rating_max: float, sequence_negative_pool_size: int, multimodal_embedding_dim: int) -> bool:
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
        negative_rating_max,
        sequence_negative_pool_size,
        multimodal_embedding_dim,
    )



def run_retrieval_preprocessing():
    settings = yaml.safe_load((CONFIG_DIR / "preprocess.yaml").read_text(encoding="utf-8")) or {}
    retrieval_settings = settings.get("retrieval", {})
    max_seq_len = int(retrieval_settings.get("max_seq_len", 10))
    positive_rating_min = float(retrieval_settings.get("positive_rating_min", 4.0))
    negative_rating_max = float(retrieval_settings.get("negative_rating_max", 2.0))
    sequence_negative_pool_size = int(retrieval_settings.get("sequence_negative_pool_size", max_seq_len))
    multimodal_embedding_dim = _multimodal_embedding_dim()
    if _can_reuse_cache(
        max_seq_len,
        positive_rating_min,
        negative_rating_max,
        sequence_negative_pool_size,
        multimodal_embedding_dim,
    ):
        samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
        feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
        logger.info(
            "Retrieval preprocessing cache hit | max_seq_len=%s | train_samples=%s | sequence_train_samples=%s | test_samples=%s",
            max_seq_len,
            len(samples["train"]["user_id"]),
            len(samples.get("sequence_train", {}).get("user_id", [])),
            len(samples["test"]["user_id"]),
        )
        return samples, feature_dict

    logger.info(
        "Retrieval preprocessing start | max_seq_len=%s | version=%s | positive_rating_min=%s | negative_rating_max=%s | sequence_negative_pool_size=%s",
        max_seq_len,
        _PREPROCESS_VERSION,
        positive_rating_min,
        negative_rating_max,
        sequence_negative_pool_size,
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
        negative_rating_max=negative_rating_max,
        sequence_negative_pool_size=sequence_negative_pool_size,
    )
    vocab_dict = {**user_vocab, **movie_vocab}
    feature_dict = {k: len(v) + 1 for k, v in vocab_dict.items()}
    feature_dict["hist_recency_bucket"] = _RECENCY_BUCKET_COUNT
    feature_dict["popularity"] = _POPULARITY_BUCKET_COUNT + 1
    feature_dict["multimodal_embedding_dim"] = multimodal_embedding_dim
    save_pickle(samples, RETRIEVAL_SAMPLE_PATH)
    save_pickle(vocab_dict, RETRIEVAL_VOCAB_DICT_PATH)
    save_pickle(feature_dict, RETRIEVAL_FEATURE_DICT_PATH)
    save_json(
        _build_preprocess_meta(
            max_seq_len,
            positive_rating_min,
            negative_rating_max,
            sequence_negative_pool_size,
            multimodal_embedding_dim,
        ),
        RETRIEVAL_PREPROCESS_META_PATH,
    )
    save_numpy(np.array(vocab_dict["movie_id"]), MOVIE_RAW_IDS_PATH)
    logger.info(
        "Retrieval preprocessing done | train_samples=%s | sequence_train_samples=%s | test_samples=%s",
        len(samples["train"]["user_id"]),
        len(samples["sequence_train"]["user_id"]),
        len(samples["test"]["user_id"]),
    )
    return samples, feature_dict
