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
_PREPROCESS_VERSION = 17
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



def _positive_target_indices(rating_sequence: np.ndarray, positive_rating_min: float) -> list[int]:
    return [idx for idx in range(1, len(rating_sequence)) if float(rating_sequence[idx]) >= float(positive_rating_min)]



def _split_positive_targets(rating_sequence: np.ndarray, positive_rating_min: float) -> tuple[list[int], int | None, int | None]:
    target_indices = _positive_target_indices(rating_sequence, positive_rating_min)
    if len(target_indices) < 2:
        return [], None, None
    return target_indices[:-2], target_indices[-2], target_indices[-1]



def _empty_retrieval_samples(sample_count: int, max_hist_seq_len: int, negative_pool_size: int) -> dict[str, np.ndarray]:
    return {
        "user_id": np.zeros(sample_count, dtype=np.int32),
        "gender": np.zeros(sample_count, dtype=np.int32),
        "age": np.zeros(sample_count, dtype=np.int32),
        "occupation": np.zeros(sample_count, dtype=np.int32),
        "zip_code": np.zeros(sample_count, dtype=np.int32),
        "hist_movie_id": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_recency_bucket": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "hist_rating": np.zeros((sample_count, max_hist_seq_len), dtype=np.float32),
        "hist_feedback": np.zeros((sample_count, max_hist_seq_len), dtype=np.int32),
        "user_negative_movie_id": np.zeros((sample_count, negative_pool_size), dtype=np.int32),
        "movie_id": np.zeros(sample_count, dtype=np.int32),
        "rating": np.zeros(sample_count, dtype=np.float32),
    }



def generate_train_eval_samples(data_df, user_columns, item_columns, max_hist_seq_len=10, max_feat_seq_len=10, padding_value=0, positive_rating_min: float = 4.0, negative_rating_max: float = 2.0, sequence_negative_pool_size: int | None = None):
    del item_columns, max_feat_seq_len
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
        target_indices, validation_target_idx, test_target_idx = _split_positive_targets(rating_sequence, positive_rating_min)
        if validation_target_idx is None or test_target_idx is None:
            continue
        train_sample_count += len(target_indices)
        validation_sample_count += 1
        test_sample_count += 1
    negative_pool_size = int(sequence_negative_pool_size or max_hist_seq_len)

    logger.info(
        "Retrieval sample allocation | users=%s | train=%s | validation=%s | test=%s | max_hist_seq_len=%s | negative_pool_size=%s",
        len(grouped_users),
        train_sample_count,
        validation_sample_count,
        test_sample_count,
        max_hist_seq_len,
        negative_pool_size,
    )

    train_data_dict = _empty_retrieval_samples(train_sample_count, max_hist_seq_len, negative_pool_size)
    validation_data_dict = _empty_retrieval_samples(validation_sample_count, max_hist_seq_len, negative_pool_size)
    test_data_dict = _empty_retrieval_samples(test_sample_count, max_hist_seq_len, negative_pool_size)

    train_row = 0
    validation_row = 0
    test_row = 0
    for user_id_value, grouped_feats in grouped_users:
        history_length = len(grouped_feats)
        if history_length < 2:
            continue

        user_features = {col: np.int32(grouped_feats[col].iloc[0]) for col in user_columns}
        movie_sequence = grouped_feats["movie_id"].to_numpy(dtype=np.int32, copy=False)
        rating_sequence = grouped_feats["rating"].to_numpy(dtype=np.float32, copy=False)
        timestamp_sequence = grouped_feats["timestamp"].to_numpy(dtype=np.int64, copy=False)
        train_target_indices, validation_target_idx, test_target_idx = _split_positive_targets(rating_sequence, positive_rating_min)
        if validation_target_idx is None or test_target_idx is None:
            continue

        validation_data_dict["user_id"][validation_row] = np.int32(user_id_value)
        for col in user_columns:
            validation_data_dict[col][validation_row] = user_features[col]
        validation_data_dict["hist_movie_id"][validation_row] = _pad_sequence(movie_sequence[:validation_target_idx], max_hist_seq_len, padding_value)
        validation_data_dict["hist_recency_bucket"][validation_row] = _recency_buckets(timestamp_sequence[:validation_target_idx], timestamp_sequence[validation_target_idx], max_hist_seq_len, padding_value)
        validation_data_dict["hist_rating"][validation_row] = _pad_sequence(rating_sequence[:validation_target_idx], max_hist_seq_len, 0.0)
        validation_data_dict["hist_feedback"][validation_row] = _feedback_tokens(
            rating_sequence[:validation_target_idx],
            max_hist_seq_len,
            positive_rating_min,
            negative_rating_max,
        )
        validation_data_dict["user_negative_movie_id"][validation_row] = _user_negative_pool(
            movie_sequence[:validation_target_idx],
            rating_sequence[:validation_target_idx],
            negative_pool_size,
            negative_rating_max,
        )
        validation_data_dict["movie_id"][validation_row] = np.int32(movie_sequence[validation_target_idx])
        validation_data_dict["rating"][validation_row] = np.float32(rating_sequence[validation_target_idx])
        validation_row += 1

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
        test_data_dict["user_negative_movie_id"][test_row] = _user_negative_pool(
            movie_sequence[:test_target_idx],
            rating_sequence[:test_target_idx],
            negative_pool_size,
            negative_rating_max,
        )
        test_data_dict["movie_id"][test_row] = np.int32(movie_sequence[test_target_idx])
        test_data_dict["rating"][test_row] = np.float32(rating_sequence[test_target_idx])
        test_row += 1

        for i in train_target_indices:
            train_data_dict["user_id"][train_row] = np.int32(user_id_value)
            for col in user_columns:
                train_data_dict[col][train_row] = user_features[col]
            train_data_dict["hist_movie_id"][train_row] = _pad_sequence(movie_sequence[:i], max_hist_seq_len, padding_value)
            train_data_dict["hist_recency_bucket"][train_row] = _recency_buckets(timestamp_sequence[:i], timestamp_sequence[i], max_hist_seq_len, padding_value)
            train_data_dict["hist_rating"][train_row] = _pad_sequence(rating_sequence[:i], max_hist_seq_len, 0.0)
            train_data_dict["hist_feedback"][train_row] = _feedback_tokens(
                rating_sequence[:i],
                max_hist_seq_len,
                positive_rating_min,
                negative_rating_max,
            )
            train_data_dict["user_negative_movie_id"][train_row] = _user_negative_pool(
                movie_sequence[:i],
                rating_sequence[:i],
                negative_pool_size,
                negative_rating_max,
            )
            train_data_dict["movie_id"][train_row] = np.int32(movie_sequence[i])
            train_data_dict["rating"][train_row] = np.float32(rating_sequence[i])
            train_row += 1

    return {
        "train": train_data_dict,
        "validation": validation_data_dict,
        "test": test_data_dict,
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
            "Retrieval preprocessing cache hit | max_seq_len=%s | train=%s | validation=%s | test=%s",
            max_seq_len,
            len(samples["train"]["user_id"]),
            len(samples["validation"]["user_id"]),
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
        "Retrieval preprocessing done | train=%s | validation=%s | test=%s",
        len(samples["train"]["user_id"]),
        len(samples["validation"]["user_id"]),
        len(samples["test"]["user_id"]),
    )
    return samples, feature_dict
