from __future__ import annotations

import numpy as np


STATIC_USER_FIELDS = ["user_id", "gender", "age", "occupation", "zip_code"]
POINTWISE_ITEM_FIELDS = ["movie_id", "genres", "isAdult", "startYear", "popularity", "averageRating"]
POINTWISE_SEQUENCE_FIELDS = ["context_movie_id"]

CONTEXT_FIELDS = ["context_movie_id", "context_rating", "context_low_rating_mask", "context_length"]
CANDIDATE_FIELDS = ["candidate_movie_id", "candidate_genres", "candidate_isAdult", "candidate_startYear", "candidate_popularity", "candidate_averageRating"]
SPARSE_ITEM_FEATURE_FIELDS = ["genres", "isAdult", "startYear", "popularity", "averageRating"]
DENSE_ITEM_FEATURE_FIELDS = ["multimodal_embedding"]
ITEM_FEATURE_FIELDS = SPARSE_ITEM_FEATURE_FIELDS + DENSE_ITEM_FEATURE_FIELDS


def _as_int_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int32, copy=False)
    return np.asarray(value, dtype=np.int32)


def get_item_feature_arrays(source: dict) -> dict[str, np.ndarray]:
    if all(field in source for field in ITEM_FEATURE_FIELDS):
        result = {field: np.asarray(source[field], dtype=np.int32) for field in SPARSE_ITEM_FEATURE_FIELDS}
        result.update({field: np.asarray(source[field], dtype=np.float32) for field in DENSE_ITEM_FEATURE_FIELDS})
        return result

    item_features = source.get("item_features")
    if item_features is None:
        raise ValueError("Item catalog must contain item_features")
    missing_fields = [field for field in ITEM_FEATURE_FIELDS if field not in item_features]
    if missing_fields:
        raise ValueError(f"Item catalog is missing item feature fields: {missing_fields}")
    result = {field: np.asarray(item_features[field], dtype=np.int32) for field in SPARSE_ITEM_FEATURE_FIELDS}
    result.update({field: np.asarray(item_features[field], dtype=np.float32) for field in DENSE_ITEM_FEATURE_FIELDS})
    return result


def get_all_item_ids(source: dict) -> np.ndarray:
    if "all_item_ids" in source:
        return np.asarray(source["all_item_ids"], dtype=np.int32)

    item_features = get_item_feature_arrays(source)
    if item_features["genres"].shape[0] <= 1:
        return np.zeros(0, dtype=np.int32)
    return np.arange(1, item_features["genres"].shape[0], dtype=np.int32)


def extract_split_sample(split_data: dict, idx: int) -> dict:
    sample = {}
    for field in STATIC_USER_FIELDS + CONTEXT_FIELDS:
        if field not in split_data:
            continue
        value = split_data[field][idx]
        if field == "context_length":
            sample[field] = int(value)
        elif field in {"context_rating", "context_low_rating_mask"}:
            sample[field] = np.asarray(value, dtype=np.float32)
        else:
            sample[field] = _as_int_array(value)
    if "target_movie_id" in split_data:
        sample["target_movie_id"] = int(split_data["target_movie_id"][idx])
    if "target_rating" in split_data:
        sample["target_rating"] = float(split_data["target_rating"][idx])
    if "user_id_original" in split_data:
        sample["user_id_original"] = int(split_data["user_id_original"][idx])
    return sample


def get_seen_movie_ids(sample: dict) -> np.ndarray:
    context = np.asarray(sample.get("context_movie_id", []), dtype=np.int32).reshape(-1)
    return np.unique(context[context > 0]).astype(np.int32, copy=False)
