from __future__ import annotations

import numpy as np


STATIC_USER_FIELDS = ["user_id", "gender", "age", "occupation", "zip_code"]
POINTWISE_ITEM_FIELDS = ["movie_id", "genres", "isAdult", "startYear", "popularity"]
POINTWISE_SEQUENCE_FIELDS = ["context_movie_id"]

CONTEXT_FIELDS = ["context_movie_id", "context_rating", "context_low_rating_mask", "context_length"]
CANDIDATE_FIELDS = ["candidate_movie_id", "candidate_genres", "candidate_isAdult", "candidate_startYear", "candidate_popularity"]
ITEM_FEATURE_FIELDS = ["genres", "isAdult", "startYear", "popularity"]


def _as_int_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int32, copy=False)
    return np.asarray(value, dtype=np.int32)


def get_item_feature_arrays(source: dict) -> dict[str, np.ndarray]:
    if all(field in source for field in ITEM_FEATURE_FIELDS):
        return {field: np.asarray(source[field], dtype=np.int32) for field in ITEM_FEATURE_FIELDS}

    item_features = source.get("item_features")
    if item_features is not None:
        return {field: np.asarray(item_features[field], dtype=np.int32) for field in ITEM_FEATURE_FIELDS}

    item_lookup = {}
    for split_name in ("train", "test"):
        split = source.get(split_name, {})
        if "target_movie_id" not in split:
            continue
        for idx, movie_id in enumerate(np.asarray(split["target_movie_id"]).tolist()):
            encoded_movie_id = int(movie_id)
            if encoded_movie_id <= 0 or encoded_movie_id in item_lookup:
                continue
            item_lookup[encoded_movie_id] = {
                "genres": np.zeros(1, dtype=np.int32),
                "isAdult": 0,
                "startYear": 0,
                "popularity": 0,
            }

    max_movie_id = max(item_lookup) if item_lookup else 0
    arrays = {
        "genres": np.zeros((max_movie_id + 1, 1), dtype=np.int32),
        "isAdult": np.zeros(max_movie_id + 1, dtype=np.int32),
        "startYear": np.zeros(max_movie_id + 1, dtype=np.int32),
        "popularity": np.zeros(max_movie_id + 1, dtype=np.int32),
    }
    return arrays


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


def sample_negative_ids(
    all_item_ids: np.ndarray,
    seen_movie_ids: np.ndarray,
    positive_movie_id: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int32)

    seen = {int(movie_id) for movie_id in np.asarray(seen_movie_ids).reshape(-1).tolist() if int(movie_id) > 0}
    seen.add(int(positive_movie_id))
    available_count = int(all_item_ids.size - len(seen))
    if available_count <= 0:
        return np.zeros(count, dtype=np.int32)

    if available_count <= count:
        available = np.asarray([int(movie_id) for movie_id in all_item_ids.tolist() if int(movie_id) not in seen], dtype=np.int32)
        if available.size == 0:
            return np.zeros(count, dtype=np.int32)
        if available.size >= count:
            return available[:count]
        extra = rng.choice(available, size=count - available.size, replace=True).astype(np.int32)
        return np.concatenate([available, extra], axis=0)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    max_attempts = max(count * 20, 50)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        candidate = int(rng.choice(all_item_ids))
        attempts += 1
        if candidate in seen or candidate in chosen_set:
            continue
        chosen.append(candidate)
        chosen_set.add(candidate)

    if len(chosen) < count:
        remaining = [
            int(movie_id)
            for movie_id in all_item_ids.tolist()
            if int(movie_id) not in seen and int(movie_id) not in chosen_set
        ]
        take = min(count - len(chosen), len(remaining))
        chosen.extend(remaining[:take])

    if len(chosen) < count:
        base = np.asarray(chosen if chosen else [int(all_item_ids[0])], dtype=np.int32)
        extra = rng.choice(base, size=count - len(chosen), replace=True).astype(np.int32)
        chosen.extend(extra.tolist())

    return np.asarray(chosen, dtype=np.int32)


def sample_negative_ids_with_candidates(
    all_item_ids: np.ndarray,
    seen_movie_ids: np.ndarray,
    positive_movie_id: int,
    count: int,
    rng: np.random.Generator,
    candidate_pool: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int32)

    seen = {int(movie_id) for movie_id in np.asarray(seen_movie_ids).reshape(-1).tolist() if int(movie_id) > 0}
    seen.add(int(positive_movie_id))
    chosen: list[int] = []
    chosen_set: set[int] = set()

    if candidate_pool is not None:
        for movie_id in np.asarray(candidate_pool, dtype=np.int32).reshape(-1).tolist():
            candidate = int(movie_id)
            if candidate <= 0 or candidate in seen or candidate in chosen_set:
                continue
            chosen.append(candidate)
            chosen_set.add(candidate)
            if len(chosen) >= count:
                return np.asarray(chosen, dtype=np.int32)

    if len(chosen) < count:
        random_tail = sample_negative_ids(
            all_item_ids,
            np.asarray(list(seen | chosen_set), dtype=np.int32),
            positive_movie_id=0,
            count=count - len(chosen),
            rng=rng,
        )
        chosen.extend(int(movie_id) for movie_id in random_tail.tolist())

    return np.asarray(chosen[:count], dtype=np.int32)
