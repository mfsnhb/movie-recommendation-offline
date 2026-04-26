from __future__ import annotations

import numpy as np

from offline.ranking.protocol import extract_split_sample, get_seen_movie_ids, sample_negative_ids_with_candidates


def build_xgboost_training_frame(
    split_data: dict,
    item_features: dict[str, np.ndarray],
    all_item_ids: np.ndarray,
    num_negatives: int,
    item_popularity: np.ndarray | None = None,
    seed: int = 42,
    fused_candidates_by_user: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    sample_weights: list[float] = []
    rng = np.random.default_rng(seed)
    for idx in range(int(len(split_data["user_id"]))):
        sample = extract_split_sample(split_data, idx)
        positive_movie_id = int(sample["target_movie_id"])
        candidate_pool = None
        if fused_candidates_by_user is not None:
            user_id_original = int(sample.get("user_id_original", sample["user_id"]))
            candidate_pool = fused_candidates_by_user.get(user_id_original)
        negatives = sample_negative_ids_with_candidates(
            all_item_ids,
            get_seen_movie_ids(sample),
            positive_movie_id,
            num_negatives,
            rng,
            candidate_pool=candidate_pool,
        )
        candidate_ids = np.concatenate([[positive_movie_id], negatives], axis=0)
        prefix_movies = np.asarray(sample["context_movie_id"], dtype=np.int32)
        prefix_movies = prefix_movies[prefix_movies > 0]
        prefix_genres = np.asarray(sample["context_genres"], dtype=np.int32)
        prefix_genres = prefix_genres[prefix_genres > 0]
        prefix_ratings = np.asarray(sample.get("context_rating", []), dtype=np.float32)
        prefix_ratings = prefix_ratings[prefix_ratings > 0]
        low_rating_mask = np.asarray(sample.get("context_low_rating_mask", []), dtype=np.float32)
        if low_rating_mask.size > 0:
            visible_mask = low_rating_mask <= 0
            prefix_movies = prefix_movies[visible_mask[-prefix_movies.size :]] if prefix_movies.size > 0 else prefix_movies
            prefix_genres = prefix_genres[visible_mask[-prefix_genres.size :]] if prefix_genres.size > 0 else prefix_genres
            prefix_ratings = prefix_ratings[visible_mask[-prefix_ratings.size :]] if prefix_ratings.size > 0 else prefix_ratings
        group_sizes.append(int(candidate_ids.shape[0]))
        for candidate_idx, candidate_id in enumerate(candidate_ids.tolist()):
            rows.append(
                _build_xgboost_feature_row(
                    sample=sample,
                    prefix_movies=prefix_movies,
                    prefix_genres=prefix_genres,
                    prefix_ratings=prefix_ratings,
                    candidate_id=int(candidate_id),
                    item_features=item_features,
                    item_popularity=item_popularity,
                )
            )
            labels.append(1 if candidate_idx == 0 else 0)
            sample_weights.append(float(sample.get("target_rating", 0.0)) if candidate_idx == 0 else 1.0)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(group_sizes, dtype=np.int32),
        np.asarray(sample_weights, dtype=np.float32),
    )


def build_xgboost_inference_frame(
    sample: dict,
    candidate_movie_ids: list[int],
    item_features: dict[str, np.ndarray],
    item_popularity: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    valid_candidates = [
        int(movie_id)
        for movie_id in candidate_movie_ids
        if 0 < int(movie_id) < item_features["genres"].shape[0]
    ]
    if not valid_candidates:
        return np.zeros((0, 0), dtype=np.float32), []
    prefix_movies = np.asarray(sample["context_movie_id"], dtype=np.int32)
    prefix_movies = prefix_movies[prefix_movies > 0]
    prefix_genres = np.asarray(sample["context_genres"], dtype=np.int32)
    prefix_genres = prefix_genres[prefix_genres > 0]
    prefix_ratings = np.asarray(sample.get("context_rating", []), dtype=np.float32)
    prefix_ratings = prefix_ratings[prefix_ratings > 0]
    rows = [
        _build_xgboost_feature_row(
            sample=sample,
            prefix_movies=prefix_movies,
            prefix_genres=prefix_genres,
            prefix_ratings=prefix_ratings,
            candidate_id=int(candidate_id),
            item_features=item_features,
            item_popularity=item_popularity,
        )
        for candidate_id in valid_candidates
    ]
    return np.asarray(rows, dtype=np.float32), valid_candidates


def _build_xgboost_feature_row(
    sample: dict,
    prefix_movies: np.ndarray,
    prefix_genres: np.ndarray,
    prefix_ratings: np.ndarray,
    candidate_id: int,
    item_features: dict[str, np.ndarray],
    item_popularity: np.ndarray | None,
) -> list[float]:
    prefix_length = int(prefix_movies.size)
    last_movie = int(prefix_movies[-1]) if prefix_length > 0 else 0
    candidate_genre = int(item_features["genres"][candidate_id])
    candidate_is_adult = int(item_features["isAdult"][candidate_id])
    candidate_start_year = int(item_features["startYear"][candidate_id])
    candidate_genre_tokens = np.asarray(item_features.get("genres_multi", item_features["genres"])[candidate_id]).reshape(-1)
    candidate_genre_tokens = candidate_genre_tokens[candidate_genre_tokens > 0]
    history_genre_hits = int(np.sum(prefix_genres == candidate_genre)) if prefix_genres.size > 0 else 0
    history_movie_hits = int(np.sum(prefix_movies == candidate_id)) if prefix_movies.size > 0 else 0
    same_last_movie = 1 if last_movie == candidate_id and prefix_length > 0 else 0
    last_genre = int(prefix_genres[-1]) if prefix_genres.size > 0 else 0
    same_last_genre = 1 if candidate_genre == last_genre and prefix_genres.size > 0 else 0
    shared_genre_count = int(np.intersect1d(prefix_genres, candidate_genre_tokens, assume_unique=False).size) if prefix_genres.size > 0 and candidate_genre_tokens.size > 0 else 0
    history_genre_overlap_ratio = float(shared_genre_count / max(candidate_genre_tokens.size, 1)) if candidate_genre_tokens.size > 0 else 0.0
    multi_same_last_genre = 1 if prefix_genres.size > 0 and candidate_genre_tokens.size > 0 and int(prefix_genres[-1]) in set(candidate_genre_tokens.tolist()) else 0
    last_rating = float(prefix_ratings[-1]) if prefix_ratings.size > 0 else 0.0
    mean_context_rating = float(prefix_ratings.mean()) if prefix_ratings.size > 0 else 0.0
    high_rating_ratio = float(np.mean(prefix_ratings >= 4.0)) if prefix_ratings.size > 0 else 0.0
    popularity = 0.0
    if item_popularity is not None and 0 <= candidate_id < int(item_popularity.shape[0]):
        popularity = float(item_popularity[candidate_id])
    return [
        float(sample["user_id"]),
        float(sample["gender"]),
        float(sample["age"]),
        float(sample["occupation"]),
        float(sample["zip_code"]),
        float(candidate_id),
        float(candidate_genre),
        float(candidate_is_adult),
        float(candidate_start_year),
        float(prefix_length),
        float(last_movie),
        float(last_genre),
        float(history_genre_hits),
        float(history_movie_hits),
        float(same_last_movie),
        float(same_last_genre),
        float(shared_genre_count),
        float(history_genre_overlap_ratio),
        float(multi_same_last_genre),
        float(last_rating),
        float(mean_context_rating),
        float(high_rating_ratio),
        float(sample.get("target_rating", 0.0)),
        float(popularity),
    ]
