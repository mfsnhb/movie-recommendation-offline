from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import Dataset

from offline.ranking.protocol import CANDIDATE_FIELDS, CANDIDATE_RECALL_FEATURE_FIELDS, HISTORY_FIELDS, SPARSE_ITEM_FEATURE_FIELDS, STATIC_USER_FIELDS, extract_split_sample


_MIN_SAMPLING_PROB = 1e-12


class SequenceRankingDataset(Dataset):
    def __init__(self, split_data: dict):
        self.split_data = split_data
        self.length = int(len(split_data["user_id"]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return extract_split_sample(self.split_data, idx)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {field: value.to(device, non_blocking=True) for field, value in batch.items()}


def build_inference_batch(
    samples: list[dict],
    candidate_movie_ids: list[np.ndarray | list[int]],
    item_features: dict[str, np.ndarray],
    candidate_recall_ranks: list[np.ndarray | list[int]] | None = None,
    candidate_recall_scores: list[np.ndarray | list[float]] | None = None,
) -> tuple[dict[str, torch.Tensor] | None, list[list[int]], list[int]]:
    valid_samples: list[dict] = []
    valid_candidates_by_sample: list[list[int]] = []
    valid_ranks_by_sample: list[np.ndarray] = []
    valid_scores_by_sample: list[np.ndarray] = []
    valid_indices: list[int] = []
    max_candidate_size = 0
    for sample_idx, (sample, raw_candidates) in enumerate(zip(samples, candidate_movie_ids, strict=False)):
        raw_candidate_array = np.asarray(raw_candidates, dtype=np.int32).reshape(-1)
        raw_rank_array = _metadata_array(candidate_recall_ranks, sample_idx, raw_candidate_array.size, dtype=np.float32)
        raw_score_array = _metadata_array(candidate_recall_scores, sample_idx, raw_candidate_array.size, dtype=np.float32)
        valid_positions = [
            pos
            for pos, movie_id in enumerate(raw_candidate_array.tolist())
            if 0 < int(movie_id) < item_features["genres"].shape[0]
        ]
        if not valid_positions:
            continue
        valid_candidates = raw_candidate_array[valid_positions].astype(np.int32, copy=False).tolist()
        valid_samples.append(sample)
        valid_candidates_by_sample.append(valid_candidates)
        valid_ranks_by_sample.append(raw_rank_array[valid_positions])
        valid_scores_by_sample.append(raw_score_array[valid_positions])
        valid_indices.append(sample_idx)
        max_candidate_size = max(max_candidate_size, len(valid_candidates))

    if not valid_samples:
        return None, [], []

    batch = _init_batch_arrays(
        batch_size=len(valid_samples),
        history_width=len(np.asarray(valid_samples[0]["hist_movie_id"]).reshape(-1)),
        candidate_size=max_candidate_size,
        genre_count=int(item_features["genres"].shape[1]),
    )
    candidate_ids = np.zeros((len(valid_samples), max_candidate_size), dtype=np.int32)
    candidate_ranks = np.zeros((len(valid_samples), max_candidate_size), dtype=np.float32)
    candidate_scores = np.zeros((len(valid_samples), max_candidate_size), dtype=np.float32)
    for row_idx, (sample, valid_candidates) in enumerate(zip(valid_samples, valid_candidates_by_sample, strict=True)):
        _fill_base_fields(batch, sample, row_idx=row_idx)
        _fill_history_slots(batch, item_features=item_features, row_idx=row_idx)
        candidate_ids[row_idx, : len(valid_candidates)] = np.asarray(valid_candidates, dtype=np.int32)
        candidate_ranks[row_idx, : len(valid_candidates)] = valid_ranks_by_sample[row_idx]
        candidate_scores[row_idx, : len(valid_candidates)] = valid_scores_by_sample[row_idx]
    _fill_candidate_slots(batch, candidate_ids=candidate_ids, item_features=item_features, candidate_recall_ranks=candidate_ranks, candidate_recall_scores=candidate_scores)
    return _numpy_batch_to_torch(batch), valid_candidates_by_sample, valid_indices


def _metadata_array(values: list[np.ndarray | list] | None, sample_idx: int, width: int, dtype) -> np.ndarray:
    if values is None or sample_idx >= len(values):
        return np.zeros(width, dtype=dtype)
    array = np.asarray(values[sample_idx], dtype=dtype).reshape(-1)
    result = np.zeros(width, dtype=dtype)
    result[: min(width, array.size)] = array[:width]
    return result


@dataclass
class SequenceRankingTrainCollator:
    item_features: dict[str, np.ndarray]
    all_item_ids: np.ndarray
    num_negatives: int
    low_rating_negative_ratio: float = 0.6
    random_negative_ratio: float = 0.4
    negative_popularity_alpha: float = 0.75
    seed: int = 42

    def __post_init__(self) -> None:
        self.all_item_ids = np.asarray(self.all_item_ids, dtype=np.int32)
        self.item_features = {field: np.asarray(self.item_features[field], dtype=np.int32) for field in SPARSE_ITEM_FEATURE_FIELDS}
        max_item_id = int(self.all_item_ids.max()) if self.all_item_ids.size else 0
        self.item_id_positions = np.full(max_item_id + 1, -1, dtype=np.int32)
        if self.all_item_ids.size:
            self.item_id_positions[self.all_item_ids] = np.arange(self.all_item_ids.size, dtype=np.int32)
        popularity = self.item_features["popularity"][self.all_item_ids] if self.all_item_ids.size else np.zeros(0, dtype=np.float32)
        self.popularity_weights = np.power(np.maximum(popularity.astype(np.float64), 1.0), float(self.negative_popularity_alpha))
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        batch = _init_batch_arrays(
            batch_size=len(samples),
            history_width=len(np.asarray(samples[0]["hist_movie_id"]).reshape(-1)),
            candidate_size=self.num_negatives + 1,
            genre_count=int(self.item_features["genres"].shape[1]),
        )
        positive_ids = np.zeros(len(samples), dtype=np.int32)
        for row_idx, sample in enumerate(samples):
            _fill_base_fields(batch, sample, row_idx=row_idx)
            _fill_history_slots(batch, item_features=self.item_features, row_idx=row_idx)
            positive_ids[row_idx] = int(sample["target_movie_id"])

        low_rating_count, popularity_count = _negative_counts_from_ratios(
            num_negatives=self.num_negatives,
            low_rating_ratio=self.low_rating_negative_ratio,
            popularity_ratio=self.random_negative_ratio,
        )
        low_rating_negative_ids, low_rating_negative_logq = _sample_low_rating_negative_pool(
            low_rating_movie_ids=batch["low_rating_movie_id"],
            positive_movie_ids=batch["positive_movie_id"],
            positive_ids=positive_ids,
            count=low_rating_count,
            rng=self.rng,
        )
        negative_ids, negative_logq = _sample_popularity_negative_ids(
            all_item_ids=self.all_item_ids,
            item_id_positions=self.item_id_positions,
            popularity_weights=self.popularity_weights,
            positive_movie_ids=batch["positive_movie_id"],
            positive_ids=positive_ids,
            count=popularity_count,
            rng=self.rng,
            extra_blocked_ids=low_rating_negative_ids,
        )
        candidate_ids = np.concatenate([positive_ids.reshape(-1, 1), low_rating_negative_ids, negative_ids], axis=1)
        positive_logq = _popularity_logq_for_ids(
            item_ids=positive_ids.reshape(-1, 1),
            item_id_positions=self.item_id_positions,
            popularity_weights=self.popularity_weights,
            sample_count=popularity_count,
        )
        candidate_logq = np.concatenate([positive_logq, low_rating_negative_logq, negative_logq], axis=1)
        candidate_ranks = np.zeros(candidate_ids.shape, dtype=np.float32)
        candidate_scores = np.zeros(candidate_ids.shape, dtype=np.float32)
        _fill_candidate_slots(batch, candidate_ids=candidate_ids, item_features=self.item_features, candidate_recall_ranks=candidate_ranks, candidate_recall_scores=candidate_scores)
        batch["candidate_logq"][:, : candidate_logq.shape[1]] = candidate_logq
        return _numpy_batch_to_torch(batch)


def _negative_counts_from_ratios(
    num_negatives: int,
    low_rating_ratio: float,
    popularity_ratio: float,
) -> tuple[int, int]:
    total = max(int(num_negatives), 0)
    if total == 0:
        return 0, 0
    ratios = np.asarray([low_rating_ratio, popularity_ratio], dtype=np.float64)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 0.0)
    if float(ratios.sum()) <= 0.0:
        ratios = np.asarray([0.5, 0.5], dtype=np.float64)
    quotas = ratios / ratios.sum() * total
    counts = np.floor(quotas).astype(np.int32)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(quotas - counts))
        counts[order[:remainder]] += 1
    return int(counts[0]), int(counts[1])


def _blocked_positive_set(positive_movie_ids: np.ndarray, positive_id: int, extra_blocked_ids: np.ndarray | None = None, row_idx: int | None = None) -> set[int]:
    blocked = set(int(item_id) for item_id in positive_movie_ids.reshape(-1).tolist() if int(item_id) > 0)
    blocked.add(int(positive_id))
    if extra_blocked_ids is not None and extra_blocked_ids.size > 0 and row_idx is not None:
        blocked.update(int(item_id) for item_id in extra_blocked_ids[row_idx].reshape(-1).tolist() if int(item_id) > 0)
    return blocked


def _sample_low_rating_negative_pool(
    low_rating_movie_ids: np.ndarray,
    positive_movie_ids: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(positive_ids.shape[0])
    if count <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32), np.zeros((batch_size, 0), dtype=np.float32)

    negatives = np.zeros((batch_size, count), dtype=np.int32)
    logq = np.zeros((batch_size, count), dtype=np.float32)
    for row_idx in range(batch_size):
        blocked = _blocked_positive_set(positive_movie_ids[row_idx], int(positive_ids[row_idx]))
        candidates = np.asarray(low_rating_movie_ids[row_idx], dtype=np.int32)
        candidates = np.asarray([int(item_id) for item_id in candidates.tolist() if int(item_id) > 0 and int(item_id) not in blocked], dtype=np.int32)
        if candidates.size == 0:
            continue
        unique_candidates = np.unique(candidates.astype(np.int32, copy=False))
        take = min(count, int(unique_candidates.size))
        if take < unique_candidates.size:
            selected = rng.choice(unique_candidates, size=take, replace=False).astype(np.int32, copy=False)
        else:
            selected = unique_candidates[-take:]
        negatives[row_idx, :take] = selected
        expected_count = min(float(count) / max(float(unique_candidates.size), 1.0), 1.0)
        logq[row_idx, :take] = np.float32(np.log(max(expected_count, _MIN_SAMPLING_PROB)))
    return negatives, logq


def _sample_popularity_negative_ids(
    all_item_ids: np.ndarray,
    item_id_positions: np.ndarray,
    popularity_weights: np.ndarray,
    positive_movie_ids: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
    extra_blocked_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = int(positive_ids.shape[0])
    if count <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32), np.zeros((batch_size, 0), dtype=np.float32)
    if all_item_ids.size == 0:
        return np.zeros((batch_size, count), dtype=np.int32), np.zeros((batch_size, count), dtype=np.float32)

    negatives = np.zeros((batch_size, count), dtype=np.int32)
    logq = np.zeros((batch_size, count), dtype=np.float32)
    base_weights = np.asarray(popularity_weights, dtype=np.float64)
    for row_idx in range(batch_size):
        weights = base_weights.copy()
        blocked = _blocked_positive_set(positive_movie_ids[row_idx], int(positive_ids[row_idx]), extra_blocked_ids, row_idx)
        if blocked:
            blocked_array = np.asarray(list(blocked), dtype=np.int32)
            valid_blocked = (blocked_array > 0) & (blocked_array < item_id_positions.shape[0])
            if np.any(valid_blocked):
                blocked_positions = item_id_positions[blocked_array[valid_blocked]]
                blocked_positions = blocked_positions[blocked_positions >= 0]
                weights[blocked_positions] = 0.0
        valid_positions = np.flatnonzero(weights > 0.0)
        take = min(count, int(valid_positions.size))
        if take <= 0:
            continue
        probabilities = weights[valid_positions]
        probabilities = probabilities / probabilities.sum()
        selected_positions = rng.choice(valid_positions, size=take, replace=False, p=probabilities)
        negatives[row_idx, :take] = all_item_ids[selected_positions].astype(np.int32, copy=False)
        selected_probabilities = probabilities[np.searchsorted(valid_positions, selected_positions)]
        expected_counts = np.minimum(float(count) * selected_probabilities, 1.0)
        logq[row_idx, :take] = np.log(np.maximum(expected_counts, _MIN_SAMPLING_PROB)).astype(np.float32, copy=False)
    return negatives, logq


def _popularity_logq_for_ids(
    item_ids: np.ndarray,
    item_id_positions: np.ndarray,
    popularity_weights: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    ids = np.asarray(item_ids, dtype=np.int32)
    logq = np.zeros(ids.shape, dtype=np.float32)
    if sample_count <= 0 or ids.size == 0 or popularity_weights.size == 0:
        return logq
    weights = np.asarray(popularity_weights, dtype=np.float64)
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return logq
    valid_ids = (ids > 0) & (ids < item_id_positions.shape[0])
    positions = np.full(ids.shape, -1, dtype=np.int32)
    positions[valid_ids] = item_id_positions[ids[valid_ids]]
    valid_positions = positions >= 0
    expected_counts = np.zeros(ids.shape, dtype=np.float64)
    expected_counts[valid_positions] = np.minimum(float(sample_count) * weights[positions[valid_positions]] / denominator, 1.0)
    logq[valid_positions] = np.log(np.maximum(expected_counts[valid_positions], _MIN_SAMPLING_PROB)).astype(np.float32, copy=False)
    return logq


def _init_batch_arrays(batch_size: int, history_width: int, candidate_size: int, genre_count: int) -> dict[str, np.ndarray]:
    batch = {field: np.zeros(batch_size, dtype=np.int32) for field in STATIC_USER_FIELDS}
    batch["user_id_raw"] = np.zeros(batch_size, dtype=np.int32)
    for field in HISTORY_FIELDS:
        if field == "hist_length":
            batch[field] = np.zeros(batch_size, dtype=np.int32)
        elif field == "hist_rating":
            batch[field] = np.zeros((batch_size, history_width), dtype=np.float32)
        else:
            batch[field] = np.zeros((batch_size, history_width), dtype=np.int32)
    batch["hist_genres"] = np.zeros((batch_size, history_width, genre_count), dtype=np.int32)
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        if field != "genres":
            batch[f"hist_{field}"] = np.zeros((batch_size, history_width), dtype=np.int32)
    batch["target_index"] = np.zeros(batch_size, dtype=np.int64)
    batch["target_rating"] = np.zeros(batch_size, dtype=np.float32)
    batch["candidate_mask"] = np.zeros((batch_size, candidate_size), dtype=bool)
    for field in CANDIDATE_FIELDS:
        source_field = field.removeprefix("candidate_")
        if source_field == "genres":
            batch[field] = np.zeros((batch_size, candidate_size, genre_count), dtype=np.int32)
        else:
            batch[field] = np.zeros((batch_size, candidate_size), dtype=np.int32)
    for field in CANDIDATE_RECALL_FEATURE_FIELDS:
        batch[field] = np.zeros((batch_size, candidate_size), dtype=np.float32)
    batch["candidate_relevance"] = np.zeros((batch_size, candidate_size), dtype=np.float32)
    batch["candidate_logq"] = np.zeros((batch_size, candidate_size), dtype=np.float32)
    return batch


def _feedback_from_rating(rating: np.ndarray) -> np.ndarray:
    rating_values = np.asarray(rating, dtype=np.float32)
    feedback = np.zeros(rating_values.shape, dtype=np.int32)
    feedback[(rating_values > 0.0) & (rating_values <= 2.0)] = 1
    feedback[np.rint(rating_values) == 3.0] = 2
    feedback[rating_values >= 4.0] = 3
    return feedback


def _fill_base_fields(batch: dict[str, np.ndarray], sample: dict, row_idx: int) -> None:
    for field in STATIC_USER_FIELDS:
        batch[field][row_idx] = int(sample[field])
    batch["user_id_raw"][row_idx] = int(sample.get("user_id_raw", 0))
    batch["hist_movie_id"][row_idx] = np.asarray(sample["hist_movie_id"], dtype=np.int32)
    hist_rating = np.asarray(sample.get("hist_rating", np.zeros_like(sample["hist_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["hist_rating"][row_idx] = hist_rating
    batch["hist_feedback"][row_idx] = np.asarray(sample.get("hist_feedback", _feedback_from_rating(hist_rating)), dtype=np.int32)
    batch["hist_time_gap_bucket"][row_idx] = np.asarray(sample.get("hist_time_gap_bucket", np.zeros_like(sample["hist_movie_id"], dtype=np.int32)), dtype=np.int32)
    batch["hist_length"][row_idx] = int(sample["hist_length"])
    batch["low_rating_movie_id"][row_idx] = np.asarray(sample.get("low_rating_movie_id", np.zeros_like(sample["hist_movie_id"], dtype=np.int32)), dtype=np.int32)
    batch["positive_movie_id"][row_idx] = np.asarray(sample.get("positive_movie_id", np.zeros_like(sample["hist_movie_id"], dtype=np.int32)), dtype=np.int32)
    batch["target_rating"][row_idx] = float(sample.get("target_rating", 0.0))


def _fill_history_slots(batch: dict[str, np.ndarray], item_features: dict[str, np.ndarray], row_idx: int) -> None:
    history_ids = batch["hist_movie_id"][row_idx]
    valid_mask = (history_ids > 0) & (history_ids < item_features["genres"].shape[0])
    safe_history_ids = np.where(valid_mask, history_ids, 0).astype(np.int32, copy=False)
    batch["hist_genres"][row_idx] = item_features["genres"][safe_history_ids]
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        if field != "genres":
            batch[f"hist_{field}"][row_idx] = item_features[field][safe_history_ids]


def _fill_candidate_slots(
    batch: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    item_features: dict[str, np.ndarray],
    candidate_recall_ranks: np.ndarray | None = None,
    candidate_recall_scores: np.ndarray | None = None,
) -> None:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
    candidate_size = min(candidate_ids.shape[1], batch["candidate_movie_id"].shape[1])
    candidate_ids = candidate_ids[:, :candidate_size]
    valid_mask = (candidate_ids > 0) & (candidate_ids < item_features["genres"].shape[0])
    safe_candidate_ids = np.where(valid_mask, candidate_ids, 0).astype(np.int32, copy=False)
    batch["target_index"][: candidate_ids.shape[0]] = 0
    batch["candidate_mask"][:, :candidate_size] = valid_mask
    batch["candidate_movie_id"][:, :candidate_size] = safe_candidate_ids
    batch["candidate_genres"][:, :candidate_size] = item_features["genres"][safe_candidate_ids]
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        if field != "genres":
            batch[f"candidate_{field}"][:, :candidate_size] = item_features[field][safe_candidate_ids]
    if candidate_recall_ranks is not None:
        batch["candidate_recall_rank"][:, :candidate_size] = np.asarray(candidate_recall_ranks[:, :candidate_size], dtype=np.float32) * valid_mask.astype(np.float32)
    if candidate_recall_scores is not None:
        batch["candidate_recall_score"][:, :candidate_size] = np.asarray(candidate_recall_scores[:, :candidate_size], dtype=np.float32) * valid_mask.astype(np.float32)
    batch["candidate_relevance"][:, :candidate_size] = 0.0
    batch["candidate_relevance"][:, 0] = batch["target_rating"]


def _numpy_batch_to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    result = {}
    for field, values in batch.items():
        if field == "candidate_mask":
            result[field] = torch.as_tensor(values, dtype=torch.bool)
        elif field == "target_index":
            result[field] = torch.as_tensor(values, dtype=torch.long)
        elif field in {"hist_rating", "target_rating", "candidate_relevance", "candidate_recall_rank", "candidate_recall_score", "candidate_logq"}:
            result[field] = torch.as_tensor(values, dtype=torch.float32)
        else:
            result[field] = torch.as_tensor(values, dtype=torch.long)
    return result
