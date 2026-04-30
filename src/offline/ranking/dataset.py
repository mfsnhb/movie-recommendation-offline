from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import Dataset

from offline.ranking.protocol import CANDIDATE_FIELDS, CANDIDATE_RECALL_FEATURE_FIELDS, CONTEXT_FIELDS, SPARSE_ITEM_FEATURE_FIELDS, STATIC_USER_FIELDS, extract_split_sample


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
        context_width=len(np.asarray(valid_samples[0]["context_movie_id"]).reshape(-1)),
        candidate_size=max_candidate_size,
        genre_count=int(item_features["genres"].shape[1]),
    )
    candidate_ids = np.zeros((len(valid_samples), max_candidate_size), dtype=np.int32)
    candidate_ranks = np.zeros((len(valid_samples), max_candidate_size), dtype=np.float32)
    candidate_scores = np.zeros((len(valid_samples), max_candidate_size), dtype=np.float32)
    for row_idx, (sample, valid_candidates) in enumerate(zip(valid_samples, valid_candidates_by_sample, strict=True)):
        _fill_base_fields(batch, sample, row_idx=row_idx)
        _fill_context_slots(batch, item_features=item_features, row_idx=row_idx)
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
    low_rating_negatives: int = 5
    recall_hard_negatives: int = 10
    random_negatives: int = 5
    fused_candidates_by_user: dict[int, np.ndarray] | None = None
    fused_scores_by_user: dict[int, np.ndarray] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        self.all_item_ids = np.asarray(self.all_item_ids, dtype=np.int32)
        self.item_features = {field: np.asarray(self.item_features[field], dtype=np.int32) for field in SPARSE_ITEM_FEATURE_FIELDS}
        max_item_id = int(self.all_item_ids.max()) if self.all_item_ids.size else 0
        self.item_id_positions = np.full(max_item_id + 1, -1, dtype=np.int32)
        if self.all_item_ids.size:
            self.item_id_positions[self.all_item_ids] = np.arange(self.all_item_ids.size, dtype=np.int32)
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        batch = _init_batch_arrays(
            batch_size=len(samples),
            context_width=len(np.asarray(samples[0]["context_movie_id"]).reshape(-1)),
            candidate_size=self.num_negatives + 1,
            genre_count=int(self.item_features["genres"].shape[1]),
        )
        positive_ids = np.zeros(len(samples), dtype=np.int32)
        for row_idx, sample in enumerate(samples):
            _fill_base_fields(batch, sample, row_idx=row_idx)
            _fill_context_slots(batch, item_features=self.item_features, row_idx=row_idx)
            positive_ids[row_idx] = int(sample["target_movie_id"])

        low_rating_count = min(max(int(self.low_rating_negatives), 0), self.num_negatives)
        hard_count = min(max(int(self.recall_hard_negatives), 0), max(self.num_negatives - low_rating_count, 0))
        random_count = min(max(int(self.random_negatives), 0), max(self.num_negatives - low_rating_count - hard_count, 0))
        random_count += max(self.num_negatives - low_rating_count - hard_count - random_count, 0)
        low_rating_negative_ids = _sample_low_rating_negative_ids(
            context_movie_ids=batch["context_movie_id"],
            context_low_rating_mask=batch["context_low_rating_mask"],
            positive_ids=positive_ids,
            count=low_rating_count,
            rng=self.rng,
        )
        hard_negative_ids, hard_negative_ranks, hard_negative_scores = _sample_recall_hard_negative_ids(
            samples=samples,
            context_movie_ids=batch["context_movie_id"],
            positive_ids=positive_ids,
            count=hard_count,
            fused_candidates_by_user=self.fused_candidates_by_user,
            fused_scores_by_user=self.fused_scores_by_user,
            extra_blocked_ids=low_rating_negative_ids,
        )
        negative_ids = _sample_batch_negative_ids(
            all_item_ids=self.all_item_ids,
            item_id_positions=self.item_id_positions,
            context_movie_ids=batch["context_movie_id"],
            positive_ids=positive_ids,
            count=random_count,
            rng=self.rng,
            extra_blocked_ids=np.concatenate([low_rating_negative_ids, hard_negative_ids], axis=1),
        )
        candidate_ids = np.concatenate([positive_ids.reshape(-1, 1), low_rating_negative_ids, hard_negative_ids, negative_ids], axis=1)
        candidate_ranks = np.zeros(candidate_ids.shape, dtype=np.float32)
        candidate_scores = np.zeros(candidate_ids.shape, dtype=np.float32)
        hard_start = 1 + low_rating_negative_ids.shape[1]
        candidate_ranks[:, hard_start : hard_start + hard_negative_ids.shape[1]] = hard_negative_ranks
        candidate_scores[:, hard_start : hard_start + hard_negative_ids.shape[1]] = hard_negative_scores
        _fill_candidate_slots(batch, candidate_ids=candidate_ids, item_features=self.item_features, candidate_recall_ranks=candidate_ranks, candidate_recall_scores=candidate_scores)
        return _numpy_batch_to_torch(batch)


def _sample_low_rating_negative_ids(
    context_movie_ids: np.ndarray,
    context_low_rating_mask: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size = int(positive_ids.shape[0])
    if count <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32)

    negatives = np.zeros((batch_size, count), dtype=np.int32)
    for row_idx in range(batch_size):
        candidates = context_movie_ids[row_idx][context_low_rating_mask[row_idx] > 0]
        candidates = candidates[(candidates > 0) & (candidates != positive_ids[row_idx])]
        if candidates.size == 0:
            continue
        unique_candidates = np.unique(candidates.astype(np.int32, copy=False))
        take = min(count, int(unique_candidates.size))
        if take < unique_candidates.size:
            selected = rng.choice(unique_candidates, size=take, replace=False).astype(np.int32, copy=False)
        else:
            selected = unique_candidates[-take:]
        negatives[row_idx, :take] = selected
    return negatives


def _sample_recall_hard_negative_ids(
    samples: list[dict],
    context_movie_ids: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    fused_candidates_by_user: dict[int, np.ndarray] | None,
    fused_scores_by_user: dict[int, np.ndarray] | None,
    extra_blocked_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(positive_ids.shape[0])
    negatives = np.zeros((batch_size, count), dtype=np.int32)
    ranks = np.zeros((batch_size, count), dtype=np.float32)
    scores = np.zeros((batch_size, count), dtype=np.float32)
    if count <= 0 or not fused_candidates_by_user:
        return negatives, ranks, scores

    for row_idx, sample in enumerate(samples):
        user_id = int(sample.get("user_id_original", sample["user_id"]))
        candidates = fused_candidates_by_user.get(user_id)
        if candidates is None:
            candidates = fused_candidates_by_user.get(int(sample["user_id"]))
        if candidates is None:
            continue
        candidate_array = np.asarray(candidates, dtype=np.int32).reshape(-1)
        score_array = None
        if fused_scores_by_user:
            score_values = fused_scores_by_user.get(user_id)
            if score_values is None:
                score_values = fused_scores_by_user.get(int(sample["user_id"]))
            if score_values is not None:
                score_array = np.asarray(score_values, dtype=np.float32).reshape(-1)
        blocked = set(int(item_id) for item_id in context_movie_ids[row_idx].tolist() if int(item_id) > 0)
        blocked.add(int(positive_ids[row_idx]))
        if extra_blocked_ids is not None and extra_blocked_ids.size > 0:
            blocked.update(int(item_id) for item_id in extra_blocked_ids[row_idx].tolist() if int(item_id) > 0)
        selected = []
        selected_ranks = []
        selected_scores = []
        for rank_idx, item_id in enumerate(candidate_array.tolist(), start=1):
            item_id = int(item_id)
            if item_id <= 0 or item_id in blocked:
                continue
            selected.append(item_id)
            selected_ranks.append(float(rank_idx))
            selected_scores.append(float(score_array[rank_idx - 1]) if score_array is not None and rank_idx - 1 < score_array.size else 0.0)
            blocked.add(item_id)
            if len(selected) >= count:
                break
        take = len(selected)
        if take > 0:
            negatives[row_idx, :take] = np.asarray(selected, dtype=np.int32)
            ranks[row_idx, :take] = np.asarray(selected_ranks, dtype=np.float32)
            scores[row_idx, :take] = np.asarray(selected_scores, dtype=np.float32)
    return negatives, ranks, scores

def _sample_batch_negative_ids(
    all_item_ids: np.ndarray,
    item_id_positions: np.ndarray,
    context_movie_ids: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
    extra_blocked_ids: np.ndarray | None = None,
) -> np.ndarray:
    batch_size = int(positive_ids.shape[0])
    if count <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32)
    if all_item_ids.size == 0:
        return np.zeros((batch_size, count), dtype=np.int32)

    scores = rng.random((batch_size, all_item_ids.size), dtype=np.float32)
    blocked_ids = np.concatenate([context_movie_ids.astype(np.int32, copy=False), positive_ids.reshape(-1, 1)], axis=1)
    if extra_blocked_ids is not None and extra_blocked_ids.size > 0:
        blocked_ids = np.concatenate([blocked_ids, extra_blocked_ids.astype(np.int32, copy=False)], axis=1)
    flat_blocked = blocked_ids.reshape(-1)
    flat_rows = np.repeat(np.arange(batch_size, dtype=np.int64), blocked_ids.shape[1])
    valid_blocked = (flat_blocked > 0) & (flat_blocked < item_id_positions.shape[0])
    if np.any(valid_blocked):
        blocked_positions = item_id_positions[flat_blocked[valid_blocked]]
        valid_positions = blocked_positions >= 0
        if np.any(valid_positions):
            scores[flat_rows[valid_blocked][valid_positions], blocked_positions[valid_positions]] = -1.0

    take = min(count, int(all_item_ids.size))
    selected_positions = np.argpartition(scores, kth=all_item_ids.size - take, axis=1)[:, -take:]
    selected_scores = np.take_along_axis(scores, selected_positions, axis=1)
    selected_ids = np.where(selected_scores >= 0.0, all_item_ids[selected_positions], 0).astype(np.int32, copy=False)
    negatives = np.zeros((batch_size, count), dtype=np.int32)
    negatives[:, :take] = selected_ids
    return negatives


def _init_batch_arrays(batch_size: int, context_width: int, candidate_size: int, genre_count: int) -> dict[str, np.ndarray]:
    batch = {field: np.zeros(batch_size, dtype=np.int32) for field in STATIC_USER_FIELDS}
    batch["user_id_original"] = np.zeros(batch_size, dtype=np.int32)
    for field in CONTEXT_FIELDS:
        if field == "context_length":
            batch[field] = np.zeros(batch_size, dtype=np.int32)
        elif field in {"context_rating", "context_low_rating_mask"}:
            batch[field] = np.zeros((batch_size, context_width), dtype=np.float32)
        else:
            batch[field] = np.zeros((batch_size, context_width), dtype=np.int32)
    batch["context_genres"] = np.zeros((batch_size, context_width, genre_count), dtype=np.int32)
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        if field != "genres":
            batch[f"context_{field}"] = np.zeros((batch_size, context_width), dtype=np.int32)
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
    return batch


def _fill_base_fields(batch: dict[str, np.ndarray], sample: dict, row_idx: int) -> None:
    for field in STATIC_USER_FIELDS:
        batch[field][row_idx] = int(sample[field])
    batch["user_id_original"][row_idx] = int(sample.get("user_id_original", sample["user_id"]))
    batch["context_movie_id"][row_idx] = np.asarray(sample["context_movie_id"], dtype=np.int32)
    batch["context_rating"][row_idx] = np.asarray(sample.get("context_rating", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["context_low_rating_mask"][row_idx] = np.asarray(sample.get("context_low_rating_mask", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["context_recency_bucket"][row_idx] = np.asarray(sample.get("context_recency_bucket", np.zeros_like(sample["context_movie_id"], dtype=np.int32)), dtype=np.int32)
    batch["context_length"][row_idx] = int(sample["context_length"])
    batch["target_rating"][row_idx] = float(sample.get("target_rating", 0.0))


def _fill_context_slots(batch: dict[str, np.ndarray], item_features: dict[str, np.ndarray], row_idx: int) -> None:
    context_ids = batch["context_movie_id"][row_idx]
    valid_mask = (context_ids > 0) & (context_ids < item_features["genres"].shape[0])
    safe_context_ids = np.where(valid_mask, context_ids, 0).astype(np.int32, copy=False)
    batch["context_genres"][row_idx] = item_features["genres"][safe_context_ids]
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        if field != "genres":
            batch[f"context_{field}"][row_idx] = item_features[field][safe_context_ids]


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
        elif field in {"context_rating", "context_low_rating_mask", "target_rating", "candidate_relevance", "candidate_recall_rank", "candidate_recall_score"}:
            result[field] = torch.as_tensor(values, dtype=torch.float32)
        else:
            result[field] = torch.as_tensor(values, dtype=torch.long)
    return result
