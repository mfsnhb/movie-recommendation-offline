from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import Dataset

from offline.ranking.protocol import CANDIDATE_FIELDS, CONTEXT_FIELDS, SPARSE_ITEM_FEATURE_FIELDS, STATIC_USER_FIELDS, extract_split_sample


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
) -> tuple[dict[str, torch.Tensor] | None, list[list[int]], list[int]]:
    valid_samples: list[dict] = []
    valid_candidates_by_sample: list[list[int]] = []
    valid_indices: list[int] = []
    max_candidate_size = 0
    for sample_idx, (sample, raw_candidates) in enumerate(zip(samples, candidate_movie_ids, strict=False)):
        valid_candidates = [
            int(movie_id)
            for movie_id in np.asarray(raw_candidates, dtype=np.int32).reshape(-1).tolist()
            if 0 < int(movie_id) < item_features["genres"].shape[0]
        ]
        if not valid_candidates:
            continue
        valid_samples.append(sample)
        valid_candidates_by_sample.append(valid_candidates)
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
    for row_idx, (sample, valid_candidates) in enumerate(zip(valid_samples, valid_candidates_by_sample, strict=True)):
        _fill_base_fields(batch, sample, row_idx=row_idx)
        _fill_context_slots(batch, item_features=item_features, row_idx=row_idx)
        candidate_ids[row_idx, : len(valid_candidates)] = np.asarray(valid_candidates, dtype=np.int32)
    _fill_candidate_slots(batch, candidate_ids=candidate_ids, item_features=item_features)
    return _numpy_batch_to_torch(batch), valid_candidates_by_sample, valid_indices


@dataclass
class SequenceRankingTrainCollator:
    item_features: dict[str, np.ndarray]
    all_item_ids: np.ndarray
    num_negatives: int
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

        negative_ids = _sample_batch_negative_ids(
            all_item_ids=self.all_item_ids,
            item_id_positions=self.item_id_positions,
            context_movie_ids=batch["context_movie_id"],
            positive_ids=positive_ids,
            count=self.num_negatives,
            rng=self.rng,
        )
        candidate_ids = np.concatenate([positive_ids.reshape(-1, 1), negative_ids], axis=1)
        _fill_candidate_slots(batch, candidate_ids=candidate_ids, item_features=self.item_features)
        return _numpy_batch_to_torch(batch)


def _sample_batch_negative_ids(
    all_item_ids: np.ndarray,
    item_id_positions: np.ndarray,
    context_movie_ids: np.ndarray,
    positive_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size = int(positive_ids.shape[0])
    if count <= 0:
        return np.zeros((batch_size, 0), dtype=np.int32)
    if all_item_ids.size == 0:
        return np.zeros((batch_size, count), dtype=np.int32)

    scores = rng.random((batch_size, all_item_ids.size), dtype=np.float32)
    blocked_ids = np.concatenate([context_movie_ids.astype(np.int32, copy=False), positive_ids.reshape(-1, 1)], axis=1)
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
    batch["candidate_relevance"] = np.zeros((batch_size, candidate_size), dtype=np.float32)
    return batch


def _fill_base_fields(batch: dict[str, np.ndarray], sample: dict, row_idx: int) -> None:
    for field in STATIC_USER_FIELDS:
        batch[field][row_idx] = int(sample[field])
    batch["user_id_original"][row_idx] = int(sample.get("user_id_original", sample["user_id"]))
    batch["context_movie_id"][row_idx] = np.asarray(sample["context_movie_id"], dtype=np.int32)
    batch["context_rating"][row_idx] = np.asarray(sample.get("context_rating", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["context_low_rating_mask"][row_idx] = np.asarray(sample.get("context_low_rating_mask", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
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
    batch["candidate_relevance"][:, :candidate_size] = 0.0
    batch["candidate_relevance"][:, 0] = batch["target_rating"]


def _numpy_batch_to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    result = {}
    for field, values in batch.items():
        if field == "candidate_mask":
            result[field] = torch.as_tensor(values, dtype=torch.bool)
        elif field == "target_index":
            result[field] = torch.as_tensor(values, dtype=torch.long)
        elif field in {"context_rating", "context_low_rating_mask", "target_rating", "candidate_relevance"}:
            result[field] = torch.as_tensor(values, dtype=torch.float32)
        else:
            result[field] = torch.as_tensor(values, dtype=torch.long)
    return result
