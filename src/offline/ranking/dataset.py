from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - preprocessing can run without torch installed
    torch = None

    class Dataset:  # type: ignore[override]
        pass


from offline.ranking.protocol import (
    CANDIDATE_FIELDS,
    CONTEXT_FIELDS,
    STATIC_USER_FIELDS,
    extract_split_sample,
    get_seen_movie_ids,
    sample_negative_ids_with_candidates,
)


class SequenceRankingDataset(Dataset):
    def __init__(self, split_data: dict):
        self.split_data = split_data
        self.length = int(len(split_data["user_id"]))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return extract_split_sample(self.split_data, idx)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError("torch is required for ranking training")
    return {field: value.to(device) for field, value in batch.items()}


def build_inference_batch(
    sample: dict,
    candidate_movie_ids: list[int],
    item_features: dict[str, np.ndarray],
) -> tuple[dict[str, torch.Tensor] | None, list[int]]:
    if torch is None:
        raise ModuleNotFoundError("torch is required for ranking inference")

    valid_candidates = [
        int(movie_id)
        for movie_id in candidate_movie_ids
        if 0 < int(movie_id) < item_features["genres"].shape[0]
    ]
    if not valid_candidates:
        return None, []

    batch = _init_batch_arrays(
        batch_size=1,
        context_width=len(np.asarray(sample["context_movie_id"]).reshape(-1)),
        candidate_size=len(valid_candidates),
    )
    _fill_base_fields(batch, sample, row_idx=0)
    candidate_ids = np.asarray(valid_candidates, dtype=np.int32)
    _fill_candidate_slot(batch, row_idx=0, candidate_ids=candidate_ids, item_features=item_features)
    return _numpy_batch_to_torch(batch), valid_candidates


def _resolve_candidate_pool(
    sample: dict,
    fused_candidates_by_user: dict[int, np.ndarray] | None,
) -> np.ndarray | None:
    if not fused_candidates_by_user:
        return None
    user_id = int(sample["user_id"])
    candidate_pool = fused_candidates_by_user.get(user_id)
    if candidate_pool is None or int(np.asarray(candidate_pool).size) == 0:
        return None
    return np.asarray(candidate_pool, dtype=np.int32).reshape(-1)


@dataclass
class SequenceRankingTrainCollator:
    item_features: dict[str, np.ndarray]
    all_item_ids: np.ndarray
    num_negatives: int
    seed: int = 42
    fused_candidates_by_user: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.all_item_ids = np.asarray(self.all_item_ids, dtype=np.int32)
        self.item_features = {field: np.asarray(values, dtype=np.int32) for field, values in self.item_features.items()}
        if self.fused_candidates_by_user is not None:
            self.fused_candidates_by_user = {
                int(user_id): np.asarray(candidate_ids, dtype=np.int32)
                for user_id, candidate_ids in self.fused_candidates_by_user.items()
            }
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        batch = _init_batch_arrays(
            batch_size=len(samples),
            context_width=len(np.asarray(samples[0]["context_movie_id"]).reshape(-1)),
            candidate_size=self.num_negatives + 1,
        )
        for row_idx, sample in enumerate(samples):
            _fill_base_fields(batch, sample, row_idx=row_idx)
            positive_movie_id = int(sample["target_movie_id"])
            negatives = sample_negative_ids_with_candidates(
                self.all_item_ids,
                get_seen_movie_ids(sample),
                positive_movie_id,
                self.num_negatives,
                self.rng,
                candidate_pool=_resolve_candidate_pool(sample, self.fused_candidates_by_user),
            )
            candidate_ids = np.concatenate([[positive_movie_id], negatives], axis=0)
            _fill_candidate_slot(batch, row_idx=row_idx, candidate_ids=candidate_ids, item_features=self.item_features)
        return _numpy_batch_to_torch(batch)


@dataclass
class SequenceRankingValidationCollator:
    item_features: dict[str, np.ndarray]
    all_item_ids: np.ndarray
    num_negatives: int
    seed: int = 97
    fused_candidates_by_user: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.all_item_ids = np.asarray(self.all_item_ids, dtype=np.int32)
        self.item_features = {field: np.asarray(values, dtype=np.int32) for field, values in self.item_features.items()}
        if self.fused_candidates_by_user is not None:
            self.fused_candidates_by_user = {
                int(user_id): np.asarray(candidate_ids, dtype=np.int32)
                for user_id, candidate_ids in self.fused_candidates_by_user.items()
            }

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        batch = _init_batch_arrays(
            batch_size=len(samples),
            context_width=len(np.asarray(samples[0]["context_movie_id"]).reshape(-1)),
            candidate_size=self.num_negatives + 1,
        )
        for row_idx, sample in enumerate(samples):
            _fill_base_fields(batch, sample, row_idx=row_idx)
            positive_movie_id = int(sample["target_movie_id"])
            rng = np.random.default_rng(self.seed + int(sample.get("user_id_original", sample["user_id"])) + positive_movie_id * 1009)
            negatives = sample_negative_ids_with_candidates(
                self.all_item_ids,
                get_seen_movie_ids(sample),
                positive_movie_id,
                self.num_negatives,
                rng,
                candidate_pool=_resolve_candidate_pool(sample, self.fused_candidates_by_user),
            )
            candidate_ids = np.concatenate([[positive_movie_id], negatives], axis=0)
            _fill_candidate_slot(batch, row_idx=row_idx, candidate_ids=candidate_ids, item_features=self.item_features)
        return _numpy_batch_to_torch(batch)


@dataclass
class SequenceRankingEvalCollator:
    item_features: dict[str, np.ndarray]
    all_item_ids: np.ndarray
    num_negatives: int
    seed: int = 193
    fused_candidates_by_user: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.all_item_ids = np.asarray(self.all_item_ids, dtype=np.int32)
        self.item_features = {field: np.asarray(values, dtype=np.int32) for field, values in self.item_features.items()}
        if self.fused_candidates_by_user is not None:
            self.fused_candidates_by_user = {
                int(user_id): np.asarray(candidate_ids, dtype=np.int32)
                for user_id, candidate_ids in self.fused_candidates_by_user.items()
            }

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        batch = _init_batch_arrays(
            batch_size=len(samples),
            context_width=len(np.asarray(samples[0]["context_movie_id"]).reshape(-1)),
            candidate_size=self.num_negatives + 1,
        )
        for row_idx, sample in enumerate(samples):
            _fill_base_fields(batch, sample, row_idx=row_idx)
            positive_movie_id = int(sample["target_movie_id"])
            rng = np.random.default_rng(self.seed + int(sample.get("user_id_original", sample["user_id"])) + positive_movie_id * 1009)
            negatives = sample_negative_ids_with_candidates(
                self.all_item_ids,
                get_seen_movie_ids(sample),
                positive_movie_id,
                self.num_negatives,
                rng,
                candidate_pool=_resolve_candidate_pool(sample, self.fused_candidates_by_user),
            )
            candidate_ids = np.concatenate([[positive_movie_id], negatives], axis=0)
            _fill_candidate_slot(batch, row_idx=row_idx, candidate_ids=candidate_ids, item_features=self.item_features)
        return _numpy_batch_to_torch(batch)


def _init_batch_arrays(batch_size: int, context_width: int, candidate_size: int) -> dict[str, np.ndarray]:
    batch = {field: np.zeros(batch_size, dtype=np.int32) for field in STATIC_USER_FIELDS}
    batch["user_id_original"] = np.zeros(batch_size, dtype=np.int32)
    for field in CONTEXT_FIELDS:
        if field == "context_length":
            batch[field] = np.zeros(batch_size, dtype=np.int32)
        elif field in {"context_rating", "context_low_rating_mask"}:
            batch[field] = np.zeros((batch_size, context_width), dtype=np.float32)
        else:
            batch[field] = np.zeros((batch_size, context_width), dtype=np.int32)
    batch["target_index"] = np.zeros(batch_size, dtype=np.int64)
    batch["target_rating"] = np.zeros(batch_size, dtype=np.float32)
    batch["candidate_mask"] = np.zeros((batch_size, candidate_size), dtype=bool)
    for field in CANDIDATE_FIELDS:
        batch[field] = np.zeros((batch_size, candidate_size), dtype=np.int32)
    return batch


def _fill_base_fields(batch: dict[str, np.ndarray], sample: dict, row_idx: int) -> None:
    for field in STATIC_USER_FIELDS:
        batch[field][row_idx] = int(sample[field])
    batch["user_id_original"][row_idx] = int(sample.get("user_id_original", sample["user_id"]))
    batch["context_movie_id"][row_idx] = np.asarray(sample["context_movie_id"], dtype=np.int32)
    batch["context_genres"][row_idx] = np.asarray(sample["context_genres"], dtype=np.int32)
    batch["context_rating"][row_idx] = np.asarray(sample.get("context_rating", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["context_low_rating_mask"][row_idx] = np.asarray(sample.get("context_low_rating_mask", np.zeros_like(sample["context_movie_id"], dtype=np.float32)), dtype=np.float32)
    batch["context_length"][row_idx] = int(sample["context_length"])
    batch["target_rating"][row_idx] = float(sample.get("target_rating", 0.0))


def _fill_candidate_slot(
    batch: dict[str, np.ndarray],
    row_idx: int,
    candidate_ids: np.ndarray,
    item_features: dict[str, np.ndarray],
) -> None:
    batch["target_index"][row_idx] = 0
    batch["candidate_mask"][row_idx, : candidate_ids.shape[0]] = True
    batch["candidate_movie_id"][row_idx, : candidate_ids.shape[0]] = candidate_ids
    batch["candidate_genres"][row_idx, : candidate_ids.shape[0]] = item_features["genres"][candidate_ids]
    batch["candidate_isAdult"][row_idx, : candidate_ids.shape[0]] = item_features["isAdult"][candidate_ids]
    batch["candidate_startYear"][row_idx, : candidate_ids.shape[0]] = item_features["startYear"][candidate_ids]


def _numpy_batch_to_torch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError("torch is required for ranking training")
    result = {}
    for field, values in batch.items():
        if field == "candidate_mask":
            result[field] = torch.as_tensor(values, dtype=torch.bool)
        elif field == "target_index":
            result[field] = torch.as_tensor(values, dtype=torch.long)
        elif field in {"context_rating", "target_rating"}:
            result[field] = torch.as_tensor(values, dtype=torch.float32)
        else:
            result[field] = torch.as_tensor(values, dtype=torch.long)
    return result
