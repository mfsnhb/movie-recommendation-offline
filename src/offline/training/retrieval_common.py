from __future__ import annotations

import numpy as np
import torch

from offline.features.item_batch import ITEM_FEATURE_FIELDS
from offline.ranking.protocol import get_all_item_ids, get_item_feature_arrays
from offline.utils.io import (
    ITEM_CATALOG_PATH,
    RETRIEVAL_FEATURE_DICT_PATH,
    RETRIEVAL_SAMPLE_PATH,
    RETRIEVAL_VOCAB_DICT_PATH,
    load_pickle,
)


def batch_indices(indices, batch_size: int, *, shuffle: bool) -> list[np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    if shuffle:
        index_array = index_array[np.random.permutation(index_array.shape[0])]
    return [index_array[start : start + batch_size] for start in range(0, index_array.shape[0], batch_size)]


def slice_train_batch(train_data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    tensor_specs: dict[str, torch.dtype] = {
        "user_id": torch.long,
        "age": torch.long,
        "gender": torch.long,
        "occupation": torch.long,
        "zip_code": torch.long,
        "hist_movie_id": torch.long,
        "hist_recency_bucket": torch.long,
        "hist_rating": torch.float32,
        "movie_id": torch.long,
        "rating": torch.float32,
    }
    batch: dict[str, torch.Tensor] = {}
    for field, dtype in tensor_specs.items():
        if field not in train_data:
            continue
        batch[field] = torch.from_numpy(np.asarray(train_data[field])[indices]).to(device=device, dtype=dtype)
    return batch


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().data.norm(2).item())
        total += grad_norm * grad_norm
    return total ** 0.5


def load_retrieval_context(settings: dict, config: dict) -> dict:
    train_eval_samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
    feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
    vocab_dict = load_pickle(RETRIEVAL_VOCAB_DICT_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    item_feature_arrays = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog).astype(np.int64)
    missing_fields = [field for field in ITEM_FEATURE_FIELDS[1:] if field not in item_feature_arrays]
    if missing_fields:
        raise ValueError(
            f"Global item catalog is missing item feature fields: {missing_fields}. "
            f"Run ranking preprocessing to regenerate {ITEM_CATALOG_PATH.name}."
        )
    train_data = train_eval_samples["train"]
    required_train_fields = [
        "user_id",
        "hist_movie_id",
        "hist_recency_bucket",
        "hist_rating",
        "movie_id",
        "rating",
    ]
    missing_train_fields = [field for field in required_train_fields if field not in train_data]
    if missing_train_fields:
        raise ValueError(
            "Retrieval training artifacts use an outdated schema. "
            f"Missing fields: {missing_train_fields}. Run retrieval preprocessing again."
        )
    if np.asarray(train_data["hist_movie_id"]).shape != np.asarray(train_data["hist_rating"]).shape:
        raise ValueError(f"Retrieval training artifacts have misaligned hist_movie_id/hist_rating shapes. Regenerate {RETRIEVAL_SAMPLE_PATH.name}.")
    sample_rating_semantics = train_eval_samples.get("rating_semantics", {})
    positive_rating_min = float(config.get("rating_semantics", {}).get("positive_rating_min", 3.0))
    eval_positive_rating_min = float(
        sample_rating_semantics.get(
            "eval_positive_rating_min",
            config.get("rating_semantics", {}).get("eval_positive_rating_min", 4.0),
        )
    )
    validate_min_target_rating(
        train_eval_samples["train"],
        rating_field="rating",
        min_rating=positive_rating_min,
        stage_name="Retrieval train",
        artifact_name=RETRIEVAL_SAMPLE_PATH.name,
    )
    for split_name in ("validation", "test"):
        validate_min_target_rating(
            train_eval_samples[split_name],
            rating_field="rating",
            min_rating=eval_positive_rating_min,
            stage_name=f"Retrieval {split_name}",
            artifact_name=RETRIEVAL_SAMPLE_PATH.name,
        )
    return {
        "settings": settings,
        "resolved_config": config,
        "train_eval_samples": train_eval_samples,
        "feature_dict": feature_dict,
        "vocab_dict": vocab_dict,
        "item_catalog": item_catalog,
        "item_feature_arrays": item_feature_arrays,
        "all_item_ids": all_item_ids,
        "train_data": train_data,
        "validation_data": train_eval_samples["validation"],
        "test_data": train_eval_samples["test"],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }


def validate_min_target_rating(
    split_data: dict[str, np.ndarray],
    rating_field: str,
    min_rating: float,
    stage_name: str,
    artifact_name: str,
) -> None:
    if rating_field not in split_data:
        raise ValueError(f"{stage_name} samples are missing '{rating_field}'. Regenerate {artifact_name}.")
    ratings = np.asarray(split_data[rating_field], dtype=np.float32)
    min_seen = float(ratings.min()) if ratings.size else float("nan")
    if min_seen >= float(min_rating):
        return
    bad_count = int(np.count_nonzero(ratings < float(min_rating)))
    raise ValueError(
        f"{stage_name} samples contain {bad_count} targets below rating {min_rating}. "
        f"min_seen={min_seen}. Regenerate {artifact_name} with the current preprocessing config."
    )
