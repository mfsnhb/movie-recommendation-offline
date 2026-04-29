from __future__ import annotations

import numpy as np
import torch

from offline.ranking.protocol import SPARSE_ITEM_FEATURE_FIELDS


ITEM_ID_FIELD = "movie_id"
ITEM_FEATURE_FIELDS = [ITEM_ID_FIELD, *SPARSE_ITEM_FEATURE_FIELDS, "multimodal_embedding"]


def item_feature_tensors(item_features: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        field: torch.as_tensor(item_features[field], dtype=torch.long, device=device)
        for field in SPARSE_ITEM_FEATURE_FIELDS
    }


def build_item_batch(
    item_ids: torch.Tensor,
    item_features: dict[str, torch.Tensor] | dict[str, np.ndarray],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    target_device = device or item_ids.device
    item_ids = item_ids.to(device=target_device, dtype=torch.long)
    if isinstance(next(iter(item_features.values())), torch.Tensor):
        batch = {ITEM_ID_FIELD: item_ids}
        for field in SPARSE_ITEM_FEATURE_FIELDS:
            batch[field] = item_features[field][item_ids]
        return batch

    movie_ids = item_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    batch = {ITEM_ID_FIELD: item_ids}
    for field in SPARSE_ITEM_FEATURE_FIELDS:
        batch[field] = torch.as_tensor(item_features[field][movie_ids], dtype=torch.long, device=target_device)
    return batch
