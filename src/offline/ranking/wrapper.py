from __future__ import annotations

import torch
from torch import nn


class PointwiseCandidateRanker(nn.Module):
    def __init__(self, scorer: nn.Module, static_fields: list[str]):
        super().__init__()
        self.scorer = scorer
        self.static_fields = static_fields

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pointwise_batch = self._to_pointwise_batch(batch)
        logits = self.scorer(pointwise_batch)
        batch_size, candidate_size = batch["candidate_movie_id"].shape
        logits = logits.view(batch_size, candidate_size)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        return logits

    def _to_pointwise_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        candidate_movie_id = batch["candidate_movie_id"]
        batch_size, candidate_size = candidate_movie_id.shape
        context_movie_id = batch["context_movie_id"]
        context_genres = batch["context_genres"]
        context_rating = batch.get("context_rating")
        context_low_rating_mask = batch.get("context_low_rating_mask")
        context_width = context_movie_id.size(1)

        hist_movie_id = context_movie_id.unsqueeze(1).expand(-1, candidate_size, -1)
        hist_genres = context_genres.unsqueeze(1).expand(-1, candidate_size, -1)

        pointwise_batch = {
            "movie_id": candidate_movie_id.reshape(-1),
            "genres": batch["candidate_genres"].reshape(-1),
            "isAdult": batch["candidate_isAdult"].reshape(-1),
            "startYear": batch["candidate_startYear"].reshape(-1),
            "hist_movie_id": hist_movie_id.reshape(-1, context_width),
            "hist_genres": hist_genres.reshape(-1, context_width),
        }
        if context_rating is not None:
            hist_rating = context_rating.unsqueeze(1).expand(-1, candidate_size, -1)
            pointwise_batch["hist_rating"] = hist_rating.reshape(-1, context_width)
        if context_low_rating_mask is not None:
            hist_low_rating_mask = context_low_rating_mask.unsqueeze(1).expand(-1, candidate_size, -1)
            pointwise_batch["hist_low_rating_mask"] = hist_low_rating_mask.reshape(-1, context_width)
        for field in self.static_fields:
            pointwise_batch[field] = batch[field].view(batch_size, 1).expand(-1, candidate_size).reshape(-1)
        return pointwise_batch


PointwiseSequenceRanker = PointwiseCandidateRanker

__all__ = ["PointwiseCandidateRanker", "PointwiseSequenceRanker"]
