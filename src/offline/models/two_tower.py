from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, UserFeatureEncoder, build_mlp


class TwoTowerRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        user_hidden_dims: list[int] | None = None,
        item_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        multimodal_table=None,
        item_feature_table=None,
        recent_history_length: int = 20,
    ):
        super().__init__()
        self.movie_encoder = MovieFeatureEncoder(
            feature_dict,
            emb_dim,
            hidden_dims=item_hidden_dims,
            dropout=dropout,
            output_norm=False,
            multimodal_table=multimodal_table,
            item_feature_table=item_feature_table,
        )
        self.user_encoder = UserFeatureEncoder(
            feature_dict,
            emb_dim,
            dropout=dropout,
            output_norm=False,
        )
        self.recent_history_length = int(recent_history_length)
        self.user_projection = build_mlp(emb_dim * 3, user_hidden_dims or [emb_dim * 4, emb_dim * 2], emb_dim, dropout=dropout)

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_embedding = self.movie_encoder({
            "movie_id": batch["hist_movie_id"],
            "interaction_rating": batch["hist_rating"],
            "interaction_time_gap_bucket": batch["hist_time_gap_bucket"],
        })
        history_mask = batch["hist_movie_id"].gt(0)
        static_user = self.user_encoder(batch)
        pooled_history = self._mean_pool(history_embedding, history_mask)
        pooled_recent = self._mean_pool(history_embedding, self._recent_mask(batch["hist_movie_id"], history_mask))
        user_repr = self.user_projection(torch.cat([static_user, pooled_history, pooled_recent], dim=-1))
        return torch.nn.functional.normalize(user_repr, dim=1)

    @staticmethod
    def _mean_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)

    def _recent_mask(self, hist_movie_id: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if self.recent_history_length <= 0:
            return history_mask.new_zeros(history_mask.shape)
        valid_seen = history_mask.long().cumsum(dim=1)
        valid_count = history_mask.long().sum(dim=1, keepdim=True)
        return hist_movie_id.gt(0) & history_mask & valid_seen.gt((valid_count - self.recent_history_length).clamp_min(0))

    def encode_item(self, item_ids: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        item_repr = self.movie_encoder(item_ids)
        return torch.nn.functional.normalize(item_repr, dim=1)
