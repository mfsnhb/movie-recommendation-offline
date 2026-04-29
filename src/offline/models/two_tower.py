from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, UserFeatureEncoder


class TwoTowerRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        user_hidden_dims: list[int] | None = None,
        item_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        rating_weighting_enabled: bool = False,
        rating_weight_neutral: float = 3.0,
        rating_weight_scale: float = 0.25,
        rating_weight_min: float = 0.0,
        rating_weight_max: float = 1.0,
        short_history_length: int = 10,
        positive_rating_min: float = 4.0,
        multimodal_table=None,
    ):
        super().__init__()
        self.movie_encoder = MovieFeatureEncoder(
            feature_dict,
            emb_dim,
            hidden_dims=item_hidden_dims,
            dropout=dropout,
            output_norm=False,
            multimodal_table=multimodal_table,
        )
        self.user_encoder = UserFeatureEncoder(
            feature_dict,
            emb_dim,
            hidden_dims=user_hidden_dims,
            dropout=dropout,
            rating_weighting_enabled=rating_weighting_enabled,
            rating_weight_neutral=rating_weight_neutral,
            rating_weight_scale=rating_weight_scale,
            rating_weight_min=rating_weight_min,
            rating_weight_max=rating_weight_max,
            short_history_length=short_history_length,
            positive_rating_min=positive_rating_min,
            output_norm=False,
        )

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_movie_embeddings = self.movie_encoder({
            "movie_id": batch["hist_movie_id"],
            "genres": batch["hist_genres"],
            "isAdult": batch["hist_isAdult"],
            "startYear": batch["hist_startYear"],
            "popularity": batch["hist_popularity"],
            "averageRating": batch["hist_averageRating"],
        })
        history_mask = batch["hist_movie_id"].gt(0)
        user_repr = self.user_encoder(batch, history_movie_embeddings, history_mask)
        return torch.nn.functional.normalize(user_repr, dim=1)

    def encode_item(self, item_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_repr = self.movie_encoder(item_batch)
        return torch.nn.functional.normalize(item_repr, dim=1)
