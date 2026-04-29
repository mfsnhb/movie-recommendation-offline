from __future__ import annotations

import numpy as np
import torch
from torch import nn


STATIC_USER_FIELDS = ("user_id", "gender", "age", "occupation", "zip_code")


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GenreEmbeddingPooling(nn.Module):
    def __init__(self, genre_count: int, emb_dim: int):
        super().__init__()
        self.genre_emb = nn.Embedding(genre_count + 1, emb_dim, padding_idx=0)

    def forward(self, genres: torch.Tensor) -> torch.Tensor:
        genre_ids = torch.arange(1, genres.size(-1) + 1, device=genres.device)
        genre_embeddings = self.genre_emb(genre_ids).view(*([1] * (genres.ndim - 1)), genres.size(-1), -1)
        mask = genres.gt(0).unsqueeze(-1)
        summed = (genre_embeddings * mask).sum(dim=-2)
        counts = mask.sum(dim=-2).clamp_min(1)
        return summed / counts


class MovieFeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        output_norm: bool = False,
        multimodal_table: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__()
        self.output_norm = bool(output_norm)
        self.movie_emb = nn.Embedding(feature_dict["movie_id"], emb_dim, padding_idx=0)
        self.genre_pool = GenreEmbeddingPooling(max(int(feature_dict["genres"]) - 1, 1), emb_dim)
        self.is_adult_emb = nn.Embedding(feature_dict["isAdult"], emb_dim, padding_idx=0)
        self.start_year_emb = nn.Embedding(feature_dict["startYear"], emb_dim, padding_idx=0)
        self.popularity_emb = nn.Embedding(feature_dict["popularity"], emb_dim, padding_idx=0)
        self.average_rating_emb = nn.Embedding(feature_dict["averageRating"], emb_dim, padding_idx=0)
        self.multimodal_dim = int(feature_dict["multimodal_embedding_dim"])
        if multimodal_table is None:
            table = torch.zeros((int(feature_dict["movie_id"]), self.multimodal_dim), dtype=torch.float32)
        else:
            table = torch.as_tensor(multimodal_table, dtype=torch.float32)
        if table.ndim != 2 or table.size(1) != self.multimodal_dim:
            raise ValueError(f"multimodal_table must have shape [num_items, {self.multimodal_dim}]")
        self.register_buffer("multimodal_table", table, persistent=False)
        self.multimodal_projection = build_mlp(self.multimodal_dim, [emb_dim * 2], emb_dim, dropout=dropout)
        self.multimodal_gate = nn.Parameter(torch.tensor(-4.0))
        self.projection = build_mlp(emb_dim * 6, hidden_dims or [emb_dim * 2], emb_dim, dropout=dropout)

    def forward(self, item_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        structured_parts = [
            self.movie_emb(item_batch["movie_id"].long()),
            self.genre_pool(item_batch["genres"].long()),
            self.is_adult_emb(item_batch["isAdult"].long()),
            self.start_year_emb(item_batch["startYear"].long()),
            self.popularity_emb(item_batch["popularity"].long()),
            self.average_rating_emb(item_batch["averageRating"].long()),
        ]
        output = self.projection(torch.cat(structured_parts, dim=-1))
        multimodal = self.multimodal_table[item_batch["movie_id"].long().clamp(min=0, max=self.multimodal_table.size(0) - 1)]
        output = output + torch.sigmoid(self.multimodal_gate) * self.multimodal_projection(multimodal)
        if self.output_norm:
            output = torch.nn.functional.normalize(output, dim=-1)
        return output


class UserFeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        rating_weighting_enabled: bool = False,
        rating_weight_neutral: float = 3.0,
        rating_weight_scale: float = 0.25,
        rating_weight_min: float = 0.0,
        rating_weight_max: float = 1.0,
        short_history_length: int = 10,
        positive_rating_min: float = 4.0,
        output_norm: bool = False,
    ):
        super().__init__()
        self.rating_weighting_enabled = bool(rating_weighting_enabled)
        self.rating_weight_neutral = float(rating_weight_neutral)
        self.rating_weight_scale = float(rating_weight_scale)
        self.rating_weight_min = float(rating_weight_min)
        self.rating_weight_max = float(rating_weight_max)
        self.short_history_length = int(short_history_length)
        self.positive_rating_min = float(positive_rating_min)
        self.output_norm = bool(output_norm)
        self.static_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in STATIC_USER_FIELDS})
        self.recency_emb = nn.Embedding(feature_dict.get("hist_recency_bucket", 2), emb_dim, padding_idx=0)
        self.projection = build_mlp(emb_dim * 9, hidden_dims or [emb_dim * 4, emb_dim * 2], emb_dim, dropout=dropout)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        history_movie_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        hist_movie_id = batch["hist_movie_id"] if "hist_movie_id" in batch else batch["context_movie_id"]
        hist_recency_bucket = batch.get("hist_recency_bucket")
        if hist_recency_bucket is None:
            hist_recency_bucket = torch.ones_like(hist_movie_id)
        hist_rating = batch.get("hist_rating", batch.get("context_rating"))

        rating_weights = self._rating_weights(hist_rating, history_mask)
        recency_weights = self._recency_weights(hist_recency_bucket, history_mask)
        combined_weights = rating_weights * recency_weights

        pooled_history = self._weighted_pool(history_movie_embeddings, combined_weights, history_mask)
        recency_embeddings = self.recency_emb(hist_recency_bucket.long())
        pooled_recency = self._weighted_pool(recency_embeddings, combined_weights, history_mask)

        short_embeddings = self._short_tail(history_movie_embeddings)
        short_mask = self._short_tail(history_mask)
        short_weights = self._short_tail(combined_weights)
        pooled_short_history = self._weighted_pool(short_embeddings, short_weights, short_mask)

        if hist_rating is None:
            positive_mask = history_mask
        else:
            positive_mask = history_mask & hist_rating.float().ge(self.positive_rating_min)
        pooled_positive_history = self._weighted_pool(history_movie_embeddings, recency_weights * positive_mask.float(), positive_mask)

        output = self.projection(torch.cat([
            self.static_embeddings["user_id"](batch["user_id"].long()),
            self.static_embeddings["gender"](batch["gender"].long()),
            self.static_embeddings["age"](batch["age"].long()),
            self.static_embeddings["occupation"](batch["occupation"].long()),
            self.static_embeddings["zip_code"](batch["zip_code"].long()),
            pooled_history,
            pooled_recency,
            pooled_short_history,
            pooled_positive_history,
        ], dim=-1))
        if self.output_norm:
            output = torch.nn.functional.normalize(output, dim=-1)
        return output

    def _rating_weights(self, hist_rating: torch.Tensor | None, history_mask: torch.Tensor) -> torch.Tensor:
        if hist_rating is None or not self.rating_weighting_enabled:
            return history_mask.float()
        weights = 1.0 + self.rating_weight_scale * (hist_rating.float() - self.rating_weight_neutral)
        weights = weights.clamp(min=self.rating_weight_min, max=self.rating_weight_max)
        return weights * history_mask.float()

    @staticmethod
    def _recency_weights(hist_recency_bucket: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        return (1.0 / hist_recency_bucket.clamp_min(1).float()) * history_mask.float()

    def _short_tail(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.short_history_length <= 0:
            return tensor
        return tensor[:, -self.short_history_length :]

    @staticmethod
    def _weighted_pool(embeddings: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted_mask = weights.unsqueeze(-1).float() * mask.unsqueeze(-1).float()
        denom = weighted_mask.sum(dim=1).clamp_min(1e-9)
        return (embeddings * weighted_mask).sum(dim=1) / denom
