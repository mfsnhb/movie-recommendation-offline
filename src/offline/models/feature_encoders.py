from __future__ import annotations

import numpy as np
import torch
from torch import nn

from offline.ranking.protocol import SPARSE_ITEM_FEATURE_FIELDS


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
        item_feature_table: dict[str, np.ndarray | torch.Tensor] | None = None,
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
        self.multimodal_gate = nn.Parameter(torch.tensor(-2.0))
        self.projection = build_mlp(emb_dim * 6, hidden_dims or [emb_dim * 2], emb_dim, dropout=dropout)
        self.interaction_rating_emb = nn.Embedding(6, emb_dim, padding_idx=0)
        self.interaction_projection = build_mlp(emb_dim * 2, [emb_dim * 2], emb_dim, dropout=dropout)
        self.has_item_feature_table = item_feature_table is not None
        if item_feature_table is not None:
            for field in SPARSE_ITEM_FEATURE_FIELDS:
                values = torch.as_tensor(item_feature_table[field], dtype=torch.long)
                self.register_buffer(f"item_feature_{field}", values, persistent=False)

    def lookup_features(self, movie_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.has_item_feature_table:
            raise ValueError("item_feature_table is required for movie_id lookup")
        safe_ids = movie_ids.long().clamp(min=0, max=getattr(self, "item_feature_genres").size(0) - 1)
        return {
            "movie_id": movie_ids.long(),
            "genres": getattr(self, "item_feature_genres")[safe_ids],
            "isAdult": getattr(self, "item_feature_isAdult")[safe_ids],
            "startYear": getattr(self, "item_feature_startYear")[safe_ids],
            "popularity": getattr(self, "item_feature_popularity")[safe_ids],
            "averageRating": getattr(self, "item_feature_averageRating")[safe_ids],
        }

    def forward(self, item_batch: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(item_batch, torch.Tensor):
            item_batch = self.lookup_features(item_batch)
        movie_ids = item_batch["movie_id"].long()
        structured_parts = [
            self.movie_emb(movie_ids),
            self.genre_pool(item_batch["genres"].long()),
            self.is_adult_emb(item_batch["isAdult"].long()),
            self.start_year_emb(item_batch["startYear"].long()),
            self.popularity_emb(item_batch["popularity"].long()),
            self.average_rating_emb(item_batch["averageRating"].long()),
        ]
        output = self.projection(torch.cat(structured_parts, dim=-1))
        multimodal = self.multimodal_table[movie_ids.clamp(min=0, max=self.multimodal_table.size(0) - 1)]
        output = output + torch.sigmoid(self.multimodal_gate) * self.multimodal_projection(multimodal)
        interaction_rating = item_batch.get("interaction_rating", item_batch.get("rating"))
        if interaction_rating is not None:
            rating_ids = interaction_rating.float().round().long().clamp(min=0, max=5)
            output = self.interaction_projection(torch.cat([output, self.interaction_rating_emb(rating_ids)], dim=-1))
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
        output_norm: bool = False,
    ):
        super().__init__()
        self.output_norm = bool(output_norm)
        self.static_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in STATIC_USER_FIELDS})
        self.projection = build_mlp(emb_dim * len(STATIC_USER_FIELDS), hidden_dims or [emb_dim * 2], emb_dim, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.projection(torch.cat([
            self.static_embeddings[field](batch[field].long())
            for field in STATIC_USER_FIELDS
        ], dim=-1))
        if self.output_norm:
            output = torch.nn.functional.normalize(output, dim=-1)
        return output
