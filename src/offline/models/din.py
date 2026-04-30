from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, STATIC_USER_FIELDS, build_mlp


class LocalActivationUnit(nn.Module):
    def __init__(self, emb_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        self.activation = build_mlp(emb_dim * 4, hidden_dims or [128, 64], 1, dropout=dropout)

    def forward(self, candidate_embedding: torch.Tensor, history_embedding: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        candidate_expanded = candidate_embedding.unsqueeze(2)
        history_expanded = history_embedding.unsqueeze(1)
        activation_input = torch.cat([
            candidate_expanded.expand(-1, -1, history_embedding.size(1), -1),
            history_expanded.expand(-1, candidate_embedding.size(1), -1, -1),
            candidate_expanded - history_expanded,
            candidate_expanded * history_expanded,
        ], dim=-1)
        activation_weight = self.activation(activation_input).squeeze(-1)
        activation_weight = activation_weight.masked_fill(~history_mask.unsqueeze(1), 0.0)
        return torch.matmul(activation_weight, history_embedding)


class DINModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        fields: list[str],
        emb_dim: int,
        dnn_hidden_dims: list[int] | None = None,
        attention_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        multimodal_table=None,
        positive_rating_min: float = 4.0,
        negative_rating_max: float = 2.0,
    ):
        super().__init__()
        del fields, positive_rating_min, negative_rating_max
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False, multimodal_table=multimodal_table)
        self.static_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in STATIC_USER_FIELDS})
        self.user_profile_projection = build_mlp(emb_dim * len(STATIC_USER_FIELDS), [emb_dim * 2], emb_dim, dropout=dropout)
        self.history_rating_projection = nn.Linear(1, emb_dim)
        self.history_recency_embedding = nn.Embedding(feature_dict.get("recency_bucket", 21), emb_dim, padding_idx=0)
        self.activation_unit = LocalActivationUnit(emb_dim, attention_hidden_dims, dropout)
        self.dnn = build_mlp(emb_dim * 4, dnn_hidden_dims or [256, 128], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_movie_id = batch["candidate_movie_id"]
        batch_size, candidate_size = candidate_movie_id.shape
        candidate_embedding = self.movie_encoder({
            "movie_id": candidate_movie_id,
            "genres": batch["candidate_genres"],
            "isAdult": batch["candidate_isAdult"],
            "startYear": batch["candidate_startYear"],
            "popularity": batch["candidate_popularity"],
            "averageRating": batch["candidate_averageRating"],
        })
        history_movie_id = batch["context_movie_id"]
        history_movie_embedding = self.movie_encoder({
            "movie_id": history_movie_id,
            "genres": batch["context_genres"],
            "isAdult": batch["context_isAdult"],
            "startYear": batch["context_startYear"],
            "popularity": batch["context_popularity"],
            "averageRating": batch["context_averageRating"],
        })
        history_mask = history_movie_id.gt(0)
        history_rating = batch.get("context_rating", torch.zeros_like(history_movie_id, dtype=history_movie_embedding.dtype)).float()
        history_rating_embedding = self.history_rating_projection((history_rating / 5.0).unsqueeze(-1)).masked_fill(~history_mask.unsqueeze(-1), 0.0)
        history_recency_bucket = batch.get("context_recency_bucket", torch.zeros_like(history_movie_id)).long()
        history_recency_embedding = self.history_recency_embedding(
            history_recency_bucket.clamp(min=0, max=self.history_recency_embedding.num_embeddings - 1)
        ).masked_fill(~history_mask.unsqueeze(-1), 0.0)
        history_embedding = history_movie_embedding + history_rating_embedding + history_recency_embedding

        activated_history = self.activation_unit(candidate_embedding, history_embedding, history_mask)
        global_history_context = self._weighted_pool(history_embedding, history_mask.float(), history_mask)
        global_recency_context = self._weighted_pool(history_recency_embedding, 1.0 / history_recency_bucket.clamp_min(1).float(), history_mask)
        context_embedding = global_history_context + global_recency_context
        user_profile = self.user_profile_projection(torch.cat([
            self.static_embeddings[field](batch[field].long())
            for field in STATIC_USER_FIELDS
        ], dim=-1))
        user_profile = user_profile.unsqueeze(1).expand(-1, candidate_size, -1)
        context_embedding = context_embedding.unsqueeze(1).expand(-1, candidate_size, -1)
        dnn_input = torch.cat([candidate_embedding, activated_history, user_profile, context_embedding], dim=-1)
        logits = self.dnn(dnn_input.reshape(batch_size * candidate_size, -1)).view(batch_size, candidate_size)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        return logits

    @staticmethod
    def _weighted_pool(embeddings: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted_mask = weights.unsqueeze(-1).float() * mask.unsqueeze(-1).float()
        denom = weighted_mask.sum(dim=1).clamp_min(1e-9)
        return (embeddings * weighted_mask).sum(dim=1) / denom
