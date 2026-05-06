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
        activation_input = torch.cat([
            candidate_expanded.expand(-1, -1, history_embedding.size(2), -1),
            history_embedding,
            candidate_expanded - history_embedding,
            candidate_expanded * history_embedding,
        ], dim=-1)
        activation_weight = self.activation(activation_input).squeeze(-1)
        activation_weight = activation_weight.masked_fill(~history_mask, 0.0)
        return torch.matmul(activation_weight.unsqueeze(2), history_embedding).squeeze(2)


class DINModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        dnn_hidden_dims: list[int] | None = None,
        attention_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        multimodal_table=None,
        top_m_history: int = 32,
    ):
        super().__init__()
        self.top_m_history = int(top_m_history)
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
        attention_history, attention_mask = self._candidate_topm_history(candidate_embedding, history_embedding, history_mask)

        activated_history = self.activation_unit(candidate_embedding, attention_history, attention_mask)
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

    def _candidate_topm_history(
        self,
        candidate_embedding: torch.Tensor,
        history_embedding: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_width = history_embedding.size(1)
        top_m = history_width if self.top_m_history <= 0 else min(self.top_m_history, history_width)
        similarity = torch.einsum("bce,bhe->bch", candidate_embedding, history_embedding)
        similarity = similarity.masked_fill(~history_mask.unsqueeze(1), torch.finfo(similarity.dtype).min)
        _, top_indices = torch.topk(similarity, k=top_m, dim=2)
        gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, history_embedding.size(-1))
        expanded_history = history_embedding.unsqueeze(1).expand(-1, candidate_embedding.size(1), -1, -1)
        selected_history = expanded_history.gather(2, gather_indices)
        selected_mask = history_mask.unsqueeze(1).expand(-1, candidate_embedding.size(1), -1).gather(2, top_indices)
        return selected_history, selected_mask

    @staticmethod
    def _weighted_pool(embeddings: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted_mask = weights.unsqueeze(-1).float() * mask.unsqueeze(-1).float()
        denom = weighted_mask.sum(dim=1).clamp_min(1e-9)
        return (embeddings * weighted_mask).sum(dim=1) / denom
