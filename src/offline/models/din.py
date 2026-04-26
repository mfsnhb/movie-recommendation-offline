from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, UserFeatureEncoder, build_mlp


class AttentionPooling(nn.Module):
    def __init__(self, emb_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        self.attention = build_mlp(emb_dim * 4, hidden_dims or [128, 64], 1, dropout=dropout)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(2)
        keys = keys.unsqueeze(1)
        attention_input = torch.cat([
            query.expand(-1, -1, keys.size(2), -1),
            keys.expand(-1, query.size(1), -1, -1),
            query - keys,
            query * keys,
        ], dim=-1)
        scores = self.attention(attention_input).squeeze(-1)
        expanded_mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2) * expanded_mask.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-9)
        return torch.matmul(weights, keys.squeeze(1))


class DINModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        fields: list[str],
        emb_dim: int,
        dnn_hidden_dims: list[int] | None = None,
        attention_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        del fields
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False)
        self.user_encoder = UserFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False)
        self.attention = AttentionPooling(emb_dim, attention_hidden_dims, dropout)
        self.interaction = build_mlp(emb_dim * 4, [emb_dim * 2], emb_dim, dropout=dropout)
        self.dnn = build_mlp(emb_dim * 4, dnn_hidden_dims or [256, 128], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_movie_id = batch["candidate_movie_id"]
        batch_size, candidate_size = candidate_movie_id.shape
        candidate_movie = self.movie_encoder({
            "movie_id": candidate_movie_id,
            "genres": batch["candidate_genres"],
            "isAdult": batch["candidate_isAdult"],
            "startYear": batch["candidate_startYear"],
            "popularity": batch["candidate_popularity"],
        })
        context_movie_id = batch["context_movie_id"]
        context_movie = self.movie_encoder({
            "movie_id": context_movie_id,
            "genres": batch["context_genres"],
            "isAdult": batch["context_isAdult"],
            "startYear": batch["context_startYear"],
            "popularity": batch["context_popularity"],
        })
        history_mask = context_movie_id.gt(0)
        if "context_low_rating_mask" in batch:
            history_mask = history_mask & ~batch["context_low_rating_mask"].gt(0)
        attended_history = self.attention(candidate_movie, context_movie, history_mask)
        user_embedding = self.user_encoder(batch, context_movie, history_mask)
        user_vector = user_embedding.unsqueeze(1).expand(-1, candidate_size, -1)
        interaction = self.interaction(torch.cat([
            candidate_movie,
            attended_history,
            candidate_movie - attended_history,
            candidate_movie * attended_history,
        ], dim=-1))
        dnn_input = torch.cat([candidate_movie, attended_history, interaction, user_vector], dim=-1)
        logits = self.dnn(dnn_input.reshape(batch_size * candidate_size, -1)).view(batch_size, candidate_size)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        return logits
