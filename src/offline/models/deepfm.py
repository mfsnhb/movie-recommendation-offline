from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, UserFeatureEncoder, build_mlp


class DeepFMModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        fields: list[str],
        emb_dim: int,
        dnn_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        history_fields: list[str] | None = None,
        multimodal_table=None,
    ):
        super().__init__()
        del fields, history_fields
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False, multimodal_table=multimodal_table)
        self.user_encoder = UserFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False)
        self.movie_linear = nn.Embedding(feature_dict["movie_id"], 1, padding_idx=0)
        self.dnn = build_mlp(emb_dim * 3, dnn_hidden_dims or [128, 64], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_movie_id = batch["candidate_movie_id"]
        batch_size, candidate_size = candidate_movie_id.shape
        candidate_movie = self.movie_encoder({
            "movie_id": candidate_movie_id,
            "genres": batch["candidate_genres"],
            "isAdult": batch["candidate_isAdult"],
            "startYear": batch["candidate_startYear"],
            "popularity": batch["candidate_popularity"],
            "averageRating": batch["candidate_averageRating"],
        })
        context_movie = self.movie_encoder({
            "movie_id": batch["context_movie_id"],
            "genres": batch["context_genres"],
            "isAdult": batch["context_isAdult"],
            "startYear": batch["context_startYear"],
            "popularity": batch["context_popularity"],
            "averageRating": batch["context_averageRating"],
        })
        history_mask = batch["context_movie_id"].gt(0)
        if "context_low_rating_mask" in batch:
            history_mask = history_mask & ~batch["context_low_rating_mask"].gt(0)
        history_weights = history_mask.unsqueeze(-1).float()
        pooled_history = (context_movie * history_weights).sum(dim=1) / history_weights.sum(dim=1).clamp_min(1e-9)
        user_embedding = self.user_encoder(batch, context_movie, history_mask)
        pooled_history = pooled_history.unsqueeze(1).expand(-1, candidate_size, -1)
        user_vector = user_embedding.unsqueeze(1).expand(-1, candidate_size, -1)
        fm_vectors = torch.stack([candidate_movie, pooled_history, user_vector], dim=2)
        summed = fm_vectors.sum(dim=2)
        fm_logit = 0.5 * ((summed * summed) - (fm_vectors * fm_vectors).sum(dim=2)).sum(dim=2)
        linear_logit = self.movie_linear(candidate_movie_id).squeeze(-1)
        dnn_input = torch.cat([candidate_movie, pooled_history, user_vector], dim=-1)
        dnn_logit = self.dnn(dnn_input.reshape(batch_size * candidate_size, -1)).view(batch_size, candidate_size)
        logits = linear_logit + fm_logit + dnn_logit
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        return logits
