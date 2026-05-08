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
        force_recent_history: int = 0,
    ):
        super().__init__()
        self.top_m_history = int(top_m_history)
        self.force_recent_history = int(force_recent_history)
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False, multimodal_table=multimodal_table)
        self.static_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in STATIC_USER_FIELDS})
        self.user_profile_projection = build_mlp(emb_dim * len(STATIC_USER_FIELDS), [emb_dim * 2], emb_dim, dropout=dropout)
        self.activation_unit = LocalActivationUnit(emb_dim, attention_hidden_dims, dropout)
        self.dnn = build_mlp(emb_dim * 3, dnn_hidden_dims or [256, 128], 1, dropout=dropout)

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
        history_movie_id = batch["hist_movie_id"]
        history_embedding = self.movie_encoder({
            "movie_id": history_movie_id,
            "genres": batch["hist_genres"],
            "isAdult": batch["hist_isAdult"],
            "startYear": batch["hist_startYear"],
            "popularity": batch["hist_popularity"],
            "averageRating": batch["hist_averageRating"],
            "interaction_rating": batch["hist_rating"],
            "interaction_time_gap_bucket": batch["hist_time_gap_bucket"],
        })
        history_mask = history_movie_id.gt(0)
        attention_history, attention_mask = self._candidate_topm_history(candidate_embedding, history_embedding, history_mask)

        activated_history = self.activation_unit(candidate_embedding, attention_history, attention_mask)
        user_profile = self.user_profile_projection(torch.cat([
            self.static_embeddings[field](batch[field].long())
            for field in STATIC_USER_FIELDS
        ], dim=-1))
        user_profile = user_profile.unsqueeze(1).expand(-1, candidate_size, -1)
        dnn_input = torch.cat([candidate_embedding, activated_history, user_profile], dim=-1)
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
        recent_count = min(max(self.force_recent_history, 0), top_m)
        candidate_count = candidate_embedding.size(1)
        emb_dim = history_embedding.size(-1)
        expanded_history = history_embedding.unsqueeze(1).expand(-1, candidate_count, -1, -1)
        expanded_mask = history_mask.unsqueeze(1).expand(-1, candidate_count, -1)

        if recent_count == 0:
            similarity = torch.einsum("bce,bhe->bch", candidate_embedding, history_embedding)
            similarity = similarity.masked_fill(~history_mask.unsqueeze(1), torch.finfo(similarity.dtype).min)
            _, top_indices = torch.topk(similarity, k=top_m, dim=2)
            gather_indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, emb_dim)
            selected_history = expanded_history.gather(2, gather_indices)
            selected_mask = expanded_mask.gather(2, top_indices)
            return selected_history, selected_mask

        valid_counts = history_mask.long().sum(dim=1)
        positions = torch.arange(history_width, device=history_embedding.device).unsqueeze(0).expand_as(history_mask)
        recent_start = (valid_counts - recent_count).clamp_min(0).unsqueeze(1)
        recent_region = history_mask & positions.ge(recent_start)
        recent_order_keys = torch.where(recent_region, positions, positions - history_width)
        recent_indices = recent_order_keys.topk(k=recent_count, dim=1).indices.sort(dim=1).values
        recent_mask = history_mask.gather(1, recent_indices)
        recent_history = history_embedding.gather(1, recent_indices.unsqueeze(-1).expand(-1, -1, emb_dim))
        recent_history = recent_history.unsqueeze(1).expand(-1, candidate_count, -1, -1)
        recent_mask = recent_mask.unsqueeze(1).expand(-1, candidate_count, -1)

        semantic_count = top_m - recent_count
        if semantic_count <= 0:
            return recent_history, recent_mask

        semantic_source_mask = history_mask & ~recent_region
        similarity = torch.einsum("bce,bhe->bch", candidate_embedding, history_embedding)
        similarity = similarity.masked_fill(~semantic_source_mask.unsqueeze(1), torch.finfo(similarity.dtype).min)
        _, semantic_indices = torch.topk(similarity, k=semantic_count, dim=2)
        semantic_history = expanded_history.gather(2, semantic_indices.unsqueeze(-1).expand(-1, -1, -1, emb_dim))
        semantic_mask = expanded_mask.gather(2, semantic_indices) & semantic_source_mask.unsqueeze(1).expand(-1, candidate_count, -1).gather(2, semantic_indices)
        selected_history = torch.cat([recent_history, semantic_history], dim=2)
        selected_mask = torch.cat([recent_mask, semantic_mask], dim=2)
        return selected_history, selected_mask

