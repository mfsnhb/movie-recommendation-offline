from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder, SequenceFeatureEncoder, STATIC_USER_FIELDS, build_mlp


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


class AttentionPoolingUnit(nn.Module):
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
        activation_weight = activation_weight.masked_fill(~history_mask, torch.finfo(activation_weight.dtype).min)
        attention = torch.softmax(activation_weight, dim=-1).masked_fill(~history_mask, 0.0)
        return torch.matmul(attention.unsqueeze(2), history_embedding).squeeze(2)


class InterestExpertGate(nn.Module):
    def __init__(self, emb_dim: int, expert_count: int, dropout: float = 0.1):
        super().__init__()
        self.gate = build_mlp(emb_dim * 4, [emb_dim * 2], expert_count, dropout=dropout)

    def forward(self, candidate_embedding: torch.Tensor, user_profile: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([
            candidate_embedding,
            user_profile,
            candidate_embedding * user_profile,
            candidate_embedding - user_profile,
        ], dim=-1)
        logits = self.gate(gate_input)
        logits = logits.masked_fill(~expert_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1).masked_fill(~expert_mask, 0.0)


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
        self.sequence_encoder = SequenceFeatureEncoder(feature_dict, emb_dim, dropout=dropout)
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
        })
        history_embedding = self.sequence_encoder(history_embedding, batch["hist_rating"], batch["hist_time_gap_bucket"], batch.get("hist_feedback"))
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


class MoEDINModel(DINModel):
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
        super().__init__(
            feature_dict,
            emb_dim,
            dnn_hidden_dims=dnn_hidden_dims,
            attention_hidden_dims=attention_hidden_dims,
            dropout=dropout,
            multimodal_table=multimodal_table,
            top_m_history=top_m_history,
            force_recent_history=force_recent_history,
        )
        self.activation_unit = AttentionPoolingUnit(emb_dim, attention_hidden_dims, dropout)
        self.expert_gate = InterestExpertGate(emb_dim, expert_count=4, dropout=dropout)

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
        })
        history_embedding = self.sequence_encoder(history_embedding, batch["hist_rating"], batch["hist_time_gap_bucket"], batch.get("hist_feedback"))
        history_mask = history_movie_id.gt(0)
        user_profile_base = self.user_profile_projection(torch.cat([
            self.static_embeddings[field](batch[field].long())
            for field in STATIC_USER_FIELDS
        ], dim=-1))
        user_profile = user_profile_base.unsqueeze(1).expand(-1, candidate_size, -1)
        interest_embedding = self._moe_interest_embedding(candidate_embedding, history_embedding, history_mask, batch.get("hist_feedback"), user_profile)
        dnn_input = torch.cat([candidate_embedding, interest_embedding, user_profile], dim=-1)
        logits = self.dnn(dnn_input.reshape(batch_size * candidate_size, -1)).view(batch_size, candidate_size)
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        return logits

    def _moe_interest_embedding(
        self,
        candidate_embedding: torch.Tensor,
        history_embedding: torch.Tensor,
        history_mask: torch.Tensor,
        hist_feedback: torch.Tensor | None,
        user_profile: torch.Tensor,
    ) -> torch.Tensor:
        global_expert, global_valid = self._masked_mean(history_embedding, history_mask)
        recent_history, recent_mask = self._recent_history(history_embedding, history_mask, candidate_size=candidate_embedding.size(1))
        recent_expert = self.activation_unit(candidate_embedding, recent_history, recent_mask)
        recent_valid = recent_mask.any(dim=-1)

        positive_mask = history_mask if hist_feedback is None else history_mask & hist_feedback.long().eq(3)
        positive_expert, positive_valid_base = self._masked_mean(history_embedding, positive_mask)
        positive_expert = positive_expert.unsqueeze(1).expand_as(candidate_embedding)
        positive_valid = positive_valid_base.unsqueeze(1).expand(candidate_embedding.size(0), candidate_embedding.size(1))

        similar_history, similar_mask = self._candidate_topm_history(candidate_embedding, history_embedding, history_mask)
        similar_expert, similar_valid = self._candidate_masked_mean(similar_history, similar_mask)

        global_expert = global_expert.unsqueeze(1).expand_as(candidate_embedding)
        experts = torch.stack([global_expert, recent_expert, positive_expert, similar_expert], dim=2)
        expert_mask = torch.stack([
            global_valid.unsqueeze(1).expand_as(recent_valid),
            recent_valid,
            positive_valid,
            similar_valid,
        ], dim=-1)
        weights = self.expert_gate(candidate_embedding, user_profile, expert_mask).unsqueeze(-1)
        return (experts * weights).sum(dim=2)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = mask.unsqueeze(-1).to(dtype=values.dtype)
        counts = mask_float.sum(dim=1).clamp_min(1.0)
        pooled = (values * mask_float).sum(dim=1) / counts
        return pooled, mask.any(dim=1)

    def _candidate_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = mask.unsqueeze(-1).to(dtype=values.dtype)
        counts = mask_float.sum(dim=2).clamp_min(1.0)
        pooled = (values * mask_float).sum(dim=2) / counts
        return pooled, mask.any(dim=2)

    def _recent_history(self, history_embedding: torch.Tensor, history_mask: torch.Tensor, candidate_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        history_width = history_embedding.size(1)
        recent_count = min(max(self.force_recent_history, 1), history_width)
        emb_dim = history_embedding.size(-1)
        valid_counts = history_mask.long().sum(dim=1)
        positions = torch.arange(history_width, device=history_embedding.device).unsqueeze(0).expand_as(history_mask)
        recent_start = (valid_counts - recent_count).clamp_min(0).unsqueeze(1)
        recent_region = history_mask & positions.ge(recent_start)
        recent_order_keys = torch.where(recent_region, positions, positions - history_width)
        recent_indices = recent_order_keys.topk(k=recent_count, dim=1).indices.sort(dim=1).values
        recent_mask = history_mask.gather(1, recent_indices)
        recent_history = history_embedding.gather(1, recent_indices.unsqueeze(-1).expand(-1, -1, emb_dim))
        return recent_history.unsqueeze(1).expand(-1, candidate_size, -1, -1), recent_mask.unsqueeze(1).expand(-1, candidate_size, -1)
