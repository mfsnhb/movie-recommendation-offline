from __future__ import annotations

import math

import torch
from torch import nn


class SASRecBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, ffn_dim: int, dropout: float, ffn_activation: str = "relu"):
        super().__init__()
        activation = nn.GELU() if str(ffn_activation).strip().lower() == "gelu" else nn.ReLU()
        self.attn_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        normed = self.attn_norm(x)
        attn_output, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = residual + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SequenceRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        num_heads: int = 2,
        num_layers: int = 2,
        ffn_dim: int = 128,
        max_len: int = 10,
        dropout: float = 0.1,
        use_recency_embedding: bool = False,
        use_feedback_embedding: bool = False,
        use_final_norm: bool = True,
        ffn_activation: str = "relu",
    ):
        super().__init__()
        self.max_len = max_len
        self.emb_scale = math.sqrt(emb_dim)
        self.use_recency_embedding = use_recency_embedding
        self.use_feedback_embedding = use_feedback_embedding
        self.movie_emb = nn.Embedding(feature_dict["movie_id"], emb_dim, padding_idx=0)
        self.recency_emb = nn.Embedding(feature_dict["hist_recency_bucket"], emb_dim, padding_idx=0)
        self.feedback_emb = nn.Embedding(4, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(emb_dim, num_heads, ffn_dim, dropout, ffn_activation=ffn_activation) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(emb_dim) if use_final_norm else nn.Identity()

    def encode_sequence(
        self,
        hist_movie_ids: torch.Tensor,
        hist_recency_bucket: torch.Tensor | None = None,
        hist_feedback: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hist_movie_ids = hist_movie_ids[:, -self.max_len :]
        positions = torch.arange(hist_movie_ids.size(1), device=hist_movie_ids.device).unsqueeze(0).expand_as(hist_movie_ids)
        key_padding_mask = hist_movie_ids.eq(0)

        x = self.movie_emb(hist_movie_ids) * self.emb_scale + self.pos_emb(positions)
        if self.use_recency_embedding and hist_recency_bucket is not None:
            hist_recency_bucket = hist_recency_bucket[:, -self.max_len :]
            x = x + self.recency_emb(hist_recency_bucket)
        if self.use_feedback_embedding and hist_feedback is not None:
            hist_feedback = hist_feedback[:, -self.max_len :]
            x = x + self.feedback_emb(hist_feedback.clamp_min(0).clamp_max(3))
        x = self.dropout(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones(hist_movie_ids.size(1), hist_movie_ids.size(1), device=hist_movie_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask)
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = self.final_norm(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        visible_mask = ~key_padding_mask
        return {
            "hidden_states": x,
            "visible_mask": visible_mask,
            "padding_mask": key_padding_mask,
        }

    def encode_user(
        self,
        hist_movie_ids: torch.Tensor,
        hist_recency_bucket: torch.Tensor | None = None,
        hist_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_outputs = self.encode_sequence(hist_movie_ids, hist_recency_bucket, hist_feedback)
        hidden_states = sequence_outputs["hidden_states"]
        visible_mask = sequence_outputs["visible_mask"]
        valid_lengths = visible_mask.sum(dim=1) - 1
        valid_lengths = valid_lengths.clamp_min(0)
        user_repr = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), valid_lengths]
        empty_history = visible_mask.sum(dim=1).eq(0)
        if empty_history.any():
            user_repr = user_repr.clone()
            user_repr[empty_history] = 0.0
        return user_repr

    def encode_item(self, item_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.movie_emb(item_batch["movie_id"])

    def next_item_logits(self, user_repr: torch.Tensor) -> torch.Tensor:
        logits = user_repr @ self.movie_emb.weight.T
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_repr = self.encode_user(
            batch["hist_movie_id"],
            batch.get("hist_recency_bucket"),
            batch.get("hist_feedback"),
        )
        item_repr = self.encode_item(batch["item"])
        logits = user_repr @ item_repr.T
        labels = torch.arange(logits.size(0), device=logits.device)
        return logits, labels
