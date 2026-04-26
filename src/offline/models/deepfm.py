from __future__ import annotations

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
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


class MeanPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor, tokens: torch.Tensor, low_rating_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = tokens.gt(0)
        if low_rating_mask is not None:
            mask = mask & ~low_rating_mask.gt(0)
        mask = mask.unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1)
        return summed / counts


class DeepFMModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        fields: list[str],
        emb_dim: int,
        dnn_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        history_fields: list[str] | None = None,
    ):
        super().__init__()
        self.fields = fields
        self.history_fields = history_fields or []
        self.history_base_fields = {
            "hist_movie_id": "movie_id",
            "hist_genres": "genres",
        }
        self.linear_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], 1, padding_idx=0) for field in fields})
        self.fm_embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in fields})
        self.pool = MeanPooling()
        self.dnn = _build_mlp((len(fields) + len(self.history_fields)) * emb_dim, dnn_hidden_dims or [128, 64], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        linear_terms = []
        fm_vectors = []
        for field in self.fields:
            values = batch[field]
            linear_terms.append(self.linear_embeddings[field](values))
            fm_vectors.append(self.fm_embeddings[field](values))
        linear_logit = torch.stack(linear_terms, dim=0).sum(dim=0).squeeze(-1)
        fm_stack = torch.stack(fm_vectors, dim=1)
        summed = fm_stack.sum(dim=1)
        fm_logit = 0.5 * ((summed * summed) - (fm_stack * fm_stack).sum(dim=1)).sum(dim=1)
        history_vectors = []
        history_low_rating_mask = batch.get("hist_low_rating_mask")
        for history_field in self.history_fields:
            base_field = self.history_base_fields[history_field]
            tokens = batch[history_field]
            embeddings = self.fm_embeddings[base_field](tokens)
            history_vectors.append(self.pool(embeddings, tokens, history_low_rating_mask))
        dnn_input = torch.cat(fm_vectors + history_vectors, dim=1)
        dnn_logit = self.dnn(dnn_input).squeeze(-1)
        return linear_logit + fm_logit + dnn_logit
