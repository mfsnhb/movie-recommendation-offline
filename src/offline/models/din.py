from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, emb_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = emb_dim * 4
        for hidden_dim in hidden_dims or [128, 64]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.attention = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = query.unsqueeze(1).expand_as(keys)
        attention_input = torch.cat([query, keys, query - keys, query * keys], dim=-1)
        scores = self.attention(attention_input).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.float()
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        weights = weights / normalizer
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)


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
        self.fields = fields
        self.embeddings = nn.ModuleDict({field: nn.Embedding(feature_dict[field], emb_dim, padding_idx=0) for field in fields})
        self.movie_embedding = self.embeddings["movie_id"]
        self.genre_embedding = self.embeddings["genres"]
        self.movie_attention = AttentionPooling(emb_dim, attention_hidden_dims, dropout)
        self.genre_attention = AttentionPooling(emb_dim, attention_hidden_dims, dropout)

        input_dim = (len(fields) + 2) * emb_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in dnn_hidden_dims or [256, 128]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        base_vectors = [self.embeddings[field](batch[field]) for field in self.fields]

        target_movie = self.movie_embedding(batch["movie_id"])
        hist_movie_tokens = batch["hist_movie_id"]
        hist_movie_emb = self.movie_embedding(hist_movie_tokens)
        movie_mask = hist_movie_tokens.gt(0)
        if "hist_low_rating_mask" in batch:
            movie_mask = movie_mask & ~batch["hist_low_rating_mask"].gt(0)
        pooled_movie_hist = self.movie_attention(target_movie, hist_movie_emb, movie_mask)

        target_genre = self.genre_embedding(batch["genres"])
        hist_genre_tokens = batch["hist_genres"]
        hist_genre_emb = self.genre_embedding(hist_genre_tokens)
        genre_mask = hist_genre_tokens.gt(0)
        if "hist_low_rating_mask" in batch:
            genre_mask = genre_mask & ~batch["hist_low_rating_mask"].gt(0)
        pooled_genre_hist = self.genre_attention(target_genre, hist_genre_emb, genre_mask)

        dnn_input = torch.cat(base_vectors + [pooled_movie_hist, pooled_genre_hist], dim=1)
        return self.dnn(dnn_input).squeeze(-1)
