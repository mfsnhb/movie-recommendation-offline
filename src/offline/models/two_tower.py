from __future__ import annotations

import torch
from torch import nn


class MeanPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens > 0).unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1)
        return summed / counts


class WeightedMeanPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).float()
        normalized_weights = weights.unsqueeze(-1).float() * mask
        denom = normalized_weights.sum(dim=1).clamp_min(1e-9)
        return (embeddings * normalized_weights).sum(dim=1) / denom



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


class TwoTowerRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        user_hidden_dims: list[int] | None = None,
        item_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        rating_weighting_enabled: bool = False,
        rating_weight_neutral: float = 3.0,
        rating_weight_scale: float = 0.25,
        rating_weight_min: float = 0.0,
        rating_weight_max: float = 1.0,
        short_history_length: int = 10,
        positive_rating_min: float = 4.0,
    ):
        super().__init__()
        self.rating_weighting_enabled = bool(rating_weighting_enabled)
        self.rating_weight_neutral = float(rating_weight_neutral)
        self.rating_weight_scale = float(rating_weight_scale)
        self.rating_weight_min = float(rating_weight_min)
        self.rating_weight_max = float(rating_weight_max)
        self.short_history_length = int(short_history_length)
        self.positive_rating_min = float(positive_rating_min)
        self.user_id_emb = nn.Embedding(feature_dict["user_id"], emb_dim, padding_idx=0)
        self.age_emb = nn.Embedding(feature_dict["age"], emb_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(feature_dict["gender"], emb_dim, padding_idx=0)
        self.occupation_emb = nn.Embedding(feature_dict["occupation"], emb_dim, padding_idx=0)
        self.zip_code_emb = nn.Embedding(feature_dict["zip_code"], emb_dim, padding_idx=0)
        self.movie_emb = nn.Embedding(feature_dict["movie_id"], emb_dim, padding_idx=0)
        self.genre_emb = nn.Embedding(feature_dict["genres"], emb_dim, padding_idx=0)
        self.recency_emb = nn.Embedding(feature_dict["hist_recency_bucket"], emb_dim, padding_idx=0)
        self.is_adult_emb = nn.Embedding(feature_dict["isAdult"], emb_dim, padding_idx=0)
        self.start_year_emb = nn.Embedding(feature_dict["startYear"], emb_dim, padding_idx=0)
        self.pool = MeanPooling()
        self.weighted_pool = WeightedMeanPooling()
        self.user_mlp = _build_mlp(emb_dim * 12, user_hidden_dims or [256, 128], emb_dim, dropout=dropout)
        self.item_mlp = _build_mlp(emb_dim * 4, item_hidden_dims or [256, 128], emb_dim, dropout=dropout)

    def _rating_weights(self, hist_rating: torch.Tensor | None, hist_movie_id: torch.Tensor) -> torch.Tensor:
        if hist_rating is None or not self.rating_weighting_enabled:
            return hist_movie_id.gt(0).float()
        weights = 1.0 + self.rating_weight_scale * (hist_rating.float() - self.rating_weight_neutral)
        weights = weights.clamp(min=self.rating_weight_min, max=self.rating_weight_max)
        return weights * hist_movie_id.gt(0).float()

    def _recency_weights(self, hist_recency_bucket: torch.Tensor | None, hist_movie_id: torch.Tensor) -> torch.Tensor:
        if hist_recency_bucket is None:
            return hist_movie_id.gt(0).float()
        return (1.0 / hist_recency_bucket.clamp_min(1).float()) * hist_movie_id.gt(0).float()

    def _short_tail(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.short_history_length <= 0:
            return tensor
        return tensor[:, -self.short_history_length :]

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hist_movie_id = batch["hist_movie_id"]
        hist_genres = batch["hist_genres"]
        hist_recency_bucket = batch["hist_recency_bucket"]
        hist_rating = batch.get("hist_rating")

        movie_embeddings = self.movie_emb(hist_movie_id)
        genre_embeddings = self.genre_emb(hist_genres)
        recency_embeddings = self.recency_emb(hist_recency_bucket)

        rating_weights = self._rating_weights(hist_rating, hist_movie_id)
        recency_weights = self._recency_weights(hist_recency_bucket, hist_movie_id)
        combined_weights = rating_weights * recency_weights

        pooled_hist_movie = self.weighted_pool(movie_embeddings, combined_weights, hist_movie_id.gt(0))
        pooled_hist_genre = self.weighted_pool(genre_embeddings, combined_weights, hist_genres.gt(0))
        pooled_hist_recency = self.weighted_pool(recency_embeddings, combined_weights, hist_recency_bucket.gt(0))

        short_movie_id = self._short_tail(hist_movie_id)
        short_movie_embeddings = self.movie_emb(short_movie_id)
        short_weights = self._short_tail(combined_weights)
        pooled_short_hist_movie = self.weighted_pool(short_movie_embeddings, short_weights, short_movie_id.gt(0))

        short_genres = self._short_tail(hist_genres)
        short_genre_embeddings = self.genre_emb(short_genres)
        pooled_short_hist_genre = self.weighted_pool(short_genre_embeddings, short_weights, short_genres.gt(0))

        if hist_rating is None:
            positive_mask = hist_movie_id.gt(0)
        else:
            positive_mask = hist_movie_id.gt(0) & hist_rating.float().ge(self.positive_rating_min)
        positive_weights = recency_weights * positive_mask.float()
        pooled_positive_hist_movie = self.weighted_pool(movie_embeddings, positive_weights, positive_mask)
        pooled_positive_hist_genre = self.weighted_pool(genre_embeddings, positive_weights, positive_mask & hist_genres.gt(0))

        user_repr = self.user_mlp(torch.cat([
            self.user_id_emb(batch["user_id"]),
            self.age_emb(batch["age"]),
            self.gender_emb(batch["gender"]),
            self.occupation_emb(batch["occupation"]),
            self.zip_code_emb(batch["zip_code"]),
            pooled_hist_movie,
            pooled_hist_genre,
            pooled_hist_recency,
            pooled_short_hist_movie,
            pooled_short_hist_genre,
            pooled_positive_hist_movie,
            pooled_positive_hist_genre,
        ], dim=1))
        return torch.nn.functional.normalize(user_repr, dim=1)

    def encode_item(self, item_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        genre_tokens = item_batch["genres"]
        if genre_tokens.ndim == 1:
            genre_tokens = genre_tokens.unsqueeze(1)
        item_repr = self.item_mlp(torch.cat([
            self.movie_emb(item_batch["movie_id"]),
            self.pool(self.genre_emb(genre_tokens), genre_tokens),
            self.is_adult_emb(item_batch["isAdult"]),
            self.start_year_emb(item_batch["startYear"]),
        ], dim=1))
        return torch.nn.functional.normalize(item_repr, dim=1)
