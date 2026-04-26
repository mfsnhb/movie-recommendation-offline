from __future__ import annotations

import torch
from torch import nn

from offline.models.feature_encoders import MovieFeatureEncoder


def _left_align_by_mask(values: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    width = values.size(1)
    columns = torch.arange(width, device=values.device).unsqueeze(0).expand_as(values)
    order_keys = torch.where(keep_mask, columns, columns + width)
    order = order_keys.argsort(dim=1)
    return _left_align_by_order(values, order, keep_mask.sum(dim=1))


def _left_align_by_order(values: torch.Tensor, order: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    width = values.size(1)
    columns = torch.arange(width, device=values.device).unsqueeze(0)
    gather_order = order.view(*order.shape, *([1] * (values.ndim - 2))).expand_as(values)
    gathered = values.gather(1, gather_order)
    valid_shape = (values.size(0), width) + (1,) * (values.ndim - 2)
    valid_positions = columns.lt(counts.unsqueeze(1)).view(valid_shape)
    return torch.where(valid_positions, gathered, torch.zeros_like(values))


def right_align_by_mask(values: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    width = values.size(1)
    compact = _left_align_by_mask(values, keep_mask)
    counts = keep_mask.sum(dim=1)
    gather_positions = torch.arange(width, device=values.device).unsqueeze(0) - (width - counts).unsqueeze(1)
    valid_positions = gather_positions.ge(0) & gather_positions.lt(counts.unsqueeze(1))
    gathered = compact.gather(1, gather_positions.clamp_min(0))
    return torch.where(valid_positions, gathered, torch.zeros_like(values))


def _right_align_sequence_states(states: torch.Tensor, compact_mask: torch.Tensor) -> torch.Tensor:
    batch_size, width, hidden_dim = states.shape
    counts = compact_mask.sum(dim=1)
    source_positions = torch.arange(width, device=states.device).unsqueeze(0) - (width - counts).unsqueeze(1)
    valid_positions = source_positions.ge(0) & source_positions.lt(counts.unsqueeze(1))
    gathered = states.gather(1, source_positions.clamp_min(0).unsqueeze(-1).expand(batch_size, width, hidden_dim))
    return torch.where(valid_positions.unsqueeze(-1), gathered, torch.zeros_like(states))


def filter_sequence_history(
    hist_movie_ids: torch.Tensor,
    hist_recency_bucket: torch.Tensor | None = None,
    hist_feedback: torch.Tensor | None = None,
    history_feedback: str = "positive",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if history_feedback != "positive" or hist_feedback is None:
        return hist_movie_ids, hist_recency_bucket, hist_feedback
    keep_mask = hist_movie_ids.gt(0) & hist_feedback.eq(3)
    filtered_movie_ids = right_align_by_mask(hist_movie_ids, keep_mask)
    filtered_recency = right_align_by_mask(hist_recency_bucket, keep_mask) if hist_recency_bucket is not None else None
    filtered_feedback = right_align_by_mask(hist_feedback, keep_mask)
    return filtered_movie_ids, filtered_recency, filtered_feedback


class SequenceRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        max_len: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = int(hidden_dim or emb_dim)
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, emb_dim) if self.hidden_dim != emb_dim else nn.Identity()
        self.final_norm = nn.LayerNorm(emb_dim)

    def encode_sequence(
        self,
        hist_movie_ids: torch.Tensor,
        hist_recency_bucket: torch.Tensor | None = None,
        hist_feedback: torch.Tensor | None = None,
        hist_item_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del hist_recency_bucket, hist_feedback
        if hist_item_features is None:
            raise ValueError("hist_item_features is required for sequence encoding")
        hist_movie_ids = hist_movie_ids[:, -self.max_len :]
        hist_item_features = {field: value[:, -self.max_len :] for field, value in hist_item_features.items()}
        visible_mask = hist_movie_ids.gt(0)
        width = hist_movie_ids.size(1)
        columns = torch.arange(width, device=hist_movie_ids.device).unsqueeze(0).expand_as(hist_movie_ids)
        order_keys = torch.where(visible_mask, columns, columns + width)
        order = order_keys.argsort(dim=1)
        counts = visible_mask.sum(dim=1)
        compact_movie_ids = _left_align_by_order(hist_movie_ids, order, counts)
        compact_features = {
            "movie_id": compact_movie_ids,
            "genres": _left_align_by_order(hist_item_features["genres"], order, counts),
            "isAdult": _left_align_by_order(hist_item_features["isAdult"], order, counts),
            "startYear": _left_align_by_order(hist_item_features["startYear"], order, counts),
            "popularity": _left_align_by_order(hist_item_features["popularity"], order, counts),
        }
        compact_mask = compact_movie_ids.gt(0)
        lengths = compact_mask.sum(dim=1).cpu()
        x = self.dropout(self.movie_encoder(compact_features))
        compact_states = torch.zeros(hist_movie_ids.size(0), hist_movie_ids.size(1), self.hidden_dim, device=hist_movie_ids.device, dtype=x.dtype)
        non_empty = lengths.gt(0)
        if non_empty.any():
            non_empty_indices = non_empty.to(device=hist_movie_ids.device)
            packed = nn.utils.rnn.pack_padded_sequence(
                x[non_empty_indices],
                lengths[non_empty],
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.gru(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=hist_movie_ids.size(1))
            compact_states[non_empty_indices] = unpacked
        hidden_states = _right_align_sequence_states(compact_states, compact_mask)
        hidden_states = self.final_norm(self.output(hidden_states))
        hidden_states = hidden_states.masked_fill(~visible_mask.unsqueeze(-1), 0.0)
        return {
            "hidden_states": hidden_states,
            "visible_mask": visible_mask,
            "padding_mask": ~visible_mask,
        }

    def encode_user(
        self,
        hist_movie_ids: torch.Tensor,
        hist_recency_bucket: torch.Tensor | None = None,
        hist_feedback: torch.Tensor | None = None,
        hist_item_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sequence_outputs = self.encode_sequence(hist_movie_ids, hist_recency_bucket, hist_feedback, hist_item_features)
        hidden_states = sequence_outputs["hidden_states"]
        visible_mask = sequence_outputs["visible_mask"]
        visible_counts = visible_mask.sum(dim=1)
        last_indices = visible_mask.size(1) - 1 - torch.flip(visible_mask, dims=[1]).long().argmax(dim=1)
        last_indices = torch.where(visible_counts.gt(0), last_indices, torch.zeros_like(last_indices))
        user_repr = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]
        empty_history = visible_counts.eq(0)
        if empty_history.any():
            user_repr = user_repr.clone()
            user_repr[empty_history] = 0.0
        return user_repr

    def encode_item(self, item_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.movie_encoder(item_batch)
