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


class SequenceRetrievalModel(nn.Module):
    def __init__(
        self,
        feature_dict: dict,
        emb_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        max_len: int = 10,
        dropout: float = 0.1,
        multimodal_table=None,
        item_feature_table=None,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = int(hidden_dim or emb_dim)
        self.movie_encoder = MovieFeatureEncoder(feature_dict, emb_dim, dropout=dropout, output_norm=False, multimodal_table=multimodal_table, item_feature_table=item_feature_table)
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
        hist_time_gap_bucket: torch.Tensor | None = None,
        hist_rating: torch.Tensor | None = None,
        hist_item_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hist_movie_ids = hist_movie_ids[:, -self.max_len :]
        if hist_time_gap_bucket is None:
            hist_time_gap_bucket = torch.zeros_like(hist_movie_ids)
        else:
            hist_time_gap_bucket = hist_time_gap_bucket[:, -self.max_len :].long()
        if hist_rating is None:
            hist_rating = torch.zeros_like(hist_movie_ids, dtype=torch.float32)
        else:
            hist_rating = hist_rating[:, -self.max_len :].float()
        visible_mask = hist_movie_ids.gt(0)
        width = hist_movie_ids.size(1)
        columns = torch.arange(width, device=hist_movie_ids.device).unsqueeze(0).expand_as(hist_movie_ids)
        order_keys = torch.where(visible_mask, columns, columns + width)
        order = order_keys.argsort(dim=1)
        counts = visible_mask.sum(dim=1)
        compact_movie_ids = _left_align_by_order(hist_movie_ids, order, counts)
        compact_rating = _left_align_by_order(hist_rating, order, counts)
        compact_time_gap = _left_align_by_order(hist_time_gap_bucket, order, counts).long()
        compact_mask = compact_movie_ids.gt(0)
        lengths = compact_mask.sum(dim=1).cpu()
        x = self.movie_encoder({
            "movie_id": compact_movie_ids,
            "interaction_rating": compact_rating,
            "interaction_time_gap_bucket": compact_time_gap,
        })
        x = self.dropout(x)
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
        hist_time_gap_bucket: torch.Tensor | None = None,
        hist_rating: torch.Tensor | None = None,
        hist_item_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sequence_outputs = self.encode_sequence(hist_movie_ids, hist_time_gap_bucket, hist_rating, hist_item_features)
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

    def encode_item(self, item_ids: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.movie_encoder(item_ids)
