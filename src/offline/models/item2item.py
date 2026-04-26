from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from offline.utils.logging import get_logger


logger = get_logger("offline.models.item2item")


class ItemEmbeddingQueue:
    def __init__(self, max_size: int, emb_dim: int):
        self.max_size = max(0, int(max_size))
        self.emb_dim = emb_dim
        self.item_ids: deque[int] = deque(maxlen=self.max_size)
        self.embeddings: deque[np.ndarray] = deque(maxlen=self.max_size)

    def push(self, item_ids: torch.Tensor, item_embeddings: torch.Tensor) -> None:
        if self.max_size <= 0:
            return
        item_ids_np = item_ids.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        item_embeddings_np = item_embeddings.detach().cpu().numpy().reshape(-1, self.emb_dim).astype(np.float32, copy=False)
        self.item_ids.extend(item_ids_np.tolist())
        self.embeddings.extend(np.asarray(item_embeddings_np, dtype=np.float32).copy())

    def get(self, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.item_ids:
            return None, None
        item_ids = torch.tensor(list(self.item_ids), dtype=torch.long, device=device)
        embeddings = torch.tensor(np.stack(self.embeddings), dtype=torch.float32, device=device)
        return item_ids, embeddings



def train_item2item_embeddings(
    train_data: dict,
    all_item_ids: np.ndarray,
    settings: dict,
    device: torch.device,
    init_embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    embedding_dim = int(settings.get("embedding_dim", 32))
    window_size = int(settings.get("window_size", 5))
    negative_samples = int(settings.get("negative_samples", 5))
    epochs = int(settings.get("epochs", 3))
    learning_rate = float(settings.get("learning_rate", 0.025))
    batch_size = int(settings.get("batch_size", 4096))
    negative_sampling = str(settings.get("negative_sampling", "random")).strip().lower()
    hard_negative_queue_size = int(settings.get("hard_negative_queue_size", 0))
    hard_negative_topk = int(settings.get("hard_negative_topk", 0))
    item_count = int(np.max(all_item_ids)) if all_item_ids.size > 0 else 0
    if item_count <= 0:
        return np.zeros((0, embedding_dim), dtype=np.float32), {"pair_count": 0, "embedding_dim": embedding_dim}

    centers, contexts, invalid_id_array, invalid_mask_array = _build_training_pairs(train_data, window_size)
    if centers.size == 0:
        return np.zeros((item_count, embedding_dim), dtype=np.float32), {"pair_count": 0, "embedding_dim": embedding_dim}

    center_tensor = torch.as_tensor(centers, dtype=torch.long, device=device)
    context_tensor = torch.as_tensor(contexts, dtype=torch.long, device=device)
    invalid_id_tensor = torch.as_tensor(invalid_id_array, dtype=torch.long, device=device)
    invalid_mask_tensor = torch.as_tensor(invalid_mask_array, dtype=torch.bool, device=device)

    in_embedding = nn.Embedding(item_count + 1, embedding_dim).to(device)
    out_embedding = nn.Embedding(item_count + 1, embedding_dim).to(device)
    nn.init.xavier_uniform_(in_embedding.weight)
    nn.init.zeros_(out_embedding.weight)
    if init_embeddings is not None:
        init_array = np.asarray(init_embeddings, dtype=np.float32)
        if init_array.shape == (item_count, embedding_dim):
            init_tensor = torch.as_tensor(init_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                in_embedding.weight[1 : item_count + 1].copy_(init_tensor)
                out_embedding.weight[1 : item_count + 1].copy_(init_tensor)

    optimizer = torch.optim.Adam(list(in_embedding.parameters()) + list(out_embedding.parameters()), lr=learning_rate)
    use_hard_negatives = negative_sampling == "hard" and hard_negative_queue_size > 0 and hard_negative_topk > 0
    memory_queue = ItemEmbeddingQueue(hard_negative_queue_size, embedding_dim)

    total_pairs = int(center_tensor.numel())
    effective_batch_size = max(batch_size, 1)
    for epoch in range(epochs):
        permutation = torch.randperm(total_pairs, device=device)
        epoch_loss = 0.0
        seen = 0
        epoch_hard = 0
        epoch_random = 0
        for start in range(0, total_pairs, effective_batch_size):
            batch_indices = permutation[start : start + effective_batch_size]
            batch_centers = center_tensor[batch_indices]
            batch_contexts = context_tensor[batch_indices]
            batch_invalid_ids = invalid_id_tensor[batch_indices]
            batch_invalid_mask = invalid_mask_tensor[batch_indices]
            optimizer.zero_grad()
            center_vecs = in_embedding(batch_centers)
            positive_vecs = out_embedding(batch_contexts)
            negative_ids, hard_count, random_count = _sample_negative_ids(
                center_vecs=center_vecs,
                batch_invalid_ids=batch_invalid_ids,
                batch_invalid_mask=batch_invalid_mask,
                negative_samples=negative_samples,
                item_count=item_count,
                memory_queue=memory_queue,
                hard_negative_topk=hard_negative_topk,
                use_hard_negatives=use_hard_negatives,
            )
            negative_vecs = out_embedding(negative_ids)
            positive_loss = F.logsigmoid((center_vecs * positive_vecs).sum(dim=1))
            negative_loss = F.logsigmoid(-(negative_vecs * center_vecs.unsqueeze(1)).sum(dim=2)).sum(dim=1)
            loss = -(positive_loss + negative_loss).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_centers.size(0)
            seen += int(batch_centers.size(0))
            epoch_hard += hard_count
            epoch_random += random_count
            memory_queue.push(batch_contexts, out_embedding(batch_contexts))
        logger.info(
            "Item2Item epoch %s/%s | avg_loss=%.4f | pairs=%s | window_size=%s | negative_sampling=%s | hard_negatives=%s | random_fallback=%s",
            epoch + 1,
            epochs,
            epoch_loss / max(seen, 1),
            total_pairs,
            window_size,
            negative_sampling,
            epoch_hard,
            epoch_random,
        )

    embeddings = 0.5 * (in_embedding.weight.detach().cpu().numpy() + out_embedding.weight.detach().cpu().numpy())
    embeddings = embeddings[1 : item_count + 1].astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-9, None)
    return embeddings, {
        "pair_count": total_pairs,
        "embedding_dim": embedding_dim,
        "window_size": window_size,
        "negative_samples": negative_samples,
        "epochs": epochs,
        "negative_sampling": negative_sampling,
        "hard_negative_queue_size": hard_negative_queue_size,
        "hard_negative_topk": hard_negative_topk,
    }



def _build_training_pairs(train_data: dict, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histories = np.asarray(train_data["hist_movie_id"], dtype=np.int64)
    targets = np.asarray(train_data["movie_id"], dtype=np.int64).reshape(-1)
    if histories.ndim != 2 or targets.ndim != 1:
        raise ValueError("Item2Item training expects hist_movie_id to be 2D and movie_id to be 1D")

    effective_window = max(min(window_size, histories.shape[1]), 1)
    window_histories = histories[:, -effective_window:]
    valid_targets = targets > 0
    pair_mask = valid_targets[:, None] & (window_histories > 0)
    if not bool(pair_mask.any()):
        width = effective_window + 1
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0, width), dtype=bool),
        )

    row_indices, context_indices = np.nonzero(pair_mask)
    forward_centers = targets[row_indices]
    forward_contexts = window_histories[row_indices, context_indices]

    blocked_ids = np.concatenate([targets[:, None], window_histories], axis=1)
    blocked_mask = np.concatenate([valid_targets[:, None], window_histories > 0], axis=1)
    pair_blocked_ids = blocked_ids[row_indices]
    pair_blocked_mask = blocked_mask[row_indices]

    centers = np.concatenate([forward_centers, forward_contexts], axis=0)
    contexts = np.concatenate([forward_contexts, forward_centers], axis=0)
    invalid_ids = np.concatenate([pair_blocked_ids, pair_blocked_ids], axis=0)
    invalid_mask = np.concatenate([pair_blocked_mask, pair_blocked_mask], axis=0)
    return centers, contexts, invalid_ids, invalid_mask



def _sample_negative_ids(
    center_vecs: torch.Tensor,
    batch_invalid_ids: torch.Tensor,
    batch_invalid_mask: torch.Tensor,
    negative_samples: int,
    item_count: int,
    memory_queue: ItemEmbeddingQueue,
    hard_negative_topk: int,
    use_hard_negatives: bool,
) -> tuple[torch.Tensor, int, int]:
    batch_size = center_vecs.size(0)
    device = center_vecs.device
    if negative_samples <= 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=device), 0, 0

    hard_packed = torch.zeros((batch_size, negative_samples), dtype=torch.long, device=device)
    hard_slots = torch.zeros(batch_size, dtype=torch.long, device=device)
    queue_item_ids, queue_embeddings = memory_queue.get(device) if use_hard_negatives else (None, None)
    hard_candidate_count = 0 if queue_item_ids is None else min(hard_negative_topk, int(queue_item_ids.numel()))
    if hard_candidate_count > 0 and queue_embeddings is not None:
        hard_scores = center_vecs @ queue_embeddings.T
        blocked_queue_mask = batch_invalid_mask.unsqueeze(2) & batch_invalid_ids.unsqueeze(2).eq(queue_item_ids.view(1, 1, -1))
        hard_scores = hard_scores.masked_fill(blocked_queue_mask.any(dim=1), torch.finfo(hard_scores.dtype).min)
        hard_values, hard_indices = torch.topk(hard_scores, k=hard_candidate_count, dim=1)
        hard_candidate_ids = queue_item_ids[hard_indices]
        hard_valid_mask = hard_values > (torch.finfo(hard_values.dtype).min / 2)
        hard_valid_mask &= _first_occurrence_mask(hard_candidate_ids)
        hard_selection_mask = hard_valid_mask & (hard_valid_mask.cumsum(dim=1) <= negative_samples)
        hard_packed = _pack_candidate_ids(hard_candidate_ids, hard_selection_mask, negative_samples)
        hard_slots = hard_selection_mask.sum(dim=1)

    packed_ids = hard_packed
    packed_mask = torch.arange(negative_samples, device=device).unsqueeze(0) < hard_slots.unsqueeze(1)

    remaining_needed = negative_samples - hard_slots
    max_remaining = int(remaining_needed.max().item()) if remaining_needed.numel() > 0 else 0
    random_slots = torch.zeros(batch_size, dtype=torch.long, device=device)
    if max_remaining > 0:
        blocked_ids = torch.cat([batch_invalid_ids, packed_ids], dim=1)
        blocked_mask = torch.cat([batch_invalid_mask, packed_mask], dim=1)
        sample_width = max(64, negative_samples * 8)
        random_candidates = torch.randint(1, item_count + 1, (batch_size, sample_width), device=device)
        random_valid_mask = _valid_candidate_mask(random_candidates, blocked_ids, blocked_mask)
        random_valid_mask &= _first_occurrence_mask(random_candidates)
        random_selection_mask = random_valid_mask & (random_valid_mask.cumsum(dim=1) <= remaining_needed.unsqueeze(1))
        random_slots = random_selection_mask.sum(dim=1)
        random_packed = _pack_candidate_ids(random_candidates, random_selection_mask, negative_samples)
        combined_ids = torch.cat([packed_ids, random_packed], dim=1)
        combined_mask = torch.cat([
            packed_mask,
            torch.arange(negative_samples, device=device).unsqueeze(0) < random_slots.unsqueeze(1),
        ], dim=1)
        packed_ids = _pack_candidate_ids(combined_ids, combined_mask, negative_samples)

        filled_counts = hard_slots + random_slots
        if bool((filled_counts < negative_samples).any()):
            packed_ids = _fill_remaining_random(
                packed_ids=packed_ids,
                filled_counts=filled_counts,
                batch_invalid_ids=batch_invalid_ids,
                batch_invalid_mask=batch_invalid_mask,
                item_count=item_count,
            )
            random_slots = negative_samples - hard_slots

    hard_count = int(hard_slots.sum().item())
    random_count = int(random_slots.sum().item())
    return packed_ids, hard_count, random_count



def _first_occurrence_mask(candidate_ids: torch.Tensor) -> torch.Tensor:
    if candidate_ids.size(1) == 0:
        return torch.zeros_like(candidate_ids, dtype=torch.bool)
    duplicate_matrix = candidate_ids.unsqueeze(2).eq(candidate_ids.unsqueeze(1))
    seen_before = torch.tril(duplicate_matrix, diagonal=-1).any(dim=2)
    return ~seen_before



def _valid_candidate_mask(candidate_ids: torch.Tensor, blocked_ids: torch.Tensor, blocked_mask: torch.Tensor) -> torch.Tensor:
    blocked_matches = candidate_ids.unsqueeze(2).eq(blocked_ids.unsqueeze(1))
    blocked_matches &= blocked_mask.unsqueeze(1)
    return ~blocked_matches.any(dim=2)



def _pack_candidate_ids(candidate_ids: torch.Tensor, selected_mask: torch.Tensor, width: int) -> torch.Tensor:
    batch_size, candidate_width = candidate_ids.shape
    if width <= 0:
        return torch.zeros((batch_size, 0), dtype=torch.long, device=candidate_ids.device)
    if candidate_width == 0:
        return torch.zeros((batch_size, width), dtype=torch.long, device=candidate_ids.device)

    positions = torch.arange(candidate_width, device=candidate_ids.device).unsqueeze(0).expand(batch_size, -1)
    sort_key = torch.where(selected_mask, positions, positions + candidate_width)
    sort_indices = sort_key.argsort(dim=1)
    packed = torch.where(selected_mask, candidate_ids, torch.zeros_like(candidate_ids)).gather(1, sort_indices)
    if candidate_width >= width:
        return packed[:, :width]
    padding = torch.zeros((batch_size, width - candidate_width), dtype=torch.long, device=candidate_ids.device)
    return torch.cat([packed, padding], dim=1)



def _fill_remaining_random(
    packed_ids: torch.Tensor,
    filled_counts: torch.Tensor,
    batch_invalid_ids: torch.Tensor,
    batch_invalid_mask: torch.Tensor,
    item_count: int,
) -> torch.Tensor:
    for row_idx in torch.nonzero(filled_counts < packed_ids.size(1), as_tuple=False).flatten().tolist():
        selected = packed_ids[row_idx, : filled_counts[row_idx]].detach().cpu().tolist()
        blocked = set(batch_invalid_ids[row_idx][batch_invalid_mask[row_idx]].detach().cpu().tolist())
        blocked.update(int(item_id) for item_id in selected)
        while len(selected) < packed_ids.size(1):
            candidate_id = int(torch.randint(1, item_count + 1, (1,), device=packed_ids.device).item())
            if candidate_id in blocked:
                continue
            selected.append(candidate_id)
            blocked.add(candidate_id)
        packed_ids[row_idx] = torch.as_tensor(selected, dtype=torch.long, device=packed_ids.device)
    return packed_ids
