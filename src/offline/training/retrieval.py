from __future__ import annotations

from collections import defaultdict, Counter
import json
import time

import numpy as np
import torch
import yaml
from torch import nn

from offline.evaluate.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, save_metrics
from offline.evaluate.multi_recall import build_multi_recall_artifacts
from offline.features.item_batch import ITEM_FEATURE_FIELDS as _ITEM_FEATURE_FIELDS, build_item_batch, item_feature_tensors as to_item_feature_tensors
from offline.models.item2item import _first_occurrence_mask, _pack_candidate_ids, _valid_candidate_mask, train_item2item_embeddings
from offline.models.sequence_retrieval import SequenceRetrievalModel, filter_sequence_history
from offline.models.two_tower import TwoTowerRetrievalModel
from offline.ranking.protocol import get_all_item_ids, get_item_feature_arrays
from offline.utils.config import resolve_retrieval_config, resolve_retrieval_route_names
from offline.utils.io import (
    CONFIG_DIR,
    GENRE_MODEL_PATH,
    ITEM2ITEM_ITEM_EMBEDDINGS_PATH,
    ITEM2ITEM_MODEL_PATH,
    ITEM2ITEM_MOVIE_IDS_PATH,
    ITEM_CATALOG_PATH,
    ITEM_CF_MODEL_PATH,
    ITEM_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MOVIE_RAW_IDS_PATH,
    POPULAR_MODEL_PATH,
    RETRIEVAL_FEATURE_DICT_PATH,
    RETRIEVAL_METRICS_PATH,
    RETRIEVAL_MODEL_PATH,
    RETRIEVAL_SAMPLE_PATH,
    RETRIEVAL_VOCAB_DICT_PATH,
    SEQUENCE_ITEM_EMBEDDINGS_PATH,
    SEQUENCE_MODEL_PATH,
    load_pickle,
    save_numpy,
    save_pickle,
)
from offline.utils.logging import format_eta, get_logger


logger = get_logger("offline.training.retrieval")




class ItemMemoryQueue:
    def __init__(self, max_size: int, emb_dim: int):
        self.max_size = max(0, int(max_size))
        self.emb_dim = emb_dim
        self.item_ids: torch.Tensor | None = None
        self.embeddings: torch.Tensor | None = None
        self.next_index = 0
        self.size = 0

    def push(self, item_ids: torch.Tensor, item_embeddings: torch.Tensor) -> None:
        if self.max_size <= 0:
            return
        item_ids = item_ids.detach().long().reshape(-1)
        item_embeddings = item_embeddings.detach().float().reshape(-1, self.emb_dim)
        if self.item_ids is None or self.embeddings is None or self.item_ids.device != item_ids.device:
            self.item_ids = torch.zeros(self.max_size, dtype=torch.long, device=item_ids.device)
            self.embeddings = torch.zeros((self.max_size, self.emb_dim), dtype=torch.float32, device=item_ids.device)
            self.next_index = 0
            self.size = 0
        if item_ids.numel() >= self.max_size:
            self.item_ids.copy_(item_ids[-self.max_size :])
            self.embeddings.copy_(item_embeddings[-self.max_size :])
            self.next_index = 0
            self.size = self.max_size
            return
        insert_count = int(item_ids.numel())
        first_count = min(insert_count, self.max_size - self.next_index)
        self.item_ids[self.next_index : self.next_index + first_count] = item_ids[:first_count]
        self.embeddings[self.next_index : self.next_index + first_count] = item_embeddings[:first_count]
        remaining = insert_count - first_count
        if remaining > 0:
            self.item_ids[:remaining] = item_ids[first_count:]
            self.embeddings[:remaining] = item_embeddings[first_count:]
        self.next_index = (self.next_index + insert_count) % self.max_size
        self.size = min(self.size + insert_count, self.max_size)

    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.size <= 0 or self.item_ids is None or self.embeddings is None:
            return None, None
        return self.item_ids[: self.size], self.embeddings[: self.size]





def _index_batch_loader(indices, batch_size: int, *, shuffle: bool) -> list[np.ndarray]:
    index_array = np.asarray(indices, dtype=np.int64)
    if shuffle:
        index_array = index_array[np.random.permutation(index_array.shape[0])]
    return [index_array[start : start + batch_size] for start in range(0, index_array.shape[0], batch_size)]



def _slice_train_batch(train_data: dict[str, np.ndarray], batch_indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    tensor_specs: dict[str, torch.dtype] = {
        "user_id": torch.long,
        "age": torch.long,
        "gender": torch.long,
        "occupation": torch.long,
        "zip_code": torch.long,
        "hist_movie_id": torch.long,
        "hist_recency_bucket": torch.long,
        "hist_rating": torch.float32,
        "hist_feedback": torch.long,
        "movie_id": torch.long,
        "rating": torch.float32,
        "user_negative_movie_id": torch.long,
    }
    batch: dict[str, torch.Tensor] = {}
    for field, dtype in tensor_specs.items():
        if field not in train_data:
            continue
        batch[field] = torch.from_numpy(np.asarray(train_data[field])[batch_indices]).to(device=device, dtype=dtype)
    return batch




def _resolve_rating_weighting_settings(settings: dict | None) -> dict[str, float | bool]:
    source = dict(settings or {})
    return {
        "enabled": bool(source.get("rating_weighting_enabled", False)),
        "neutral": float(source.get("rating_weight_neutral", 3.0)),
        "scale": float(source.get("rating_weight_scale", 0.25)),
        "min": float(source.get("rating_weight_min", 0.0)),
        "max": float(source.get("rating_weight_max", 1.0)),
    }



def _rating_weights(values, settings: dict | None) -> np.ndarray:
    ratings = np.asarray(values, dtype=np.float32)
    config = _resolve_rating_weighting_settings(settings)
    if not bool(config["enabled"]):
        return np.ones_like(ratings, dtype=np.float32)
    weights = 1.0 + float(config["scale"]) * (ratings - float(config["neutral"]))
    return np.clip(weights, float(config["min"]), float(config["max"])).astype(np.float32, copy=False)



def _rating_weight(value: float, settings: dict | None) -> float:
    return float(_rating_weights(np.asarray([value], dtype=np.float32), settings)[0])



def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().data.norm(2).item())
        total += grad_norm * grad_norm
    return total ** 0.5



def run_retrieval_training(routes=None, warm_start: bool = True, build_multi_recall: bool = False, evaluate: bool = False):
    settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
    selected_routes = resolve_retrieval_route_names(settings, routes)
    artifacts = train_retrieval_routes(routes=selected_routes, warm_start=warm_start)
    result: dict[str, object] = {"trained_routes": selected_routes, "artifacts": artifacts}

    if evaluate and any(route in selected_routes for route in ("two_tower", "sequence")):
        metrics = {
            route: evaluate_retrieval(route=route)
            for route in selected_routes
            if route in {"two_tower", "sequence"}
        }
        result["retrieval"] = metrics

    if build_multi_recall:
        resolved_config = resolve_retrieval_config(settings)
        multi_recall_routes = resolved_config["multi_recall_settings"].get("routes")
        artifacts = build_multi_recall_artifacts(
            topk=int(resolved_config["evaluation_settings"].get("topk", 200)),
            routes=multi_recall_routes,
        )
        result["multi_recall"] = artifacts["metrics"]
    return result



def train_retrieval_routes(routes=None, warm_start: bool = True):
    settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_retrieval_config(settings)
    selected_routes = resolve_retrieval_route_names(settings, routes)
    context = _load_retrieval_context(settings, resolved_config)
    results: dict[str, dict] = {}

    if "two_tower" in selected_routes:
        results["two_tower"] = train_two_tower(context=context, warm_start=warm_start)
    if "sequence" in selected_routes:
        results["sequence"] = train_sequence(context=context, warm_start=warm_start)

    if "item2item" in selected_routes:
        results["item2item"] = train_item2item(context, warm_start=warm_start)
    if "item_cf" in selected_routes:
        results["item_cf"] = train_item_cf(context)
    if "genre" in selected_routes:
        results["genre"] = train_genre(context)
    if "popular" in selected_routes:
        results["popular"] = train_popular(context)
    return results



def train_two_tower(context=None, warm_start: bool = True):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    settings = context["settings"]
    resolved_config = context["resolved_config"]
    two_tower_settings = resolved_config["two_tower_settings"]
    training_settings = resolved_config["training_settings"]

    train_data = context["train_data"]
    test_data = context["test_data"]
    feature_dict = context["feature_dict"]
    vocab_dict = context["vocab_dict"]
    item_feature_arrays = context["item_feature_arrays"]
    all_item_ids = context["all_item_ids"]
    emb_dim = int(resolved_config["embedding_dim"])
    batch_size = int(training_settings.get("batch_size", 256))
    epochs = int(training_settings.get("epochs", 10))
    learning_rate = float(training_settings.get("learning_rate", 0.001))
    weight_decay = float(training_settings.get("weight_decay", 1e-5))
    log_every_n_batches = int(training_settings.get("log_every_n_batches", 100))
    early_stopping_patience = int(training_settings.get("early_stopping_patience", 2))
    min_delta = float(training_settings.get("min_delta", 1e-4))
    negative_sampling = str(training_settings.get("negative_sampling", "in_batch")).strip().lower()
    hard_negative_queue_size = int(training_settings.get("hard_negative_queue_size", 0))
    hard_negative_topk = int(training_settings.get("hard_negative_topk", 0))
    two_tower_num_sampled_negatives = int(training_settings.get("two_tower_num_sampled_negatives", 64))
    export_batch_size = int(training_settings.get("export_batch_size", 4096))
    two_tower_temperature = float(training_settings.get("two_tower_temperature", 0.05))

    device = context["device"]
    item_feature_tensors = to_item_feature_tensors(item_feature_arrays, device)
    all_item_id_tensor = torch.as_tensor(all_item_ids, dtype=torch.long, device=device)
    two_tower_model = TwoTowerRetrievalModel(
        feature_dict,
        emb_dim,
        user_hidden_dims=two_tower_settings.get("user_hidden_dims"),
        item_hidden_dims=two_tower_settings.get("item_hidden_dims"),
        dropout=float(two_tower_settings.get("dropout", 0.1)),
        rating_weighting_enabled=bool(training_settings.get("rating_weighting_enabled", False)),
        rating_weight_neutral=float(training_settings.get("rating_weight_neutral", 3.0)),
        rating_weight_scale=float(training_settings.get("rating_weight_scale", 0.25)),
        rating_weight_min=float(training_settings.get("rating_weight_min", 0.0)),
        rating_weight_max=float(training_settings.get("rating_weight_max", 1.0)),
        short_history_length=int(two_tower_settings.get("short_history_length", 10)),
        positive_rating_min=float(training_settings.get("positive_rating_min", 4.0)),
        multimodal_table=item_feature_arrays["multimodal_embedding"],
    ).to(device)

    if warm_start and RETRIEVAL_MODEL_PATH.exists():
        retrieval_checkpoint = torch.load(RETRIEVAL_MODEL_PATH, map_location=device)
        two_tower_model.load_state_dict(retrieval_checkpoint["state_dict"])

    two_tower_optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_size = int(len(train_data["user_id"]))
    val_ratio = 0.1
    val_size = max(1, int(train_size * val_ratio))
    train_size = train_size - val_size
    all_indices = np.arange(len(train_data["user_id"]), dtype=np.int64)
    rng = np.random.default_rng(42)
    rng.shuffle(all_indices)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    use_hard_negatives = negative_sampling in {"in_batch_hard", "mixed"} and hard_negative_queue_size > 0 and hard_negative_topk > 0
    use_sampled_negatives = negative_sampling in {"sampled", "mixed"} and two_tower_num_sampled_negatives > 0
    tt_queue = ItemMemoryQueue(hard_negative_queue_size, emb_dim)

    logger.info(
        "TwoTower training start | device=%s | train_samples=%s | val_samples=%s | test_users=%s | batch_size=%s | epochs=%s | negative_sampling=%s | hard_negative_queue_size=%s | hard_negative_topk=%s | sampled_negatives=%s | export_batch_size=%s | temperature=%s | warm_start=%s",
        device,
        train_size,
        val_size,
        len(test_data["user_id"]),
        batch_size,
        epochs,
        negative_sampling,
        hard_negative_queue_size,
        hard_negative_topk,
        two_tower_num_sampled_negatives if use_sampled_negatives else 0,
        export_batch_size,
        two_tower_temperature,
        warm_start,
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    total_batches = max(int(np.ceil(train_size / max(batch_size, 1))), 1)

    for epoch in range(epochs):
        two_tower_model.train()
        total_tt_loss = 0.0
        seen_examples = 0
        total_grad_norm = 0.0
        grad_norm_steps = 0
        epoch_start = time.perf_counter()
        train_batches = _index_batch_loader(train_indices, batch_size, shuffle=True)

        for batch_idx, batch_indices in enumerate(train_batches, start=1):
            batch_dict = _slice_train_batch(train_data, batch_indices, device)
            hist_item_batch = build_item_batch(batch_dict["hist_movie_id"], item_feature_tensors, device)
            batch_dict.update({f"hist_{field}": value for field, value in hist_item_batch.items() if field != "movie_id"})
            batch_dict["item"] = build_item_batch(batch_dict["movie_id"], item_feature_tensors, device)

            two_tower_optimizer.zero_grad()
            user_repr_tt = two_tower_model.encode_user(batch_dict)
            item_repr_tt = two_tower_model.encode_item(batch_dict["item"])
            sampled_negative_ids = None
            sampled_negative_repr = None
            if use_sampled_negatives:
                sampled_negative_ids = _sample_context_negative_ids(
                    positive_ids=batch_dict["movie_id"],
                    context_ids=batch_dict["hist_movie_id"],
                    all_item_ids=all_item_id_tensor,
                    num_negatives=two_tower_num_sampled_negatives,
                )
                sampled_negative_batch = build_item_batch(sampled_negative_ids.reshape(-1), item_feature_tensors, device)
                sampled_negative_repr = two_tower_model.encode_item(sampled_negative_batch).reshape(
                    sampled_negative_ids.size(0),
                    sampled_negative_ids.size(1),
                    -1,
                )
            logits_tt, labels_tt = _build_training_logits(
                user_repr_tt,
                item_repr_tt,
                batch_dict["movie_id"],
                batch_dict["hist_movie_id"],
                tt_queue,
                hard_negative_topk,
                use_hard_negatives,
                two_tower_temperature,
                sampled_negative_ids=sampled_negative_ids,
                sampled_negative_repr=sampled_negative_repr,
            )
            loss_tt = torch.nn.functional.cross_entropy(logits_tt, labels_tt)
            loss_tt.backward()
            should_log = batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
            grad_norm_tt = _gradient_norm(two_tower_model.parameters()) if should_log else 0.0
            two_tower_optimizer.step()
            tt_queue.push(batch_dict["movie_id"], item_repr_tt)

            batch_size_now = batch_dict["user_id"].size(0)
            total_tt_loss += float(loss_tt.item()) * batch_size_now
            seen_examples += batch_size_now
            if should_log:
                total_grad_norm += grad_norm_tt
                grad_norm_steps += 1

                elapsed_epoch = time.perf_counter() - epoch_start
                avg_batch_time = elapsed_epoch / batch_idx
                remaining_batches = total_batches - batch_idx
                remaining_epochs = epochs - epoch - 1
                eta_seconds = avg_batch_time * (remaining_batches + remaining_epochs * total_batches)
                logger.info(
                    "TwoTower epoch %s/%s | batch %s/%s | loss=%.4f | avg_loss=%.4f | grad_norm=%.6f | avg_grad_norm=%.6f | eta=%s",
                    epoch + 1,
                    epochs,
                    batch_idx,
                    total_batches,
                    float(loss_tt.item()),
                    total_tt_loss / max(seen_examples, 1),
                    grad_norm_tt,
                    total_grad_norm / max(grad_norm_steps, 1),
                    format_eta(eta_seconds),
                )

        train_tt_loss = total_tt_loss / max(train_size, 1)
        val_metrics = _evaluate_loss(
            two_tower_model,
            None,
            train_data,
            val_indices,
            device,
            item_feature_tensors,
            batch_size=batch_size,
            evaluate_two_tower=True,
            evaluate_sequence=False,
            two_tower_temperature=two_tower_temperature,
        )
        logger.info(
            "TwoTower epoch %s/%s done | train_loss=%.4f | val_loss=%.4f | avg_grad_norm=%.6f | epoch_time=%s | elapsed=%s",
            epoch + 1,
            epochs,
            train_tt_loss,
            val_metrics["loss_blended"],
            total_grad_norm / max(grad_norm_steps, 1),
            format_eta(time.perf_counter() - epoch_start),
            format_eta(time.perf_counter() - training_start),
        )

        if val_metrics["loss_blended"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss_blended"]
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in two_tower_model.state_dict().items()}
            logger.info("TwoTower validation improved | best_val_loss=%.4f", best_val_loss)
        else:
            stale_epochs += 1
            logger.info("TwoTower validation stale | stale_epochs=%s/%s", stale_epochs, early_stopping_patience)
            if stale_epochs >= early_stopping_patience:
                logger.info("TwoTower early stopping triggered | best_val_loss=%.4f", best_val_loss)
                break

    if best_state is not None:
        two_tower_model.load_state_dict(best_state)

    two_tower_model.eval()
    encoded_movie_ids = np.asarray(all_item_ids, dtype=np.int64)
    logger.info("TwoTower embedding export start | item_count=%s", len(encoded_movie_ids))
    item_embeddings, _ = _export_item_embeddings(
        two_tower_model,
        None,
        encoded_movie_ids,
        item_feature_arrays,
        device,
        export_batch_size,
        export_two_tower=True,
        export_sequence=False,
    )
    save_numpy(encoded_movie_ids, MOVIE_IDS_PATH)
    save_numpy(np.array(vocab_dict["movie_id"]), MOVIE_RAW_IDS_PATH)
    save_numpy(item_embeddings, ITEM_EMBEDDINGS_PATH)
    torch.save(
        {
            "state_dict": two_tower_model.state_dict(),
            "feature_dict": feature_dict,
            "emb_dim": emb_dim,
            "config_schema": "model_registry_v2" if "models" in settings else "legacy_v1",
            "model_settings": two_tower_settings,
            "training_settings": training_settings,
            "item_feature_fields": list(_ITEM_FEATURE_FIELDS),
        },
        RETRIEVAL_MODEL_PATH,
    )
    return {"model_path": str(RETRIEVAL_MODEL_PATH), "embedding_path": str(ITEM_EMBEDDINGS_PATH), "movie_ids_path": str(MOVIE_IDS_PATH)}



def train_sequence(context=None, warm_start: bool = True):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    settings = context["settings"]
    resolved_config = context["resolved_config"]
    sequence_settings = resolved_config["sequence_settings"]
    training_settings = resolved_config.get("sequence_training_settings", resolved_config["training_settings"])

    train_data = context["sequence_train_data"]
    test_data = context["test_data"]
    feature_dict = context["feature_dict"]
    vocab_dict = context["vocab_dict"]
    item_feature_arrays = context["item_feature_arrays"]
    all_item_ids = context["all_item_ids"]
    emb_dim = int(resolved_config["embedding_dim"])
    batch_size = int(training_settings.get("batch_size", 256))
    epochs = int(training_settings.get("epochs", 10))
    learning_rate = float(training_settings.get("learning_rate", 0.001))
    weight_decay = float(training_settings.get("weight_decay", 1e-5))
    log_every_n_batches = int(training_settings.get("log_every_n_batches", 100))
    early_stopping_patience = int(training_settings.get("early_stopping_patience", 2))
    min_delta = float(training_settings.get("min_delta", 1e-4))
    export_batch_size = int(training_settings.get("export_batch_size", 4096))
    sequence_num_negatives = int(training_settings.get("sequence_num_negatives", 32))
    sequence_user_negative_ratio = float(training_settings.get("sequence_user_negative_ratio", 0.75))
    sequence_loss = str(training_settings.get("sequence_loss", "bce")).strip().lower()
    sequence_history_feedback = str(training_settings.get("sequence_history_feedback", "positive")).strip().lower()

    device = context["device"]
    item_feature_tensors = to_item_feature_tensors(item_feature_arrays, device)
    all_item_id_tensor = torch.as_tensor(all_item_ids, dtype=torch.long, device=device)
    sequence_model = SequenceRetrievalModel(
        feature_dict,
        emb_dim,
        hidden_dim=int(sequence_settings.get("hidden_dim", emb_dim)),
        num_layers=int(sequence_settings.get("num_layers", 1)),
        max_len=int(sequence_settings.get("max_len", 10)),
        dropout=float(sequence_settings.get("dropout", 0.1)),
        multimodal_table=item_feature_arrays["multimodal_embedding"],
    ).to(device)

    if warm_start and SEQUENCE_MODEL_PATH.exists():
        sequence_checkpoint = torch.load(SEQUENCE_MODEL_PATH, map_location=device)
        sequence_model.load_state_dict(sequence_checkpoint["state_dict"])

    sequence_optimizer = torch.optim.Adam(sequence_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_samples = int(len(train_data["user_id"]))
    val_ratio = 0.1
    val_size = max(1, int(total_samples * val_ratio)) if total_samples > 1 else 0
    train_size = max(total_samples - val_size, 0)
    all_indices = np.arange(total_samples, dtype=np.int64)
    rng = np.random.default_rng(42)
    rng.shuffle(all_indices)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    logger.info(
        "Sequence training start | device=%s | train_samples=%s | val_samples=%s | test_users=%s | batch_size=%s | epochs=%s | export_batch_size=%s | sequence_num_negatives=%s | sequence_user_negative_ratio=%s | sequence_loss=%s | sequence_history_feedback=%s | warm_start=%s",
        device,
        train_size,
        val_size,
        len(test_data["user_id"]),
        batch_size,
        epochs,
        export_batch_size,
        sequence_num_negatives,
        sequence_user_negative_ratio,
        sequence_loss,
        sequence_history_feedback,
        warm_start,
    )

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    total_batches = max(int(np.ceil(train_size / max(batch_size, 1))), 1) if train_size > 0 else 1

    for epoch in range(epochs):
        sequence_model.train()
        total_seq_loss = 0.0
        seen_positions = 0
        epoch_start = time.perf_counter()
        train_batches = _index_batch_loader(train_indices, batch_size, shuffle=True) if train_size > 0 else []

        for batch_idx, batch_indices in enumerate(train_batches, start=1):
            batch_dict = _slice_train_batch(train_data, batch_indices, device)
            sequence_optimizer.zero_grad()
            loss_seq, supervised_positions = _compute_sequence_loss(
                sequence_model,
                batch_dict,
                item_feature_tensors,
                all_item_ids=all_item_id_tensor,
                num_negatives=sequence_num_negatives,
                user_negative_ratio=sequence_user_negative_ratio,
                loss_type=sequence_loss,
                history_feedback=sequence_history_feedback,
            )
            if supervised_positions == 0:
                continue
            loss_seq.backward()
            sequence_optimizer.step()

            total_seq_loss += float(loss_seq.item()) * supervised_positions
            seen_positions += supervised_positions

            should_log = batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
            if should_log:
                elapsed_epoch = time.perf_counter() - epoch_start
                avg_batch_time = elapsed_epoch / batch_idx
                remaining_batches = total_batches - batch_idx
                remaining_epochs = epochs - epoch - 1
                eta_seconds = avg_batch_time * (remaining_batches + remaining_epochs * total_batches)
                logger.info(
                    "Sequence epoch %s/%s | batch %s/%s | loss=%.4f | avg_loss=%.4f | supervised_positions=%s | eta=%s",
                    epoch + 1,
                    epochs,
                    batch_idx,
                    total_batches,
                    float(loss_seq.item()),
                    total_seq_loss / max(seen_positions, 1),
                    supervised_positions,
                    format_eta(eta_seconds),
                )

        train_seq_loss = total_seq_loss / max(seen_positions, 1)
        val_metrics = _evaluate_loss(
            None,
            sequence_model,
            train_data,
            val_indices,
            device,
            item_feature_arrays,
            batch_size=batch_size,
            evaluate_two_tower=False,
            evaluate_sequence=True,
            all_item_ids=all_item_id_tensor,
            sequence_num_negatives=sequence_num_negatives,
            sequence_user_negative_ratio=sequence_user_negative_ratio,
            sequence_loss=sequence_loss,
            sequence_history_feedback=sequence_history_feedback,
        )
        logger.info(
            "Sequence epoch %s/%s done | train_loss=%.4f | val_loss=%.4f | epoch_time=%s | elapsed=%s",
            epoch + 1,
            epochs,
            train_seq_loss,
            val_metrics["loss_blended"],
            format_eta(time.perf_counter() - epoch_start),
            format_eta(time.perf_counter() - training_start),
        )

        if val_metrics["loss_blended"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss_blended"]
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in sequence_model.state_dict().items()}
            logger.info("Sequence validation improved | best_val_loss=%.4f", best_val_loss)
        else:
            stale_epochs += 1
            logger.info("Sequence validation stale | stale_epochs=%s/%s", stale_epochs, early_stopping_patience)
            if stale_epochs >= early_stopping_patience:
                logger.info("Sequence early stopping triggered | best_val_loss=%.4f", best_val_loss)
                break

    if best_state is not None:
        sequence_model.load_state_dict(best_state)

    sequence_model.eval()
    encoded_movie_ids = np.asarray(all_item_ids, dtype=np.int64)
    logger.info("Sequence embedding export start | item_count=%s", len(encoded_movie_ids))
    _, sequence_item_embeddings = _export_item_embeddings(
        None,
        sequence_model,
        encoded_movie_ids,
        item_feature_arrays,
        device,
        export_batch_size,
        export_two_tower=False,
        export_sequence=True,
    )
    save_numpy(encoded_movie_ids, MOVIE_IDS_PATH)
    save_numpy(np.array(vocab_dict["movie_id"]), MOVIE_RAW_IDS_PATH)
    save_numpy(sequence_item_embeddings, SEQUENCE_ITEM_EMBEDDINGS_PATH)
    torch.save(
        {
            "state_dict": sequence_model.state_dict(),
            "feature_dict": feature_dict,
            "emb_dim": emb_dim,
            "config_schema": "model_registry_v2" if "models" in settings else "legacy_v1",
            "model_settings": sequence_settings,
            "training_settings": training_settings,
            "item_feature_fields": list(_ITEM_FEATURE_FIELDS),
        },
        SEQUENCE_MODEL_PATH,
    )
    return {"model_path": str(SEQUENCE_MODEL_PATH), "embedding_path": str(SEQUENCE_ITEM_EMBEDDINGS_PATH), "movie_ids_path": str(MOVIE_IDS_PATH)}



def train_item2item(context=None, warm_start: bool = True):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    resolved_config = context["resolved_config"]
    item2item_settings = resolved_config["item2item_settings"]
    if not bool(item2item_settings.get("enabled", True)):
        logger.info("Item2Item training skipped | enabled=false")
        return {"skipped": True, "reason": "disabled"}

    init_embeddings = None
    if warm_start and ITEM2ITEM_ITEM_EMBEDDINGS_PATH.exists():
        init_embeddings = np.load(ITEM2ITEM_ITEM_EMBEDDINGS_PATH)
    encoded_movie_ids = np.asarray(context["all_item_ids"], dtype=np.int64)
    logger.info("Item2Item training start | item_count=%s | warm_start=%s", len(encoded_movie_ids), warm_start)
    item2item_embeddings, item2item_metadata = train_item2item_embeddings(
        train_data=context["train_data"],
        all_item_ids=np.asarray(context["all_item_ids"], dtype=np.int64),
        settings=item2item_settings,
        device=context["device"],
        init_embeddings=init_embeddings,
    )
    save_numpy(item2item_embeddings, ITEM2ITEM_ITEM_EMBEDDINGS_PATH)
    save_numpy(encoded_movie_ids, ITEM2ITEM_MOVIE_IDS_PATH)
    torch.save(item2item_metadata, ITEM2ITEM_MODEL_PATH)
    logger.info(
        "Item2Item training done | embedding_dim=%s | pairs=%s | negative_sampling=%s",
        item2item_embeddings.shape[1] if item2item_embeddings.ndim == 2 and item2item_embeddings.size else 0,
        item2item_metadata.get("pair_count", 0),
        item2item_metadata.get("negative_sampling"),
    )
    return {
        "model_path": str(ITEM2ITEM_MODEL_PATH),
        "embedding_path": str(ITEM2ITEM_ITEM_EMBEDDINGS_PATH),
        "movie_ids_path": str(ITEM2ITEM_MOVIE_IDS_PATH),
        "metadata": item2item_metadata,
    }



def train_item_cf(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    train_data = dict(context["train_data"])
    train_data["rating_weighting_settings"] = context["resolved_config"].get("training_settings", {})
    model = _build_item_cf_model(train_data)
    save_pickle(model, ITEM_CF_MODEL_PATH)
    return {"model_path": str(ITEM_CF_MODEL_PATH), "anchor_items": len(model)}



def train_genre(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    genre_to_items = defaultdict(list)
    item_features = context["item_feature_arrays"]
    genre_matrix = np.asarray(item_features["genres"], dtype=np.int32)
    for item_id in context["all_item_ids"].tolist():
        valid_genres = [int(idx + 1) for idx in np.flatnonzero(genre_matrix[item_id]).tolist()]
        for genre_id in valid_genres:
            genre_to_items[int(genre_id)].append(int(item_id))
    save_pickle(dict(genre_to_items), GENRE_MODEL_PATH)
    return {"model_path": str(GENRE_MODEL_PATH), "genre_count": len(genre_to_items)}



def train_popular(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = _load_retrieval_context(settings, resolve_retrieval_config(settings))
    popularity = Counter(int(np.asarray(movie_id).reshape(-1)[0]) for movie_id in context["train_data"]["movie_id"])
    ranked = dict(popularity.most_common(len(popularity)))
    save_pickle(ranked, POPULAR_MODEL_PATH)
    return {"model_path": str(POPULAR_MODEL_PATH), "item_count": len(ranked)}



def evaluate_retrieval(route: str = "two_tower", topk: int | None = None):
    normalized_route = route.strip().lower()
    if normalized_route not in {"two_tower", "sequence"}:
        raise ValueError("Retrieval evaluation only supports routes 'two_tower' and 'sequence'")
    settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_retrieval_config(settings)
    resolved_topk = int(topk or resolved_config["evaluation_settings"].get("topk", 200))

    if normalized_route == "two_tower":
        if not RETRIEVAL_MODEL_PATH.exists() or not ITEM_EMBEDDINGS_PATH.exists() or not MOVIE_IDS_PATH.exists():
            raise FileNotFoundError("Missing two_tower retrieval artifacts for evaluation")
        checkpoint_path = RETRIEVAL_MODEL_PATH
        embedding_path = ITEM_EMBEDDINGS_PATH
        model_cls = TwoTowerRetrievalModel
    else:
        if not SEQUENCE_MODEL_PATH.exists() or not SEQUENCE_ITEM_EMBEDDINGS_PATH.exists() or not MOVIE_IDS_PATH.exists():
            raise FileNotFoundError("Missing sequence retrieval artifacts for evaluation")
        checkpoint_path = SEQUENCE_MODEL_PATH
        embedding_path = SEQUENCE_ITEM_EMBEDDINGS_PATH
        model_cls = SequenceRetrievalModel

    train_eval_samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
    feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_settings = checkpoint.get("model_settings", {})
    training_settings = checkpoint.get("training_settings", {})
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    item_feature_arrays = get_item_feature_arrays(item_catalog)
    if normalized_route == "two_tower":
        model = model_cls(
            feature_dict,
            int(checkpoint["emb_dim"]),
            user_hidden_dims=model_settings.get("user_hidden_dims"),
            item_hidden_dims=model_settings.get("item_hidden_dims"),
            dropout=float(model_settings.get("dropout", 0.1)),
            rating_weighting_enabled=bool(training_settings.get("rating_weighting_enabled", False)),
            rating_weight_neutral=float(training_settings.get("rating_weight_neutral", 3.0)),
            rating_weight_scale=float(training_settings.get("rating_weight_scale", 0.25)),
            rating_weight_min=float(training_settings.get("rating_weight_min", 0.0)),
            rating_weight_max=float(training_settings.get("rating_weight_max", 1.0)),
            short_history_length=int(model_settings.get("short_history_length", 10)),
            positive_rating_min=float(training_settings.get("positive_rating_min", 4.0)),
            multimodal_table=item_feature_arrays["multimodal_embedding"],
        ).to(device)
    else:
        model = model_cls(
            feature_dict,
            int(checkpoint.get("emb_dim", model_settings.get("embedding_dim", 32))),
            hidden_dim=int(model_settings.get("hidden_dim", checkpoint.get("emb_dim", model_settings.get("embedding_dim", 32)))),
            num_layers=int(model_settings.get("num_layers", 1)),
            max_len=int(model_settings.get("max_len", model_settings.get("sequence_max_len", 10))),
            dropout=float(model_settings.get("dropout", model_settings.get("sequence_dropout", 0.1))),
            multimodal_table=item_feature_arrays["multimodal_embedding"],
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    encoded_movie_ids = np.load(MOVIE_IDS_PATH, mmap_mode="r").astype(np.int64)
    item_embeddings = np.load(embedding_path, mmap_mode="r")
    history_feedback = str(training_settings.get("sequence_history_feedback", "positive")).strip().lower()
    metrics = _evaluate_retrieval(model, train_eval_samples["test"], encoded_movie_ids, item_embeddings, device, resolved_topk, route=normalized_route, sequence_history_feedback=history_feedback, item_feature_arrays=item_feature_arrays)
    all_metrics = {}
    if RETRIEVAL_METRICS_PATH.exists():
        all_metrics = json.loads(RETRIEVAL_METRICS_PATH.read_text(encoding="utf-8"))
    if not isinstance(all_metrics, dict):
        all_metrics = {}
    all_metrics[normalized_route] = metrics
    save_metrics(RETRIEVAL_METRICS_PATH, all_metrics)
    logger.info("Retrieval metrics saved | route=%s | metrics=%s", normalized_route, metrics)
    return metrics



def _load_retrieval_context(settings: dict, resolved_config: dict) -> dict:
    train_eval_samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
    feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
    vocab_dict = load_pickle(RETRIEVAL_VOCAB_DICT_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    item_feature_arrays = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog).astype(np.int64)
    missing_fields = [field for field in _ITEM_FEATURE_FIELDS[1:] if field not in item_feature_arrays]
    if missing_fields:
        raise ValueError(
            f"Global item catalog is missing item feature fields: {missing_fields}. "
            f"Run ranking preprocessing to regenerate {ITEM_CATALOG_PATH.name}."
        )
    sequence_train_data = train_eval_samples.get("sequence_train", train_eval_samples["train"])
    required_sequence_fields = [
        "user_id",
        "hist_movie_id",
        "hist_recency_bucket",
        "hist_feedback",
        "movie_id",
        "rating",
        "user_negative_movie_id",
    ]
    missing_sequence_fields = [field for field in required_sequence_fields if field not in sequence_train_data]
    if missing_sequence_fields:
        raise ValueError(
            "Sequence training artifacts use an outdated schema. "
            f"Missing fields: {missing_sequence_fields}. Run retrieval preprocessing again."
        )
    return {
        "settings": settings,
        "resolved_config": resolved_config,
        "train_eval_samples": train_eval_samples,
        "feature_dict": feature_dict,
        "vocab_dict": vocab_dict,
        "item_catalog": item_catalog,
        "item_feature_arrays": item_feature_arrays,
        "all_item_ids": all_item_ids,
        "train_data": train_eval_samples["train"],
        "sequence_train_data": sequence_train_data,
        "test_data": train_eval_samples["test"],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }



def _export_item_embeddings(
    two_tower_model,
    sequence_model,
    encoded_movie_ids: np.ndarray,
    item_feature_arrays: dict[str, np.ndarray],
    device: torch.device,
    export_batch_size: int,
    export_two_tower: bool = True,
    export_sequence: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    two_tower_chunks: list[np.ndarray] = []
    sequence_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(encoded_movie_ids), max(export_batch_size, 1)):
            batch_ids_np = encoded_movie_ids[start : start + max(export_batch_size, 1)]
            batch_ids = torch.as_tensor(batch_ids_np, dtype=torch.long, device=device)
            item_batch = build_item_batch(batch_ids, item_feature_arrays, device)
            if export_two_tower:
                two_tower_chunks.append(two_tower_model.encode_item(item_batch).cpu().numpy())
            if export_sequence:
                sequence_chunks.append(sequence_model.encode_item(item_batch).cpu().numpy())
    two_tower_embeddings = np.concatenate(two_tower_chunks, axis=0) if two_tower_chunks else None
    sequence_embeddings = np.concatenate(sequence_chunks, axis=0) if sequence_chunks else None
    return two_tower_embeddings, sequence_embeddings



def _build_training_logits(
    user_repr: torch.Tensor,
    positive_item_repr: torch.Tensor,
    positive_item_ids: torch.Tensor,
    hist_movie_ids: torch.Tensor,
    memory_queue: ItemMemoryQueue | None,
    hard_negative_topk: int,
    use_hard_negatives: bool,
    temperature: float,
    sampled_negative_ids: torch.Tensor | None = None,
    sampled_negative_repr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (user_repr @ positive_item_repr.T) / temperature
    duplicate_positive_mask = positive_item_ids.unsqueeze(0).eq(positive_item_ids.unsqueeze(1))
    in_batch_history_mask = hist_movie_ids.unsqueeze(2).eq(positive_item_ids.unsqueeze(0).unsqueeze(0)).any(dim=1)
    invalid_in_batch_mask = duplicate_positive_mask | in_batch_history_mask
    invalid_in_batch_mask.fill_diagonal_(False)
    logits = logits.masked_fill(invalid_in_batch_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(logits.size(0), device=logits.device)
    extra_logits = []

    if sampled_negative_ids is not None and sampled_negative_repr is not None and sampled_negative_ids.numel() > 0:
        sampled_scores = torch.einsum("bd,bnd->bn", user_repr, sampled_negative_repr) / temperature
        sampled_invalid_mask = sampled_negative_ids.le(0)
        sampled_invalid_mask |= sampled_negative_ids.eq(positive_item_ids.unsqueeze(1))
        sampled_invalid_mask |= hist_movie_ids.unsqueeze(2).eq(sampled_negative_ids.unsqueeze(1)).any(dim=1)
        sampled_scores = sampled_scores.masked_fill(sampled_invalid_mask, torch.finfo(sampled_scores.dtype).min)
        extra_logits.append(sampled_scores)

    if use_hard_negatives and memory_queue is not None:
        queue_item_ids, queue_item_embeddings = memory_queue.get()
        if queue_item_ids is not None and queue_item_embeddings is not None and queue_item_ids.numel() > 0:
            hard_scores = (user_repr @ queue_item_embeddings.T) / temperature
            invalid_mask = queue_item_ids.unsqueeze(0).eq(positive_item_ids.unsqueeze(1))
            invalid_mask |= hist_movie_ids.unsqueeze(2).eq(queue_item_ids.unsqueeze(0).unsqueeze(0)).any(dim=1)
            hard_scores = hard_scores.masked_fill(invalid_mask, torch.finfo(hard_scores.dtype).min)
            topk = min(hard_negative_topk, hard_scores.size(1))
            hard_values, _ = torch.topk(hard_scores, k=topk, dim=1)
            valid_hard_mask = torch.isfinite(hard_values) & (hard_values > torch.finfo(hard_values.dtype).min / 2)
            hard_values = torch.where(valid_hard_mask, hard_values, torch.full_like(hard_values, torch.finfo(hard_values.dtype).min))
            extra_logits.append(hard_values)

    if extra_logits:
        logits = torch.cat([logits, *extra_logits], dim=1)
    return logits, labels



def _sample_context_negative_ids(
    positive_ids: torch.Tensor,
    context_ids: torch.Tensor,
    all_item_ids: torch.Tensor,
    num_negatives: int,
    candidate_pools: torch.Tensor | None = None,
    candidate_pool_ratio: float = 0.0,
) -> torch.Tensor:
    batch_size = positive_ids.size(0)
    device = positive_ids.device
    if batch_size == 0 or num_negatives <= 0 or all_item_ids.numel() == 0:
        return torch.zeros((batch_size, max(num_negatives, 0)), dtype=torch.long, device=device)

    blocked_ids = torch.cat([context_ids.long(), positive_ids.long().unsqueeze(1)], dim=1)
    blocked_mask = blocked_ids.gt(0)
    packed_ids = torch.zeros((batch_size, num_negatives), dtype=torch.long, device=device)
    filled_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

    preferred_width = 0
    if candidate_pools is not None and candidate_pool_ratio > 0:
        preferred_width = min(candidate_pools.size(1), max(0, int(round(num_negatives * candidate_pool_ratio))))
    if preferred_width > 0:
        preferred_candidates = candidate_pools[:, -preferred_width:].long()
        preferred_valid = preferred_candidates.gt(0)
        preferred_valid &= _valid_candidate_mask(preferred_candidates, blocked_ids, blocked_mask)
        preferred_valid &= _first_occurrence_mask(preferred_candidates)
        preferred_selection = preferred_valid & (preferred_valid.cumsum(dim=1) <= num_negatives)
        packed_ids = _pack_candidate_ids(preferred_candidates, preferred_selection, num_negatives)
        filled_counts = preferred_selection.sum(dim=1).clamp_max(num_negatives)

    packed_mask = torch.arange(num_negatives, device=device).unsqueeze(0) < filled_counts.unsqueeze(1)
    remaining_needed = num_negatives - filled_counts
    random_width = max(num_negatives * 16, num_negatives + context_ids.size(1) + 32)
    random_candidates = all_item_ids[torch.randint(0, all_item_ids.numel(), (batch_size, random_width), device=device)]
    random_blocked_ids = torch.cat([blocked_ids, packed_ids], dim=1)
    random_blocked_mask = torch.cat([blocked_mask, packed_mask], dim=1)
    random_valid = _valid_candidate_mask(random_candidates, random_blocked_ids, random_blocked_mask)
    random_valid &= _first_occurrence_mask(random_candidates)
    random_selection = random_valid & (random_valid.cumsum(dim=1) <= remaining_needed.unsqueeze(1))
    random_counts = random_selection.sum(dim=1).clamp_max(num_negatives)
    random_packed = _pack_candidate_ids(random_candidates, random_selection, num_negatives)

    combined_ids = torch.cat([packed_ids, random_packed], dim=1)
    combined_mask = torch.cat([
        packed_mask,
        torch.arange(num_negatives, device=device).unsqueeze(0) < random_counts.unsqueeze(1),
    ], dim=1)
    sampled_ids = _pack_candidate_ids(combined_ids, combined_mask, num_negatives)
    return sampled_ids



def _compute_sequence_loss(
    sequence_model: SequenceRetrievalModel,
    batch_dict: dict[str, torch.Tensor],
    item_feature_tensors: dict[str, torch.Tensor],
    all_item_ids: torch.Tensor,
    num_negatives: int,
    user_negative_ratio: float,
    loss_type: str,
    history_feedback: str,
) -> tuple[torch.Tensor, int]:
    positive_ids = batch_dict["movie_id"]
    valid_mask = positive_ids.gt(0)
    if not valid_mask.any():
        return torch.zeros((), device=batch_dict["hist_movie_id"].device), 0

    if not valid_mask.all():
        positive_ids = positive_ids[valid_mask]
        hist_movie_ids = batch_dict["hist_movie_id"][valid_mask]
        context_movie_ids = hist_movie_ids
        hist_recency_bucket = batch_dict.get("hist_recency_bucket")
        hist_feedback = batch_dict.get("hist_feedback")
        user_negative_ids = batch_dict["user_negative_movie_id"][valid_mask]
        if hist_recency_bucket is not None:
            hist_recency_bucket = hist_recency_bucket[valid_mask]
        if hist_feedback is not None:
            hist_feedback = hist_feedback[valid_mask]
    else:
        hist_movie_ids = batch_dict["hist_movie_id"]
        context_movie_ids = hist_movie_ids
        hist_recency_bucket = batch_dict.get("hist_recency_bucket")
        hist_feedback = batch_dict.get("hist_feedback")
        user_negative_ids = batch_dict["user_negative_movie_id"]

    hist_movie_ids, hist_recency_bucket, hist_feedback = filter_sequence_history(
        hist_movie_ids,
        hist_recency_bucket,
        hist_feedback,
        history_feedback,
    )
    hist_item_features = build_item_batch(hist_movie_ids, item_feature_tensors, hist_movie_ids.device)
    hidden_states = sequence_model.encode_user(
        hist_movie_ids,
        hist_recency_bucket,
        hist_feedback,
        {field: value for field, value in hist_item_features.items() if field != "movie_id"},
    )
    negative_ids = _sample_context_negative_ids(
        positive_ids=positive_ids,
        context_ids=context_movie_ids,
        all_item_ids=all_item_ids,
        num_negatives=num_negatives,
        candidate_pools=user_negative_ids,
        candidate_pool_ratio=user_negative_ratio,
    )
    positive_emb = sequence_model.encode_item(build_item_batch(positive_ids, item_feature_tensors, positive_ids.device))
    negative_emb = sequence_model.encode_item(build_item_batch(negative_ids.reshape(-1), item_feature_tensors, negative_ids.device)).reshape(negative_ids.size(0), negative_ids.size(1), -1)
    positive_logits = (hidden_states * positive_emb).sum(dim=1)
    negative_logits = torch.einsum("bd,bnd->bn", hidden_states, negative_emb)
    negative_mask = negative_ids.gt(0)
    if loss_type == "softmax":
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits.masked_fill(~negative_mask, torch.finfo(negative_logits.dtype).min)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels), int(logits.size(0))

    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits), reduction="sum")
    if negative_mask.any():
        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(negative_logits[negative_mask], torch.zeros_like(negative_logits[negative_mask]), reduction="sum")
        denominator = positive_logits.numel() + int(negative_mask.sum().item())
        loss = (positive_loss + negative_loss) / max(denominator, 1)
    else:
        loss = positive_loss / max(int(positive_logits.numel()), 1)
    return loss, int(positive_logits.numel())



def _evaluate_loss(
    two_tower_model,
    sequence_model,
    train_data: dict[str, np.ndarray],
    eval_indices: np.ndarray,
    device: torch.device,
    item_feature_arrays: dict[str, np.ndarray] | dict[str, torch.Tensor],
    batch_size: int,
    evaluate_two_tower: bool = True,
    evaluate_sequence: bool = True,
    all_item_ids: torch.Tensor | None = None,
    sequence_num_negatives: int = 0,
    sequence_user_negative_ratio: float = 0.0,
    sequence_loss: str = "bce",
    sequence_history_feedback: str = "positive",
    two_tower_temperature: float = 1.0,
) -> dict[str, float]:
    if evaluate_two_tower and two_tower_model is None:
        raise ValueError("two_tower_model is required when evaluate_two_tower=True")
    if evaluate_sequence and sequence_model is None:
        raise ValueError("sequence_model is required when evaluate_sequence=True")
    if two_tower_model is not None:
        two_tower_model.eval()
    if sequence_model is not None:
        sequence_model.eval()
    total_tt_loss, total_seq_loss = 0.0, 0.0
    total_tt_examples, total_seq_examples = 0, 0
    eval_batches = _index_batch_loader(eval_indices, batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, batch_indices in enumerate(eval_batches, start=1):
            batch_dict = _slice_train_batch(train_data, batch_indices, device)
            if evaluate_two_tower:
                hist_item_batch = build_item_batch(batch_dict["hist_movie_id"], item_feature_arrays, device)
                batch_dict.update({f"hist_{field}": value for field, value in hist_item_batch.items() if field != "movie_id"})
                batch_dict["item"] = build_item_batch(batch_dict["movie_id"], item_feature_arrays, device)
            loss_tt = torch.tensor(0.0, device=device)
            loss_seq = torch.tensor(0.0, device=device)
            if evaluate_two_tower:
                user_repr_tt = two_tower_model.encode_user(batch_dict)
                item_repr_tt = two_tower_model.encode_item(batch_dict["item"])
                logits_tt, labels_tt = _build_training_logits(
                    user_repr_tt,
                    item_repr_tt,
                    batch_dict["movie_id"],
                    batch_dict["hist_movie_id"],
                    None,
                    hard_negative_topk=0,
                    use_hard_negatives=False,
                    temperature=two_tower_temperature,
                )
                loss_tt = torch.nn.functional.cross_entropy(logits_tt, labels_tt)
            if evaluate_sequence:
                if all_item_ids is None:
                    raise ValueError("all_item_ids is required when evaluate_sequence=True")
                loss_seq, supervised_positions = _compute_sequence_loss(
                    sequence_model,
                    batch_dict,
                    item_feature_arrays,
                    all_item_ids=all_item_ids,
                    num_negatives=sequence_num_negatives,
                    user_negative_ratio=sequence_user_negative_ratio,
                    loss_type=sequence_loss,
                    history_feedback=sequence_history_feedback,
                )
                if supervised_positions == 0:
                    continue
            loss_blended = 0.5 * (loss_tt + loss_seq) if evaluate_two_tower and evaluate_sequence else (loss_tt + loss_seq)
            if not torch.isfinite(loss_blended):
                logger.warning(
                    "Retrieval validation produced non-finite loss | batch=%s | two_tower_enabled=%s | sequence_enabled=%s | two_tower_loss=%s | sequence_loss=%s",
                    batch_idx,
                    evaluate_two_tower,
                    evaluate_sequence,
                    float(loss_tt.detach().cpu().item()) if torch.isfinite(loss_tt) else float("nan"),
                    float(loss_seq.detach().cpu().item()) if torch.isfinite(loss_seq) else float("nan"),
                )
                continue
            batch_size_now = batch_dict["user_id"].size(0)
            total_tt_loss += float(loss_tt.item()) * batch_size_now
            total_tt_examples += batch_size_now
            if evaluate_sequence:
                total_seq_loss += float(loss_seq.item()) * supervised_positions
                total_seq_examples += supervised_positions
            else:
                total_seq_loss += float(loss_seq.item()) * batch_size_now
                total_seq_examples += batch_size_now
    if total_tt_examples == 0 and total_seq_examples == 0:
        return {"loss_tt": float("inf"), "loss_seq": float("inf"), "loss_blended": float("inf")}
    avg_tt = total_tt_loss / total_tt_examples if evaluate_two_tower and total_tt_examples > 0 else 0.0
    avg_seq = total_seq_loss / total_seq_examples if evaluate_sequence and total_seq_examples > 0 else 0.0
    blended = 0.5 * (avg_tt + avg_seq) if evaluate_two_tower and evaluate_sequence else (avg_tt + avg_seq)
    return {"loss_tt": avg_tt, "loss_seq": avg_seq, "loss_blended": blended}



def _evaluate_retrieval(model, test_data, encoded_movie_ids, item_embeddings, device: torch.device, topk: int, route: str = "two_tower", sequence_history_feedback: str = "positive", item_feature_arrays: dict[str, np.ndarray] | None = None) -> dict:
    labels, scores, user_ids = [], [], []
    item_embeddings = np.asarray(item_embeddings)
    item_feature_tensors = to_item_feature_tensors(item_feature_arrays, device) if item_feature_arrays is not None else None
    with torch.no_grad():
        for idx, user_id in enumerate(test_data["user_id"]):
            batch = {
                "user_id": torch.tensor([int(user_id)], dtype=torch.long, device=device),
                "age": torch.tensor([int(test_data["age"][idx])], dtype=torch.long, device=device),
                "gender": torch.tensor([int(test_data["gender"][idx])], dtype=torch.long, device=device),
                "occupation": torch.tensor([int(test_data["occupation"][idx])], dtype=torch.long, device=device),
                "zip_code": torch.tensor([int(test_data["zip_code"][idx])], dtype=torch.long, device=device),
                "hist_movie_id": torch.tensor(np.asarray(test_data["hist_movie_id"][idx]).reshape(1, -1), dtype=torch.long, device=device),
                "hist_recency_bucket": torch.tensor(np.asarray(test_data["hist_recency_bucket"][idx]).reshape(1, -1), dtype=torch.long, device=device),
                "hist_rating": torch.tensor(np.asarray(test_data["hist_rating"][idx]).reshape(1, -1), dtype=torch.float32, device=device),
            }
            if route == "sequence":
                batch["hist_feedback"] = torch.tensor(np.asarray(test_data["hist_feedback"][idx]).reshape(1, -1), dtype=torch.long, device=device)
                hist_movie_ids, hist_recency_bucket, hist_feedback = filter_sequence_history(
                    batch["hist_movie_id"],
                    batch["hist_recency_bucket"],
                    batch["hist_feedback"],
                    sequence_history_feedback,
                )
                hist_item_features = build_item_batch(hist_movie_ids, item_feature_tensors, device)
                user_embedding = model.encode_user(
                    hist_movie_ids,
                    hist_recency_bucket,
                    hist_feedback,
                    {field: value for field, value in hist_item_features.items() if field != "movie_id"},
                ).cpu().numpy()[0]
            else:
                hist_item_batch = build_item_batch(batch["hist_movie_id"], item_feature_tensors, device)
                batch.update({f"hist_{field}": value for field, value in hist_item_batch.items() if field != "movie_id"})
                user_embedding = model.encode_user(batch).cpu().numpy()[0]
            candidate_scores = item_embeddings @ user_embedding
            top_indices = np.argsort(candidate_scores)[::-1][:topk]
            target = int(np.asarray(test_data["movie_id"][idx]).reshape(-1)[0])
            for item_idx in top_indices:
                labels.append(1.0 if int(encoded_movie_ids[item_idx]) == target else 0.0)
                scores.append(float(candidate_scores[item_idx]))
                user_ids.append(int(user_id))
    labels = np.array(labels)
    scores = np.array(scores)
    user_ids = np.array(user_ids)
    metrics = {}
    for k in (50, 100, 200):
        metrics[f"recall@{k}"] = recall_at_k(labels, scores, user_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(labels, scores, user_ids, k)
        metrics[f"hr@{k}"] = hit_rate_at_k(labels, scores, user_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, scores, user_ids, k)
    return metrics



def _build_item_cf_model(train_data):
    logger.info("ItemCF build start | train_samples=%s", len(train_data["movie_id"]))
    transitions = defaultdict(Counter)
    rating_settings = _resolve_rating_weighting_settings(train_data.get("rating_weighting_settings"))
    hist_ratings = train_data.get("hist_rating")
    target_ratings = train_data.get("rating")
    hist_recency_buckets = train_data.get("hist_recency_bucket")
    for idx, (history, target) in enumerate(zip(train_data["hist_movie_id"], train_data["movie_id"])):
        target_id = int(np.asarray(target).reshape(-1)[0])
        target_rating = float(np.asarray(target_ratings[idx]).reshape(-1)[0]) if target_ratings is not None else float(rating_settings["neutral"])
        target_weight = _rating_weight(target_rating, rating_settings)
        valid_history = [int(x) for x in np.asarray(history).reshape(-1) if int(x) > 0]
        if not valid_history:
            continue
        history_tail = valid_history[-3:]
        hist_rating_tail = []
        hist_recency_tail = []
        if hist_ratings is not None:
            hist_rating_tail = [float(x) for x in np.asarray(hist_ratings[idx]).reshape(-1) if float(x) > 0][-len(history_tail):]
        if hist_recency_buckets is not None:
            hist_recency_tail = [int(x) for x in np.asarray(hist_recency_buckets[idx]).reshape(-1) if int(x) > 0][-len(history_tail):]
        if len(hist_rating_tail) < len(history_tail):
            hist_rating_tail = [float(rating_settings["neutral"])] * (len(history_tail) - len(hist_rating_tail)) + hist_rating_tail
        if len(hist_recency_tail) < len(history_tail):
            hist_recency_tail = [1] * (len(history_tail) - len(hist_recency_tail)) + hist_recency_tail
        for movie_id, hist_rating, hist_bucket in zip(history_tail, hist_rating_tail, hist_recency_tail, strict=False):
            recency_weight = 1.0 / max(int(hist_bucket), 1)
            edge_weight = _rating_weight(hist_rating, rating_settings) * target_weight * recency_weight
            transitions[movie_id][target_id] += float(edge_weight)
    total_edges = sum(len(counter) for counter in transitions.values())
    logger.info("ItemCF build done | anchor_items=%s | transition_edges=%s | rating_weighting=%s", len(transitions), total_edges, bool(rating_settings["enabled"]))
    return transitions
