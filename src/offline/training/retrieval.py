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
from offline.features.item_batch import ITEM_FEATURE_FIELDS, build_item_batch, item_feature_tensors as to_item_feature_tensors
from offline.models.sequence_retrieval import SequenceRetrievalModel, filter_sequence_history
from offline.models.two_tower import TwoTowerRetrievalModel
from offline.ranking.protocol import get_item_feature_arrays
from offline.training.retrieval_common import batch_indices, gradient_norm, load_retrieval_context, rating_weight, slice_train_batch
from offline.utils.config import resolve_retrieval_config, resolve_retrieval_route_names
from offline.utils.io import (
    CONFIG_DIR,
    GENRE_MODEL_PATH,
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
    SEQUENCE_ITEM_EMBEDDINGS_PATH,
    SEQUENCE_MODEL_PATH,
    load_pickle,
    save_numpy,
    save_pickle,
)
from offline.utils.logging import format_eta, get_logger


logger = get_logger("offline.training.retrieval")






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
    context = load_retrieval_context(settings, resolved_config)
    results: dict[str, dict] = {}

    if "two_tower" in selected_routes:
        results["two_tower"] = train_two_tower(context=context, warm_start=warm_start)
    if "sequence" in selected_routes:
        results["sequence"] = train_sequence(context=context, warm_start=warm_start)

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
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
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
        multimodal_table=item_feature_arrays["multimodal_embedding"],
        recent_history_length=int(two_tower_settings.get("recent_history_length", 20)),
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

    use_hard_negatives = negative_sampling in {"in_batch_hard", "mixed"} and hard_negative_topk > 0
    use_sampled_negatives = negative_sampling in {"sampled", "mixed"} and two_tower_num_sampled_negatives > 0

    logger.info(
        "TwoTower training start | device=%s | train_samples=%s | val_samples=%s | test_users=%s | batch_size=%s | epochs=%s | negative_sampling=%s | low_rating_hard_negatives=%s | sampled_negatives=%s | export_batch_size=%s | temperature=%s | warm_start=%s",
        device,
        train_size,
        val_size,
        len(test_data["user_id"]),
        batch_size,
        epochs,
        negative_sampling,
        hard_negative_topk if use_hard_negatives else 0,
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
        train_batches = batch_indices(train_indices, batch_size, shuffle=True)

        for batch_idx, indices in enumerate(train_batches, start=1):
            batch_dict = slice_train_batch(train_data, indices, device)
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
            low_rating_negative_ids = None
            low_rating_negative_repr = None
            if use_hard_negatives:
                low_rating_negative_ids = _sample_user_negative_ids(
                    user_negative_ids=batch_dict["user_negative_movie_id"],
                    positive_ids=batch_dict["movie_id"],
                    context_ids=batch_dict["hist_movie_id"],
                    num_negatives=hard_negative_topk,
                )
                low_rating_negative_batch = build_item_batch(low_rating_negative_ids.reshape(-1), item_feature_tensors, device)
                low_rating_negative_repr = two_tower_model.encode_item(low_rating_negative_batch).reshape(
                    low_rating_negative_ids.size(0),
                    low_rating_negative_ids.size(1),
                    -1,
                )
            logits_tt, labels_tt = _build_training_logits(
                user_repr_tt,
                item_repr_tt,
                batch_dict["movie_id"],
                batch_dict["hist_movie_id"],
                two_tower_temperature,
                sampled_negative_ids=sampled_negative_ids,
                sampled_negative_repr=sampled_negative_repr,
                low_rating_negative_ids=low_rating_negative_ids,
                low_rating_negative_repr=low_rating_negative_repr,
            )
            loss_tt = torch.nn.functional.cross_entropy(logits_tt, labels_tt)
            loss_tt.backward()
            should_log = batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
            grad_norm_tt = gradient_norm(two_tower_model.parameters()) if should_log else 0.0
            two_tower_optimizer.step()

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
            "item_feature_fields": list(ITEM_FEATURE_FIELDS),
        },
        RETRIEVAL_MODEL_PATH,
    )
    return {"model_path": str(RETRIEVAL_MODEL_PATH), "embedding_path": str(ITEM_EMBEDDINGS_PATH), "movie_ids_path": str(MOVIE_IDS_PATH)}



def train_sequence(context=None, warm_start: bool = True):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
    settings = context["settings"]
    resolved_config = context["resolved_config"]
    sequence_settings = resolved_config["sequence_settings"]
    training_settings = resolved_config.get("sequence_training_settings", resolved_config["training_settings"])

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
        train_batches = batch_indices(train_indices, batch_size, shuffle=True) if train_size > 0 else []

        for batch_idx, indices in enumerate(train_batches, start=1):
            batch_dict = slice_train_batch(train_data, indices, device)
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
            "item_feature_fields": list(ITEM_FEATURE_FIELDS),
        },
        SEQUENCE_MODEL_PATH,
    )
    return {"model_path": str(SEQUENCE_MODEL_PATH), "embedding_path": str(SEQUENCE_ITEM_EMBEDDINGS_PATH), "movie_ids_path": str(MOVIE_IDS_PATH)}



def train_item_cf(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
    train_data = context["train_data"]
    logger.info("ItemCF build start | samples=%s", len(train_data["user_id"]))
    transitions = defaultdict(Counter)
    rating_settings = context["resolved_config"].get("training_settings", {})
    neutral_rating = float(rating_settings.get("rating_weight_neutral", 3.0))
    rating_weighting_enabled = bool(rating_settings.get("rating_weighting_enabled", False))
    hist_ratings = train_data.get("hist_rating")
    for history, ratings, target_id, target_rating in zip(train_data["hist_movie_id"], hist_ratings, train_data["movie_id"], train_data["rating"], strict=False):
        valid_history = [int(x) for x in np.asarray(history).reshape(-1) if int(x) > 0]
        valid_ratings = [float(x) for x in np.asarray(ratings).reshape(-1) if float(x) > 0]
        target_id = int(target_id)
        if target_id <= 0:
            continue
        valid_history.append(target_id)
        valid_ratings.append(float(target_rating))
        if len(valid_history) < 2:
            continue
        if len(valid_ratings) < len(valid_history):
            valid_ratings = [neutral_rating] * (len(valid_history) - len(valid_ratings)) + valid_ratings
        window_size = 5
        for target_pos in range(1, len(valid_history)):
            target_id = int(valid_history[target_pos])
            target_weight = rating_weight(float(valid_ratings[target_pos]), rating_settings)
            start_pos = max(0, target_pos - window_size)
            for source_pos in range(start_pos, target_pos):
                source_id = int(valid_history[source_pos])
                distance = target_pos - source_pos
                source_weight = rating_weight(float(valid_ratings[source_pos]), rating_settings)
                transitions[source_id][target_id] += float(source_weight * target_weight / max(distance, 1))
    total_edges = sum(len(counter) for counter in transitions.values())
    logger.info("ItemCF build done | anchor_items=%s | transition_edges=%s | rating_weighting=%s", len(transitions), total_edges, rating_weighting_enabled)
    save_pickle(transitions, ITEM_CF_MODEL_PATH)
    return {"model_path": str(ITEM_CF_MODEL_PATH), "anchor_items": len(transitions)}



def train_genre(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
    genre_scores = defaultdict(Counter)
    item_features = context["item_feature_arrays"]
    genre_matrix = np.asarray(item_features["genres"], dtype=np.int32)
    rating_settings = context["resolved_config"].get("training_settings", {})
    positive_rating_min = float(context["resolved_config"].get("rating_semantics", {}).get("positive_rating_min", 4.0))
    for history, ratings, target_id, target_rating in zip(context["train_data"]["hist_movie_id"], context["train_data"]["hist_rating"], context["train_data"]["movie_id"], context["train_data"]["rating"], strict=False):
        history_ids = np.concatenate([np.asarray(history).reshape(-1), np.asarray([target_id])])
        history_ratings = np.concatenate([np.asarray(ratings).reshape(-1), np.asarray([target_rating])])
        for movie_id, rating in zip(history_ids, history_ratings, strict=False):
            movie_id = int(movie_id)
            rating = float(rating)
            if movie_id <= 0 or movie_id >= genre_matrix.shape[0] or rating < positive_rating_min:
                continue
            item_weight = rating_weight(rating, rating_settings)
            for genre_idx in np.flatnonzero(genre_matrix[movie_id]).tolist():
                genre_scores[int(genre_idx) + 1][movie_id] += float(item_weight)
    genre_to_items = {
        genre_id: [int(item_id) for item_id, _ in counter.most_common()]
        for genre_id, counter in genre_scores.items()
    }
    save_pickle(genre_to_items, GENRE_MODEL_PATH)
    return {"model_path": str(GENRE_MODEL_PATH), "genre_count": len(genre_to_items)}



def train_popular(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
    popularity = Counter()
    rating_settings = context["resolved_config"].get("training_settings", {})
    positive_rating_min = float(context["resolved_config"].get("rating_semantics", {}).get("positive_rating_min", 4.0))
    for history, ratings, target_id, target_rating in zip(context["train_data"]["hist_movie_id"], context["train_data"]["hist_rating"], context["train_data"]["movie_id"], context["train_data"]["rating"], strict=False):
        history_ids = np.concatenate([np.asarray(history).reshape(-1), np.asarray([target_id])])
        history_ratings = np.concatenate([np.asarray(ratings).reshape(-1), np.asarray([target_rating])])
        for movie_id, rating in zip(history_ids, history_ratings, strict=False):
            movie_id = int(movie_id)
            rating = float(rating)
            if movie_id <= 0 or rating < positive_rating_min:
                continue
            popularity[movie_id] += rating_weight(rating, rating_settings)
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
    temperature: float,
    sampled_negative_ids: torch.Tensor | None = None,
    sampled_negative_repr: torch.Tensor | None = None,
    low_rating_negative_ids: torch.Tensor | None = None,
    low_rating_negative_repr: torch.Tensor | None = None,
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

    if low_rating_negative_ids is not None and low_rating_negative_repr is not None and low_rating_negative_ids.numel() > 0:
        low_rating_scores = torch.einsum("bd,bnd->bn", user_repr, low_rating_negative_repr) / temperature
        low_rating_invalid_mask = low_rating_negative_ids.le(0)
        low_rating_invalid_mask |= low_rating_negative_ids.eq(positive_item_ids.unsqueeze(1))
        low_rating_invalid_mask |= hist_movie_ids.unsqueeze(2).eq(low_rating_negative_ids.unsqueeze(1)).any(dim=1)
        low_rating_scores = low_rating_scores.masked_fill(low_rating_invalid_mask, torch.finfo(low_rating_scores.dtype).min)
        extra_logits.append(low_rating_scores)

    if extra_logits:
        logits = torch.cat([logits, *extra_logits], dim=1)
    return logits, labels


def _valid_candidate_mask(candidates: torch.Tensor, blocked_ids: torch.Tensor, blocked_mask: torch.Tensor) -> torch.Tensor:
    return candidates.gt(0) & ~((candidates.unsqueeze(2) == blocked_ids.unsqueeze(1)) & blocked_mask.unsqueeze(1)).any(dim=2)



def _first_occurrence_mask(candidates: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(candidates.size(1), device=candidates.device).view(1, -1, 1)
    previous_positions = torch.arange(candidates.size(1), device=candidates.device).view(1, 1, -1)
    duplicate_previous = candidates.unsqueeze(2).eq(candidates.unsqueeze(1)) & previous_positions.lt(positions)
    return ~duplicate_previous.any(dim=2)



def _pack_candidate_ids(candidates: torch.Tensor, selection: torch.Tensor, width: int) -> torch.Tensor:
    batch_size = candidates.size(0)
    packed = torch.zeros((batch_size, width), dtype=candidates.dtype, device=candidates.device)
    if width <= 0 or candidates.numel() == 0:
        return packed
    rank = selection.long().cumsum(dim=1) - 1
    valid = selection & rank.lt(width)
    if valid.any():
        row_indices = torch.arange(batch_size, device=candidates.device).unsqueeze(1).expand_as(candidates)[valid]
        col_indices = rank[valid]
        packed[row_indices, col_indices] = candidates[valid]
    return packed



def _sample_user_negative_ids(
    user_negative_ids: torch.Tensor,
    positive_ids: torch.Tensor,
    context_ids: torch.Tensor,
    num_negatives: int,
) -> torch.Tensor:
    batch_size = positive_ids.size(0)
    if batch_size == 0 or num_negatives <= 0:
        return torch.zeros((batch_size, max(num_negatives, 0)), dtype=torch.long, device=positive_ids.device)
    candidates = user_negative_ids.long()
    valid = candidates.gt(0)
    valid &= candidates.ne(positive_ids.unsqueeze(1))
    valid &= ~context_ids.unsqueeze(2).eq(candidates.unsqueeze(1)).any(dim=1)
    valid &= _first_occurrence_mask(candidates)
    selection = valid & (valid.cumsum(dim=1) <= num_negatives)
    return _pack_candidate_ids(candidates, selection, num_negatives)


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
        hist_rating = batch_dict.get("hist_rating")
        user_negative_ids = batch_dict["user_negative_movie_id"][valid_mask]
        if hist_recency_bucket is not None:
            hist_recency_bucket = hist_recency_bucket[valid_mask]
        if hist_rating is not None:
            hist_rating = hist_rating[valid_mask]
        if hist_feedback is not None:
            hist_feedback = hist_feedback[valid_mask]
    else:
        hist_movie_ids = batch_dict["hist_movie_id"]
        context_movie_ids = hist_movie_ids
        hist_recency_bucket = batch_dict.get("hist_recency_bucket")
        hist_rating = batch_dict.get("hist_rating")
        hist_feedback = batch_dict.get("hist_feedback")
        user_negative_ids = batch_dict["user_negative_movie_id"]

    hist_movie_ids, hist_recency_bucket, hist_rating, hist_feedback = filter_sequence_history(
        hist_movie_ids,
        hist_recency_bucket,
        hist_rating,
        hist_feedback,
        history_feedback,
    )
    hist_item_features = build_item_batch(hist_movie_ids, item_feature_tensors, hist_movie_ids.device)
    hidden_states = sequence_model.encode_user(
        hist_movie_ids,
        hist_recency_bucket,
        hist_rating,
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
    eval_batches = batch_indices(eval_indices, batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, indices in enumerate(eval_batches, start=1):
            batch_dict = slice_train_batch(train_data, indices, device)
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
                    two_tower_temperature,
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
                hist_movie_ids, hist_recency_bucket, hist_rating, hist_feedback = filter_sequence_history(
                    batch["hist_movie_id"],
                    batch["hist_recency_bucket"],
                    batch["hist_rating"],
                    batch["hist_feedback"],
                    sequence_history_feedback,
                )
                hist_item_features = build_item_batch(hist_movie_ids, item_feature_tensors, device)
                user_embedding = model.encode_user(
                    hist_movie_ids,
                    hist_recency_bucket,
                    hist_rating,
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
