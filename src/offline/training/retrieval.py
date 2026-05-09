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
from offline.features.item_batch import ITEM_FEATURE_FIELDS
from offline.models.sequence_retrieval import SequenceRetrievalModel
from offline.models.two_tower import TwoTowerRetrievalModel
from offline.ranking.protocol import get_item_feature_arrays
from offline.training.retrieval_common import batch_indices, gradient_norm, load_retrieval_context, slice_train_batch
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
    validation_data = context["validation_data"]
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
    two_tower_num_negatives = int(training_settings.get("two_tower_num_negatives", 256))
    hard_negative_ratio = float(training_settings.get("hard_negative_ratio", 0.1))
    negative_popularity_alpha = float(training_settings.get("negative_popularity_alpha", 0.75))
    export_batch_size = int(training_settings.get("export_batch_size", 4096))
    two_tower_temperature = float(training_settings.get("two_tower_temperature", 0.05))

    device = context["device"]
    all_item_id_tensor = torch.as_tensor(all_item_ids, dtype=torch.long, device=device)
    item_id_positions = _build_item_position_lookup(all_item_id_tensor)
    popularity_weights = _build_retrieval_popularity_weights(
        all_item_id_tensor,
        item_feature_arrays,
        device,
        negative_popularity_alpha,
    )
    multimodal_matrix = torch.nn.functional.normalize(
        torch.as_tensor(item_feature_arrays["multimodal_embedding"], dtype=torch.float32, device=device),
        dim=1,
    )
    two_tower_model = TwoTowerRetrievalModel(
        feature_dict,
        emb_dim,
        user_hidden_dims=two_tower_settings.get("user_hidden_dims"),
        item_hidden_dims=two_tower_settings.get("item_hidden_dims"),
        dropout=float(two_tower_settings.get("dropout", 0.1)),
        multimodal_table=item_feature_arrays["multimodal_embedding"],
        item_feature_table=item_feature_arrays,
        recent_history_length=int(two_tower_settings.get("recent_history_length", 20)),
    ).to(device)

    if warm_start and RETRIEVAL_MODEL_PATH.exists():
        retrieval_checkpoint = torch.load(RETRIEVAL_MODEL_PATH, map_location=device)
        two_tower_model.load_state_dict(retrieval_checkpoint["state_dict"])

    two_tower_optimizer = torch.optim.Adam(two_tower_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_size = int(len(train_data["user_id"]))
    train_indices = np.arange(train_size, dtype=np.int64)

    use_explicit_negatives = two_tower_num_negatives > 0

    logger.info(
        "TwoTower training start | device=%s | train_samples=%s | validation_users=%s | test_users=%s | batch_size=%s | epochs=%s | negatives=%s | hard_negative_ratio=%s | negative_popularity_alpha=%s | export_batch_size=%s | temperature=%s | warm_start=%s",
        device,
        train_size,
        len(validation_data["user_id"]),
        len(test_data["user_id"]),
        batch_size,
        epochs,
        two_tower_num_negatives if use_explicit_negatives else 0,
        hard_negative_ratio if use_explicit_negatives else 0.0,
        negative_popularity_alpha if use_explicit_negatives else 0.0,
        export_batch_size,
        two_tower_temperature,
        warm_start,
    )

    best_state = None
    best_val_metric = float("-inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    total_batches = max(int(np.ceil(train_size / max(batch_size, 1))), 1)
    encoded_movie_ids = np.asarray(all_item_ids, dtype=np.int64)
    evaluation_topk = int(resolved_config["evaluation_settings"].get("topk", 200))

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

            two_tower_optimizer.zero_grad()
            user_repr_tt = two_tower_model.encode_user(batch_dict)
            positive_ids = batch_dict["movie_id"]
            positive_ratings = batch_dict["rating"]
            item_repr_tt = two_tower_model.encode_item(positive_ids)
            explicit_negative_ids = None
            explicit_negative_repr = None
            explicit_negative_logq = None
            if use_explicit_negatives:
                explicit_negative_ids, explicit_negative_logq = _sample_retrieval_negative_ids(
                    positive_ids=positive_ids,
                    context_ids=batch_dict["hist_movie_id"],
                    all_item_ids=all_item_id_tensor,
                    item_id_positions=item_id_positions,
                    popularity_weights=popularity_weights,
                    multimodal_matrix=multimodal_matrix,
                    num_negatives=two_tower_num_negatives,
                    hard_negative_ratio=hard_negative_ratio,
                )
                explicit_negative_repr = two_tower_model.encode_item(explicit_negative_ids.reshape(-1)).reshape(
                    explicit_negative_ids.size(0),
                    explicit_negative_ids.size(1),
                    -1,
                )
            logits_tt, labels_tt = _build_training_logits(
                user_repr_tt,
                item_repr_tt,
                positive_ids,
                batch_dict["hist_movie_id"],
                two_tower_temperature,
                explicit_negative_ids=explicit_negative_ids,
                explicit_negative_repr=explicit_negative_repr,
                explicit_negative_logq=explicit_negative_logq,
            )
            row_loss = torch.nn.functional.cross_entropy(logits_tt, labels_tt, reduction="none")
            loss_tt = _weighted_mean(row_loss, positive_ratings)
            loss_tt.backward()
            should_log = batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
            grad_norm_tt = gradient_norm(two_tower_model.parameters()) if should_log else 0.0
            two_tower_optimizer.step()

            batch_size_now = positive_ids.size(0)
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
        two_tower_model.eval()
        validation_item_embeddings, _ = _export_item_embeddings(
            two_tower_model,
            None,
            encoded_movie_ids,
            item_feature_arrays,
            device,
            export_batch_size,
            export_two_tower=True,
            export_sequence=False,
        )
        val_metrics = _evaluate_retrieval(
            two_tower_model,
            validation_data,
            encoded_movie_ids,
            validation_item_embeddings,
            device,
            evaluation_topk,
            route="two_tower",
            item_feature_arrays=item_feature_arrays,
        )
        val_metric = float(val_metrics.get("ndcg@200", val_metrics.get("ndcg@100", 0.0)))
        logger.info(
            "TwoTower epoch %s/%s done | train_loss=%.4f | val_ndcg@200=%.6f | val_hr@200=%.6f | avg_grad_norm=%.6f | epoch_time=%s | elapsed=%s",
            epoch + 1,
            epochs,
            train_tt_loss,
            val_metric,
            float(val_metrics.get("hr@200", 0.0)),
            total_grad_norm / max(grad_norm_steps, 1),
            format_eta(time.perf_counter() - epoch_start),
            format_eta(time.perf_counter() - training_start),
        )

        if val_metric > best_val_metric + min_delta:
            best_val_metric = val_metric
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in two_tower_model.state_dict().items()}
            logger.info("TwoTower validation improved | best_val_ndcg@200=%.6f", best_val_metric)
        else:
            stale_epochs += 1
            logger.info("TwoTower validation stale | stale_epochs=%s/%s", stale_epochs, early_stopping_patience)
            if stale_epochs >= early_stopping_patience:
                logger.info("TwoTower early stopping triggered | best_val_ndcg@200=%.6f", best_val_metric)
                break

    if best_state is not None:
        two_tower_model.load_state_dict(best_state)

    two_tower_model.eval()
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
    validation_data = context["validation_data"]
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
    sequence_num_negatives = int(training_settings.get("sequence_num_negatives", 128))
    sequence_loss = str(training_settings.get("sequence_loss", "bce")).strip().lower()
    hard_negative_ratio = float(training_settings.get("hard_negative_ratio", 0.1))
    negative_popularity_alpha = float(training_settings.get("negative_popularity_alpha", 0.75))

    device = context["device"]
    all_item_id_tensor = torch.as_tensor(all_item_ids, dtype=torch.long, device=device)
    item_id_positions = _build_item_position_lookup(all_item_id_tensor)
    popularity_weights = _build_retrieval_popularity_weights(
        all_item_id_tensor,
        item_feature_arrays,
        device,
        negative_popularity_alpha,
    )
    multimodal_matrix = torch.nn.functional.normalize(
        torch.as_tensor(item_feature_arrays["multimodal_embedding"], dtype=torch.float32, device=device),
        dim=1,
    )
    sequence_model = SequenceRetrievalModel(
        feature_dict,
        emb_dim,
        hidden_dim=int(sequence_settings.get("hidden_dim", emb_dim)),
        num_layers=int(sequence_settings.get("num_layers", 1)),
        max_len=int(sequence_settings.get("max_len", 10)),
        dropout=float(sequence_settings.get("dropout", 0.1)),
        multimodal_table=item_feature_arrays["multimodal_embedding"],
        item_feature_table=item_feature_arrays,
    ).to(device)

    if warm_start and SEQUENCE_MODEL_PATH.exists():
        sequence_checkpoint = torch.load(SEQUENCE_MODEL_PATH, map_location=device)
        sequence_model.load_state_dict(sequence_checkpoint["state_dict"])

    sequence_optimizer = torch.optim.Adam(sequence_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_samples = int(len(train_data["user_id"]))
    train_indices = np.arange(total_samples, dtype=np.int64)

    logger.info(
        "Sequence training start | device=%s | train_samples=%s | validation_users=%s | test_users=%s | batch_size=%s | epochs=%s | export_batch_size=%s | sequence_num_negatives=%s | sequence_loss=%s | hard_negative_ratio=%s | negative_popularity_alpha=%s | warm_start=%s",
        device,
        total_samples,
        len(validation_data["user_id"]),
        len(test_data["user_id"]),
        batch_size,
        epochs,
        export_batch_size,
        sequence_num_negatives,
        sequence_loss,
        hard_negative_ratio,
        negative_popularity_alpha,
        warm_start,
    )

    best_state = None
    best_val_metric = float("-inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    total_batches = max(int(np.ceil(total_samples / max(batch_size, 1))), 1) if total_samples > 0 else 1
    encoded_movie_ids = np.asarray(all_item_ids, dtype=np.int64)
    evaluation_topk = int(resolved_config["evaluation_settings"].get("topk", 200))

    for epoch in range(epochs):
        sequence_model.train()
        total_seq_loss = 0.0
        seen_positions = 0
        epoch_start = time.perf_counter()
        train_batches = batch_indices(train_indices, batch_size, shuffle=True) if total_samples > 0 else []

        for batch_idx, indices in enumerate(train_batches, start=1):
            batch_dict = slice_train_batch(train_data, indices, device)
            sequence_optimizer.zero_grad()
            loss_seq, supervised_positions = _compute_sequence_loss(
                sequence_model,
                batch_dict,
                all_item_ids=all_item_id_tensor,
                item_id_positions=item_id_positions,
                popularity_weights=popularity_weights,
                multimodal_matrix=multimodal_matrix,
                num_negatives=sequence_num_negatives,
                hard_negative_ratio=hard_negative_ratio,
                loss_type=sequence_loss,
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
        sequence_model.eval()
        _, validation_item_embeddings = _export_item_embeddings(
            None,
            sequence_model,
            encoded_movie_ids,
            item_feature_arrays,
            device,
            export_batch_size,
            export_two_tower=False,
            export_sequence=True,
        )
        val_metrics = _evaluate_retrieval(
            sequence_model,
            validation_data,
            encoded_movie_ids,
            validation_item_embeddings,
            device,
            evaluation_topk,
            route="sequence",
            item_feature_arrays=item_feature_arrays,
        )
        val_metric = float(val_metrics.get("ndcg@200", val_metrics.get("ndcg@100", 0.0)))
        logger.info(
            "Sequence epoch %s/%s done | train_loss=%.4f | val_ndcg@200=%.6f | val_hr@200=%.6f | epoch_time=%s | elapsed=%s",
            epoch + 1,
            epochs,
            train_seq_loss,
            val_metric,
            float(val_metrics.get("hr@200", 0.0)),
            format_eta(time.perf_counter() - epoch_start),
            format_eta(time.perf_counter() - training_start),
        )

        if val_metric > best_val_metric + min_delta:
            best_val_metric = val_metric
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in sequence_model.state_dict().items()}
            logger.info("Sequence validation improved | best_val_ndcg@200=%.6f", best_val_metric)
        else:
            stale_epochs += 1
            logger.info("Sequence validation stale | stale_epochs=%s/%s", stale_epochs, early_stopping_patience)
            if stale_epochs >= early_stopping_patience:
                logger.info("Sequence early stopping triggered | best_val_ndcg@200=%.6f", best_val_metric)
                break

    if best_state is not None:
        sequence_model.load_state_dict(best_state)

    sequence_model.eval()
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
    neutral_rating = float(context["resolved_config"].get("training_settings", {}).get("neutral_rating", 3.0))
    hist_ratings = train_data.get("hist_rating")
    for history, ratings in zip(train_data["hist_movie_id"], hist_ratings, strict=False):
        valid_history = [int(x) for x in np.asarray(history).reshape(-1) if int(x) > 0]
        valid_ratings = [float(x) for x in np.asarray(ratings).reshape(-1) if float(x) > 0]
        if len(valid_history) < 2:
            continue
        if len(valid_ratings) < len(valid_history):
            valid_ratings = [neutral_rating] * (len(valid_history) - len(valid_ratings)) + valid_ratings
        window_size = 10
        for target_pos in range(1, len(valid_history)):
            target_id = int(valid_history[target_pos])
            target_weight = float(valid_ratings[target_pos])
            start_pos = max(0, target_pos - window_size)
            for source_pos in range(start_pos, target_pos):
                source_id = int(valid_history[source_pos])
                distance = target_pos - source_pos
                source_weight = float(valid_ratings[source_pos])
                transitions[source_id][target_id] += float(source_weight * target_weight / max(distance, 1))
    total_edges = sum(len(counter) for counter in transitions.values())
    logger.info("ItemCF build done | anchor_items=%s | transition_edges=%s", len(transitions), total_edges)
    save_pickle(transitions, ITEM_CF_MODEL_PATH)
    return {"model_path": str(ITEM_CF_MODEL_PATH), "anchor_items": len(transitions)}



def train_genre(context=None):
    if context is None:
        settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
        context = load_retrieval_context(settings, resolve_retrieval_config(settings))
    genre_scores = defaultdict(Counter)
    item_features = context["item_feature_arrays"]
    genre_matrix = np.asarray(item_features["genres"], dtype=np.int32)
    positive_rating_min = float(context["resolved_config"].get("rating_semantics", {}).get("positive_rating_min", 3.0))
    for history, ratings in zip(context["train_data"]["hist_movie_id"], context["train_data"]["hist_rating"], strict=False):
        history_ids = np.asarray(history).reshape(-1)
        history_ratings = np.asarray(ratings).reshape(-1)
        for movie_id, rating in zip(history_ids, history_ratings, strict=False):
            movie_id = int(movie_id)
            rating = float(rating)
            if movie_id <= 0 or movie_id >= genre_matrix.shape[0] or rating < positive_rating_min:
                continue
            item_weight = rating
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
    positive_rating_min = float(context["resolved_config"].get("rating_semantics", {}).get("positive_rating_min", 3.0))
    for history, ratings in zip(context["train_data"]["hist_movie_id"], context["train_data"]["hist_rating"], strict=False):
        history_ids = np.asarray(history).reshape(-1)
        history_ratings = np.asarray(ratings).reshape(-1)
        for movie_id, rating in zip(history_ids, history_ratings, strict=False):
            movie_id = int(movie_id)
            rating = float(rating)
            if movie_id <= 0 or rating < positive_rating_min:
                continue
            popularity[movie_id] += rating
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
            multimodal_table=item_feature_arrays["multimodal_embedding"],
            item_feature_table=item_feature_arrays,
            recent_history_length=int(model_settings.get("recent_history_length", 20)),
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
            item_feature_table=item_feature_arrays,
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    encoded_movie_ids = np.load(MOVIE_IDS_PATH, mmap_mode="r").astype(np.int64)
    item_embeddings = np.load(embedding_path, mmap_mode="r")
    metrics = _evaluate_retrieval(model, train_eval_samples["test"], encoded_movie_ids, item_embeddings, device, resolved_topk, route=normalized_route, item_feature_arrays=item_feature_arrays)
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
            if export_two_tower:
                two_tower_chunks.append(two_tower_model.encode_item(batch_ids).cpu().numpy())
            if export_sequence:
                sequence_chunks.append(sequence_model.encode_item(batch_ids).cpu().numpy())
    two_tower_embeddings = np.concatenate(two_tower_chunks, axis=0) if two_tower_chunks else None
    sequence_embeddings = np.concatenate(sequence_chunks, axis=0) if sequence_chunks else None
    return two_tower_embeddings, sequence_embeddings



def _build_training_logits(
    user_repr: torch.Tensor,
    positive_item_repr: torch.Tensor,
    positive_item_ids: torch.Tensor,
    hist_movie_ids: torch.Tensor,
    temperature: float,
    explicit_negative_ids: torch.Tensor | None = None,
    explicit_negative_repr: torch.Tensor | None = None,
    explicit_negative_logq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (user_repr @ positive_item_repr.T) / temperature
    duplicate_positive_mask = positive_item_ids.unsqueeze(0).eq(positive_item_ids.unsqueeze(1))
    in_batch_history_mask = hist_movie_ids.unsqueeze(2).eq(positive_item_ids.unsqueeze(0).unsqueeze(0)).any(dim=1)
    invalid_in_batch_mask = duplicate_positive_mask | in_batch_history_mask
    invalid_in_batch_mask.fill_diagonal_(False)
    logits = logits.masked_fill(invalid_in_batch_mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(logits.size(0), device=logits.device)
    extra_logits = []

    if explicit_negative_ids is not None and explicit_negative_repr is not None and explicit_negative_ids.numel() > 0:
        explicit_scores = torch.einsum("bd,bnd->bn", user_repr, explicit_negative_repr) / temperature
        if explicit_negative_logq is not None:
            explicit_scores = explicit_scores - explicit_negative_logq.to(device=explicit_scores.device, dtype=explicit_scores.dtype)
        explicit_invalid_mask = explicit_negative_ids.le(0)
        explicit_invalid_mask |= explicit_negative_ids.eq(positive_item_ids.unsqueeze(1))
        explicit_invalid_mask |= hist_movie_ids.unsqueeze(2).eq(explicit_negative_ids.unsqueeze(1)).any(dim=1)
        explicit_scores = explicit_scores.masked_fill(explicit_invalid_mask, torch.finfo(explicit_scores.dtype).min)
        extra_logits.append(explicit_scores)

    if extra_logits:
        logits = torch.cat([logits, *extra_logits], dim=1)
    return logits, labels


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    safe_weights = weights.float().clamp_min(1.0)
    return (loss * safe_weights).sum() / safe_weights.sum().clamp_min(1e-9)





def _valid_candidate_mask(candidates: torch.Tensor, blocked_ids: torch.Tensor, blocked_mask: torch.Tensor) -> torch.Tensor:
    valid = candidates.gt(0)
    for start in range(0, blocked_ids.size(1), 128):
        stop = min(start + 128, blocked_ids.size(1))
        blocked_chunk = blocked_ids[:, start:stop]
        mask_chunk = blocked_mask[:, start:stop]
        valid &= ~((candidates.unsqueeze(2) == blocked_chunk.unsqueeze(1)) & mask_chunk.unsqueeze(1)).any(dim=2)
    return valid



def _first_occurrence_mask(candidates: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(candidates.size(1), device=candidates.device).view(1, -1, 1)
    previous_positions = torch.arange(candidates.size(1), device=candidates.device).view(1, 1, -1)
    duplicate_previous = candidates.unsqueeze(2).eq(candidates.unsqueeze(1)) & previous_positions.lt(positions)
    return ~duplicate_previous.any(dim=2)


def _build_item_position_lookup(all_item_ids: torch.Tensor) -> torch.Tensor:
    if all_item_ids.numel() == 0:
        return torch.full((1,), -1, dtype=torch.long, device=all_item_ids.device)
    positions = torch.full(
        (int(all_item_ids.max().detach().cpu().item()) + 1,),
        -1,
        dtype=torch.long,
        device=all_item_ids.device,
    )
    positions[all_item_ids.long()] = torch.arange(all_item_ids.numel(), dtype=torch.long, device=all_item_ids.device)
    return positions


def _build_retrieval_popularity_weights(
    all_item_ids: torch.Tensor,
    item_feature_arrays: dict[str, np.ndarray],
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    if all_item_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    popularity = np.asarray(item_feature_arrays.get("popularity"), dtype=np.float32)
    safe_ids = all_item_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    values = np.ones(safe_ids.shape[0], dtype=np.float32)
    valid = (safe_ids >= 0) & (safe_ids < popularity.shape[0])
    values[valid] = np.maximum(popularity[safe_ids[valid]], 1.0)
    weights = torch.as_tensor(values, dtype=torch.float32, device=device).pow(float(alpha))
    if not torch.isfinite(weights).all() or float(weights.sum().detach().cpu().item()) <= 0.0:
        weights = torch.ones_like(weights)
    return weights


def _lookup_item_positions(item_id_positions: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
    safe_ids = item_ids.long().clamp(min=0, max=item_id_positions.numel() - 1)
    positions = item_id_positions[safe_ids]
    return torch.where((item_ids > 0) & (item_ids < item_id_positions.numel()), positions, torch.full_like(positions, -1))



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



def _sample_popularity_unseen_negative_ids(
    positive_ids: torch.Tensor,
    context_ids: torch.Tensor,
    all_item_ids: torch.Tensor,
    item_id_positions: torch.Tensor,
    popularity_weights: torch.Tensor,
    num_negatives: int,
    existing_negative_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = positive_ids.size(0)
    device = positive_ids.device
    if batch_size == 0 or num_negatives <= 0 or all_item_ids.numel() == 0:
        shape = (batch_size, max(num_negatives, 0))
        return torch.zeros(shape, dtype=torch.long, device=device), torch.zeros(shape, dtype=torch.float32, device=device)

    blocked_parts = [context_ids.long(), positive_ids.long().unsqueeze(1)]
    if existing_negative_ids is not None and existing_negative_ids.numel() > 0:
        blocked_parts.append(existing_negative_ids.long())
    blocked_ids = torch.cat(blocked_parts, dim=1)
    blocked_mask = blocked_ids.gt(0)
    sampled_ids = torch.zeros((batch_size, num_negatives), dtype=torch.long, device=device)
    sampled_logq = torch.zeros((batch_size, num_negatives), dtype=torch.float32, device=device)
    base_weights = popularity_weights.float().clamp_min(0.0)
    for row_idx in range(batch_size):
        weights = base_weights.clone()
        blocked_row = blocked_ids[row_idx][blocked_mask[row_idx]]
        if blocked_row.numel() > 0:
            blocked_positions = _lookup_item_positions(item_id_positions, blocked_row)
            blocked_positions = blocked_positions[blocked_positions >= 0]
            if blocked_positions.numel() > 0:
                weights[blocked_positions] = 0.0
        valid_positions = torch.nonzero(weights > 0.0, as_tuple=False).reshape(-1)
        take = min(num_negatives, int(valid_positions.numel()))
        if take <= 0:
            continue
        probabilities = weights[valid_positions]
        probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
        selected_local = torch.multinomial(probabilities, num_samples=take, replacement=False)
        selected_positions = valid_positions[selected_local]
        sampled_ids[row_idx, :take] = all_item_ids[selected_positions]
        expected_counts = (float(num_negatives) * probabilities[selected_local]).clamp(max=1.0)
        sampled_logq[row_idx, :take] = expected_counts.clamp_min(1e-12).log()
    return sampled_ids, sampled_logq


def _sample_retrieval_negative_ids(
    positive_ids: torch.Tensor,
    context_ids: torch.Tensor,
    all_item_ids: torch.Tensor,
    item_id_positions: torch.Tensor,
    popularity_weights: torch.Tensor,
    multimodal_matrix: torch.Tensor,
    num_negatives: int,
    hard_negative_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = positive_ids.size(0)
    device = positive_ids.device
    if batch_size == 0 or num_negatives <= 0:
        shape = (batch_size, max(num_negatives, 0))
        return torch.zeros(shape, dtype=torch.long, device=device), torch.zeros(shape, dtype=torch.float32, device=device)
    hard_count = min(num_negatives, max(0, int(round(num_negatives * float(hard_negative_ratio)))))
    simple_count = num_negatives - hard_count
    simple_ids, simple_logq = _sample_popularity_unseen_negative_ids(
        positive_ids,
        context_ids,
        all_item_ids,
        item_id_positions,
        popularity_weights,
        simple_count,
    )
    if hard_count <= 0 or multimodal_matrix.numel() == 0:
        return simple_ids, simple_logq

    positive_vectors = multimodal_matrix[positive_ids.clamp(min=0, max=multimodal_matrix.size(0) - 1)]
    candidate_vectors = multimodal_matrix[all_item_ids.clamp(min=0, max=multimodal_matrix.size(0) - 1)]
    scores = positive_vectors @ candidate_vectors.T
    blocked_ids = torch.cat([context_ids.long(), positive_ids.long().unsqueeze(1), simple_ids], dim=1)
    blocked_mask = blocked_ids.gt(0)
    blocked_candidate_mask = ((all_item_ids.view(1, -1, 1) == blocked_ids.unsqueeze(1)) & blocked_mask.unsqueeze(1)).any(dim=2)
    scores = scores.masked_fill(blocked_candidate_mask, torch.finfo(scores.dtype).min)
    top_count = min(max(hard_count * 8, hard_count), all_item_ids.numel())
    hard_candidates = all_item_ids[scores.topk(top_count, dim=1).indices]
    hard_valid = hard_candidates.gt(0)
    hard_valid &= _valid_candidate_mask(hard_candidates, blocked_ids, blocked_mask)
    hard_valid &= _first_occurrence_mask(hard_candidates)
    hard_selection = hard_valid & (hard_valid.cumsum(dim=1) <= hard_count)
    hard_ids = _pack_candidate_ids(hard_candidates, hard_selection, hard_count)
    hard_logq = torch.zeros((batch_size, hard_count), dtype=torch.float32, device=device)
    return torch.cat([simple_ids, hard_ids], dim=1), torch.cat([simple_logq, hard_logq], dim=1)




def _compute_sequence_loss(
    sequence_model: SequenceRetrievalModel,
    batch_dict: dict[str, torch.Tensor],
    all_item_ids: torch.Tensor,
    item_id_positions: torch.Tensor,
    popularity_weights: torch.Tensor,
    multimodal_matrix: torch.Tensor,
    num_negatives: int,
    hard_negative_ratio: float,
    loss_type: str,
) -> tuple[torch.Tensor, int]:
    positive_ids = batch_dict["movie_id"]
    valid_mask = positive_ids.gt(0)
    if not valid_mask.any():
        return torch.zeros((), device=batch_dict["hist_movie_id"].device), 0

    positive_ids = positive_ids[valid_mask]
    positive_ratings = batch_dict["rating"][valid_mask].float().clamp_min(0.0)
    hist_movie_ids = batch_dict["hist_movie_id"][valid_mask]
    hist_time_gap_bucket = batch_dict.get("hist_time_gap_bucket")
    hist_rating = batch_dict.get("hist_rating")
    if hist_time_gap_bucket is not None:
        hist_time_gap_bucket = hist_time_gap_bucket[valid_mask]
    if hist_rating is not None:
        hist_rating = hist_rating[valid_mask]

    hidden_states = sequence_model.encode_user(
        hist_movie_ids,
        hist_time_gap_bucket,
        hist_rating,
    )
    negative_ids, negative_logq = _sample_retrieval_negative_ids(
        positive_ids=positive_ids,
        context_ids=hist_movie_ids,
        all_item_ids=all_item_ids,
        item_id_positions=item_id_positions,
        popularity_weights=popularity_weights,
        multimodal_matrix=multimodal_matrix,
        num_negatives=num_negatives,
        hard_negative_ratio=hard_negative_ratio,
    )
    positive_emb = sequence_model.encode_item(positive_ids)
    negative_emb = sequence_model.encode_item(negative_ids.reshape(-1)).reshape(negative_ids.size(0), negative_ids.size(1), -1)
    positive_logits = (hidden_states * positive_emb).sum(dim=1)
    negative_logits = torch.einsum("bd,bnd->bn", hidden_states, negative_emb)
    corrected_negative_logits = negative_logits - negative_logq.to(device=negative_logits.device, dtype=negative_logits.dtype)
    negative_mask = negative_ids.gt(0)
    if loss_type == "softmax":
        logits = torch.cat([positive_logits.unsqueeze(1), corrected_negative_logits.masked_fill(~negative_mask, torch.finfo(negative_logits.dtype).min)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        row_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return _weighted_mean(row_loss, positive_ratings), int(logits.size(0))

    positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits), reduction="none")
    weighted_positive_loss = (positive_loss * positive_ratings.clamp_min(1.0)).sum()
    positive_weight = positive_ratings.clamp_min(1.0).sum()
    if negative_mask.any():
        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(corrected_negative_logits[negative_mask], torch.zeros_like(corrected_negative_logits[negative_mask]), reduction="sum")
        denominator = positive_weight + negative_mask.sum().float().clamp_min(1.0)
        loss = (weighted_positive_loss + negative_loss) / denominator
    else:
        loss = weighted_positive_loss / positive_weight.clamp_min(1e-9)
    return loss, int(positive_logits.numel())




def _evaluate_retrieval(model, test_data, encoded_movie_ids, item_embeddings, device: torch.device, topk: int, route: str = "two_tower", item_feature_arrays: dict[str, np.ndarray] | None = None) -> dict:
    labels, scores, user_ids = [], [], []
    item_embeddings = np.asarray(item_embeddings)
    with torch.no_grad():
        for idx, user_id in enumerate(test_data["user_id"]):
            batch = {
                "user_id": torch.tensor([int(user_id)], dtype=torch.long, device=device),
                "age": torch.tensor([int(test_data["age"][idx])], dtype=torch.long, device=device),
                "gender": torch.tensor([int(test_data["gender"][idx])], dtype=torch.long, device=device),
                "occupation": torch.tensor([int(test_data["occupation"][idx])], dtype=torch.long, device=device),
                "zip_code": torch.tensor([int(test_data["zip_code"][idx])], dtype=torch.long, device=device),
                "hist_movie_id": torch.tensor(np.asarray(test_data["hist_movie_id"][idx]).reshape(1, -1), dtype=torch.long, device=device),
                "hist_time_gap_bucket": torch.tensor(np.asarray(test_data["hist_time_gap_bucket"][idx]).reshape(1, -1), dtype=torch.long, device=device),
                "hist_rating": torch.tensor(np.asarray(test_data["hist_rating"][idx]).reshape(1, -1), dtype=torch.float32, device=device),
            }
            if route == "sequence":
                user_embedding = model.encode_user(
                    batch["hist_movie_id"],
                    batch["hist_time_gap_bucket"],
                    batch["hist_rating"],
                ).cpu().numpy()[0]
            else:
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
