from __future__ import annotations

import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from offline.evaluate.metrics import hit_rate_at_k, ndcg_at_k, ranking_metrics, save_metrics
from offline.evaluate.multi_recall import evaluate_final_candidates
from offline.models.deepfm import DeepFMModel
from offline.models.din import DINModel
from offline.ranking.dataset import (
    SequenceRankingDataset,
    SequenceRankingTrainCollator,
    batch_to_device,
    build_inference_batch,
)
from offline.ranking.protocol import (
    POINTWISE_ITEM_FIELDS,
    POINTWISE_SEQUENCE_FIELDS,
    STATIC_USER_FIELDS,
    extract_split_sample,
    get_all_item_ids,
    get_item_feature_arrays,
)
from offline.utils.config import resolve_ranking_config, resolve_ranking_model_names
from offline.utils.io import (
    CONFIG_DIR,
    ITEM_CATALOG_PATH,
    MULTI_RECALL_ARTIFACTS_PATH,
    RANKING_FEATURE_DICT_PATH,
    RANKING_SAMPLE_PATH,
    get_final_metrics_path,
    get_ranking_metrics_path,
    get_ranking_model_config_path,
    get_ranking_model_path,
    load_pickle,
    save_pickle,
)
from offline.utils.logging import format_eta, get_logger


logger = get_logger("offline.training.ranking")
POINTWISE_RANKING_FIELDS = STATIC_USER_FIELDS + POINTWISE_ITEM_FIELDS
_TORCH_RANKING_MODELS = {"deepfm", "din"}
_RANKING_INFERENCE_BATCH_SIZE = 64
_FULL_CATALOG_CANDIDATE_CHUNK_SIZE = 256
_DEFAULT_MAX_VALIDATION_USERS = 6040



def _resolve_torch_device(training_settings: dict | None = None) -> torch.device:
    device_setting = str((training_settings or {}).get("device", "auto")).strip().lower()
    if device_setting == "cpu":
        return torch.device("cpu")
    if device_setting.startswith("cuda"):
        return torch.device(device_setting if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def run_ranking_training(model_name: str | None = None, models=None, warm_start: bool = True, evaluate: bool = False, final_evaluate: bool = False):
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    selected_models = resolve_ranking_model_names(settings, models if models is not None else model_name)
    results = train_ranking_models(models=selected_models, warm_start=warm_start)

    if evaluate:
        for selected_model in selected_models:
            results.setdefault(selected_model, {})["ranking"] = evaluate_ranking_model(selected_model)

    if final_evaluate:
        for selected_model in selected_models:
            results.setdefault(selected_model, {})["final"] = evaluate_final_ranking(selected_model)
    return results



def train_ranking_models(models=None, warm_start: bool = True):
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    selected_models = resolve_ranking_model_names(settings, models)
    results = {}
    for model_name in selected_models:
        if model_name == "deepfm":
            results[model_name] = train_deepfm(warm_start=warm_start)
        elif model_name == "din":
            results[model_name] = train_din(warm_start=warm_start)
        else:
            raise ValueError(f"Unsupported ranking model: {model_name}")
    return results



def train_deepfm(warm_start: bool = True):
    return _train_torch_ranking_model("deepfm", warm_start=warm_start)



def train_din(warm_start: bool = True):
    return _train_torch_ranking_model("din", warm_start=warm_start)



def evaluate_ranking_model(model_name: str) -> dict:
    normalized_model_name = model_name.strip().lower()
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_ranking_config(settings, requested_model_name=normalized_model_name)
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)
    fused_candidates_by_user = _resolve_ranking_user_candidate_pool(ranking_samples)
    if not fused_candidates_by_user:
        raise FileNotFoundError("Missing fused multi-recall candidates for ranking evaluation")

    feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
    model_settings = resolved_config["model_settings"]
    emb_dim = int(model_settings.get("embedding_dim", 16))
    final_topk = int(resolved_config["evaluation_settings"].get("final_topk", 20))
    device = _resolve_torch_device(resolved_config.get("training_settings", {}))
    checkpoint = torch.load(get_ranking_model_path(normalized_model_name), map_location=device)
    model = _build_model(normalized_model_name, feature_dict, model_settings, emb_dim, item_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = _evaluate_fused_candidates_subset(
        model,
        ranking_samples["test"],
        list(range(int(len(ranking_samples["test"]["user_id"])))),
        item_features,
        all_item_ids,
        device,
        final_topk=final_topk,
        fused_candidates_by_user=fused_candidates_by_user,
    )
    save_metrics(get_ranking_metrics_path(normalized_model_name), metrics)
    logger.info("Ranking metrics saved | model=%s | path=%s | metrics=%s", normalized_model_name, get_ranking_metrics_path(normalized_model_name).name, metrics)
    return metrics



def evaluate_final_ranking(model_name: str) -> dict:
    normalized_model_name = model_name.strip().lower()
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_ranking_config(settings, requested_model_name=normalized_model_name)
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)
    final_topk = int(resolved_config["evaluation_settings"].get("final_topk", 20))
    final_metrics_path = get_final_metrics_path(normalized_model_name)

    feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
    model_settings = resolved_config["model_settings"]
    emb_dim = int(model_settings.get("embedding_dim", 16))
    device = _resolve_torch_device(resolved_config.get("training_settings", {}))
    checkpoint = torch.load(get_ranking_model_path(normalized_model_name), map_location=device)
    model = _build_model(normalized_model_name, feature_dict, model_settings, emb_dim, item_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    final_metrics = _evaluate_final_candidates(model, ranking_samples, item_features, all_item_ids, device, final_topk)

    save_metrics(final_metrics_path, final_metrics)
    logger.info("Final metrics saved | model=%s | path=%s | metrics=%s", normalized_model_name, final_metrics_path.name, final_metrics)
    return final_metrics



def _train_torch_ranking_model(model_name: str, warm_start: bool = True):
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_ranking_config(settings, requested_model_name=model_name)
    model_settings = resolved_config["model_settings"]
    training_settings = resolved_config["training_settings"]
    evaluation_settings = resolved_config["evaluation_settings"]
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)

    train_dataset = SequenceRankingDataset(ranking_samples["train"])
    validation_data = ranking_samples.get("validation")
    if validation_data is None:
        raise ValueError("Ranking samples must contain an explicit validation split")
    fused_candidates_by_user = _resolve_ranking_user_candidate_pool(ranking_samples)

    batch_size = int(training_settings.get("batch_size", 256))
    epochs = int(training_settings.get("epochs", 3))
    learning_rate = float(training_settings.get("learning_rate", 0.001))
    weight_decay = float(training_settings.get("weight_decay", 1e-5))
    log_every_n_batches = int(training_settings.get("log_every_n_batches", 100))
    early_stopping_patience = int(training_settings.get("early_stopping_patience", 2))
    min_delta = float(training_settings.get("min_delta", 1e-4))
    emb_dim = int(model_settings.get("embedding_dim", 16))
    train_negatives = int(training_settings.get("train_negatives", 7))

    device = _resolve_torch_device(resolved_config.get("training_settings", {}))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=SequenceRankingTrainCollator(
            item_features=item_features,
            all_item_ids=all_item_ids,
            num_negatives=train_negatives,
            low_rating_negatives=int(training_settings.get("low_rating_negatives", 0)),
            seed=42,
        ),
    )

    model = _build_model(model_name, feature_dict, model_settings, emb_dim, item_features).to(device)
    ranking_model_path = get_ranking_model_path(model_name)
    ranking_model_config_path = get_ranking_model_config_path(model_name)
    if warm_start and ranking_model_path.exists():
        checkpoint = torch.load(ranking_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info(
        "Ranking training start | model=%s | protocol=%s | device=%s | train_samples=%s | validation_samples=%s | validation_users=%s | batch_size=%s | epochs=%s | train_negatives=%s | low_rating_negatives=%s | validation_negatives=%s | warm_start=%s",
        model_name,
        ranking_samples.get("protocol", "unknown"),
        device,
        len(ranking_samples["train"]["user_id"]),
        len(validation_data["user_id"]),
        len(np.unique(np.asarray(validation_data["user_id"], dtype=np.int32))),
        batch_size,
        epochs,
        train_negatives,
        int(training_settings.get("low_rating_negatives", 0)),
        int(training_settings.get("validation_negatives", 1000)),
        warm_start,
    )

    best_state = None
    best_val_ndcg = float("-inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    total_batches = max(len(train_loader), 1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_groups = 0
        epoch_start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = _listwise_candidate_loss(logits, batch["candidate_relevance"], batch["candidate_mask"])
            loss.backward()
            optimizer.step()

            batch_loss = float(loss.item())
            batch_groups = int(batch["target_index"].numel())
            total_loss += batch_loss * batch_groups
            seen_groups += batch_groups

            should_log = batch_idx % log_every_n_batches == 0 or batch_idx == total_batches
            if should_log:
                elapsed_epoch = time.perf_counter() - epoch_start
                avg_batch_time = elapsed_epoch / batch_idx
                remaining_batches = total_batches - batch_idx
                remaining_epochs = epochs - epoch - 1
                eta_seconds = avg_batch_time * (remaining_batches + remaining_epochs * total_batches)
                logger.info(
                    "Ranking epoch %s/%s | model=%s | batch %s/%s | batch_loss=%.4f | avg_loss=%.4f | candidate_groups=%s | eta=%s",
                    epoch + 1,
                    epochs,
                    model_name,
                    batch_idx,
                    total_batches,
                    batch_loss,
                    total_loss / max(seen_groups, 1),
                    batch_groups,
                    format_eta(eta_seconds),
                )

        train_loss = total_loss / max(seen_groups, 1)
        val_metrics = _evaluate_hard_negative_validation(
            model,
            validation_data,
            item_features,
            all_item_ids,
            device,
            final_topk=int(evaluation_settings.get("final_topk", 20)),
            negative_count=int(training_settings.get("validation_negatives", 1000)),
            max_users=int(training_settings.get("validation_max_users", _DEFAULT_MAX_VALIDATION_USERS)),
            fused_candidates_by_user=fused_candidates_by_user,
        )
        val_ndcg20 = float(val_metrics.get("ndcg@20", float("-inf")))

        logger.info(
            "Ranking epoch %s/%s done | model=%s | train_loss=%.4f | val_ndcg@20=%s | epoch_time=%s | elapsed=%s",
            epoch + 1,
            epochs,
            model_name,
            train_loss,
            f"{val_ndcg20:.4f}" if np.isfinite(val_ndcg20) else "n/a",
            format_eta(time.perf_counter() - epoch_start),
            format_eta(time.perf_counter() - training_start),
        )

        if val_ndcg20 > best_val_ndcg + min_delta:
            best_val_ndcg = val_ndcg20
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            logger.info("Ranking validation improved | model=%s | best_ndcg@20=%.4f | metrics=%s", model_name, best_val_ndcg, val_metrics)
        else:
            stale_epochs += 1
            logger.info("Ranking validation stale | model=%s | ndcg@20=%.4f | stale_epochs=%s/%s", model_name, val_ndcg20, stale_epochs, early_stopping_patience)
            if stale_epochs >= early_stopping_patience:
                logger.info("Ranking early stopping triggered | model=%s | best_ndcg@20=%.4f", model_name, best_val_ndcg)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "config_schema": "model_registry_v2" if "models" in settings else "legacy_v1",
            "protocol": ranking_samples.get("protocol", "unknown"),
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "feature_dict": feature_dict,
            "pointwise_fields": POINTWISE_RANKING_FIELDS,
            "history_fields": POINTWISE_SEQUENCE_FIELDS,
            "emb_dim": emb_dim,
            "model_settings": model_settings,
            "training_settings": training_settings,
        },
        ranking_model_path,
    )
    save_pickle(
        {
            "config_schema": "model_registry_v2" if "models" in settings else "legacy_v1",
            "protocol": ranking_samples.get("protocol", "unknown"),
            "model_name": model_name,
            "feature_dict": feature_dict,
            "pointwise_fields": POINTWISE_RANKING_FIELDS,
            "history_fields": POINTWISE_SEQUENCE_FIELDS,
            "model_settings": model_settings,
            "training_settings": training_settings,
            "evaluation_settings": evaluation_settings,
        },
        ranking_model_config_path,
    )
    return {"model_name": model_name, "model_path": str(ranking_model_path), "config_path": str(ranking_model_config_path)}



def _build_fused_candidates_by_user() -> dict[int, np.ndarray] | None:
    if not MULTI_RECALL_ARTIFACTS_PATH.exists():
        return None
    artifacts = load_pickle(MULTI_RECALL_ARTIFACTS_PATH)
    user_ids = np.asarray(artifacts.get("user_ids", []), dtype=np.int32)
    fused_candidates = artifacts.get("fused_candidates", []) or []
    if user_ids.size == 0 or not fused_candidates:
        return None
    return {
        int(user_id): np.asarray(candidate_ids, dtype=np.int32)
        for user_id, candidate_ids in zip(user_ids.tolist(), fused_candidates, strict=False)
    }



def _resolve_ranking_user_candidate_pool(ranking_samples: dict) -> dict[int, np.ndarray] | None:
    del ranking_samples
    return _build_fused_candidates_by_user()




def _build_model(model_name: str, feature_dict: dict, model_settings: dict, emb_dim: int, item_features: dict[str, np.ndarray]):
    multimodal_table = item_features["multimodal_embedding"]
    if model_name == "deepfm":
        scorer = DeepFMModel(
            feature_dict,
            POINTWISE_RANKING_FIELDS,
            emb_dim,
            dnn_hidden_dims=model_settings.get("dnn_hidden_dims"),
            dropout=float(model_settings.get("dropout", 0.1)),
            history_fields=POINTWISE_SEQUENCE_FIELDS,
            multimodal_table=multimodal_table,
        )
        return scorer

    if model_name == "din":
        scorer = DINModel(
            feature_dict,
            POINTWISE_RANKING_FIELDS,
            emb_dim,
            dnn_hidden_dims=model_settings.get("dnn_hidden_dims"),
            attention_hidden_dims=model_settings.get("attention_hidden_dims"),
            dropout=float(model_settings.get("dropout", 0.1)),
            multimodal_table=multimodal_table,
        )
        return scorer

    raise ValueError(f"Unsupported ranking model: {model_name}")



def _listwise_candidate_loss(
    logits: torch.Tensor,
    relevance: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    mask = candidate_mask.bool()
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    gains = torch.clamp(relevance.float(), min=0.0) * mask.float()
    gain_sums = gains.sum(dim=1, keepdim=True)
    valid_rows = mask.any(dim=1) & gain_sums.squeeze(1).gt(0)
    if not valid_rows.any():
        return torch.zeros((), dtype=logits.dtype, device=logits.device)

    target_dist = gains[valid_rows] / gain_sums[valid_rows].clamp_min(1e-9)
    pred_log_probs = torch.log_softmax(masked_logits[valid_rows], dim=1)
    row_losses = -(target_dist * pred_log_probs).sum(dim=1)
    return row_losses.mean()



def _evaluate_hard_negative_validation(
    model,
    split_data: dict,
    item_features: dict[str, np.ndarray],
    all_item_ids: np.ndarray,
    device: torch.device,
    final_topk: int,
    negative_count: int,
    max_users: int,
    fused_candidates_by_user: dict[int, np.ndarray] | None = None,
) -> dict:
    sample_count = int(len(split_data["user_id"]))
    if max_users > 0:
        sample_count = min(sample_count, int(max_users))
    sample_indices = list(range(sample_count))
    rng = np.random.default_rng(20260429)
    candidate_user_ids: list[int] = []
    candidate_ids: list[list[int]] = []
    candidate_scores: list[list[float]] = []
    candidate_labels: list[list[float]] = []
    flat_user_ids: list[int] = []
    flat_scores: list[float] = []
    flat_labels: list[float] = []
    candidate_pool_size = max(int(negative_count), final_topk) + 1

    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_indices), _RANKING_INFERENCE_BATCH_SIZE):
            batch_indices = sample_indices[start : start + _RANKING_INFERENCE_BATCH_SIZE]
            samples = []
            candidate_pools = []
            for sample_idx in batch_indices:
                sample = extract_split_sample(split_data, sample_idx)
                target = int(sample["target_movie_id"])
                user_id = int(sample["user_id"])
                seen_ids = set(int(item_id) for item_id in np.asarray(sample.get("context_movie_id", []), dtype=np.int32).reshape(-1).tolist() if int(item_id) > 0)
                seen_ids.discard(target)
                selected: list[int] = [target] if target > 0 else []
                blocked = set(seen_ids)
                blocked.update(selected)

                hard_pool = None if fused_candidates_by_user is None else fused_candidates_by_user.get(user_id)
                if hard_pool is not None:
                    hard_candidates = [
                        int(item_id)
                        for item_id in np.asarray(hard_pool, dtype=np.int32).reshape(-1).tolist()
                        if int(item_id) > 0 and int(item_id) not in blocked
                    ]
                    if hard_candidates:
                        unique_hard = np.asarray(list(dict.fromkeys(hard_candidates)), dtype=np.int32)
                        take = min(int(unique_hard.size), max(candidate_pool_size - len(selected), 0))
                        selected.extend(unique_hard[:take].astype(int).tolist())
                        blocked.update(selected)

                remaining = candidate_pool_size - len(selected)
                if remaining > 0:
                    random_pool = all_item_ids[~np.isin(all_item_ids, np.asarray(list(blocked), dtype=np.int32))]
                    if random_pool.size > 0:
                        take = min(remaining, int(random_pool.size))
                        chosen = rng.choice(random_pool, size=take, replace=False) if take < random_pool.size else random_pool[:take]
                        selected.extend(chosen.astype(int).tolist())

                if len(selected) <= 1:
                    continue
                samples.append(sample)
                candidate_pools.append(np.asarray(selected, dtype=np.int32))

            if not samples:
                continue

            batch, valid_candidates_by_sample, valid_indices = build_inference_batch(samples, candidate_pools, item_features)
            if batch is None:
                continue

            batch = batch_to_device(batch, device)
            batch_scores = model(batch).cpu().numpy()
            for row_idx, valid_candidates in enumerate(valid_candidates_by_sample):
                sample = samples[valid_indices[row_idx]]
                scores = batch_scores[row_idx, : len(valid_candidates)]
                candidates_array = np.asarray(valid_candidates, dtype=np.int32)
                target = int(sample["target_movie_id"])
                labels = (candidates_array == target).astype(float)
                user_id = int(sample["user_id"])
                take = min(final_topk, int(scores.shape[0]))
                if take <= 0:
                    continue
                if take < scores.shape[0]:
                    top_positions = np.argpartition(scores, -take)[-take:]
                    top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
                else:
                    top_positions = np.argsort(scores)[::-1]
                flat_user_ids.extend([user_id] * int(scores.shape[0]))
                flat_scores.extend(scores.astype(float).tolist())
                flat_labels.extend(labels.tolist())
                candidate_user_ids.append(user_id)
                candidate_ids.append(candidates_array[top_positions].astype(int).tolist())
                candidate_scores.append(scores[top_positions].astype(float).tolist())
                candidate_labels.append(labels[top_positions].astype(float).tolist())

    metrics = evaluate_final_candidates(
        np.asarray(candidate_user_ids, dtype=np.int32),
        candidate_ids,
        candidate_scores,
        candidate_labels,
        all_item_ids=all_item_ids,
        output_path=None,
    )
    if flat_labels:
        labels = np.asarray(flat_labels, dtype=np.float32)
        scores = np.asarray(flat_scores, dtype=np.float32)
        user_ids = np.asarray(flat_user_ids, dtype=np.int32)
        metrics.update(ranking_metrics(labels, scores, user_ids))
        for k in (10, 20):
            metrics[f"hr@{k}"] = hit_rate_at_k(labels, scores, user_ids, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(labels, scores, user_ids, k)
    metrics["evaluated_users"] = int(len(candidate_user_ids))
    metrics["validation_max_users"] = int(max_users)
    metrics["validation_negatives"] = int(negative_count)
    metrics["candidate_source"] = "hard_negative_validation"
    return metrics

def _evaluate_fused_candidates_subset(
    model,
    split_data: dict,
    sample_indices: list[int],
    item_features: dict[str, np.ndarray],
    all_item_ids: np.ndarray,
    device: torch.device,
    final_topk: int,
    fused_candidates_by_user: dict[int, np.ndarray],
) -> dict:
    candidate_user_ids: list[int] = []
    candidate_ids: list[list[int]] = []
    candidate_scores: list[list[float]] = []
    candidate_labels: list[list[float]] = []
    flat_user_ids: list[int] = []
    flat_scores: list[float] = []
    flat_labels: list[float] = []
    skipped_missing_candidates = 0
    inference_batch_size = _RANKING_INFERENCE_BATCH_SIZE

    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_indices), inference_batch_size):
            batch_indices = sample_indices[start : start + inference_batch_size]
            samples = []
            candidate_pools = []
            for sample_idx in batch_indices:
                sample = extract_split_sample(split_data, sample_idx)
                user_id = int(sample["user_id"])
                candidates = fused_candidates_by_user.get(user_id)
                if candidates is None or int(np.asarray(candidates).size) == 0:
                    skipped_missing_candidates += 1
                    continue
                samples.append(sample)
                candidate_pools.append(np.asarray(candidates, dtype=np.int32))

            if not samples:
                continue

            batch, valid_candidates_by_sample, valid_indices = build_inference_batch(samples, candidate_pools, item_features)
            if batch is None:
                continue

            batch = batch_to_device(batch, device)
            batch_scores = model(batch).cpu().numpy()
            for row_idx, valid_candidates in enumerate(valid_candidates_by_sample):
                sample = samples[valid_indices[row_idx]]
                scores = batch_scores[row_idx, : len(valid_candidates)]
                candidates_array = np.asarray(valid_candidates, dtype=np.int32)
                take = min(final_topk, int(scores.shape[0]))
                if take <= 0:
                    continue
                if take < scores.shape[0]:
                    top_positions = np.argpartition(scores, -take)[-take:]
                    top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
                else:
                    top_positions = np.argsort(scores)[::-1]
                ranked_candidates_array = candidates_array[top_positions]
                ranked_scores_array = scores[top_positions]
                target = int(sample["target_movie_id"])
                labels = (candidates_array == target).astype(float)
                user_id = int(sample["user_id"])
                flat_user_ids.extend([user_id] * int(scores.shape[0]))
                flat_scores.extend(scores.astype(float).tolist())
                flat_labels.extend(labels.tolist())

                candidate_user_ids.append(user_id)
                candidate_ids.append(ranked_candidates_array.astype(int).tolist())
                candidate_scores.append(ranked_scores_array.astype(float).tolist())
                candidate_labels.append(labels[top_positions].astype(float).tolist())

    metrics = evaluate_final_candidates(
        np.asarray(candidate_user_ids, dtype=np.int32),
        candidate_ids,
        candidate_scores,
        candidate_labels,
        all_item_ids=all_item_ids,
        output_path=None,
    )
    if flat_labels:
        metrics.update(
            ranking_metrics(
                np.asarray(flat_labels, dtype=np.float32),
                np.asarray(flat_scores, dtype=np.float32),
                np.asarray(flat_user_ids, dtype=np.int32),
            )
        )
    metrics["evaluated_users"] = int(len(candidate_user_ids))
    metrics["skipped_missing_candidates"] = int(skipped_missing_candidates)
    return metrics



def _evaluate_final_candidates(
    model,
    ranking_samples: dict,
    item_features: dict[str, np.ndarray],
    all_item_ids: np.ndarray,
    device: torch.device,
    final_topk: int,
) -> dict:
    fused_candidates_by_user = _build_fused_candidates_by_user()
    if not fused_candidates_by_user:
        raise FileNotFoundError("Missing fused multi-recall candidates for final evaluation")
    ranking_test = ranking_samples["test"]
    sample_indices = list(range(int(len(ranking_test["user_id"]))))
    return _evaluate_fused_candidates_subset(
        model,
        ranking_test,
        sample_indices,
        item_features,
        all_item_ids,
        device,
        final_topk,
        fused_candidates_by_user,
    )
