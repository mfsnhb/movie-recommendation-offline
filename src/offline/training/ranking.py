from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, random_split

from offline.evaluate.metrics import ranking_metrics, save_metrics
from offline.evaluate.multi_recall import evaluate_final_candidates
from offline.models.deepfm import DeepFMModel
from offline.models.din import DINModel
from offline.models.xgboost import (
    evaluate_final_candidates_xgboost,
    evaluate_xgboost_scores,
    predict_xgboost_scores,
    train_xgboost_model,
)
from offline.ranking.dataset import (
    SequenceRankingDataset,
    SequenceRankingEvalCollator,
    SequenceRankingTrainCollator,
    SequenceRankingValidationCollator,
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
from offline.ranking.wrapper import PointwiseCandidateRanker
from offline.ranking.xgboost_features import build_xgboost_training_frame
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
_XGBOOST_RANKING_MODELS = {"xgboost", "xgboost_ranker"}



def _resolve_rating_weighting_settings(training_settings: dict | None) -> dict[str, float | bool]:
    source = dict(training_settings or {})
    return {
        "enabled": bool(source.get("rating_weighting_enabled", False)),
        "neutral": float(source.get("rating_weight_neutral", 3.0)),
        "scale": float(source.get("rating_weight_scale", 0.25)),
        "min": float(source.get("rating_weight_min", 0.5)),
        "max": float(source.get("rating_weight_max", 1.5)),
    }



def _target_rating_weights(target_ratings: torch.Tensor, training_settings: dict | None) -> torch.Tensor:
    config = _resolve_rating_weighting_settings(training_settings)
    if not bool(config["enabled"]):
        return torch.ones_like(target_ratings, dtype=torch.float32)
    weights = 1.0 + float(config["scale"]) * (target_ratings.float() - float(config["neutral"]))
    return weights.clamp(min=float(config["min"]), max=float(config["max"])).to(dtype=torch.float32)



def run_ranking_training(model_name: str | None = None, models=None, warm_start: bool = True, evaluate: bool = False, final_evaluate: bool = False):
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    selected_models = resolve_ranking_model_names(settings, models if models is not None else model_name)
    results = train_ranking_models(models=selected_models, warm_start=warm_start)

    if evaluate:
        for selected_model in selected_models:
            if selected_model in _TORCH_RANKING_MODELS:
                results.setdefault(selected_model, {})["ranking"] = evaluate_ranking_model(selected_model)
            else:
                results.setdefault(selected_model, {})["ranking"] = load_pickle(get_ranking_model_config_path(selected_model)).get("latest_ranking_metrics")

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
        elif model_name == "xgboost":
            results[model_name] = train_xgboost(warm_start=warm_start)
        elif model_name == "xgboost_ranker":
            results[model_name] = train_xgboost_ranker(warm_start=warm_start)
        else:
            raise ValueError(f"Unsupported ranking model: {model_name}")
    return results



def train_deepfm(warm_start: bool = True):
    return _train_torch_ranking_model("deepfm", warm_start=warm_start)



def train_din(warm_start: bool = True):
    return _train_torch_ranking_model("din", warm_start=warm_start)



def train_xgboost(warm_start: bool = True):
    return _run_xgboost_training("xgboost", warm_start=warm_start)



def train_xgboost_ranker(warm_start: bool = True):
    return _run_xgboost_training("xgboost_ranker", warm_start=warm_start)



def evaluate_ranking_model(model_name: str) -> dict:
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name in _XGBOOST_RANKING_MODELS:
        ranking_metrics_path = get_ranking_metrics_path(normalized_model_name)
        if not ranking_metrics_path.exists():
            raise FileNotFoundError(f"Missing ranking metrics for {normalized_model_name}")
        import json

        return json.loads(ranking_metrics_path.read_text(encoding="utf-8"))

    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_ranking_config(settings, requested_model_name=normalized_model_name)
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)
    training_settings = resolved_config["training_settings"]
    model_settings = resolved_config["model_settings"]
    emb_dim = int(model_settings.get("embedding_dim", 16))
    eval_negatives = int(training_settings.get("eval_negatives", 49))
    batch_size = int(training_settings.get("batch_size", 256))
    if normalized_model_name == "din":
        batch_size = int(training_settings.get("din_batch_size", batch_size))

    test_dataset = SequenceRankingDataset(ranking_samples["test"])
    fused_candidates_by_user = _resolve_ranking_user_candidate_pool(ranking_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SequenceRankingEvalCollator(
            item_features=item_features,
            all_item_ids=all_item_ids,
            num_negatives=eval_negatives,
            seed=193,
            fused_candidates_by_user=fused_candidates_by_user,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(get_ranking_model_path(normalized_model_name), map_location=device)
    model = _build_model(normalized_model_name, feature_dict, model_settings, emb_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    metrics = _evaluate_ranking(model, test_loader, device)
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
    item_popularity = np.asarray(ranking_samples.get("item_popularity", np.zeros(0, dtype=np.int64)))
    final_topk = int(resolved_config["evaluation_settings"].get("final_topk", 20))
    final_metrics_path = get_final_metrics_path(normalized_model_name)

    if normalized_model_name in _XGBOOST_RANKING_MODELS:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(get_ranking_model_path(normalized_model_name))
        final_metrics = evaluate_final_candidates_xgboost(
            booster,
            ranking_samples,
            item_features=item_features,
            item_popularity=item_popularity,
            all_item_ids=all_item_ids,
            final_topk=final_topk,
            use_ranker=normalized_model_name == "xgboost_ranker",
        )
    else:
        feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
        model_settings = resolved_config["model_settings"]
        emb_dim = int(model_settings.get("embedding_dim", 16))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(get_ranking_model_path(normalized_model_name), map_location=device)
        model = _build_model(normalized_model_name, feature_dict, model_settings, emb_dim).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
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
    fused_candidates_by_user = _resolve_ranking_user_candidate_pool(ranking_samples)
    val_ratio = float(training_settings.get("validation_ratio", 0.1))
    val_size = max(1, int(len(train_dataset) * val_ratio)) if len(train_dataset) > 1 else 0
    train_size = max(len(train_dataset) - val_size, 0)
    val_indices: list[int] = []
    if train_size == 0:
        train_subset = train_dataset
        val_subset = None
    else:
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        val_indices = [int(index) for index in val_subset.indices]

    batch_size = int(training_settings.get("batch_size", 256))
    epochs = int(training_settings.get("epochs", 3))
    learning_rate = float(training_settings.get("learning_rate", 0.001))
    weight_decay = float(training_settings.get("weight_decay", 1e-5))
    log_every_n_batches = int(training_settings.get("log_every_n_batches", 100))
    early_stopping_patience = int(training_settings.get("early_stopping_patience", 2))
    min_delta = float(training_settings.get("min_delta", 1e-4))
    emb_dim = int(model_settings.get("embedding_dim", 16))
    train_negatives = int(training_settings.get("train_negatives", 7))
    eval_negatives = int(training_settings.get("eval_negatives", 49))
    if model_name == "din":
        batch_size = int(training_settings.get("din_batch_size", batch_size))
        log_every_n_batches = int(training_settings.get("din_log_every_n_batches", log_every_n_batches))

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SequenceRankingTrainCollator(
            item_features=item_features,
            all_item_ids=all_item_ids,
            num_negatives=train_negatives,
            seed=42,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, feature_dict, model_settings, emb_dim).to(device)
    ranking_model_path = get_ranking_model_path(model_name)
    ranking_model_config_path = get_ranking_model_config_path(model_name)
    if warm_start and ranking_model_path.exists():
        checkpoint = torch.load(ranking_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info(
        "Ranking training start | model=%s | protocol=%s | device=%s | train_samples=%s | val_samples=%s | batch_size=%s | epochs=%s | train_negatives=%s | fused_validation=%s | warm_start=%s",
        model_name,
        ranking_samples.get("protocol", "unknown"),
        device,
        len(train_dataset),
        len(val_indices),
        batch_size,
        epochs,
        train_negatives,
        bool(fused_candidates_by_user),
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
            sample_weight = _target_rating_weights(batch["target_rating"], training_settings)
            loss = _candidate_loss(logits, batch["target_index"], sample_weight)
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
        val_metrics = None
        if val_indices and fused_candidates_by_user:
            val_metrics = _evaluate_fused_candidates_subset(
                model,
                ranking_samples["train"],
                val_indices,
                item_features,
                all_item_ids,
                device,
                final_topk=int(evaluation_settings.get("final_topk", 20)),
                fused_candidates_by_user=fused_candidates_by_user,
            )
        val_ndcg20 = float(val_metrics.get("ndcg@20", float("-inf"))) if val_metrics is not None else float("-inf")

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

        if val_metrics is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            logger.info("Ranking validation skipped | model=%s | reason=missing_fused_candidates_or_val_subset", model_name)
            break

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



def _build_model(model_name: str, feature_dict: dict, model_settings: dict, emb_dim: int):
    if model_name == "deepfm":
        scorer = DeepFMModel(
            feature_dict,
            POINTWISE_RANKING_FIELDS,
            emb_dim,
            dnn_hidden_dims=model_settings.get("dnn_hidden_dims"),
            dropout=float(model_settings.get("dropout", 0.1)),
            history_fields=POINTWISE_SEQUENCE_FIELDS,
        )
        return PointwiseCandidateRanker(scorer, static_fields=STATIC_USER_FIELDS)

    if model_name == "din":
        scorer = DINModel(
            feature_dict,
            POINTWISE_RANKING_FIELDS,
            emb_dim,
            dnn_hidden_dims=model_settings.get("dnn_hidden_dims"),
            attention_hidden_dims=model_settings.get("attention_hidden_dims"),
            dropout=float(model_settings.get("dropout", 0.1)),
        )
        return PointwiseCandidateRanker(scorer, static_fields=STATIC_USER_FIELDS)

    raise ValueError(f"Unsupported ranking model: {model_name}")



def _run_xgboost_training(model_name: str, warm_start: bool = True):
    settings = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_ranking_config(settings, requested_model_name=model_name)
    training_settings = resolved_config["training_settings"]
    evaluation_settings = resolved_config["evaluation_settings"]
    xgboost_params = resolved_config["xgboost_params"]
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    feature_dict = load_pickle(RANKING_FEATURE_DICT_PATH)
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)
    fused_candidates_by_user = _resolve_ranking_user_candidate_pool(ranking_samples)
    item_popularity = np.asarray(ranking_samples.get("item_popularity", np.zeros(0, dtype=np.int64)))
    train_negatives = int(training_settings.get("train_negatives", 7))
    eval_negatives = int(training_settings.get("eval_negatives", 49))
    ranking_model_path = get_ranking_model_path(model_name)
    ranking_model_config_path = get_ranking_model_config_path(model_name)
    ranking_metrics_path = get_ranking_metrics_path(model_name)

    train_rows, train_labels, train_groups, train_weights = build_xgboost_training_frame(
        ranking_samples["train"],
        item_features=item_features,
        all_item_ids=all_item_ids,
        num_negatives=train_negatives,
        item_popularity=item_popularity,
        seed=42,
    )
    test_rows, test_labels, test_groups, _ = build_xgboost_training_frame(
        ranking_samples["test"],
        item_features=item_features,
        all_item_ids=all_item_ids,
        num_negatives=eval_negatives,
        item_popularity=item_popularity,
        seed=193,
        fused_candidates_by_user=fused_candidates_by_user,
    )
    if train_rows.size == 0 or test_rows.size == 0:
        raise ValueError("XGBoost ranking received empty training or evaluation rows")

    model_params = dict(xgboost_params or {})
    model_params.setdefault("n_estimators", 300)
    model_params.setdefault("max_depth", 6)
    model_params.setdefault("learning_rate", 0.05)
    model_params.setdefault("subsample", 0.9)
    model_params.setdefault("colsample_bytree", 0.9)
    model_params.setdefault("reg_lambda", 1.0)
    model_params.setdefault("min_child_weight", 1.0)
    model_params.setdefault("random_state", 42)
    model_params.setdefault("tree_method", "hist")
    model_params.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "XGBoost training start | model=%s | protocol=%s | train_rows=%s | test_rows=%s | train_negatives=%s | eval_negatives=%s | warm_start=%s",
        model_name,
        ranking_samples.get("protocol", "unknown"),
        int(train_rows.shape[0]),
        int(test_rows.shape[0]),
        train_negatives,
        eval_negatives,
        warm_start,
    )

    warm_start_path = ranking_model_path if warm_start and ranking_model_path.exists() else None
    booster = train_xgboost_model(
        model_name,
        train_rows=train_rows,
        train_labels=train_labels,
        train_groups=train_groups,
        model_params=model_params,
        warm_start_model_path=warm_start_path,
        sample_weights=train_weights,
    )
    test_scores = predict_xgboost_scores(booster, test_rows, use_ranker=model_name == "xgboost_ranker")
    metrics = evaluate_xgboost_scores(test_labels, test_scores, test_groups)
    save_metrics(ranking_metrics_path, metrics)
    booster.save_model(ranking_model_path)
    save_pickle(
        {
            "protocol": ranking_samples.get("protocol", "unknown"),
            "model_name": model_name,
            "feature_dict": feature_dict,
            "model_settings": model_params,
            "training_settings": training_settings,
            "evaluation_settings": evaluation_settings,
            "train_negatives": train_negatives,
            "eval_negatives": eval_negatives,
            "latest_ranking_metrics": metrics,
        },
        ranking_model_config_path,
    )
    logger.info("Ranking metrics saved | model=%s | path=%s | metrics=%s", model_name, ranking_metrics_path.name, metrics)
    return {"model_name": model_name, "model_path": str(ranking_model_path), "config_path": str(ranking_model_config_path), "ranking": metrics}



def _candidate_loss(logits: torch.Tensor, target_index: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    losses = F.cross_entropy(logits, target_index, reduction="none")
    if sample_weight is None:
        return losses.mean()
    normalized = sample_weight.float() / sample_weight.float().mean().clamp_min(1e-9)
    return (losses * normalized).mean()



def _evaluate_loss(model, data_loader, device: torch.device, training_settings: dict | None = None) -> float:
    model.eval()
    total_loss = 0.0
    total_groups = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            logits = model(batch)
            sample_weight = _target_rating_weights(batch["target_rating"], training_settings)
            loss = _candidate_loss(logits, batch["target_index"], sample_weight)
            batch_groups = int(batch["target_index"].numel())
            total_loss += float(loss.item()) * batch_groups
            total_groups += batch_groups
    return total_loss / max(total_groups, 1)



def _evaluate_ranking(model, data_loader, device: torch.device) -> dict:
    model.eval()
    flat_labels: list[np.ndarray] = []
    flat_scores: list[np.ndarray] = []
    flat_user_ids: list[np.ndarray] = []
    group_losses: list[float] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=-1)
            target_index = batch["target_index"]
            positive_probabilities = probabilities.gather(1, target_index.unsqueeze(1)).squeeze(1)
            group_losses.extend((-torch.log(positive_probabilities.clamp_min(1e-9))).cpu().numpy().tolist())

            labels = torch.zeros_like(probabilities)
            labels.scatter_(1, target_index.unsqueeze(1), 1.0)
            user_ids = batch["user_id_original"].cpu().numpy()

            flat_labels.append(labels.cpu().numpy().reshape(-1))
            flat_scores.append(probabilities.cpu().numpy().reshape(-1))
            flat_user_ids.append(np.repeat(user_ids, probabilities.size(1)))

    labels = np.concatenate(flat_labels) if flat_labels else np.zeros(0, dtype=np.float32)
    scores = np.concatenate(flat_scores) if flat_scores else np.zeros(0, dtype=np.float32)
    user_ids = np.concatenate(flat_user_ids) if flat_user_ids else np.zeros(0, dtype=np.int32)
    metrics = ranking_metrics(labels, scores, user_ids)
    metrics["group_cross_entropy"] = float(np.mean(group_losses)) if group_losses else 0.0
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
    skipped_missing_candidates = 0

    model.eval()
    with torch.no_grad():
        for sample_idx in sample_indices:
            sample = extract_split_sample(split_data, sample_idx)
            user_id = int(sample["user_id"])
            candidates = fused_candidates_by_user.get(user_id)
            if candidates is None or int(np.asarray(candidates).size) == 0:
                skipped_missing_candidates += 1
                continue
            batch, valid_candidates = build_inference_batch(sample, np.asarray(candidates, dtype=np.int32).tolist(), item_features)
            if batch is None or not valid_candidates:
                continue

            batch = batch_to_device(batch, device)
            scores = model(batch)[0].cpu().numpy().tolist()
            target = int(sample["target_movie_id"])
            ranked_pairs = sorted(zip(valid_candidates, scores), key=lambda pair: pair[1], reverse=True)[:final_topk]
            ranked_candidates = [int(candidate) for candidate, _ in ranked_pairs]
            ranked_scores = [float(score) for _, score in ranked_pairs]
            ranked_labels = [1.0 if int(candidate) == target else 0.0 for candidate in ranked_candidates]

            candidate_user_ids.append(user_id)
            candidate_ids.append(ranked_candidates)
            candidate_scores.append(ranked_scores)
            candidate_labels.append(ranked_labels)

    metrics = evaluate_final_candidates(
        np.asarray(candidate_user_ids, dtype=np.int32),
        candidate_ids,
        candidate_scores,
        candidate_labels,
        all_item_ids=all_item_ids,
        output_path=None,
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
