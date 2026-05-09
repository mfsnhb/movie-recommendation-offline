from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
import yaml

from offline.evaluate.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, save_metrics
from offline.models.sequence_retrieval import SequenceRetrievalModel
from offline.models.two_tower import TwoTowerRetrievalModel
from offline.ranking.protocol import get_all_item_ids, get_item_feature_arrays, get_seen_movie_ids
from offline.utils.config import resolve_multi_recall_route_names, resolve_retrieval_config
from offline.utils.io import (
    CONFIG_DIR,
    GENRE_MODEL_PATH,
    ITEM_CF_MODEL_PATH,
    ITEM_CATALOG_PATH,
    ITEM_EMBEDDINGS_PATH,
    MOVIE_IDS_PATH,
    MULTI_RECALL_ARTIFACTS_PATH,
    MULTI_RECALL_METRICS_PATH,
    POPULAR_MODEL_PATH,
    RANKING_SAMPLE_PATH,
    RETRIEVAL_FEATURE_DICT_PATH,
    RETRIEVAL_MODEL_PATH,
    RETRIEVAL_SAMPLE_PATH,
    SEQUENCE_ITEM_EMBEDDINGS_PATH,
    SEQUENCE_MODEL_PATH,
    load_pickle,
    save_pickle,
)
from offline.utils.logging import get_logger


logger = get_logger("offline.evaluate.multi_recall")
DEFAULT_ROUTE_ORDER = ["two_tower", "sequence", "item_cf", "multimodal", "genre", "popular"]
RRF_K = 60
MIN_ROUTE_WEIGHT = 0.3
MAX_ROUTE_WEIGHT = 1.0



def run_multi_recall_build(routes=None, topk: int | None = None):
    settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_retrieval_config(settings)
    selected_routes = resolve_multi_recall_route_names(settings, routes)
    resolved_topk = int(topk or resolved_config["evaluation_settings"].get("topk", 200))
    return build_multi_recall_artifacts(topk=resolved_topk, routes=selected_routes)



def build_multi_recall_artifacts(topk: int, routes: list[str] | None = None):
    retrieval_samples = load_pickle(RETRIEVAL_SAMPLE_PATH)
    retrieval_samples.pop("train", None)
    _validate_retrieval_ranking_targets(retrieval_samples)
    test = retrieval_samples["test"]
    route_outputs = build_route_outputs(topk=topk, routes=routes, split_name="test", retrieval_samples=retrieval_samples)
    route_order = list(routes or DEFAULT_ROUTE_ORDER)
    metric_ks = sorted({50, 100, min(200, topk)})
    allocation_k = min(200, topk)

    logger.info(
        "Multi-recall artifact build start | users=%s | topk=%s | routes=%s | fusion_method=rrf | rrf_k=%s",
        len(test["user_id"]),
        topk,
        route_order,
        RRF_K,
    )

    validation_data = retrieval_samples["validation"]
    validation_route_outputs = build_route_outputs(topk=topk, routes=routes, split_name="validation", retrieval_samples=retrieval_samples)
    validation_metrics, validation_route_recalls = evaluate_route_outputs(validation_route_outputs, validation_data, topk)
    incremental_stats = _compute_incremental_route_recalls(route_order, validation_route_outputs, validation_data, topk, validation_route_recalls)
    weight_source = "validation"
    route_weights, prioritized_routes = _allocate_route_weights(
        route_order,
        incremental_stats["route_incremental_recalls"],
        effective_routes=incremental_stats["effective_routes"],
    )
    clipped_route_weights = _clip_route_weights(route_order, route_weights, incremental_stats["effective_routes"])
    logger.info(
        "Multi-recall fusion setup | weight_source=%s | method=rrf | rrf_k=%s | metric=incremental_recall@%s | prioritized_routes=%s | route_weights=%s | clipped_route_weights=%s | incremental_recalls=%s | effective_routes=%s",
        weight_source,
        RRF_K,
        allocation_k,
        prioritized_routes,
        route_weights,
        clipped_route_weights,
        incremental_stats["route_incremental_recalls"],
        incremental_stats["effective_routes"],
    )

    per_route_metrics, route_recalls = evaluate_route_outputs(route_outputs, test, topk)
    fused_candidates, fused_scores, labels, user_ids = [], [], [], []
    for idx, user_id in enumerate(test["user_id"]):
        target = int(np.asarray(test["movie_id"][idx]).reshape(-1)[0])
        route_candidate_lists = {name: list(output.get(int(user_id), {}).items()) for name, output in route_outputs.items()}
        ranked_items = _rrf_candidates(route_candidate_lists, prioritized_routes, clipped_route_weights, topk, RRF_K)
        candidates = [int(item) for item, _ in ranked_items]
        scores = [float(score) for _, score in ranked_items]
        fused_candidates.append(candidates)
        fused_scores.append(scores)
        user_ids.extend([int(user_id)] * len(candidates))
        labels.extend([1.0 if item == target else 0.0 for item in candidates])

    fused_metrics = _topk_metrics(
        np.array(labels),
        np.array([score for row in fused_scores for score in row]),
        np.array(user_ids),
        metric_ks,
    )
    metric_summary = {
        "routes": per_route_metrics,
        "fused": fused_metrics,
        "route_order": route_order,
        "prioritized_routes": prioritized_routes,
        "route_weights": route_weights,
        "route_weights_clipped": clipped_route_weights,
        "route_single_recalls": incremental_stats["route_single_recalls"],
        "route_incremental_recalls": incremental_stats["route_incremental_recalls"],
        "effective_routes": incremental_stats["effective_routes"],
        "weight_source": weight_source,
        "validation_metrics": validation_metrics,
        "allocation_metric": f"incremental_recall@{allocation_k}",
        "fusion_method": "rrf",
        "rrf_k": RRF_K,
        "route_weight_clip": {"min": MIN_ROUTE_WEIGHT, "max": MAX_ROUTE_WEIGHT},
        "score_normalization": "rrf_rank_weighted_by_clipped_incremental_recall",
    }
    save_metrics(MULTI_RECALL_METRICS_PATH, metric_summary)
    artifacts = {
        "route_order": route_order,
        "prioritized_routes": prioritized_routes,
        "route_weights": route_weights,
        "route_weights_clipped": clipped_route_weights,
        "route_single_recalls": incremental_stats["route_single_recalls"],
        "route_incremental_recalls": incremental_stats["route_incremental_recalls"],
        "effective_routes": incremental_stats["effective_routes"],
        "weight_source": weight_source,
        "fusion_method": "rrf",
        "rrf_k": RRF_K,
        "route_weight_clip": {"min": MIN_ROUTE_WEIGHT, "max": MAX_ROUTE_WEIGHT},
        "score_normalization": "rrf_rank_weighted_by_clipped_incremental_recall",
        "route_outputs": route_outputs,
        "fused_candidates": fused_candidates,
        "fused_scores": fused_scores,
        "user_ids": np.asarray(test["user_id"]),
        "targets": np.asarray(test["movie_id"]),
        "metrics": metric_summary,
    }
    save_pickle(artifacts, MULTI_RECALL_ARTIFACTS_PATH)
    logger.info("Multi-recall artifact build done | fused_metrics=%s", fused_metrics)
    return artifacts


def _validate_retrieval_ranking_targets(retrieval_samples: dict) -> None:
    if not RANKING_SAMPLE_PATH.exists():
        logger.warning(
            "Retrieval/ranking target alignment check skipped | reason=missing_ranking_samples | path=%s",
            RANKING_SAMPLE_PATH.name,
        )
        return
    ranking_samples = load_pickle(RANKING_SAMPLE_PATH)
    for split_name in ("validation", "test"):
        retrieval_split = retrieval_samples.get(split_name)
        ranking_split = ranking_samples.get(split_name)
        if not isinstance(retrieval_split, dict) or not isinstance(ranking_split, dict):
            raise ValueError(f"Missing {split_name} split in retrieval or ranking samples. Rerun preprocessing.")
        retrieval_users = np.asarray(retrieval_split["user_id"], dtype=np.int64).reshape(-1)
        retrieval_targets = np.asarray(retrieval_split["movie_id"], dtype=np.int64).reshape(-1)
        ranking_users = np.asarray(ranking_split["user_id"], dtype=np.int64).reshape(-1)
        ranking_targets = np.asarray(ranking_split["target_movie_id"], dtype=np.int64).reshape(-1)
        if retrieval_users.shape[0] != ranking_users.shape[0]:
            raise ValueError(
                f"Retrieval/ranking {split_name} target count mismatch: "
                f"retrieval={retrieval_users.shape[0]} ranking={ranking_users.shape[0]}. Rerun preprocessing."
            )
        if not np.array_equal(retrieval_users, ranking_users) or not np.array_equal(retrieval_targets, ranking_targets):
            mismatches = np.flatnonzero((retrieval_users != ranking_users) | (retrieval_targets != ranking_targets))
            first_idx = int(mismatches[0]) if mismatches.size else 0
            raise ValueError(
                f"Retrieval/ranking {split_name} targets are not aligned. "
                f"first_mismatch_idx={first_idx} "
                f"retrieval=(user={int(retrieval_users[first_idx])}, target={int(retrieval_targets[first_idx])}) "
                f"ranking=(user={int(ranking_users[first_idx])}, target={int(ranking_targets[first_idx])}). "
                "Rerun ranking and retrieval preprocessing."
            )
        logger.info("Retrieval/ranking target alignment ok | split=%s | users=%s", split_name, retrieval_users.shape[0])



def build_route_outputs(topk: int, routes: list[str] | None = None, split_name: str = "test", retrieval_samples: dict | None = None):
    retrieval_samples = retrieval_samples or load_pickle(RETRIEVAL_SAMPLE_PATH)
    item_catalog = load_pickle(ITEM_CATALOG_PATH)
    settings = yaml.safe_load((CONFIG_DIR / "retrieval.yaml").read_text(encoding="utf-8")) or {}
    resolved_config = resolve_retrieval_config(settings)
    split = retrieval_samples[split_name]
    item_features = get_item_feature_arrays(item_catalog)
    all_item_ids = get_all_item_ids(item_catalog)
    route_order = list(routes or DEFAULT_ROUTE_ORDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    route_outputs: dict[str, dict[int, dict[int, float]]] = {}
    route_topk = topk

    encoded_movie_ids = None
    if MOVIE_IDS_PATH.exists():
        encoded_movie_ids = np.load(MOVIE_IDS_PATH, mmap_mode="r").astype(np.int64)

    route_builders = {
        "two_tower": lambda: _build_two_tower_output(split, encoded_movie_ids, item_features, device, route_topk),
        "sequence": lambda: _build_sequence_output(split, encoded_movie_ids, item_features, device, route_topk),
        "item_cf": lambda: _build_item_cf_output(split, route_topk),
        "multimodal": lambda: _build_multimodal_output(split, all_item_ids, item_features, route_topk, resolved_config["rating_semantics"]),
        "genre": lambda: _build_genre_output(split, all_item_ids, item_features, route_topk),
        "popular": lambda: _build_popular_output(split, route_topk),
    }
    for route_name in route_order:
        route_builder = route_builders.get(route_name)
        if route_builder is None:
            logger.warning("Multi-recall route skipped | route=%s | reason=unknown_route", route_name)
            route_outputs[route_name] = _empty_route_output(split["user_id"])
        else:
            route_outputs[route_name] = route_builder()
    route_outputs = _filter_seen_route_outputs(route_outputs, split, topk)
    return route_outputs



def evaluate_route_outputs(route_outputs: dict[str, dict[int, dict[int, float]]], test_data: dict, topk: int):
    metric_ks = sorted({50, 100, min(200, topk)})
    allocation_k = min(200, topk)
    per_route_metrics = {}
    route_recalls = {}
    for route_name, output in route_outputs.items():
        route_labels, route_scores, route_users = _evaluate_route(output, test_data, topk)
        route_metrics = _topk_metrics(route_labels, route_scores, route_users, metric_ks)
        per_route_metrics[route_name] = route_metrics
        route_recalls[route_name] = float(route_metrics.get(f"recall@{allocation_k}", 0.0))
        logger.info(
            "Multi-recall route metrics | route=%s | recall@%s=%.4f | precision@%s=%.4f | hr@%s=%.4f | ndcg@%s=%.4f",
            route_name,
            allocation_k,
            route_metrics.get(f"recall@{allocation_k}", 0.0),
            allocation_k,
            route_metrics.get(f"precision@{allocation_k}", 0.0),
            allocation_k,
            route_metrics.get(f"hr@{allocation_k}", 0.0),
            allocation_k,
            route_metrics.get(f"ndcg@{allocation_k}", 0.0),
        )
    return per_route_metrics, route_recalls



def evaluate_final_candidates(user_ids: np.ndarray, candidate_ids: list[list[int]], candidate_scores: list[list[float]], candidate_labels: list[list[float]], all_item_ids: np.ndarray | None = None, output_path=None):
    metrics = {}
    for k in (10, 20):
        flat_users, flat_scores, flat_labels, recommended_items = [], [], [], []
        for uid, ids, scores, labels in zip(user_ids, candidate_ids, candidate_scores, candidate_labels):
            limit = min(k, len(scores), len(labels), len(ids))
            flat_users.extend([int(uid)] * limit)
            flat_scores.extend(float(score) for score in scores[:limit])
            flat_labels.extend(float(label) for label in labels[:limit])
            recommended_items.extend(int(item_id) for item_id in ids[:limit])
        metrics[f"recall@{k}"] = recall_at_k(np.array(flat_labels), np.array(flat_scores), np.array(flat_users), k)
        metrics[f"hr@{k}"] = hit_rate_at_k(np.array(flat_labels), np.array(flat_scores), np.array(flat_users), k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(np.array(flat_labels), np.array(flat_scores), np.array(flat_users), k)
        if all_item_ids is not None and len(all_item_ids) > 0:
            metrics[f"coverage@{k}"] = float(len(set(recommended_items)) / len(set(np.asarray(all_item_ids).tolist())))
    if output_path is not None:
        save_metrics(output_path, metrics)
    return metrics



def _empty_route_output(test_user_ids) -> dict[int, dict[int, float]]:
    return {int(uid): {} for uid in np.asarray(test_user_ids).tolist()}




def _uniform_incremental_stats(route_order: list[str]) -> dict:
    if not route_order:
        return {"route_single_recalls": {}, "route_incremental_recalls": {}, "effective_routes": []}
    weight = round(1.0 / len(route_order), 6)
    return {
        "route_single_recalls": {route: weight for route in route_order},
        "route_incremental_recalls": {route: weight for route in route_order},
        "effective_routes": list(route_order),
    }




def _filter_seen_route_outputs(route_outputs: dict[str, dict[int, dict[int, float]]], test_data: dict, topk: int):
    seen_by_user = {
        int(user_id): set(get_seen_movie_ids({"hist_movie_id": test_data["hist_movie_id"][idx]}).tolist())
        for idx, user_id in enumerate(test_data["user_id"])
    }
    filtered_outputs = {}
    for route_name, user_outputs in route_outputs.items():
        filtered_user_outputs = {}
        for user_id, candidates in user_outputs.items():
            seen_items = seen_by_user.get(int(user_id), set())
            ranked = [
                (int(item_id), float(score))
                for item_id, score in candidates.items()
                if int(item_id) > 0 and int(item_id) not in seen_items
            ][:topk]
            filtered_user_outputs[int(user_id)] = dict(ranked)
        filtered_outputs[route_name] = filtered_user_outputs
    return filtered_outputs



def _build_two_tower_output(test_data: dict, encoded_movie_ids: np.ndarray | None, item_features: dict[str, np.ndarray], device: torch.device, topk: int):
    if encoded_movie_ids is None or not RETRIEVAL_MODEL_PATH.exists() or not ITEM_EMBEDDINGS_PATH.exists() or not RETRIEVAL_FEATURE_DICT_PATH.exists():
        logger.info("Multi-recall route skipped | route=two_tower | reason=missing_artifact")
        return _empty_route_output(test_data["user_id"])

    retrieval_checkpoint = torch.load(RETRIEVAL_MODEL_PATH, map_location=device)
    feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
    embeddings = np.load(ITEM_EMBEDDINGS_PATH, mmap_mode="r")
    if len(encoded_movie_ids) != int(embeddings.shape[0]):
        raise ValueError(f"Item embedding alignment mismatch: movie_ids={len(encoded_movie_ids)}, two_tower={embeddings.shape[0]}")

    model_settings = retrieval_checkpoint.get("model_settings", {})
    model = TwoTowerRetrievalModel(
        feature_dict,
        int(retrieval_checkpoint["emb_dim"]),
        user_hidden_dims=model_settings.get("user_hidden_dims"),
        item_hidden_dims=model_settings.get("item_hidden_dims"),
        dropout=float(model_settings.get("dropout", 0.1)),
        multimodal_table=item_features["multimodal_embedding"],
        item_feature_table=item_features,
        recent_history_length=int(model_settings.get("recent_history_length", 20)),
    ).to(device)
    model.load_state_dict(retrieval_checkpoint["state_dict"])
    model.eval()
    return _two_tower_candidates(test_data, model, embeddings, encoded_movie_ids, device, topk)



def _build_sequence_output(test_data: dict, encoded_movie_ids: np.ndarray | None, item_features: dict[str, np.ndarray], device: torch.device, topk: int):
    if encoded_movie_ids is None or not SEQUENCE_MODEL_PATH.exists() or not SEQUENCE_ITEM_EMBEDDINGS_PATH.exists() or not RETRIEVAL_FEATURE_DICT_PATH.exists():
        logger.info("Multi-recall route skipped | route=sequence | reason=missing_artifact")
        return _empty_route_output(test_data["user_id"])

    sequence_checkpoint = torch.load(SEQUENCE_MODEL_PATH, map_location=device)
    feature_dict = load_pickle(RETRIEVAL_FEATURE_DICT_PATH)
    embeddings = np.load(SEQUENCE_ITEM_EMBEDDINGS_PATH, mmap_mode="r")
    if len(encoded_movie_ids) != int(embeddings.shape[0]):
        raise ValueError(f"Item embedding alignment mismatch: movie_ids={len(encoded_movie_ids)}, sequence={embeddings.shape[0]}")

    model_settings = sequence_checkpoint.get("model_settings", {})
    emb_dim = int(sequence_checkpoint.get("emb_dim", model_settings.get("embedding_dim", 32)))
    model = SequenceRetrievalModel(
        feature_dict,
        emb_dim,
        hidden_dim=int(model_settings.get("hidden_dim", emb_dim)),
        num_layers=int(model_settings.get("num_layers", 1)),
        max_len=int(model_settings.get("max_len", model_settings.get("sequence_max_len", 10))),
        dropout=float(model_settings.get("dropout", model_settings.get("sequence_dropout", 0.1))),
        multimodal_table=item_features["multimodal_embedding"],
        item_feature_table=item_features,
    ).to(device)
    model.load_state_dict(sequence_checkpoint["state_dict"])
    model.eval()
    return _sequence_candidates(test_data, model, embeddings, encoded_movie_ids, device, topk)



def _build_item_cf_output(test_data: dict, topk: int):
    if not ITEM_CF_MODEL_PATH.exists():
        logger.info("Multi-recall route skipped | route=item_cf | reason=missing_artifact")
        return _empty_route_output(test_data["user_id"])
    return _item_cf_candidates(test_data, load_pickle(ITEM_CF_MODEL_PATH), topk)



def _build_genre_output(test_data: dict, all_item_ids: np.ndarray, item_features: dict[str, np.ndarray], topk: int):
    if GENRE_MODEL_PATH.exists():
        genre_to_items = load_pickle(GENRE_MODEL_PATH)
    else:
        logger.info("Multi-recall route skipped | route=genre | reason=missing_artifact")
        return _empty_route_output(test_data["user_id"])
    return _genre_candidates(test_data, genre_to_items, item_features, all_item_ids, topk)



def _build_popular_output(test_data: dict, topk: int):
    if not POPULAR_MODEL_PATH.exists():
        logger.info("Multi-recall route skipped | route=popular | reason=missing_artifact")
        return _empty_route_output(test_data["user_id"])
    ranked = load_pickle(POPULAR_MODEL_PATH)
    return _popular_candidates(test_data, ranked, topk)


def _build_multimodal_output(test_data: dict, all_item_ids: np.ndarray, item_features: dict[str, np.ndarray], topk: int, rating_semantics: dict):
    embeddings = np.asarray(item_features["multimodal_embedding"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        logger.info("Multi-recall route skipped | route=multimodal | reason=missing_embedding")
        return _empty_route_output(test_data["user_id"])
    return _multimodal_candidates(
        test_data,
        embeddings,
        all_item_ids,
        topk,
        positive_rating_min=float(rating_semantics.get("positive_rating_min", 3.0)),
        history_size=int(rating_semantics.get("multimodal_history_size", 10)),
    )



def _two_tower_candidates(test_data, model, embeddings, encoded_movie_ids, device, topk):
    outputs = {}
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
            user_embedding = model.encode_user(batch).cpu().numpy()[0]
            scores = embeddings @ user_embedding
            top_indices = np.argsort(scores)[::-1][:topk]
            outputs[int(user_id)] = {int(encoded_movie_ids[i]): float(scores[i]) for i in top_indices}
    return outputs



def _sequence_candidates(test_data, model, embeddings, encoded_movie_ids, device, topk):
    outputs = {}
    with torch.no_grad():
        for idx, user_id in enumerate(test_data["user_id"]):
            hist_movie_ids = torch.tensor(np.asarray(test_data["hist_movie_id"][idx]).reshape(1, -1), dtype=torch.long, device=device)
            hist_time_gap_bucket = torch.tensor(np.asarray(test_data["hist_time_gap_bucket"][idx]).reshape(1, -1), dtype=torch.long, device=device)
            hist_rating = torch.tensor(np.asarray(test_data["hist_rating"][idx]).reshape(1, -1), dtype=torch.float32, device=device)
            user_embedding = model.encode_user(
                hist_movie_ids,
                hist_time_gap_bucket,
                hist_rating,
            ).cpu().numpy()[0]
            scores = embeddings @ user_embedding
            top_indices = np.argsort(scores)[::-1][:topk]
            outputs[int(user_id)] = {int(encoded_movie_ids[i]): float(scores[i]) for i in top_indices}
    return outputs



def _item_cf_candidates(test_data, item_cf_model, topk):
    logger.info("ItemCF candidate generation start | users=%s | topk=%s", len(test_data["user_id"]), topk)
    outputs = {}
    fallback_users = 0
    for idx, user_id in enumerate(test_data["user_id"]):
        history = [int(x) for x in np.asarray(test_data["hist_movie_id"][idx]).reshape(-1) if int(x) > 0]
        scores = Counter()
        for rank, movie_id in enumerate(reversed(history[-50:]), start=1):
            weight = 1.0 / float(np.log2(rank + 1))
            for candidate_id, candidate_score in item_cf_model.get(movie_id, {}).items():
                scores[int(candidate_id)] += float(candidate_score) * weight
        if not scores:
            fallback_users += 1
        outputs[int(user_id)] = dict(scores.most_common(topk))
    logger.info("ItemCF candidate generation done | users=%s | empty_users=%s", len(outputs), fallback_users)
    return outputs


def _multimodal_candidates(test_data, embeddings: np.ndarray, all_item_ids: np.ndarray, topk: int, positive_rating_min: float, history_size: int):
    normalized_embeddings = _normalize_rows(embeddings)
    candidate_ids = np.asarray(all_item_ids, dtype=np.int64)
    candidate_matrix = normalized_embeddings[candidate_ids]
    outputs = {}
    for idx, user_id in enumerate(test_data["user_id"]):
        history_ids = np.asarray(test_data["hist_movie_id"][idx], dtype=np.int32).reshape(-1)
        ratings = np.asarray(test_data["hist_rating"][idx], dtype=np.float32).reshape(-1)
        valid_mask = (history_ids > 0) & (history_ids < normalized_embeddings.shape[0]) & (ratings >= positive_rating_min)
        valid_ids = history_ids[valid_mask][-history_size:]
        valid_ratings = ratings[valid_mask][-history_size:]
        if valid_ids.size == 0:
            outputs[int(user_id)] = {}
            continue
        user_vector = (normalized_embeddings[valid_ids] * valid_ratings[:, None]).sum(axis=0) / valid_ratings.sum().clip(min=1e-9)
        user_norm = np.linalg.norm(user_vector)
        if user_norm > 0:
            user_vector = user_vector / user_norm
        scores = candidate_matrix @ user_vector
        seen_items = set(get_seen_movie_ids({"hist_movie_id": history_ids}).tolist())
        take = min(max(topk + len(seen_items), topk), scores.shape[0])
        top_positions = np.argpartition(scores, kth=scores.shape[0] - take)[-take:]
        top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
        ranked = {}
        for position in top_positions.tolist():
            item_id = int(candidate_ids[position])
            if item_id in seen_items:
                continue
            ranked[item_id] = float(scores[position])
            if len(ranked) >= topk:
                break
        outputs[int(user_id)] = ranked
    return outputs


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)



def _genre_candidates(test_data, genre_to_items: dict[int, list[int]], item_features: dict[str, np.ndarray], all_item_ids: np.ndarray, topk: int):
    if not genre_to_items:
        genre_to_items = defaultdict(list)
        genre_matrix = np.asarray(item_features["genres"], dtype=np.int32)
        for item_id in all_item_ids.tolist():
            for genre_id in (idx + 1 for idx in np.flatnonzero(genre_matrix[item_id]).tolist()):
                genre_to_items[int(genre_id)].append(int(item_id))
    outputs = {}
    for idx, user_id in enumerate(test_data["user_id"]):
        history_ids = np.asarray(test_data["hist_movie_id"][idx], dtype=np.int32).reshape(-1)
        valid_history = history_ids[(history_ids > 0) & (history_ids < item_features["genres"].shape[0])]
        preference_scores = Counter()
        for rank, movie_id in enumerate(reversed(valid_history[-10:]), start=1):
            movie_genres = np.flatnonzero(np.asarray(item_features["genres"][movie_id], dtype=np.int32)).tolist()
            for genre_idx in movie_genres:
                preference_scores[int(genre_idx) + 1] += 1.0 / rank
        scores = {}
        for genre, genre_weight in preference_scores.most_common(3):
            for item_rank, item_id in enumerate(genre_to_items.get(int(genre), [])[: topk // 2 or 1], start=1):
                scores[item_id] = max(scores.get(item_id, 0.0), float(genre_weight) / item_rank)
        outputs[int(user_id)] = scores
    return outputs



def _popular_candidates(test_data: dict, ranked_items: dict[int, float], topk: int):
    ranked_pairs = [(int(item_id), float(score)) for item_id, score in ranked_items.items()]
    outputs = {}
    for idx, user_id in enumerate(test_data["user_id"]):
        seen_items = set(get_seen_movie_ids({"hist_movie_id": test_data["hist_movie_id"][idx]}).tolist())
        candidates = [(item_id, score) for item_id, score in ranked_pairs if item_id not in seen_items][:topk]
        outputs[int(user_id)] = dict(candidates)
    return outputs


def _normalize_route_scores(
    route_outputs: dict[str, dict[int, dict[int, float]]],
    route_weights: dict[str, float],
) -> dict[str, dict[int, dict[int, float]]]:
    return {
        route: {
            int(user_id): _normalize_candidate_scores(candidates, route_weight=float(route_weights.get(route, 0.0)))
            for user_id, candidates in user_outputs.items()
        }
        for route, user_outputs in route_outputs.items()
    }


def _normalize_candidate_scores(candidates: dict[int, float], route_weight: float) -> dict[int, float]:
    ranked = sorted(
        ((int(item_id), float(score)) for item_id, score in candidates.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0][0]: float(route_weight)}

    scores = np.asarray([score for _, score in ranked], dtype=np.float32)
    if np.all(np.isfinite(scores)) and float(scores.max()) > float(scores.min()):
        normalized = (scores - float(scores.min())) / (float(scores.max()) - float(scores.min()))
    else:
        normalized = 1.0 - (np.arange(len(ranked), dtype=np.float32) / max(len(ranked) - 1, 1))
    return {item_id: float(score * route_weight) for (item_id, _), score in zip(ranked, normalized, strict=True)}



def _compute_incremental_route_recalls(
    route_order: list[str],
    route_outputs: dict[str, dict[int, dict[int, float]]],
    test_data: dict,
    topk: int,
    route_recalls: dict[str, float],
) -> dict:
    allocation_k = min(200, topk)
    route_rank = {route: idx for idx, route in enumerate(route_order)}
    candidate_routes = [route for route in route_order if route in route_outputs]
    evaluation_order = sorted(
        candidate_routes,
        key=lambda route: (-float(route_recalls.get(route, 0.0)), route_rank[route]),
    )
    target_by_user = {
        int(user_id): int(np.asarray(test_data["movie_id"][idx]).reshape(-1)[0])
        for idx, user_id in enumerate(test_data["user_id"])
    }
    total_users = max(len(target_by_user), 1)
    route_hits: dict[str, set[int]] = {}
    for route in candidate_routes:
        hits = set()
        output = route_outputs.get(route, {})
        for user_id, target in target_by_user.items():
            ranked = list(output.get(user_id, {}).keys())[:allocation_k]
            if target in {int(item_id) for item_id in ranked}:
                hits.add(user_id)
        route_hits[route] = hits

    covered_users: set[int] = set()
    incremental_recalls = {route: 0.0 for route in candidate_routes}
    effective_routes: list[str] = []
    for route in evaluation_order:
        new_hits = route_hits[route] - covered_users
        incremental_recall = len(new_hits) / total_users
        incremental_recalls[route] = float(incremental_recall)
        if new_hits:
            effective_routes.append(route)
            covered_users.update(new_hits)

    if not effective_routes and evaluation_order:
        fallback_route = evaluation_order[0]
        effective_routes = [fallback_route]
        incremental_recalls[fallback_route] = max(float(route_recalls.get(fallback_route, 0.0)), 1.0 / total_users)

    rounded_incremental = {route: round(float(incremental_recalls.get(route, 0.0)), 6) for route in candidate_routes}
    rounded_single = {route: round(float(route_recalls.get(route, 0.0)), 6) for route in candidate_routes}
    return {
        "route_single_recalls": rounded_single,
        "route_incremental_recalls": rounded_incremental,
        "effective_routes": effective_routes,
    }



def _allocate_route_weights(
    route_order: list[str],
    route_recalls: dict[str, float],
    effective_routes: list[str] | None = None,
):
    route_rank = {route: idx for idx, route in enumerate(route_order)}
    available_routes = [route for route in route_order if route in route_recalls]
    if effective_routes is not None:
        effective_set = set(effective_routes)
        available_routes = [route for route in available_routes if route in effective_set]
    raw_weights = {route: max(float(route_recalls.get(route, 0.0)), 0.0) for route in available_routes}
    total_weight = sum(raw_weights.values())
    if total_weight <= 0 and available_routes:
        fallback_route = max(available_routes, key=lambda route: (float(route_recalls.get(route, 0.0)), -route_rank[route]))
        raw_weights = {route: 1.0 if route == fallback_route else 0.0 for route in available_routes}
        total_weight = 1.0
    normalized_weights = {route: (weight / total_weight if total_weight > 0 else 0.0) for route, weight in raw_weights.items()}
    prioritized_routes = sorted(
        available_routes,
        key=lambda route: (-normalized_weights[route], route_rank[route]),
    )
    rounded_weights = {route: round(normalized_weights[route], 6) for route in prioritized_routes}
    return rounded_weights, prioritized_routes



def _clip_route_weights(
    route_order: list[str],
    route_weights: dict[str, float],
    effective_routes: list[str] | None = None,
    min_weight: float = MIN_ROUTE_WEIGHT,
    max_weight: float = MAX_ROUTE_WEIGHT,
) -> dict[str, float]:
    effective_set = set(effective_routes or route_order)
    available_routes = [route for route in route_order if route in route_weights and route in effective_set]
    if not available_routes:
        available_routes = [route for route in route_order if route in route_weights]
    if not available_routes:
        return {}

    positive_weights = [float(route_weights.get(route, 0.0)) for route in available_routes if float(route_weights.get(route, 0.0)) > 0]
    if not positive_weights:
        return {route: 1.0 for route in available_routes}

    mean_weight = float(np.mean(positive_weights))
    clipped = {}
    for route in available_routes:
        relative_weight = float(route_weights.get(route, 0.0)) / mean_weight if mean_weight > 0 else 1.0
        clipped[route] = round(float(np.clip(relative_weight, min_weight, max_weight)), 6)
    return clipped



def _rrf_candidates(
    route_candidate_lists: dict[str, list[tuple[int, float]]],
    prioritized_routes: list[str],
    route_weights: dict[str, float],
    topk: int,
    rrf_k: int = RRF_K,
):
    fusion_scores = Counter()
    route_rank = {route: idx for idx, route in enumerate(prioritized_routes)}
    best_rank = {}
    best_route = {}

    for route in prioritized_routes:
        weight = float(route_weights.get(route, 0.0))
        if weight <= 0:
            continue
        ranked = sorted(
            ((int(item_id), float(score)) for item_id, score in route_candidate_lists.get(route, [])),
            key=lambda item: item[1],
            reverse=True,
        )
        for rank, (item_id, _) in enumerate(ranked, start=1):
            if item_id <= 0:
                continue
            fusion_scores[item_id] += weight / float(rrf_k + rank)
            current_best_rank = best_rank.get(item_id)
            if current_best_rank is None or rank < current_best_rank:
                best_rank[item_id] = rank
                best_route[item_id] = route

    ranked_items = sorted(
        fusion_scores.items(),
        key=lambda item: (-float(item[1]), best_rank.get(item[0], 0), route_rank.get(best_route.get(item[0], ""), len(route_rank)), item[0]),
    )
    return [(int(item_id), float(score)) for item_id, score in ranked_items[:topk]]



def _evaluate_route(route_output, test_data, topk):
    labels, scores, user_ids = [], [], []
    for idx, user_id in enumerate(test_data["user_id"]):
        target = int(np.asarray(test_data["movie_id"][idx]).reshape(-1)[0])
        ranked = list(route_output.get(int(user_id), {}).items())[:topk]
        for movie_id, score in ranked:
            labels.append(1.0 if int(movie_id) == target else 0.0)
            scores.append(float(score))
            user_ids.append(int(user_id))
    return np.array(labels), np.array(scores), np.array(user_ids)



def _topk_metrics(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray, ks: list[int]):
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(labels, scores, user_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(labels, scores, user_ids, k)
        metrics[f"hr@{k}"] = hit_rate_at_k(labels, scores, user_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, scores, user_ids, k)
    return metrics
