import json
from collections import defaultdict
from pathlib import Path

import numpy as np



def _binary_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(np.mean(losses)) if losses.size > 0 else 0.0



def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end
    positive_ranks = ranks[sorted_labels == 1]
    auc = (positive_ranks.sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)



def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))



def _group_rankings(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray):
    grouped = defaultdict(list)
    for uid, label, score in zip(user_ids, labels, scores):
        grouped[int(uid)].append((float(label), float(score)))
    return grouped



def recall_at_k(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray, k: int) -> float:
    grouped = _group_rankings(labels, scores, user_ids)
    recalls = []
    for group in grouped.values():
        positives = sum(label > 0 for label, _ in group)
        if positives == 0:
            recalls.append(0.0)
            continue
        ranked = sorted(group, key=lambda x: x[1], reverse=True)[:k]
        hits = sum(label > 0 for label, _ in ranked)
        recalls.append(hits / positives)
    return float(np.mean(recalls)) if recalls else 0.0



def hit_rate_at_k(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray, k: int) -> float:
    grouped = _group_rankings(labels, scores, user_ids)
    hits = []
    for group in grouped.values():
        ranked = sorted(group, key=lambda x: x[1], reverse=True)[:k]
        hits.append(float(any(label > 0 for label, _ in ranked)))
    return float(np.mean(hits)) if hits else 0.0



def precision_at_k(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray, k: int) -> float:
    grouped = _group_rankings(labels, scores, user_ids)
    precisions = []
    for group in grouped.values():
        ranked = sorted(group, key=lambda x: x[1], reverse=True)[:k]
        if not ranked:
            precisions.append(0.0)
            continue
        hits = sum(label > 0 for label, _ in ranked)
        precisions.append(hits / len(ranked))
    return float(np.mean(precisions)) if precisions else 0.0



def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray, k: int) -> float:
    grouped = _group_rankings(labels, scores, user_ids)
    ndcgs = []
    for group in grouped.values():
        ranked = sorted(group, key=lambda x: x[1], reverse=True)
        topk = np.array([label for label, _ in ranked[:k]], dtype=np.float32)
        ideal = np.sort(np.array([label for label, _ in group], dtype=np.float32))[::-1][:k]
        denom = _dcg(ideal)
        ndcgs.append(0.0 if denom == 0 else _dcg(topk) / denom)
    return float(np.mean(ndcgs)) if ndcgs else 0.0



def ranking_metrics(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    return {
        "auc": float(_binary_auc(labels, scores)),
        "gauc": float(_group_auc(labels, scores, user_ids)),
        "logloss": float(_binary_logloss(labels, scores)),
    }



def _group_auc(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray) -> float:
    grouped = defaultdict(lambda: {"labels": [], "scores": []})
    for uid, label, score in zip(user_ids, labels, scores):
        grouped[int(uid)]["labels"].append(int(label))
        grouped[int(uid)]["scores"].append(float(score))

    total_weight = 0
    weighted_auc = 0.0
    for group in grouped.values():
        group_labels = np.array(group["labels"])
        if len(np.unique(group_labels)) < 2:
            continue
        group_scores = np.array(group["scores"])
        weight = len(group_labels)
        weighted_auc += _binary_auc(group_labels, group_scores) * weight
        total_weight += weight
    return 0.0 if total_weight == 0 else weighted_auc / total_weight



def save_metrics(path: Path, metrics: dict):
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
