from __future__ import annotations

import numpy as np

from offline.evaluate.metrics import ranking_metrics
from offline.evaluate.multi_recall import evaluate_final_candidates
from offline.ranking.protocol import extract_split_sample
from offline.ranking.xgboost_features import build_xgboost_inference_frame
from offline.utils.io import MULTI_RECALL_ARTIFACTS_PATH, load_pickle


try:
    import xgboost as xgb
except ImportError as exc:
    xgb = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None



def is_xgboost_ranker(model_name: str) -> bool:
    return model_name.strip().lower() == "xgboost_ranker"



def ensure_xgboost_available() -> None:
    if _XGBOOST_IMPORT_ERROR is not None:
        raise ImportError(
            "xgboost is required for ranking models 'xgboost' and 'xgboost_ranker'. Install xgboost to enable these baselines."
        ) from _XGBOOST_IMPORT_ERROR



def _normalize_training_params(model_params: dict) -> tuple[dict, int]:
    params = dict(model_params)
    num_boost_round = int(params.pop("n_estimators", params.pop("num_boost_round", 300)))
    learning_rate = params.pop("learning_rate", None)
    if learning_rate is not None and "eta" not in params:
        params["eta"] = learning_rate
    random_state = params.pop("random_state", None)
    if random_state is not None and "seed" not in params:
        params["seed"] = random_state
    return params, num_boost_round



def train_xgboost_model(
    model_name: str,
    train_rows: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    model_params: dict,
    warm_start_model_path: str | None = None,
    sample_weights: np.ndarray | None = None,
):
    ensure_xgboost_available()
    params, num_boost_round = _normalize_training_params(model_params)
    train_matrix = xgb.DMatrix(np.asarray(train_rows, dtype=np.float32), label=np.asarray(train_labels, dtype=np.float32), weight=None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32))

    if is_xgboost_ranker(model_name):
        train_matrix.set_group(np.asarray(train_groups, dtype=np.uint32))
        params.setdefault("objective", "rank:ndcg")
        params.setdefault("eval_metric", "ndcg")
    else:
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")

    init_model = None
    if warm_start_model_path:
        booster = xgb.Booster()
        booster.load_model(warm_start_model_path)
        init_model = booster
    return xgb.train(params=params, dtrain=train_matrix, num_boost_round=num_boost_round, xgb_model=init_model)



def predict_xgboost_scores(booster, rows: np.ndarray, use_ranker: bool) -> np.ndarray:
    if rows.size == 0:
        return np.zeros(0, dtype=np.float32)
    matrix = xgb.DMatrix(np.asarray(rows, dtype=np.float32))
    scores = booster.predict(matrix)
    return np.asarray(scores, dtype=np.float32)



def evaluate_xgboost_scores(labels: np.ndarray, scores: np.ndarray, group_sizes: np.ndarray) -> dict:
    flat_user_ids: list[int] = []
    for group_idx, group_size in enumerate(group_sizes.tolist()):
        flat_user_ids.extend([int(group_idx)] * int(group_size))
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    user_ids = np.asarray(flat_user_ids, dtype=np.int32)
    metrics = ranking_metrics(labels, scores, user_ids)

    group_losses: list[float] = []
    offset = 0
    for group_size in group_sizes.tolist():
        group_scores = scores[offset : offset + group_size]
        positive_index = int(np.argmax(labels[offset : offset + group_size]))
        shifted = group_scores - np.max(group_scores)
        probs = np.exp(shifted)
        probs = probs / np.clip(probs.sum(), 1e-9, None)
        group_losses.append(float(-np.log(np.clip(probs[positive_index], 1e-9, None))))
        offset += group_size
    metrics["group_cross_entropy"] = float(np.mean(group_losses)) if group_losses else 0.0
    return metrics



def evaluate_final_candidates_xgboost(
    booster,
    ranking_samples: dict,
    item_features: dict[str, np.ndarray],
    item_popularity: np.ndarray,
    all_item_ids: np.ndarray,
    final_topk: int,
    use_ranker: bool,
) -> dict:
    artifacts = load_pickle(MULTI_RECALL_ARTIFACTS_PATH)
    ranking_test = ranking_samples["test"]
    user_index = {int(user_id): idx for idx, user_id in enumerate(np.asarray(ranking_test["user_id"]).tolist())}
    eval_user_ids, candidate_ids, candidate_scores, candidate_labels = [], [], [], []

    for artifact_idx, user_id in enumerate(np.asarray(artifacts["user_ids"]).tolist()):
        sample_idx = user_index.get(int(user_id))
        if sample_idx is None:
            continue
        candidates = artifacts["fused_candidates"][artifact_idx]
        if not candidates:
            continue

        sample = extract_split_sample(ranking_test, sample_idx)
        rows, valid_candidates = build_xgboost_inference_frame(
            sample,
            candidates,
            item_features=item_features,
            item_popularity=item_popularity,
        )
        if rows.size == 0 or not valid_candidates:
            continue

        scores = predict_xgboost_scores(booster, rows, use_ranker=use_ranker)
        target = int(ranking_test["target_movie_id"][sample_idx])
        ranked_pairs = sorted(zip(valid_candidates, scores.tolist()), key=lambda pair: pair[1], reverse=True)[:final_topk]
        ranked_candidates = [int(candidate) for candidate, _ in ranked_pairs]
        ranked_scores = [float(score) for _, score in ranked_pairs]
        ranked_labels = [1.0 if int(candidate) == target else 0.0 for candidate in ranked_candidates]
        eval_user_ids.append(int(user_id))
        candidate_ids.append(ranked_candidates)
        candidate_scores.append(ranked_scores)
        candidate_labels.append(ranked_labels)

    return evaluate_final_candidates(
        np.asarray(eval_user_ids, dtype=np.int32),
        candidate_ids,
        candidate_scores,
        candidate_labels,
        all_item_ids=all_item_ids,
        output_path=None,
    )
