from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "funrec-movielens-1m"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

for path in (PROCESSED_DIR, MODELS_DIR, METRICS_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)


RETRIEVAL_SAMPLE_PATH = PROCESSED_DIR / "train_eval_sample_final.pkl"
RETRIEVAL_FEATURE_DICT_PATH = PROCESSED_DIR / "feature_dict.pkl"
RETRIEVAL_VOCAB_DICT_PATH = PROCESSED_DIR / "vocab_dict.pkl"
RETRIEVAL_PREPROCESS_META_PATH = PROCESSED_DIR / "retrieval_preprocess_meta.json"
ITEM_EMBEDDINGS_PATH = PROCESSED_DIR / "item_embeddings.npy"
SEQUENCE_ITEM_EMBEDDINGS_PATH = PROCESSED_DIR / "sequence_item_embeddings.npy"
MOVIE_IDS_PATH = PROCESSED_DIR / "movie_ids.npy"
MOVIE_RAW_IDS_PATH = PROCESSED_DIR / "movie_raw_ids.npy"
MULTI_RECALL_ARTIFACTS_PATH = PROCESSED_DIR / "multi_recall_artifacts.pkl"
ITEM_CATALOG_PATH = PROCESSED_DIR / "item_catalog.pkl"
ITEM2ITEM_ITEM_EMBEDDINGS_PATH = PROCESSED_DIR / "item2item_item_embeddings.npy"
ITEM2ITEM_MOVIE_IDS_PATH = PROCESSED_DIR / "item2item_movie_ids.npy"
ITEM_CF_MODEL_PATH = MODELS_DIR / "item_cf_model.pkl"
GENRE_MODEL_PATH = MODELS_DIR / "genre_model.pkl"
POPULAR_MODEL_PATH = MODELS_DIR / "popular_model.pkl"

RANKING_SAMPLE_PATH = PROCESSED_DIR / "ranking_train_eval_sample.pkl"
RANKING_FEATURE_DICT_PATH = PROCESSED_DIR / "ranking_feature_dict.pkl"
RANKING_VOCAB_DICT_PATH = PROCESSED_DIR / "ranking_vocab_dict.pkl"
RANKING_MODEL_CONFIG_PATH = PROCESSED_DIR / "ranking_model_config.pkl"

RETRIEVAL_MODEL_PATH = MODELS_DIR / "retrieval_model.pt"
SEQUENCE_MODEL_PATH = MODELS_DIR / "sequence_model.pt"
ITEM2ITEM_MODEL_PATH = MODELS_DIR / "item2item_model.pt"
RANKING_MODEL_PATH = MODELS_DIR / "ranking_model.pt"

RETRIEVAL_METRICS_PATH = METRICS_DIR / "retrieval_metrics.json"
MULTI_RECALL_METRICS_PATH = METRICS_DIR / "multi_recall_metrics.json"
RANKING_METRICS_PATH = METRICS_DIR / "ranking_metrics.json"
FINAL_METRICS_PATH = METRICS_DIR / "final_metrics.json"



def _normalize_model_name(model_name: str | None) -> str:
    return (model_name or "deepfm").strip().lower()



def get_ranking_model_path(model_name: str | None = None) -> Path:
    normalized = _normalize_model_name(model_name)
    if normalized == "deepfm":
        return RANKING_MODEL_PATH
    if normalized in {"xgboost", "xgboost_ranker"}:
        return MODELS_DIR / f"ranking_model_{normalized}.json"
    return MODELS_DIR / f"ranking_model_{normalized}.pt"



def get_ranking_model_config_path(model_name: str | None = None) -> Path:
    normalized = _normalize_model_name(model_name)
    if normalized == "deepfm":
        return RANKING_MODEL_CONFIG_PATH
    return PROCESSED_DIR / f"ranking_model_config_{normalized}.pkl"



def get_ranking_metrics_path(model_name: str | None = None) -> Path:
    normalized = _normalize_model_name(model_name)
    if normalized == "deepfm":
        return RANKING_METRICS_PATH
    return METRICS_DIR / f"ranking_metrics_{normalized}.json"



def get_final_metrics_path(model_name: str | None = None) -> Path:
    normalized = _normalize_model_name(model_name)
    if normalized == "deepfm":
        return FINAL_METRICS_PATH
    return METRICS_DIR / f"final_metrics_{normalized}.json"



def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)



def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)



def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))



def save_numpy(array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
