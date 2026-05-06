from __future__ import annotations

from pathlib import Path

import yaml


RETRIEVAL_ROUTE_NAMES = ("two_tower", "sequence", "item_cf", "multimodal", "genre", "popular")
RANKING_MODEL_NAMES = ("deepfm", "din")


def load_yaml_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}



def _merge_dict(base: dict | None, override: dict | None) -> dict:
    merged = dict(base or {})
    merged.update(override or {})
    return merged



def resolve_rating_semantics(settings: dict | None) -> dict[str, float | bool]:
    source = dict(settings or {})
    return {
        "positive_rating_min": float(source.get("positive_rating_min", 4.0)),
        "negative_rating_max": float(source.get("negative_rating_max", 2.0)),
        "neutral_rating": float(source.get("neutral_rating", 3.0)),
    }



def _normalize_name_list(requested_names, default_names: list[str]) -> list[str]:
    if requested_names is None:
        names = list(default_names)
    elif isinstance(requested_names, str):
        names = [name.strip().lower() for name in requested_names.split(",") if name.strip()]
    else:
        names = [str(name).strip().lower() for name in requested_names if str(name).strip()]

    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped



def resolve_ranking_model_names(settings: dict, requested_models=None) -> list[str]:
    if "models" in settings:
        available_models = [str(name).strip().lower() for name in settings.get("models", {}).keys()]
        default_model = (settings.get("default_model") or (available_models[0] if available_models else "deepfm")).strip().lower()
    else:
        available_models = list(RANKING_MODEL_NAMES)
        default_model = str(settings.get("model", {}).get("name", "deepfm")).strip().lower()

    requested = _normalize_name_list(requested_models, [default_model])
    unknown = sorted(set(requested) - set(available_models))
    if unknown:
        raise ValueError(f"Unknown ranking models {unknown}. Available models: {sorted(available_models)}")
    return requested



def resolve_retrieval_route_names(settings: dict, requested_routes=None) -> list[str]:
    if "models" in settings:
        models = settings.get("models", {}) or {}
        common_settings = settings.get("common", {}) or {}
        configured_routes = list((common_settings.get("multi_recall", {}) or {}).get("routes", []) or [])
        enabled_learned_routes: list[str] = []
        for route_name in ("two_tower", "sequence"):
            enabled_learned_routes.append(route_name)
        default_routes: list[str] = []
        for route_name in enabled_learned_routes + configured_routes:
            normalized = str(route_name).strip().lower()
            if normalized in RETRIEVAL_ROUTE_NAMES and normalized not in default_routes:
                default_routes.append(normalized)
    else:
        configured_routes = list((settings.get("multi_recall", {}) or {}).get("routes", []) or [])
        default_routes = ["two_tower", "sequence"]
        for route_name in configured_routes:
            normalized = str(route_name).strip().lower()
            if normalized in RETRIEVAL_ROUTE_NAMES and normalized not in default_routes:
                default_routes.append(normalized)

    if not default_routes:
        default_routes = ["two_tower", "sequence"]

    requested = _normalize_name_list(requested_routes, default_routes)
    unknown = sorted(set(requested) - set(RETRIEVAL_ROUTE_NAMES))
    if unknown:
        raise ValueError(f"Unknown retrieval routes {unknown}. Available routes: {list(RETRIEVAL_ROUTE_NAMES)}")
    return requested



def resolve_multi_recall_route_names(settings: dict, requested_routes=None) -> list[str]:
    if "models" in settings:
        common_settings = settings.get("common", {}) or {}
        configured_routes = list((common_settings.get("multi_recall", {}) or {}).get("routes", []) or [])
    else:
        configured_routes = list((settings.get("multi_recall", {}) or {}).get("routes", []) or [])

    default_routes = []
    for route_name in configured_routes:
        normalized = str(route_name).strip().lower()
        if normalized in RETRIEVAL_ROUTE_NAMES and normalized not in default_routes:
            default_routes.append(normalized)
    if not default_routes:
        default_routes = resolve_retrieval_route_names(settings)

    requested = _normalize_name_list(requested_routes, default_routes)
    unknown = sorted(set(requested) - set(RETRIEVAL_ROUTE_NAMES))
    if unknown:
        raise ValueError(f"Unknown multi-recall routes {unknown}. Available routes: {list(RETRIEVAL_ROUTE_NAMES)}")
    return requested



def resolve_ranking_config(settings: dict, requested_model_name: str | None = None) -> dict:
    if "models" in settings:
        models = settings.get("models", {})
        default_model = (requested_model_name or settings.get("default_model") or "deepfm").strip().lower()
        if default_model not in models:
            raise ValueError(f"Unknown ranking model '{default_model}'. Available models: {sorted(models)}")

        common_settings = settings.get("common", {})
        common_training = common_settings.get("training", {})
        common_evaluation = common_settings.get("evaluation", {})
        model_entry = models[default_model] or {}
        return {
            "model_name": default_model,
            "framework": str(model_entry.get("framework", "torch")).strip().lower(),
            "model_settings": dict(model_entry.get("architecture", {})),
            "training_settings": _merge_dict(common_training, model_entry.get("training", {})),
            "evaluation_settings": _merge_dict(common_evaluation, model_entry.get("evaluation", {})),
        }

    legacy_model_settings = settings.get("model", {})
    legacy_model_name = (requested_model_name or legacy_model_settings.get("name", "deepfm")).strip().lower()
    return {
        "model_name": legacy_model_name,
        "framework": "torch",
        "model_settings": dict(legacy_model_settings),
        "training_settings": dict(settings.get("training", {})),
        "evaluation_settings": dict(settings.get("evaluation", {})),
    }



def resolve_retrieval_config(settings: dict) -> dict:
    if "models" in settings:
        models = settings.get("models", {})
        common_settings = settings.get("common", {})
        common_training = dict(common_settings.get("training", {}))
        common_evaluation = dict(common_settings.get("evaluation", {}))
        multi_recall_settings = _merge_dict(common_settings.get("multi_recall", {}), settings.get("multi_recall", {}))

        two_tower_entry = models.get("two_tower", {}) or {}
        sequence_entry = models.get("sequence", {}) or {}
        rating_semantics = resolve_rating_semantics(common_training)
        two_tower_training = _merge_dict(_merge_dict(common_training, two_tower_entry.get("training", {})), rating_semantics)
        sequence_training = _merge_dict(_merge_dict(common_training, sequence_entry.get("training", {})), rating_semantics)

        two_tower_settings = dict(two_tower_entry.get("architecture", {}))
        sequence_settings = dict(sequence_entry.get("architecture", {}))
        tt_emb_dim = int(two_tower_settings.get("embedding_dim", 32))
        seq_emb_dim = int(sequence_settings.get("embedding_dim", tt_emb_dim))
        if tt_emb_dim != seq_emb_dim:
            raise ValueError(f"two_tower.embedding_dim ({tt_emb_dim}) must match sequence.embedding_dim ({seq_emb_dim})")

        return {
            "two_tower_settings": two_tower_settings,
            "sequence_settings": sequence_settings,
            "training_settings": two_tower_training,
            "two_tower_training_settings": two_tower_training,
            "sequence_training_settings": sequence_training,
            "evaluation_settings": common_evaluation,
            "multi_recall_settings": multi_recall_settings,
            "rating_semantics": rating_semantics,
            "embedding_dim": tt_emb_dim,
        }

    legacy_model_settings = dict(settings.get("model", {}))
    legacy_training = dict(settings.get("training", {}))
    rating_semantics = resolve_rating_semantics(legacy_training)
    merged_legacy_training = _merge_dict(legacy_training, rating_semantics)
    return {
        "two_tower_settings": {
            "embedding_dim": legacy_model_settings.get("embedding_dim", 32),
            "user_hidden_dims": legacy_model_settings.get("user_hidden_dims"),
            "item_hidden_dims": legacy_model_settings.get("item_hidden_dims"),
            "dropout": legacy_model_settings.get("dropout", 0.1),
        },
        "sequence_settings": {
            "embedding_dim": legacy_model_settings.get("embedding_dim", 32),
            "hidden_dim": legacy_model_settings.get("sequence_hidden_dim", legacy_model_settings.get("embedding_dim", 32)),
            "num_layers": legacy_model_settings.get("sequence_num_layers", 1),
            "max_len": legacy_model_settings.get("sequence_max_len", 10),
            "dropout": legacy_model_settings.get("sequence_dropout", 0.1),
        },
        "training_settings": merged_legacy_training,
        "two_tower_training_settings": merged_legacy_training,
        "sequence_training_settings": merged_legacy_training,
        "evaluation_settings": dict(settings.get("evaluation", {})),
        "multi_recall_settings": dict(settings.get("multi_recall", {})),
        "rating_semantics": rating_semantics,
        "embedding_dim": int(legacy_model_settings.get("embedding_dim", 32)),
    }
