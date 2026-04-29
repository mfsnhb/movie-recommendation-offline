from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from offline.utils.io import (
    OPENCLIP_ITEM_EMBEDDINGS_PATH,
    OPENCLIP_MOVIE_IDS_PATH,
    OPENCLIP_PREPROCESS_META_PATH,
    RAW_DATA_DIR,
    save_json,
    save_numpy,
)
from offline.utils.logging import get_logger


logger = get_logger("offline.features.multimodal")
_MULTIMODAL_SCHEMA_VERSION = 2
_MULTIMODAL_TEXT_FIELDS = ["title", "primaryTitle", "originalTitle", "description"]
_MULTIMODAL_FUSION_SEED = 20260428
_MULTIMODAL_FUSION_HIDDEN_DIM = 1536
_MULTIMODAL_FUSION_OUTPUT_DIM = 768


@dataclass(frozen=True)
class MultimodalSettings:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    batch_size: int = 64
    device: str = "auto"


@dataclass(frozen=True)
class MultimodalArtifacts:
    embeddings: np.ndarray
    movie_ids: np.ndarray
    metadata: dict


def resolve_multimodal_settings(settings: dict | None) -> MultimodalSettings:
    source = dict(settings or {})
    return MultimodalSettings(
        model_name=str(source.get("model_name", "ViT-L-14")),
        pretrained=str(source.get("pretrained", "openai")),
        batch_size=int(source.get("batch_size", 64)),
        device=str(source.get("device", "auto")),
    )


def load_multimodal_embeddings() -> MultimodalArtifacts | None:
    if not OPENCLIP_ITEM_EMBEDDINGS_PATH.exists() or not OPENCLIP_MOVIE_IDS_PATH.exists() or not OPENCLIP_PREPROCESS_META_PATH.exists():
        return None
    embeddings = np.load(OPENCLIP_ITEM_EMBEDDINGS_PATH).astype(np.float32, copy=False)
    movie_ids = np.load(OPENCLIP_MOVIE_IDS_PATH).astype(np.int32, copy=False)
    metadata = _load_json(OPENCLIP_PREPROCESS_META_PATH)
    return MultimodalArtifacts(embeddings=embeddings, movie_ids=movie_ids, metadata=metadata)


def build_or_load_multimodal_embeddings(
    df_movies,
    movie_vocab: dict,
    settings: dict | MultimodalSettings | None = None,
    force: bool = False,
) -> MultimodalArtifacts:
    resolved = settings if isinstance(settings, MultimodalSettings) else resolve_multimodal_settings(settings)
    expected_raw_movie_ids = np.asarray(movie_vocab["movie_id"], dtype=np.int32)
    cached = load_multimodal_embeddings()
    if not force and cached is not None and _cache_matches(cached, resolved, expected_raw_movie_ids):
        logger.info(
            "OpenCLIP multimodal cache hit | movies=%s | dim=%s | path=%s",
            cached.embeddings.shape[0] - 1,
            cached.embeddings.shape[1],
            OPENCLIP_ITEM_EMBEDDINGS_PATH.name,
        )
        return cached

    return build_multimodal_embeddings(df_movies, movie_vocab, resolved)


def build_multimodal_embeddings(df_movies, movie_vocab: dict, settings: MultimodalSettings) -> MultimodalArtifacts:
    device = _resolve_device(settings.device)
    model, _, preprocess = open_clip.create_model_and_transforms(settings.model_name, pretrained=settings.pretrained)
    tokenizer = open_clip.get_tokenizer(settings.model_name)
    model = model.to(device).eval()

    raw_movie_ids = np.asarray(movie_vocab["movie_id"], dtype=np.int32)
    movie_lookup = df_movies.drop_duplicates(subset=["movie_id"]).assign(movie_id=lambda df: df["movie_id"].astype(np.int32)).set_index("movie_id", drop=False)
    batch_size = max(int(settings.batch_size), 1)
    zero_image = _zero_image_template(raw_movie_ids, preprocess)
    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    missing_posters = 0
    missing_text = 0

    logger.info(
        "OpenCLIP multimodal extraction start | model=%s | pretrained=%s | movies=%s | batch_size=%s | device=%s",
        settings.model_name,
        settings.pretrained,
        raw_movie_ids.size,
        batch_size,
        device,
    )

    with torch.no_grad():
        for start in range(0, raw_movie_ids.size, batch_size):
            batch_ids = raw_movie_ids[start : start + batch_size]
            images = []
            texts = []
            for raw_movie_id in batch_ids.tolist():
                row = movie_lookup.loc[int(raw_movie_id)]
                image, image_missing = _load_poster(int(raw_movie_id), preprocess, zero_image)
                text, text_missing = _movie_text(row)
                images.append(image)
                texts.append(text)
                missing_posters += int(image_missing)
                missing_text += int(text_missing)

            image_tensor = torch.stack(images).to(device, non_blocking=True)
            text_tensor = tokenizer(texts).to(device, non_blocking=True)
            image_embedding = torch.nn.functional.normalize(model.encode_image(image_tensor).float(), dim=1)
            text_embedding = torch.nn.functional.normalize(model.encode_text(text_tensor).float(), dim=1)
            image_embeddings.append(image_embedding.cpu().numpy().astype(np.float32, copy=False))
            text_embeddings.append(text_embedding.cpu().numpy().astype(np.float32, copy=False))

            processed = min(start + batch_size, raw_movie_ids.size)
            if processed == raw_movie_ids.size or processed % max(batch_size * 10, 1) == 0:
                logger.info("OpenCLIP multimodal extraction progress | movies=%s/%s", processed, raw_movie_ids.size)

    image_matrix = np.concatenate(image_embeddings, axis=0) if image_embeddings else np.zeros((0, 0), dtype=np.float32)
    text_matrix = np.concatenate(text_embeddings, axis=0) if text_embeddings else np.zeros((0, 0), dtype=np.float32)
    fused = _fuse_clip_embeddings(image_matrix, text_matrix)
    aligned = np.zeros((raw_movie_ids.size + 1, fused.shape[1]), dtype=np.float32)
    aligned[1:] = fused
    movie_ids = raw_movie_ids.astype(np.int32, copy=True)
    metadata = {
        "schema_version": _MULTIMODAL_SCHEMA_VERSION,
        "model_name": settings.model_name,
        "pretrained": settings.pretrained,
        "embedding_dim": int(aligned.shape[1]),
        "image_embedding_dim": int(image_matrix.shape[1]) if image_matrix.ndim == 2 else 0,
        "text_embedding_dim": int(text_matrix.shape[1]) if text_matrix.ndim == 2 else 0,
        "fusion_hidden_dim": _MULTIMODAL_FUSION_HIDDEN_DIM,
        "fusion_output_dim": _MULTIMODAL_FUSION_OUTPUT_DIM,
        "fusion_seed": _MULTIMODAL_FUSION_SEED,
        "text_fields": list(_MULTIMODAL_TEXT_FIELDS),
        "movie_count": int(raw_movie_ids.size),
        "missing_posters": int(missing_posters),
        "missing_text": int(missing_text),
        "raw_data_dir": str(RAW_DATA_DIR),
    }
    save_numpy(aligned, OPENCLIP_ITEM_EMBEDDINGS_PATH)
    save_numpy(movie_ids, OPENCLIP_MOVIE_IDS_PATH)
    save_json(metadata, OPENCLIP_PREPROCESS_META_PATH)
    logger.info(
        "OpenCLIP multimodal extraction done | movies=%s | dim=%s | missing_posters=%s | missing_text=%s",
        raw_movie_ids.size,
        aligned.shape[1],
        missing_posters,
        missing_text,
    )
    return MultimodalArtifacts(embeddings=aligned, movie_ids=movie_ids, metadata=metadata)


def _cache_matches(artifacts: MultimodalArtifacts, settings: MultimodalSettings, expected_raw_movie_ids: np.ndarray) -> bool:
    metadata = artifacts.metadata
    expected_count = int(expected_raw_movie_ids.shape[0])
    return (
        artifacts.embeddings.ndim == 2
        and artifacts.embeddings.shape == (expected_count + 1, _MULTIMODAL_FUSION_OUTPUT_DIM)
        and np.array_equal(artifacts.movie_ids.astype(np.int32, copy=False), expected_raw_movie_ids.astype(np.int32, copy=False))
        and int(metadata.get("schema_version", -1)) == _MULTIMODAL_SCHEMA_VERSION
        and str(metadata.get("model_name")) == settings.model_name
        and str(metadata.get("pretrained")) == settings.pretrained
        and int(metadata.get("movie_count", -1)) == expected_count
        and list(metadata.get("text_fields", [])) == _MULTIMODAL_TEXT_FIELDS
    )


def _fuse_clip_embeddings(image_matrix: np.ndarray, text_matrix: np.ndarray) -> np.ndarray:
    if image_matrix.size == 0 or text_matrix.size == 0:
        return np.zeros((image_matrix.shape[0], _MULTIMODAL_FUSION_OUTPUT_DIM), dtype=np.float32)
    fused_input = np.concatenate([image_matrix, text_matrix], axis=1).astype(np.float32, copy=False)
    rng = np.random.default_rng(_MULTIMODAL_FUSION_SEED)
    input_dim = int(fused_input.shape[1])
    hidden_weight = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, _MULTIMODAL_FUSION_HIDDEN_DIM)).astype(np.float32)
    output_weight = rng.normal(0.0, 1.0 / np.sqrt(_MULTIMODAL_FUSION_HIDDEN_DIM), size=(_MULTIMODAL_FUSION_HIDDEN_DIM, _MULTIMODAL_FUSION_OUTPUT_DIM)).astype(np.float32)
    hidden = _gelu(fused_input @ hidden_weight)
    output = hidden @ output_weight
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return (output / np.clip(norms, 1e-9, None)).astype(np.float32, copy=False)


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * np.power(values, 3))))


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _zero_image_template(raw_movie_ids: np.ndarray, preprocess):
    for raw_movie_id in raw_movie_ids.tolist():
        image_path = RAW_DATA_DIR / "image" / f"{int(raw_movie_id)}.png"
        if image_path.exists():
            with Image.open(image_path) as image:
                return torch.zeros_like(preprocess(image.convert("RGB")))
    return torch.zeros((3, 224, 224), dtype=torch.float32)


def _load_poster(raw_movie_id: int, preprocess, zero_image):
    image_path = RAW_DATA_DIR / "image" / f"{raw_movie_id}.png"
    if not image_path.exists():
        return zero_image.clone(), True
    with Image.open(image_path) as image:
        return preprocess(image.convert("RGB")), False


def _movie_text(row) -> tuple[str, bool]:
    parts = [_clean_text(row.get(field, "")) for field in _MULTIMODAL_TEXT_FIELDS]
    parts = [part for part in parts if part]
    if not parts:
        return "unknown movie", True
    return " | ".join(parts), False


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text == "\\N" or text.lower() == "nan":
        return ""
    return text
