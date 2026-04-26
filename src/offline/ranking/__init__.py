from .dataset import (
    SequenceRankingDataset,
    SequenceRankingEvalCollator,
    SequenceRankingTrainCollator,
    SequenceRankingValidationCollator,
    batch_to_device,
    build_inference_batch,
)
from .protocol import (
    CANDIDATE_FIELDS,
    CONTEXT_FIELDS,
    ITEM_FEATURE_FIELDS,
    POINTWISE_ITEM_FIELDS,
    POINTWISE_RANKING_FIELDS,
    POINTWISE_SEQUENCE_FIELDS,
    STATIC_USER_FIELDS,
    extract_split_sample,
    get_all_item_ids,
    get_item_feature_arrays,
    get_seen_movie_ids,
    sample_negative_ids,
)
from .xgboost_features import build_xgboost_inference_frame, build_xgboost_training_frame

try:
    from .wrapper import PointwiseCandidateRanker, PointwiseSequenceRanker
except ModuleNotFoundError:  # pragma: no cover - preprocessing can run without torch installed
    PointwiseCandidateRanker = None  # type: ignore[assignment]
    PointwiseSequenceRanker = None  # type: ignore[assignment]

__all__ = [
    "CANDIDATE_FIELDS",
    "CONTEXT_FIELDS",
    "ITEM_FEATURE_FIELDS",
    "POINTWISE_ITEM_FIELDS",
    "POINTWISE_RANKING_FIELDS",
    "POINTWISE_SEQUENCE_FIELDS",
    "STATIC_USER_FIELDS",
    "SequenceRankingDataset",
    "SequenceRankingEvalCollator",
    "SequenceRankingTrainCollator",
    "SequenceRankingValidationCollator",
    "PointwiseCandidateRanker",
    "PointwiseSequenceRanker",
    "batch_to_device",
    "build_inference_batch",
    "sample_negative_ids",
    "build_xgboost_inference_frame",
    "build_xgboost_training_frame",
    "extract_split_sample",
    "get_all_item_ids",
    "get_item_feature_arrays",
    "get_seen_movie_ids",
]
