from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline.utils.logging import format_eta, get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
STEP_FUNCS = {
    "retrieval_preprocess": ("offline.features.retrieval", "run_retrieval_preprocessing"),
    "ranking_preprocess": ("offline.ranking.preprocess", "run_ranking_preprocessing"),
    "retrieval_train": ("offline.training.retrieval", "run_retrieval_training"),
    "retrieval_evaluate": ("offline.training.retrieval", "evaluate_retrieval"),
    "multi_recall_build": ("offline.evaluate.multi_recall", "run_multi_recall_build"),
    "ranking_train": ("offline.training.ranking", "run_ranking_training"),
    "ranking_evaluate": ("offline.training.ranking", "evaluate_ranking_model"),
    "final_evaluate": ("offline.training.ranking", "evaluate_final_ranking"),
}



def _load_step_func(step: str):
    module_name, func_name = STEP_FUNCS[step]
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)



def _load_default_ranking_model() -> str:
    ranking_config = yaml.safe_load((CONFIG_DIR / "ranking.yaml").read_text(encoding="utf-8")) or {}
    if "models" in ranking_config:
        return str(ranking_config.get("default_model") or "deepfm").strip().lower()
    return str((ranking_config.get("model") or {}).get("name", "deepfm")).strip().lower()



def _run_pipeline_step(step: str):
    step_func = _load_step_func(step)
    if step == "retrieval_train":
        return step_func(build_multi_recall=False, evaluate=False)
    if step == "retrieval_evaluate":
        return step_func(route="two_tower")
    if step == "ranking_train":
        return step_func(evaluate=False, final_evaluate=False)
    if step in {"ranking_evaluate", "final_evaluate"}:
        return step_func(_load_default_ranking_model())
    return step_func()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="all")
    args = parser.parse_args()
    logger = get_logger("offline.pipeline")
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    if "all" in steps:
        steps = [
            "ranking_preprocess",
            "retrieval_preprocess",
            "retrieval_train",
            "retrieval_evaluate",
            "multi_recall_build",
            "ranking_train",
            "ranking_evaluate",
            "final_evaluate",
        ]

    total_steps = len(steps)
    pipeline_start = time.perf_counter()
    finished_durations: list[float] = []
    logger.info("Pipeline start | total_steps=%s | steps=%s", total_steps, ", ".join(steps))

    for idx, step in enumerate(steps, start=1):
        avg_step_time = sum(finished_durations) / len(finished_durations) if finished_durations else None
        eta_before = avg_step_time * (total_steps - idx + 1) if avg_step_time is not None else None
        logger.info(
            "Step %s/%s start | name=%s | elapsed=%s | eta=%s",
            idx,
            total_steps,
            step,
            format_eta(time.perf_counter() - pipeline_start),
            format_eta(eta_before),
        )
        step_start = time.perf_counter()
        _run_pipeline_step(step)
        step_duration = time.perf_counter() - step_start
        finished_durations.append(step_duration)
        avg_step_time = sum(finished_durations) / len(finished_durations)
        eta_after = avg_step_time * (total_steps - idx)
        logger.info(
            "Step %s/%s done | name=%s | step_time=%s | elapsed=%s | eta=%s",
            idx,
            total_steps,
            step,
            format_eta(step_duration),
            format_eta(time.perf_counter() - pipeline_start),
            format_eta(eta_after),
        )

    logger.info("Pipeline done | total_time=%s", format_eta(time.perf_counter() - pipeline_start))
