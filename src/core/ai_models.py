"""AI Models facade.

Re-exports model definitions, recommendations, and geometry functions from models.py.
"""

from __future__ import annotations

from src.core.models import (
    AIModelSpec,
    AVAILABLE_MODELS,
    ModelEngine,
    ModelRecommendation,
    auto_tile_size,
    calculate_target_dimensions,
    check_local_model_availability,
    choose_best_model_for_video,
    get_model_display_name,
    get_optimal_thread_spec,
    is_heavy_model,
    recommend_model_for_video,
    resolve_realcugan_options,
)

__all__ = [
    "AIModelSpec",
    "AVAILABLE_MODELS",
    "ModelEngine",
    "ModelRecommendation",
    "auto_tile_size",
    "calculate_target_dimensions",
    "check_local_model_availability",
    "choose_best_model_for_video",
    "get_model_display_name",
    "get_optimal_thread_spec",
    "is_heavy_model",
    "recommend_model_for_video",
    "resolve_realcugan_options",
]
