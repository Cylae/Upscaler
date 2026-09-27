"""Pipeline facade.

Re-exports VideoUpscalerPipeline and UpscaleJobConfig from video_pipeline.
"""

from __future__ import annotations

from typing import Optional

from src.core.video_pipeline import (
    UpscaleJobConfig,
    VideoUpscalerPipeline,
)


def generate_frame_comparison(
    video_path: str,
    model_id: str = "auto",
    target_preset: str = "4k",
    hardware: Optional[object] = None,
):
    """Convenience helper to extract and upscale a single frame for Before/After comparison."""
    pipeline = VideoUpscalerPipeline(hardware)
    return pipeline.extract_and_upscale_single_frame(
        video_path,
        model_id=model_id,
        target_preset=target_preset,
    )


__all__ = [
    "UpscaleJobConfig",
    "VideoUpscalerPipeline",
    "generate_frame_comparison",
]
