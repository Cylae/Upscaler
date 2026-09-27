"""Batch Manager facade.

Re-exports BatchManager, QueueItem, and EpisodeStatus from batch.py.
"""

from __future__ import annotations

from src.core.batch import (
    BatchManager,
    EpisodeStatus,
    QueueItem,
)
from src.core.pipeline import UpscaleJobConfig, VideoUpscalerPipeline

__all__ = [
    "BatchManager",
    "EpisodeStatus",
    "QueueItem",
    "UpscaleJobConfig",
    "VideoUpscalerPipeline",
]
