import os
import sys

if __name__ == "__main__" or "src" not in sys.modules:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.core.ai_models import AVAILABLE_MODELS, recommend_model_for_video
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import (
    export_diagnostic_report,
    get_hardware_info,
    print_doctor_report,
)
from src.core.presets import ENHANCEMENT_PRESETS, RESOLUTION_TAGS, get_preset
from src.core.scene_renamer import SceneNamingConfig
from src.core.video_pipeline import probe_video
from src.engine.cli import run_cli as engine_run_cli


def run_cli(argv: list[str] | None = None) -> int:
    """Routes execution to the unified high-performance engine CLI with full backward compatibility."""
    return engine_run_cli(argv)


__all__ = [
    "AVAILABLE_MODELS",
    "ENHANCEMENT_PRESETS",
    "RESOLUTION_TAGS",
    "BatchManager",
    "EpisodeStatus",
    "QueueItem",
    "SceneNamingConfig",
    "export_diagnostic_report",
    "get_hardware_info",
    "get_preset",
    "print_doctor_report",
    "probe_video",
    "recommend_model_for_video",
    "run_cli",
]


if __name__ == "__main__":
    sys.exit(run_cli())
