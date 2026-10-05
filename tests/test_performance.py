import os
import pytest
from src.core.video_pipeline import get_optimal_thread_spec, UpscaleJobConfig
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import HardwareInfo


def test_get_optimal_thread_spec():
    """Test thread specification resolution based on CPU core count."""
    assert get_optimal_thread_spec("4:4:4") == "4:4:4"
    assert get_optimal_thread_spec("2:4:2") == "2:4:2"
    assert get_optimal_thread_spec("1:2:2") == "1:2:2"

    auto_spec = get_optimal_thread_spec("auto")
    assert isinstance(auto_spec, str)
    parts = auto_spec.split(":")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_upscale_job_config_performance_defaults():
    """Test UpscaleJobConfig defaults for performance."""
    job = UpscaleJobConfig(input_file="test.mp4", output_dir="out")
    assert job.threads == "auto"
    assert job.frame_format == "jpg"
    assert job.video_codec == "hevc_nvenc"
    assert job.target_preset == "4k"


def test_batch_manager_stop_request():
    """Test BatchManager stop and cancellation mechanism."""
    hw = HardwareInfo(gpu_name="RTX 3070", gpu_vram_gb=8.0, cpu_cores=32)
    bm = BatchManager(hw)

    assert not bm.is_running
    bm.stop()
    assert bm._stop_requested


def test_format_eta():
    """Test format_eta helper for durations."""
    from src.core.video_pipeline import format_eta
    assert format_eta(0) == "0s"
    assert format_eta(45) == "45s"
    assert format_eta(60) == "1m 00s"
    assert format_eta(125) == "2m 05s"
    assert format_eta(3600) == "1h 00m"
    assert format_eta(3665) == "1h 01m"
    assert format_eta(-10) == "--"
    assert format_eta(None) == "--"
    assert format_eta(float("nan")) == "--"
    assert format_eta(float("inf")) == "--"
    assert format_eta(float("-inf")) == "--"
    assert format_eta(99999999999) == "--"


def test_queue_item_file_eta():
    """Test QueueItem file_eta attribute."""
    item = QueueItem(
        index=1,
        source_path="test.mp4",
        relative_source_path="test.mp4",
        target_scene_name="test.mp4",
        target_full_path="out/test.mp4",
        file_eta="1m 30s"
    )
    assert item.file_eta == "1m 30s"


def test_local_models_presence():
    """Test check_local_model_availability returns complete model availability dictionary."""
    from src.core.ai_models import check_local_model_availability
    status = check_local_model_availability()
    assert isinstance(status, dict)
    assert "auto" in status
    assert "realesr-animevideov3" in status
    assert "realesrgan-x4plus-anime" in status
    assert "realesrgan-x4plus" in status
    assert "realesrnet-x4plus" in status
    assert "realcugan-pro" in status
    assert "realcugan-se" in status
    assert status["auto"] is True
