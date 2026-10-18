"""8K batch planning regressions using synthetic queue entries only."""

from types import SimpleNamespace

import pytest

from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo


def _manager_with_queue():
    manager = BatchManager(HardwareInfo())
    manager.queue = [
        QueueItem(
            index=1,
            source_path="synthetic-pending.mp4",
            relative_source_path="synthetic-pending.mp4",
            target_scene_name="pending.mp4",
            target_full_path="output/pending.mp4",
            file_size_mb=100.0,
        ),
        QueueItem(
            index=2,
            source_path="synthetic-complete.mp4",
            relative_source_path="synthetic-complete.mp4",
            target_scene_name="complete.mp4",
            target_full_path="output/complete.mp4",
            file_size_mb=200.0,
            status=EpisodeStatus.COMPLETED,
        ),
    ]
    return manager


def test_8k_estimate_is_four_times_4k_and_only_counts_pending():
    manager = _manager_with_queue()

    four_k = manager.get_preflight_summary("", target_preset="4k", video_codec="libx265")
    eight_k = manager.get_preflight_summary("", target_preset="8k", video_codec="libx265")

    assert four_k["estimated_output_mb"] == 250.0
    assert eight_k["estimated_output_mb"] == 1000.0
    assert eight_k["estimated_output_mb"] == 4 * four_k["estimated_output_mb"]
    assert eight_k["total_source_size_mb"] == 300.0
    assert eight_k["pending_items"] == 1
    assert "Rough estimate" in eight_k["output_estimate_note"]
    assert "additional disk space" in eight_k["output_estimate_note"]


@pytest.mark.parametrize("preset", ["8k", "8K", " 8K "])
def test_8k_estimate_normalizes_preset(preset):
    summary = _manager_with_queue().get_preflight_summary("", target_preset=preset)

    assert summary["estimated_output_mb"] == 1000.0


def test_8k_estimate_detects_disk_shortage_missed_by_4k(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.core.batch.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=500 * 1024**2),
    )
    manager = _manager_with_queue()

    four_k = manager.get_preflight_summary(str(tmp_path), "4k", "libx265")
    eight_k = manager.get_preflight_summary(str(tmp_path), "8k", "libx265")

    assert not four_k["warnings"]
    assert any("Low disk space" in warning for warning in eight_k["warnings"])
    assert any("output alone" in warning for warning in eight_k["warnings"])


def test_completed_queue_does_not_reserve_8k_output_space():
    manager = _manager_with_queue()
    manager.queue[0].status = EpisodeStatus.COMPLETED

    summary = manager.get_preflight_summary("", "8k", "libx265")

    assert summary["estimated_output_mb"] == 0.0
    assert summary["pending_items"] == 0
