import os
import sys
import tempfile
import time
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

from src.core.video_pipeline import (
    is_file_complete_and_valid,
    cleanup_stale_temp_directories,
    register_active_job_dir,
    unregister_active_job_dir,
    safe_atomic_move,
    is_process_alive,
    VideoMetadata,
)
from src.core.batch_manager import (
    BatchManager,
    QueueItem,
    EpisodeStatus,
)
from src.core.scene_renamer import SceneNamingConfig
from src.core.detector import (
    generate_diagnostic_report,
    export_diagnostic_report,
)


def test_queue_item_stable_id():
    """Verify QueueItem has a unique stable immutable id."""
    item1 = QueueItem(index=1, source_path="test1.mp4", relative_source_path="test1.mp4", target_scene_name="out1.mp4", target_full_path="C:/out/out1.mp4", file_size_mb=10.0)
    item2 = QueueItem(index=2, source_path="test2.mp4", relative_source_path="test2.mp4", target_scene_name="out2.mp4", target_full_path="C:/out/out2.mp4", file_size_mb=10.0)
    assert isinstance(item1.id, str)
    assert len(item1.id) >= 16
    assert item1.id != item2.id


def test_is_file_complete_and_valid_dimension_and_duration_checks():
    """Verify is_file_complete_and_valid strictly checks resolution and duration tolerance."""
    dummy_path = "C:/fake/video.mp4"

    # Mock os.path.exists and os.path.getsize
    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=50 * 1024 * 1024), \
         patch("src.core.video_pipeline.probe_video") as mock_probe:
        
        # Valid output matching expected dimensions and duration
        mock_probe.return_value = VideoMetadata(
            filepath=dummy_path,
            width=3840,
            height=2160,
            duration=120.0,
            fps=24.0,
            nb_frames=2880,
            codec_name="hevc",
            has_audio=True,
            audio_codec="aac",
            has_subtitles=False
        )

        # 1. Matching expectation -> True
        assert is_file_complete_and_valid(
            dummy_path,
            expected_width=3840,
            expected_height=2160,
            expected_duration=120.0,
            duration_tolerance_s=1.5
        ) is True

        # 2. Wrong width -> False
        assert is_file_complete_and_valid(
            dummy_path,
            expected_width=1920,
            expected_height=2160
        ) is False

        # 3. Truncated duration beyond tolerance -> False
        assert is_file_complete_and_valid(
            dummy_path,
            expected_duration=180.0,
            duration_tolerance_s=2.0
        ) is False

        # 4. Missing expected audio -> False
        mock_probe.return_value.has_audio = False
        assert is_file_complete_and_valid(
            dummy_path,
            expect_audio=True
        ) is False


def test_active_job_temp_cleanup_protection():
    """Verify active jobs are protected by in-memory registry and PID lease locks during cleanup."""
    with tempfile.TemporaryDirectory() as tmp_root:
        temp_root = os.path.join(tmp_root, ".temp_upscaler")
        os.makedirs(temp_root, exist_ok=True)

        # Create an active job dir and an old abandoned dir
        active_job_dir = os.path.join(temp_root, "job_active_123")
        os.makedirs(active_job_dir, exist_ok=True)
        lock_file = os.path.join(active_job_dir, ".job_lock")
        with open(lock_file, "w") as f:
            json.dump({"pid": os.getpid()}, f)

        stale_job_dir = os.path.join(temp_root, "job_stale_999")
        os.makedirs(stale_job_dir, exist_ok=True)
        # Stale job with a non-existent dead PID
        stale_lock = os.path.join(stale_job_dir, ".job_lock")
        with open(stale_lock, "w") as f:
            json.dump({"pid": 99999999}, f)
        # Set mtime to 1 hour ago so it qualifies as stale
        past_time = time.time() - 3600
        os.utime(stale_job_dir, (past_time, past_time))

        # Register active job in memory as well
        register_active_job_dir(active_job_dir)
        try:
            # Force max_age_seconds=0 to test cleanup logic
            deleted_count = cleanup_stale_temp_directories(root_dir=tmp_root, max_age_seconds=0)
            
            # Active job directory must still exist!
            assert os.path.exists(active_job_dir)
            
            # Dead job directory was cleaned up
            assert not os.path.exists(stale_job_dir)
            assert deleted_count >= 1
        finally:
            unregister_active_job_dir(active_job_dir)


def test_safe_atomic_move():
    """Verify safe_atomic_move moves files correctly and cleans up source."""
    with tempfile.TemporaryDirectory() as tmp_root:
        src = os.path.join(tmp_root, "source.mp4")
        dst = os.path.join(tmp_root, "subdir", "dest.mp4")
        with open(src, "w") as f:
            f.write("upscaled video content")

        safe_atomic_move(src, dst)
        assert not os.path.exists(src)
        assert os.path.exists(dst)
        with open(dst, "r") as f:
            assert f.read() == "upscaled video content"


def test_batch_manager_retry_failed_items():
    """Verify retry_failed_items resets only FAILED and CANCELLED items to PENDING."""
    bm = BatchManager()
    item1 = QueueItem(index=1, source_path="1.mp4", relative_source_path="1.mp4", target_scene_name="1.mp4", target_full_path="1.mp4", file_size_mb=1.0, status=EpisodeStatus.COMPLETED)
    item2 = QueueItem(index=2, source_path="2.mp4", relative_source_path="2.mp4", target_scene_name="2.mp4", target_full_path="2.mp4", file_size_mb=1.0, status=EpisodeStatus.FAILED)
    item3 = QueueItem(index=3, source_path="3.mp4", relative_source_path="3.mp4", target_scene_name="3.mp4", target_full_path="3.mp4", file_size_mb=1.0, status=EpisodeStatus.CANCELLED)
    item4 = QueueItem(index=4, source_path="4.mp4", relative_source_path="4.mp4", target_scene_name="4.mp4", target_full_path="4.mp4", file_size_mb=1.0, status=EpisodeStatus.SKIPPED)
    bm.queue = [item1, item2, item3, item4]

    retried = bm.retry_failed_items()
    assert retried == 2
    assert item1.status == EpisodeStatus.COMPLETED
    assert item2.status == EpisodeStatus.PENDING
    assert item3.status == EpisodeStatus.PENDING
    assert item4.status == EpisodeStatus.SKIPPED


def test_batch_manager_manifest_save_and_load():
    """Verify queue state can be saved to and loaded from a JSON manifest."""
    bm = BatchManager()
    item1 = QueueItem(index=1, source_path="s01e01.mp4", relative_source_path="s01e01.mp4", target_scene_name="out1.mp4", target_full_path="C:/out/out1.mp4", file_size_mb=12.5, status=EpisodeStatus.COMPLETED)
    item2 = QueueItem(index=2, source_path="s01e02.mp4", relative_source_path="s01e02.mp4", target_scene_name="out2.mp4", target_full_path="C:/out/out2.mp4", file_size_mb=14.2, status=EpisodeStatus.FAILED, status_detail="Disk full")
    bm.queue = [item1, item2]

    with tempfile.TemporaryDirectory() as tmp_root:
        manifest_path = os.path.join(tmp_root, "upscale_manifest.json")
        saved = bm.save_manifest(manifest_path)
        assert saved is True
        assert os.path.exists(manifest_path)

        new_bm = BatchManager()
        loaded = new_bm.load_manifest(manifest_path)
        assert loaded == 2
        assert len(new_bm.queue) == 2
        assert new_bm.queue[0].source_path == "s01e01.mp4"
        assert new_bm.queue[0].status == EpisodeStatus.COMPLETED
        assert new_bm.queue[1].source_path == "s01e02.mp4"
        assert new_bm.queue[1].status == EpisodeStatus.FAILED
        assert new_bm.queue[1].status_detail == "Disk full"


def test_batch_manager_plan_dry_run():
    """Verify plan_dry_run reports output mapping and detects collisions."""
    bm = BatchManager()
    item1 = QueueItem(index=1, source_path="C:/src/ep1.mp4", relative_source_path="ep1.mp4", target_scene_name="Show.S01E01.mp4", target_full_path="C:/out/Show.S01E01.mp4", file_size_mb=50.0)
    item2 = QueueItem(index=2, source_path="C:/src/dup/ep1.mp4", relative_source_path="dup/ep1.mp4", target_scene_name="Show.S01E01.mp4", target_full_path="C:/out/Show.S01E01.mp4", file_size_mb=50.0)
    bm.queue = [item1, item2]

    plan = bm.plan_dry_run("C:/out", preserve_folders=False)
    assert len(plan) == 2
    assert plan[0]["collision"] is True
    assert plan[1]["collision"] is True


def test_diagnostic_report_privacy_redaction():
    """Verify generate_diagnostic_report redacts user home directory when redact_paths is True."""
    report = generate_diagnostic_report(redact_paths=True)
    assert "SYSTEM ENVIRONMENT DIAGNOSTIC" in report
    assert "NVIDIA GPU Detected" in report
    user_home = os.path.expanduser("~")
    if len(user_home) > 3:
        assert user_home not in report
        assert "<USER_HOME>" in report
