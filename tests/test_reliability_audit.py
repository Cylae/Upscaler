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


def test_probe_video_fps_parsing_variations():
    """Verify probe_video handles integer, decimal, rational, and fallback FPS strings."""
    from src.core.probe import probe_video

    def make_ffprobe_json(avg_fps, r_fps="24/1"):
        return json.dumps({
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": avg_fps,
                    "r_frame_rate": r_fps,
                    "duration": "10.0",
                    "nb_frames": "240",
                    "codec_name": "h264"
                }
            ],
            "format": {"duration": "10.0"}
        })

    with patch("subprocess.check_output") as mock_subproc:
        # 1. Plain integer FPS string e.g. "25"
        mock_subproc.return_value = make_ffprobe_json("25", "25")
        meta = probe_video("dummy.mp4")
        assert meta.fps == 25.0

        # 2. Decimal FPS string e.g. "29.97"
        mock_subproc.return_value = make_ffprobe_json("29.97", "29.97")
        meta = probe_video("dummy.mp4")
        assert meta.fps == 29.97

        # 3. Rational FPS string e.g. "30000/1001"
        mock_subproc.return_value = make_ffprobe_json("30000/1001", "30000/1001")
        meta = probe_video("dummy.mp4")
        assert abs(meta.fps - (30000 / 1001)) < 0.0001

        # 4. Invalid/zero FPS string falls back to 24.0
        mock_subproc.return_value = make_ffprobe_json("0/0", "0")
        meta = probe_video("dummy.mp4")
        assert meta.fps == 24.0


def test_resolve_realcugan_options_with_subfolder_and_fallbacks():
    """Verify RealCUGAN option resolution handles direct subfolders, scale fallback, and denoise filtering."""
    from src.core.models import resolve_realcugan_options

    with tempfile.TemporaryDirectory() as tmp_root:
        pro_dir = os.path.join(tmp_root, "models-pro")
        se_dir = os.path.join(tmp_root, "models-se")
        os.makedirs(pro_dir, exist_ok=True)
        os.makedirs(se_dir, exist_ok=True)

        # 1. Direct models-pro dir, scale 2 -> accepts models-pro with denoise 2
        scale, denoise, model_name = resolve_realcugan_options(pro_dir, requested_scale=2, requested_denoise=2)
        assert scale == 2
        assert denoise == 2
        assert model_name == "models-pro"

        # 2. Direct models-pro dir, scale 3 -> falls back to models-se (Pro only supports 2x)
        scale, denoise, model_name = resolve_realcugan_options(pro_dir, requested_scale=3, requested_denoise=0)
        assert scale == 3
        assert denoise == 0
        assert model_name == "models-se"

        # 3. Direct models-se dir, scale 2, denoise 1 -> remains models-se (SE supports denoise 1)
        scale, denoise, model_name = resolve_realcugan_options(se_dir, requested_scale=2, requested_denoise=1)
        assert scale == 2
        assert denoise == 1
        assert model_name == "models-se"

        # 4. Invalid denoise for models-pro (e.g. 1) falls back to -1
        scale, denoise, model_name = resolve_realcugan_options(pro_dir, requested_scale=2, requested_denoise=1)
        assert scale == 2
        assert denoise == -1
        assert model_name == "models-pro"


def test_batch_manager_scan_folder_rejects_non_video_files():
    """Verify scan_folder ignores non-video files even when passed directly as source."""
    with tempfile.TemporaryDirectory() as tmp_root:
        txt_file = os.path.join(tmp_root, "readme.txt")
        with open(txt_file, "w") as f:
            f.write("not a video")

        bm = BatchManager()
        items = bm.scan_folder(txt_file, tmp_root)
        assert len(items) == 0

        vid_file = os.path.join(tmp_root, "episode.mp4")
        with open(vid_file, "wb") as f:
            f.write(b"\x00" * 1024)

        items = bm.scan_folder(vid_file, tmp_root)
        assert len(items) == 1
        assert items[0].source_path == vid_file


def test_batch_manager_manifest_corrupt_file_handling():
    """Verify load_manifest safely handles corrupt or empty JSON files without crashing."""
    bm = BatchManager()
    with tempfile.TemporaryDirectory() as tmp_root:
        # Non-existent
        assert bm.load_manifest(os.path.join(tmp_root, "missing.json")) == 0

        # Corrupt JSON
        corrupt_file = os.path.join(tmp_root, "corrupt.json")
        with open(corrupt_file, "w") as f:
            f.write("{{{ invalid json")
        assert bm.load_manifest(corrupt_file) == 0

        # Empty file
        empty_file = os.path.join(tmp_root, "empty.json")
        with open(empty_file, "w") as f:
            pass
        assert bm.load_manifest(empty_file) == 0

        # JSON array instead of dict
        array_file = os.path.join(tmp_root, "array.json")
        with open(array_file, "w") as f:
            f.write("[]")
        assert bm.load_manifest(array_file) == 0


def test_save_manifest_creates_parent_directory():
    """Verify save_manifest creates non-existent parent directories automatically."""
    bm = BatchManager()
    item = QueueItem(index=1, source_path="s.mp4", relative_source_path="s.mp4", target_scene_name="t.mp4", target_full_path="t.mp4")
    bm.queue = [item]

    with tempfile.TemporaryDirectory() as tmp_root:
        nested_file = os.path.join(tmp_root, "deep", "nested", "manifest.json")
        assert bm.save_manifest(nested_file) is True
        assert os.path.isfile(nested_file)


def test_get_preflight_summary_nested_nonexistent_output_dir():
    """Verify get_preflight_summary calculates free disk space even when output_dir is deeply nested and non-existent."""
    bm = BatchManager()
    item = QueueItem(index=1, source_path="test.mp4", relative_source_path="test.mp4", target_scene_name="out.mp4", target_full_path="out.mp4", file_size_mb=100.0)
    bm.queue = [item]

    with tempfile.TemporaryDirectory() as tmp_root:
        nested_target = os.path.join(tmp_root, "non_existent_folder_a", "non_existent_folder_b")
        summary = bm.get_preflight_summary(nested_target, target_preset="4k")
        assert summary["total_items"] == 1
        assert summary["free_disk_mb"] > 0
        assert summary["estimated_output_mb"] == 250.0


def test_export_diagnostic_report_nested_dir():
    """Verify export_diagnostic_report creates parent directories when exporting to file."""
    with tempfile.TemporaryDirectory() as tmp_root:
        out_report = os.path.join(tmp_root, "logs", "doctor", "report.txt")
        report = export_diagnostic_report(out_report)
        assert os.path.isfile(out_report)
        assert len(report) > 0


def test_validate_job_config_case_insensitivity(tmp_path):
    """Verify validate_job_config accepts uppercase target preset, codec, and audio mode."""
    from src.core.validator import validate_job_config
    from src.core.video_pipeline import UpscaleJobConfig

    dummy_vid = tmp_path / "input.mp4"
    dummy_vid.write_bytes(b"\x00" * 1024)

    cfg = UpscaleJobConfig(
        input_file=str(dummy_vid),
        output_dir=str(tmp_path / "out"),
        target_preset="4K",
        video_codec="HEVC_NVENC",
        audio_mode="COPY",
        nvenc_preset="P4",
    )
    # Should not raise ValueError
    validate_job_config(cfg)


def test_format_clean_name_resolution_tags():
    """Verify format_clean_name standardizes resolution tags to uppercase 4K and clean naming."""
    from src.core.scene_renamer import format_clean_name

    assert format_clean_name("Episode_01.mp4", resolution_tag="4k") == "Episode_01_4K.mp4"
    assert format_clean_name("Episode_01.mp4", resolution_tag="2160p") == "Episode_01_4K.mp4"
    assert format_clean_name("Episode_01.mp4", resolution_tag="1440p") == "Episode_01_1440p.mp4"
    assert format_clean_name("Episode_01.mp4", resolution_tag="1080p") == "Episode_01_1080p.mp4"
    assert format_clean_name("Episode_01.mp4", resolution_tag="720p") == "Episode_01_720p.mp4"
    assert format_clean_name("mfhg 14 10 08 dee dee lynn 2160p.mp4", resolution_tag="4k") == "mfhg 14 10 08 dee dee lynn_4K.mp4"


def test_cli_manifest_restore_without_source(tmp_path, capsys):
    """Verify CLI can restore and dry-run from a manifest without requiring --source."""
    from src.cli import run_cli

    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Episode_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    manifest_file = tmp_path / "session_manifest.json"
    manifest_data = {
        "version": 1,
        "timestamp": 123456789.0,
        "items": [
            {
                "id": "item1234567890abcdef",
                "index": 1,
                "source_path": str(video_file),
                "relative_source_path": "Episode_01.mp4",
                "target_scene_name": "Show.S01E01.mp4",
                "target_full_path": str(out_dir / "Show.S01E01.mp4"),
                "status": "Pending",
                "file_size_mb": 1.0,
            }
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    dummy_meta = VideoMetadata(
        filepath=str(video_file),
        width=1280,
        height=720,
        fps=24.0,
        duration=10.0,
        nb_frames=240,
        codec_name="h264",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False,
    )

    with patch("src.cli.probe_video", return_value=dummy_meta):
        ret = run_cli([
            "--manifest", str(manifest_file),
            "--dry-run",
            "--yes",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Restoring queue from manifest" in captured.out
        assert "DRY-RUN OUTPUT PLAN" in captured.out

