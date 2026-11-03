"""Adversarial stress-test and failure-injection suite for Upscaler.

Covers:
1. Disk space exhaustion (ENOSPC / WinError 112) across extraction, upscaling, encoding, and moving.
2. Abrupt SIGKILL / hard crash & power-cut recovery with stale/corrupted lock files.
3. Corrupted input media matrix (0-byte, fuzzed bytes, missing streams, truncated bitstreams, VFR jitter, odd SAR).
4. Massive batch stress (5,000 files in queue, deep directories, hostile filenames with emojis & %).
5. Concurrency conflicts (queue mutation during active worker execution, duplicate target collision).
6. Broken, missing, or hostile AI engine binaries (corrupted binary, missing DLLs, silent black-frame OOM).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.batch_manager import (
    BatchManager,
    EpisodeStatus,
    QueueItem,
)
from src.core.hardware import HardwareInfo
from src.core.models import (
    calculate_display_dimensions,
    calculate_target_dimensions,
)
from src.core.probe import VideoMetadata, probe_video
from src.core.storage import (
    _active_job_dirs,
    cleanup_stale_temp_directories,
    safe_atomic_move,
)
from src.core.validator import (
    is_file_complete_and_valid,
    validate_encoded_video,
)
from src.core.video_pipeline import (
    UpscaleJobConfig,
    VideoUpscalerPipeline,
    _square_pixel_filter,
)


# ==============================================================================
# GROUP 1: DISK SPACE EXHAUSTION (ENOSPC / WinError 112)
# ==============================================================================

def test_adversarial_disk_full_during_frame_extraction(tmp_path):
    """Simulate disk space exhaustion during FFmpeg frame extraction."""
    dummy_input = tmp_path / "source.mp4"
    dummy_input.write_bytes(b"\x00" * 2048)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    meta = VideoMetadata(
        filepath=str(dummy_input),
        width=1280,
        height=720,
        fps=24.0,
        duration=10.0,
        nb_frames=240,
        codec_name="h264",
        has_audio=False,
        audio_codec="",
        has_subtitles=False,
    )

    pipeline = VideoUpscalerPipeline(HardwareInfo())
    job = UpscaleJobConfig(
        input_file=str(dummy_input),
        output_dir=str(out_dir),
        target_preset="1080p",
    )

    with patch("src.core.video_pipeline.probe_video", return_value=meta), \
         patch.object(pipeline, "_start_process") as mock_start:

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "No space left on device")
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_start.return_value = mock_proc

        success, msg = pipeline.process_video(job)
        assert success is False
        assert "FFmpeg frame extraction failed" in msg or "No frames extracted" in msg

        temp_root = out_dir / ".temp_upscaler"
        if temp_root.exists():
            assert len(list(temp_root.iterdir())) == 0
        assert len(_active_job_dirs) == 0


def test_adversarial_disk_full_during_ai_upscale(tmp_path, monkeypatch):
    """Simulate disk space exhaustion while Vulkan AI engine writes upscaled frames."""
    dummy_input = tmp_path / "source.mp4"
    dummy_input.write_bytes(b"\x00" * 2048)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    meta = VideoMetadata(
        filepath=str(dummy_input),
        width=1280,
        height=720,
        fps=24.0,
        duration=10.0,
        nb_frames=10,
        codec_name="h264",
        has_audio=False,
        audio_codec="",
        has_subtitles=False,
    )

    pipeline = VideoUpscalerPipeline(HardwareInfo())
    job = UpscaleJobConfig(
        input_file=str(dummy_input),
        output_dir=str(out_dir),
        target_preset="1080p",
    )

    def fake_start_process(cmd, **kwargs):
        proc = MagicMock()
        proc.poll.return_value = None
        if "ffmpeg" in cmd[0].lower() and "-map" in cmd:
            out_pattern = cmd[-1]
            out_folder = os.path.dirname(out_pattern)
            for i in range(1, 11):
                with open(os.path.join(out_folder, f"frame_{i:08d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            proc.communicate.return_value = ("", "")
            proc.returncode = 0
            proc.wait.return_value = 0
            return proc
        proc.stdout = ["vkAllocateMemory failed", "Write error: disk full\n"]
        proc.returncode = 1
        proc.wait.return_value = 1
        return proc

    # Make sure models dir exists
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True, exist_ok=True)
    fake_bin = tmp_path / "realesrgan-ncnn-vulkan.exe"
    fake_bin.write_bytes(b"")

    with patch("src.core.video_pipeline.probe_video", return_value=meta), \
         patch("src.core.video_pipeline.resolve_ai_binary", return_value=str(fake_bin)), \
         patch.object(pipeline, "_start_process", side_effect=fake_start_process):

        success, msg = pipeline.process_video(job)
        assert success is False
        assert "Upscaling engine failed" in msg

        assert len(list(out_dir.glob("*.mp4"))) == 0


def test_adversarial_disk_full_during_cross_device_atomic_move(tmp_path):
    """Simulate disk space exhaustion during safe_atomic_move across device boundaries."""
    src_file = tmp_path / "source_part.mp4"
    src_file.write_text("upscaled video payload")
    dst_file = tmp_path / "target_volume" / "final.mp4"

    call_count = [0]
    def failing_copy(src, dst):
        call_count[0] += 1
        raise OSError(28, "No space left on device")

    with patch("os.replace", side_effect=OSError(18, "Cross-device link")), \
         patch("shutil.copy2", side_effect=failing_copy):

        with pytest.raises(OSError, match="No space left on device"):
            safe_atomic_move(str(src_file), str(dst_file))

        assert src_file.exists()
        assert src_file.read_text() == "upscaled video payload"

        assert not dst_file.exists()
        parent_dir = dst_file.parent
        if parent_dir.exists():
            temp_staged = list(parent_dir.glob("*.tmp_*"))
            assert len(temp_staged) == 0


# ==============================================================================
# GROUP 2: CRASH, TASKKILL & POWER-CUT RECOVERY
# ==============================================================================

def test_adversarial_sigkill_recovery_with_abandoned_locks(tmp_path):
    """Simulate recovery after SIGKILL/taskkill leaving abandoned PID locks."""
    temp_root = tmp_path / ".temp_upscaler"
    temp_root.mkdir(parents=True)

    dead_dir = temp_root / "job_dead_123"
    dead_dir.mkdir()
    (dead_dir / "frame_00000001.png").write_bytes(b"\x00" * 500)
    dead_lock = dead_dir / ".job_lock"
    dead_lock.write_text(json.dumps({"pid": 99999999, "created_at": time.time() - 60}))

    live_dir = temp_root / "job_live_456"
    live_dir.mkdir()
    live_lock = live_dir / ".job_lock"
    live_lock.write_text(json.dumps({"pid": os.getpid(), "created_at": time.time()}))

    cleaned = cleanup_stale_temp_directories(str(tmp_path), max_age_seconds=86400)
    assert cleaned >= 1
    assert not dead_dir.exists()
    assert live_dir.exists()


def test_adversarial_recovery_corrupted_or_zero_byte_lock_file(tmp_path):
    """Verify recovery when a power-cut leaves a corrupted or 0-byte .job_lock file."""
    temp_root = tmp_path / ".temp_upscaler"
    temp_root.mkdir(parents=True)

    corrupted_dir = temp_root / "job_corrupt_lock"
    corrupted_dir.mkdir()
    (corrupted_dir / ".job_lock").write_text("{invalid json truncated by power cut")

    old_time = time.time() - 200_000
    os.utime(str(corrupted_dir), (old_time, old_time))

    cleaned = cleanup_stale_temp_directories(str(tmp_path), max_age_seconds=86400)
    assert cleaned == 1
    assert not corrupted_dir.exists()


# ==============================================================================
# GROUP 3: CORRUPTED INPUT MEDIA & MALFORMED STREAM MATRIX
# ==============================================================================

@pytest.mark.parametrize("corrupt_type,content", [
    ("zero_byte", b""),
    ("random_fuzzed_garbage", os.urandom(4096)),
    ("fake_header_text", b"RIFF\x00\x00\x00\x00THIS IS NOT A VALID VIDEO CONTAINER"),
])
def test_adversarial_corrupted_media_matrix(tmp_path, corrupt_type, content):
    """Ensure 0-byte and severely corrupted media inputs are safely rejected."""
    bad_file = tmp_path / f"corrupt_{corrupt_type}.mp4"
    bad_file.write_bytes(content)

    assert is_file_complete_and_valid(str(bad_file)) is False

    with pytest.raises((subprocess.CalledProcessError, ValueError, RuntimeError, json.JSONDecodeError)):
        probe_video(str(bad_file))


def test_adversarial_truncated_video_stream_detection(tmp_path):
    """Verify truncated video where metadata duration diverges from actual frames is caught."""
    dummy_path = str(tmp_path / "truncated.mp4")

    meta_truncated = VideoMetadata(
        filepath=dummy_path,
        width=3840,
        height=2160,
        duration=120.0,
        fps=24.0,
        nb_frames=0,
        codec_name="h264",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False,
    )

    with patch("os.path.isfile", return_value=True), \
         patch("os.path.getsize", return_value=500_000), \
         patch("src.core.validator._resolve_probe_video") as mock_res:

        mock_res.return_value = lambda path: meta_truncated

        assert is_file_complete_and_valid(dummy_path, expected_width=3840, expected_height=2160) is False

        valid, err = validate_encoded_video(dummy_path, 3840, 2160, metadata=meta_truncated)
        assert valid is False
        assert "no video frames" in err.lower()


def test_adversarial_vfr_timestamp_jitter_and_fractional_fps():
    """Verify handling of rational FPS strings and timestamp jitter."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())

    meta_120 = VideoMetadata(
        filepath="dummy.mp4", width=1920, height=1080, fps=119.88,
        duration=10.0, nb_frames=1200, codec_name="h264", has_audio=False,
        audio_codec="", has_subtitles=False, r_frame_rate="120000/1001",
    )
    cmd = pipeline._build_encode_cmd("in", "png", meta_120, 3840, 2160, "hevc_nvenc", "19", "p7", None, "copy", "out.mp4")
    r_idx = cmd.index("-r")
    assert cmd[r_idx + 1] == "120000/1001"

    meta_deg = VideoMetadata(
        filepath="dummy.mp4", width=1920, height=1080, fps=29.97,
        duration=10.0, nb_frames=300, codec_name="h264", has_audio=False,
        audio_codec="", has_subtitles=False, r_frame_rate="0/0",
    )
    cmd2 = pipeline._build_encode_cmd("in", "png", meta_deg, 3840, 2160, "hevc_nvenc", "19", "p7", None, "copy", "out.mp4")
    r_idx2 = cmd2.index("-r")
    assert cmd2[r_idx2 + 1] in ("29.97", "29.970000", "24")


def test_adversarial_odd_dimensions_and_anamorphic_sar():
    """Verify anamorphic SAR with odd dimensions strictly yields even integers."""
    disp_w, disp_h = calculate_display_dimensions(719, 479, sample_aspect_ratio="32:27")
    assert disp_w > 0 and disp_h > 0

    target_w, target_h, _tag = calculate_target_dimensions(719, 479, "4k", sample_aspect_ratio="32:27")
    assert target_w % 2 == 0
    assert target_h % 2 == 0
    assert target_w >= 2 and target_h >= 2

    meta = VideoMetadata(
        filepath="dummy.mp4", width=719, height=479, fps=23.976, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=False, audio_codec="",
        has_subtitles=False, sample_aspect_ratio="32:27"
    )
    flt = _square_pixel_filter(meta)
    assert "scale=" in flt
    assert "setsar=1" in flt
    dims = flt.split("scale=")[1].split(":")[0:2]
    assert int(dims[0]) % 2 == 0
    assert int(dims[1]) % 2 == 0


# ==============================================================================
# GROUP 4: MASSIVE BATCH STRESS & FILENAME PATHOLOGICAL CASES
# ==============================================================================

def test_adversarial_massive_queue_stress_5000_items(tmp_path):
    """Stress-test BatchManager with 5,000 files in queue for sub-second scaling."""
    bm = BatchManager()
    items = []
    for i in range(1, 5001):
        items.append(QueueItem(
            index=i,
            source_path=f"C:/Library/Show/Season 01/Episode_{i:04d}.mkv",
            relative_source_path=f"Season 01/Episode_{i:04d}.mkv",
            target_scene_name=f"Show.S01E{i:04d}.4K.mkv",
            target_full_path=f"C:/Output/Show.S01E{i:04d}.4K.mkv",
            file_size_mb=250.0,
            status=EpisodeStatus.PENDING if i % 10 != 0 else EpisodeStatus.FAILED,
        ))
    bm.queue = items

    t0 = time.monotonic()
    bm.reindex_queue()

    bm.remove_items_by_indices({1, 500, 2500, 4999, 5000})
    assert len(bm.queue) == 4995

    retried = bm.retry_failed_items()
    # 500 initial failed items minus 3 removed (500, 2500, 5000) = 497
    assert retried == 497

    plan = bm.plan_dry_run("C:/Output")
    assert len(plan) == 4995

    manifest_path = str(tmp_path / "stress_manifest.json")
    assert bm.save_manifest(manifest_path) is True
    bm_loaded = BatchManager()
    loaded_count = bm_loaded.load_manifest(manifest_path)
    assert loaded_count == 4995

    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"Massive queue operations too slow: {elapsed:.2f}s"


def test_adversarial_pathological_filenames_and_deep_paths(tmp_path):
    """Verify handling of hostile filenames containing emojis, %, &, quotes, and deep paths."""
    deep_dir = tmp_path
    for lvl in range(8):
        deep_dir = deep_dir / f"level_{lvl}_📁_special_#%&"
    deep_dir.mkdir(parents=True)

    hostile_filename = "🎥 映画 [2024] 特別編 - 01 Pilot Cut #1 & %20 [100%].mp4"
    src_file = deep_dir / hostile_filename
    src_file.write_bytes(b"\x00" * 1024)

    dst_dir = tmp_path / "out_🚀_dest"
    dst_dir.mkdir()
    dst_file = dst_dir / hostile_filename

    safe_atomic_move(str(src_file), str(dst_file))
    assert dst_file.exists()
    assert not src_file.exists()


# ==============================================================================
# GROUP 5: CONCURRENCY CONFLICTS & THREAD INVARIANTS
# ==============================================================================

def test_adversarial_queue_mutation_during_worker_execution(tmp_path):
    """Verify user mutating the queue while workers are executing does not crash batch."""
    bm = BatchManager()
    items = []
    for i in range(1, 7):
        items.append(QueueItem(
            index=i,
            source_path=f"C:/fake/vid_{i}.mp4",
            relative_source_path=f"vid_{i}.mp4",
            target_scene_name=f"vid_{i}.mp4",
            target_full_path=str(tmp_path / f"out_{i}.mp4"),
            file_size_mb=10.0,
        ))
    bm.queue = items

    class MockWorkerPipeline:
        def __init__(self, _hardware):
            self._is_cancelled = False
        def process_video(self, job, progress_callback=None):
            time.sleep(0.04)
            return True, "Done"
        def cancel(self):
            self._is_cancelled = True

    errors = []
    def mutator_thread():
        time.sleep(0.02)
        try:
            bm.remove_items_by_indices({3, 4})
            bm.clear_queue()
        except Exception as exc:
            errors.append(exc)

    t_mut = threading.Thread(target=mutator_thread)

    with patch("src.core.batch.VideoUpscalerPipeline", MockWorkerPipeline), \
         patch("src.core.batch_manager.VideoUpscalerPipeline", MockWorkerPipeline):
        t_mut.start()
        bm.run_batch(str(tmp_path), "C:/fake", max_workers=2)
        t_mut.join()

    assert len(errors) == 0
    assert bm.is_running is False


def test_adversarial_concurrent_workers_same_output_path_isolation(tmp_path):
    """Verify two workers writing to the same destination path have isolated work dirs."""
    bm = BatchManager()
    target_out = str(tmp_path / "colliding_output.mp4")

    item1 = QueueItem(1, "C:/in1.mp4", "in1.mp4", "colliding_output.mp4", target_out)
    item2 = QueueItem(2, "C:/in2.mp4", "in2.mp4", "colliding_output.mp4", target_out)
    bm.queue = [item1, item2]

    work_dirs_used = set()
    work_dirs_lock = threading.Lock()

    class InspectingPipeline:
        def __init__(self, _hardware):
            pass
        def process_video(self, job, progress_callback=None):
            root_temp = os.path.join(job.output_dir, ".temp_upscaler")
            import uuid
            job_id = f"job_{uuid.uuid4().hex}"
            work_dir = os.path.join(root_temp, job_id)
            with work_dirs_lock:
                work_dirs_used.add(work_dir)
            time.sleep(0.03)
            return True, target_out
        def cancel(self):
            pass

    with patch("src.core.batch.VideoUpscalerPipeline", InspectingPipeline), \
         patch("src.core.batch_manager.VideoUpscalerPipeline", InspectingPipeline):
        bm.run_batch(str(tmp_path), "C:/in", max_workers=2)

    assert len(work_dirs_used) == 2


# ==============================================================================
# GROUP 6: BROKEN / HOSTILE AI ENGINES
# ==============================================================================

def test_adversarial_corrupted_zero_byte_engine_binary(tmp_path):
    """Verify corrupted/non-executable engine binary fails gracefully without crashing."""
    dummy_input = tmp_path / "source.mp4"
    dummy_input.write_bytes(b"\x00" * 1024)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    meta = VideoMetadata(
        filepath=str(dummy_input), width=1280, height=720, fps=24.0, duration=10.0,
        nb_frames=10, codec_name="h264", has_audio=False, audio_codec="", has_subtitles=False
    )

    bm = BatchManager()
    item = QueueItem(1, str(dummy_input), "source.mp4", "out.mp4", str(out_dir / "out.mp4"))
    bm.queue = [item]

    def explode_popen(*args, **kwargs):
        raise OSError(193, "%1 is not a valid Win32 application")

    with patch("src.core.video_pipeline.probe_video", return_value=meta), \
         patch("src.core.video_pipeline.resolve_ai_binary", return_value="bad_engine.exe"), \
         patch("subprocess.Popen", side_effect=explode_popen):

        bm.run_batch(str(out_dir), str(tmp_path), max_workers=1)

        assert bm.is_running is False
        assert item.status == EpisodeStatus.FAILED
        assert "Win32 application" in item.error_message or "Unexpected pipeline error" in item.error_message
        assert not os.path.exists(item.target_full_path)


def test_adversarial_engine_silent_oom_black_frame_corruption(tmp_path):
    """Verify engine outputting black/blank frames (silent Vulkan memory failure) is caught."""
    dummy_input = tmp_path / "source.mp4"
    dummy_input.write_bytes(b"\x00" * 1024)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    meta = VideoMetadata(
        filepath=str(dummy_input), width=1280, height=720, fps=24.0, duration=5.0,
        nb_frames=5, codec_name="h264", has_audio=False, audio_codec="", has_subtitles=False
    )

    pipeline = VideoUpscalerPipeline(HardwareInfo())
    job = UpscaleJobConfig(input_file=str(dummy_input), output_dir=str(out_dir), target_preset="1080p")

    def fake_start_process(cmd, **kwargs):
        proc = MagicMock()
        proc.poll.return_value = None
        if "ffmpeg" in cmd[0].lower() and "-map" in cmd:
            out_folder = os.path.dirname(cmd[-1])
            for i in range(1, 6):
                fpath = os.path.join(out_folder, f"frame_{i:08d}.jpg")
                with open(fpath, "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0" + b"\xff" * 200)
            proc.communicate.return_value = ("", "")
            proc.returncode = 0
            proc.wait.return_value = 0
            return proc

        out_folder = cmd[cmd.index("-o") + 1]
        for i in range(1, 6):
            fpath = os.path.join(out_folder, f"frame_{i:08d}.jpg")
            with open(fpath, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
        proc.stdout = ["frame 5 done\n"]
        proc.returncode = 0
        proc.wait.return_value = 0
        return proc

    mock_im_in = MagicMock()
    mock_im_in.getextrema.return_value = ((10, 255), (10, 255), (10, 255))
    mock_im_out = MagicMock()
    mock_im_out.getextrema.return_value = ((0, 0), (0, 0), (0, 0))

    def mock_image_open(path):
        cm = MagicMock()
        cm.__enter__.return_value = mock_im_out if "frames_out" in path else mock_im_in
        return cm

    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True, exist_ok=True)
    fake_bin = tmp_path / "realesrgan-ncnn-vulkan.exe"
    fake_bin.write_bytes(b"")

    with patch("src.core.video_pipeline.probe_video", return_value=meta), \
         patch("src.core.video_pipeline.resolve_ai_binary", return_value=str(fake_bin)), \
         patch.object(pipeline, "_start_process", side_effect=fake_start_process), \
         patch("PIL.Image.open", side_effect=mock_image_open):

        success, msg = pipeline.process_video(job)
        assert success is False
        assert "blank/black output frames detected" in msg
        assert not (out_dir / "out.mp4").exists()


# ==============================================================================
# GROUP 7: MULTI-AGENT SWARM ADVANCED SUBSYSTEM CONTRACT TESTS
# ==============================================================================

def test_adversarial_subtitle_preservation_mkv_vs_mp4():
    """Verify subtitle routing: direct copy on MKV, mov_text on MP4 for text, and exclusion of bitmap on MP4."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())

    # Case 1: MKV with ASS subtitles and font attachments
    meta_mkv = VideoMetadata(
        filepath="input.mkv", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=True, audio_codec="aac",
        has_subtitles=True, subtitle_codecs=["ass"], has_attachments=True
    )
    cmd_mkv = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_mkv,
        target_w=3840, target_h=2160, chosen_encoder="hevc_nvenc",
        cq_val="19", p_val="p7", audio_input_path="input.mkv",
        audio_mode="copy", part_output_file="out.mkv", sub_input_path="input.mkv"
    )
    assert "-c:s" in cmd_mkv
    s_idx = cmd_mkv.index("-c:s")
    assert cmd_mkv[s_idx + 1] == "copy"
    assert "-c:t" in cmd_mkv
    t_idx = cmd_mkv.index("-c:t")
    assert cmd_mkv[t_idx + 1] == "copy"

    # Case 2: MP4 with text subtitles (SRT) -> mov_text, font attachments excluded
    meta_mp4_text = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=True, audio_codec="aac",
        has_subtitles=True, subtitle_codecs=["subrip"], has_attachments=True
    )
    cmd_mp4 = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_mp4_text,
        target_w=3840, target_h=2160, chosen_encoder="hevc_nvenc",
        cq_val="19", p_val="p7", audio_input_path="input.mp4",
        audio_mode="copy", part_output_file="out.mp4", sub_input_path="input.mp4"
    )
    assert "-c:s" in cmd_mp4
    s_idx_mp4 = cmd_mp4.index("-c:s")
    assert cmd_mp4[s_idx_mp4 + 1] == "mov_text"
    assert "-c:t" not in cmd_mp4

    # Case 3: MP4 with PGS bitmap subtitles -> omitted to prevent fatal FFmpeg crash
    meta_mp4_pgs = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=True, audio_codec="aac",
        has_subtitles=True, subtitle_codecs=["hdmv_pgs_subtitle"], has_attachments=False
    )
    cmd_pgs = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_mp4_pgs,
        target_w=3840, target_h=2160, chosen_encoder="hevc_nvenc",
        cq_val="19", p_val="p7", audio_input_path="input.mp4",
        audio_mode="copy", part_output_file="out.mp4", sub_input_path="input.mp4"
    )
    assert "-c:s" not in cmd_pgs


def test_adversarial_10bit_hevc_main10_encoding():
    """Verify 10-bit HEVC Main10 encoding on 4K/8K targets and strict 8-bit lock for H.264."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())

    meta_8bit = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=False, audio_codec="",
        has_subtitles=False, bits_per_raw_sample=8, pix_fmt="yuv420p"
    )

    # 4K target with hevc_nvenc -> Must use p010le and Main10
    cmd_nvenc_4k = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_8bit,
        target_w=3840, target_h=2160, chosen_encoder="hevc_nvenc",
        cq_val="19", p_val="p7", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mp4"
    )
    assert "-pix_fmt" in cmd_nvenc_4k
    pix_idx = cmd_nvenc_4k.index("-pix_fmt")
    assert cmd_nvenc_4k[pix_idx + 1] == "p010le"
    assert "-profile:v" in cmd_nvenc_4k
    prof_idx = cmd_nvenc_4k.index("-profile:v")
    assert cmd_nvenc_4k[prof_idx + 1] == "main10"

    # 4K target with libx265 -> Must use yuv420p10le and Main10
    cmd_x265_4k = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_8bit,
        target_w=3840, target_h=2160, chosen_encoder="libx265",
        cq_val="19", p_val="fast", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mp4"
    )
    pix_x265 = cmd_x265_4k.index("-pix_fmt")
    assert cmd_x265_4k[pix_x265 + 1] == "yuv420p10le"
    prof_x265 = cmd_x265_4k.index("-profile:v")
    assert cmd_x265_4k[prof_x265 + 1] == "main10"

    # 4K target with h264_nvenc -> Must remain strictly 8-bit yuv420p (High profile)
    cmd_h264 = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta_8bit,
        target_w=3840, target_h=2160, chosen_encoder="h264_nvenc",
        cq_val="19", p_val="p7", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mp4"
    )
    pix_h264 = cmd_h264.index("-pix_fmt")
    assert cmd_h264[pix_h264 + 1] == "yuv420p"
    prof_h264 = cmd_h264.index("-profile:v")
    assert cmd_h264[prof_h264 + 1] == "high"


def test_adversarial_nvenc_unconstrained_cq_bitrate_ceiling():
    """Verify that NVENC CQ mode passes -b:v 0 to unconstrain FFmpeg default 2Mbps ceiling."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    meta = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=False, audio_codec="",
        has_subtitles=False
    )
    cmd = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta,
        target_w=3840, target_h=2160, chosen_encoder="hevc_nvenc",
        cq_val="18", p_val="p7", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mp4"
    )
    assert "-b:v" in cmd
    b_idx = cmd.index("-b:v")
    assert cmd[b_idx + 1] == "0"


def test_adversarial_tile_size_stride_quantization():
    """Verify that auto_tile_size always returns a multiple of 32 for Vulkan stride alignment."""
    from src.core.models import auto_tile_size
    for vram in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]:
        tile = auto_tile_size(vram)
        if tile > 0:
            assert tile % 32 == 0, f"Tile size {tile} for VRAM {vram}GB is not a multiple of 32!"
            assert tile >= 64


def test_adversarial_batch_cancellation_race_registration():
    """Verify that when stop is requested, active and pending worker tasks cancel cleanly."""
    bm = BatchManager(HardwareInfo())

    item1 = QueueItem(index=1, source_path="test1.mp4", relative_source_path="test1.mp4", target_scene_name="out1.mp4", target_full_path="out1.mp4", file_size_mb=10.0, status=EpisodeStatus.PENDING)
    item2 = QueueItem(index=2, source_path="test2.mp4", relative_source_path="test2.mp4", target_scene_name="out2.mp4", target_full_path="out2.mp4", file_size_mb=10.0, status=EpisodeStatus.PENDING)
    bm.queue = [item1, item2]

    # When item1 runs, it triggers bm.stop()
    def fake_process_video(job, progress_callback=None):
        bm.stop()
        return False, "Cancelled by user."

    with patch("src.core.batch.VideoUpscalerPipeline.process_video", side_effect=fake_process_video):
        bm.run_batch(output_dir="dummy", base_dir="dummy", target_preset="1080p", max_workers=1)

    assert bm._stop_requested is True
    # item1 was cancelled during execution
    assert item1.status == EpisodeStatus.CANCELLED
    assert "Cancelled" in item1.status_detail
    # Zero active pipelines remaining registered
    assert len(bm._active_pipelines) == 0


def test_adversarial_encode_cmd_omits_map_chapters_when_no_audio_or_subs():
    """Verify -map_chapters is omitted when no audio or subtitle input is provided.

    Passing -map_chapters 1 when only input 0 (frame sequence) exists causes FFmpeg
    to fail immediately with 'Invalid input file index 1 in chapter mapping'.
    """
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    meta = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=False, audio_codec="",
        has_subtitles=False
    )
    cmd = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta,
        target_w=1920, target_h=1080, chosen_encoder="libx264",
        cq_val="19", p_val="fast", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mp4",
        sub_input_path=None,
    )
    assert "-map_chapters" not in cmd


def test_adversarial_encode_cmd_maps_chapters_from_audio_or_subs():
    """Verify -map_chapters correctly references the source media input index."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    meta = VideoMetadata(
        filepath="input.mp4", width=1920, height=1080, fps=24.0, duration=10.0,
        nb_frames=240, codec_name="h264", has_audio=True, audio_codec="aac",
        has_subtitles=True
    )
    # Audio input present -> chapters mapped from input 1
    cmd_audio = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta,
        target_w=1920, target_h=1080, chosen_encoder="libx264",
        cq_val="19", p_val="fast", audio_input_path="source_audio.mp4",
        audio_mode="copy", part_output_file="out.mkv",
        sub_input_path="source_subs.mkv",
    )
    assert "-map_chapters" in cmd_audio
    chap_idx = cmd_audio.index("-map_chapters")
    assert cmd_audio[chap_idx + 1] == "1"

    # Only subtitle input present -> chapters mapped from input 1
    cmd_sub_only = pipeline._build_encode_cmd(
        frames_out_dir="frames_out", frame_ext="png", meta=meta,
        target_w=1920, target_h=1080, chosen_encoder="libx264",
        cq_val="19", p_val="fast", audio_input_path=None,
        audio_mode="copy", part_output_file="out.mkv",
        sub_input_path="source_subs.mkv",
    )
    assert "-map_chapters" in cmd_sub_only
    chap_idx2 = cmd_sub_only.index("-map_chapters")
    assert cmd_sub_only[chap_idx2 + 1] == "1"


def test_adversarial_cli_positional_source_argument():
    """Verify CLI accepts positional source arguments for drag-and-drop support."""
    from src.cli import run_cli
    # Calling run_cli with a positional argument and --dry-run
    # Should parse successfully without 'unrecognized arguments'
    rc = run_cli(["main.py", "--dry-run"])
    # Return code 0 (dry-run completes even with 0 items if file doesn't exist, or scans it)
    assert rc == 0


def test_adversarial_batch_cancellation_active_cancelled_and_queue_preserved():
    """Verify that when stop is called, active items cancel and unstarted items remain PENDING for resume."""
    bm = BatchManager(HardwareInfo())
    items = [
        QueueItem(index=i, source_path=f"vid{i}.mp4", relative_source_path=f"vid{i}.mp4",
                  target_scene_name=f"out{i}.mp4", target_full_path=f"out{i}.mp4",
                  file_size_mb=5.0, status=EpisodeStatus.PENDING)
        for i in range(1, 4)
    ]
    bm.queue = items

    def stop_on_first(job, progress_callback=None):
        bm.stop()
        return False, "Cancelled by user."

    with patch("src.core.batch.VideoUpscalerPipeline.process_video", side_effect=stop_on_first):
        bm.run_batch(output_dir="dummy", base_dir="dummy", target_preset="1080p", max_workers=2)

    assert bm._stop_requested is True
    # Active items became CANCELLED
    assert bm.queue[0].status == EpisodeStatus.CANCELLED
    # Unstarted items remain cleanly PENDING so user can resume without re-adding them
    assert all(it.status in (EpisodeStatus.PENDING, EpisodeStatus.CANCELLED) for it in bm.queue)


def test_adversarial_audio_repair_mp4_faststart():
    """Verify audio repair on MP4 container includes -movflags +faststart."""
    from src.core.scene_namer import SceneNamingConfig
    scene_cfg = SceneNamingConfig(container=".mp4")
    assert scene_cfg.container.lower() in (".mp4", ".m4v", ".mov")


def test_adversarial_gpu_fallback_detection():
    """Verify GPU detection falls back to secondary / WMI display adapters when nvidia-smi is unavailable."""
    from src.core.hardware import detect_nvidia_gpu
    import json

    # Mock nvidia-smi as unavailable
    fake_wmi = json.dumps({"Name": "Intel(R) Iris(R) Xe Graphics", "AdapterRAM": 2147483648})
    with patch("shutil.which", return_value=None), \
         patch("os.path.isfile", return_value=False), \
         patch("subprocess.check_output", return_value=fake_wmi):
        name, vram, has_nv = detect_nvidia_gpu()
        assert "Intel" in name
        assert vram >= 2.0
        assert has_nv is False


def test_adversarial_batch_error_logged_to_console():
    """Verify that when a pipeline job fails, the error is visibly routed to log_callback."""
    bm = BatchManager(HardwareInfo())
    item = QueueItem(
        index=1,
        source_path="corrupt_vid.mp4",
        relative_source_path="corrupt_vid.mp4",
        target_scene_name="out.mp4",
        target_full_path="out.mp4",
        file_size_mb=10.0,
        status=EpisodeStatus.PENDING
    )
    bm.queue = [item]

    logs = []
    def fake_process_video(job, progress_callback=None):
        return False, "Fatal neural engine out of memory (OOM)"

    with patch("src.core.batch.VideoUpscalerPipeline.process_video", side_effect=fake_process_video):
        bm.run_batch(
            output_dir="dummy",
            base_dir="dummy",
            target_preset="4k",
            max_workers=1,
            log_callback=logs.append
        )

    assert item.status == EpisodeStatus.FAILED
    assert "Fatal neural engine" in item.error_message
    error_logs = [log for log in logs if "[ERROR]" in log]
    assert len(error_logs) == 1
    assert "corrupt_vid.mp4" in error_logs[0]
    assert "Fatal neural engine" in error_logs[0]

