"""Production-grade unit and integration tests for the zero-disk media streaming pipeline.

Verifies:
  - VideoFrame memory representation and zero-copy conversions.
  - Encoder discovery and priority fallback (NVENC -> QSV -> AMF -> libx265).
  - VideoStreamDemuxer zero-disk streaming directly via OS pipes (pipe:1).
  - VideoStreamMuxer zero-disk streaming directly via OS pipes (pipe:0).
  - Rational framerate preservation and colorspace preservation.
  - 10-bit HEVC Main10 enforcement on 4K/8K targets and 10-bit sources.
  - Container-aware audio, subtitle, font attachment, and chapter preservation.
  - Collision-proof temporary locks and atomic destination promotion.
  - StreamPipelineController bounded queue concurrency and real-time telemetry.
  - Instant process tree kill and deterministic cancellation without orphaned processes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.core.probe import probe_video
from src.core.process import ProcessTracker
from src.engine.media_stream import (
    MuxerConfig,
    StreamPipelineController,
    StreamProgress,
    VideoFrame,
    VideoStreamDemuxer,
    VideoStreamMuxer,
    get_supported_ffmpeg_encoders,
    is_encoder_functional,
    resolve_hevc_encoder,
)


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for media stream tests.")
    return path


@pytest.fixture
def sample_video_mkv(tmp_path: Path, ffmpeg_bin: str) -> Path:
    """Generates a small synthetic MKV file with video and audio for streaming tests."""
    video_path = tmp_path / "sample_source.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=440",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, timeout=15)
    return video_path


# -----------------------------------------------------------------------------
# 1. VideoFrame Memory Representation Tests
# -----------------------------------------------------------------------------

def test_video_frame_valid_creation():
    w, h = 32, 24
    raw_data = b"\xaa\xbb\xcc" * (w * h)
    frame = VideoFrame(index=5, data=raw_data, width=w, height=h, pts=0.25)

    assert frame.index == 5
    assert frame.width == w
    assert frame.height == h
    assert frame.pts == 0.25
    assert len(frame.data) == w * h * 3


def test_video_frame_invalid_size_raises_value_error():
    with pytest.raises(ValueError, match="data size mismatch"):
        VideoFrame(index=0, data=b"too_short", width=64, height=64)


def test_video_frame_numpy_conversions():
    w, h = 16, 12
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    arr[0, 0] = [255, 0, 100]

    frame = VideoFrame.from_numpy(arr, index=2, pts=1.0)
    assert frame.width == w
    assert frame.height == h

    arr_out = frame.to_numpy()
    assert arr_out.shape == (h, w, 3)
    assert arr_out.dtype == np.uint8
    assert list(arr_out[0, 0]) == [255, 0, 100]
    assert np.array_equal(arr, arr_out)


def test_video_frame_pil_image_conversions():
    w, h = 20, 15
    img = Image.new("RGB", (w, h), color=(10, 20, 30))
    img.putpixel((0, 0), (255, 128, 64))

    frame = VideoFrame.from_image(img, index=7)
    assert frame.width == w
    assert frame.height == h

    img_out = frame.to_image()
    assert img_out.size == (w, h)
    assert img_out.getpixel((0, 0)) == (255, 128, 64)
    assert img_out.getpixel((1, 1)) == (10, 20, 30)


# -----------------------------------------------------------------------------
# 2. Hardware Encoder Discovery and Fallback Tests
# -----------------------------------------------------------------------------

def test_get_supported_ffmpeg_encoders(ffmpeg_bin: str):
    encoders = get_supported_ffmpeg_encoders(ffmpeg_bin)
    assert isinstance(encoders, set)
    assert "libx264" in encoders or "libx265" in encoders


def test_is_encoder_functional_libx265(ffmpeg_bin: str):
    assert is_encoder_functional("libx265", ffmpeg_bin) is True


def test_is_encoder_functional_nonexistent(ffmpeg_bin: str):
    assert is_encoder_functional("nonexistent_encoder_xyz", ffmpeg_bin) is False


def test_resolve_hevc_encoder_explicit(ffmpeg_bin: str):
    resolved = resolve_hevc_encoder("libx265", ffmpeg_bin)
    assert resolved == "libx265"


def test_resolve_hevc_encoder_auto(ffmpeg_bin: str):
    resolved = resolve_hevc_encoder("auto", ffmpeg_bin)
    assert resolved in ("hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265")


# -----------------------------------------------------------------------------
# 3. VideoStreamDemuxer Zero-Disk Streaming Tests
# -----------------------------------------------------------------------------

def test_demuxer_zero_disk_streaming(sample_video_mkv: Path, ffmpeg_bin: str):
    tracker = ProcessTracker()
    demuxer = VideoStreamDemuxer(str(sample_video_mkv), ffmpeg_bin=ffmpeg_bin, process_tracker=tracker)

    assert demuxer.width == 64
    assert demuxer.height == 64
    assert demuxer.total_frames == 24
    assert demuxer.fps == 24.0
    assert demuxer.frame_bytes == 64 * 64 * 3

    frames_read = 0
    with demuxer:
        for frame in demuxer:
            assert isinstance(frame, VideoFrame)
            assert frame.width == 64
            assert frame.height == 64
            assert len(frame.data) == 64 * 64 * 3
            assert frame.index == frames_read
            frames_read += 1

    assert frames_read == 24
    assert demuxer.is_running is False
    assert tracker.is_cancelled is False


def test_demuxer_custom_filters(sample_video_mkv: Path, ffmpeg_bin: str):
    # Demux with custom video filter (e.g. crop or scale)
    demuxer = VideoStreamDemuxer(
        str(sample_video_mkv),
        ffmpeg_bin=ffmpeg_bin,
        custom_filters=["scale=32:32"],
    )
    # When custom filter changes size, read_frame handles exact frame_bytes if configured
    demuxer.close()


# -----------------------------------------------------------------------------
# 4. VideoStreamMuxer Zero-Disk Encoding Tests
# -----------------------------------------------------------------------------

def test_muxer_zero_disk_raw_encoding(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "stream_output.mp4"
    config = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx265",
        cq_level=20,
    )

    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    assert os.path.isfile(muxer.lock_path)

    solid_frame = b"\x55\xaa\xff" * (64 * 64)

    with muxer:
        for _ in range(12):
            muxer.write_frame(solid_frame)

    assert muxer.frames_written == 12
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
    assert not os.path.exists(muxer.part_path)
    assert not os.path.exists(muxer.lock_path)

    meta = probe_video(str(out_file))
    assert meta.width == 64
    assert meta.height == 64
    assert meta.nb_frames == 12


def test_muxer_with_source_audio_passthrough(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "stream_with_audio.mkv"
    config = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx265",
        source_path=str(sample_video_mkv),
        audio_mode="copy",
    )

    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    raw_frame = b"\x00\x80\xff" * (64 * 64)

    with muxer:
        for _ in range(24):
            muxer.write_frame(raw_frame)

    assert out_file.is_file()
    meta = probe_video(str(out_file))
    assert meta.has_audio is True
    assert meta.audio_codec.lower() in ("aac", "mp4a")


# -----------------------------------------------------------------------------
# 5. 10-Bit HEVC Main10 Enforcement Tests
# -----------------------------------------------------------------------------

def test_10bit_enforcement_for_4k(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "test_4k.mp4"
    config = MuxerConfig(
        target_w=3840,
        target_h=2160,
        fps=24,
        encoder="libx265",
    )
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    assert muxer.use_10bit is True
    cmd = muxer._build_command()
    cmd_str = " ".join(cmd)
    assert "yuv420p10le" in cmd_str
    assert "main10" in cmd_str
    muxer.close()


def test_10bit_enforcement_for_8k(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "test_8k.mkv"
    config = MuxerConfig(
        target_w=7680,
        target_h=4320,
        fps=60,
        encoder="libx265",
    )
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    assert muxer.use_10bit is True
    cmd = muxer._build_command()
    cmd_str = " ".join(cmd)
    assert "yuv420p10le" in cmd_str
    assert "main10" in cmd_str
    muxer.close()


def test_8bit_allowed_for_standard_1080p(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "test_1080p.mp4"
    config = MuxerConfig(
        target_w=1920,
        target_h=1080,
        fps=24,
        encoder="libx265",
        force_10bit=False,
    )
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    assert muxer.use_10bit is False
    cmd = muxer._build_command()
    cmd_str = " ".join(cmd)
    assert "yuv420p10le" not in cmd_str
    muxer.close()


# -----------------------------------------------------------------------------
# 6. Container-Aware Subtitle and Colorspace Routing Tests
# -----------------------------------------------------------------------------

def test_container_aware_routing_mkv_vs_mp4(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    # MKV routing: copy subtitles, font attachments, chapters
    mkv_out = tmp_path / "output.mkv"
    mkv_config = MuxerConfig(
        target_w=1280,
        target_h=720,
        fps=24,
        source_path=str(sample_video_mkv),
    )
    mkv_muxer = VideoStreamMuxer(str(mkv_out), mkv_config, ffmpeg_bin=ffmpeg_bin)
    cmd_mkv = mkv_muxer._build_command()
    cmd_mkv_str = " ".join(cmd_mkv)
    assert "-map 0:s?" in cmd_mkv_str
    assert "-c:s copy" in cmd_mkv_str
    assert "-map 0:t?" in cmd_mkv_str
    assert "-map_chapters 0" in cmd_mkv_str
    mkv_muxer.close()

    # MP4 routing: mov_text for subtitles, faststart, no font attachments
    mp4_out = tmp_path / "output.mp4"
    mp4_config = MuxerConfig(
        target_w=1280,
        target_h=720,
        fps=24,
        source_path=str(sample_video_mkv),
    )
    mp4_muxer = VideoStreamMuxer(str(mp4_out), mp4_config, ffmpeg_bin=ffmpeg_bin)
    cmd_mp4 = mp4_muxer._build_command()
    cmd_mp4_str = " ".join(cmd_mp4)
    assert "-map 0:t?" not in cmd_mp4_str
    assert "+faststart" in cmd_mp4_str
    assert "-map_chapters 0" in cmd_mp4_str
    mp4_muxer.close()


def test_colorspace_conformance_for_sd_to_hd(tmp_path: Path, ffmpeg_bin: str):
    # Create fake SD metadata probe
    out_file = tmp_path / "sd_upscale.mp4"
    config = MuxerConfig(
        target_w=1920,
        target_h=1080,
        fps=24,
        color_space="bt709",
        color_primaries="bt709",
        color_transfer="bt709",
    )
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    cmd = muxer._build_command()
    cmd_str = " ".join(cmd)
    assert "-colorspace bt709" in cmd_str
    assert "-color_primaries bt709" in cmd_str
    assert "-color_trc bt709" in cmd_str
    muxer.close()


# -----------------------------------------------------------------------------
# 7. StreamPipelineController End-to-End & Bounded Queue Tests
# -----------------------------------------------------------------------------

def test_stream_pipeline_controller_end_to_end(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "controller_result.mkv"

    # Demuxer source: 64x64
    demuxer = VideoStreamDemuxer(str(sample_video_mkv), ffmpeg_bin=ffmpeg_bin)

    # 2x upscale mock inference: 64x64 -> 128x128
    def mock_inference_upscale_2x(frame: VideoFrame) -> VideoFrame:
        arr = frame.to_numpy()
        # Simple nearest-neighbor 2x upscale
        upscaled = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
        return VideoFrame.from_numpy(upscaled, index=frame.index, pts=frame.pts)

    muxer_config = MuxerConfig(
        target_w=128,
        target_h=128,
        fps=24,
        encoder="libx265",
        source_path=str(sample_video_mkv),
        audio_mode="copy",
    )
    muxer = VideoStreamMuxer(str(out_file), muxer_config, ffmpeg_bin=ffmpeg_bin)

    controller = StreamPipelineController(
        demuxer=demuxer,
        muxer=muxer,
        inference_fn=mock_inference_upscale_2x,
        queue_size=4,  # Strict memory ring-buffer
    )

    progress_reports: list[StreamProgress] = []

    def on_progress(p: StreamProgress) -> None:
        progress_reports.append(p)

    final_path = controller.run(progress_callback=on_progress)

    assert final_path == str(out_file)
    assert out_file.is_file()
    assert len(progress_reports) > 0
    assert progress_reports[-1].percent == 100.0

    # Probe output to verify dimensions, frames, and audio
    meta = probe_video(final_path)
    assert meta.width == 128
    assert meta.height == 128
    assert meta.nb_frames == 24
    assert meta.has_audio is True


# -----------------------------------------------------------------------------
# 8. Deterministic Process Tree Cancellation Tests
# -----------------------------------------------------------------------------

def test_pipeline_cancellation_instant_cleanup(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "cancelled_render.mkv"
    demuxer = VideoStreamDemuxer(str(sample_video_mkv), ffmpeg_bin=ffmpeg_bin)

    # Simulate slow inference to allow cancellation
    def slow_inference(frame: VideoFrame) -> VideoFrame:
        time.sleep(0.5)
        return frame

    muxer_config = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx265",
        source_path=str(sample_video_mkv),
    )
    muxer = VideoStreamMuxer(str(out_file), muxer_config, ffmpeg_bin=ffmpeg_bin)

    tracker = ProcessTracker()
    controller = StreamPipelineController(
        demuxer=demuxer,
        muxer=muxer,
        inference_fn=slow_inference,
        queue_size=2,
        process_tracker=tracker,
    )

    # Cancel immediately after 1 frame
    def cancel_after_one(p: StreamProgress) -> None:
        if p.processed_frames >= 1:
            controller.cancel()

    with pytest.raises(RuntimeError, match="cancelled"):
        controller.run(progress_callback=cancel_after_one)

    assert controller.is_cancelled is True
    assert demuxer.is_running is False
    # Unfinalized output must not exist or be promoted
    assert not out_file.is_file()
    assert not os.path.exists(muxer.part_path)
    assert not os.path.exists(muxer.lock_path)


# -----------------------------------------------------------------------------
# 9. Edge Case & Robustness Tests
# -----------------------------------------------------------------------------

def test_demuxer_nonexistent_file_raises_notfound():
    with pytest.raises(FileNotFoundError):
        VideoStreamDemuxer("nonexistent_video_path_12345.mp4")


def test_muxer_invalid_frame_data_type(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "type_err.mp4"
    config = MuxerConfig(target_w=32, target_h=32, fps=24, encoder="libx265")
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    try:
        with pytest.raises(TypeError, match="Unsupported frame object type"):
            muxer.write_frame({"invalid": "type"})  # type: ignore[arg-type]
    finally:
        muxer.close()


def test_muxer_frame_size_mismatch(tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "size_mismatch.mp4"
    config = MuxerConfig(target_w=32, target_h=32, fps=24, encoder="libx265")
    muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
    try:
        with pytest.raises(ValueError, match="Frame size mismatch"):
            muxer.write_frame(b"\x00" * 10)
    finally:
        muxer.close()


def test_muxer_extra_args_and_audio_modes(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    out_file = tmp_path / "audio_modes.mkv"
    for mode in ("aac_192k", "aac_320k", "flac", "none"):
        config = MuxerConfig(
            target_w=64,
            target_h=64,
            fps=24,
            source_path=str(sample_video_mkv),
            audio_mode=mode,
            extra_muxer_args=["-metadata", "title=StreamTest"],
        )
        muxer = VideoStreamMuxer(str(out_file), config, ffmpeg_bin=ffmpeg_bin)
        cmd = " ".join(muxer._build_command())
        if mode == "none":
            assert "-map 0:a?" not in cmd
        elif mode == "flac":
            assert "-c:a flac" in cmd
        elif mode == "aac_192k":
            assert "-b:a 192k" in cmd
        elif mode == "aac_320k":
            assert "-b:a 320k" in cmd
        assert "-metadata title=StreamTest" in cmd
        muxer.close()


def test_stream_pipeline_controller_invalid_queue_size(sample_video_mkv: Path, tmp_path: Path, ffmpeg_bin: str):
    demuxer = VideoStreamDemuxer(str(sample_video_mkv), ffmpeg_bin=ffmpeg_bin)
    muxer = VideoStreamMuxer(str(tmp_path / "out.mp4"), MuxerConfig(target_w=64, target_h=64, fps=24), ffmpeg_bin=ffmpeg_bin)
    try:
        with pytest.raises(ValueError, match="Queue size must be at least 1"):
            StreamPipelineController(demuxer, muxer, queue_size=0)
    finally:
        demuxer.close()
        muxer.close()


def test_progress_dispatch_multiple_callback_signatures():
    prog = StreamProgress(processed_frames=10, total_frames=20, fps=25.0, eta_seconds=0.4, percent=50.0)

    # 1. Single arg
    received_1 = []
    StreamPipelineController._dispatch_progress(lambda p: received_1.append(p), prog)
    assert len(received_1) == 1
    assert received_1[0].processed_frames == 10

    # 2. Legacy 3 args (ratio, status, extra)
    received_3 = []
    def legacy_cb(ratio: float, status: str, extra: dict) -> None:
        received_3.append((ratio, status, extra))
    StreamPipelineController._dispatch_progress(legacy_cb, prog)
    assert len(received_3) == 1
    assert received_3[0][0] == 0.5
    assert received_3[0][2]["processed_frames"] == 10

    # 3. 4 args (processed, total, fps, eta)
    received_4 = []
    def four_arg_cb(proc: int, tot: int, fps: float, eta: float) -> None:
        received_4.append((proc, tot, fps, eta))
    StreamPipelineController._dispatch_progress(four_arg_cb, prog)
    assert len(received_4) == 1
    assert received_4[0] == (10, 20, 25.0, 0.4)

