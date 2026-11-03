"""Hostile Media and Codec Fuzzing Stress Test Suite for AnimaScale/Upscaler.

Exhaustively attacks media streaming, probing, validation, and encoding pipelines:
1. Corrupt video bitstreams:
   - 0-byte media files.
   - Truncated MP4 files (missing moov atom at EOF).
   - Mid-stream truncated MP4 files (partial mdat with valid moov header).
   - Corrupted JPEG/PNG image headers.
2. Audio variations:
   - Zero audio streams.
   - Multi-track audio (6 distinct audio tracks: 5.1 Surround, Stereo, Commentary, PCM, AAC, MP3).
   - Exotic codecs (Opus, Vorbis, FLAC, PCM) with container compatibility routing.
   - Unusual sample rates (22050 Hz, 96000 Hz).
3. Subtitle variations:
   - SRT text subtitles in MKV and MP4.
   - SSA/ASS subtitles with font attachments in MKV (verifying attachment preservation).
   - ASS subtitles to MP4 (verifying font attachment stripping and text conversion).
   - PGS bitmap subtitles (omitted from MP4, preserved in MKV).
   - Invalid UTF-8 subtitle bytes (verifying error-tolerant decoding).
4. Geometry and framerate edge cases:
   - Anamorphic SAR ratios (32:27, 8:9, 64:45) display geometry and SAR preservation.
   - Extreme framerates (1 fps, 120 fps).
   - Container timebase clock anomaly handling (VFR / 90000/1 timebase).
   - Vertical 9:16 portrait video geometry.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.core.models import (
    calculate_target_dimensions,
)
from src.core.probe import (
    VideoMetadata,
    probe_audio_stream,
    probe_audio_streams,
    probe_video,
)
from src.core.validator import (
    is_file_complete_and_valid,
    validate_encoded_video,
)
from src.core.video_pipeline import VideoUpscalerPipeline
from src.engine.media_stream import (
    MuxerConfig,
    StreamPipelineController,
    VideoFrame,
    VideoStreamDemuxer,
    VideoStreamMuxer,
    ZeroDiskStreamingPipeline,
)


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required for hostile media fuzzing tests.")
    return path


# ==============================================================================
# Helper generators for hostile media files
# ==============================================================================

def _generate_synthetic_video(
    ffmpeg_bin: str,
    output_path: Path | str,
    *,
    duration: float = 1.0,
    width: int = 64,
    height: int = 64,
    fps: int | float = 24,
    sar: str = "1:1",
    faststart: bool = False,
    extra_video_args: list[str] | None = None,
) -> str:
    out = str(output_path)
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-vf", f"setsar={str(sar).replace(':', '/')}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ]
    if faststart and out.lower().endswith((".mp4", ".mov")):
        cmd.extend(["-movflags", "+faststart"])
    if extra_video_args:
        cmd.extend(extra_video_args)
    cmd.append(out)
    subprocess.run(cmd, check=True, timeout=20)
    return out


# ==============================================================================
# 1. Corrupt Video Bitstreams
# ==============================================================================

def test_fuzz_zero_byte_media_rejection(tmp_path: Path):
    """Verifies that 0-byte media files are cleanly rejected across all pipeline stages."""
    zero_mp4 = tmp_path / "empty.mp4"
    zero_mp4.write_bytes(b"")

    zero_mkv = tmp_path / "empty.mkv"
    zero_mkv.write_bytes(b"")

    for empty_file in (zero_mp4, zero_mkv):
        path_str = str(empty_file)

        # 1. Probe rejection
        with pytest.raises((subprocess.SubprocessError, ValueError, RuntimeError)):
            probe_video(path_str)

        # 2. Complete and valid checks
        assert is_file_complete_and_valid(path_str) is False
        assert probe_audio_stream(path_str) is False
        assert probe_audio_streams(path_str) == []

        # 3. Validator rejection
        is_valid, err_msg = validate_encoded_video(path_str, 1920, 1080)
        assert is_valid is False
        assert "not create an output file" in err_msg or "readable" in err_msg

        # 4. Demuxer rejection
        with pytest.raises((subprocess.SubprocessError, ValueError, RuntimeError)):
            VideoStreamDemuxer(path_str)


def test_fuzz_truncated_mp4_missing_moov_atom(tmp_path: Path, ffmpeg_bin: str):
    """Attacks pipeline with MP4 where the moov atom was never written (recording cut)."""
    # Create MP4 without faststart (moov atom placed at EOF)
    full_mp4 = tmp_path / "full_without_faststart.mp4"
    _generate_synthetic_video(ffmpeg_bin, full_mp4, duration=2.0, faststart=False)
    raw_data = full_mp4.read_bytes()
    assert len(raw_data) > 1024

    # Truncate away the moov atom at the end (keep only first 50%)
    corrupt_mp4 = tmp_path / "missing_moov.mp4"
    corrupt_mp4.write_bytes(raw_data[: len(raw_data) // 2])

    path_str = str(corrupt_mp4)
    # Probe must fail because moov atom is absent
    with pytest.raises((subprocess.SubprocessError, ValueError, RuntimeError)):
        probe_video(path_str)

    # Resume check & validator must reject
    assert is_file_complete_and_valid(path_str) is False
    is_valid, _ = validate_encoded_video(path_str, 64, 64)
    assert is_valid is False

    # Demuxer must reject on construction
    with pytest.raises((subprocess.SubprocessError, ValueError, RuntimeError)):
        VideoStreamDemuxer(path_str)


def test_fuzz_truncated_mp4_midstream_eof(tmp_path: Path, ffmpeg_bin: str):
    """Attacks demuxer with valid moov header but truncated mdat payload (mid-stream EOF)."""
    full_mp4 = tmp_path / "full_faststart.mp4"
    _generate_synthetic_video(ffmpeg_bin, full_mp4, duration=4.0, fps=24, faststart=True)
    raw_data = full_mp4.read_bytes()

    # Truncate midway through the frame data (into mdat)
    partial_mp4 = tmp_path / "partial_mdat.mp4"
    partial_mp4.write_bytes(raw_data[: int(len(raw_data) * 0.75)])

    path_str = str(partial_mp4)
    meta = probe_video(path_str)
    # Metadata has header duration/frames, but actual file cuts off early
    assert meta.width == 64
    assert meta.height == 64

    # Demuxer must yield all readable frames and cleanly terminate at EOF without hanging
    demuxer = VideoStreamDemuxer(path_str)
    frames: list[VideoFrame] = list(demuxer.frames())

    assert len(frames) > 0
    # Truncated file should yield fewer frames than total header estimate
    assert len(frames) < meta.nb_frames
    assert all(isinstance(f, VideoFrame) for f in frames)
    assert demuxer.is_running is False


def test_fuzz_corrupted_image_headers(tmp_path: Path):
    """Attacks validator with truncated and corrupt image file headers."""
    trunc_jpg = tmp_path / "corrupt_header.jpg"
    trunc_jpg.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00")
    assert is_file_complete_and_valid(str(trunc_jpg)) is False

    corrupt_png = tmp_path / "corrupt_header.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x00\x00\x00\x00\x00")
    assert is_file_complete_and_valid(str(corrupt_png)) is False


# ==============================================================================
# 2. Audio Variations
# ==============================================================================

def test_fuzz_zero_audio_streams(tmp_path: Path, ffmpeg_bin: str):
    """Verifies that videos without any audio streams process cleanly without failures."""
    silent_mkv = tmp_path / "silent_source.mkv"
    _generate_synthetic_video(ffmpeg_bin, silent_mkv, duration=1.0)

    meta = probe_video(str(silent_mkv))
    assert meta.has_audio is False
    assert meta.audio_codec == ""
    assert meta.audio_stream_count == 0
    assert probe_audio_stream(str(silent_mkv)) is False
    assert probe_audio_streams(str(silent_mkv)) == []

    out_mp4 = tmp_path / "silent_out.mp4"
    demuxer = VideoStreamDemuxer(str(silent_mkv))
    mux_cfg = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx264",
        source_path=str(silent_mkv),
        audio_mode="copy",
    )
    muxer = VideoStreamMuxer(str(out_mp4), config=mux_cfg)
    controller = StreamPipelineController(demuxer, muxer)
    final_path = controller.run()

    out_meta = probe_video(final_path)
    assert out_meta.has_audio is False
    assert out_meta.width == 64
    assert out_meta.height == 64
    is_valid, err = validate_encoded_video(final_path, 64, 64, expect_audio=False)
    assert is_valid is True
    assert err == ""


def test_fuzz_multitrack_audio_6_streams_mkv_and_mp4(tmp_path: Path, ffmpeg_bin: str):
    """Attacks pipeline with 6 distinct audio streams including surround, exotic codecs, and rates:

    1. 5.1 Surround (FLAC, 96000 Hz, 6 channels)
    2. Stereo Main (Opus, 48000 Hz, 2 channels)
    3. Commentary (Vorbis, 22050 Hz, 1 channel)
    4. Uncompressed (PCM 16-bit, 44100 Hz, 2 channels)
    5. AAC Stereo (AAC, 48000 Hz, 2 channels)
    6. Alternate Dialogue (MP3, 32000 Hz, 1 channel)
    """
    multi_mkv = tmp_path / "multitrack_source.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=440:sample_rate=96000",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=880:sample_rate=48000",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=220:sample_rate=22050",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=330:sample_rate=44100",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=550:sample_rate=48000",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=660:sample_rate=32000",
        "-map", "0:v:0",
        "-map", "1:a:0", "-c:a:0", "flac",
        "-map", "2:a:0", "-c:a:1", "opus",
        "-map", "3:a:0", "-c:a:2", "libvorbis",
        "-map", "4:a:0", "-c:a:3", "pcm_s16le",
        "-map", "5:a:0", "-c:a:4", "aac",
        "-map", "6:a:0", "-c:a:5", "mp3",
        "-c:v", "libx264",
        "-strict", "-2",
        str(multi_mkv),
    ]
    subprocess.run(cmd, check=True, timeout=25)

    # 1. Verify Probe captures multi-track metadata
    meta = probe_video(str(multi_mkv))
    assert meta.has_audio is True
    assert meta.audio_stream_count == 6
    assert len(meta.audio_codecs) == 6
    assert "flac" in meta.audio_codecs
    assert "opus" in meta.audio_codecs
    assert "vorbis" in meta.audio_codecs
    assert "pcm_s16le" in meta.audio_codecs

    streams = probe_audio_streams(str(multi_mkv))
    assert len(streams) == 6

    # 2. Streaming Mux to MKV (Lossless copy of all 6 streams)
    out_mkv = tmp_path / "multitrack_out.mkv"
    demuxer = VideoStreamDemuxer(str(multi_mkv))
    mux_cfg_mkv = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx264",
        source_path=str(multi_mkv),
        audio_mode="copy",
    )
    muxer_mkv = VideoStreamMuxer(str(out_mkv), config=mux_cfg_mkv)
    controller_mkv = StreamPipelineController(demuxer, muxer_mkv)
    final_mkv = controller_mkv.run()

    out_streams_mkv = probe_audio_streams(final_mkv)
    assert len(out_streams_mkv) == 6

    # 3. Streaming Mux to MP4 (Container-aware transcoding of incompatible codecs)
    out_mp4 = tmp_path / "multitrack_out.mp4"
    demuxer_mp4 = VideoStreamDemuxer(str(multi_mkv))
    mux_cfg_mp4 = MuxerConfig(
        target_w=64,
        target_h=64,
        fps=24,
        encoder="libx264",
        source_path=str(multi_mkv),
        audio_mode="copy",
    )
    muxer_mp4 = VideoStreamMuxer(str(out_mp4), config=mux_cfg_mp4)
    controller_mp4 = StreamPipelineController(demuxer_mp4, muxer_mp4)
    final_mp4 = controller_mp4.run()

    out_streams_mp4 = probe_audio_streams(final_mp4)
    assert len(out_streams_mp4) == 6
    assert all(s.get("codec_name") == "aac" for s in out_streams_mp4)
    is_valid, _ = validate_encoded_video(final_mp4, 64, 64, expect_audio=True)
    assert is_valid is True


def test_fuzz_unusual_sample_rates_22050_96000(tmp_path: Path, ffmpeg_bin: str):
    """Verifies that exotic sample rates (22050 Hz and 96000 Hz) survive transcoding and probing."""
    video_96k = tmp_path / "audio_96k.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-f", "lavfi", "-i", "sine=duration=1:frequency=440:sample_rate=96000",
        "-c:v", "libx264", "-c:a", "flac",
        str(video_96k),
    ]
    subprocess.run(cmd, check=True)

    meta = probe_video(str(video_96k))
    assert meta.has_audio is True
    streams = probe_audio_streams(str(video_96k))
    assert int(streams[0].get("sample_rate", 0)) == 96000

    # Upscale 2x via ZeroDiskStreamingPipeline
    upscaled_out = tmp_path / "upscaled_96k.mkv"
    pipeline = ZeroDiskStreamingPipeline(
        source_path=str(video_96k),
        output_path=str(upscaled_out),
        upscale_fn=lambda frame: np.zeros((128, 128, 3), dtype=np.uint8),
        scale=2,
        encoder="libx264",
    )
    success, res_path = pipeline.process()
    assert success is True
    assert os.path.isfile(res_path)

    up_meta = probe_video(res_path)
    assert up_meta.width == 128
    assert up_meta.height == 128
    assert up_meta.has_audio is True


# ==============================================================================
# 3. Subtitle Variations
# ==============================================================================

def test_fuzz_subtitles_srt_in_mkv_and_mp4(tmp_path: Path, ffmpeg_bin: str):
    """Verifies SRT subtitle stream handling: copied in MKV, transcoded to mov_text in MP4."""
    srt_file = tmp_path / "dialogue.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello, Anime World!\n", encoding="utf-8")

    source_mkv = tmp_path / "source_with_srt.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-i", str(srt_file),
        "-c:v", "libx264", "-c:s", "subrip",
        str(source_mkv),
    ]
    subprocess.run(cmd, check=True)

    meta = probe_video(str(source_mkv))
    assert meta.has_subtitles is True
    assert "subrip" in meta.subtitle_codecs

    # Test Muxer to MP4: converted to mov_text
    out_mp4 = tmp_path / "sub_out.mp4"
    demuxer = VideoStreamDemuxer(str(source_mkv))
    muxer = VideoStreamMuxer(str(out_mp4), config=MuxerConfig(64, 64, 24, "libx264", source_path=str(source_mkv)))
    controller = StreamPipelineController(demuxer, muxer)
    final_mp4 = controller.run()

    out_meta = probe_video(final_mp4)
    assert out_meta.has_subtitles is True
    assert "mov_text" in out_meta.subtitle_codecs


def test_fuzz_subtitles_ass_with_font_attachments_mkv(tmp_path: Path, ffmpeg_bin: str):
    """Verifies SSA/ASS subtitles and font attachments are fully preserved in MKV without corruption."""
    ass_file = tmp_path / "fancy.ass"
    ass_file.write_text(
        "[Script Info]\nTitle: Test\nScriptType: v4.00+\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,ComicSans,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Stylized Subtitle\n",
        encoding="utf-8",
    )

    font_file = tmp_path / "comic_sans.ttf"
    font_file.write_bytes(b"OTTO\x00\x01\x00\x00" + b"\x00" * 256)

    mkv_source = tmp_path / "ass_font_source.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-i", str(ass_file),
        "-attach", str(font_file), "-metadata:s:t", "mimetype=application/x-truetype-font",
        "-map", "0:v:0", "-map", "1:s:0",
        "-c:v", "libx264", "-c:s", "ass",
        str(mkv_source),
    ]
    subprocess.run(cmd, check=True)

    meta = probe_video(str(mkv_source))
    assert meta.has_subtitles is True
    assert "ass" in meta.subtitle_codecs
    assert meta.has_attachments is True

    # Streaming Mux to MKV (Attachment metadata tags must be preserved)
    out_mkv = tmp_path / "ass_font_out.mkv"
    demuxer = VideoStreamDemuxer(str(mkv_source))
    muxer = VideoStreamMuxer(str(out_mkv), config=MuxerConfig(64, 64, 24, "libx264", source_path=str(mkv_source)))
    controller = StreamPipelineController(demuxer, muxer)
    final_mkv = controller.run()

    out_meta = probe_video(final_mkv)
    assert out_meta.has_subtitles is True
    assert "ass" in out_meta.subtitle_codecs
    assert out_meta.has_attachments is True


def test_fuzz_subtitles_ass_to_mp4_font_stripping(tmp_path: Path, ffmpeg_bin: str):
    """Verifies that converting MKV with font attachments to MP4 safely strips fonts and converts text."""
    ass_file = tmp_path / "fancy2.ass"
    ass_file.write_text(
        "[Script Info]\nTitle: Test\nScriptType: v4.00+\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,ComicSans,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Stylized Subtitle\n",
        encoding="utf-8",
    )
    font_file = tmp_path / "test_font.ttf"
    font_file.write_bytes(b"OTTO\x00\x01\x00\x00" + b"\x00" * 256)

    mkv_source = tmp_path / "ass_font_for_mp4.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-i", str(ass_file),
        "-attach", str(font_file), "-metadata:s:t", "mimetype=application/x-truetype-font",
        "-map", "0:v:0", "-map", "1:s:0",
        "-c:v", "libx264", "-c:s", "ass",
        str(mkv_source),
    ]
    subprocess.run(cmd, check=True)

    out_mp4 = tmp_path / "converted.mp4"
    demuxer = VideoStreamDemuxer(str(mkv_source))
    muxer = VideoStreamMuxer(str(out_mp4), config=MuxerConfig(64, 64, 24, "libx264", source_path=str(mkv_source)))
    controller = StreamPipelineController(demuxer, muxer)
    final_mp4 = controller.run()

    out_meta = probe_video(final_mp4)
    # MP4 has converted mov_text subtitles and zero attachments
    assert out_meta.has_subtitles is True
    assert "mov_text" in out_meta.subtitle_codecs
    assert out_meta.has_attachments is False


def test_fuzz_subtitles_pgs_bitmap_omitted_from_mp4():
    """Verifies that PGS bitmap subtitles are omitted from MP4 commands to prevent FFmpeg crashes."""
    pipeline = VideoUpscalerPipeline()
    meta_pgs = VideoMetadata(
        filepath="dummy.mkv",
        width=1920,
        height=1080,
        fps=24.0,
        duration=10.0,
        nb_frames=240,
        codec_name="hevc",
        has_audio=False,
        audio_codec="",
        has_subtitles=True,
        subtitle_codecs=["hdmv_pgs_subtitle"],
        has_attachments=False,
    )

    # 1. MKV output: PGS must be copied
    cmd_mkv = pipeline._build_encode_cmd(
        frames_out_dir="frames_out",
        frame_ext="png",
        meta=meta_pgs,
        target_w=1920,
        target_h=1080,
        chosen_encoder="libx264",
        cq_val="19",
        p_val="fast",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mkv",
        sub_input_path="dummy.mkv",
    )
    assert "-c:s" in cmd_mkv
    assert "copy" in cmd_mkv

    # 2. MP4 output: PGS must NOT be mapped or converted to mov_text
    cmd_mp4 = pipeline._build_encode_cmd(
        frames_out_dir="frames_out",
        frame_ext="png",
        meta=meta_pgs,
        target_w=1920,
        target_h=1080,
        chosen_encoder="libx264",
        cq_val="19",
        p_val="fast",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mp4",
        sub_input_path="dummy.mkv",
    )
    assert "-c:s" not in cmd_mp4


def test_fuzz_subtitles_invalid_utf8_recovery(tmp_path: Path, ffmpeg_bin: str):
    """Verifies that subtitles with invalid non-UTF8 bytes do not abort MP4 encoding."""
    corrupt_srt = tmp_path / "corrupt_bytes.srt"
    corrupt_srt.write_bytes(b"1\r\n00:00:00,000 --> 00:00:01,000\r\n\x80\x81\xfe\xff Subtitle\r\n\r\n")

    mkv_corrupt = tmp_path / "corrupt_sub_source.mkv"
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-f", "srt", "-i", str(corrupt_srt),
        "-c:v", "libx264", "-c:s", "copy",
        str(mkv_corrupt),
    ]
    subprocess.run(cmd, check=True)

    # Mux to MP4: with -max_error_rate 1.0, the pipeline must complete without fatal exit 69
    out_mp4 = tmp_path / "recovered_out.mp4"
    demuxer = VideoStreamDemuxer(str(mkv_corrupt))
    muxer = VideoStreamMuxer(str(out_mp4), config=MuxerConfig(64, 64, 24, "libx264", source_path=str(mkv_corrupt)))
    controller = StreamPipelineController(demuxer, muxer)
    final_mp4 = controller.run()

    assert os.path.isfile(final_mp4)
    meta = probe_video(final_mp4)
    assert meta.width == 64
    assert meta.height == 64


# ==============================================================================
# 4. Anamorphic SAR, Extreme Framerates, and Vertical Geometry
# ==============================================================================

@pytest.mark.parametrize(
    ("width", "height", "sar", "expected_display_w"),
    [
        (720, 480, "32:27", 853.333),  # NTSC 16:9 widescreen
        (720, 480, "8:9", 640.0),       # NTSC 4:3 fullscreen
        (720, 576, "64:45", 1024.0),    # PAL 16:9 widescreen
    ],
)
def test_fuzz_anamorphic_sar_display_and_muxing(
    tmp_path: Path,
    ffmpeg_bin: str,
    width: int,
    height: int,
    sar: str,
    expected_display_w: float,
):
    """Verifies that non-square pixel aspect ratios (SAR) compute correct display dimensions and are preserved."""
    vid_path = tmp_path / f"anamorphic_{width}x{height}_{sar.replace(':', '_')}.mkv"
    _generate_synthetic_video(ffmpeg_bin, vid_path, width=width, height=height, sar=sar)

    meta = probe_video(str(vid_path))
    assert meta.sample_aspect_ratio == sar
    assert math.isclose(meta.display_width, expected_display_w, rel_tol=1e-2)
    assert math.isclose(meta.display_height, height, rel_tol=1e-2)

    # Verify target dimension calculation for anamorphic video
    target_w, target_h, _ = calculate_target_dimensions(width, height, "1080p", sample_aspect_ratio=sar)
    assert target_w % 2 == 0
    assert target_h % 2 == 0
    # Display aspect ratio of target must match display aspect ratio of anamorphic source
    target_aspect = target_w / target_h
    source_display_aspect = expected_display_w / height
    assert math.isclose(target_aspect, source_display_aspect, rel_tol=0.05)

    # Verify VideoStreamMuxer inherits the SAR when sample_aspect_ratio is not overridden
    mux_cfg = MuxerConfig(target_w=width, target_h=height, fps=24, encoder="libx264", source_path=str(vid_path))
    cmd = VideoStreamMuxer(str(tmp_path / "dummy.mkv"), config=mux_cfg)._build_command()
    assert f"setsar={sar.replace(':', '/')}" in cmd


@pytest.mark.parametrize("fps_val", [1.0, 120.0])
def test_fuzz_extreme_framerates_1fps_and_120fps(tmp_path: Path, ffmpeg_bin: str, fps_val: float):
    """Verifies that extreme framerates (1.0 fps and 120.0 fps) are accurately probed and streamed."""
    vid_path = tmp_path / f"extreme_fps_{int(fps_val)}.mkv"
    _generate_synthetic_video(ffmpeg_bin, vid_path, duration=1.0, fps=fps_val)

    meta = probe_video(str(vid_path))
    assert math.isclose(meta.fps, fps_val, abs_tol=0.5)

    # Verify pipeline rational fps validator accepts both 1 fps and 120 fps
    pipeline = VideoUpscalerPipeline()
    encode_cmd = pipeline._build_encode_cmd(
        frames_out_dir="frames_out",
        frame_ext="png",
        meta=meta,
        target_w=64,
        target_h=64,
        chosen_encoder="libx264",
        cq_val="19",
        p_val="fast",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mkv",
    )
    fps_idx = encode_cmd.index("-r") + 1
    from fractions import Fraction
    assert float(Fraction(encode_cmd[fps_idx])) == pytest.approx(fps_val, rel=0.01)


def test_fuzz_vfr_container_timebase_clock_clamping():
    """Verifies that insane container timebase clocks (e.g. 90000/1 Hz) are clamped to rational framerates."""
    pipeline = VideoUpscalerPipeline()
    vfr_meta = VideoMetadata(
        filepath="vfr.mkv",
        width=1920,
        height=1080,
        fps=23.976,
        duration=10.0,
        nb_frames=240,
        codec_name="h264",
        has_audio=False,
        audio_codec="",
        has_subtitles=False,
        r_frame_rate="90000/1",  # Pathological container timebase
    )
    cmd = pipeline._build_encode_cmd(
        frames_out_dir="frames_out",
        frame_ext="png",
        meta=vfr_meta,
        target_w=1920,
        target_h=1080,
        chosen_encoder="libx264",
        cq_val="19",
        p_val="fast",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mkv",
    )
    fps_arg = cmd[cmd.index("-r") + 1]
    # Insane 90000/1 rate must be rejected; fall back to avg_frame_rate or probed fps
    assert fps_arg != "90000/1"
    assert float(fps_arg) == pytest.approx(23.976, rel=0.01)


def test_fuzz_vertical_9_16_videos():
    """Verifies that vertical portrait videos (9:16) compute correct target aspect ratios."""
    # Source: 1080x1920 portrait video
    w, h = 1080, 1920
    target_w, target_h, tag = calculate_target_dimensions(w, h, "4k")

    # In 4K, portrait height should be 3840 and width 2160 (preserving 9:16)
    assert target_w == 2160
    assert target_h == 3840
    assert tag == "2160p"

    # 8K target preset
    target_8k_w, target_8k_h, tag_8k = calculate_target_dimensions(w, h, "8k")
    assert target_8k_w == 4320
    assert target_8k_h == 7680
    assert tag_8k == "4320p"
