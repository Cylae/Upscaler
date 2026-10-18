"""Output verification, job contract validation, and integrity auditing.

Enforces strict compliance with requested resolution, duration tolerances,
and stream integrity.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from src.core.models import AVAILABLE_MODELS
from src.core.probe import VideoMetadata, probe_video


def _resolve_probe_video():
    import sys
    vp = sys.modules.get("src.core.video_pipeline")
    if vp and hasattr(vp, "probe_video"):
        return vp.probe_video
    return probe_video


def is_file_complete_and_valid(
    filepath: str,
    expect_audio: bool = False,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_duration: Optional[float] = None,
    duration_tolerance_s: float = 2.0,
    duration_tolerance_ratio: float = 0.02,
) -> bool:
    """Checks if an already existing output file is valid and complete, matching the expected job contract."""
    if not os.path.isfile(filepath):
        return False
    if os.path.getsize(filepath) < 100_000:  # < 100KB is incomplete
        return False
    try:
        probe_fn = _resolve_probe_video()
        meta = probe_fn(filepath)
        if meta.duration <= 0.5 or meta.width <= 0 or meta.height <= 0:
            return False
        # nb_frames > 0 guards against truncated files where the MP4 container
        # header (moov atom) was written successfully but the frame data is
        # incomplete or missing — ffprobe can read metadata from these files
        # without error, causing a false positive on the resume check.
        if meta.nb_frames <= 0:
            return False
        if expect_audio and not meta.has_audio:
            return False
        if expected_width is not None and expected_height is not None:
            if meta.width != expected_width or meta.height != expected_height:
                return False
        if expected_duration is not None and expected_duration > 1.0:
            allowed_diff = max(duration_tolerance_s, expected_duration * duration_tolerance_ratio)
            if abs(meta.duration - expected_duration) > allowed_diff:
                return False
        return True
    except Exception:
        return False


def validate_encoded_video(
    filepath: str,
    expected_width: int,
    expected_height: int,
    expect_audio: bool = False,
    metadata: Optional[VideoMetadata] = None,
) -> Tuple[bool, str]:
    """Validates that FFmpeg produced a playable video with valid dimensions and streams."""
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        return False, "Encoder did not create an output file."
    if metadata is None:
        try:
            probe_fn = _resolve_probe_video()
            meta = probe_fn(filepath)
        except Exception as exc:
            return False, f"Final output has no readable video stream: {exc}"
    else:
        meta = metadata

    if meta.width != expected_width or meta.height != expected_height:
        return False, (
            f"Final output resolution is {meta.width}x{meta.height}, expected "
            f"{expected_width}x{expected_height}."
        )
    if meta.duration <= 0 or meta.nb_frames <= 0:
        return False, "Final output contains no video frames."
    if expect_audio and not meta.has_audio:
        return False, "Final output is missing its expected audio stream."
    return True, ""


def validate_job_config(job) -> None:
    """Fails fast on invalid programmatic jobs before creating temporary files."""
    if not os.path.isfile(job.input_file):
        raise ValueError(f"Input video does not exist: {job.input_file}")
    if not job.output_dir:
        raise ValueError("Output directory is required.")
    if job.target_preset.lower() not in {"8k", "4k", "1440p", "1080p", "720p"}:
        raise ValueError(f"Unsupported target resolution: {job.target_preset}")
    if job.video_codec.lower() not in {"hevc_nvenc", "h264_nvenc", "libx265", "libx264"}:
        raise ValueError(f"Unsupported video codec: {job.video_codec}")
    if job.model_id not in AVAILABLE_MODELS and job.model_id != "auto":
        raise ValueError(f"Unknown AI model: {job.model_id}")
    if job.frame_format.lower() not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported frame format: {job.frame_format}")
    if not 14 <= job.cq_level <= 28:
        raise ValueError("CQ level must be between 14 and 28.")
    if job.nvenc_preset.lower() not in {f"p{index}" for index in range(1, 8)}:
        raise ValueError(f"Unsupported NVENC preset: {job.nvenc_preset}")
    if job.denoise_level not in {-1, 0, 1, 2, 3}:
        raise ValueError("RealCUGAN denoise level must be between -1 and 3.")
    if job.audio_mode.lower() not in {"copy", "aac_192k", "aac_320k", "flac"}:
        raise ValueError(f"Unsupported audio mode: {job.audio_mode}")
    if job.tile_size != 0 and job.tile_size < 32:
        raise ValueError("Tile size must be 0 (auto) or at least 32 pixels.")
    if job.threads != "auto":
        match = re.fullmatch(r"(\d+):(\d+):(\d+)", job.threads)
        if not match or any(int(value) < 1 for value in match.groups()):
            raise ValueError("Threads must be 'auto' or LOAD:PROC:SAVE with positive integers.")
    if job.preview_duration_s is not None and job.preview_duration_s <= 0:
        raise ValueError("Preview duration must be greater than zero.")
