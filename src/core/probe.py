"""FFprobe video metadata inspector and stream analyzer.

Accurately extracts resolution, rational frame rate, duration, audio streams,
and color space attributes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VideoMetadata:
    filepath: str
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int
    codec_name: str
    has_audio: bool
    audio_codec: str
    has_subtitles: bool
    r_frame_rate: str = ""
    color_range: str = ""
    color_space: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    audio_stream_count: int = 1


def resolve_ffprobe_bin(ffprobe_bin: Optional[str] = None) -> str:
    """Finds the usable ffprobe executable path."""
    if ffprobe_bin and (os.path.isfile(ffprobe_bin) or shutil.which(ffprobe_bin)):
        return ffprobe_bin
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg_found = shutil.which("ffmpeg")
    if ffmpeg_found:
        candidate = os.path.join(
            os.path.dirname(ffmpeg_found),
            "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if os.path.isfile(candidate):
            return candidate
    return "ffprobe"


def probe_video(filepath: str, ffprobe_bin: Optional[str] = None) -> VideoMetadata:
    """Uses ffprobe to extract complete, structured video and stream metadata."""
    bin_name = resolve_ffprobe_bin(ffprobe_bin)
    cmd = [
        bin_name,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        filepath
    ]
    raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=30)
    data = json.loads(raw)

    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ValueError(f"No video stream found in {filepath}")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))

    fps = 0.0
    for fps_str in (v_stream.get("avg_frame_rate"), v_stream.get("r_frame_rate")):
        if not fps_str:
            continue
        try:
            fps_val_str = str(fps_str).strip()
            if "/" in fps_val_str:
                numerator, denominator = fps_val_str.split("/", 1)
                den = float(denominator)
                if den > 0:
                    val = float(numerator) / den
                    if val > 0:
                        fps = val
                        break
            else:
                val = float(fps_val_str)
                if val > 0:
                    fps = val
                    break
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    if fps <= 0:
        fps = 24.0

    duration = 0.0
    for raw_duration in (data.get("format", {}).get("duration"), v_stream.get("duration")):
        try:
            duration = float(raw_duration)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    nb_frames_raw = v_stream.get("nb_frames")
    nb_frames = int(nb_frames_raw) if nb_frames_raw and str(nb_frames_raw).isdigit() else max(1, int(duration * fps))
    if duration <= 0 and nb_frames > 0:
        duration = nb_frames / fps

    a_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    has_audio = len(a_streams) > 0
    audio_codec = a_streams[0].get("codec_name", "") if has_audio else ""

    s_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"), None)
    has_subtitles = s_stream is not None

    r_frame_rate = str(v_stream.get("r_frame_rate", "") or "")
    color_range = str(v_stream.get("color_range", "") or "")
    color_space = str(v_stream.get("color_space", "") or "")
    color_primaries = str(v_stream.get("color_primaries", "") or "")
    color_transfer = str(v_stream.get("color_transfer", "") or "")

    return VideoMetadata(
        filepath=filepath,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
        codec_name=v_stream.get("codec_name", ""),
        has_audio=has_audio,
        audio_codec=audio_codec,
        has_subtitles=has_subtitles,
        r_frame_rate=r_frame_rate,
        color_range=color_range,
        color_space=color_space,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        audio_stream_count=len(a_streams)
    )


def probe_audio_streams(filepath: str, ffprobe_bin: Optional[str] = None) -> List[Dict]:
    """Returns list of audio stream dictionaries."""
    bin_name = resolve_ffprobe_bin(ffprobe_bin)
    cmd = [
        bin_name,
        "-v", "error",
        "-show_streams",
        "-select_streams", "a",
        "-print_format", "json",
        filepath
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=20)
        data = json.loads(raw)
        return data.get("streams", [])
    except Exception:
        return []


def probe_audio_stream(filepath: str, ffprobe_bin: Optional[str] = None) -> bool:
    """Uses ffprobe to verify that the file contains at least one valid audio stream."""
    if not os.path.isfile(filepath) or os.path.getsize(filepath) < 500:
        return False
    streams = probe_audio_streams(filepath, ffprobe_bin)
    return len(streams) > 0


def format_eta(seconds: Optional[float]) -> str:
    """Formats seconds into human-readable text (e.g. '45s', '1m 24s', '2h 15m')."""
    if seconds is None:
        return "--"
    try:
        if math.isnan(seconds) or math.isinf(seconds) or seconds < 0 or seconds > 86400 * 365:
            return "--"
    except (TypeError, ValueError):
        return "--"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    minutes = s // 60
    rem_s = s % 60
    if minutes < 60:
        return f"{minutes}m {rem_s:02d}s"
    hours = minutes // 60
    rem_m = minutes % 60
    return f"{hours}h {rem_m:02d}m"


def clean_metadata_args(include_video: bool = True) -> list:
    """Returns FFmpeg command line arguments to strip and wipe all metadata."""
    args = [
        "-map_metadata", "-1",
        "-metadata", "title=",
        "-metadata", "comment=",
        "-metadata", "description=",
        "-metadata", "encoder=",
        "-metadata", "synopsis=",
        "-metadata", "genre=",
        "-metadata", "artist=",
        "-metadata", "album="
    ]
    if include_video:
        args.extend([
            "-metadata:s:v:0", "title=",
            "-metadata:s:v:0", "comment=",
            "-metadata:s:v:0", "description=",
            "-metadata:s:v:0", "handler_name="
        ])
    return args


def verify_clean_media_metadata(filepath: str, ffprobe_bin: Optional[str] = None) -> tuple[bool, list[str], str]:
    """Inspects output media file tags via ffprobe for forbidden markers."""
    from src.core.scene_renamer import find_forbidden_output_markers
    bin_path = resolve_ffprobe_bin(ffprobe_bin)
    try:
        raw = subprocess.check_output(
            [bin_path, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", filepath],
            text=True,
            errors="ignore",
            timeout=15,
        )
        markers = find_forbidden_output_markers(raw)
        return len(markers) == 0, markers, ""
    except Exception as exc:
        return False, [], str(exc)


def find_forbidden_media_metadata(filepath: str, ffprobe_bin: Optional[str] = None) -> list:
    """Checks output media file tags via ffprobe for any forbidden markers."""
    is_clean, violations, err = verify_clean_media_metadata(filepath, ffprobe_bin)
    if err:
        raise RuntimeError(f"Failed to inspect media metadata for {filepath}: {err}")
    return violations

