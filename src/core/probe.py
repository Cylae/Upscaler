"""FFprobe video metadata inspector and stream analyzer.

Accurately extracts resolution, rational frame rate, duration, audio streams,
and color space attributes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.models import calculate_display_dimensions, normalize_rotation, normalize_sample_aspect_ratio


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
    sample_aspect_ratio: str = "1:1"
    rotation: float = 0

    @property
    def display_width(self) -> float:
        return self._display_dimensions()[0]

    @property
    def display_height(self) -> float:
        return self._display_dimensions()[1]

    def _display_dimensions(self) -> Tuple[float, float]:
        return calculate_display_dimensions(
            self.width,
            self.height,
            sample_aspect_ratio=self.sample_aspect_ratio,
            rotation=self.rotation,
        )


# Keep complete probe documents together: one inspection covers video, all audio
# tracks, subtitles, display geometry, and the tags used by the privacy audit.
_PROBE_CACHE_LIMIT = 256
_PROBE_CACHE: OrderedDict[tuple, Dict] = OrderedDict()
_IN_FLIGHT_PROBES: Dict[tuple, Future] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_GENERATION = 0


def clear_probe_cache() -> None:
    """Clears cached snapshots without letting an older active probe refill them."""
    global _CACHE_GENERATION
    with _CACHE_LOCK:
        _PROBE_CACHE.clear()
        _IN_FLIGHT_PROBES.clear()
        _CACHE_GENERATION += 1


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


def _file_identity(filepath: str) -> Optional[tuple]:
    """Use one stat call and nanosecond timestamps, including replacement identity."""
    try:
        info = os.stat(filepath)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (
        os.path.normcase(os.path.abspath(filepath)),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _probe_media(filepath: str, ffprobe_bin: Optional[str], timeout: int) -> Dict:
    """Read a bounded, shared snapshot; never publish a file changed mid-probe."""
    bin_name = resolve_ffprobe_bin(ffprobe_bin)
    resolved_bin = shutil.which(bin_name) or bin_name
    bin_identity = _file_identity(resolved_bin) or os.path.normcase(os.path.abspath(resolved_bin))
    # A writer may finish during the first inspection. Retry it once after the
    # change instead of caching or validating a snapshot of the earlier file.
    for attempt in range(2):
        identity = _file_identity(filepath)
        cache_key = (identity, bin_identity) if identity is not None else None
        owner = False
        with _CACHE_LOCK:
            generation = _CACHE_GENERATION
            if cache_key is not None:
                cached = _PROBE_CACHE.get(cache_key)
                if cached is not None:
                    _PROBE_CACHE.move_to_end(cache_key)
                    return cached
                pending = _IN_FLIGHT_PROBES.get(cache_key)
                if pending is None:
                    pending = Future()
                    _IN_FLIGHT_PROBES[cache_key] = pending
                    owner = True
        if cache_key is None:
            return _read_probe_document(filepath, bin_name, timeout)
        if not owner:
            data = pending.result()
            if data is not None:
                return data
            continue
        try:
            data = _read_probe_document(filepath, bin_name, timeout)
            if _file_identity(filepath) != identity:
                data = None
            with _CACHE_LOCK:
                if data is not None and generation == _CACHE_GENERATION:
                    _PROBE_CACHE[cache_key] = data
                    _PROBE_CACHE.move_to_end(cache_key)
                    while len(_PROBE_CACHE) > _PROBE_CACHE_LIMIT:
                        _PROBE_CACHE.popitem(last=False)
                if _IN_FLIGHT_PROBES.get(cache_key) is pending:
                    del _IN_FLIGHT_PROBES[cache_key]
            pending.set_result(data)
            if data is not None:
                return data
        except BaseException as exc:
            with _CACHE_LOCK:
                if _IN_FLIGHT_PROBES.get(cache_key) is pending:
                    del _IN_FLIGHT_PROBES[cache_key]
            pending.set_exception(exc)
            raise
    raise RuntimeError(f"Media file changed while ffprobe was inspecting it: {filepath}")


def _read_probe_document(filepath: str, bin_name: str, timeout: int) -> Dict:
    cmd = [
        bin_name,
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        filepath
    ]
    raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=timeout)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid ffprobe response for {filepath}")
    return data


def probe_video(filepath: str, ffprobe_bin: Optional[str] = None) -> VideoMetadata:
    """Extract complete video metadata, sharing fresh bounded snapshots with audits."""
    data = _probe_media(filepath, ffprobe_bin, timeout=30)

    v_stream = next((
        s for s in data.get("streams", [])
        if s.get("codec_type") == "video"
        and (s.get("disposition") or {}).get("attached_pic", 0) not in (1, "1", True)
    ), None)
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
    sample_aspect_ratio = normalize_sample_aspect_ratio(v_stream.get("sample_aspect_ratio", "1:1"))
    rotation = normalize_rotation((v_stream.get("tags") or {}).get("rotate", 0))
    for side_data in v_stream.get("side_data_list") or []:
        if "rotation" not in side_data:
            continue
        try:
            side_rotation = float(side_data["rotation"])
            if math.isfinite(side_rotation):
                rotation = side_rotation
                break
        except (ValueError, TypeError, OverflowError):
            continue

    meta = VideoMetadata(
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
        audio_stream_count=len(a_streams),
        sample_aspect_ratio=sample_aspect_ratio,
        rotation=rotation,
    )

    return meta


def probe_audio_streams(filepath: str, ffprobe_bin: Optional[str] = None) -> List[Dict]:
    """Return every audio track without exposing mutable cached stream dictionaries."""
    try:
        data = _probe_media(filepath, ffprobe_bin, timeout=20)
        return deepcopy([stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"])
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
    try:
        data = _probe_media(filepath, ffprobe_bin, timeout=15)
        tags = {
            "format": {"tags": data.get("format", {}).get("tags", {})},
            "streams": [{"tags": stream.get("tags", {})} for stream in data.get("streams", [])],
        }
        markers = find_forbidden_output_markers(json.dumps(tags, ensure_ascii=False))
        return len(markers) == 0, markers, ""
    except Exception as exc:
        return False, [], str(exc)


def find_forbidden_media_metadata(filepath: str, ffprobe_bin: Optional[str] = None) -> list:
    """Checks output media file tags via ffprobe for any forbidden markers."""
    is_clean, violations, err = verify_clean_media_metadata(filepath, ffprobe_bin)
    if err:
        raise RuntimeError(f"Failed to inspect media metadata for {filepath}: {err}")
    return violations

