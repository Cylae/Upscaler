import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

_active_job_dirs_lock = threading.Lock()
_active_job_dirs: set = set()


def register_active_job_dir(work_dir: str, job_info: Optional[Dict] = None) -> str:
    """Registers an active work directory and writes a PID lease lock file."""
    norm_path = os.path.normcase(os.path.abspath(work_dir))
    with _active_job_dirs_lock:
        _active_job_dirs.add(norm_path)

    os.makedirs(work_dir, exist_ok=True)
    lock_file = os.path.join(work_dir, ".job_lock")
    payload = {
        "pid": os.getpid(),
        "created_at": time.time(),
        "last_heartbeat": time.time(),
        "info": job_info or {}
    }
    try:
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
    return lock_file


def unregister_active_job_dir(work_dir: str) -> None:
    """Unregisters an active work directory."""
    norm_path = os.path.normcase(os.path.abspath(work_dir))
    with _active_job_dirs_lock:
        _active_job_dirs.discard(norm_path)


def is_process_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is actively running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            STILL_ACTIVE = 259
            return exit_code.value == STILL_ACTIVE
        except Exception:
            return True  # Be conservative: assume alive if error occurs
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

from src.core.ai_models import (
    AVAILABLE_MODELS,
    AIModelSpec,
    ModelEngine,
    calculate_target_dimensions,
    choose_best_model_for_video
)
from src.core.detector import HardwareInfo, get_hardware_info
from src.core.scene_renamer import (
    SceneNamingConfig,
    find_forbidden_output_markers,
    format_scene_name,
    get_scene_relative_output_path
)


def format_eta(seconds: Optional[float]) -> str:
    """Formats a duration in seconds into clean human-readable text (e.g. '45s', '1m 24s', '2h 15m')."""
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
    remaining_secs = s % 60
    if minutes < 60:
        return f"{minutes}m {remaining_secs:02d}s"
    hours = minutes // 60
    remaining_mins = minutes % 60
    return f"{hours}h {remaining_mins:02d}m"


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


def verify_clean_media_metadata(filepath: str, ffprobe_bin: Optional[str] = None) -> Tuple[bool, List[str], str]:
    """Inspects output media file tags via ffprobe for forbidden markers.

    Returns (is_clean, violations, error_message).
    is_clean is True only if inspection succeeded and no forbidden markers were found.
    """
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
    """Checks output media file tags via ffprobe for any forbidden markers.

    Raises RuntimeError if ffprobe inspection fails, ensuring verification errors
    are never silently treated as clean output.
    """
    is_clean, violations, err = verify_clean_media_metadata(filepath, ffprobe_bin)
    if err:
        raise RuntimeError(f"Failed to inspect media metadata for {filepath}: {err}")
    return violations


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


@dataclass
class UpscaleJobConfig:
    input_file: str
    output_dir: str
    base_dir: Optional[str] = None
    target_preset: str = "4k"  # "4k", "1440p", "1080p", "720p"
    video_codec: str = "hevc_nvenc"  # "hevc_nvenc", "h264_nvenc", "libx265", "libx264"
    model_id: str = "auto"
    scene_config: Optional[SceneNamingConfig] = None
    preserve_folders: bool = True
    overwrite: bool = False
    temp_dir: Optional[str] = None
    tile_size: int = 0  # 0 for auto
    threads: str = "auto"  # "auto", "4:4:4", "2:4:2", "1:2:2"
    frame_format: str = "jpg"  # "jpg" (fastest I/O) or "png" (lossless)
    cq_level: int = 19
    nvenc_preset: str = "p7"
    enable_tta: bool = False
    denoise_level: int = -1
    audio_mode: str = "copy"  # "copy", "aac_192k", "aac_320k", "flac"
    log_callback: Optional[Callable[[str], None]] = None
    # When set, only `preview_duration_s` seconds are rendered (starting 10% into the
    # video so the clip is representative rather than a cold open).  The output file
    # receives a "[Preview Xsec]" suffix and never overwrites a full-length encode.
    preview_duration_s: Optional[float] = None


def validate_job_config(job: UpscaleJobConfig) -> None:
    """Fails fast on invalid programmatic jobs before creating temporary files."""
    if not os.path.isfile(job.input_file):
        raise ValueError(f"Input video does not exist: {job.input_file}")
    if not job.output_dir:
        raise ValueError("Output directory is required.")
    if job.target_preset not in {"4k", "1440p", "1080p", "720p"}:
        raise ValueError(f"Unsupported target resolution: {job.target_preset}")
    if job.video_codec not in {"hevc_nvenc", "h264_nvenc", "libx265", "libx264"}:
        raise ValueError(f"Unsupported video codec: {job.video_codec}")
    if job.model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown AI model: {job.model_id}")
    if job.frame_format.lower() not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported frame format: {job.frame_format}")
    if not 14 <= job.cq_level <= 28:
        raise ValueError("CQ level must be between 14 and 28.")
    if job.nvenc_preset not in {f"p{index}" for index in range(1, 8)}:
        raise ValueError(f"Unsupported NVENC preset: {job.nvenc_preset}")
    if job.denoise_level not in {-1, 0, 1, 2, 3}:
        raise ValueError("RealCUGAN denoise level must be between -1 and 3.")
    if job.audio_mode not in {"copy", "aac_192k", "aac_320k", "flac"}:
        raise ValueError(f"Unsupported audio mode: {job.audio_mode}")
    if job.tile_size != 0 and job.tile_size < 32:
        raise ValueError("Tile size must be 0 (auto) or at least 32 pixels.")
    if job.threads != "auto":
        match = re.fullmatch(r"(\d+):(\d+):(\d+)", job.threads)
        if not match or any(int(value) < 1 for value in match.groups()):
            raise ValueError("Threads must be 'auto' or LOAD:PROC:SAVE with positive integers.")
    if job.preview_duration_s is not None and job.preview_duration_s <= 0:
        raise ValueError("Preview duration must be greater than zero.")


def get_optimal_thread_spec(threads_param: str = "auto", gpu_vram_gb: float = 0.0) -> str:
    """Calculates optimal load:proc:save thread count based on CPU cores and GPU VRAM.

    Higher GPU VRAM allows more parallel processing threads in the AI upscaler,
    reducing per-frame latency significantly on modern GPUs.
    """
    if threads_param and threads_param != "auto":
        return threads_param
    cpu_count = os.cpu_count() or 4

    # Determine proc threads based on available VRAM — more VRAM = more parallelism
    if gpu_vram_gb >= 10.0:
        proc = 8
    elif gpu_vram_gb >= 6.0:
        proc = 6
    elif gpu_vram_gb >= 4.0:
        proc = 4
    else:
        proc = 4  # Safe default

    # Determine load/save threads based on CPU core count
    if cpu_count >= 16:
        load_save = 4
    elif cpu_count >= 8:
        load_save = 3
    elif cpu_count >= 4:
        load_save = 2
    else:
        load_save = 1

    return f"{load_save}:{proc}:{load_save}"


def _auto_tile_size(gpu_vram_gb: float) -> int:
    """Computes optimal tile_size for ncnn-vulkan upscalers based on available GPU VRAM.

    Larger tiles reduce kernel-launch overhead but require proportionally more VRAM.
    Over-sized tiles cause GPU OOM and fall back to software, which is catastrophically slow.
    """
    if gpu_vram_gb >= 10.0:
        return 0    # No tiling needed — entire frame fits in VRAM
    elif gpu_vram_gb >= 6.0:
        return 400  # Large tiles, minimal overhead
    elif gpu_vram_gb >= 4.0:
        return 200  # Medium tiles
    elif gpu_vram_gb >= 2.0:
        return 100  # Small tiles for low-VRAM GPUs
    else:
        return 64   # Conservative fallback (integrated / very old GPU)


def resolve_realcugan_options(models_dir: str, requested_scale: int, requested_denoise: int) -> Tuple[int, int, str]:
    """Resolves a RealCUGAN request against the weights actually installed.

    RealCUGAN Pro distributions are often intentionally sparse (this bundle has
    2x/3x and denoise -1/0/3 only). Selecting a missing variant makes the native
    binary fail late and leaves users with an opaque exit code. This resolver
    chooses the nearest available variant and returns a user-facing explanation.
    """
    available: Dict[int, List[int]] = {}
    noise_names = {-1: "no-denoise", 0: "conservative", 1: "denoise1x", 2: "denoise2x", 3: "denoise3x"}
    for scale in (1, 2, 3, 4):
        levels = [
            level for level, name in noise_names.items()
            if os.path.isfile(os.path.join(models_dir, f"up{scale}x-{name}.param"))
            and os.path.isfile(os.path.join(models_dir, f"up{scale}x-{name}.bin"))
        ]
        if levels:
            available[scale] = levels

    if not available:
        raise ValueError(f"No valid RealCUGAN weights found in {models_dir}")

    scale = requested_scale if requested_scale in available else min(available, key=lambda value: abs(value - requested_scale))
    levels = available[scale]
    denoise = requested_denoise if requested_denoise in levels else min(levels, key=lambda value: abs(value - requested_denoise))

    changes = []
    if scale != requested_scale:
        changes.append(f"scale {requested_scale}x → {scale}x")
    if denoise != requested_denoise:
        changes.append(f"denoise {requested_denoise} → {denoise}")
    return scale, denoise, "; ".join(changes)


def resolve_ffprobe_bin(ffprobe_bin: Optional[str] = None) -> str:
    """Finds ffprobe binary in environment or adjacent to ffmpeg."""
    if ffprobe_bin and (os.path.isfile(ffprobe_bin) or shutil.which(ffprobe_bin)):
        return ffprobe_bin
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg_found = shutil.which("ffmpeg")
    if ffmpeg_found:
        candidate = os.path.join(os.path.dirname(ffmpeg_found), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(candidate):
            return candidate
    return "ffprobe"


def probe_video(filepath: str, ffprobe_bin: Optional[str] = None) -> VideoMetadata:
    """Uses ffprobe to extract complete video metadata."""
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
        try:
            numerator, denominator = str(fps_str).split("/", 1)
            if float(denominator) > 0:
                fps = float(numerator) / float(denominator)
                if fps > 0:
                    break
        except (TypeError, ValueError):
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


def probe_audio_stream(filepath: str, ffprobe_bin: Optional[str] = None) -> bool:
    """Uses ffprobe to verify that the file contains at least one valid audio stream."""
    if not os.path.isfile(filepath) or os.path.getsize(filepath) < 500:
        return False
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
        raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=15)
        data = json.loads(raw)
        streams = data.get("streams", [])
        return len(streams) > 0
    except Exception:
        return False


def is_file_complete_and_valid(
    filepath: str,
    expect_audio: bool = False,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_duration: Optional[float] = None,
    duration_tolerance_s: float = 2.0,
    duration_tolerance_ratio: float = 0.02,
) -> bool:
    """Checks if an already existing output file is valid and complete, matching the expected job contract.
    
    Verifies:
      - File exists and is > 100 KB
      - Valid video stream with positive dimensions and duration
      - Resolution matches expected_width and expected_height (if specified)
      - Duration matches expected_duration within tolerance (detects truncated/incomplete renders)
      - Audio stream presence matches expect_audio
    """
    if not os.path.isfile(filepath):
        return False
    if os.path.getsize(filepath) < 100_000:  # < 100KB is incomplete
        return False
    try:
        meta = probe_video(filepath)
        if meta.duration <= 0.5 or meta.width <= 0 or meta.height <= 0:
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


def safe_atomic_move(src: str, dst: str) -> None:
    """Atomically moves src to dst, with safe fallback for cross-filesystem moves.

    If src and dst live on different volumes/drives (WinError 17 on Windows or
    EXDEV errno 18 on POSIX), copies to a temporary file on the destination
    directory first, then performs atomic replace.
    """
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 17 or getattr(exc, "errno", None) == 18:
            temp_dst = os.path.join(dst_dir, f".moving_{uuid.uuid4().hex[:8]}_{os.path.basename(dst)}")
            try:
                shutil.copy2(src, temp_dst)
                os.replace(temp_dst, dst)
                try:
                    os.remove(src)
                except OSError:
                    pass
            finally:
                if os.path.exists(temp_dst):
                    try:
                        os.remove(temp_dst)
                    except OSError:
                        pass
        else:
            raise


def validate_encoded_video(
    filepath: str,
    expected_width: int,
    expected_height: int,
    expect_audio: bool = False,
    metadata: Optional[VideoMetadata] = None,
) -> Tuple[bool, str]:
    """Validates that FFmpeg produced a playable video, not just a non-empty file.

    This is deliberately stricter than a size check: an audio-only MP4 can be a
    few megabytes large but is not a successful video render.
    """
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        return False, "Encoder did not create an output file."
    if metadata is None:
        try:
            meta = probe_video(filepath)
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


def _probe_cuda_support() -> bool:
    """Returns True if FFmpeg was compiled with CUDA/cuvid hwaccel support."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
            timeout=10,
        )
        return "cuda" in out.lower() or "cuvid" in out.lower()
    except Exception:
        return False


class VideoUpscalerPipeline:
    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or get_hardware_info()
        self._is_cancelled = False
        self._current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()
        # Detect CUDA hwaccel once at construction — avoids repeated subprocess calls per job
        self._cuda_available: Optional[bool] = None

    def cancel(self):
        """Signals cancellation and terminates running subprocesses cleanly."""
        self._is_cancelled = True
        with self._process_lock:
            processes = list(self._active_processes)
        for proc in processes:
            try:
                if proc.poll() is None:
                    if os.name == "nt" and proc.pid:
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=3
                            )
                        except Exception:
                            pass
                    proc.kill()
            except Exception:
                pass

    def _start_process(self, cmd: list, **popen_kwargs) -> subprocess.Popen:
        """Starts and registers a child process so cancellation reaches every stage."""
        if self._is_cancelled:
            raise RuntimeError("Operation cancelled by user.")
        if os.name == "nt":
            creationflags = popen_kwargs.get("creationflags", 0)
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
            popen_kwargs["creationflags"] = creationflags
        proc = subprocess.Popen(cmd, **popen_kwargs)
        with self._process_lock:
            self._active_processes.add(proc)
        return proc

    def _forget_process(self, proc: subprocess.Popen) -> None:
        with self._process_lock:
            self._active_processes.discard(proc)

    def _wait_for_process(self) -> int:
        """Waits for _current_process to finish, killing it immediately if cancelled."""
        proc = self._current_process
        if proc is None:
            return 0
        while True:
            try:
                proc.wait(timeout=0.05)
                return proc.returncode
            except subprocess.TimeoutExpired:
                if self._is_cancelled:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.wait()
                    return -1

    def _run_tracked(self, cmd: list, **popen_kwargs) -> int:
        """Launches a subprocess, tracks it in _current_process, waits with cancel support."""
        proc = self._start_process(cmd, **popen_kwargs)
        self._current_process = proc
        try:
            return self._wait_for_process()
        finally:
            self._forget_process(proc)
            if self._current_process is proc:
                self._current_process = None

    def reset_cancel(self):
        self._is_cancelled = False
        self._current_process = None

    def _has_cuda(self) -> bool:
        """Returns CUDA hwaccel availability, caching the result after the first check."""
        if self._cuda_available is None:
            self._cuda_available = self.hardware.has_nvidia and _probe_cuda_support()
        return self._cuda_available

    def _build_extract_frames_cmd(
        self,
        input_file: str,
        frames_in_dir: str,
        frame_ext: str,
        use_hwdecode: bool,
        start_offset_s: float = 0.0,
        duration_s: Optional[float] = None,
    ) -> list:
        """Builds an optimised FFmpeg command to extract video frames.

        When use_hwdecode is True and a CUDA-capable GPU is detected, the video
        is decoded entirely on the GPU (cuvid / NVDEC) which is typically 2–4x
        faster than CPU decoding and frees CPU threads for other pipeline stages.
        Falls back to the optimised CPU path otherwise.

        Args:
            start_offset_s: Seek to this position before decoding (preview mode).
                            Placed *before* -i to benefit from fast stream-copy seek.
            duration_s:     If set, only decode this many seconds (preview mode).

        Additional flags applied to both paths:
          -fflags +fastseek         — seeks by keyframe rather than byte scan
          -probesize 32M            — larger probe avoids false-positive format errors
          -analyzeduration 10M      — analyse more of the stream before decoding
          -filter_threads 0         — use all logical CPU cores for filter graph
          -filter_complex_threads 0 — same for complex filter graphs
        """
        common_input_flags = [
            "-fflags", "+fastseek",
            "-probesize", "32M",
            "-analyzeduration", "10M",
        ]

        # Pre-input seek (fast keyframe seek — accurate enough for preview)
        seek_flags = []
        if start_offset_s > 0:
            seek_flags = ["-ss", f"{start_offset_s:.3f}"]

        if use_hwdecode:
            cmd = [
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
            ] + common_input_flags + seek_flags + [
                "-i", input_file,
                "-vf", "hwdownload,format=nv12,format=yuvj420p",
                "-filter_threads", "0",
                "-filter_complex_threads", "0",
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-threads", "0",
            ] + common_input_flags + seek_flags + [
                "-i", input_file,
                "-filter_threads", "0",
                "-filter_complex_threads", "0",
            ]

        if duration_s is not None:
            cmd.extend(["-t", f"{duration_s:.3f}"])

        if frame_ext == "jpg":
            cmd.extend(["-qscale:v", "1", "-qmin", "1"])

        cmd.append(os.path.join(frames_in_dir, f"frame_%08d.{frame_ext}"))
        return cmd

    def _build_encode_cmd(
        self,
        frames_out_dir: str,
        frame_ext: str,
        meta: VideoMetadata,
        target_w: int,
        target_h: int,
        chosen_encoder: str,
        cq_val: str,
        p_val: str,
        audio_input_path: Optional[str],
        audio_mode: str,
        part_output_file: str,
        audio_start_offset_s: float = 0.0,
        audio_duration_s: Optional[float] = None,
    ) -> list:
        """Builds an optimised FFmpeg command to re-encode upscaled frames to the target format.

        Improvements over the previous version:
          - Uses -r instead of the deprecated -framerate flag for frame rate injection.
          - Adds -movflags +faststart so the moov atom is written at the start of
            the file — enables progressive playback and faster seeks in media players.
          - Adds -rc-lookahead 32 and -surfaces 64 for NVENC encoders to improve
            B-frame prediction quality without any notable speed cost.
          - CPU fallback presets changed from 'medium' to 'fast': roughly 2x faster
            with negligible (< 1 dB PSNR) quality difference at the same CRF.
          - filter_threads / filter_complex_threads set to 0 for maximum CPU usage
            during the Lanczos scale filter.
        """
        fps_arg = meta.r_frame_rate if (meta.r_frame_rate and "/" in meta.r_frame_rate and meta.r_frame_rate != "0/0") else str(meta.fps)
        cmd = [
            "ffmpeg", "-y",
            "-threads", "0",
            "-r", fps_arg,
            "-i", os.path.join(frames_out_dir, f"frame_%08d.{frame_ext}"),
        ]

        if audio_input_path:
            # A direct source-audio fallback must use the preview window too;
            # otherwise a preview that starts mid-episode gets audio from 0:00.
            if audio_start_offset_s > 0:
                cmd.extend(["-ss", f"{audio_start_offset_s:.3f}"])
            cmd.extend(["-i", audio_input_path])
            if audio_duration_s is not None:
                cmd.extend(["-t", f"{audio_duration_s:.3f}"])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        else:
            cmd.extend(["-map", "0:v:0"])

        # High-quality Lanczos resize with all CPU cores assigned
        cmd.extend([
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-filter_threads", "0",
            "-filter_complex_threads", "0",
        ])

        # Colorspace and dynamic range preservation
        if meta.color_range and meta.color_range in ("tv", "pc", "limited", "full"):
            cmd.extend(["-color_range", meta.color_range])
        if meta.color_space and meta.color_space not in ("unknown", "unspecified"):
            cmd.extend(["-colorspace", meta.color_space])
        if meta.color_primaries and meta.color_primaries not in ("unknown", "unspecified"):
            cmd.extend(["-color_primaries", meta.color_primaries])
        if meta.color_transfer and meta.color_transfer not in ("unknown", "unspecified"):
            cmd.extend(["-color_trc", meta.color_transfer])

        # Encoder-specific parameters
        if chosen_encoder == "hevc_nvenc":
            cmd.extend([
                "-c:v", "hevc_nvenc",
                "-preset", p_val,
                "-cq", cq_val,
                "-rc-lookahead", "32",
                "-surfaces", "64",
                "-spatial-aq", "1",
                "-temporal-aq", "1",
                "-b_ref_mode", "middle",
                "-pix_fmt", "yuv420p",
            ])
        elif chosen_encoder == "h264_nvenc":
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-preset", p_val,
                "-cq", cq_val,
                "-rc-lookahead", "32",
                "-surfaces", "64",
                "-spatial-aq", "1",
                "-temporal-aq", "1",
                "-b_ref_mode", "middle",
                "-pix_fmt", "yuv420p",
            ])
        elif chosen_encoder == "libx265":
            # 'fast' is ~2x quicker than 'medium' with < 1 dB PSNR difference at same CRF
            cmd.extend([
                "-c:v", "libx265",
                "-crf", cq_val,
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
            ])
        else:  # libx264
            cmd.extend([
                "-c:v", "libx264",
                "-crf", cq_val,
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
            ])

        # Audio codec selection
        if audio_input_path:
            if audio_mode == "aac_192k":
                cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            elif audio_mode == "aac_320k":
                cmd.extend(["-c:a", "aac", "-b:a", "320k"])
            elif audio_mode == "flac":
                cmd.extend(["-c:a", "flac"])
            else:  # "copy"
                cmd.extend(["-c:a", "copy"])
            cmd.append("-shortest")

        # movflags: faststart puts moov atom at file head for streaming-friendly output
        cmd.extend(["-movflags", "+faststart"])

        # Strip all metadata
        cmd.extend(clean_metadata_args(include_video=True))

        cmd.append(part_output_file)
        return cmd

    def process_video(
        self,
        job: UpscaleJobConfig,
        progress_callback: Optional[Callable[[float, str, Dict], None]] = None
    ) -> Tuple[bool, str]:
        """
        Executes end-to-end upscale pipeline:
        1. Probe input video.
        2. Resolve model & target resolution.
        3. Check if output already exists (resume feature).
        4. Extract audio & frames in parallel (threaded).
        5. Run upscaling engine with progress tracking.
        6. Encode to target resolution with NVENC (or CPU fallback).
        7. Atomic rename & cleanup.

        Key optimisations applied vs the previous sequential pipeline:
        - Audio extraction and frame extraction run concurrently on separate threads,
          saving the full duration of audio extraction from the critical path.
        - Frame extraction uses GPU CUDA hwdecode when available (2–4x faster).
        - tile_size is automatically computed from GPU VRAM if not set by the user.
        - Thread spec for the AI upscaler accounts for GPU VRAM capacity.
        - FFmpeg encode command uses rc-lookahead, surfaces, movflags+faststart,
          and CPU fallback preset 'fast' instead of 'medium'.
        """
        self.reset_cancel()
        t_file_start = time.time()

        try:
            validate_job_config(job)
        except ValueError as exc:
            return False, f"Invalid job configuration: {exc}"

        # Preview mode: limit clip to `preview_duration_s` seconds, starting at 10 %
        # of the total duration so the clip is representative, not a cold open.
        # The offset is resolved after probing so we can compute 10 % of duration.
        _preview_mode = job.preview_duration_s is not None and job.preview_duration_s > 0

        def log_cmd(msg: str):
            if job.log_callback:
                try:
                    job.log_callback(msg)
                except Exception:
                    pass

        def report(ratio: float, status: str, extra: Optional[Dict] = None):
            if progress_callback:
                progress_callback(ratio, status, extra or {})

        report(0.01, "Analyzing source video...")

        # 1. Probe input video
        try:
            meta = probe_video(job.input_file)
        except Exception as e:
            return False, f"Video analysis error: {e}"

        # 2. Resolve target dimensions & scene naming
        target_w, target_h, res_tag = calculate_target_dimensions(meta.width, meta.height, job.target_preset)

        # Every concurrent job gets an isolated naming configuration. Mutating
        # the shared batch object here caused cross-worker naming races.
        scene_cfg = replace(job.scene_config) if job.scene_config else SceneNamingConfig()
        scene_cfg.resolution_tag = res_tag

        # Set codec tag in scene name according to chosen encoder
        if "hevc" in job.video_codec or "x265" in job.video_codec:
            scene_cfg.codec_tag = "H.265"
        else:
            scene_cfg.codec_tag = "H.264"

        rel_out = get_scene_relative_output_path(
            job.input_file,
            job.base_dir,
            scene_cfg,
            preserve_folders=job.preserve_folders
        )
        final_output_path = os.path.join(job.output_dir, rel_out)

        # In preview mode, suffix the filename so it never overwrites the full encode.
        if _preview_mode:
            dur_label = int(job.preview_duration_s)  # type: ignore[arg-type]
            stem, ext = os.path.splitext(final_output_path)
            final_output_path = f"{stem} [Preview {dur_label}sec]{ext}"

        out_folder = os.path.dirname(final_output_path)
        os.makedirs(out_folder, exist_ok=True)

        # 3. Check resume (already completed?) — skipped in preview mode, always re-render
        if not _preview_mode and not job.overwrite and is_file_complete_and_valid(
            final_output_path,
            expect_audio=meta.has_audio,
            expected_width=target_w,
            expected_height=target_h,
            expected_duration=meta.duration,
        ):
            report(1.0, "Episode already present (Skipped)", {"output_file": final_output_path, "skipped": True})
            return True, f"Already completed: {os.path.basename(final_output_path)}"

        # 4. Resolve Model
        if job.model_id == "auto" or job.model_id not in AVAILABLE_MODELS:
            selected_model, chosen_scale = choose_best_model_for_video(
                meta.width, meta.height, job.target_preset
            )
        else:
            selected_model = AVAILABLE_MODELS[job.model_id]
            chosen_scale = selected_model.default_scale

        report(0.05, f"Selected model: {selected_model.name} (Scale {chosen_scale}x)")

        # Prepare workspace & temp directory
        root_temp = job.temp_dir or os.path.join(job.output_dir, ".temp_upscaler")
        # UUID makes concurrent/restarted jobs collision-proof, even for duplicate
        # source names started within the same clock tick.
        job_id = f"job_{uuid.uuid4().hex}"
        work_dir = os.path.join(root_temp, job_id)
        frames_in_dir = os.path.join(work_dir, "frames_in")
        frames_out_dir = os.path.join(work_dir, "frames_out")
        audio_file = os.path.join(work_dir, "audio.m4a")
        part_output_file = os.path.join(work_dir, "output_part" + scene_cfg.container)

        os.makedirs(frames_in_dir, exist_ok=True)
        os.makedirs(frames_out_dir, exist_ok=True)
        register_active_job_dir(work_dir)

        try:
            frame_ext = "jpg" if job.frame_format.lower() in ("jpg", "jpeg") else "png"

            # Compute preview segment boundaries once (needs meta.duration)
            _preview_start_s: float = 0.0
            _preview_dur_s: Optional[float] = None
            if _preview_mode:
                _preview_dur_s = float(job.preview_duration_s)  # type: ignore[arg-type]
                if meta.duration and meta.duration > 0:
                    _preview_dur_s = min(_preview_dur_s, meta.duration)
                    # Start at ~10 % into the video (avoids cold opens / recaps)
                    _preview_start_s = max(0.0, meta.duration * 0.10)
                    # Clamp: don't start so late that we'd run past the end
                    max_start = max(0.0, meta.duration - _preview_dur_s - 1.0)
                    _preview_start_s = min(_preview_start_s, max_start)
                else:
                    _preview_start_s = 0.0
                log_cmd(
                    f"[PREVIEW] Rendering {_preview_dur_s:.0f}s clip starting at "
                    f"{format_eta(_preview_start_s)} "
                    f"(total duration: {format_eta(meta.duration)})"
                )

            # ------------------------------------------------------------------
            # 5. Parallel audio + frame extraction
            #    Both operations are I/O bound and independent; running them on
            #    separate threads removes the audio extraction time from the
            #    critical path entirely (typically 5–20 s on a 24-min episode).
            # ------------------------------------------------------------------

            # Shared results written by threads
            _audio_result: Dict = {"extracted": False, "error": None}
            _frames_result: Dict = {"extracted": 0, "error": None}
            _audio_proc: Dict = {"proc": None}

            def _extract_audio_thread():
                """Extracts audio stream from source; first tries copy, then AAC transcode.

                In preview mode the same time window as the video extraction is used
                so audio and video are always in sync in the preview clip.
                """
                if not meta.has_audio:
                    return

                audio_cmd = ["ffmpeg", "-y"]
                if _preview_start_s > 0:
                    audio_cmd.extend(["-ss", f"{_preview_start_s:.3f}"])
                audio_cmd.extend(["-i", job.input_file, "-vn", "-map", "0:a:0"])
                if _preview_dur_s is not None:
                    audio_cmd.extend(["-t", f"{_preview_dur_s:.3f}"])
                audio_cmd.extend(["-c:a", "copy", audio_file])
                log_cmd(f"[FFMPEG AUDIO] {' '.join(audio_cmd)}")
                try:
                    proc = self._start_process(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    _audio_proc["proc"] = proc
                    try:
                        proc.wait()
                    finally:
                        self._forget_process(proc)
                except Exception as exc:
                    _audio_result["error"] = str(exc)
                    return

                if self._is_cancelled:
                    return

                if probe_audio_stream(audio_file):
                    _audio_result["extracted"] = True
                    return

                # Fallback: transcode to AAC when direct copy fails (e.g. PCM / Vorbis)
                audio_cmd = ["ffmpeg", "-y"]
                if _preview_start_s > 0:
                    audio_cmd.extend(["-ss", f"{_preview_start_s:.3f}"])
                audio_cmd.extend(["-i", job.input_file, "-vn", "-map", "0:a:0"])
                if _preview_dur_s is not None:
                    audio_cmd.extend(["-t", f"{_preview_dur_s:.3f}"])
                audio_cmd.extend(["-c:a", "aac", "-b:a", "192k", audio_file])
                log_cmd(f"[FFMPEG AUDIO FALLBACK] {' '.join(audio_cmd)}")
                try:
                    proc = self._start_process(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    _audio_proc["proc"] = proc
                    try:
                        proc.wait()
                    finally:
                        self._forget_process(proc)
                except Exception as exc:
                    _audio_result["error"] = str(exc)
                    return

                _audio_result["extracted"] = probe_audio_stream(audio_file)

            def _extract_frames_thread():
                """Extracts video frames; tries GPU hwdecode first, falls back to CPU.

                In preview mode only `_preview_dur_s` seconds starting at
                `_preview_start_s` are decoded.
                """
                use_hw = self._has_cuda()
                extract_cmd = self._build_extract_frames_cmd(
                    job.input_file, frames_in_dir, frame_ext,
                    use_hwdecode=use_hw,
                    start_offset_s=_preview_start_s,
                    duration_s=_preview_dur_s,
                )
                log_cmd(f"[FFMPEG EXTRACT ({'CUDA' if use_hw else 'CPU'})] {' '.join(extract_cmd)}")
                try:
                    proc = self._start_process(
                        extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        text=True, errors="ignore"
                    )
                    try:
                        _, stderr_out = proc.communicate()
                    finally:
                        self._forget_process(proc)
                    rc = proc.returncode
                except Exception as exc:
                    _frames_result["error"] = str(exc)
                    return

                # If CUDA hwdecode failed (e.g. unsupported codec), retry with CPU path
                if rc != 0 and use_hw:
                    log_cmd("[FFMPEG EXTRACT] CUDA hwdecode failed, retrying with CPU decoder...")
                    self._cuda_available = False  # Disable for subsequent jobs in this session
                    extract_cmd = self._build_extract_frames_cmd(
                        job.input_file, frames_in_dir, frame_ext,
                        use_hwdecode=False,
                        start_offset_s=_preview_start_s,
                        duration_s=_preview_dur_s,
                    )
                    log_cmd(f"[FFMPEG EXTRACT (CPU)] {' '.join(extract_cmd)}")
                    # Clear any partial frames before retry
                    for f in os.listdir(frames_in_dir):
                        try:
                            os.remove(os.path.join(frames_in_dir, f))
                        except OSError:
                            pass
                    try:
                        proc = self._start_process(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        try:
                            proc.wait()
                        finally:
                            self._forget_process(proc)
                        rc = proc.returncode
                    except Exception as exc:
                        _frames_result["error"] = str(exc)
                        return

                if rc != 0:
                    _frames_result["error"] = f"FFmpeg frame extraction failed (exit code {rc})"
                    return

                ff = [f for f in os.listdir(frames_in_dir) if f.startswith("frame_") and f.endswith(f".{frame_ext}")]
                _frames_result["extracted"] = len(ff)

            report(0.08, "Extracting audio and frames in parallel...")

            audio_thread = threading.Thread(target=_extract_audio_thread, daemon=True)
            frames_thread = threading.Thread(target=_extract_frames_thread, daemon=True)

            audio_thread.start()
            frames_thread.start()

            # Poll for cancellation while waiting for both threads to complete
            while audio_thread.is_alive() or frames_thread.is_alive():
                if self._is_cancelled:
                    # Kill audio subprocess if running
                    ap = _audio_proc.get("proc")
                    if ap:
                        try:
                            ap.kill()
                        except Exception:
                            pass
                    return False, "Cancelled by user."
                time.sleep(0.1)

            audio_thread.join()
            frames_thread.join()

            if self._is_cancelled:
                return False, "Cancelled by user."

            # Interpret thread results
            has_extracted_audio = _audio_result["extracted"]

            if meta.has_audio:
                if has_extracted_audio:
                    log_cmd(f"[AUDIO] Source soundtrack extracted and verified ({meta.audio_codec.upper() if meta.audio_codec else 'AAC'})")
                else:
                    log_cmd("[AUDIO WARNING] Intermediate audio extraction failed, will mux directly from source video.")

            total_extracted = _frames_result["extracted"]
            if total_extracted == 0:
                return False, "No frames extracted from source video."

            log_cmd(f"[FRAMES] {total_extracted} frames extracted.")

            # ------------------------------------------------------------------
            # 6. Compute optimised AI upscaler parameters
            # ------------------------------------------------------------------
            thread_spec = get_optimal_thread_spec(job.threads, gpu_vram_gb=self.hardware.gpu_vram_gb)

            # Resolve tile_size: respect user override, otherwise auto-compute from VRAM
            effective_tile_size = job.tile_size if job.tile_size > 0 else _auto_tile_size(self.hardware.gpu_vram_gb)

            mode_label = f" — Preview {int(_preview_dur_s)}s" if _preview_mode and _preview_dur_s else ""
            report(0.20, f"Upscaling with {selected_model.name} (Threads: {thread_spec}, Tile: {effective_tile_size or 'auto'}){mode_label}...")
            t_start_ai = time.time()

            # ------------------------------------------------------------------
            # 7. Run Upscaler (Tuned Multithreading -j load:proc:save)
            # ------------------------------------------------------------------
            if selected_model.engine == ModelEngine.REALESRGAN:
                bin_path = self.hardware.realesrgan_bin
                if not bin_path or not os.path.isfile(bin_path):
                    return False, f"Real-ESRGAN binary not found: {bin_path}"

                models_dir = os.path.join(os.path.dirname(bin_path), "models")
                if not os.path.isdir(models_dir):
                    return False, f"Local Real-ESRGAN models directory not found: {models_dir}"

                ai_cmd = [
                    bin_path,
                    "-i", frames_in_dir,
                    "-o", frames_out_dir,
                    "-m", models_dir,
                    "-n", selected_model.internal_name,
                    "-s", str(chosen_scale),
                    "-j", thread_spec,
                    "-f", frame_ext,
                    "-v"
                ]
                if effective_tile_size > 0:
                    ai_cmd.extend(["-t", str(effective_tile_size)])
                if job.enable_tta:
                    ai_cmd.append("-x")

            elif selected_model.engine == ModelEngine.REALCUGAN:
                bin_path = self.hardware.realcugan_bin
                if not bin_path or not os.path.isfile(bin_path):
                    return False, f"RealCUGAN binary not found: {bin_path}"

                models_dir = os.path.join(os.path.dirname(bin_path), selected_model.internal_name)
                if not os.path.isdir(models_dir):
                    return False, f"Local RealCUGAN models directory not found: {models_dir}"

                try:
                    chosen_scale, effective_denoise, adjustment = resolve_realcugan_options(
                        models_dir,
                        chosen_scale,
                        job.denoise_level if job.denoise_level is not None else -1,
                    )
                except ValueError as exc:
                    return False, str(exc)
                if adjustment:
                    log_cmd(f"[REALCUGAN] Requested variant unavailable; using {adjustment} based on installed weights.")
                    report(0.20, f"RealCUGAN compatibility adjustment: {adjustment}")

                ai_cmd = [
                    bin_path,
                    "-i", frames_in_dir,
                    "-o", frames_out_dir,
                    "-m", models_dir,
                    "-s", str(chosen_scale),
                    "-j", thread_spec,
                    "-f", frame_ext,
                    "-v"
                ]
                if effective_tile_size > 0:
                    ai_cmd.extend(["-t", str(effective_tile_size)])
                if job.enable_tta:
                    ai_cmd.append("-x")
                ai_cmd.extend(["-n", str(effective_denoise)])
            else:
                return False, f"Unknown engine: {selected_model.engine}"

            log_cmd(f"[VULKAN AI] {' '.join(ai_cmd)}")

            self._current_process = self._start_process(
                ai_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
                bufsize=1
            )

            # Monitor progress per completed frame
            completed_frames = 0
            last_report_time = 0.0
            last_report_frames = 0
            est_encode_s = max(2.0, total_extracted / 120.0)

            report(
                0.15,
                f"Upscaling: 0/{total_extracted} frames (Starting engine...)",
                {"stage": "upscaling", "completed_frames": 0, "total_frames": total_extracted, "pct": 0.0, "file_eta_str": "Calculating..."}
            )

            if self._current_process.stdout:
                for line in self._current_process.stdout:
                    if self._is_cancelled:
                        self._current_process.terminate()
                        return False, "Cancelled by user."

                    line_clean = line.strip()
                    # Real-ESRGAN and RealCUGAN output: "... -> ... done"
                    if "->" in line_clean and line_clean.lower().endswith("done"):
                        completed_frames += 1
                        now = time.time()

                        # Update every 4 frames or 0.25s, or at completion
                        if (completed_frames - last_report_frames >= 4) or (now - last_report_time >= 0.25) or (completed_frames >= total_extracted):
                            last_report_time = now
                            last_report_frames = completed_frames

                            pct = min(100.0, (completed_frames / total_extracted) * 100.0)
                            elapsed_ai = max(0.01, now - t_start_ai)
                            fps_est = completed_frames / elapsed_ai

                            remaining_frames = max(0, total_extracted - completed_frames)
                            inference_eta_s = remaining_frames / max(0.1, fps_est)
                            file_total_eta_s = inference_eta_s + est_encode_s
                            eta_str = format_eta(file_total_eta_s)

                            global_ratio = 0.15 + (pct / 100.0) * 0.70
                            report(
                                global_ratio,
                                f"Upscaling: {pct:.1f}% ({completed_frames}/{total_extracted} frames · {fps_est:.1f} fps · File ETA: {eta_str})",
                                {
                                    "stage": "upscaling",
                                    "completed_frames": completed_frames,
                                    "total_frames": total_extracted,
                                    "pct": pct,
                                    "fps": fps_est,
                                    "file_eta_s": file_total_eta_s,
                                    "file_eta_str": eta_str
                                }
                            )

            ai_process = self._current_process
            ai_process.wait()
            self._forget_process(ai_process)
            if ai_process.returncode != 0 and not self._is_cancelled:
                return False, f"Upscaling engine error (exit code {ai_process.returncode})"

            if self._is_cancelled:
                return False, "Cancelled by user."

            output_frames = [
                name for name in os.listdir(frames_out_dir)
                if name.startswith("frame_") and name.endswith(f".{frame_ext}")
                and os.path.getsize(os.path.join(frames_out_dir, name)) > 0
            ]
            if len(output_frames) != total_extracted:
                return False, (
                    "Upscaling produced an incomplete frame sequence "
                    f"({len(output_frames)}/{total_extracted} frames)."
                )

            # ------------------------------------------------------------------
            # 8. Encode to Target Resolution with Hardware NVENC (Multithreaded)
            # ------------------------------------------------------------------
            chosen_encoder = job.video_codec
            if "nvenc" in chosen_encoder:
                if "hevc" in chosen_encoder and not self.hardware.has_nvenc_hevc:
                    chosen_encoder = "libx265"
                elif "h264" in chosen_encoder and not self.hardware.has_nvenc_h264:
                    chosen_encoder = "libx264"

            encode_eta_str = format_eta(est_encode_s)
            report(
                0.86,
                f"Encoding final video ({chosen_encoder} · File ETA: ~{encode_eta_str})...",
                {"stage": "encoding", "file_eta_s": est_encode_s, "file_eta_str": encode_eta_str}
            )

            audio_input_path = None
            if has_extracted_audio:
                audio_input_path = audio_file
            elif meta.has_audio:
                audio_input_path = job.input_file

            cq_val = str(job.cq_level) if job.cq_level else "19"
            p_val = job.nvenc_preset or "p7"

            encode_cmd = self._build_encode_cmd(
                frames_out_dir=frames_out_dir,
                frame_ext=frame_ext,
                meta=meta,
                target_w=target_w,
                target_h=target_h,
                chosen_encoder=chosen_encoder,
                cq_val=cq_val,
                p_val=p_val,
                audio_input_path=audio_input_path,
                audio_mode=job.audio_mode,
                part_output_file=part_output_file,
                audio_start_offset_s=_preview_start_s if audio_input_path == job.input_file else 0.0,
                audio_duration_s=_preview_dur_s if audio_input_path == job.input_file else None,
            )

            log_cmd(f"[FFMPEG ENCODE] {' '.join(encode_cmd)}")
            encode_rc = self._run_tracked(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if self._is_cancelled:
                return False, "Cancelled by user."
            if encode_rc != 0:
                return False, f"Final video encoding failed (exit code {encode_rc})."

            if not os.path.isfile(part_output_file) or os.path.getsize(part_output_file) == 0:
                return False, "Failed to encode final video."

            if not meta.has_audio:
                valid_video, validation_error = validate_encoded_video(
                    part_output_file, target_w, target_h, expect_audio=False
                )
                if not valid_video:
                    return False, validation_error

            # 8b. Strict Audio Preservation Integrity Check & Auto-Repair Pass
            if meta.has_audio:
                needs_repair = False
                try:
                    part_meta = probe_video(part_output_file)
                    if not part_meta.has_audio:
                        needs_repair = True
                except Exception:
                    needs_repair = True

                if needs_repair:
                    report(0.96, "Soundtrack missing in initial encode. Repairing soundtrack stream from source...")
                    log_cmd("[AUDIO REPAIR] Warning: Soundtrack missing in initial encode. Triggering guaranteed remux from source.")
                    repair_file = part_output_file + ".fixed" + scene_cfg.container
                    repair_cmd = [
                        "ffmpeg", "-y",
                        "-i", part_output_file,
                    ]
                    if _preview_start_s > 0:
                        repair_cmd.extend(["-ss", f"{_preview_start_s:.3f}"])
                    repair_cmd.extend([
                        "-i", job.input_file,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                    ])
                    if _preview_dur_s is not None:
                        repair_cmd.extend(["-t", f"{_preview_dur_s:.3f}"])
                    repair_cmd.append("-shortest")
                    repair_cmd.extend(clean_metadata_args(include_video=False))
                    repair_cmd.append(repair_file)
                    log_cmd(f"[AUDIO REPAIR REMUX] {' '.join(repair_cmd)}")
                    repair_rc = self._run_tracked(repair_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if self._is_cancelled:
                        return False, "Cancelled by user."
                    if repair_rc == 0 and os.path.isfile(repair_file) and os.path.getsize(repair_file) > 0:
                        try:
                            rep_meta = probe_video(repair_file)
                            if rep_meta.has_audio:
                                os.replace(repair_file, part_output_file)
                                log_cmd("[AUDIO REPAIR] Successfully restored soundtrack to output file.")
                        except Exception:
                            pass

                # Final strict guarantee check:
                try:
                    final_probe = probe_video(part_output_file)
                except Exception as exc:
                    return False, f"Audio integrity check failed: final video stream is unreadable ({exc})."
                if not final_probe.has_audio:
                    return False, "Audio integrity check failed: Soundtrack is missing from final output file."
                log_cmd(f"[AUDIO VERIFIED] Soundtrack confirmed intact in output ({final_probe.audio_codec.upper()}).")
                valid_video, validation_error = validate_encoded_video(
                    part_output_file,
                    target_w,
                    target_h,
                    expect_audio=True,
                    metadata=final_probe,
                )
                if not valid_video:
                    return False, validation_error

            # 9. Atomic Move to Final Scene Output Destination
            # Both paths live below output_dir, but safe_atomic_move also handles
            # cross-filesystem/cross-drive moves cleanly without data loss.
            safe_atomic_move(part_output_file, final_output_path)

            total_duration_s = max(0.1, time.time() - t_file_start)
            final_size_mb = round(os.path.getsize(final_output_path) / (1024**2), 1)
            duration_str = format_eta(total_duration_s)

            report(1.0, f"Completed in {duration_str} ({final_size_mb} MB)", {
                "output_file": final_output_path,
                "size_mb": final_size_mb,
                "duration_s": total_duration_s,
                "duration_str": duration_str
            })

            return True, final_output_path

        finally:
            unregister_active_job_dir(work_dir)
            # Clean up temporary frames and audio with retry loop to handle Windows file locking delays
            self._current_process = None
            if os.path.exists(work_dir):
                for attempt in range(4):
                    try:
                        shutil.rmtree(work_dir, ignore_errors=False)
                        break
                    except Exception:
                        time.sleep(0.2)
                if os.path.exists(work_dir):
                    try:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    except Exception:
                        pass

    def extract_and_upscale_single_frame(
        self,
        video_path: str,
        model_id: str = "auto",
        target_preset: str = "4k",
        offset_seconds: Optional[float] = None
    ) -> Tuple[bool, str, str, Dict]:
        """Extracts a single representative frame from video and upscales it with AI model.

        Returns (success, original_frame_path, upscaled_frame_path, info_dict).
        Takes ~0.5s and provides instant before/after visual verification before batch encoding.
        """
        if not os.path.isfile(video_path):
            return False, "", "", {"error": f"Video file not found: {video_path}"}

        try:
            meta = probe_video(video_path, self.hardware.ffprobe_path)
        except Exception as exc:
            return False, "", "", {"error": f"Video analysis error: {exc}"}

        if offset_seconds is None:
            offset_seconds = max(0.5, meta.duration * 0.15) if meta.duration > 1 else 0.0

        target_w, target_h, res_tag = calculate_target_dimensions(meta.width, meta.height, target_preset)

        if model_id == "auto" or model_id not in AVAILABLE_MODELS:
            selected_model, chosen_scale = choose_best_model_for_video(meta.width, meta.height, target_preset)
        else:
            selected_model = AVAILABLE_MODELS[model_id]
            chosen_scale = selected_model.default_scale

        temp_dir = ""
        try:
            base_dir = os.path.dirname(video_path)
            candidate_dir = os.path.join(base_dir, ".temp_upscaler", f"preview_{uuid.uuid4().hex[:8]}")
            os.makedirs(candidate_dir, exist_ok=True)
            temp_dir = candidate_dir
        except Exception:
            temp_dir = os.path.join(tempfile.gettempdir(), "upscaler_previews", f"preview_{uuid.uuid4().hex[:8]}")
            os.makedirs(temp_dir, exist_ok=True)

        orig_frame = os.path.join(temp_dir, "original_frame.png")
        up_frame = os.path.join(temp_dir, "upscaled_frame.png")

        # 1. Extract 1 single frame via ffmpeg
        ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        cmd_extract = [
            ffmpeg_bin, "-y",
            "-ss", f"{offset_seconds:.3f}",
            "-i", video_path,
            "-vframes", "1",
            orig_frame
        ]
        rc = subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc != 0 or not os.path.isfile(orig_frame) or os.path.getsize(orig_frame) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": "Cannot extract frame from source file."}

        # 2. Run AI upscaler on single frame
        t0 = time.time()
        thread_spec = get_optimal_thread_spec("auto", gpu_vram_gb=self.hardware.gpu_vram_gb)
        effective_tile_size = _auto_tile_size(self.hardware.gpu_vram_gb)

        if selected_model.engine == ModelEngine.REALESRGAN:
            bin_path = self.hardware.realesrgan_bin
            if not bin_path or not os.path.isfile(bin_path):
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "", "", {"error": f"Real-ESRGAN engine not found: {bin_path}"}
            models_dir = os.path.join(os.path.dirname(bin_path), "models")
            ai_cmd = [
                bin_path,
                "-i", orig_frame,
                "-o", up_frame,
                "-m", models_dir,
                "-n", selected_model.internal_name,
                "-s", str(chosen_scale),
                "-j", thread_spec,
                "-f", "png"
            ]
            if effective_tile_size > 0:
                ai_cmd.extend(["-t", str(effective_tile_size)])
        elif selected_model.engine == ModelEngine.REALCUGAN:
            bin_path = self.hardware.realcugan_bin
            if not bin_path or not os.path.isfile(bin_path):
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, "", "", {"error": f"RealCUGAN engine not found: {bin_path}"}
            models_dir = os.path.join(os.path.dirname(bin_path), selected_model.internal_name)
            chosen_scale, eff_denoise, _ = resolve_realcugan_options(models_dir, chosen_scale, -1)
            ai_cmd = [
                bin_path,
                "-i", orig_frame,
                "-o", up_frame,
                "-m", models_dir,
                "-s", str(chosen_scale),
                "-j", thread_spec,
                "-f", "png",
                "-n", str(eff_denoise)
            ]
            if effective_tile_size > 0:
                ai_cmd.extend(["-t", str(effective_tile_size)])
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": f"Unknown AI engine: {selected_model.engine}"}

        proc = subprocess.run(ai_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        infer_time = time.time() - t0
        if proc.returncode != 0 or not os.path.isfile(up_frame) or os.path.getsize(up_frame) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": f"Image upscaling failed (code {proc.returncode})"}

        return True, orig_frame, up_frame, {
            "model_name": selected_model.name,
            "orig_width": meta.width,
            "orig_height": meta.height,
            "target_width": target_w,
            "target_height": target_h,
            "scale": chosen_scale,
            "time_seconds": round(infer_time, 2),
            "temp_dir": temp_dir
        }


def cleanup_stale_temp_directories(
    root_dir: Optional[str] = None,
    max_age_seconds: int = 86400,
    max_age_hours: Optional[float] = None
) -> int:
    """Scans and removes stale .temp_upscaler directories left over from terminated sessions.

    Safely protects active jobs:
    1. Checks in-memory active job directories for the current process.
    2. Checks PID-based lease lock files (.job_lock) to avoid deleting directories
       of concurrent processes currently performing work.
    3. Uses a conservative default max age (24 hours instead of 15 minutes).
    """
    if max_age_hours is not None:
        max_age_seconds = int(max_age_hours * 3600)
    cleaned = 0
    search_dirs = [root_dir] if root_dir else []
    try:
        from src.core.detector import get_base_dir
        search_dirs.append(get_base_dir())
    except Exception:
        pass

    for s_dir in search_dirs:
        if not s_dir or not os.path.isdir(s_dir):
            continue
        temp_dir = os.path.join(s_dir, ".temp_upscaler")
        if os.path.isdir(temp_dir):
            try:
                for entry in os.listdir(temp_dir):
                    sub = os.path.join(temp_dir, entry)
                    if not os.path.isdir(sub):
                        continue

                    # 1. Skip if registered as active in current process
                    norm_sub = os.path.normcase(os.path.abspath(sub))
                    with _active_job_dirs_lock:
                        if norm_sub in _active_job_dirs:
                            continue

                    # 2. Check lock file for active foreign processes
                    lock_file = os.path.join(sub, ".job_lock")
                    if os.path.isfile(lock_file):
                        try:
                            with open(lock_file, "r", encoding="utf-8") as f:
                                lock_data = json.load(f)
                            owner_pid = lock_data.get("pid")
                            if owner_pid and is_process_alive(int(owner_pid)):
                                # Process is alive, do not clean!
                                continue
                        except Exception:
                            pass

                    # 3. Check modification age
                    try:
                        mtime = os.path.getmtime(sub)
                        if time.time() - mtime >= max_age_seconds:
                            shutil.rmtree(sub, ignore_errors=True)
                            cleaned += 1
                    except Exception:
                        pass
            except Exception:
                pass
    return cleaned
