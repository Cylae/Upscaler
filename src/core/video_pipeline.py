"""Core video processing pipeline for local neural upscaling and transcoding."""

from __future__ import annotations

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

from src.core.hardware import (
    HardwareInfo,
    detect_ai_binaries,
    detect_ffmpeg,
    detect_nvidia_gpu,
    get_base_dir,
    get_hardware_info,
    resolve_ai_binary,
)
from src.core.models import (
    AVAILABLE_MODELS,
    AIModelSpec,
    ModelEngine,
    auto_tile_size,
    calculate_target_dimensions,
    choose_best_model_for_video,
    get_optimal_thread_spec,
    is_heavy_model,
    recommend_model_for_video,
    resolve_realcugan_options,
)
from src.core.probe import (
    VideoMetadata,
    clean_metadata_args,
    find_forbidden_media_metadata,
    format_eta,
    probe_audio_stream,
    probe_audio_streams,
    probe_video,
    resolve_ffprobe_bin,
    verify_clean_media_metadata,
)
from src.core.process import (
    ProcessTracker,
    safe_kill_process_tree,
)
from src.core.scene_namer import (
    SceneNamingConfig,
    clean_scene_text,
    find_forbidden_output_markers,
    format_clean_name,
    format_output_name,
    format_scene_name,
    get_scene_relative_output_path,
    parse_episode_metadata,
    scrub_text_of_internal_markers,
    strip_forbidden_output_markers,
)
from src.core.storage import (
    cleanup_stale_temp_directories,
    is_process_alive,
    register_active_job_dir,
    safe_atomic_move,
    unregister_active_job_dir,
)
from src.core.validator import (
    is_file_complete_and_valid,
    validate_encoded_video,
    validate_job_config,
)

_auto_tile_size = auto_tile_size

# Image files and neural engines do not reliably honor pixel-aspect metadata.
# FFmpeg autorotates first; normalize the resulting display geometry before AI.
_SQUARE_PIXEL_FILTER = "scale=w='max(2,round(iw*sar/2)*2)':h='max(2,round(ih/2)*2)',setsar=1"


def _display_frame_dimensions(meta: VideoMetadata) -> Tuple[int, int]:
    return tuple(max(2, int(math.floor(value / 2 + 0.5)) * 2)
                 for value in (meta.display_width, meta.display_height))


def _square_pixel_filter(meta: Optional[VideoMetadata] = None) -> str:
    if meta is None:
        return _SQUARE_PIXEL_FILTER
    width, height = _display_frame_dimensions(meta)
    return f"scale={width}:{height}:flags=lanczos,setsar=1"


def _select_model_scale(model: AIModelSpec, required_scale: float) -> int:
    scales = model.supported_scales or [model.default_scale]
    return min((scale for scale in scales if scale >= required_scale - 0.05), default=max(scales))


def _list_nonempty_frames(directory: str, frame_ext: str) -> List[str]:
    """Reuse directory-entry metadata instead of statting every output path."""
    suffix = f".{frame_ext}"
    with os.scandir(directory) as entries:
        return sorted(
            entry.name for entry in entries
            if entry.name.startswith("frame_") and entry.name.endswith(suffix)
            and entry.is_file() and entry.stat().st_size > 0
        )


def _complete_frame_sequence(names: List[str], frame_ext: str, expected: int) -> bool:
    """A missing frame must not be masked by a stray or misnumbered image."""
    return len(names) == expected and all(
        name == f"frame_{index:08d}.{frame_ext}"
        for index, name in enumerate(names, 1)
    )



@dataclass
class UpscaleJobConfig:
    input_file: str
    output_dir: str
    base_dir: Optional[str] = None
    target_preset: str = "4k"  # "8k", "4k", "1440p", "1080p", "720p"
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


def _probe_cuda_support(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Returns True if FFmpeg was compiled with CUDA/cuvid hwaccel support."""
    try:
        out = subprocess.check_output(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
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
        with self._process_lock:
            # Cancellation must see either no child or an already registered one.
            if self._is_cancelled:
                raise RuntimeError("Operation cancelled by user.")
            proc = subprocess.Popen(cmd, **popen_kwargs)
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
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
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
            ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
            self._cuda_available = self.hardware.has_nvidia and _probe_cuda_support(ffmpeg_bin)
        return self._cuda_available

    def _check_nvenc_target(
        self,
        meta: VideoMetadata,
        target_w: int,
        target_h: int,
        encoder: str,
        cq_val: str,
        p_val: str,
    ) -> Tuple[bool, str]:
        """Test the actual encoder configuration using one synthetic frame.

        A compiled NVENC encoder does not guarantee that the GPU supports the
        target geometry. Check before extracting thousands of frames or running
        neural inference. Results are not cached: available sessions/VRAM change.
        """
        cmd = self._build_encode_cmd(
            frames_out_dir="", frame_ext="png", meta=meta,
            target_w=target_w, target_h=target_h, chosen_encoder=encoder,
            cq_val=cq_val, p_val=p_val, audio_input_path=None,
            audio_mode="copy", part_output_file="-",
        )
        input_pos = cmd.index("-i")
        fps_arg = cmd[cmd.index("-r") + 1]
        # Replace the image-sequence input, including its input -r, with lavfi.
        cmd = cmd[:input_pos - 2] + [
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", f"color=c=black:s={target_w}x{target_h}:r={fps_arg}",
        ] + cmd[input_pos + 2:-1] + ["-frames:v", "1", "-f", "null", "-"]
        proc = self._start_process(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace",
        )
        try:
            try:
                _, stderr_out = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                safe_kill_process_tree(proc)
                proc.communicate()
                return False, "Encoder check timed out after 30 seconds."
            if self._is_cancelled:
                raise RuntimeError("Operation cancelled by user.")
            if proc.returncode == 0:
                return True, ""
            detail = (stderr_out or "").strip()
            return False, detail[-2000:] or f"Encoder check failed (exit code {proc.returncode})."
        finally:
            self._forget_process(proc)

    def _resolve_encoder(
        self,
        job: UpscaleJobConfig,
        meta: VideoMetadata,
        target_w: int,
        target_h: int,
        log: Callable[[str], None],
    ) -> str:
        encoder = job.video_codec.lower()
        fallback = {"hevc_nvenc": "libx265", "h264_nvenc": "libx264"}
        if encoder not in fallback:
            return encoder
        available = self.hardware.has_nvenc_hevc if encoder == "hevc_nvenc" else self.hardware.has_nvenc_h264
        if not available:
            log(f"[ENCODER] {encoder} unavailable; using {fallback[encoder]} on CPU.")
            return fallback[encoder]
        supported, reason = self._check_nvenc_target(
            meta, target_w, target_h, encoder, str(job.cq_level), job.nvenc_preset.lower(),
        )
        if self._is_cancelled:
            raise RuntimeError("Operation cancelled by user.")
        if supported:
            log(f"[ENCODER] {encoder} verified for {target_w}x{target_h}.")
            return encoder
        log(f"[ENCODER] {encoder} check failed: {reason}\nUsing {fallback[encoder]} on CPU.")
        return fallback[encoder]

    def _build_extract_frames_cmd(
        self,
        input_file: str,
        frames_in_dir: str,
        frame_ext: str,
        use_hwdecode: bool,
        start_offset_s: float = 0.0,
        duration_s: Optional[float] = None,
        meta: Optional[VideoMetadata] = None,
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

        ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        if use_hwdecode:
            cmd = [
                ffmpeg_bin, "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
            ] + common_input_flags + seek_flags + [
                "-i", input_file,
                "-map", "0:V:0", "-an", "-sn",
                "-vf", f"hwdownload,format=nv12,{_square_pixel_filter(meta)}",
                "-filter_threads", "0",
                "-filter_complex_threads", "0",
            ]
        else:
            cmd = [
                ffmpeg_bin, "-y",
                "-threads", "0",
            ] + common_input_flags + seek_flags + [
                "-i", input_file,
                "-map", "0:V:0", "-an", "-sn",
                "-vf", _square_pixel_filter(meta),
                "-filter_threads", "0",
                "-filter_complex_threads", "0",
            ]

        if duration_s is not None:
            cmd.extend(["-t", f"{duration_s:.3f}"])

        if frame_ext == "jpg":
            cmd.extend(["-qscale:v", "1", "-qmin", "1"])
        else:
            # PNG stays lossless. Spend less CPU compressing temporary images;
            # the model receives exactly the same decoded pixels.
            cmd.extend(["-compression_level", "1"])

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
          - Retains NVENC lookahead, surfaces, AQ and the requested CQ/preset.
          - Retains the existing CPU fallback preset and requested CRF.
          - filter_threads / filter_complex_threads set to 0 for maximum CPU usage
            during the Lanczos scale filter.
        """
        ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        fps_arg = meta.r_frame_rate if (meta.r_frame_rate and "/" in meta.r_frame_rate and meta.r_frame_rate != "0/0") else str(meta.fps)
        cmd = [
            ffmpeg_bin, "-y",
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
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos,setsar=1",
            "-filter_threads", "0",
            "-filter_complex_threads", "0",
        ])

        # Colorspace and dynamic range preservation
        is_hd_or_higher = target_h >= 720 or target_w >= 1280
        is_sd_source = 0 < meta.height < 720 and 0 < meta.width < 1280

        if is_hd_or_higher and is_sd_source:
            # Upscaling SD (BT.601 / SMPTE 170M) to HD/4K: conform to standard BT.709
            cmd.extend([
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
            ])
            if meta.color_range and meta.color_range in ("tv", "pc", "limited", "full"):
                cmd.extend(["-color_range", meta.color_range])
        else:
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
        # Batch workers allocate a fresh pipeline for each job. Do not erase a
        # stop request delivered just before this call; reuse requires reset_cancel().
        if self._is_cancelled:
            return False, "Cancelled by user."
        t_file_start = time.monotonic()

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
        target_w, target_h, res_tag = calculate_target_dimensions(
            meta.width, meta.height, job.target_preset,
            sample_aspect_ratio=meta.sample_aspect_ratio, rotation=meta.rotation,
        )

        # Every concurrent job gets an isolated naming configuration. Mutating
        # the shared batch object here caused cross-worker naming races.
        scene_cfg = replace(job.scene_config) if job.scene_config else SceneNamingConfig()
        scene_cfg.resolution_tag = res_tag

        # Set codec tag in scene name according to chosen encoder
        if "hevc" in job.video_codec.lower() or "x265" in job.video_codec.lower():
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
        resolved_model_id = job.model_id
        req_scale = max(target_w / max(1, meta.display_width), target_h / max(1, meta.display_height))
        if resolved_model_id in AVAILABLE_MODELS and resolved_model_id != "auto":
            selected_model = AVAILABLE_MODELS[resolved_model_id]
            chosen_scale = _select_model_scale(selected_model, req_scale)
        else:
            found = False
            if resolved_model_id and resolved_model_id != "auto":
                for m_id, spec in AVAILABLE_MODELS.items():
                    if m_id != "auto" and (resolved_model_id in (spec.name, spec.internal_name, spec.name.split(" (")[0])):
                        selected_model = spec
                        chosen_scale = _select_model_scale(spec, req_scale)
                        found = True
                        break
            if not found:
                selected_model, chosen_scale = choose_best_model_for_video(
                    meta.display_width, meta.display_height, job.target_preset
                )

        report(0.05, f"Selected model: {selected_model.name} (Scale {chosen_scale}x)", {"model_name": selected_model.name, "scale": chosen_scale})
        model_reason = f"Auto-selected for {round(meta.display_width)}×{round(meta.display_height)} display size" if job.model_id == "auto" else "User selected"
        log_cmd(f"[AI MODEL] {selected_model.name} ×{chosen_scale} ({model_reason})")
        if req_scale > chosen_scale + 0.05:
            log_cmd(
                f"[RESIZE] Target exceeds the model's native {chosen_scale}x output; "
                f"a final Lanczos resize produces {target_w}x{target_h}."
            )

        # A long-lived GUI may have started before the portable engines were
        # installed. Resolve again before doing expensive extraction/encoding.
        try:
            bin_path = resolve_ai_binary(self.hardware, selected_model.engine)
        except (FileNotFoundError, ValueError) as exc:
            return False, str(exc)

        report(0.06, f"Checking encoder for {target_w}x{target_h}...")
        try:
            chosen_encoder = self._resolve_encoder(job, meta, target_w, target_h, log_cmd)
        except (OSError, RuntimeError) as exc:
            return False, "Cancelled by user." if self._is_cancelled else f"Encoder check failed: {exc}"

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

                ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
                audio_cmd = [ffmpeg_bin, "-y"]
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
                audio_cmd = [ffmpeg_bin, "-y"]
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
                    meta=meta,
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
                        meta=meta,
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

                _frames_result["extracted"] = sum(
                    1 for name in os.listdir(frames_in_dir)
                    if name.startswith("frame_") and name.endswith(f".{frame_ext}")
                )

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
                    break
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
                err_msg = _frames_result.get("error") or "No frames extracted from source video."
                return False, err_msg

            log_cmd(f"[FRAMES] {total_extracted} frames extracted.")

            # ------------------------------------------------------------------
            # 6. Compute optimised AI upscaler parameters
            # ------------------------------------------------------------------
            thread_spec = get_optimal_thread_spec(job.threads, gpu_vram_gb=self.hardware.gpu_vram_gb, model=selected_model)

            # Resolve tile_size: respect user override, otherwise auto-compute from VRAM and model weight
            effective_tile_size = job.tile_size if job.tile_size > 0 else _auto_tile_size(self.hardware.gpu_vram_gb, model=selected_model)

            mode_label = f" — Preview {int(_preview_dur_s)}s" if _preview_mode and _preview_dur_s else ""
            report(0.20, f"Upscaling with {selected_model.name} (Threads: {thread_spec}, Tile: {effective_tile_size or 'auto'}){mode_label}...")
            t_start_ai = time.monotonic()

            # ------------------------------------------------------------------
            # 7. Run Upscaler with Automatic GPU Memory Protection & Fallback
            # ------------------------------------------------------------------
            max_ai_attempts = 2
            ai_attempt = 1

            while ai_attempt <= max_ai_attempts:
                if selected_model.engine == ModelEngine.REALESRGAN:
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
                    models_dir = os.path.join(os.path.dirname(bin_path), selected_model.internal_name)
                    if not os.path.isdir(models_dir):
                        return False, f"Local RealCUGAN models directory not found: {models_dir}"

                    try:
                        chosen_scale, effective_denoise, target_model_name = resolve_realcugan_options(
                            models_dir,
                            chosen_scale,
                            job.denoise_level if job.denoise_level is not None else -1,
                        )
                    except ValueError as exc:
                        return False, str(exc)

                    if target_model_name != selected_model.internal_name:
                        alt_dir = os.path.join(os.path.dirname(bin_path), target_model_name)
                        if os.path.isdir(alt_dir):
                            models_dir = alt_dir
                        log_cmd(f"[REALCUGAN] Requested model '{selected_model.internal_name}' unavailable for scale {chosen_scale}; using {target_model_name} instead.")
                        report(0.20, f"RealCUGAN compatibility adjustment: {target_model_name}")
                    elif effective_denoise != (job.denoise_level if job.denoise_level is not None else -1):
                        log_cmd(f"[REALCUGAN] Requested denoise level unavailable; adjusted to {effective_denoise}.")
                        report(0.20, f"RealCUGAN denoise adjusted to {effective_denoise}")

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
                last_report_time = time.monotonic()
                est_encode_s = max(2.0, total_extracted / 120.0)
                vulkan_errors: List[str] = []

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
                        low = line_clean.lower()
                        if any(err_token in low for err_token in (
                            "vkallocatememory failed",
                            "vkqueuesubmit failed",
                            "vkwaitforfences failed",
                            "vkcreateimage failed",
                            "out of device memory",
                            "out of memory"
                        )):
                            vulkan_errors.append(line_clean)

                        # Real-ESRGAN and RealCUGAN output: "... -> ... done"
                        if "->" in line_clean and line_clean.lower().endswith("done"):
                            completed_frames += 1
                            now = time.monotonic()

                            # Bound UI work on fast models; always publish completion.
                            if now - last_report_time >= 0.25 or completed_frames >= total_extracted:
                                last_report_time = now

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

                if self._is_cancelled:
                    return False, "Cancelled by user."

                output_frames = _list_nonempty_frames(frames_out_dir, frame_ext)
                sequence_complete = _complete_frame_sequence(output_frames, frame_ext, total_extracted)

                # Check if output frames are corrupted blank/black (e.g. silent Vulkan OOM failure)
                is_corrupted_black = False
                if sequence_complete and total_extracted > 0:
                    sample_indices = [
                        min(len(output_frames) - 1, max(0, total_extracted // 4)),
                        min(len(output_frames) - 1, max(0, total_extracted // 2)),
                        min(len(output_frames) - 1, max(0, (total_extracted * 3) // 4)),
                    ]
                    checked_any = False
                    all_black_while_input_not = True
                    for s_idx in sample_indices:
                        sample_name = output_frames[s_idx]
                        sample_in = os.path.join(frames_in_dir, sample_name)
                        sample_out = os.path.join(frames_out_dir, sample_name)
                        if os.path.isfile(sample_in) and os.path.isfile(sample_out):
                            try:
                                from PIL import Image
                                with Image.open(sample_in) as im_in, Image.open(sample_out) as im_out:
                                    ext_in = im_in.getextrema()
                                    ext_out = im_out.getextrema()
                                    def _is_ext_black(ext):
                                        if isinstance(ext[0], tuple):
                                            return all(e == (0, 0) for e in ext)
                                        return ext == (0, 0)
                                    if not _is_ext_black(ext_in):
                                        checked_any = True
                                        if not _is_ext_black(ext_out):
                                            all_black_while_input_not = False
                                            break
                            except Exception:
                                pass
                    if checked_any and all_black_while_input_not:
                        is_corrupted_black = True

                has_failure = (
                    ai_process.returncode != 0
                    or bool(vulkan_errors)
                    or not sequence_complete
                    or is_corrupted_black
                )

                if has_failure:
                    err_detail = (
                        vulkan_errors[0] if vulkan_errors
                        else ("Corrupted blank/black output frames detected" if is_corrupted_black
                              else f"Incomplete or misnumbered output frames ({len(output_frames)}/{total_extracted}) with exit code {ai_process.returncode}")
                    )
                    if ai_attempt < max_ai_attempts:
                        # Clear incomplete/corrupted frames and retry with safe profile
                        for f in os.listdir(frames_out_dir):
                            try:
                                os.remove(os.path.join(frames_out_dir, f))
                            except Exception:
                                pass

                        new_tile = 200 if (effective_tile_size > 200 or effective_tile_size == 0) else 100
                        new_threads = "2:2:2"
                        log_cmd(
                            f"[VULKAN RETRY] GPU memory pressure detected ({err_detail}). "
                            f"Retrying with safe parameters (tile: {new_tile}, threads: {new_threads})..."
                        )
                        effective_tile_size = new_tile
                        thread_spec = new_threads
                        ai_attempt += 1
                        continue
                    else:
                        return False, f"Upscaling engine failed ({err_detail})"

                break

            # ------------------------------------------------------------------
            # 8. Encode to Target Resolution with Hardware NVENC (Multithreaded)
            # ------------------------------------------------------------------
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
            p_val = (job.nvenc_preset or "p7").lower()

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
                    ffmpeg_bin = self.hardware.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
                    repair_cmd = [
                        ffmpeg_bin, "-y",
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

            total_duration_s = max(0.1, time.monotonic() - t_file_start)
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
        offset_seconds: Optional[float] = None,
        *,
        threads: str = "auto",
        tile_size: int = 0,
        denoise_level: int = -1,
        enable_tta: bool = False,
    ) -> Tuple[bool, str, str, Dict]:
        """Extracts a single representative frame from video and upscales it with AI model.

        Returns (success, original_frame_path, upscaled_frame_path, info_dict).
        The returned preview uses the same display geometry and final dimensions
        as the full encode, including targets beyond the model's native scale,
        and honors the selected inference settings used by the full batch.
        """
        if self._is_cancelled:
            return False, "", "", {"error": "Cancelled by user."}
        if not os.path.isfile(video_path):
            return False, "", "", {"error": f"Video file not found: {video_path}"}

        try:
            meta = probe_video(video_path, self.hardware.ffprobe_path)
        except Exception as exc:
            return False, "", "", {"error": f"Video analysis error: {exc}"}

        if offset_seconds is None:
            offset_seconds = max(0.5, meta.duration * 0.15) if meta.duration > 1 else 0.0

        target_w, target_h, res_tag = calculate_target_dimensions(
            meta.width, meta.height, target_preset,
            sample_aspect_ratio=meta.sample_aspect_ratio, rotation=meta.rotation,
        )

        if model_id == "auto" or model_id not in AVAILABLE_MODELS:
            selected_model, chosen_scale = choose_best_model_for_video(
                meta.display_width, meta.display_height, target_preset,
            )
        else:
            selected_model = AVAILABLE_MODELS[model_id]
            required_scale = max(target_w / max(1, meta.display_width), target_h / max(1, meta.display_height))
            chosen_scale = _select_model_scale(selected_model, required_scale)

        try:
            bin_path = resolve_ai_binary(self.hardware, selected_model.engine)
        except (FileNotFoundError, ValueError) as exc:
            return False, "", "", {"error": str(exc)}

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
            "-map", "0:V:0", "-an", "-sn",
            "-vf", _square_pixel_filter(meta),
            "-vframes", "1",
            orig_frame
        ]
        try:
            rc = self._run_tracked(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, RuntimeError) as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": str(exc)}
        if rc != 0 or not os.path.isfile(orig_frame) or os.path.getsize(orig_frame) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": "Cannot extract frame from source file."}

        # 2. Run AI upscaler on single frame
        t0 = time.time()
        thread_spec = get_optimal_thread_spec(threads, gpu_vram_gb=self.hardware.gpu_vram_gb, model=selected_model)
        effective_tile_size = tile_size if tile_size > 0 else _auto_tile_size(self.hardware.gpu_vram_gb, model=selected_model)

        if selected_model.engine == ModelEngine.REALESRGAN:
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
            models_dir = os.path.join(os.path.dirname(bin_path), selected_model.internal_name)
            chosen_scale, eff_denoise, target_model_name = resolve_realcugan_options(
                models_dir, chosen_scale, denoise_level if denoise_level is not None else -1,
            )
            if target_model_name != selected_model.internal_name:
                alt_dir = os.path.join(os.path.dirname(bin_path), target_model_name)
                if os.path.isdir(alt_dir):
                    models_dir = alt_dir
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

        if enable_tta:
            ai_cmd.append("-x")

        try:
            ai_rc = self._run_tracked(ai_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, RuntimeError) as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": str(exc)}
        infer_time = time.time() - t0
        if ai_rc != 0 or not os.path.isfile(up_frame) or os.path.getsize(up_frame) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": f"Image upscaling failed (code {ai_rc})"}

        final_frame = os.path.join(temp_dir, "preview_target.png")
        resize_cmd = [
            ffmpeg_bin, "-y", "-i", up_frame,
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos,setsar=1",
            "-frames:v", "1", final_frame,
        ]
        try:
            resize_rc = self._run_tracked(
                resize_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, RuntimeError) as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": str(exc)}
        if resize_rc != 0 or not os.path.isfile(final_frame) or os.path.getsize(final_frame) == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, "", "", {"error": "Cannot resize preview to the requested output dimensions."}
        os.replace(final_frame, up_frame)
        infer_time = time.time() - t0

        return True, orig_frame, up_frame, {
            "model_name": selected_model.name,
            "orig_width": _display_frame_dimensions(meta)[0],
            "orig_height": _display_frame_dimensions(meta)[1],
            "target_width": target_w,
            "target_height": target_h,
            "scale": chosen_scale,
            "time_seconds": round(infer_time, 2),
            "temp_dir": temp_dir
        }


