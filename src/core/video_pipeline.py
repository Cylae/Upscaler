import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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


def find_forbidden_media_metadata(filepath: str, ffprobe_bin: Optional[str] = None) -> list:
    """Checks output media file tags via ffprobe for any forbidden markers."""
    bin_path = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    try:
        raw = subprocess.check_output(
            [bin_path, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", filepath],
            text=True,
            errors="ignore"
        )
        return find_forbidden_output_markers(raw)
    except Exception:
        return []


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


def probe_video(filepath: str) -> VideoMetadata:
    """Uses ffprobe to extract complete video metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        filepath
    ]
    raw = subprocess.check_output(cmd, text=True, errors="ignore")
    data = json.loads(raw)

    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ValueError(f"No video stream found in {filepath}")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))

    fps_str = v_stream.get("r_frame_rate", "24/1")
    parts = fps_str.split("/")
    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 and float(parts[1]) > 0 else 24.0

    duration = float(data.get("format", {}).get("duration", v_stream.get("duration", 0.0)))
    nb_frames_raw = v_stream.get("nb_frames")
    nb_frames = int(nb_frames_raw) if nb_frames_raw and str(nb_frames_raw).isdigit() else max(1, int(duration * fps))

    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    has_audio = a_stream is not None
    audio_codec = a_stream.get("codec_name", "") if a_stream else ""

    s_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"), None)
    has_subtitles = s_stream is not None

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
        has_subtitles=has_subtitles
    )


def is_file_complete_and_valid(filepath: str) -> bool:
    """Checks if an already existing output file is valid and complete."""
    if not os.path.isfile(filepath):
        return False
    if os.path.getsize(filepath) < 100_000:  # < 100KB is incomplete
        return False
    try:
        meta = probe_video(filepath)
        return meta.duration > 0.5 and meta.width > 0
    except Exception:
        return False


class VideoUpscalerPipeline:
    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or get_hardware_info()
        self._is_cancelled = False
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        """Signals cancellation and terminates running subprocess."""
        self._is_cancelled = True
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def reset_cancel(self):
        self._is_cancelled = False
        self._current_process = None

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
        4. Extract frames & audio.
        5. Run upscaling engine with progress tracking.
        6. Encode to target 4K with NVENC.
        7. Atomic rename & cleanup.
        """
        self.reset_cancel()

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

        scene_cfg = job.scene_config or SceneNamingConfig()
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
        out_folder = os.path.dirname(final_output_path)
        os.makedirs(out_folder, exist_ok=True)

        # 3. Check resume (already completed?)
        if not job.overwrite and is_file_complete_and_valid(final_output_path):
            report(1.0, "Episode already present (Skipped)", {"output_file": final_output_path, "skipped": True})
            return True, f"Already completed: {os.path.basename(final_output_path)}"

        # 4. Resolve Model
        if job.model_id == "auto" or job.model_id not in AVAILABLE_MODELS:
            selected_model, chosen_scale = choose_best_model_for_video(meta.width, meta.height)
        else:
            selected_model = AVAILABLE_MODELS[job.model_id]
            chosen_scale = selected_model.default_scale

        report(0.05, f"Selected model: {selected_model.name} (Scale {chosen_scale}x)")

        # Prepare workspace & temp directory
        root_temp = job.temp_dir or os.path.join(job.output_dir, ".temp_upscaler")
        job_id = f"job_{int(time.time())}_{abs(hash(job.input_file)) % 100000}"
        work_dir = os.path.join(root_temp, job_id)
        frames_in_dir = os.path.join(work_dir, "frames_in")
        frames_out_dir = os.path.join(work_dir, "frames_out")
        audio_file = os.path.join(work_dir, "audio.m4a")
        part_output_file = os.path.join(work_dir, "output_part" + scene_cfg.container)

        os.makedirs(frames_in_dir, exist_ok=True)
        os.makedirs(frames_out_dir, exist_ok=True)

        try:
            # 5. Extract Audio
            if meta.has_audio:
                report(0.08, "Extracting audio stream...")
                audio_cmd = [
                    "ffmpeg", "-y",
                    "-i", job.input_file,
                    "-vn",
                    "-map", "0:a:0",
                    "-c:a", "copy",
                    audio_file
                ]
                subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if not os.path.isfile(audio_file) or os.path.getsize(audio_file) == 0:
                    # Fallback to AAC encode if direct stream copy fails
                    audio_cmd = [
                        "ffmpeg", "-y",
                        "-i", job.input_file,
                        "-vn",
                        "-map", "0:a:0",
                        "-c:a", "aac", "-b:a", "192k",
                        audio_file
                    ]
                    subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

            if self._is_cancelled:
                return False, "Cancelled by user."

            # 6. Extract Frames
            report(0.12, f"Extracting frames ({meta.nb_frames} images)...")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", job.input_file,
                "-qscale:v", "1",
                "-qmin", "1",
                os.path.join(frames_in_dir, "frame_%08d.png")
            ]
            self._current_process = subprocess.Popen(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current_process.wait()

            if self._is_cancelled:
                return False, "Cancelled by user."

            frame_files = [f for f in os.listdir(frames_in_dir) if f.startswith("frame_") and f.endswith(".png")]
            total_extracted = len(frame_files)
            if total_extracted == 0:
                return False, "No frames extracted from source video."

            # 7. Run Upscaler
            report(0.20, f"Upscaling with {selected_model.name}...")
            t_start_ai = time.time()

            if selected_model.engine == ModelEngine.REALESRGAN:
                bin_path = self.hardware.realesrgan_bin
                if not bin_path or not os.path.isfile(bin_path):
                    return False, f"Real-ESRGAN binary not found: {bin_path}"

                ai_cmd = [
                    bin_path,
                    "-i", frames_in_dir,
                    "-o", frames_out_dir,
                    "-n", selected_model.internal_name,
                    "-s", str(chosen_scale),
                    "-f", "png",
                    "-v"
                ]
                if job.tile_size > 0:
                    ai_cmd.extend(["-t", str(job.tile_size)])

            elif selected_model.engine == ModelEngine.REALCUGAN:
                bin_path = self.hardware.realcugan_bin
                if not bin_path or not os.path.isfile(bin_path):
                    return False, f"RealCUGAN binary not found: {bin_path}"

                ai_cmd = [
                    bin_path,
                    "-i", frames_in_dir,
                    "-o", frames_out_dir,
                    "-m", selected_model.internal_name,
                    "-s", str(chosen_scale),
                    "-f", "png",
                    "-v"
                ]
                if job.tile_size > 0:
                    ai_cmd.extend(["-t", str(job.tile_size)])
            else:
                return False, f"Unknown engine: {selected_model.engine}"

            self._current_process = subprocess.Popen(
                ai_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
                bufsize=1
            )

            # Monitor progress
            pct_regex = re.compile(r"(\d+(?:\.\d+)?)%")
            last_pct = 0.0

            if self._current_process.stdout:
                for line in self._current_process.stdout:
                    if self._is_cancelled:
                        self._current_process.terminate()
                        return False, "Cancelled by user."
                    m = pct_regex.search(line)
                    if m:
                        try:
                            pct = float(m.group(1))
                            if pct != last_pct:
                                last_pct = pct
                                global_ratio = 0.20 + (pct / 100.0) * 0.60
                                elapsed = time.time() - t_start_ai
                                fps_est = (total_extracted * (pct / 100.0)) / max(0.1, elapsed)
                                eta_sec = (total_extracted * (1.0 - pct / 100.0)) / max(0.1, fps_est)
                                report(
                                    global_ratio,
                                    f"Upscaling: {pct:.1f}% ({fps_est:.1f} fps - ETA: {int(eta_sec)}s)",
                                    {"pct": pct, "fps": fps_est, "eta_s": eta_sec}
                                )
                        except Exception:
                            pass

            self._current_process.wait()
            if self._current_process.returncode != 0 and not self._is_cancelled:
                return False, f"Upscaling engine error (exit code {self._current_process.returncode})"

            if self._is_cancelled:
                return False, "Cancelled by user."

            # 8. Encode to Target Resolution with Hardware NVENC
            report(0.85, f"Encoding final video to {target_w}x{target_h} ({job.video_codec})...")

            chosen_encoder = job.video_codec
            if "nvenc" in chosen_encoder:
                if "hevc" in chosen_encoder and not self.hardware.has_nvenc_hevc:
                    chosen_encoder = "libx265"
                elif "h264" in chosen_encoder and not self.hardware.has_nvenc_h264:
                    chosen_encoder = "libx264"

            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(meta.fps),
                "-i", os.path.join(frames_out_dir, "frame_%08d.png")
            ]

            if os.path.isfile(audio_file):
                encode_cmd.extend(["-i", audio_file])

            # Video filter: scale to exact target resolution with high-grade Lanczos
            encode_cmd.extend(["-vf", f"scale={target_w}:{target_h}:flags=lanczos"])

            # Encoder parameters
            if chosen_encoder == "hevc_nvenc":
                encode_cmd.extend([
                    "-c:v", "hevc_nvenc",
                    "-preset", "p7",
                    "-cq", "19",
                    "-pix_fmt", "yuv420p"
                ])
            elif chosen_encoder == "h264_nvenc":
                encode_cmd.extend([
                    "-c:v", "h264_nvenc",
                    "-preset", "p7",
                    "-cq", "18",
                    "-pix_fmt", "yuv420p"
                ])
            elif chosen_encoder == "libx265":
                encode_cmd.extend([
                    "-c:v", "libx265",
                    "-crf", "19",
                    "-preset", "medium",
                    "-pix_fmt", "yuv420p"
                ])
            else:  # libx264
                encode_cmd.extend([
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "medium",
                    "-pix_fmt", "yuv420p"
                ])

            if os.path.isfile(audio_file):
                encode_cmd.extend(["-c:a", "copy", "-shortest"])

            # Clean and sanitize metadata: absolute zero trace of internal tooling or tags
            encode_cmd.extend(clean_metadata_args(include_video=True))

            encode_cmd.append(part_output_file)

            self._current_process = subprocess.Popen(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current_process.wait()

            if self._is_cancelled:
                return False, "Cancelled by user."

            if not os.path.isfile(part_output_file) or os.path.getsize(part_output_file) == 0:
                return False, "Failed to encode final video."

            # 9. Atomic Move to Final Scene Output Destination
            if os.path.isfile(final_output_path):
                os.remove(final_output_path)
            shutil.move(part_output_file, final_output_path)

            report(1.0, "4K Upscaling completed successfully!", {
                "output_file": final_output_path,
                "size_mb": round(os.path.getsize(final_output_path) / (1024**2), 2)
            })

            return True, final_output_path

        finally:
            # Clean up temporary frames and audio
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
            self._current_process = None
