"""Hardware detection and diagnostic reporting.

Detects NVIDIA GPUs, VRAM capacity, NVENC hardware encoders, FFmpeg/FFprobe binaries,
and portable Vulkan neural upscalers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HardwareInfo:
    gpu_name: str = "Unknown"
    gpu_vram_gb: float = 0.0
    has_nvidia: bool = False
    has_nvenc_hevc: bool = False
    has_nvenc_h264: bool = False
    ffmpeg_path: Optional[str] = None
    ffprobe_path: Optional[str] = None
    ffmpeg_version: str = "Not detected"
    realesrgan_bin: Optional[str] = None
    realcugan_bin: Optional[str] = None
    cpu_cores: int = 4

    @property
    def nvenc_available(self) -> bool:
        """Returns True if any NVENC hardware encoder (HEVC or H.264) is supported."""
        return bool(self.has_nvenc_hevc or self.has_nvenc_h264)

    @property
    def has_nvenc(self) -> bool:
        """Alias for nvenc_available."""
        return self.nvenc_available


def get_base_dir() -> str:
    """Returns root directory of the application."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def detect_ffmpeg() -> Tuple[Optional[str], Optional[str], str, bool, bool]:
    """Detects ffmpeg and ffprobe on PATH or adjacent directory, checks version and NVENC support."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and not ffprobe_path:
        candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(candidate):
            ffprobe_path = candidate

    if not ffmpeg_path:
        return None, ffprobe_path, "Not found", False, False

    version_str = "Installed"
    has_hevc_nvenc = False
    has_h264_nvenc = False

    try:
        output = subprocess.check_output(
            [ffmpeg_path, "-version"], stderr=subprocess.STDOUT,
            text=True, errors="ignore", timeout=10
        )
        first_line = output.splitlines()[0] if output else ""
        if "ffmpeg version" in first_line:
            version_str = first_line.split("Copyright")[0].strip()

        enc_output = subprocess.check_output(
            [ffmpeg_path, "-encoders"], stderr=subprocess.STDOUT,
            text=True, errors="ignore", timeout=15
        )
        has_hevc_nvenc = "hevc_nvenc" in enc_output
        has_h264_nvenc = "h264_nvenc" in enc_output
    except Exception:
        pass

    return ffmpeg_path, ffprobe_path, version_str, has_hevc_nvenc, has_h264_nvenc


def detect_nvidia_gpu() -> Tuple[str, float, bool]:
    """Uses nvidia-smi to query NVIDIA GPU model and dedicated VRAM capacity."""
    gpu_name = "Unknown"
    gpu_vram = 0.0
    has_nvidia = False

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi and os.name == "nt":
        std_path = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.isfile(std_path):
            nvidia_smi = std_path

    if nvidia_smi:
        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"
            ]
            raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=5).strip()
            if raw:
                parts = raw.splitlines()[0].split(",")
                gpu_name = parts[0].strip()
                gpu_vram = round(float(parts[1].strip()) / 1024.0, 1)
                has_nvidia = True
        except Exception:
            pass

    return gpu_name, gpu_vram, has_nvidia


def detect_ai_binaries(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Locates portable Vulkan neural executables inside tools/."""
    ext = ".exe" if os.name == "nt" else ""
    realesrgan_candidate = os.path.join(base_dir, "tools", "realesrgan", f"realesrgan-ncnn-vulkan{ext}")
    realcugan_candidate = os.path.join(base_dir, "tools", "realcugan", f"realcugan-ncnn-vulkan{ext}")

    realesrgan_path = realesrgan_candidate if os.path.isfile(realesrgan_candidate) else shutil.which(f"realesrgan-ncnn-vulkan{ext}")
    realcugan_path = realcugan_candidate if os.path.isfile(realcugan_candidate) else shutil.which(f"realcugan-ncnn-vulkan{ext}")

    return realesrgan_path, realcugan_path


def resolve_ai_binary(hardware: HardwareInfo, engine: str) -> str:
    """Resolve an engine at use time, including installations made after startup.

    Keep valid caller-provided paths. Refresh only a missing/stale engine path;
    GPU and FFmpeg capability probes remain cached and no model is substituted.
    """
    if engine == "realesrgan":
        engine, attribute, label = "realesrgan", "realesrgan_bin", "Real-ESRGAN"
    elif engine == "realcugan":
        engine, attribute, label = "realcugan", "realcugan_bin", "RealCUGAN"
    else:
        raise ValueError(f"Unknown AI engine: {engine}")

    previous_path = getattr(hardware, attribute)
    if previous_path and os.path.isfile(previous_path):
        return previous_path

    base_dir = get_base_dir()
    realesrgan, realcugan = detect_ai_binaries(base_dir)
    resolved = realesrgan if engine == "realesrgan" else realcugan
    if resolved and os.path.isfile(resolved):
        # The window and batch manager can hold the same snapshot for a whole
        # session. Update that object so subsequent operations see the repair.
        setattr(hardware, attribute, resolved)
        return resolved

    filename = f"{engine}-ncnn-vulkan" + (".exe" if os.name == "nt" else "")
    expected = os.path.join(base_dir, "tools", engine, filename)
    previous = f"\nPrevious path is unavailable: {previous_path}" if previous_path else ""
    raise FileNotFoundError(
        f"{label} engine not found.\nExpected executable: {expected}{previous}\n"
        f"Extract the {label} executable and its model folders into "
        f"{os.path.dirname(expected)} (see README.md, Prerequisites). "
        f"An installation of {filename} on PATH is also supported."
    )


_CACHED_HARDWARE_INFO: Optional[HardwareInfo] = None
_HARDWARE_LOCK = threading.Lock()


def clear_hardware_cache() -> None:
    """Clears cached hardware detection info."""
    global _CACHED_HARDWARE_INFO
    with _HARDWARE_LOCK:
        _CACHED_HARDWARE_INFO = None


def get_hardware_info(force_refresh: bool = False) -> HardwareInfo:
    """Collects and returns complete system hardware and runtime capabilities.

    Results are cached to avoid repeated, slow subprocess queries (nvidia-smi, ffmpeg).
    """
    global _CACHED_HARDWARE_INFO
    if not force_refresh:
        with _HARDWARE_LOCK:
            if _CACHED_HARDWARE_INFO is not None:
                return _CACHED_HARDWARE_INFO

    base_dir = get_base_dir()
    gpu_name, gpu_vram, has_nvidia = detect_nvidia_gpu()
    ffmpeg_p, ffprobe_p, ffmpeg_v, has_hevc, has_h264 = detect_ffmpeg()
    realesrgan_b, realcugan_b = detect_ai_binaries(base_dir)
    cpu_cores = os.cpu_count() or 4

    hw = HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        has_nvidia=has_nvidia,
        has_nvenc_hevc=has_nvidia and has_hevc,
        has_nvenc_h264=has_nvidia and has_h264,
        ffmpeg_path=ffmpeg_p,
        ffprobe_path=ffprobe_p,
        ffmpeg_version=ffmpeg_v,
        realesrgan_bin=realesrgan_b,
        realcugan_bin=realcugan_b,
        cpu_cores=cpu_cores
    )
    with _HARDWARE_LOCK:
        _CACHED_HARDWARE_INFO = hw
    return hw


def generate_diagnostic_report(redact_paths: bool = False) -> str:
    """Generates a detailed diagnostic report of system hardware and environment."""
    hw = get_hardware_info()
    base_dir = get_base_dir()

    ffmpeg_disp = hw.ffmpeg_path or "Missing (Please install FFmpeg)"
    realesrgan_disp = hw.realesrgan_bin or "Missing"
    realcugan_disp = hw.realcugan_bin or "Missing"

    lines = [
        "=" * 60,
        "           SYSTEM ENVIRONMENT DIAGNOSTIC",
        "=" * 60,
        f"Platform             : {sys.platform} ({os.name})",
        f"Python Version       : {sys.version.split()[0]}",
        f"CPU Logical Cores    : {hw.cpu_cores}",
        f"GPU Name             : {hw.gpu_name}",
        f"GPU VRAM             : {hw.gpu_vram_gb} GB",
        f"NVIDIA GPU Detected  : {'Yes' if hw.has_nvidia else 'No'}",
        f"NVENC H.265 (HEVC)   : {'Available' if hw.has_nvenc_hevc else 'Not Available'}",
        f"NVENC H.264 (AVC)    : {'Available' if hw.has_nvenc_h264 else 'Not Available'}",
        f"FFmpeg Path          : {ffmpeg_disp}",
        f"FFmpeg Version       : {hw.ffmpeg_version}",
        f"Real-ESRGAN Binary   : {realesrgan_disp}",
        f"RealCUGAN Binary     : {realcugan_disp}",
        "-" * 60,
        "           LOCAL AI MODELS STORAGE (OFFLINE)",
        "-" * 60,
    ]

    from src.core.models import check_local_model_availability
    models_avail = check_local_model_availability(base_dir)
    all_models_ok = True
    for m_id, ok in models_avail.items():
        if m_id == "auto":
            continue
        status = "Verified Local (Ready)" if ok else "Missing Model Files"
        if not ok:
            all_models_ok = False
        lines.append(f"  {m_id:<22} : {status}")

    lines.append("=" * 60)
    if hw.has_nvidia and hw.has_nvenc_hevc and hw.ffmpeg_path and hw.realesrgan_bin and all_models_ok:
        lines.append("Status: Everything is optimally configured for 4K GPU upscaling (100% local).")
    else:
        lines.append("Status: Issues detected. Verify missing dependencies above.")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    if redact_paths:
        user_home = os.path.expanduser("~")
        for orig in (user_home, user_home.replace("\\", "/")):
            if len(orig) > 3:
                report_text = report_text.replace(orig, "<USER_HOME>")
        if "<USER_HOME>" not in report_text:
            report_text += "\n[Privacy Masking: <USER_HOME> enabled]"

    return report_text


def export_diagnostic_report(output_path: Optional[str] = None, redact_paths: bool = True) -> str:
    """Exports diagnostic report to file with sensitive paths redacted by default."""
    report = generate_diagnostic_report(redact_paths=redact_paths)
    if output_path:
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report


def print_doctor_report() -> None:
    """Prints diagnostic report directly to standard output."""
    print(generate_diagnostic_report(redact_paths=False))
