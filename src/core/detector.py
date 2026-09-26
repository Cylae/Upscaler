import os
import shutil
import subprocess
import sys
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

        # Check encoders
        enc_output = subprocess.check_output(
            [ffmpeg_path, "-encoders"], stderr=subprocess.STDOUT,
            text=True, errors="ignore", timeout=15
        )
        if "hevc_nvenc" in enc_output:
            has_hevc_nvenc = True
        if "h264_nvenc" in enc_output:
            has_h264_nvenc = True
    except Exception:
        pass

    return ffmpeg_path, ffprobe_path, version_str, has_hevc_nvenc, has_h264_nvenc


def detect_gpu() -> Tuple[str, float, bool]:
    """Detects primary GPU and VRAM using nvidia-smi or Windows CIM/WMI."""
    gpu_name = "Unknown"
    gpu_vram_gb = 0.0
    has_nvidia = False

    # Try nvidia-smi first
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            errors="ignore",
            timeout=10,
        ).strip()
        if smi_out:
            lines = smi_out.splitlines()
            parts = lines[0].split(",")
            gpu_name = parts[0].strip()
            if len(parts) > 1:
                gpu_vram_gb = round(float(parts[1].strip()) / 1024.0, 1)
            has_nvidia = True
            return gpu_name, gpu_vram_gb, has_nvidia
    except Exception:
        pass

    # Fallback to PowerShell Win32_VideoController
    try:
        ps_cmd = "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name, AdapterRAM | ConvertTo-Json"
        ps_out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stderr=subprocess.DEVNULL, text=True, errors="ignore", timeout=10
        )
        import json
        data = json.loads(ps_out)
        gpu_name = data.get("Name", "Unknown")
        ram_bytes = data.get("AdapterRAM", 0)
        if ram_bytes:
            gpu_vram_gb = round(float(ram_bytes) / (1024**3), 1)
        if "nvidia" in gpu_name.lower():
            has_nvidia = True
    except Exception:
        pass

    return gpu_name, gpu_vram_gb, has_nvidia


def detect_tools() -> Tuple[Optional[str], Optional[str]]:
    """Detects portable Real-ESRGAN and RealCUGAN binaries in tools/."""
    base_dir = get_base_dir()
    realesrgan_exe = os.path.join(base_dir, "tools", "realesrgan", "realesrgan-ncnn-vulkan.exe")
    realcugan_exe = os.path.join(base_dir, "tools", "realcugan", "realcugan-ncnn-vulkan.exe")

    realesrgan_path = realesrgan_exe if os.path.isfile(realesrgan_exe) else None
    realcugan_path = realcugan_exe if os.path.isfile(realcugan_exe) else None

    return realesrgan_path, realcugan_path


def get_hardware_info() -> HardwareInfo:
    """Gathers all hardware and environment information into a dataclass."""
    gpu_name, gpu_vram, has_nvidia = detect_gpu()
    ffmpeg_path, ffprobe_path, ffmpeg_ver, has_hevc, has_h264 = detect_ffmpeg()
    realesrgan_bin, realcugan_bin = detect_tools()

    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        has_nvidia=has_nvidia,
        has_nvenc_hevc=has_nvidia and has_hevc,
        has_nvenc_h264=has_nvidia and has_h264,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        ffmpeg_version=ffmpeg_ver,
        realesrgan_bin=realesrgan_bin,
        realcugan_bin=realcugan_bin,
        cpu_cores=os.cpu_count() or 4
    )


def _redact_path(p: Optional[str]) -> str:
    """Redacts user home directory from paths to protect privacy."""
    if not p:
        return "Not Found"
    try:
        user_home = os.path.expanduser("~")
        if user_home and user_home in p:
            return p.replace(user_home, "<USER_HOME>")
    except Exception:
        pass
    return p


def generate_diagnostic_report(redact_paths: bool = False) -> str:
    """Generates a complete diagnostic report with optional path redaction for privacy."""
    hw = get_hardware_info()
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
        f"FFmpeg Path          : {_redact_path(hw.ffmpeg_path) if redact_paths else (hw.ffmpeg_path or 'Not Found')}",
        f"FFmpeg Version       : {hw.ffmpeg_version}",
        f"Real-ESRGAN Binary   : {_redact_path(hw.realesrgan_bin) if redact_paths else (hw.realesrgan_bin or 'Not Found')}",
        f"RealCUGAN Binary     : {_redact_path(hw.realcugan_bin) if redact_paths else (hw.realcugan_bin or 'Not Found')}",
        "-" * 60,
        "           LOCAL AI MODELS STORAGE (OFFLINE)",
        "-" * 60,
    ]
    from src.core.ai_models import check_local_model_availability
    models_status = check_local_model_availability(get_base_dir())
    for mid, ok in models_status.items():
        if mid == "auto":
            continue
        status_str = "Verified Local (Ready)" if ok else "Missing"
        lines.append(f"  {mid:<23}: {status_str}")
    lines.append("=" * 60)
    all_models_ok = all(models_status.values())
    if hw.has_nvidia and hw.has_nvenc_hevc and hw.ffmpeg_path and hw.realesrgan_bin and all_models_ok:
        lines.append("Status: Everything is optimally configured for 4K GPU upscaling (100% local).")
    else:
        lines.append("Notice: Some acceleration components were not detected.")
    lines.append("=" * 60)
    return "\n".join(lines)


def export_diagnostic_report(output_path: Optional[str] = None, redact_paths: bool = True) -> str:
    """Exports diagnostic report to a text file or returns it as a string."""
    report = generate_diagnostic_report(redact_paths=redact_paths)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report


def print_doctor_report():
    """Prints diagnostic report of the system environment to stdout."""
    print(generate_diagnostic_report(redact_paths=False))

