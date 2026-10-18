"""Hardware detection facade.

Re-exports all hardware detection functions and classes from hardware.py.
"""

from __future__ import annotations

from src.core.hardware import (
    HardwareInfo,
    clear_hardware_cache,
    detect_ai_binaries,
    detect_ffmpeg,
    detect_nvidia_gpu,
    export_diagnostic_report,
    generate_diagnostic_report,
    get_base_dir,
    get_hardware_info,
    print_doctor_report,
    resolve_ai_binary,
)

__all__ = [
    "HardwareInfo",
    "clear_hardware_cache",
    "detect_ai_binaries",
    "detect_ffmpeg",
    "detect_nvidia_gpu",
    "export_diagnostic_report",
    "generate_diagnostic_report",
    "get_base_dir",
    "get_hardware_info",
    "print_doctor_report",
    "resolve_ai_binary",
]
