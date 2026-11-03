"""High-Level Engine Orchestrator and Unified Hardware Manager.

Seamlessly bridges DirectML/CUDA/TensorRT/CPU ONNX inference and Vulkan NCNN engines
with automatic VRAM budgeting and zero-disk streaming media pipelines.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any, Callable
import numpy as np

from src.engine.media_stream import ZeroDiskStreamingPipeline
from src.engine.onnx_backend import HAS_ONNXRUNTIME, ONNXInferenceEngine
from src.engine.vulkan_backend import VulkanNCNNBackend

logger = logging.getLogger(__name__)


class EngineOrchestrator:
    """Unified entry point for AI video and image super-resolution."""

    def __init__(
        self,
        backend_type: str = "auto",
        model_name_or_path: str = "auto",
        scale: int = 2,
        tile_size: int = 0,
        overlap: int = 32,
        encoder: str = "auto",
        crf: int = 18,
    ):
        self.backend_type = backend_type.lower()
        self.model_name_or_path = model_name_or_path
        self.scale = scale
        self.tile_size = tile_size
        self.overlap = overlap
        self.encoder = encoder
        self.crf = crf

        self.hardware_info = self.detect_system_hardware()
        self.resolved_encoder = self._resolve_encoder(encoder)
        self.effective_tile_size = self._resolve_tile_size(tile_size)

        self._active_stream: ZeroDiskStreamingPipeline | None = None

    @staticmethod
    def detect_system_hardware() -> dict[str, Any]:
        """Probes system hardware capabilities (GPU model, VRAM, DirectML, NVENC, Vulkan)."""
        gpu_name = "CPU / Integrated"
        gpu_vram_gb = 0.0
        has_nvidia = False
        has_directml = False
        has_cuda = False
        has_vulkan_binaries = False

        # 1. Probe ONNX Providers
        if HAS_ONNXRUNTIME:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            has_directml = "DmlExecutionProvider" in providers
            has_cuda = "CUDAExecutionProvider" in providers

        # 2. Fast sub-millisecond Windows Registry enumeration + WMI fallback
        if os.name == "nt":
            try:
                import winreg
                key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as k:
                    for i in range(10):
                        try:
                            subkey_name = winreg.EnumKey(k, i)
                            with winreg.OpenKey(k, subkey_name) as subk:
                                try:
                                    desc = str(winreg.QueryValueEx(subk, "DriverDesc")[0]).strip()
                                    low_desc = desc.lower()
                                    if desc and not any(v in low_desc for v in ("basic render", "remote", "rdp", "virtual", "vga")):
                                        gpu_name = desc
                                        if any(nv in low_desc for nv in ("geforce", "rtx", "gtx", "quadro", "tesla", "nvidia")):
                                            has_nvidia = True
                                        if any(igpu in low_desc for igpu in ("intel", "iris", "uhd", "radeon", "vega", "arc")):
                                            gpu_vram_gb = 2.0
                                        break
                                except FileNotFoundError:
                                    pass
                        except OSError:
                            break
            except Exception:
                pass

            try:
                cmd = [
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json -Compress"
                ]
                raw = subprocess.check_output(cmd, text=True, errors="ignore", timeout=6).strip()
                if raw:
                    import json
                    data = json.loads(raw)
                    if isinstance(data, list):
                        data = data[0]
                    name = data.get("Name", "")
                    if name and gpu_name in ("CPU / Integrated", "Unknown"):
                        gpu_name = name
                    raw_ram = data.get("AdapterRAM", 0) or 0
                    if raw_ram < 0:
                        raw_ram = raw_ram & 0xFFFFFFFF
                    ram_gb = round(raw_ram / (1024**3), 1)
                    if ram_gb > 0:
                        gpu_vram_gb = ram_gb
                    elif gpu_vram_gb == 0.0 and any(igpu in gpu_name.lower() for igpu in ("intel", "iris", "uhd", "radeon", "vega")):
                        gpu_vram_gb = 2.0
                    if any(nv in gpu_name.lower() for nv in ("geforce", "rtx", "gtx", "quadro", "tesla", "nvidia")):
                        has_nvidia = True
            except Exception:
                pass

        # 3. Check portable Vulkan binaries
        cwd = os.getcwd()
        re_bin = os.path.join(cwd, "tools", "realesrgan", "realesrgan-ncnn-vulkan.exe")
        rc_bin = os.path.join(cwd, "tools", "realcugan", "realcugan-ncnn-vulkan.exe")
        has_vulkan_binaries = os.path.isfile(re_bin) or os.path.isfile(rc_bin) or (shutil.which("realesrgan-ncnn-vulkan") is not None)

        return {
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram_gb,
            "has_nvidia": has_nvidia,
            "has_directml": has_directml,
            "has_cuda": has_cuda,
            "has_vulkan_binaries": has_vulkan_binaries,
            "cpu_cores": os.cpu_count() or 4,
        }

    def _resolve_tile_size(self, requested_tile: int) -> int:
        if requested_tile > 0:
            return requested_tile

        vram = self.hardware_info.get("gpu_vram_gb", 0.0)
        if vram <= 2.0:
            return 384
        elif vram <= 6.0:
            return 512
        elif vram <= 12.0:
            return 768
        return 1024

    def _resolve_encoder(self, requested_encoder: str) -> str:
        if requested_encoder != "auto":
            return requested_encoder

        # Check ffmpeg encoder support
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        try:
            res = subprocess.check_output([ffmpeg_bin, "-encoders"], stderr=subprocess.STDOUT, text=True, errors="ignore")
            if "hevc_nvenc" in res and self.hardware_info.get("has_nvidia", False):
                return "hevc_nvenc"
            if "hevc_qsv" in res and "intel" in self.hardware_info.get("gpu_name", "").lower():
                return "hevc_qsv"
            if "hevc_amf" in res and "radeon" in self.hardware_info.get("gpu_name", "").lower():
                return "hevc_amf"
        except Exception:
            pass
        return "libx265"

    def upscale_image_file(
        self,
        input_image_path: str,
        output_image_path: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> bool:
        """Upscales an individual image file using seamless feathered tiling."""
        from PIL import Image

        if not os.path.isfile(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        if progress_callback:
            progress_callback(0.1, "Loading image...")

        with Image.open(input_image_path) as pil_img:
            img_arr = np.array(pil_img)

        # Build inference function
        if os.path.isfile(self.model_name_or_path):
            engine = ONNXInferenceEngine(self.model_name_or_path, scale=self.scale)
            upscaled = engine.upscale_image(img_arr, tile_size=self.effective_tile_size, overlap=self.overlap)
        else:
            vulkan = VulkanNCNNBackend(
                engine_type="realesrgan" if "cugan" not in self.model_name_or_path else "realcugan",
                scale=self.scale,
                tile_size=self.effective_tile_size,
            )
            upscaled = vulkan.upscale_image(img_arr)

        if progress_callback:
            progress_callback(0.9, "Saving upscaled image...")

        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        out_pil = Image.fromarray(upscaled)
        out_pil.save(output_image_path)

        if progress_callback:
            progress_callback(1.0, "Complete")

        return True

    def upscale_video_file(
        self,
        input_video_path: str,
        output_video_path: str,
        preview_duration_s: float | None = None,
        progress_callback: Callable[[float, str, dict[str, Any]], None] | None = None,
    ) -> tuple[bool, str]:
        """Upscales a video file end-to-end via zero-disk in-memory streaming pipes."""
        if not os.path.isfile(input_video_path):
            return False, f"Input video not found: {input_video_path}"

        # Initialize inference function
        if os.path.isfile(self.model_name_or_path):
            onnx_engine = ONNXInferenceEngine(self.model_name_or_path, scale=self.scale)
            tile_sz = self.effective_tile_size
            ovlp = self.overlap

            def _infer(frame: np.ndarray) -> np.ndarray:
                return onnx_engine.upscale_image(frame, tile_size=tile_sz, overlap=ovlp)
        else:
            vulkan = VulkanNCNNBackend(
                engine_type="realesrgan" if "cugan" not in self.model_name_or_path else "realcugan",
                scale=self.scale,
                tile_size=self.effective_tile_size,
            )

            def _infer(frame: np.ndarray) -> np.ndarray:
                return vulkan.upscale_image(frame)

        pipeline = ZeroDiskStreamingPipeline(
            source_path=input_video_path,
            output_path=output_video_path,
            upscale_fn=_infer,
            scale=self.scale,
            encoder=self.resolved_encoder,
            crf=self.crf,
            preview_duration_s=preview_duration_s,
            progress_callback=progress_callback,
        )
        self._active_stream = pipeline

        try:
            return pipeline.process()
        finally:
            self._active_stream = None

    def cancel(self) -> None:
        """Cancels any active video streaming pipeline immediately."""
        if self._active_stream is not None:
            self._active_stream.cancel()
