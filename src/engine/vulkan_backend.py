"""Native integration with portable Vulkan NCNN neural upscaling binaries.

Provides high-performance, direct execution of:
    - Real-ESRGAN (realesrgan-ncnn-vulkan)
    - RealCUGAN (realcugan-ncnn-vulkan)
with memory-bounded tile sizing, multi-threaded pipeline topology,
in-memory image streaming, and robust process supervision.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Windows creation flag to suppress command prompt window popup
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


@dataclass
class VulkanConfig:
    """Execution parameters for Vulkan NCNN neural upscaling engines."""
    scale: int = 4
    tile_size: int = 0          # 0 = auto-calculate based on VRAM, or explicit (e.g. 256, 400, 512)
    gpu_id: int | str = "auto"  # "auto", 0, 1, or -1 for CPU
    threads: str = "1:2:2"      # load:proc:save thread count
    enable_tta: bool = False    # Test-time augmentation (8x slower, higher quality)
    output_format: str = "png"  # "png", "jpg", "webp"

    # RealCUGAN specific
    model_family: str = "models-se"  # "models-se", "models-pro", "models-nose"
    denoise_level: int = -1          # -1 (conservative), 0, 1, 2, 3
    sync_gap_mode: int = 3           # sync gap mode (0/1/2/3)


def get_project_root() -> str:
    """Returns repository root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def find_vulkan_binary(engine_name: str, custom_path: str | None = None) -> str:
    """Locates the portable Vulkan NCNN executable on disk or PATH.

    Args:
        engine_name: "realesrgan" or "realcugan".
        custom_path: User-provided binary path override.

    Returns:
        Absolute path to the executable binary.

    Raises:
        FileNotFoundError: If the binary cannot be located.
    """
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    base_dir = get_project_root()
    ext = ".exe" if os.name == "nt" else ""
    filename = f"{engine_name}-ncnn-vulkan{ext}"

    # 1. Search in tools/<engine_name>/
    candidate = os.path.join(base_dir, "tools", engine_name, filename)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # 2. Search on system PATH
    which_path = shutil.which(filename)
    if which_path and os.path.isfile(which_path):
        return os.path.abspath(which_path)

    raise FileNotFoundError(
        f"Vulkan engine '{engine_name}' not found. Expected executable at: {candidate} "
        f"or available on PATH."
    )


class VulkanNCNNBackend:
    """High-performance manager for portable Vulkan NCNN neural models."""

    def __init__(
        self,
        engine_type: str = "realesrgan",
        model_name: str = "realesr-animevideov3",
        binary_path: str | None = None,
        models_dir: str | None = None,
        config: VulkanConfig | None = None,
        scale: int | None = None,
        tile_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.engine_type = engine_type.lower()
        if self.engine_type not in ("realesrgan", "realcugan"):
            raise ValueError(f"Unknown Vulkan engine type: {self.engine_type}. Expected 'realesrgan' or 'realcugan'.")

        self.model_name = model_name
        self.config = config or VulkanConfig()
        if scale is not None:
            self.config.scale = scale
        if tile_size is not None:
            self.config.tile_size = tile_size
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.binary_path = find_vulkan_binary(self.engine_type, binary_path)

        # Resolve models directory
        base_dir = os.path.dirname(self.binary_path)
        if models_dir and os.path.isdir(models_dir):
            self.models_dir = os.path.abspath(models_dir)
        else:
            if self.engine_type == "realesrgan":
                self.models_dir = os.path.join(base_dir, "models")
            else:
                self.models_dir = os.path.join(base_dir, self.config.model_family)

        self._active_process: subprocess.Popen[str] | None = None

    def build_command_args(
        self,
        input_path: str,
        output_path: str,
        scale: int | None = None,
        tile_size: int | None = None,
    ) -> list[str]:
        """Constructs CLI arguments tailored to the specific engine and model."""
        s = scale if scale is not None else self.config.scale
        t = tile_size if tile_size is not None else self.config.tile_size

        cmd = [
            self.binary_path,
            "-i", os.path.abspath(input_path),
            "-o", os.path.abspath(output_path),
            "-s", str(s),
            "-t", str(t),
            "-m", self.models_dir,
            "-g", str(self.config.gpu_id),
            "-j", str(self.config.threads),
            "-f", self.config.output_format,
        ]

        if self.config.enable_tta:
            cmd.append("-x")

        if self.engine_type == "realesrgan":
            cmd.extend(["-n", self.model_name])
        elif self.engine_type == "realcugan":
            cmd.extend([
                "-n", str(self.config.denoise_level),
                "-c", str(self.config.sync_gap_mode),
            ])

        return cmd

    def run_file(
        self,
        input_path: str,
        output_path: str,
        scale: int | None = None,
        tile_size: int | None = None,
        timeout: float = 300.0,
    ) -> None:
        """Executes upscaling on a single image file or directory with process supervision."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cmd = self.build_command_args(input_path, output_path, scale=scale, tile_size=tile_size)

        logger.info("Executing Vulkan NCNN command: %s", " ".join(cmd))
        try:
            self._active_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=CREATE_NO_WINDOW,
            )
            stdout, stderr = self._active_process.communicate(timeout=timeout)
            ret_code = self._active_process.returncode

            if ret_code != 0:
                err_msg = stderr.strip() if stderr else stdout.strip()
                raise RuntimeError(
                    f"Vulkan NCNN engine '{self.engine_type}' exited with code {ret_code}: {err_msg}"
                )

            if not os.path.exists(output_path) and not os.path.isdir(input_path):
                raise RuntimeError(
                    f"Vulkan NCNN finished with code 0 but output file was not created: {output_path}"
                )

        except subprocess.TimeoutExpired:
            self.cancel()
            raise TimeoutError(f"Vulkan NCNN upscaling timed out after {timeout} seconds.")
        finally:
            self._active_process = None

    def cancel(self) -> None:
        """Terminates active running Vulkan subprocess cleanly."""
        if self._active_process is not None:
            try:
                self._active_process.terminate()
                self._active_process.wait(timeout=2.0)
            except Exception:
                try:
                    self._active_process.kill()
                except Exception:
                    pass
            finally:
                self._active_process = None

    def upscale_image(
        self,
        image: Image.Image | np.ndarray,
        scale: int | None = None,
        tile_size: int | None = None,
        timeout: float = 120.0,
    ) -> np.ndarray:
        """In-memory image upscaling using fast atomic temporary file exchange.

        Args:
            image: PIL Image or HWC NumPy array.
            scale: Target scale factor.
            tile_size: Spatial tile size (0 for auto).
            timeout: Subprocess execution timeout in seconds.

        Returns:
            Contiguous HWC uint8 upscaled NumPy array.
        """
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)

        # Create temporary working directory for zero-leak IO exchange
        with tempfile.TemporaryDirectory(prefix="vulkan_ncnn_") as tmp_dir:
            in_file = os.path.join(tmp_dir, "input.png")
            out_file = os.path.join(tmp_dir, "output.png")

            pil_img.save(in_file, format="PNG")
            self.run_file(in_file, out_file, scale=scale, tile_size=tile_size, timeout=timeout)

            with Image.open(out_file) as result_img:
                result_img.load()
                out_arr = np.asarray(result_img)

        return np.ascontiguousarray(out_arr)
