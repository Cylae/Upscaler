"""Hardware-accelerated ONNX Runtime inference backend.

Prioritizes execution providers dynamically:
    TensorRT -> CUDA -> DirectML (DmlExecutionProvider) -> CPU
Tunes graph optimization levels, memory arena allocations, and thread topology,
with dynamic shape inspection and automatic fallback on hardware exceptions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.engine.tensor import (
    ColorFormat,
    Normalization,
    TensorLayout,
    image_to_tensor,
    pad_to_multiple,
    tensor_to_image,
    unpad,
)
from src.engine.tiler import SpatialTiler

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

HAS_ONNXRUNTIME = _ORT_AVAILABLE

logger = logging.getLogger(__name__)

# Standard Execution Provider priority hierarchy
DEFAULT_PROVIDER_PRIORITY: list[str] = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class ONNXConfig:
    """Tuning and configuration parameters for ONNX Runtime InferenceSession."""
    preferred_providers: list[str] = field(default_factory=lambda: list(DEFAULT_PROVIDER_PRIORITY))
    device_id: int = 0
    enable_fp16: bool = False
    graph_optimization_level: str = "all"  # "all", "extended", "basic", "disabled"
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int = 1
    enable_memory_arena: bool = True
    enable_mem_pattern: bool = True
    trt_engine_cache_enable: bool = True
    trt_cache_path: str | None = None
    cuda_mem_limit: int | None = None  # in bytes, None for unlimited
    color_format: ColorFormat = ColorFormat.RGB
    normalization: Normalization = Normalization.ZERO_TO_ONE
    scale: int = 4


@dataclass(frozen=True, slots=True)
class ModelIOSpec:
    """Inspection metadata for model input and output tensors."""
    name: str
    shape: list[int | str | None]
    dtype: str
    is_dynamic: bool
    static_spatial_dims: tuple[int, int] | None  # (height, width) if fixed


class ONNXInferenceEngine:
    """Production-grade ONNX neural inference engine with automated hardware fallback."""

    def __init__(
        self,
        model_path: str,
        config: ONNXConfig | None = None,
        scale: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. Please install onnxruntime or onnxruntime-directml/gpu."
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.config = config or ONNXConfig()
        if scale is not None:
            self.config.scale = scale
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.session: ort.InferenceSession | None = None
        self.active_provider: str = "None"
        self.active_provider_options: dict[str, Any] = {}

        self.input_spec: ModelIOSpec | None = None
        self.output_spec: ModelIOSpec | None = None

        # Initialize session with provider fallback hierarchy
        self._initialize_session()

    def _create_session_options(self) -> ort.SessionOptions:
        """Configures graph optimizations, memory arenas, and thread concurrency."""
        opts = ort.SessionOptions()

        # Graph optimization level
        level_map = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }
        opts.graph_optimization_level = level_map.get(
            self.config.graph_optimization_level.lower(),
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

        # Memory allocations
        opts.enable_mem_pattern = self.config.enable_mem_pattern
        opts.enable_cpu_mem_arena = self.config.enable_memory_arena

        # CPU thread limits
        intra_threads = self.config.intra_op_num_threads
        if intra_threads is None:
            intra_threads = min(os.cpu_count() or 4, 8)
        opts.intra_op_num_threads = max(1, intra_threads)
        opts.inter_op_num_threads = max(1, self.config.inter_op_num_threads)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        return opts

    def _build_provider_options(self, provider: str) -> dict[str, Any]:
        """Builds hardware-tuned options for a specific execution provider."""
        opts: dict[str, Any] = {}

        if provider == "TensorrtExecutionProvider":
            opts["device_id"] = self.config.device_id
            opts["trt_fp16_enable"] = self.config.enable_fp16
            if self.config.trt_engine_cache_enable:
                opts["trt_engine_cache_enable"] = True
                cache_dir = self.config.trt_cache_path or os.path.join(
                    os.path.dirname(self.model_path), ".trt_cache"
                )
                os.makedirs(cache_dir, exist_ok=True)
                opts["trt_engine_cache_path"] = cache_dir

        elif provider == "CUDAExecutionProvider":
            opts["device_id"] = self.config.device_id
            opts["arena_extend_strategy"] = "kNextPowerOfTwo"
            opts["cudnn_conv_algo_search"] = "EXHAUSTIVE"
            opts["do_copy_in_default_stream"] = True
            if self.config.cuda_mem_limit:
                opts["gpu_mem_limit"] = self.config.cuda_mem_limit

        elif provider == "DmlExecutionProvider":
            opts["device_id"] = self.config.device_id

        elif provider == "CPUExecutionProvider":
            opts["arena_extend_strategy"] = "kSameAsRequested"

        return opts

    def _initialize_session(self) -> None:
        """Iterates through preferred execution providers and binds the first working backend."""
        available = ort.get_available_providers()
        logger.info("Available ONNX Execution Providers on system: %s", available)

        # Filter and prioritize
        candidates = [p for p in self.config.preferred_providers if p in available]
        if not candidates:
            # Fall back to CPU unconditionally if none matched
            candidates = ["CPUExecutionProvider"]

        session_opts = self._create_session_options()
        last_error: Exception | None = None

        for provider in candidates:
            provider_opts = self._build_provider_options(provider)
            try:
                logger.info("Attempting to initialize ONNX session with %s...", provider)
                sess = ort.InferenceSession(
                    self.model_path,
                    sess_options=session_opts,
                    providers=[provider],
                    provider_options=[provider_opts],
                )
                self.session = sess
                self.active_provider = provider
                self.active_provider_options = provider_opts
                logger.info("Successfully bound ONNX session to %s", provider)
                break
            except Exception as exc:
                logger.warning(
                    "Execution Provider %s failed initialization: %s. Falling back...",
                    provider,
                    exc,
                )
                last_error = exc

        if self.session is None:
            raise RuntimeError(
                f"Failed to initialize ONNX session for {self.model_path} across all providers "
                f"{candidates}. Last error: {last_error}"
            )

        self._inspect_io()

    def _inspect_io(self) -> None:
        """Inspects inputs and outputs of the initialized ONNX model."""
        assert self.session is not None

        # Inspect input
        raw_inputs = self.session.get_inputs()
        if not raw_inputs:
            raise ValueError(f"Model {self.model_path} has no inputs.")
        first_in = raw_inputs[0]
        in_shape = list(first_in.shape)
        is_dynamic = any(dim is None or isinstance(dim, str) or dim <= 0 for dim in in_shape)

        static_spatial: tuple[int, int] | None = None
        if len(in_shape) == 4 and not is_dynamic:
            # Assuming NCHW: (B, C, H, W)
            static_spatial = (int(in_shape[2]), int(in_shape[3]))  # type: ignore[arg-type]

        self.input_spec = ModelIOSpec(
            name=first_in.name,
            shape=in_shape,
            dtype=first_in.type,
            is_dynamic=is_dynamic,
            static_spatial_dims=static_spatial,
        )

        # Inspect output
        raw_outputs = self.session.get_outputs()
        if not raw_outputs:
            raise ValueError(f"Model {self.model_path} has no outputs.")
        first_out = raw_outputs[0]
        out_shape = list(first_out.shape)
        out_is_dynamic = any(dim is None or isinstance(dim, str) or dim <= 0 for dim in out_shape)

        out_static_spatial: tuple[int, int] | None = None
        if len(out_shape) == 4 and not out_is_dynamic:
            out_static_spatial = (int(out_shape[2]), int(out_shape[3]))  # type: ignore[arg-type]

        self.output_spec = ModelIOSpec(
            name=first_out.name,
            shape=out_shape,
            dtype=first_out.type,
            is_dynamic=out_is_dynamic,
            static_spatial_dims=out_static_spatial,
        )

        # Deduce scale factor if static
        if static_spatial and out_static_spatial:
            in_h, _ = static_spatial
            out_h, _ = out_static_spatial
            if in_h > 0:
                self.config.scale = round(out_h / in_h)

        logger.info(
            "Model I/O specs: Input '%s' %s (%s) -> Output '%s' %s (%s) [scale=%dx]",
            self.input_spec.name,
            self.input_spec.shape,
            self.input_spec.dtype,
            self.output_spec.name,
            self.output_spec.shape,
            self.output_spec.dtype,
            self.config.scale,
        )

    def run_raw(self, tensor: np.ndarray) -> np.ndarray:
        """Executes raw inference with automatic provider recovery on GPU failure."""
        if self.session is None or self.input_spec is None or self.output_spec is None:
            raise RuntimeError("ONNX Session is not initialized.")

        # Ensure correct dtype
        target_dtype = np.float16 if "float16" in self.input_spec.dtype else np.float32
        tensor_in = tensor.astype(target_dtype, copy=False)

        feed = {self.input_spec.name: tensor_in}

        try:
            outputs = self.session.run([self.output_spec.name], feed)
            return outputs[0]
        except Exception as exc:
            # If a GPU execution provider crashes (e.g. CUDA OOM or DML device removal), fall back to CPU
            if self.active_provider != "CPUExecutionProvider":
                logger.error(
                    "Inference failed on %s with error: %s. Initiating emergency fallback to CPUExecutionProvider.",
                    self.active_provider,
                    exc,
                )
                self._fallback_to_cpu()
                outputs = self.session.run([self.output_spec.name], feed)
                return outputs[0]
            raise

    def _fallback_to_cpu(self) -> None:
        """Emergency fallback replacing the active session with CPU execution."""
        session_opts = self._create_session_options()
        cpu_opts = self._build_provider_options("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
            provider_options=[cpu_opts],
        )
        self.active_provider = "CPUExecutionProvider"
        self.active_provider_options = cpu_opts
        logger.warning("Session successfully migrated to CPUExecutionProvider fallback.")

    def upscale_tile(self, tile_hwc: np.ndarray) -> np.ndarray:
        """Processes a single HWC uint8/float tile through tensor prep, inference, and conversion."""
        # 1. Image -> NCHW FP32/FP16 Tensor
        tensor = image_to_tensor(
            tile_hwc,
            layout=TensorLayout.NCHW,
            color_format=self.config.color_format,
            normalization=self.config.normalization,
            dtype=np.float32,
            add_batch_dim=True,
        )

        # 2. Check if padding to multiple or fixed shape is required
        unpad_info = None
        if self.input_spec and self.input_spec.static_spatial_dims:
            expected_h, expected_w = self.input_spec.static_spatial_dims
            cur_h, cur_w = tensor.shape[2], tensor.shape[3]
            if cur_h != expected_h or cur_w != expected_w:
                raise ValueError(
                    f"Tile dimension ({cur_h}, {cur_w}) does not match static model input requirement "
                    f"({expected_h}, {expected_w})"
                )
        else:
            # UNet-style models typically require divisibility by 32
            tensor, unpad_info = pad_to_multiple(tensor, multiple=32, layout=TensorLayout.NCHW)

        # 3. Model inference
        out_tensor = self.run_raw(tensor)

        # 4. Unpad if padded
        if unpad_info is not None and (unpad_info.top > 0 or unpad_info.left > 0):
            out_tensor = unpad(out_tensor, unpad_info, scale=self.config.scale)

        # 5. Tensor -> HWC uint8
        out_hwc = tensor_to_image(
            out_tensor,
            layout=TensorLayout.NCHW,
            color_format=self.config.color_format,
            target_format=self.config.color_format,
            normalization=self.config.normalization,
            to_pil=False,
            clamp=True,
        )
        return np.asarray(out_hwc)

    def upscale_image(
        self,
        image: Image.Image | np.ndarray,
        tile_size: int = 512,
        overlap: int = 64,
        progress_callback: Any = None,
    ) -> np.ndarray:
        """Upscales an entire image using spatial cosine-weighted tiled blending.

        Args:
            image: PIL Image or HWC NumPy array.
            tile_size: Tile spatial dimension (e.g. 256, 384, 512, 768, 1024).
            overlap: Tile boundary feathering overlap (e.g. 32, 48, 64).
            progress_callback: Callback receiving (completed_tiles, total_tiles).

        Returns:
            Seamlessly upscaled HWC uint8 image.
        """
        # If static spatial dimensions are enforced by model, use model's tile size
        if self.input_spec and self.input_spec.static_spatial_dims:
            sh, sw = self.input_spec.static_spatial_dims
            tile_size = min(sh, sw)
            overlap = min(overlap, tile_size // 4)

        tiler = SpatialTiler(
            tile_size=tile_size,
            overlap=overlap,
            scale=self.config.scale,
        )

        return tiler.process(
            image=image,
            inference_fn=self.upscale_tile,
            progress_callback=progress_callback,
            out_dtype=np.uint8,
        )
