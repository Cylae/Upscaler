"""AnimaScale Next-Gen Neural Video & Image Upscaling Engine.

Zero-disk streaming media architecture with hardware-accelerated DirectML/TensorRT/CUDA/CPU/Vulkan inference,
mathematically seamless 2D cosine-weighted feathered tiling, and bounded-memory concurrency.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "2.0.0"

_EXPORTS: dict[str, str] = {
    # Media Stream
    "MuxerConfig": "src.engine.media_stream",
    "StreamPipelineController": "src.engine.media_stream",
    "StreamProgress": "src.engine.media_stream",
    "VideoFrame": "src.engine.media_stream",
    "VideoStreamDemuxer": "src.engine.media_stream",
    "VideoStreamMuxer": "src.engine.media_stream",
    "get_supported_ffmpeg_encoders": "src.engine.media_stream",
    "is_encoder_functional": "src.engine.media_stream",
    "resolve_hevc_encoder": "src.engine.media_stream",
    # Tensor
    "ColorFormat": "src.engine.tensor",
    "LetterboxInfo": "src.engine.tensor",
    "Normalization": "src.engine.tensor",
    "PaddingInfo": "src.engine.tensor",
    "PaddingMode": "src.engine.tensor",
    "TensorLayout": "src.engine.tensor",
    "allocate_simd_aligned": "src.engine.tensor",
    "image_to_tensor": "src.engine.tensor",
    "letterbox": "src.engine.tensor",
    "pad_to_multiple": "src.engine.tensor",
    "tensor_to_image": "src.engine.tensor",
    "unletterbox": "src.engine.tensor",
    "unpad": "src.engine.tensor",
    # Tiler
    "SeamlessRecombiner": "src.engine.tiler",
    "SpatialTiler": "src.engine.tiler",
    "TileLocation": "src.engine.tiler",
    "calculate_optimal_tile_size": "src.engine.tiler",
    "compute_1d_feather_ramp": "src.engine.tiler",
    "generate_2d_cosine_window": "src.engine.tiler",
    "plan_tile_grid": "src.engine.tiler",
    # Vulkan Backend
    "VulkanConfig": "src.engine.vulkan_backend",
    "VulkanNCNNBackend": "src.engine.vulkan_backend",
    "find_vulkan_binary": "src.engine.vulkan_backend",
    # ONNX Backend
    "DEFAULT_PROVIDER_PRIORITY": "src.engine.onnx_backend",
    "ModelIOSpec": "src.engine.onnx_backend",
    "ONNXConfig": "src.engine.onnx_backend",
    "ONNXInferenceEngine": "src.engine.onnx_backend",
    # Orchestrator & CLI
    "EngineOrchestrator": "src.engine.orchestrator",
    "run_cli": "src.engine.cli",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = [
    "DEFAULT_PROVIDER_PRIORITY",
    "ColorFormat",
    "EngineOrchestrator",
    "LetterboxInfo",
    "ModelIOSpec",
    "MuxerConfig",
    "Normalization",
    "ONNXConfig",
    "ONNXInferenceEngine",
    "PaddingInfo",
    "PaddingMode",
    "SeamlessRecombiner",
    "SpatialTiler",
    "StreamPipelineController",
    "StreamProgress",
    "TensorLayout",
    "TileLocation",
    "VideoFrame",
    "VideoStreamDemuxer",
    "VideoStreamMuxer",
    "VulkanConfig",
    "VulkanNCNNBackend",
    "allocate_simd_aligned",
    "calculate_optimal_tile_size",
    "compute_1d_feather_ramp",
    "find_vulkan_binary",
    "generate_2d_cosine_window",
    "get_supported_ffmpeg_encoders",
    "image_to_tensor",
    "is_encoder_functional",
    "letterbox",
    "pad_to_multiple",
    "plan_tile_grid",
    "resolve_hevc_encoder",
    "run_cli",
    "tensor_to_image",
    "unletterbox",
    "unpad",
]
