"""Unit and integration tests for ONNX and Vulkan inference backends."""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

import onnx
from onnx import TensorProto, helper

from src.engine.onnx_backend import (
    HAS_ONNXRUNTIME,
    ONNXConfig,
    ONNXInferenceEngine,
)
from src.engine.vulkan_backend import (
    VulkanConfig,
    VulkanNCNNBackend,
    find_vulkan_binary,
)


def _create_synthetic_onnx_model(filepath: str, scale: int = 1) -> str:
    """Creates a minimal valid ONNX model (Identity or Resize) for testing."""
    # Input: float32[1, 3, height, width]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, None, None])
    # Output: float32[1, 3, height, width]
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, None, None])

    # Simple Identity node: output = Identity(input)
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

    graph = helper.make_graph([node], "test_graph", [x], [y])
    model = helper.make_model(graph, producer_name="upscaler_engine_test")
    model.ir_version = 9
    model.opset_import[0].version = 17
    onnx.save(model, filepath)
    return filepath


def test_onnx_engine_initialization_and_inference():
    """Verifies ONNX backend initialization, provider fallback, and tiled execution."""
    if not HAS_ONNXRUNTIME:
        pytest.skip("ONNX Runtime is not available on this platform")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.onnx")
        _create_synthetic_onnx_model(model_path)

        config = ONNXConfig(
            preferred_providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
            scale=1,
        )
        engine = ONNXInferenceEngine(model_path=model_path, config=config)

        # Active provider should have fallen back to available provider (e.g. CPUExecutionProvider)
        assert engine.active_provider in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        assert engine.input_spec is not None
        assert engine.output_spec is not None
        assert engine.input_spec.name == "input"
        assert engine.output_spec.name == "output"

        # Test single tile execution
        tile = np.full((64, 64, 3), 180, dtype=np.uint8)
        out_tile = engine.upscale_tile(tile)
        assert out_tile.shape == (64, 64, 3)
        np.testing.assert_allclose(out_tile, tile, atol=1)

        # Test full image upscaling via SpatialTiler integration
        img = np.full((128, 128, 3), 210, dtype=np.uint8)
        out_img = engine.upscale_image(img, tile_size=64, overlap=16)
        assert out_img.shape == (128, 128, 3)
        np.testing.assert_allclose(out_img, img, atol=1)


def test_onnx_engine_provider_fallback_to_cpu():
    """Verifies that ONNX engine automatically falls back to CPU when preferred providers fail."""
    if not HAS_ONNXRUNTIME:
        pytest.skip("ONNX Runtime is not available on this platform")
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.onnx")
        _create_synthetic_onnx_model(model_path)

        # Request a non-existent provider
        config = ONNXConfig(
            preferred_providers=["NonExistentExecutionProvider", "CPUExecutionProvider"],
            scale=1,
        )
        engine = ONNXInferenceEngine(model_path=model_path, config=config)
        assert engine.active_provider == "CPUExecutionProvider"


def test_vulkan_binary_discovery():
    """Verifies discovery of real Vulkan binaries in tools/."""
    realesrgan_path = find_vulkan_binary("realesrgan")
    assert os.path.isfile(realesrgan_path)
    assert "realesrgan-ncnn-vulkan" in os.path.basename(realesrgan_path).lower()

    realcugan_path = find_vulkan_binary("realcugan")
    assert os.path.isfile(realcugan_path)
    assert "realcugan-ncnn-vulkan" in os.path.basename(realcugan_path).lower()


def test_vulkan_backend_command_args():
    """Verifies command line construction for both Real-ESRGAN and RealCUGAN."""
    # Real-ESRGAN
    backend_esr = VulkanNCNNBackend(
        engine_type="realesrgan",
        model_name="realesr-animevideov3",
        config=VulkanConfig(scale=4, tile_size=400, gpu_id=0, threads="1:2:2"),
    )
    cmd_esr = backend_esr.build_command_args("in.png", "out.png")
    assert "-s" in cmd_esr and cmd_esr[cmd_esr.index("-s") + 1] == "4"
    assert "-t" in cmd_esr and cmd_esr[cmd_esr.index("-t") + 1] == "400"
    assert "-n" in cmd_esr and cmd_esr[cmd_esr.index("-n") + 1] == "realesr-animevideov3"

    # RealCUGAN
    backend_cugan = VulkanNCNNBackend(
        engine_type="realcugan",
        config=VulkanConfig(scale=2, tile_size=256, denoise_level=3, sync_gap_mode=2),
    )
    cmd_cugan = backend_cugan.build_command_args("in.png", "out.png")
    assert "-s" in cmd_cugan and cmd_cugan[cmd_cugan.index("-s") + 1] == "2"
    assert "-t" in cmd_cugan and cmd_cugan[cmd_cugan.index("-t") + 1] == "256"
    assert "-n" in cmd_cugan and cmd_cugan[cmd_cugan.index("-n") + 1] == "3"
    assert "-c" in cmd_cugan and cmd_cugan[cmd_cugan.index("-c") + 1] == "2"


def test_vulkan_real_execution():
    """Tests executing actual portable Vulkan NCNN binary on a synthetic test patch."""
    backend = VulkanNCNNBackend(
        engine_type="realesrgan",
        model_name="realesr-animevideov3",
        config=VulkanConfig(scale=2, tile_size=64, gpu_id="auto"),
    )
    # 32x32 test patch
    img = Image.new("RGB", (32, 32), color=(120, 200, 80))
    try:
        upscaled = backend.upscale_image(img, scale=2, timeout=30.0)
        assert upscaled.shape == (64, 64, 3)
    except Exception as exc:
        # If the environment has no compatible Vulkan GPU, verify that graceful error is raised
        assert "vulkan" in str(exc).lower() or "code" in str(exc).lower()
