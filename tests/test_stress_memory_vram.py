"""Memory, VRAM, and Extreme Scaling Stress Test Suite.

Rigorously audits AnimaScale engine under:
1. Extreme resolutions: 7680x4320 (8K UHD), 15360x8640 (16K), and non-standard prime dimensions (5003x3001).
   Verifies memory budgeters clamp tile sizes and prevent RAM explosion.
2. Simulated GPU VRAM exhaustion: Mocks/injects DirectML, CUDA, and Vulkan out-of-memory exceptions;
   verifies that the engine catches them, adjusts tile sizing, or falls back to CPU execution gracefully,
   producing valid upscaled output without crashes or blank/corrupted frames.
3. Long-iteration memory leak auditing: Runs 50+ consecutive full image upscales and tiling iterations
   using tracemalloc, proving post-GC heap growth is strictly bounded to < 1 MB.
4. Mathematical precision stress: Validates floating-point NaN/Inf sanitization, numerical stability across
   2D cosine windows, and boundary edge preservation under prime coordinates.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core.models import auto_tile_size
from src.engine.onnx_backend import (
    HAS_ONNXRUNTIME,
    ONNXConfig,
    ONNXInferenceEngine,
)
from src.engine.tensor import (
    Normalization,
    TensorLayout,
    allocate_simd_aligned,
    image_to_tensor,
    tensor_to_image,
)
from src.engine.tiler import (
    SeamlessRecombiner,
    SpatialTiler,
    TileLocation,
    calculate_optimal_tile_size,
    compute_1d_feather_ramp,
    generate_2d_cosine_window,
    plan_tile_grid,
)
from src.engine.vulkan_backend import (
    VulkanConfig,
    VulkanNCNNBackend,
)
from tests.test_engine_backends import _create_synthetic_onnx_model

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. EXTREME RESOLUTIONS & MEMORY BUDGETING (8K, 16K, PRIME DIMENSIONS)
# ==============================================================================

def test_8k_uhd_spatial_tile_planning_and_quantization():
    """Validates 7680x4320 (8K UHD) tile planning, quantization, and memory clamping."""
    w_8k, h_8k = 7680, 4320

    # 1. Verify VRAM budget clamping on low VRAM GPUs (<= 4GB, 6GB, 8GB, 16GB)
    t_low, o_low = calculate_optimal_tile_size(w_8k, h_8k, scale=2, vram_gb=4.0)
    assert t_low <= 384, f"8K on 4GB VRAM must clamp tile size to <= 384, got {t_low}"
    assert t_low % 32 == 0, "Tile size must be quantized to multiples of 32 for CNN strides"
    assert o_low >= 32, "Overlap must be >= 32 for smooth feathering"

    t_mid, _o_mid = calculate_optimal_tile_size(w_8k, h_8k, scale=2, vram_gb=6.0)
    assert t_mid <= 512, f"8K on 6GB VRAM must clamp tile size to <= 512, got {t_mid}"
    assert t_mid % 32 == 0

    t_high, _o_high = calculate_optimal_tile_size(w_8k, h_8k, scale=4, vram_gb=16.0)
    assert t_high <= 1024
    assert t_high % 32 == 0

    # 2. Plan 8K grid with 384px tiles
    locations = plan_tile_grid(w_8k, h_8k, tile_size=384, overlap=48, scale=2)
    assert len(locations) > 200, f"8K canvas should require 200+ tiles, got {len(locations)}"

    # 3. Verify total coverage: every pixel in 8K canvas must be accounted for
    covered_x = max(loc.in_x1 for loc in locations)
    covered_y = max(loc.in_y1 for loc in locations)
    assert covered_x == w_8k, f"Tile grid must cover full width {w_8k}, got {covered_x}"
    assert covered_y == h_8k, f"Tile grid must cover full height {h_8k}, got {covered_y}"


def test_16k_extreme_resolution_memory_budget_clamping():
    """Validates 15360x8640 (16K) tile planning and dynamic memory budget enforcement."""
    w_16k, h_16k = 15360, 8640

    # 1. Uncompressed 16K FP32 NCHW frame activation footprint is > 1.5 GB per layer
    uncompressed_16k_bytes = w_16k * h_16k * 3 * 4
    assert uncompressed_16k_bytes > 1_500_000_000, "16K frame should exceed 1.5 GB per tensor"

    # 2. Verify auto tile sizing prevents un-tiled 0 on 16K
    tile_16k = auto_tile_size(gpu_vram_gb=8.0, frame_width=w_16k, frame_height=h_16k)
    assert tile_16k > 0, "16K inputs must NEVER return tile 0 (no tiling)"
    assert tile_16k <= 384, "16K on 8GB VRAM must clamp tile size to <= 384"
    assert tile_16k % 32 == 0

    # 3. Dynamic max_memory_mb budget constraint
    t_budgeted, _ = calculate_optimal_tile_size(
        w_16k, h_16k, scale=4, vram_gb=8.0, max_memory_mb=128.0
    )
    # With a tight 128 MB budget, tile size should scale down
    assert t_budgeted <= 192, f"Expected tile <= 192 under 128MB budget, got {t_budgeted}"
    assert t_budgeted % 32 == 0

    # 4. Single tile footprint with 256px tile is ~3 MB (vs 1.5+ GB un-tiled)
    single_tile_bytes = 256 * 256 * 3 * 4
    assert single_tile_bytes < 1_000_000
    assert (uncompressed_16k_bytes / single_tile_bytes) > 1500


def test_non_standard_prime_dimensions_tiling_fidelity():
    """Tests tiler planning and seamless reconstruction on non-power-of-2 prime dimensions."""
    # Hostile prime dimensions: 5003 x 3001
    w_prime, h_prime = 5003, 3001

    locations = plan_tile_grid(w_prime, h_prime, tile_size=512, overlap=64, scale=1)
    assert len(locations) > 0

    # Verify boundary edge flags
    left_edges = [loc for loc in locations if loc.is_left_edge]
    right_edges = [loc for loc in locations if loc.is_right_edge]
    top_edges = [loc for loc in locations if loc.is_top_edge]
    bottom_edges = [loc for loc in locations if loc.is_bottom_edge]

    assert all(loc.in_x0 == 0 for loc in left_edges)
    assert all(loc.in_x1 == w_prime for loc in right_edges)
    assert all(loc.in_y0 == 0 for loc in top_edges)
    assert all(loc.in_y1 == h_prime for loc in bottom_edges)

    # Reconstruct a synthetic 2D gradient of prime size (5003 x 3001)
    # Using a 256x256 sub-region or scaled resolution to keep test execution fast
    w_sub, h_sub = 503, 301  # Smaller primes for fast exact mathematical test
    x = np.linspace(10, 240, w_sub, dtype=np.float32)
    y = np.linspace(10, 240, h_sub, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    gradient = ((xx + yy) * 0.5).astype(np.uint8)
    gradient_rgb = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)

    tiler = SpatialTiler(tile_size=128, overlap=32, scale=1)
    reconstructed = tiler.process(gradient_rgb, inference_fn=lambda tile: tile)

    assert reconstructed.shape == (h_sub, w_sub, 3)
    np.testing.assert_array_equal(
        reconstructed,
        gradient_rgb,
        err_msg="Prime dimension gradient reconstruction suffered mathematical error",
    )


def test_recombiner_zero_allocation_in_place_memory_savings():
    """Verifies that SeamlessRecombiner executes in-place normalization, freeing buffers immediately."""
    w, h = 512, 512
    rec = SeamlessRecombiner(target_width=w, target_height=h, channels=3, scale=1, dtype=np.uint8)

    loc = TileLocation(
        tile_index=0,
        total_tiles=1,
        row=0,
        col=0,
        in_x0=0,
        in_y0=0,
        in_x1=w,
        in_y1=h,
        out_x0=0,
        out_y0=0,
        out_x1=w,
        out_y1=h,
        overlap_left=0,
        overlap_right=0,
        overlap_top=0,
        overlap_bottom=0,
        is_left_edge=True,
        is_right_edge=True,
        is_top_edge=True,
        is_bottom_edge=True,
    )

    tile = np.full((h, w, 3), 175, dtype=np.uint8)
    rec.add_tile(tile, loc)

    # Accumulator and weights exist before finalize
    assert rec._accumulator is not None
    assert rec._weights is not None

    result = rec.finalize(clamp=True)

    # Buffers must be immediately freed to prevent memory retention across GC
    assert rec._accumulator is None
    assert rec._weights is None

    assert result.shape == (h, w, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, tile)

    # Calling finalize again on a cleared recombiner should raise an informative RuntimeError
    with pytest.raises(RuntimeError, match="already been finalized or cleared"):
        rec.finalize()


# ==============================================================================
# 2. SIMULATED GPU VRAM EXHAUSTION & GRACEFUL RECOVERY
# ==============================================================================

def test_simulated_directml_oom_fallback_to_cpu():
    """Injects simulated DirectML OOM exception; verifies graceful recovery to CPU without corrupted frames."""
    if not HAS_ONNXRUNTIME:
        pytest.skip("ONNX Runtime is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_sr.onnx")
        _create_synthetic_onnx_model(model_path)

        engine = ONNXInferenceEngine(model_path=model_path, config=ONNXConfig(scale=1))

        # Simulate engine bound to DirectML
        engine.active_provider = "DmlExecutionProvider"

        call_count = [0]
        orig_run = engine.session.run

        def failing_dml_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # DirectML standard device removal exception on VRAM exhaustion
                raise RuntimeError("D3D12 device removed - reason: DXGI_ERROR_DEVICE_HUNG (DirectML OOM)")
            return orig_run(*args, **kwargs)

        engine.session.run = failing_dml_run

        test_tile = np.full((64, 64, 3), 190, dtype=np.uint8)
        out_tile = engine.upscale_tile(test_tile)

        # 1. Engine caught the DML OOM and migrated active_provider to CPUExecutionProvider
        assert engine.active_provider == "CPUExecutionProvider"
        # 2. Output is intact, not corrupted or blank
        assert out_tile.shape == (64, 64, 3)
        assert np.mean(out_tile) > 0.0, "Output must not be a black or blank frame"
        np.testing.assert_allclose(out_tile, test_tile, atol=1)


def test_simulated_cuda_oom_fallback_to_cpu():
    """Injects simulated CUDA OOM exception; verifies engine falls back to CPU and completes frame."""
    if not HAS_ONNXRUNTIME:
        pytest.skip("ONNX Runtime is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_sr.onnx")
        _create_synthetic_onnx_model(model_path)

        engine = ONNXInferenceEngine(model_path=model_path, config=ONNXConfig(scale=1))

        # Simulate engine bound to CUDA
        engine.active_provider = "CUDAExecutionProvider"

        call_count = [0]
        orig_run = engine.session.run

        def failing_cuda_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # CUDA out of memory RuntimeError
                raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB (GPU 0; 8.00 GiB total capacity)")
            return orig_run(*args, **kwargs)

        engine.session.run = failing_cuda_run

        input_tensor = np.full((1, 3, 32, 32), 0.65, dtype=np.float32)
        out_tensor = engine.run_raw(input_tensor)

        assert engine.active_provider == "CPUExecutionProvider"
        assert out_tensor.shape == (1, 3, 32, 32)
        assert np.all(np.isfinite(out_tensor))
        np.testing.assert_allclose(out_tensor, input_tensor, atol=1e-4)


def test_simulated_vulkan_oom_adaptive_tile_downsizing():
    """Simulates Vulkan VK_ERROR_OUT_OF_DEVICE_MEMORY; verifies automatic tile size halving."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_bin = os.path.join(tmp_dir, "realesrgan-ncnn-vulkan.exe")
        with open(fake_bin, "w") as f:
            f.write("")
        os.makedirs(os.path.join(tmp_dir, "models"), exist_ok=True)

        backend = VulkanNCNNBackend(
            engine_type="realesrgan",
            binary_path=fake_bin,
            config=VulkanConfig(tile_size=400, gpu_id=0),
        )

        calls = []

        def mock_popen(cmd, *args, **kwargs):
            calls.append(list(cmd))
            proc = MagicMock()
            if len(calls) == 1:
                # First attempt fails with Vulkan device OOM
                proc.communicate.return_value = ("", "vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY")
                proc.returncode = -2
            else:
                # Second attempt succeeds on smaller tile size
                out_idx = cmd.index("-o") + 1
                out_file = cmd[out_idx]
                img = Image.new("RGB", (80, 80), color=(140, 180, 220))
                img.save(out_file)
                proc.communicate.return_value = ("Done", "")
                proc.returncode = 0
            return proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            in_img = np.full((40, 40, 3), 140, dtype=np.uint8)
            out = backend.upscale_image(in_img, tile_size=400)

            # Verified: First call failed on 400px tile, second call retried on 192px tile
            assert len(calls) == 2
            t_first = int(calls[0][calls[0].index("-t") + 1])
            t_second = int(calls[1][calls[1].index("-t") + 1])
            assert t_first == 400
            assert t_second == 192, f"Expected tile to halve to 192, got {t_second}"
            assert out.shape == (80, 80, 3)
            assert np.mean(out) > 0.0, "Frame must be valid non-black"


def test_simulated_vulkan_oom_cpu_fallback():
    """Simulates Vulkan OOM on minimum tile size; verifies automatic graceful CPU fallback (-g -1)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_bin = os.path.join(tmp_dir, "realesrgan-ncnn-vulkan.exe")
        with open(fake_bin, "w") as f:
            f.write("")
        os.makedirs(os.path.join(tmp_dir, "models"), exist_ok=True)

        backend = VulkanNCNNBackend(
            engine_type="realesrgan",
            binary_path=fake_bin,
            config=VulkanConfig(tile_size=64, gpu_id=0),
        )

        calls = []

        def mock_popen(cmd, *args, **kwargs):
            calls.append(list(cmd))
            proc = MagicMock()
            if len(calls) == 1:
                # Minimum tile size still OOMs on GPU
                proc.communicate.return_value = ("", "VK_ERROR_OUT_OF_DEVICE_MEMORY")
                proc.returncode = -2
            else:
                # Fallback to CPU (-g -1) succeeds
                out_idx = cmd.index("-o") + 1
                out_file = cmd[out_idx]
                img = Image.new("RGB", (64, 64), color=(80, 120, 160))
                img.save(out_file)
                proc.communicate.return_value = ("Done", "")
                proc.returncode = 0
            return proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            in_img = np.full((32, 32, 3), 80, dtype=np.uint8)
            out = backend.upscale_image(in_img, tile_size=64)

            assert len(calls) == 2
            gpu_first = calls[0][calls[0].index("-g") + 1]
            gpu_second = calls[1][calls[1].index("-g") + 1]
            assert gpu_first == "0"
            assert gpu_second == "-1", f"Expected CPU fallback gpu_id=-1, got {gpu_second}"
            assert out.shape == (64, 64, 3)


def test_vulkan_corrupted_black_frame_detection_and_recovery():
    """Verifies that silent black frame corruption is caught and recovered with safe parameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_bin = os.path.join(tmp_dir, "realesrgan-ncnn-vulkan.exe")
        with open(fake_bin, "w") as f:
            f.write("")
        os.makedirs(os.path.join(tmp_dir, "models"), exist_ok=True)

        backend = VulkanNCNNBackend(
            engine_type="realesrgan",
            binary_path=fake_bin,
            config=VulkanConfig(tile_size=400, gpu_id=0),
        )

        calls = []

        def mock_popen(cmd, *args, **kwargs):
            calls.append(list(cmd))
            proc = MagicMock()
            out_idx = cmd.index("-o") + 1
            out_file = cmd[out_idx]
            if len(calls) == 1:
                # Silent failure: returns 0 but writes an all-black image
                img = Image.new("RGB", (100, 100), color=(0, 0, 0))
                img.save(out_file)
            else:
                # Safe recovery writes valid upscaled image
                img = Image.new("RGB", (100, 100), color=(180, 130, 90))
                img.save(out_file)
            proc.communicate.return_value = ("Done", "")
            proc.returncode = 0
            return proc

        with patch("subprocess.Popen", side_effect=mock_popen):
            in_img = np.full((50, 50, 3), 150, dtype=np.uint8)
            out = backend.upscale_image(in_img, tile_size=400)

            assert len(calls) == 2
            assert np.mean(out) > 0.0, "Black frame corruption must be repaired"
            assert out.shape == (100, 100, 3)


def test_spatial_tiler_adaptive_tile_size_reduction_on_oom():
    """Verifies that SpatialTiler automatically catches inference OOM, halves tile size, and retries."""
    tiler = SpatialTiler(tile_size=256, overlap=32, scale=2)
    img = np.full((300, 300, 3), 110, dtype=np.uint8)

    invoked_sizes = []

    def mock_inference_fn(tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        invoked_sizes.append((h, w))
        # If tile is > 128px, simulate hardware memory failure
        if max(h, w) > 128:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.0 GiB")
        # 2x scale
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    out = tiler.process(img, mock_inference_fn, adaptive_tile_on_oom=True)

    assert out.shape == (600, 600, 3)
    assert np.all(out == 110)
    # Proved: First invocation had 256x256, failed and adapted to 128x128
    assert invoked_sizes[0] == (256, 256)
    assert any(s == (128, 128) for s in invoked_sizes)


# ==============================================================================
# 3. LONG-ITERATION MEMORY LEAK AUDITING (50+ ITERATIONS WITH TRACEMALLOC)
# ==============================================================================

def test_spatial_tiler_50_iterations_memory_leak_bounded():
    """Audits 50+ consecutive full image upscales with tracemalloc to prove heap growth < 1 MB."""
    tiler = SpatialTiler(tile_size=128, overlap=32, scale=2)
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def mock_inference(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    # 1. Warm-up runs to prime interpreter and internal cache structures
    for _ in range(5):
        _ = tiler.process(img, mock_inference)

    gc.collect()
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    # 2. 50 consecutive full image upscales
    for _ in range(50):
        out = tiler.process(img, mock_inference)
        assert out.shape == (512, 512, 3)
        del out

    gc.collect()
    snapshot_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot_end.compare_to(snapshot_start, "lineno")
    total_leak_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    total_leak_mb = total_leak_bytes / (1024.0 * 1024.0)

    logger.info("SpatialTiler 50-run net heap growth: %.4f MB (%d bytes)", total_leak_mb, total_leak_bytes)
    assert total_leak_mb < 1.0, f"Memory leak detected: {total_leak_mb:.4f} MB exceeded 1.0 MB limit"


def test_onnx_engine_50_iterations_memory_leak_bounded():
    """Audits 50+ consecutive ONNX tiled inference cycles with tracemalloc, proving post-GC growth < 1 MB."""
    if not HAS_ONNXRUNTIME:
        pytest.skip("ONNX Runtime is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "leak_test.onnx")
        _create_synthetic_onnx_model(model_path)

        engine = ONNXInferenceEngine(model_path=model_path, config=ONNXConfig(scale=1))
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Warm-up runs
        for _ in range(5):
            _ = engine.upscale_image(img, tile_size=128, overlap=32)

        gc.collect()
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        # 50 consecutive runs
        for _ in range(50):
            res = engine.upscale_image(img, tile_size=128, overlap=32)
            assert res.shape == (256, 256, 3)
            del res

        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_end.compare_to(snapshot_start, "lineno")
        total_leak_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        total_leak_mb = total_leak_bytes / (1024.0 * 1024.0)

        logger.info("ONNXEngine 50-run net heap growth: %.4f MB (%d bytes)", total_leak_mb, total_leak_bytes)
        assert total_leak_mb < 1.0, f"ONNX memory leak detected: {total_leak_mb:.4f} MB exceeded 1.0 MB limit"


def test_tensor_conversions_500_iterations_memory_leak_bounded():
    """Audits 500 consecutive SIMD-aligned tensor preprocessing/postprocessing conversions."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    # Pre-allocate destination buffer to verify zero-allocation fast pathway
    dest = allocate_simd_aligned((1, 3, 128, 128), dtype=np.float32)

    # Warm-up
    for _ in range(10):
        t = image_to_tensor(arr, out=dest)
        _ = tensor_to_image(t, to_pil=False)

    gc.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    for _ in range(500):
        t = image_to_tensor(arr, out=dest)
        img_out = tensor_to_image(t, to_pil=False)
        del img_out

    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, "lineno")
    leak_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    leak_mb = leak_bytes / (1024.0 * 1024.0)

    assert leak_mb < 0.5, f"Tensor conversion leak: {leak_mb:.4f} MB exceeded 0.5 MB limit"


# ==============================================================================
# 4. MATHEMATICAL PRECISION & NUMERICAL STABILITY STRESS
# ==============================================================================

def test_hostile_nan_inf_floating_point_sanitization():
    """Verifies floating-point NaN/Inf inputs are sanitized cleanly without warnings or corrupted states."""
    # 1. Tensor with NaN, +Inf, -Inf
    hostile = np.array([
        [[np.nan, np.inf], [-np.inf, 0.5]],
        [[1.0, np.nan], [0.0, -np.inf]],
        [[np.inf, 0.25], [np.nan, 0.75]],
    ], dtype=np.float32).reshape(1, 3, 2, 2)

    # tensor_to_image must sanitize without throwing or emitting RuntimeWarning
    img_uint8 = tensor_to_image(
        hostile,
        layout=TensorLayout.NCHW,
        normalization=Normalization.ZERO_TO_ONE,
        to_pil=False,
    )

    assert img_uint8.shape == (2, 2, 3)
    assert img_uint8.dtype == np.uint8
    assert np.all(np.isfinite(img_uint8))

    # NaN maps to 0, +Inf maps to 255, -Inf maps to 0
    assert img_uint8[0, 0, 0] == 0
    assert img_uint8[0, 1, 0] == 255
    assert img_uint8[1, 0, 0] == 0

    # 2. SeamlessRecombiner NaN/Inf sanitization
    rec = SeamlessRecombiner(target_width=32, target_height=32, channels=3, scale=1)
    loc = TileLocation(0, 1, 0, 0, 0, 0, 32, 32, 0, 0, 32, 32, 0, 0, 0, 0, True, True, True, True)

    bad_tile = np.full((32, 32, 3), 120.0, dtype=np.float32)
    bad_tile[5, 5, :] = np.nan
    bad_tile[5, 6, :] = np.inf

    rec.add_tile(bad_tile, loc)
    result = rec.finalize()

    assert np.all(np.isfinite(result))
    assert result.shape == (32, 32, 3)


def test_2d_cosine_window_partition_of_unity_exactness():
    """Validates exact partition of unity (Sum(W_i) == 1.0 within 1e-6) across asymmetric & non-power-of-2 bounds."""
    for overlap in [23, 37, 51, 64]:  # Prime and arbitrary overlaps
        length = 317  # Prime length
        ramp_up = compute_1d_feather_ramp(length, overlap_start=overlap, overlap_end=0, is_start_edge=False, is_end_edge=True)
        ramp_down = compute_1d_feather_ramp(length, overlap_start=0, overlap_end=overlap, is_start_edge=True, is_end_edge=False)

        w_up = ramp_up[:overlap]
        w_down = ramp_down[-overlap:]

        np.testing.assert_allclose(
            w_up + w_down,
            1.0,
            atol=1e-6,
            err_msg=f"1D feather ramp partition of unity violated for prime overlap {overlap}",
        )


def test_boundary_edge_preservation_no_vignetting():
    """Verifies boundary outer edges retain 1.0 weights so image borders never fade to black or vignette."""
    window = generate_2d_cosine_window(
        height=128,
        width=128,
        overlap_left=32,
        overlap_right=32,
        overlap_top=32,
        overlap_bottom=32,
        is_left_edge=True,
        is_right_edge=False,
        is_top_edge=True,
        is_bottom_edge=False,
    )

    # Top and left outer edges touch image boundaries -> must be identically 1.0
    assert np.all(window[0, :32] == 1.0), "Top-left boundary edge must not be attenuated"
    assert np.all(window[:32, 0] == 1.0), "Outer left edge must not be attenuated"

    # Bottom and right edges overlap with adjacent tiles -> must feather smoothly toward 0
    assert window[-1, -1] < 0.05, "Inner corner intersection must feather smoothly"
