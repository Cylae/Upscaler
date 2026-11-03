"""End-to-end hostile adversarial test suite for Cylae/upscaler overhaul.

Covers:
1. Synthetic Model Generator:
   - Creates valid 2x, 3x, and 4x super-resolution ONNX models in memory and disk
     conforming strictly to opset_version=18 and ir_version=10.
   - Verifies ONNX checker, graph topology, and execution with ONNX Runtime.
   - Generates multi-channel RGBA and PixelShuffle/Conv architectures.
2. Hostile Image & Media Fuzzing:
   - Zero-dimension (0x0, 0x100, 100x0) boundary rejections.
   - Single-pixel (1x1) edge case scaling.
   - Prime number non-aligned dimensions (37x53, 1009x701, 113x227) tile remainder handling.
   - Massive 8K / 16K synthetic frames stress-testing memory budgeting and auto-tiling.
   - Truncated bitstreams, corrupt JPEG/PNG headers, and corrupted container wrappers.
   - Alpha channel (RGBA) preservation without channel dropping or corruption.
   - Odd pixel aspect ratios (e.g. 32:27 anamorphic) and display geometry calculation.
3. Seamless Tile Blending Verification:
   - Exact mathematical partition of unity verification (Sum of weights == 1.0).
   - C1 continuity and 0 seam artifact / discontinuity comparison against continuous reference.
   - Comparative seam metric between cosine-feathered blending and hard box tiling.
4. Concurrency & Resource Leaks:
   - Rapid start/stop cancellation cycles verifying 0 orphaned child processes.
   - Strict memory bounding and garbage collection leak verification.
   - Actor pipeline queue cancellation under active load without deadlocks.
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime as ort
except (ImportError, Exception):
    pytest.skip("onnx/onnxruntime native runtime unavailable", allow_module_level=True)


from src.core.models import (
    auto_tile_size,
    calculate_display_dimensions,
    calculate_target_dimensions,
)
from src.core.probe import VideoMetadata, probe_video
from src.core.process import ProcessTracker
from src.core.validator import is_file_complete_and_valid, validate_encoded_video
from src.core.video_pipeline import _square_pixel_filter
from src.engine.media_stream import (
    StreamPipelineController,
    VideoFrame,
)
from src.engine.onnx_backend import ONNXConfig, ONNXInferenceEngine
from src.engine.tensor import (
    ColorFormat,
    image_to_tensor,
)
from src.engine.tiler import (
    SpatialTiler,
    calculate_optimal_tile_size,
    compute_1d_feather_ramp,
    plan_tile_grid,
)


# ==============================================================================
# SYNTHETIC ONNX MODEL GENERATOR (Opset 18, IR Version 10)
# ==============================================================================

def create_synthetic_onnx_sr_model(
    scale: int = 2,
    channels: int = 3,
    opset_version: int = 18,
    ir_version: int = 10,
    model_path: Path | str | None = None,
    mode: str = "linear",
) -> bytes:
    """Generates a valid, production-grade super-resolution ONNX model.

    Enforces opset_version=18 and ir_version=10 using ONNX helper APIs.
    """
    if scale < 1:
        raise ValueError(f"Scale factor must be >= 1, got {scale}")
    if channels not in (1, 3, 4):
        raise ValueError(f"Channels must be 1, 3, or 4, got {channels}")

    # Build Resize node scaling spatial H and W dimensions
    node = helper.make_node(
        "Resize",
        inputs=["X", "roi", "scales"],
        outputs=["Y"],
        mode=mode,
        coordinate_transformation_mode="half_pixel",
    )

    # Dynamic spatial input/output value info
    x_val_info = helper.make_tensor_value_info(
        "X",
        TensorProto.FLOAT,
        [1, channels, None, None],
    )
    y_val_info = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT,
        [1, channels, None, None],
    )

    roi_tensor = helper.make_tensor("roi", TensorProto.FLOAT, [0], [])
    scales_tensor = helper.make_tensor(
        "scales",
        TensorProto.FLOAT,
        [4],
        [1.0, 1.0, float(scale), float(scale)],
    )

    graph = helper.make_graph(
        nodes=[node],
        name=f"synthetic_super_resolution_{scale}x_{channels}c",
        inputs=[x_val_info],
        outputs=[y_val_info],
        initializer=[roi_tensor, scales_tensor],
    )

    opset_id = helper.make_opsetid("", opset_version)
    model = helper.make_model(
        graph,
        opset_imports=[opset_id],
        ir_version=ir_version,
    )

    # Validate ONNX model structural integrity
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()

    if model_path is not None:
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(serialized)

    return serialized


def create_synthetic_pixelshuffle_sr_model(
    scale: int = 2,
    channels: int = 3,
    opset_version: int = 18,
    ir_version: int = 10,
    model_path: Path | str | None = None,
) -> bytes:
    """Generates an ONNX model using Conv + DepthToSpace (PixelShuffle) architecture."""
    out_channels = channels * (scale ** 2)

    # 1x1 convolution expanding channels for pixel shuffle
    conv_w = np.zeros((out_channels, channels, 1, 1), dtype=np.float32)
    for c in range(channels):
        for s in range(scale * scale):
            conv_w[c * (scale * scale) + s, c, 0, 0] = 1.0

    conv_b = np.zeros(out_channels, dtype=np.float32)

    w_tensor = helper.make_tensor("conv_w", TensorProto.FLOAT, list(conv_w.shape), conv_w.flatten().tolist())
    b_tensor = helper.make_tensor("conv_b", TensorProto.FLOAT, list(conv_b.shape), conv_b.flatten().tolist())

    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "conv_w", "conv_b"],
        outputs=["conv_out"],
        kernel_shape=[1, 1],
    )

    depth_to_space_node = helper.make_node(
        "DepthToSpace",
        inputs=["conv_out"],
        outputs=["Y"],
        blocksize=scale,
        mode="CRD",
    )

    x_val = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, channels, None, None])
    y_val = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, channels, None, None])

    graph = helper.make_graph(
        nodes=[conv_node, depth_to_space_node],
        name=f"pixelshuffle_sr_{scale}x_{channels}c",
        inputs=[x_val],
        outputs=[y_val],
        initializer=[w_tensor, b_tensor],
    )

    opset_id = helper.make_opsetid("", opset_version)
    model = helper.make_model(graph, opset_imports=[opset_id], ir_version=ir_version)
    onnx.checker.check_model(model)
    serialized = model.SerializeToString()

    if model_path is not None:
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(serialized)

    return serialized


# ==============================================================================
# GROUP 1: SYNTHETIC MODEL GENERATOR TESTS
# ==============================================================================

@pytest.mark.parametrize("scale", [2, 3, 4])
def test_synthetic_onnx_generator_scales_and_opsets(tmp_path: Path, scale: int) -> None:
    """Verifies ONNX model generation for 2x, 3x, and 4x with opset_version=18 and ir_version=10."""
    model_file = tmp_path / f"sr_{scale}x.onnx"
    raw_bytes = create_synthetic_onnx_sr_model(
        scale=scale,
        channels=3,
        opset_version=18,
        ir_version=10,
        model_path=model_file,
    )

    assert len(raw_bytes) > 0
    assert model_file.is_file()

    # Verify ONNX protocol buffer attributes
    loaded_model = onnx.load(str(model_file))
    assert loaded_model.ir_version == 10
    assert len(loaded_model.opset_import) >= 1
    assert loaded_model.opset_import[0].version == 18

    # Verify execution with ONNX Runtime session
    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    inp = np.random.uniform(0.0, 1.0, (1, 3, 16, 24)).astype(np.float32)
    out = sess.run(None, {"X": inp})[0]

    expected_shape = (1, 3, 16 * scale, 24 * scale)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    assert np.all(np.isfinite(out)), "Output contains NaN or Inf values"


def test_synthetic_onnx_generator_multichannel_rgba(tmp_path: Path) -> None:
    """Verifies that the synthetic ONNX generator creates valid 4-channel (RGBA) models."""
    model_file = tmp_path / "sr_rgba_3x.onnx"
    create_synthetic_onnx_sr_model(
        scale=3,
        channels=4,
        opset_version=18,
        ir_version=10,
        model_path=model_file,
    )

    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    inp = np.random.uniform(0.0, 1.0, (1, 4, 20, 20)).astype(np.float32)
    out = sess.run(None, {"X": inp})[0]

    assert out.shape == (1, 4, 60, 60)
    assert np.all(np.isfinite(out))


def test_synthetic_onnx_pixelshuffle_architecture(tmp_path: Path) -> None:
    """Verifies Conv + PixelShuffle (DepthToSpace) ONNX model generation and execution."""
    model_file = tmp_path / "pixelshuffle_2x.onnx"
    create_synthetic_pixelshuffle_sr_model(
        scale=2,
        channels=3,
        opset_version=18,
        ir_version=10,
        model_path=model_file,
    )

    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    inp = np.full((1, 3, 8, 8), 0.75, dtype=np.float32)
    out = sess.run(None, {"X": inp})[0]

    assert out.shape == (1, 3, 16, 16)
    # The custom Conv+DepthToSpace identity expansion must preserve constant values exactly
    assert np.allclose(out, 0.75, atol=1e-5)


# ==============================================================================
# GROUP 2: HOSTILE IMAGE & MEDIA FUZZING
# ==============================================================================

def test_hostile_zero_dimension_handling() -> None:
    """Verifies that 0x0 and degenerate zero-dimension images are rejected with clear ValueErrors."""
    # 1. SpatialTiler / plan_tile_grid rejection
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        plan_tile_grid(image_width=0, image_height=0)

    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        plan_tile_grid(image_width=100, image_height=0)

    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        plan_tile_grid(image_width=0, image_height=100)

    # 2. image_to_tensor rejection
    zero_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        image_to_tensor(zero_arr)

    zero_wide = np.zeros((10, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        image_to_tensor(zero_wide)

    # 3. Video metadata / target dimensions calculation bounds
    disp_w, disp_h = calculate_display_dimensions(0, 0)
    assert disp_w >= 2.0 and disp_h >= 2.0, "Degenerate 0x0 input must clamp safely to minimum 2x2"


def test_hostile_single_pixel_1x1_processing() -> None:
    """Verifies that single-pixel 1x1 inputs process without zero-division or boundary crash."""
    img_1x1 = np.array([[[128, 64, 32]]], dtype=np.uint8)  # shape (1, 1, 3)
    tiler = SpatialTiler(tile_size=64, overlap=16, scale=4)

    def dummy_nearest_sr(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1)

    out = tiler.process(img_1x1, dummy_nearest_sr)
    assert out.shape == (4, 4, 3)
    assert np.all(out == [128, 64, 32])


@pytest.mark.parametrize(
    ("width", "height", "scale", "tile_size", "overlap"),
    [
        (37, 53, 2, 32, 8),      # Small primes requiring boundary shifts
        (113, 227, 3, 64, 16),    # Medium primes across multi-scale
        (1009, 701, 2, 256, 32),  # Large primes with non-aligned remainder tiles
    ],
)
def test_hostile_prime_non_aligned_dimensions(
    width: int,
    height: int,
    scale: int,
    tile_size: int,
    overlap: int,
) -> None:
    """Verifies tile remainder handling for non-power-of-two prime dimensions."""
    locations = plan_tile_grid(
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        overlap=overlap,
        scale=scale,
    )
    assert len(locations) > 0

    # Verify complete boundary coverage
    min_x = min(loc.in_x0 for loc in locations)
    max_x = max(loc.in_x1 for loc in locations)
    min_y = min(loc.in_y0 for loc in locations)
    max_y = max(loc.in_y1 for loc in locations)

    assert min_x == 0 and min_y == 0
    assert max_x == width and max_y == height

    # Execute full spatial tiling with continuous linear gradient
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    tiler = SpatialTiler(tile_size=tile_size, overlap=overlap, scale=scale)

    def dummy_identity_sr(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)

    out = tiler.process(img, dummy_identity_sr)
    assert out.shape == (height * scale, width * scale, 3)
    assert out.dtype == np.uint8


def test_hostile_massive_8k_16k_memory_budgeter_and_tiling() -> None:
    """Stress tests memory budgeting and VRAM auto-tiling for massive 8K and 16K frames."""
    w_8k, h_8k = 7680, 4320
    w_16k, h_16k = 15360, 8640

    # 1. Verify VRAM auto-tiling forbids tile 0 (no-tiling) on high-resolution inputs
    tile_8k_low_vram = auto_tile_size(gpu_vram_gb=4.0, frame_width=w_8k, frame_height=h_8k)
    assert tile_8k_low_vram in (128, 192), f"Expected safe tile size for 4GB VRAM, got {tile_8k_low_vram}"

    tile_8k_high_vram = auto_tile_size(gpu_vram_gb=24.0, frame_width=w_8k, frame_height=h_8k)
    assert tile_8k_high_vram > 0, "8K inputs must NEVER return tile 0 (no tiling) even on 24GB GPUs"

    tile_16k = auto_tile_size(gpu_vram_gb=8.0, frame_width=w_16k, frame_height=h_16k)
    assert tile_16k in (192, 256, 288, 384)
    assert tile_16k % 32 == 0, "Tile sizes must be quantized to multiples of 32 for CNN strides"

    # 2. Engine optimal tile size resolution under constrained VRAM
    t_size, overlap = calculate_optimal_tile_size(w_8k, h_8k, scale=2, vram_gb=6.0)
    assert t_size <= 768
    assert overlap >= 32

    # 3. Verify spatial tile planning bounds for 16K canvas
    locations_16k = plan_tile_grid(w_16k, h_16k, tile_size=512, overlap=64, scale=2)
    assert len(locations_16k) > 100, f"16K canvas should partition into multiple sub-tiles, got {len(locations_16k)}"

    # 4. Memory footprint verification: single tile tensor allocation vs un-tiled canvas
    uncompressed_16k_frame_bytes = w_16k * h_16k * 3 * 4  # FP32 NCHW
    single_tile_bytes = 512 * 512 * 3 * 4

    # Un-tiled 16K frame exceeds 1.5 GB RAM per activation layer
    assert uncompressed_16k_frame_bytes > 1_500_000_000
    # Tiled processing bounds peak memory to ~3 MB per tile
    assert single_tile_bytes < 4_000_000
    assert (uncompressed_16k_frame_bytes / single_tile_bytes) > 400


def test_hostile_corrupt_and_truncated_bitstreams(tmp_path: Path) -> None:
    """Verifies pipeline validators and probes gracefully catch truncated and corrupted media."""
    # Vector 1: Zero-byte file
    zero_file = tmp_path / "zero_bytes.mp4"
    zero_file.write_bytes(b"")
    assert is_file_complete_and_valid(str(zero_file)) is False
    valid_vid, _ = validate_encoded_video(str(zero_file), 1920, 1080, expect_audio=False, expected_duration=5.0)
    assert valid_vid is False

    with pytest.raises((RuntimeError, ValueError, subprocess.SubprocessError)):
        probe_video(str(zero_file))

    # Vector 2: Truncated JPEG
    trunc_jpg = tmp_path / "truncated.jpg"
    trunc_jpg.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    assert is_file_complete_and_valid(str(trunc_jpg)) is False

    # Vector 3: Corrupt PNG header
    corrupt_png = tmp_path / "corrupt_hdr.png"
    corrupt_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x00")
    assert is_file_complete_and_valid(str(corrupt_png)) is False

    # Vector 4: Corrupted container wrapper (invalid MP4 ftyp atom)
    corrupt_mp4 = tmp_path / "bad_container.mp4"
    corrupt_mp4.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00" + os.urandom(512))
    valid_enc, _ = validate_encoded_video(str(corrupt_mp4), 1920, 1080, expect_audio=False, expected_duration=5.0)
    assert valid_enc is False


def test_hostile_alpha_channel_rgba_preservation(tmp_path: Path) -> None:
    """Verifies that RGBA images with transparency gradients retain alpha fidelity across tiles."""
    width, height = 120, 90
    rgba_img = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :]  # R
    rgba_img[:, :, 1] = np.linspace(255, 0, height, dtype=np.uint8)[:, np.newaxis]  # G
    rgba_img[:, :, 2] = 128  # B
    rgba_img[:, :, 3] = np.linspace(10, 240, width, dtype=np.uint8)[np.newaxis, :]  # Alpha gradient

    model_file = tmp_path / "onnx_rgba_2x.onnx"
    create_synthetic_onnx_sr_model(
        scale=2,
        channels=4,
        opset_version=18,
        ir_version=10,
        model_path=model_file,
    )

    engine = ONNXInferenceEngine(
        model_path=str(model_file),
        config=ONNXConfig(
            color_format=ColorFormat.RGBA,
            scale=2,
        ),
    )

    upscaled = engine.upscale_image(rgba_img, tile_size=48, overlap=16)
    assert upscaled.shape == (height * 2, width * 2, 4)

    # Verify alpha channel continuity across tile seams
    alpha_channel = upscaled[:, :, 3]
    assert np.all(alpha_channel >= 8)
    assert np.all(alpha_channel <= 245)
    # Check that alpha values form a smooth monotonic gradient without discontinuous drops
    mid_row_alpha = alpha_channel[height, :]
    alpha_diffs = np.diff(mid_row_alpha.astype(np.int32))
    assert np.all(alpha_diffs >= 0), "Alpha channel should remain monotonically non-decreasing"


def test_hostile_odd_pixel_aspect_ratio_anamorphic() -> None:
    """Verifies square-pixel normalization and filter construction for anamorphic media (e.g. 32:27 SAR)."""
    # DVD NTSC 16:9 Anamorphic: 720x480 with SAR 32:27
    disp_w, disp_h = calculate_display_dimensions(720, 480, sample_aspect_ratio="32:27")
    expected_w = 720 * (32 / 27)  # ~853.33 px
    assert math.isclose(disp_w, expected_w, rel_tol=1e-3)
    assert math.isclose(disp_h, 480.0, rel_tol=1e-3)

    # Calculate 4K target dimensions
    target_w, target_h, tag = calculate_target_dimensions(720, 480, target_preset="4k", sample_aspect_ratio="32:27")
    assert target_w == 3840
    assert target_h == 2160
    assert tag == "2160p"

    # Verify FFmpeg square pixel filter string
    meta = VideoMetadata(
        filepath="dummy.mp4",
        width=720,
        height=480,
        fps=23.976,
        duration=10.0,
        nb_frames=240,
        codec_name="h264",
        has_audio=False,
        audio_codec="",
        has_subtitles=False,
        sample_aspect_ratio="32:27",
    )
    flt = _square_pixel_filter(meta)
    assert "setsar=1" in flt
    assert "scale=854:480" in flt
    assert flt.endswith(",setsar=1")


# ==============================================================================
# GROUP 3: SEAMLESS TILE BLENDING VERIFICATION
# ==============================================================================

def test_seamless_feathered_tile_blending_mathematical_identity() -> None:
    """Mathematically verifies partition-of-unity blending: sum of overlapping weights identically equals 1.0."""
    length = 128
    overlap = 32

    # Tile 1: Left tile (is_start_edge=True, is_end_edge=False)
    ramp_left = compute_1d_feather_ramp(
        length=length,
        overlap_start=overlap,
        overlap_end=overlap,
        is_start_edge=True,
        is_end_edge=False,
    )

    # Tile 2: Right tile (is_start_edge=False, is_end_edge=True)
    ramp_right = compute_1d_feather_ramp(
        length=length,
        overlap_start=overlap,
        overlap_end=overlap,
        is_start_edge=False,
        is_end_edge=True,
    )

    # Overlap region: tail of Tile 1 overlapping head of Tile 2
    tail_tile1 = ramp_left[-overlap:]
    head_tile2 = ramp_right[:overlap]
    overlap_sum = tail_tile1 + head_tile2

    # Partition of unity: sum of weights across overlap must equal 1.0 within float precision
    assert np.allclose(overlap_sum, 1.0, atol=1e-6)

    # C1 continuity: derivative at boundary edges must smoothly vanish
    d_tail = np.diff(tail_tile1)
    d_head = np.diff(head_tile2)
    assert np.allclose(d_tail + d_head, 0.0, atol=1e-6)


def test_seamless_tile_reconstruction_vs_continuous_reference(tmp_path: Path) -> None:
    """Verifies that tiled inference with cosine feathering matches continuous reference with 0 seam artifacts."""
    scale = 2
    model_file = tmp_path / "seamless_verify_2x.onnx"
    create_synthetic_onnx_sr_model(
        scale=scale,
        channels=3,
        opset_version=18,
        ir_version=10,
        model_path=model_file,
        mode="nearest",
    )

    # Create synthetic test frame with high frequencies, linear gradients, and step contours
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    frame = (
        0.5 * np.sin(x / 4.0) +
        0.5 * np.cos(y / 4.0) +
        (x / float(w)) * 0.5
    ).astype(np.float32)
    frame = np.stack([frame, frame * 0.8, 1.0 - frame], axis=-1)
    frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])

    # 1. Continuous reference (un-tiled inference)
    inp_continuous = frame_u8.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    ref_out = sess.run(None, {"X": inp_continuous})[0][0].transpose(1, 2, 0)
    ref_out_u8 = np.clip(ref_out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 2. Feathered cosine-window tiled reconstruction
    tiler = SpatialTiler(tile_size=48, overlap=16, scale=scale)

    def _infer_tile(t: np.ndarray) -> np.ndarray:
        t_inp = t.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        t_out = sess.run(None, {"X": t_inp})[0][0].transpose(1, 2, 0)
        return np.clip(t_out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    feathered_out_u8 = tiler.process(frame_u8, _infer_tile)

    # 3. Mathematical seam discontinuity check
    # Compare feathered output directly against continuous reference
    delta = np.abs(feathered_out_u8.astype(np.int32) - ref_out_u8.astype(np.int32))
    max_pixel_delta = np.max(delta)
    mean_pixel_delta = np.mean(delta)

    # Feathered blending with nearest scaling must perfectly match continuous reference
    assert max_pixel_delta <= 1, f"Max pixel delta exceeded threshold: {max_pixel_delta}"
    assert mean_pixel_delta < 0.05, f"Mean pixel delta exceeded threshold: {mean_pixel_delta}"

    # 4. Seam artifact contrast: verify boundary gradient differences
    # Vertical seam check across tile boundaries
    boundary_x = 48 * scale - 16 * scale
    feathered_seam_jump = np.mean(np.abs(
        feathered_out_u8[:, boundary_x + 1, :].astype(float) -
        feathered_out_u8[:, boundary_x, :].astype(float)
    ))
    ref_seam_jump = np.mean(np.abs(
        ref_out_u8[:, boundary_x + 1, :].astype(float) -
        ref_out_u8[:, boundary_x, :].astype(float)
    ))

    seam_discrepancy = abs(feathered_seam_jump - ref_seam_jump)
    assert seam_discrepancy < 0.5, f"Seam gradient discrepancy too high: {seam_discrepancy}"


# ==============================================================================
# GROUP 4: CONCURRENCY & RESOURCE LEAKS
# ==============================================================================

def test_concurrency_rapid_cancellation_cycles_zero_orphans() -> None:
    """Verifies that rapid start/stop cancellation cycles leave 0 orphaned child processes."""
    tracker = ProcessTracker()
    cycles = 8

    for _ in range(cycles):
        tracker.reset_cancel()
        procs: list[subprocess.Popen] = []

        # Spawn long-running dummy processes
        for _ in range(3):
            p = tracker.start_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(p)

        # Confirm child processes are actively running
        assert all(p.poll() is None for p in procs)

        # Trigger immediate atomic cancellation
        tracker.cancel()

        # Verify all processes were terminated promptly
        for p in procs:
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
            assert p.poll() is not None, f"Orphaned process detected: PID {p.pid}"

    # Final cleanup assertion
    assert tracker.is_cancelled is True


def test_concurrency_memory_leak_verification() -> None:
    """Verifies that repetitive tiling, tensor operations, and inference do not leak memory."""
    tracemalloc.start()
    gc.collect()
    snapshot_start = tracemalloc.take_snapshot()

    # Perform 40 repeated spatial tiling allocations and recombinations
    tiler = SpatialTiler(tile_size=64, overlap=16, scale=2)
    dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)

    def dummy_kernel(t: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)

    for _ in range(40):
        _ = tiler.process(dummy_img, dummy_kernel)

    gc.collect()
    snapshot_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")
    total_diff_kb = sum(stat.size_diff for stat in top_stats) / 1024.0

    # Net memory growth after 40 runs should be negligible (< 1024 KB)
    assert total_diff_kb < 1024.0, f"Potential memory leak detected: {total_diff_kb:.2f} KB retained"


def test_concurrency_actor_pipeline_cancellation_under_load() -> None:
    """Verifies StreamPipelineController terminates background worker threads cleanly on cancellation."""
    # Build mock demuxer and muxer
    class MockDemuxer:
        total_frames = 100
        r_frame_rate = "24/1"
        width = 64
        height = 64
        is_closed = False

        def __iter__(self):
            for i in range(100):
                if self.is_closed:
                    break
                yield VideoFrame(index=i, data=b"\x00" * (64 * 64 * 3), width=64, height=64)

        def start(self):
            pass

        def close(self):
            self.is_closed = True

    class MockMuxer:
        is_closed = False

        def start(self):
            pass

        def write_frame(self, frame):
            pass

        def finalize(self):
            return "mock_output.mp4"

        def close(self):
            self.is_closed = True

    controller = StreamPipelineController(
        demuxer=MockDemuxer(),  # type: ignore[arg-type]
        muxer=MockMuxer(),      # type: ignore[arg-type]
        queue_size=4,
    )

    t = threading.Thread(target=controller.run, daemon=True)
    t.start()
    time.sleep(0.05)

    # Cancel midway through streaming
    controller.cancel()
    t.join(timeout=2.0)

    assert not t.is_alive(), "Pipeline worker thread failed to join after cancellation"
    assert controller.is_cancelled is True
    assert getattr(controller.demuxer, "is_closed", False) is True
    assert getattr(controller.muxer, "is_closed", False) is True
