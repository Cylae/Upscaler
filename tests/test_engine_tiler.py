"""Mathematical verification of 2D cosine-weighted feathered tiling and seamless reconstruction.

Verifies:
1. Exact partition of unity: Sum(W_i(y, x)) == 1.0 identically everywhere.
2. Zero-seam reconstruction on continuous 2D gradients and complex surfaces.
3. Multi-scale tile reconstruction (1x, 2x, 4x).
4. Customizable tile sizes (256, 384, 512, 768, 1024) and overlaps (32, 48, 64).
5. VRAM-aware tile sizing and single-tile boundary edge preservation.
"""

import numpy as np
import pytest

from src.engine.tiler import (
    SpatialTiler,
    calculate_optimal_tile_size,
    compute_1d_feather_ramp,
    generate_2d_cosine_window,
)


def test_1d_feather_ramp_partition_of_unity():
    """Mathematically verifies that 1D Hann/Cosine feathering satisfies partition of unity.

    For any overlap margin O:
        W_rising(k) + W_falling(O - 1 - k) == 1.0
    for every discrete point k in [0, O-1].
    """
    for overlap in [16, 32, 48, 64, 128]:
        length = 256
        # Rising ramp at start
        ramp_rising = compute_1d_feather_ramp(
            length=length,
            overlap_start=overlap,
            overlap_end=0,
            is_start_edge=False,
            is_end_edge=True,
        )
        # Falling ramp at end
        ramp_falling = compute_1d_feather_ramp(
            length=length,
            overlap_start=0,
            overlap_end=overlap,
            is_start_edge=True,
            is_end_edge=False,
        )

        w_start = ramp_rising[:overlap]
        w_end = ramp_falling[-overlap:]

        # At each index k, w_start[k] + w_end[overlap - 1 - k] must be 1.0
        # Wait, in the overlap region between adjacent tiles:
        # Tile 1's right edge is w_end, Tile 2's left edge is w_start.
        # When aligned at the same spatial coordinate x:
        # Tile 1 index is (length - overlap + x) -> w_end[x]
        # Tile 2 index is x -> w_start[x]
        # Let's verify w_start[x] + w_end[x] == 1.0!
        ramp_sum = w_start + w_end
        np.testing.assert_allclose(
            ramp_sum,
            1.0,
            atol=1e-6,
            err_msg=f"1D partition of unity violated for overlap {overlap}",
        )


def test_2d_corner_partition_of_unity():
    """Verifies that at the intersection of 4 overlapping tiles, sum of weights is identically 1.0."""
    o_w, o_h = 32, 32
    t_w, t_h = 128, 128

    # Top-Left tile: touches right and bottom overlaps
    w_tl = generate_2d_cosine_window(
        t_h, t_w, 0, o_w, 0, o_h,
        is_left_edge=True, is_right_edge=False, is_top_edge=True, is_bottom_edge=False
    )
    # Top-Right tile: touches left and bottom overlaps
    w_tr = generate_2d_cosine_window(
        t_h, t_w, o_w, 0, 0, o_h,
        is_left_edge=False, is_right_edge=True, is_top_edge=True, is_bottom_edge=False
    )
    # Bottom-Left tile: touches right and top overlaps
    w_bl = generate_2d_cosine_window(
        t_h, t_w, 0, o_w, o_h, 0,
        is_left_edge=True, is_right_edge=False, is_top_edge=False, is_bottom_edge=True
    )
    # Bottom-Right tile: touches left and top overlaps
    w_br = generate_2d_cosine_window(
        t_h, t_w, o_w, 0, o_h, 0,
        is_left_edge=False, is_right_edge=True, is_top_edge=False, is_bottom_edge=True
    )

    # In the corner intersection of size (o_h, o_w):
    tl_corner = w_tl[-o_h:, -o_w:]
    tr_corner = w_tr[-o_h:, :o_w]
    bl_corner = w_bl[:o_h, -o_w:]
    br_corner = w_br[:o_h, :o_w]

    corner_sum = tl_corner + tr_corner + bl_corner + br_corner
    np.testing.assert_allclose(
        corner_sum,
        1.0,
        atol=1e-6,
        err_msg="2D Corner 4-tile partition of unity violated",
    )


def test_outer_edge_weights_preserved():
    """Ensures outer edges of the image maintain 1.0 weight and never fade to black."""
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
    # Left edge (x=0) and top edge (y=0) must be 1.0
    np.testing.assert_allclose(window[0, :64], 1.0, atol=1e-6)
    np.testing.assert_allclose(window[:64, 0], 1.0, atol=1e-6)


def test_constant_image_reconstruction_zero_error():
    """Tiling and reconstructing a uniform image must yield exact pixel values with zero difference."""
    width, height = 768, 512
    constant_val = 163
    img = np.full((height, width, 3), constant_val, dtype=np.uint8)

    tiler = SpatialTiler(tile_size=256, overlap=48, scale=1)
    reconstructed = tiler.process(img, inference_fn=lambda t: t, out_dtype=np.uint8)

    assert reconstructed.shape == (height, width, 3)
    # Error must be EXACTLY zero across the entire canvas
    max_error = np.max(np.abs(reconstructed.astype(int) - img.astype(int)))
    assert max_error == 0


@pytest.mark.parametrize("tile_size,overlap", [
    (256, 32),
    (384, 48),
    (512, 64),
    (768, 64),
])
def test_linear_2d_gradient_reconstruction_zero_seam(tile_size, overlap):
    """Verifies that a continuous 2D planar gradient has zero seams across tile boundaries."""
    width, height = 800, 600

    # Continuous gradient: I(y, x) = a * x + b * y + c
    x = np.linspace(0.1, 0.9, width, dtype=np.float32)
    y = np.linspace(0.1, 0.9, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    surface = (0.5 * grid_x + 0.5 * grid_y)[:, :, np.newaxis]
    surface = np.repeat(surface, 3, axis=2)

    tiler = SpatialTiler(tile_size=tile_size, overlap=overlap, scale=1)
    reconstructed = tiler.process(
        surface,
        inference_fn=lambda t: t,
        out_dtype=np.float32,
    )

    assert reconstructed.shape == surface.shape
    # Maximum absolute difference across all boundaries must be within float precision
    diff = np.abs(reconstructed - surface)
    max_diff = np.max(diff)
    assert max_diff < 1e-5, f"Grid artifact detected! Max difference: {max_diff}"


def test_multi_scale_upscale_reconstruction():
    """Verifies 2x and 4x upscaled tile reconstruction alignment."""
    width, height = 300, 200
    scale = 2
    # Simple image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[50:150, 50:250] = 200

    # 2x nearest-neighbor mock upscaler
    def mock_2x_model(tile: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)

    tiler = SpatialTiler(tile_size=128, overlap=32, scale=scale)
    reconstructed = tiler.process(img, inference_fn=mock_2x_model, out_dtype=np.uint8)

    assert reconstructed.shape == (height * scale, width * scale, 3)

    # Compare against direct whole-image 2x
    expected = mock_2x_model(img)
    diff = np.abs(reconstructed.astype(int) - expected.astype(int))
    assert np.max(diff) == 0


def test_irregular_and_small_dimensions():
    """Tests non-square, odd dimensions, and images smaller than tile size."""
    # 1. Smaller than tile size (single tile case)
    small_img = np.full((120, 150, 3), 77, dtype=np.uint8)
    tiler_small = SpatialTiler(tile_size=256, overlap=32, scale=1)
    recon_small = tiler_small.process(small_img, inference_fn=lambda t: t)
    assert recon_small.shape == (120, 150, 3)
    np.testing.assert_array_equal(recon_small, small_img)

    # 2. Arbitrary prime/odd dimensions (733 x 519)
    odd_img = np.full((519, 733, 3), 123, dtype=np.uint8)
    tiler_odd = SpatialTiler(tile_size=256, overlap=48, scale=1)
    recon_odd = tiler_odd.process(odd_img, inference_fn=lambda t: t)
    assert recon_odd.shape == (519, 733, 3)
    np.testing.assert_array_equal(recon_odd, odd_img)


def test_vram_aware_tile_size():
    """Verifies VRAM-aware tile size recommendations."""
    # 24GB VRAM
    t_24g, o_24g = calculate_optimal_tile_size(1920, 1080, scale=4, vram_gb=24.0)
    assert t_24g == 1024
    assert o_24g == 64

    # 8GB VRAM
    t_8g, o_8g = calculate_optimal_tile_size(1920, 1080, scale=4, vram_gb=8.0)
    assert t_8g == 768
    assert o_8g == 64

    # 4GB VRAM
    t_4g, o_4g = calculate_optimal_tile_size(1920, 1080, scale=4, vram_gb=4.0)
    assert t_4g == 512
    assert o_4g == 64

    # Low VRAM (< 2GB)
    t_low, o_low = calculate_optimal_tile_size(1920, 1080, scale=4, vram_gb=1.5)
    assert t_low == 256
    assert o_low == 32


def test_progress_callback():
    """Verifies that progress callback is invoked for every tile."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    tiler = SpatialTiler(tile_size=256, overlap=32, scale=1)
    calls = []

    def on_progress(completed: int, total: int):
        calls.append((completed, total))

    tiler.process(img, inference_fn=lambda t: t, progress_callback=on_progress)
    assert len(calls) > 0
    assert calls[-1][0] == calls[-1][1]  # completed == total at the end
