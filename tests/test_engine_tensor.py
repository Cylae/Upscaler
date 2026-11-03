"""Comprehensive unit tests for engine/tensor.py.

Verifies SIMD memory alignment, NCHW/NHWC conversions, FP32/FP16 normalizations,
zero-copy pre-allocated buffers, padding, unpadding, and roundtrip fidelity.
"""

import numpy as np
from PIL import Image

from src.engine.tensor import (
    ColorFormat,
    Normalization,
    PaddingMode,
    TensorLayout,
    allocate_simd_aligned,
    image_to_tensor,
    letterbox,
    pad_to_multiple,
    tensor_to_image,
    unletterbox,
    unpad,
)


def test_allocate_simd_aligned():
    """Verifies that allocated arrays are C-contiguous and byte-aligned."""
    shape = (1, 3, 256, 256)
    arr = allocate_simd_aligned(shape, dtype=np.float32, alignment=64)
    assert arr.shape == shape
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    # Check 64-byte alignment via ctypes address
    address = arr.ctypes.data
    assert address % 64 == 0


def test_image_to_tensor_basic_rgb():
    """Verifies basic PIL Image to NCHW FP32 [0.0, 1.0] tensor conversion."""
    width, height = 64, 48
    raw_img = Image.new("RGB", (width, height), color=(255, 128, 0))
    tensor = image_to_tensor(raw_img, layout=TensorLayout.NCHW, normalization=Normalization.ZERO_TO_ONE)

    assert tensor.shape == (1, 3, height, width)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]

    # Channel 0 (R): 255 / 255 = 1.0
    # Channel 1 (G): 128 / 255 ≈ 0.50196
    # Channel 2 (B): 0 / 255 = 0.0
    np.testing.assert_allclose(tensor[0, 0, :, :], 1.0, atol=1e-5)
    np.testing.assert_allclose(tensor[0, 1, :, :], 128.0 / 255.0, atol=1e-5)
    np.testing.assert_allclose(tensor[0, 2, :, :], 0.0, atol=1e-5)


def test_image_to_tensor_bgr_conversion():
    """Verifies RGB to BGR channel permutation."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 100))
    tensor_bgr = image_to_tensor(
        img,
        color_format=ColorFormat.BGR,
        source_format=ColorFormat.RGB,
        normalization=Normalization.NONE,
        dtype=np.uint8,
    )
    assert tensor_bgr.shape == (1, 3, 32, 32)
    # BGR: channel 0 is B (100), channel 1 is G (0), channel 2 is R (255)
    assert tensor_bgr[0, 0, 0, 0] == 100
    assert tensor_bgr[0, 1, 0, 0] == 0
    assert tensor_bgr[0, 2, 0, 0] == 255


def test_image_to_tensor_preallocated_out_buffer():
    """Verifies zero-allocation path when pre-allocated buffer is provided."""
    img = Image.new("RGB", (64, 64), color=(50, 150, 200))
    out_buf = np.empty((1, 3, 64, 64), dtype=np.float32)

    ret = image_to_tensor(img, out=out_buf)
    # Verify exact same object was returned
    assert ret is out_buf
    np.testing.assert_allclose(out_buf[0, 0, 0, 0], 50.0 / 255.0, atol=1e-5)
    np.testing.assert_allclose(out_buf[0, 1, 0, 0], 150.0 / 255.0, atol=1e-5)
    np.testing.assert_allclose(out_buf[0, 2, 0, 0], 200.0 / 255.0, atol=1e-5)


def test_normalization_regimes():
    """Verifies ZERO_TO_ONE, MINUS_ONE_TO_ONE, and NONE normalizations."""
    arr = np.array([[[0, 128, 255]]], dtype=np.uint8)  # 1x1x3

    # ZERO_TO_ONE
    t_01 = image_to_tensor(arr, normalization=Normalization.ZERO_TO_ONE, add_batch_dim=False)
    np.testing.assert_allclose(t_01[:, 0, 0], [0.0, 128.0 / 255.0, 1.0], atol=1e-4)

    # MINUS_ONE_TO_ONE
    t_m11 = image_to_tensor(arr, normalization=Normalization.MINUS_ONE_TO_ONE, add_batch_dim=False)
    np.testing.assert_allclose(t_m11[:, 0, 0], [-1.0, (128.0 / 127.5) - 1.0, 1.0], atol=1e-3)

    # NONE
    t_none = image_to_tensor(arr, normalization=Normalization.NONE, dtype=np.float32, add_batch_dim=False)
    np.testing.assert_allclose(t_none[:, 0, 0], [0.0, 128.0, 255.0], atol=1e-5)


def test_roundtrip_fidelity_rgb():
    """Verifies Image -> Tensor -> Image preserves pixel values within discrete rounding bounds."""
    # Create random synthetic RGB image
    rng = np.random.default_rng(42)
    orig_arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    orig_img = Image.fromarray(orig_arr)

    # Convert to NCHW FP32 tensor [0, 1]
    tensor = image_to_tensor(orig_img, layout=TensorLayout.NCHW, normalization=Normalization.ZERO_TO_ONE)

    # Reconstruct back to PIL Image
    recon_img = tensor_to_image(
        tensor,
        layout=TensorLayout.NCHW,
        normalization=Normalization.ZERO_TO_ONE,
        to_pil=True,
    )
    recon_arr = np.asarray(recon_img)

    # Difference due to float32 discretization must be at most 1 pixel level
    diff = np.abs(orig_arr.astype(int) - recon_arr.astype(int))
    max_diff = np.max(diff)
    assert max_diff <= 1
    # Over 99% of pixels must match exactly
    exact_matches = np.mean(diff == 0)
    assert exact_matches > 0.99


def test_pad_to_multiple_and_unpad():
    """Verifies padding to multiple of 32 and exact inverse unpadding restoration."""
    # Arbitrary odd dimensions
    h, w = 117, 233
    tensor = np.ones((1, 3, h, w), dtype=np.float32)

    padded, info = pad_to_multiple(tensor, multiple=32, mode=PaddingMode.REFLECT, layout=TensorLayout.NCHW)
    assert padded.shape[2] % 32 == 0
    assert padded.shape[3] % 32 == 0
    assert padded.shape[2] == 128
    assert padded.shape[3] == 256
    assert info.original_height == h
    assert info.original_width == w

    # Unpad at 1x
    restored = unpad(padded, info, scale=1)
    assert restored.shape == (1, 3, h, w)
    np.testing.assert_array_equal(restored, tensor)

    # Test scaling unpad (e.g. 4x upscale)
    scale = 4
    upscaled_padded = np.ones((1, 3, padded.shape[2] * scale, padded.shape[3] * scale), dtype=np.float32)
    restored_4x = unpad(upscaled_padded, info, scale=scale)
    assert restored_4x.shape == (1, 3, h * scale, w * scale)


def test_letterbox_and_unletterbox():
    """Verifies letterboxing to fixed aspect ratio and restoration."""
    img = Image.new("RGB", (200, 100), color=(100, 150, 200))
    target_size = (256, 256)

    canvas, info = letterbox(img, target_size=target_size, fill_value=0)
    assert canvas.shape == (256, 256, 3)
    assert info.target_width == 256
    assert info.target_height == 256

    # Restore unletterbox
    restored = unletterbox(canvas, info, scale_multiplier=1.0)
    assert isinstance(restored, np.ndarray)
    assert restored.shape == (100, 200, 3)
