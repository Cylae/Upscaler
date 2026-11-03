"""Ultra-fast, SIMD-aligned tensor preprocessing and postprocessing.

Provides production-grade conversions between PIL Images, NumPy arrays, and NCHW/NHWC tensors,
supporting FP32/FP16 normalization, zero-copy buffer reuse, SIMD alignment, and letterboxing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

# Multiplier constant for fast FP32 division (multiplication is vectorized and faster than division)
_INV_255_FP32 = np.float32(1.0 / 255.0)
_INV_127_5_FP32 = np.float32(1.0 / 127.5)

# Standard ImageNet normalization parameters (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class TensorLayout(str, enum.Enum):
    """Memory layout for multi-dimensional image tensors."""
    NCHW = "NCHW"
    NHWC = "NHWC"
    CHW = "CHW"
    HWC = "HWC"


class ColorFormat(str, enum.Enum):
    """Color channel order."""
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    RGBA = "RGBA"
    BGRA = "BGRA"


class Normalization(str, enum.Enum):
    """Tensor normalization modes."""
    ZERO_TO_ONE = "zero_to_one"        # [0.0, 1.0]
    MINUS_ONE_TO_ONE = "minus_one_to_one"  # [-1.0, 1.0]
    IMAGENET = "imagenet"              # (x - mean) / std
    NONE = "none"                      # Raw [0, 255] or unchanged


class PaddingMode(str, enum.Enum):
    """Border padding modes for alignment and letterboxing."""
    REFLECT = "reflect"
    EDGE = "edge"
    CONSTANT = "constant"
    SYMMETRIC = "symmetric"


@dataclass(frozen=True, slots=True)
class PaddingInfo:
    """Metadata recorded during padding to allow exact inverse cropping (unpadding)."""
    top: int
    bottom: int
    left: int
    right: int
    original_height: int
    original_width: int
    target_height: int
    target_width: int
    layout: TensorLayout


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Metadata recorded during letterbox scaling to allow exact inverse restoration."""
    scale: float
    pad_left: int
    pad_top: int
    original_height: int
    original_width: int
    target_height: int
    target_width: int


def allocate_simd_aligned(
    shape: Sequence[int],
    dtype: np.dtype | type = np.float32,
    alignment: int = 64,
) -> np.ndarray:
    """Allocates a NumPy array with byte-level memory alignment (e.g. 64-byte for AVX-512).

    Args:
        shape: Dimensions of the target tensor.
        dtype: Data type (default: float32).
        alignment: Memory boundary in bytes (default: 64 for cache line / AVX-512).

    Returns:
        A contiguous, SIMD-aligned C-order NumPy array.
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    total_elements = int(np.prod(shape))
    total_bytes = total_elements * itemsize

    # Allocate raw backing storage with extra margin for alignment
    raw = np.empty(total_bytes + alignment, dtype=np.uint8)
    raw_addr = raw.ctypes.data
    offset = (alignment - (raw_addr % alignment)) % alignment

    # Create array slicing the aligned sub-region with explicit offset
    arr = np.ndarray(shape, dtype=dtype, buffer=raw, offset=offset)
    return arr


def image_to_tensor(
    image: Image.Image | np.ndarray,
    *,
    layout: TensorLayout | str = TensorLayout.NCHW,
    color_format: ColorFormat | str = ColorFormat.RGB,
    source_format: ColorFormat | str = ColorFormat.RGB,
    normalization: Normalization | str = Normalization.ZERO_TO_ONE,
    dtype: np.dtype | type = np.float32,
    add_batch_dim: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Converts an image (PIL Image or NumPy array) into a contiguous neural inference tensor.

    Ruthlessly optimized with SIMD multiplication, contiguous layouts, and zero-allocation
    pathways when a reusable destination buffer `out` is supplied.

    Args:
        image: PIL Image or NumPy ndarray (HWC or HW).
        layout: Target layout (NCHW, NHWC, CHW, HWC).
        color_format: Target channel order (RGB or BGR).
        source_format: Source image channel order (RGB or BGR).
        normalization: Normalization regime (ZERO_TO_ONE, MINUS_ONE_TO_ONE, IMAGENET, NONE).
        dtype: Output data type (np.float32, np.float16, np.uint8).
        add_batch_dim: Whether to prepend batch dimension if layout lacks one.
        out: Optional pre-allocated contiguous array to avoid memory allocations.

    Returns:
        C-contiguous NumPy tensor ready for hardware inference engines.
    """
    layout = TensorLayout(layout)
    color_format = ColorFormat(color_format)
    source_format = ColorFormat(source_format)
    normalization = Normalization(normalization)
    target_dtype = np.dtype(dtype)

    # 1. Convert PIL Image to NumPy array
    if isinstance(image, Image.Image):
        if image.mode == "RGBA" and color_format in (ColorFormat.RGB, ColorFormat.BGR):
            # Fast alpha removal with white background blending or direct RGB extraction
            img_rgb = image.convert("RGB")
            arr = np.asarray(img_rgb)
        elif image.mode != "RGB" and color_format in (ColorFormat.RGB, ColorFormat.BGR):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")

    if arr.ndim == 2:
        # Grayscale (H, W) -> (H, W, 1) or expand to 3 channels if RGB/BGR requested
        if color_format in (ColorFormat.RGB, ColorFormat.BGR):
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        else:
            arr = arr[:, :, np.newaxis]
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array, got ndim={arr.ndim} with shape {arr.shape}")

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Image dimensions must be positive, got shape {arr.shape}")

    # 2. Channel order transformation (RGB <-> BGR)
    if source_format != color_format:
        if (source_format, color_format) in ((ColorFormat.RGB, ColorFormat.BGR), (ColorFormat.BGR, ColorFormat.RGB)) and arr.shape[2] >= 3:
            arr = arr[:, :, [2, 1, 0]]
        elif (source_format, color_format) in ((ColorFormat.RGBA, ColorFormat.BGRA), (ColorFormat.BGRA, ColorFormat.RGBA)) and arr.shape[2] >= 4:
            arr = arr[:, :, [2, 1, 0, 3]]

    h, w, c = arr.shape

    # 3. Compute target shape based on layout and batching
    is_nchw = layout in (TensorLayout.NCHW, TensorLayout.CHW)
    if is_nchw:
        target_shape = (1, c, h, w) if add_batch_dim else (c, h, w)
    else:
        target_shape = (1, h, w, c) if add_batch_dim else (h, w, c)

    # 4. Check or prepare destination buffer
    if out is not None:
        if out.shape != target_shape:
            raise ValueError(f"Provided output buffer shape {out.shape} does not match target shape {target_shape}")
        if out.dtype != target_dtype:
            raise ValueError(f"Provided output buffer dtype {out.dtype} does not match requested dtype {target_dtype}")
        tensor = out
    else:
        tensor = np.empty(target_shape, dtype=target_dtype)

    # 5. Fast normalization and layout transposition
    if is_nchw:
        # Destination view for (c, h, w)
        dest_chw = tensor[0] if tensor.ndim == 4 else tensor

        if normalization == Normalization.ZERO_TO_ONE:
            # Fast vectorized FP32 multiply instead of division
            # Convert to float and transpose: (H, W, C) -> (C, H, W)
            scaled = arr.astype(target_dtype, copy=False) * _INV_255_FP32
            np.copyto(dest_chw, np.transpose(scaled, (2, 0, 1)))

        elif normalization == Normalization.MINUS_ONE_TO_ONE:
            # (x / 127.5) - 1.0 == x * _INV_127_5_FP32 - 1.0
            scaled = arr.astype(target_dtype, copy=False) * _INV_127_5_FP32 - np.float32(1.0)
            np.copyto(dest_chw, np.transpose(scaled, (2, 0, 1)))

        elif normalization == Normalization.IMAGENET:
            scaled = arr.astype(target_dtype, copy=False) * _INV_255_FP32
            transposed = np.transpose(scaled, (2, 0, 1))
            norm = (transposed - _IMAGENET_MEAN.astype(target_dtype)) / _IMAGENET_STD.astype(target_dtype)
            np.copyto(dest_chw, norm)

        elif normalization == Normalization.NONE:
            transposed = np.transpose(arr.astype(target_dtype, copy=False), (2, 0, 1))
            np.copyto(dest_chw, transposed)

    else:
        # NHWC / HWC
        dest_hwc = tensor[0] if tensor.ndim == 4 else tensor

        if normalization == Normalization.ZERO_TO_ONE:
            scaled = arr.astype(target_dtype, copy=False) * _INV_255_FP32
            np.copyto(dest_hwc, scaled)

        elif normalization == Normalization.MINUS_ONE_TO_ONE:
            scaled = arr.astype(target_dtype, copy=False) * _INV_127_5_FP32 - np.float32(1.0)
            np.copyto(dest_hwc, scaled)

        elif normalization == Normalization.IMAGENET:
            scaled = arr.astype(target_dtype, copy=False) * _INV_255_FP32
            mean = _IMAGENET_MEAN.reshape(1, 1, 3).astype(target_dtype)
            std = _IMAGENET_STD.reshape(1, 1, 3).astype(target_dtype)
            norm = (scaled - mean) / std
            np.copyto(dest_hwc, norm)

        elif normalization == Normalization.NONE:
            np.copyto(dest_hwc, arr.astype(target_dtype, copy=False))

    return tensor


def tensor_to_image(
    tensor: np.ndarray,
    *,
    layout: TensorLayout | str = TensorLayout.NCHW,
    color_format: ColorFormat | str = ColorFormat.RGB,
    target_format: ColorFormat | str = ColorFormat.RGB,
    normalization: Normalization | str = Normalization.ZERO_TO_ONE,
    to_pil: bool = True,
    clamp: bool = True,
) -> Image.Image | np.ndarray:
    """Converts a neural inference output tensor back into a contiguous HWC uint8 image or PIL Image.

    Accurately rounds FP32 values (+ 0.5) before uint8 clipping to prevent integer truncation bias.

    Args:
        tensor: Multi-dimensional NumPy array (NCHW, CHW, NHWC, HWC).
        layout: Layout of the input tensor (NCHW, NHWC, CHW, HWC).
        color_format: Channel order inside the tensor (RGB or BGR).
        target_format: Channel order desired in output image (RGB or BGR).
        normalization: Normalization mode used by the model (ZERO_TO_ONE, MINUS_ONE_TO_ONE, IMAGENET, NONE).
        to_pil: If True, returns a PIL.Image; if False, returns a contiguous (H, W, C) uint8 ndarray.
        clamp: If True, securely clamps pixel values to [0, 255].

    Returns:
        PIL Image or C-contiguous uint8 NumPy array.
    """
    layout = TensorLayout(layout)
    color_format = ColorFormat(color_format)
    target_format = ColorFormat(target_format)
    normalization = Normalization(normalization)

    arr = tensor

    # Squeeze batch dimension if present
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 for single image reconstruction, got shape {arr.shape}")
        arr = arr[0]

    # Convert to HWC layout
    if layout in (TensorLayout.NCHW, TensorLayout.CHW):
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D (C, H, W) tensor after batch extraction, got ndim={arr.ndim}")
        # Transpose (C, H, W) -> (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))
    elif layout in (TensorLayout.NHWC, TensorLayout.HWC):
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D (H, W, C) tensor after batch extraction, got ndim={arr.ndim}")

    arr_float = arr.astype(np.float32, copy=False)

    # Denormalize
    if normalization == Normalization.ZERO_TO_ONE:
        # [0.0, 1.0] -> [0.0, 255.0]
        denorm = arr_float * np.float32(255.0)
    elif normalization == Normalization.MINUS_ONE_TO_ONE:
        # [-1.0, 1.0] -> [0.0, 255.0]
        denorm = (arr_float + np.float32(1.0)) * np.float32(127.5)
    elif normalization == Normalization.IMAGENET:
        mean = _IMAGENET_MEAN.reshape(1, 1, 3)
        std = _IMAGENET_STD.reshape(1, 1, 3)
        denorm = (arr_float * std + mean) * np.float32(255.0)
    elif normalization == Normalization.NONE:
        denorm = arr_float
    else:
        denorm = arr_float

    # Clamp and round with half-integer offset for unbiased discretization
    denorm = np.clip(denorm + np.float32(0.5), 0.0, 255.0) if clamp else denorm + np.float32(0.5)

    out_uint8 = denorm.astype(np.uint8)

    # Channel order conversion (e.g. BGR -> RGB)
    if color_format != target_format:
        if (color_format, target_format) in ((ColorFormat.BGR, ColorFormat.RGB), (ColorFormat.RGB, ColorFormat.BGR)) and out_uint8.shape[2] >= 3:
            out_uint8 = out_uint8[:, :, [2, 1, 0]]
        elif (color_format, target_format) in ((ColorFormat.BGRA, ColorFormat.RGBA), (ColorFormat.RGBA, ColorFormat.BGRA)) and out_uint8.shape[2] >= 4:
            out_uint8 = out_uint8[:, :, [2, 1, 0, 3]]

    out_uint8 = np.ascontiguousarray(out_uint8)

    if to_pil:
        channels = out_uint8.shape[2]
        if channels == 1:
            return Image.fromarray(out_uint8[:, :, 0], mode="L")
        elif channels == 3:
            return Image.fromarray(out_uint8, mode="RGB")
        elif channels == 4:
            return Image.fromarray(out_uint8, mode="RGBA")
        else:
            raise ValueError(f"Cannot convert array with {channels} channels to PIL Image")

    return out_uint8


def pad_to_multiple(
    tensor: np.ndarray,
    multiple: int = 32,
    mode: PaddingMode | str = PaddingMode.REFLECT,
    constant_value: float = 0.0,
    layout: TensorLayout | str = TensorLayout.NCHW,
) -> tuple[np.ndarray, PaddingInfo]:
    """Pads a tensor so height and width are strictly divisible by `multiple`.

    Crucial for UNet, Vision Transformer, and multi-scale super-resolution architectures.

    Args:
        tensor: Input tensor (NCHW or NHWC).
        multiple: Required dimension modulus (e.g. 8, 16, 32, 64).
        mode: Padding mode (reflect, edge, constant, symmetric).
        constant_value: Value used when mode is constant.
        layout: Tensor layout (NCHW or NHWC).

    Returns:
        tuple of (padded_tensor, padding_info)
    """
    layout = TensorLayout(layout)
    mode = PaddingMode(mode)

    if layout == TensorLayout.NCHW:
        if tensor.ndim == 4:
            _b, _c, h, w = tensor.shape
        elif tensor.ndim == 3:
            _c, h, w = tensor.shape
        else:
            raise ValueError(f"Invalid tensor ndim {tensor.ndim} for NCHW layout")
    elif layout == TensorLayout.NHWC:
        if tensor.ndim == 4:
            _b, h, w, _c = tensor.shape
        elif tensor.ndim == 3:
            h, w, _c = tensor.shape
        else:
            raise ValueError(f"Invalid tensor ndim {tensor.ndim} for NHWC layout")
    else:
        raise ValueError(f"Unsupported layout for padding: {layout}")

    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        info = PaddingInfo(
            top=0,
            bottom=0,
            left=0,
            right=0,
            original_height=h,
            original_width=w,
            target_height=h,
            target_width=w,
            layout=layout,
        )
        return tensor, info

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_width: tuple[tuple[int, int], ...]
    if layout == TensorLayout.NCHW:
        if tensor.ndim == 4:
            pad_width = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    else:
        if tensor.ndim == 4:
            pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    pad_kwargs: dict[str, Any] = {}
    if mode == PaddingMode.CONSTANT:
        pad_kwargs["constant_values"] = constant_value

    padded = np.pad(tensor, pad_width, mode=mode.value, **pad_kwargs)
    padded = np.ascontiguousarray(padded)

    info = PaddingInfo(
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        original_height=h,
        original_width=w,
        target_height=h + pad_h,
        target_width=w + pad_w,
        layout=layout,
    )
    return padded, info


def unpad(
    tensor: np.ndarray,
    info: PaddingInfo,
    scale: int = 1,
) -> np.ndarray:
    """Removes boundary padding from an upscaled tensor using recorded PaddingInfo.

    Args:
        tensor: Padded tensor after inference.
        info: PaddingInfo returned by pad_to_multiple.
        scale: Upscaling factor applied between pad and unpad (e.g. 1, 2, 4).

    Returns:
        Exact cropped tensor matching original dimensions * scale.
    """
    top = info.top * scale
    bottom = (info.target_height - info.bottom) * scale
    left = info.left * scale
    right = (info.target_width - info.right) * scale

    layout = info.layout
    if layout == TensorLayout.NCHW:
        cropped = tensor[:, :, top:bottom, left:right] if tensor.ndim == 4 else tensor[:, top:bottom, left:right]
    elif layout == TensorLayout.NHWC:
        cropped = tensor[:, top:bottom, left:right, :] if tensor.ndim == 4 else tensor[top:bottom, left:right, :]
    else:
        raise ValueError(f"Unsupported layout in PaddingInfo: {layout}")

    return np.ascontiguousarray(cropped)


def letterbox(
    image: Image.Image | np.ndarray,
    target_size: tuple[int, int],
    fill_value: int | float = 0,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resizes an image preserving aspect ratio and pads the remaining border to target_size (W, H).

    Args:
        image: PIL Image or NumPy ndarray (H, W, C).
        target_size: Target (width, height).
        fill_value: Pixel value for the letterbox bars.

    Returns:
        tuple of (letterboxed_hwc_array, letterbox_info)
    """
    target_w, target_h = target_size
    if isinstance(image, Image.Image):
        orig_w, orig_h = image.size
        arr = np.asarray(image)
    else:
        orig_h, orig_w = image.shape[:2]
        arr = image

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    # Resize image with high-quality area/bilinear interpolation
    pil_img = Image.fromarray(arr) if not isinstance(image, Image.Image) else image
    resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.asarray(resized)

    # Compute symmetric pad
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    c = resized_arr.shape[2] if resized_arr.ndim == 3 else 1
    canvas = np.full((target_h, target_w, c) if c > 1 else (target_h, target_w), fill_value, dtype=resized_arr.dtype)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_arr

    info = LetterboxInfo(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        original_height=orig_h,
        original_width=orig_w,
        target_height=target_h,
        target_width=target_w,
    )
    return np.ascontiguousarray(canvas), info


def unletterbox(
    image: Image.Image | np.ndarray,
    info: LetterboxInfo,
    scale_multiplier: float = 1.0,
) -> Image.Image | np.ndarray:
    """Restores an image to its original unletterboxed aspect ratio and dimension."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
        is_pil = True
    else:
        arr = image
        is_pil = False

    pad_left = round(info.pad_left * scale_multiplier)
    pad_top = round(info.pad_top * scale_multiplier)
    scaled_w = round(info.original_width * info.scale * scale_multiplier)
    scaled_h = round(info.original_height * info.scale * scale_multiplier)

    cropped = arr[pad_top : pad_top + scaled_h, pad_left : pad_left + scaled_w]
    pil_cropped = Image.fromarray(cropped)
    target_orig_w = round(info.original_width * scale_multiplier)
    target_orig_h = round(info.original_height * scale_multiplier)

    restored = pil_cropped.resize((target_orig_w, target_orig_h), Image.Resampling.LANCZOS)
    if is_pil:
        return restored
    return np.ascontiguousarray(np.asarray(restored))
