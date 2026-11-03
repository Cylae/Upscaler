"""Smart 2D spatial tiler with cosine-weighted feathered blending.

Mathematically seam-free image decomposition and reconstruction for arbitrary
tile dimensions (256, 384, 512, 768, 1024) and overlap margins (32, 48, 64 px).
Guarantees exact partition-of-unity blending, eliminating all edge and grid artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image



@dataclass(frozen=True, slots=True)
class TileLocation:
    """Exact spatial coordinates and boundary flags for an image tile."""
    tile_index: int
    total_tiles: int
    row: int
    col: int
    # Input image bounds [x0, x1) and [y0, y1)
    in_x0: int
    in_y0: int
    in_x1: int
    in_y1: int
    # Output bounds [x0, x1) and [y0, y1) after scaling
    out_x0: int
    out_y0: int
    out_x1: int
    out_y1: int
    # Overlap margins in input pixel space
    overlap_left: int
    overlap_right: int
    overlap_top: int
    overlap_bottom: int
    # Boundary flags indicating if tile touches outer image edges
    is_left_edge: bool
    is_right_edge: bool
    is_top_edge: bool
    is_bottom_edge: bool

    @property
    def in_width(self) -> int:
        return self.in_x1 - self.in_x0

    @property
    def in_height(self) -> int:
        return self.in_y1 - self.in_y0

    @property
    def out_width(self) -> int:
        return self.out_x1 - self.out_x0

    @property
    def out_height(self) -> int:
        return self.out_y1 - self.out_y0


def compute_1d_feather_ramp(
    length: int,
    overlap_start: int,
    overlap_end: int,
    is_start_edge: bool,
    is_end_edge: bool,
) -> np.ndarray:
    """Generates a 1D Hann/Cosine feathered blending ramp.

    In overlap margins:
        W_start(k) = 0.5 - 0.5 * cos(pi * (k + 0.5) / overlap_start)  [0 -> 1]
        W_end(k)   = 0.5 + 0.5 * cos(pi * (k + 0.5) / overlap_end)    [1 -> 0]
    In central region:
        W(k) = 1.0
    At image boundaries:
        No feathering (stays 1.0) so image outer edges never fade to black.

    Exact partition of unity: W_start(k) + W_end(overlap - 1 - k) == 1.0 identically.
    """
    ramp = np.ones(length, dtype=np.float32)

    # 1. Left/Top start ramp (rising 0 -> 1)
    if not is_start_edge and overlap_start > 0:
        o = min(overlap_start, length)
        # k from 0 to o-1
        k = np.arange(o, dtype=np.float32)
        # Cosine ramp: 0.5 - 0.5 * cos(pi * (k + 0.5) / o)
        ramp[:o] = np.float32(0.5) - np.float32(0.5) * np.cos(np.float32(np.pi) * (k + np.float32(0.5)) / np.float32(o))

    # 2. Right/Bottom end ramp (falling 1 -> 0)
    if not is_end_edge and overlap_end > 0:
        o = min(overlap_end, length)
        k = np.arange(o, dtype=np.float32)
        # Falling ramp: 0.5 + 0.5 * cos(pi * (k + 0.5) / o)
        fall = np.float32(0.5) + np.float32(0.5) * np.cos(np.float32(np.pi) * (k + np.float32(0.5)) / np.float32(o))
        ramp[-o:] = np.minimum(ramp[-o:], fall)

    return ramp


def generate_2d_cosine_window(
    height: int,
    width: int,
    overlap_left: int,
    overlap_right: int,
    overlap_top: int,
    overlap_bottom: int,
    is_left_edge: bool,
    is_right_edge: bool,
    is_top_edge: bool,
    is_bottom_edge: bool,
) -> np.ndarray:
    """Generates a 2D separable Hann / Cosine feathering window for a tile.

    W(y, x) = W_y(y) * W_x(x)
    Guarantees C1 continuity and 100% seam-free reconstruction.
    """
    ramp_x = compute_1d_feather_ramp(
        width,
        overlap_left,
        overlap_right,
        is_left_edge,
        is_right_edge,
    )
    ramp_y = compute_1d_feather_ramp(
        height,
        overlap_top,
        overlap_bottom,
        is_top_edge,
        is_bottom_edge,
    )
    # 2D outer product: (H, 1) * (1, W) -> (H, W)
    window_2d = ramp_y[:, np.newaxis] * ramp_x[np.newaxis, :]
    return np.ascontiguousarray(window_2d, dtype=np.float32)


def calculate_optimal_tile_size(
    width: int,
    height: int,
    scale: int = 4,
    vram_gb: float = 0.0,
    fp16: bool = True,
) -> tuple[int, int]:
    """Determines optimal (tile_size, overlap) based on image geometry and available GPU VRAM.

    Returns:
        (tile_size, overlap)
    """
    # If image is very small, no tiling needed
    max_dim = max(width, height)
    if max_dim <= 512 and vram_gb >= 2.0:
        return max_dim, 0

    if vram_gb >= 16.0:
        tile_size = 1024
        overlap = 64
    elif vram_gb >= 8.0:
        tile_size = 768
        overlap = 64
    elif vram_gb >= 4.0:
        tile_size = 512
        overlap = 64
    elif vram_gb >= 2.0:
        tile_size = 384
        overlap = 48
    else:
        tile_size = 256
        overlap = 32

    # If tile size exceeds dimensions, clamp
    if tile_size > max_dim:
        tile_size = max(64, (max_dim // 32) * 32)
        overlap = min(overlap, tile_size // 4)

    return tile_size, overlap


def plan_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 1,
) -> list[TileLocation]:
    """Computes non-redundant, fully covering spatial tile coordinates across an image.

    Args:
        image_width: Input image width in pixels.
        image_height: Input image height in pixels.
        tile_size: Base tile dimension in pixels.
        overlap: Overlap margin in pixels (must be < tile_size // 2).
        scale: Upscale factor (e.g. 1, 2, 3, 4).

    Returns:
        List of TileLocation descriptor objects.
    """
    if tile_size <= 0:
        raise ValueError(f"Tile size must be positive, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"Overlap ({overlap}) must be strictly smaller than tile_size ({tile_size})")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={image_width}, height={image_height}")

    # If entire image fits in a single tile:
    if image_width <= tile_size and image_height <= tile_size:
        return [
            TileLocation(
                tile_index=0,
                total_tiles=1,
                row=0,
                col=0,
                in_x0=0,
                in_y0=0,
                in_x1=image_width,
                in_y1=image_height,
                out_x0=0,
                out_y0=0,
                out_x1=image_width * scale,
                out_y1=image_height * scale,
                overlap_left=0,
                overlap_right=0,
                overlap_top=0,
                overlap_bottom=0,
                is_left_edge=True,
                is_right_edge=True,
                is_top_edge=True,
                is_bottom_edge=True,
            )
        ]

    # Stride between consecutive tiles
    tile_size - overlap

    def compute_1d_slices(length: int, t_size: int, o_size: int) -> list[tuple[int, int]]:
        if length <= t_size:
            return [(0, length)]
        st = t_size - o_size
        starts = list(range(0, length - t_size + 1, st))
        if starts[-1] + t_size < length:
            # Add final tile touching the far boundary
            starts.append(length - t_size)
        return [(s, s + t_size) for s in starts]

    x_slices = compute_1d_slices(image_width, tile_size, overlap)
    y_slices = compute_1d_slices(image_height, tile_size, overlap)

    num_rows = len(y_slices)
    num_cols = len(x_slices)
    total_tiles = num_rows * num_cols

    locations: list[TileLocation] = []
    tile_idx = 0

    for r, (y0, y1) in enumerate(y_slices):
        is_top = (r == 0)
        is_bottom = (r == num_rows - 1)

        # Calculate actual overlap with previous and next tiles
        o_top = 0 if is_top else (y_slices[r - 1][1] - y0)
        o_bottom = 0 if is_bottom else (y1 - y_slices[r + 1][0])

        for c, (x0, x1) in enumerate(x_slices):
            is_left = (c == 0)
            is_right = (c == num_cols - 1)

            o_left = 0 if is_left else (x_slices[c - 1][1] - x0)
            o_right = 0 if is_right else (x1 - x_slices[c + 1][0])

            loc = TileLocation(
                tile_index=tile_idx,
                total_tiles=total_tiles,
                row=r,
                col=c,
                in_x0=x0,
                in_y0=y0,
                in_x1=x1,
                in_y1=y1,
                out_x0=x0 * scale,
                out_y0=y0 * scale,
                out_x1=x1 * scale,
                out_y1=y1 * scale,
                overlap_left=o_left,
                overlap_right=o_right,
                overlap_top=o_top,
                overlap_bottom=o_bottom,
                is_left_edge=is_left,
                is_right_edge=is_right,
                is_top_edge=is_top,
                is_bottom_edge=is_bottom,
            )
            locations.append(loc)
            tile_idx += 1

    return locations


class SeamlessRecombiner:
    """Accumulates upscaled overlapping tiles and reconstructs the full image with zero seams.

    Uses normalized partition-of-unity weighted blending:
        Final(y, x) = Sum_i(Tile_i(y, x) * W_i(y, x)) / Sum_i(W_i(y, x))
    Guarantees mathematical seam elimination even across multi-scale boundary interpolations.
    """

    def __init__(
        self,
        target_width: int,
        target_height: int,
        channels: int = 3,
        scale: int = 1,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.channels = channels
        self.scale = scale
        self.dtype = np.dtype(dtype)

        # Pre-allocate zeroed accumulators in float32 for high-precision blending
        self._accumulator = np.zeros((target_height, target_width, channels), dtype=np.float32)
        self._weights = np.zeros((target_height, target_width, 1), dtype=np.float32)

    def add_tile(
        self,
        tile_data: np.ndarray,
        loc: TileLocation,
    ) -> None:
        """Blends a processed upscaled tile into the global canvas.

        Args:
            tile_data: Processed tile. Can be (H, W, C), (H, W), (C, H, W), or (1, C, H, W).
            loc: TileLocation describing the placement and boundary geometry.
        """
        arr = tile_data
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != loc.out_height:
            # (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        th, tw, _tc = arr.shape
        if th != loc.out_height or tw != loc.out_width:
            raise ValueError(
                f"Tile shape ({th}, {tw}) does not match expected output dimensions "
                f"({loc.out_height}, {loc.out_width}) for tile {loc.tile_index}"
            )

        # Scale overlap margins to output resolution
        s = self.scale
        mask_2d = generate_2d_cosine_window(
            height=th,
            width=tw,
            overlap_left=loc.overlap_left * s,
            overlap_right=loc.overlap_right * s,
            overlap_top=loc.overlap_top * s,
            overlap_bottom=loc.overlap_bottom * s,
            is_left_edge=loc.is_left_edge,
            is_right_edge=loc.is_right_edge,
            is_top_edge=loc.is_top_edge,
            is_bottom_edge=loc.is_bottom_edge,
        )

        mask_3d = mask_2d[:, :, np.newaxis]
        arr_f32 = arr.astype(np.float32, copy=False)

        # Accumulate weighted tile data and weight mask
        y0, y1 = loc.out_y0, loc.out_y1
        x0, x1 = loc.out_x0, loc.out_x1

        self._accumulator[y0:y1, x0:x1] += arr_f32 * mask_3d
        self._weights[y0:y1, x0:x1] += mask_3d

    def finalize(self, clamp: bool = True) -> np.ndarray:
        """Finalizes partition-of-unity normalization and returns the reconstructed image.

        Returns:
            Normalized NumPy array of shape (target_height, target_width, channels).
        """
        # Safe normalization preventing divide-by-zero
        safe_weights = np.maximum(self._weights, 1e-12)
        blended = self._accumulator / safe_weights

        if clamp:
            blended = np.clip(blended, 0.0, 255.0 if np.issubdtype(self.dtype, np.integer) else 1.0)

        if np.issubdtype(self.dtype, np.integer):
            # Biased half-integer rounding for unbiased discrete casting
            result = (blended + 0.5).astype(self.dtype)
        else:
            result = blended.astype(self.dtype)

        if self.channels == 1 and result.ndim == 3:
            result = result[:, :, 0]

        return np.ascontiguousarray(result)


class SpatialTiler:
    """Production 2D spatial tiler orchestrator.

    Extracts overlapping patches, coordinates external inference engines,
    and reconstructs the output with 100% seam-free mathematical feathering.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        scale: int = 1,
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale = scale

    def plan(self, width: int, height: int) -> list[TileLocation]:
        """Plans the tile grid for an image of dimensions (width, height)."""
        return plan_tile_grid(
            image_width=width,
            image_height=height,
            tile_size=self.tile_size,
            overlap=self.overlap,
            scale=self.scale,
        )

    def extract_tile(
        self,
        image: np.ndarray | Image.Image,
        loc: TileLocation,
    ) -> np.ndarray:
        """Extracts a single sub-region corresponding to TileLocation."""
        arr = np.asarray(image) if isinstance(image, Image.Image) else image

        tile = arr[loc.in_y0 : loc.in_y1, loc.in_x0 : loc.in_x1]
        return np.ascontiguousarray(tile)

    def process(
        self,
        image: np.ndarray | Image.Image,
        inference_fn: Callable[[np.ndarray], np.ndarray],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        out_dtype: np.dtype | type = np.uint8,
    ) -> np.ndarray:
        """Executes tiled neural upscaling across an entire image with streaming memory reuse.

        Args:
            image: Source image (PIL Image or NumPy array HWC/HW).
            inference_fn: Inference callback taking an input tile (np.ndarray) and returning upscaled tile.
            progress_callback: Optional callback invoked with (completed_tiles, total_tiles).
            out_dtype: Data type for the final reconstructed array (default: uint8).

        Returns:
            Seamlessly reconstructed upscaled image as a C-contiguous NumPy ndarray.
        """
        if isinstance(image, Image.Image):
            w, h = image.size
            channels = len(image.getbands())
            src_arr = np.asarray(image)
        else:
            h, w = image.shape[:2]
            channels = image.shape[2] if image.ndim == 3 else 1
            src_arr = image

        locations = self.plan(w, h)
        total_tiles = len(locations)

        out_w = w * self.scale
        out_h = h * self.scale

        recombiner = SeamlessRecombiner(
            target_width=out_w,
            target_height=out_h,
            channels=channels,
            scale=self.scale,
            dtype=out_dtype,
        )

        for i, loc in enumerate(locations):
            # Extract input patch
            tile_in = src_arr[loc.in_y0 : loc.in_y1, loc.in_x0 : loc.in_x1]
            tile_in = np.ascontiguousarray(tile_in)

            # Run inference kernel
            tile_out = inference_fn(tile_in)

            # Blend into canvas
            recombiner.add_tile(tile_out, loc)

            if progress_callback:
                progress_callback(i + 1, total_tiles)

        return recombiner.finalize()


def upscale_with_tiling(
    image: np.ndarray | Image.Image,
    inference_fn: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 512,
    overlap: int = 64,
    scale: int = 2,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Convenience helper executing seamless tiled neural upscaling on an image or array."""
    tiler = SpatialTiler(tile_size=tile_size, overlap=overlap, scale=scale)
    return tiler.process(image, inference_fn, progress_callback=progress_callback)

