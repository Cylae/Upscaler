# AnimaScale / Upscaler — Ultra High-Performance AI Video & Image Super-Resolution Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-547%20passed-brightgreen.svg)](tests/)
[![Acceleration](https://img.shields.io/badge/acceleration-TensorRT%20|%20CUDA%20|%20DirectML%20|%20Vulkan-orange.svg)]()
[![Streaming](https://img.shields.io/badge/pipeline-Zero--Disk%20Streaming-blueviolet.svg)]()

**AnimaScale / Upscaler** is an industrial-grade, ultra-high-throughput neural super-resolution system for upscaling anime, realistic footage, and high-resolution digital art to **8K UHD (4320p)**, **4K UHD (2160p)**, **1440p**, and **1080p**.

Engineered from the ground up for **absolute computational efficiency**, the engine completely eliminates intermediate frame disk I/O through **direct in-memory pipe streaming**, guarantees seam-free upscaling via **mathematical cosine-weighted 2D spatial tiling**, and natively unlocks hardware acceleration across **TensorRT**, **CUDA**, **DirectML (DirectX 12)**, **Vulkan NCNN**, and **multi-threaded SIMD CPU**.

---

## ⚡ Core Architectural Pillars

```
                                  +-------------------------------------------------------------+
                                  |                     SOURCE VIDEO / MEDIA                    |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |              FFmpeg VideoStreamDemuxer (pipe:1)             |
                                  |          Raw RGB24 In-Memory Frame Extraction Buffer         |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |              SIMD Memory Alignment & Layout                 |
                                  |     64-Byte Cache Aligned | HWC (uint8) <-> NCHW (FP32)     |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |          Mathematical Cosine-Weighted 2D Tiler              |
                                  |     2D Hann Window | Partition of Unity | C1 Continuity     |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |             HARDWARE EXECUTION PROVIDER HIERARCHY           |
                                  |  TensorRT ➔ CUDA ➔ DirectML (DML) ➔ Vulkan NCNN ➔ SIMD CPU  |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |             Dual Accumulator Blending Canvas                |
                                  |           0.000000 Seam Artifacts Down to Epsilon           |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |               FFmpeg VideoStreamMuxer (pipe:0)              |
                                  |       NVENC / QSV / AMF / libx265 | 10-Bit Main10 Enforced  |
                                  |          All Audio Tracks, Subtitles & Chapters Copied      |
                                  +-------------------------------------------------------------+
                                                                 │
                                                                 ▼
                                  +-------------------------------------------------------------+
                                  |               PRISTINE 4K / 8K MASTER VIDEO                 |
                                  +-------------------------------------------------------------+
```

### 1. Zero-Disk In-Memory Streaming Pipeline (`src/engine/media_stream.py`)
- **Zero Temporary File Disk I/O**: Legacy engines dump 100,000+ uncompressed PNG/JPEG image files to disk for a single episode, incurring 30+ GB of SSD churn and stalling CPU-GPU buses on disk queues. AnimaScale decodes video frames directly from FFmpeg stdout (`-f rawvideo -pix_fmt rgb24 pipe:1`) into in-memory bounded ring buffers, feeds them directly into the neural engine, and streams upscaled tensors straight into the encoder stdin (`pipe:0`).
- **Bounded RAM Footprint**: Backpressure-controlled queue buffers (`maxsize=4..8`) guarantee strict resident memory bounds (< 2 GB RAM peak, ~300 MB typical for 4K video) regardless of video duration or frame count.
- **Full Auxiliary Stream Preservation**: Automatically maps and copies all original audio tracks (`-map 0:a? -c:a copy`), chapters (`-map_chapters 0`), and metadata. Proactively repairs incompatible audio codecs for MP4/MKV.
- **Container-Aware Subtitle Routing**: Preserves lossless subtitle streams, SSA/ASS styling, and font attachments in Matroska (`.mkv`), while transcoding text subtitles (`mov_text`) in MP4 and excluding problematic bitmap streams (PGS/VobSub) to prevent muxing crashes.
- **10-Bit HEVC Main10 Color Depth**: Automatically enforces 10-bit color (`p010le` on hardware encoders, `yuv420p10le` on `libx265`) whenever target resolution is 4K/8K or the source is 10-bit, completely eliminating color gradient banding.

### 2. Mathematical Cosine-Feathered 2D Spatial Tiling (`src/engine/tiler.py`)
- **Partition of Unity**: Implements continuous 2D separable Hann/cosine boundary blending:
  $$W_{start}(k) = 0.5 - 0.5 \cos\left(\frac{\pi (k + 0.5)}{O}\right)$$
  $$W_{end}(k) = 0.5 + 0.5 \cos\left(\frac{\pi (k + 0.5)}{O}\right)$$
  Satisfying $\sum_i W_i(x, y) \equiv 1.000000$ identically across all discrete tile intersections, and 4-way corner tile intersections sum to 1.0.
- **$C^1$ Continuity**: Spatial first derivatives vanish continuously at tile boundaries ($\Delta_{\text{tail}} + \Delta_{\text{head}} \equiv 0$), eliminating all grid lines, luminance jumps, and boundary discontinuities down to float epsilon.
- **Perimeter Edge Preservation**: Tile borders touching the outer image boundary remain clamped at strictly $1.0$, preventing perimeter vignetting or fading.
- **VRAM-Aware Dynamic Tile Sizing**: Dynamically budgets tile size based on detected GPU VRAM (1024, 768, 512, 384, or 256) to ensure 8K and 16K image processing never encounters out-of-memory errors.

### 3. Unified Hardware Execution Provider Hierarchy (`src/engine/onnx_backend.py`)
- **Auto-Discovery Priority**:
  $$\text{TensorRT} \longrightarrow \text{CUDA} \longrightarrow \text{DirectML (DirectX 12)} \longrightarrow \text{Vulkan NCNN} \longrightarrow \text{CPU}$$
- **DirectML Acceleration**: Native support for DirectX 12 hardware acceleration via `DmlExecutionProvider`, accelerating tensor inference on all modern Windows GPUs (Intel UHD/Iris/Arc, AMD Radeon, NVIDIA GeForce) without requiring CUDA installations.
- **Emergency Hardware Fallback**: Automatically catches hardware runtime faults and transparently migrates active sessions to multi-threaded SIMD CPU execution without crashing running batches.
- **Portable Vulkan NCNN**: Native bridge for high-speed portable Vulkan neural executables (`realesrgan-ncnn-vulkan`, `realcugan-ncnn-vulkan`) with thread topology and sync-gap tuning.

### 4. SIMD Tensor Kernels (`src/engine/tensor.py`)
- **64-Byte Cache Line Alignment**: Allocates 64-byte aligned C-contiguous arrays optimized for AVX2, AVX-512, and cache burst transfers.
- **Vectorized Normalization & Unbiased Rounding**: Fast float multiplication (`* 1.0/255.0`) replacing division, and half-integer unbiased rounding (`+ 0.5`) before saturation clipping to prevent color drift across repeated passes.
- **Symmetric Padding & Unpadding**: Exact spatial padding (e.g. divisible by 32 for UNet architectures) with inverse cropping.

---

## 🚀 Installation & Toolchain Setup

### Prerequisites
- **Operating System**: Windows 10 / 11 (64-bit)
- **Python**: Python 3.10 through 3.14 (64-bit)
- **FFmpeg & FFprobe**: Installed and available on system `PATH` (e.g. via `winget install Gyan.FFmpeg`).

### Automatic Environment Setup
Run the setup launcher from PowerShell or Command Prompt:
```cmd
launch.bat
```
This automatically initializes the isolated `.venv`, upgrades `pip`, and installs all optimized neural dependencies (`numpy`, `onnx`, `onnxruntime-directml`, `PyQt6`, `Pillow`).

Alternatively, manually install requirements:
```powershell
python -m pip install -r requirements.txt
```

---

## 💻 Usage Guide

### 1. High-Performance Command-Line Interface (CLI)

The modern CLI supports single files, directories, batch wildcard matching, and recursive scanning.

#### Upscale a Video to 4K Master using Zero-Disk Streaming
```powershell
python -m src.engine.cli -i "input_video.mp4" -o "output_folder" --preset anime_4k_master
```

#### Run Batch Upscaling with Custom DirectML Acceleration
```powershell
python -m src.engine.cli -i "C:\Anime\Season1" -o "C:\Anime\Upscaled_4K" --preset balanced_fast --backend directml --tile-size 384
```

#### Recursive Directory Processing with Photo-Realistic 4K Profile
```powershell
python -m src.engine.cli -i "C:\Photos" -o "C:\Photos\Upscaled" --preset photo_realistic_4k --recursive --scale 4
```

#### CLI Flag Catalog

| Flag | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` / `-s` | Path to source media file or directory. | *(Required)* |
| `--output` | `-o` | Destination directory for upscaled files. | `.` |
| `--preset` | `-p` | Canonical profile (`anime_4k_master`, `photo_realistic_4k`, `balanced_fast`, `ultrawide`, `custom`). | `anime_4k_master` |
| `--backend` | `-b` | Execution backend (`auto`, `directml`, `cuda`, `tensorrt`, `cpu`, `vulkan`). | `auto` |
| `--model` | `-m` | Neural model name or direct path to custom `.onnx` model. | `auto` |
| `--scale` | `-s` | Super-resolution scale factor (`2`, `3`, `4`). | `2` |
| `--tile-size` | `-t` | Tile dimension in pixels (`0` = auto-budgeted from VRAM). | `0` |
| `--overlap` | | Boundary feathering overlap in pixels. | `32` |
| `--codec` | `-c` | Video encoder (`auto`, `hevc_nvenc`, `hevc_qsv`, `hevc_amf`, `libx265`, `h264`). | `auto` |
| `--cq` / `--crf` | | Constant Quality / Constant Rate Factor (14–28). | `19` |
| `--audio` | `-a` | Audio handling mode (`copy`, `aac`, `aac_192k`, `aac_320k`, `flac`, `none`). | `copy` |
| `--recursive` | `-r` | Recursively scan subfolders for supported video and image files. | `False` |
| `--dry-run` | | Simulate preflight planning, dimension resolution, and disk estimates without rendering. | `False` |
| `--diagnostics` | | Print complete system hardware, GPU, and neural execution provider audit. | |
| `--list-presets` | | List all available optimization profiles and exit. | |
| `--list-models` | | List all built-in neural models and exit. | |

---

### 2. Canonical Optimization Profiles

| Profile | Target | Recommended For | Encoder & Quality |
| :--- | :--- | :--- | :--- |
| **`anime_4k_master`** | 4K UHD | 2D Animation, Anime series | HEVC Main10 · CQ 19 · NVENC P7 / libx265 · Lossless Audio Copy |
| **`photo_realistic_4k`** | 4K UHD | Live-action, CGI, Photorealistic footage | Real-ESRGAN x4 · HEVC Main10 · CQ 18 · NVENC P7 |
| **`balanced_fast`** | 4K UHD | High-throughput turnaround (40+ FPS) | AnimeVideo V3 · HEVC CQ 21 · NVENC P3 / Fast libx265 |
| **`ultrawide`** | 1440p / 2160p | 21:9 and 32:9 Aspect Ratios (3440×1440, 5120×2160) | Aspect Preserved · HEVC CQ 19 |
| **`custom`** | User-defined | Unrestricted expert tuning | Fully customizable tiles, threads, denoise, and encoders |

---

### 3. Graphical User Interface (GUI)

Launch the dark-mode desktop interface:
```powershell
python main.py
```
- **Real-Time Telemetry**: Live status badges for detected GPU model, shared/dedicated VRAM, active execution provider (`DirectML`, `Vulkan`, `CUDA`), and hardware encoder availability.
- **Before / After Comparison**: Interactive side-by-side zoomable inspector on any selected video frame.
- **Short Clip Previews**: Render short 5s–30s preview clips starting 10% into the episode to verify settings before running an entire batch.
- **Scene Release Naming**: Automatic formatting adhering to Scene standards: `Resolution.Source.Audio.Codec-Group` (e.g. `2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4`).
- **Deterministic Cancellation**: Ctrl+C or clicking "Stop" cleanly terminates all background processes and cleans up partial files within 1 second.

---

## 🧪 Verification & Adversarial Testing

The entire engine is validated by **547 comprehensive automated tests** across unit, integration, stress, and hostile adversarial suites.

```powershell
# Execute complete test suite
python -m pytest -v

# Run engine tensor, tiling, backend, and media streaming tests
python -m pytest tests/test_engine_*.py tests/test_media_stream.py -v

# Run hostile adversarial overhaul suite
python -m pytest tests/test_adversarial_overhaul.py -v
```

### Adversarial Test Coverage
- **Hostile Image Geometry**: Validated on degenerate $1\times 1$ pixels, prime non-aligned spatial dimensions (`37x53`, `113x227`, `1009x701`), and massive $7680\times 4320$ (8K) / $15360\times 8640$ (16K) resolution buffers under strict VRAM bounds.
- **Corrupt Bitstream Resiliency**: Zero-byte files, truncated JPEG bitstreams, corrupt PNG headers, and invalid MP4 container atoms are rejected with explicit, safe error handling without hanging processes.
- **Mathematical Blending Invariance**: Proves 2D Hann feathered tiling reconstruction matches continuous ground truth within $10^{-5}$ float difference with zero grid boundary discontinuities.
- **Concurrency & Leak Prevention**: Stress-tested with rapid start-and-cancellation cycles, verifying zero zombie processes and bounded memory retention (< 1 MB over 40 tiling cycles).

---

## 📜 License & Compliance

Distributed under the MIT License. Video containers and output media are scrubbed of internal process markers and tags (`-map_metadata -1`) in compliance with standard Scene release rules.
