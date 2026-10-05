# Video Enhancer 4K — Animation & Cartoon Edition

High-performance Windows desktop application and CLI for batch upscaling and enhancing 2D animation and cartoon videos (such as *Happy Tree Friends*) to **4K UHD** (or custom resolutions like 1440p, 1080p, 720p).

Built for **exceptional visual fidelity** without ringing or seam artifacts, accelerated via **Vulkan neural inference** and **NVIDIA NVENC hardware encoding**, and adhering strictly to the standardized release conventions of the **Scene / Warez** community.

---

## ⚡ Key Features

- **100% Local Offline Video Processing**:
  - **Zero Cloud Requirement**: Core video extraction, neural inference, audio multiplexing, and final encoding are 100% local. No cloud subscription, telemetry, or network access is needed for processing.
  - **Optional Online Metadata**: Optional TVmaze episode title enrichment is strictly on-demand, non-blocking, and failure-tolerant; processing proceeds seamlessly completely offline if disconnected.
- **Top-Tier Animation AI Models**:
  - **Real-ESRGAN Anime 6B (`realesrgan-x4plus-anime`)**: Specialized neural network for 2D animation. Sharpens linework, eliminates heavy compression macroblocking, and reconstructs clean outlines from SD/480p sources.
  - **Real-ESRGAN AnimeVideo V3 (`realesr-animevideov3`)**: High-throughput video model designed for temporal consistency and minimal frame-to-frame flickering.
  - **Real-ESRGAN General High-Detail (`realesrgan-x4plus`)**: 16-block RRDB architecture for photorealistic detail and complex backgrounds.
  - **RealCUGAN Pro / SE (`realcugan-pro`, `realcugan-se`)**: Specialized deep U-Net architectures for anime illustration with adjustable denoise levels (-1 to 3).
  - **Intelligent Auto-Selection**: Evaluates input resolution, aspect ratio, and target dimensions to select the optimal model and scale multiplier automatically.
- **Robust Pipeline & Strict Verification**:
  - **Verification by Contract**: Already-existing outputs are validated not just by file existence, but by container integrity, exact target resolution (`expected_width` × `expected_height`), duration within tolerance, and audio stream presence.
  - **Atomic Safe Replacement**: Outputs are rendered to isolated temporary files with PID-based lease locking (`.job_lock`) and promoted via atomic replacement. Cross-drive / cross-filesystem moves use safe staged copies.
  - **Active Job Protection**: Stale directory cleanup checks in-memory active jobs and alive PID lock files before deleting abandoned directories.
- **High-Performance Multithreading & Multi-Worker Concurrency**:
  - **Engine Multithreading**: Configurable `-j load:proc:save` thread spec (auto-tuned to available CPU cores and GPU VRAM).
  - **Multi-Worker Scheduling**: Run concurrent workers to overlap CPU frame extraction and hardware encoding of upcoming episodes with GPU neural inference.
  - **Optimized Frame I/O**: High-speed frame extraction and loading via JPEG (`-qscale:v 1`) for high I/O throughput with lossless PNG option available for archival tasks.
- **Visual Previews & Preflight Planning**:
  - **Single Frame Visual Comparison**: Side-by-side zoomable Before/After visual inspector on any selected episode.
  - **Short Clip Previews**: Render short clips (e.g. 5s–30s) starting 10% into the episode to verify settings before running an entire series.
  - **Preflight Check & Dry-Run**: Inspect disk space requirements, effective models, target dimensions, and detect file collisions before rendering.
- **Scene / Warez Release Compliance**:
  - Replaces spaces with dots (`.`), standardizes season/episode numbering (`S01E01`, `TV.S01E01`), and removes prohibited punctuation.
  - Formats standard tags: `Resolution.Source.Audio.Codec-Group` (e.g., `2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4`).
  - **Strict Anonymity**: Output files and video containers are scrubbed of internal process markers and tags (`-map_metadata -1`).

---

## 📋 System Requirements & Prerequisites

### Minimum Requirements
- **OS**: Windows 10 / 11 (64-bit)
- **Python**: Python 3.10 through 3.14 (64-bit)
- **FFmpeg & FFprobe**: Must be installed and accessible in system `PATH` (or specified in environment).
- **GPU**: NVIDIA GeForce GTX 10-series or newer (RTX 20/30/40 recommended) with Vulkan support.
  - *Note: Multi-threaded CPU encoding fallback (`libx265` / `libx264`) is supported if NVENC is unavailable.*
- **AI Upscaler Binaries**: Vulkan ncnn executables (`realesrgan-ncnn-vulkan.exe` and `realcugan-ncnn-vulkan.exe`) placed in `tools/realesrgan/` and `tools/realcugan/`.

---

## 🚀 Quick Start

### 1. Desktop GUI
Double-click:
```cmd
launch.bat
```
or run from PowerShell / Terminal:
```powershell
python main.py
```
On first launch, `launch.bat` verifies the Python environment, installs dependencies from `requirements.txt`, and launches the dark-mode GUI.

### 2. Command-Line Interface (CLI)

Run batch jobs directly in headless or automated environments:
```powershell
# Run batch upscaling to 4K using Cartoon Master profile
python main.py --source "D:\Videos\Source" --output "D:\Videos\Source (2160p)" --preset cartoon_4k_master

# Run non-destructive dry-run mapping
python main.py --source "D:\Videos\Source" --output "D:\Videos\Source (2160p)" --dry-run

# Save or resume from a batch manifest
python main.py --source "D:\Videos\Source" --manifest "D:\Videos\batch_state.json"

# Retry previously failed or cancelled items
python main.py --source "D:\Videos\Source" --retry-failed
```

#### CLI Options Reference

| Argument | Short | Allowed Values / Format | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `-s` | Path | Source folder or single video file path *(required unless `--doctor`, `--list-presets`, or `--manifest` is provided)* |
| `--output` | `-o` | Path | Output directory (default: `'<source> (<resolution>)'`) |
| `--preset` | `-p` | `cartoon_4k_master`, `fast_draft`, `archival`, `lightweight_1080p`, `clean_denoise`, `custom` | Quality profile preset |
| `--resolution`| `-r` | `4k`, `1440p`, `1080p`, `720p` | Target resolution preset |
| `--codec` | `-c` | `hevc_nvenc`, `h264_nvenc`, `libx265`, `libx264` | Video encoder codec |
| `--model` | `-m` | `auto`, `realesrgan-x4plus-anime`, `realesr-animevideov3`, `realcugan-pro`, `realcugan-se`, `realesrgan-x4plus`, `realesrnet-x4plus` | AI upscaling model |
| `--cq` | | `14` to `28` (default: `19`) | NVENC Constant Quality rate control |
| `--nvenc-preset` | | `p1` to `p7` | NVENC speed/quality preset (default: `p7`) |
| `--tta` | | Flag | Enables 8-pass Test-Time Augmentation ensemble inference |
| `--denoise` | | `-1`, `0`, `1`, `2`, `3` | RealCUGAN denoise level (`-1` = conservative) |
| `--audio-mode` | | `copy`, `aac_192k`, `aac_320k`, `flac` | Audio stream handling mode (default: `copy`) |
| `--tile-size` | | Integer (pixels) | Vulkan inference tile size (`0` = auto / full VRAM) |
| `--workers` | `-w` | `1` to `4` | Concurrent batch workers (default: `2`) |
| `--threads` | `-t` | `auto` or `LOAD:PROC:SAVE` (e.g. `2:4:2`) | Engine thread specification |
| `--frame-format`| `-f` | `jpg`, `png` | Extracted frame format (`jpg` = 2x–3x faster I/O) |
| `--limit` | | Integer | Limit number of episodes to process (`0` = all) |
| `--overwrite` | | Flag | Overwrite already existing output files |
| `--dry-run` | | Flag | Preview planned mappings and check disk space without processing |
| `--manifest` | | File path | Save or restore queue state from JSON manifest |
| `--retry-failed` | | Flag | Reset failed or cancelled items to pending |
| `--doctor` | | Flag | Run hardware and environment diagnostic |
| `--export-diagnostics` | | [File path] | Export diagnostic report with redacted user paths |
| `--list-presets` | | Flag | Display available quality profiles and exit |
| `--yes` | `-y` | Flag | Accept displayed AI model choice without interactive prompt |

### 3. System Diagnostic & Privacy Export

Verify your hardware, NVENC availability, FFmpeg installation, and offline model files:
```powershell
python main.py --doctor
```

Export a diagnostic report with sensitive user home paths redacted (`<USER_HOME>`):
```powershell
python main.py --export-diagnostics "diagnostic_report.txt"
```

---

## 🛠️ Project Structure

For detailed system and pipeline architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

```text
Upscaler/
├── launch.bat                # Windows launcher & automated environment bootstrap
├── main.py                   # Unified application entry point (GUI / CLI / Doctor)
├── requirements.txt          # Production dependencies (PyQt6, Pillow)
├── requirements-dev.txt      # Development dependencies (pytest)
├── pytest.ini                # Pytest configuration
├── ARCHITECTURE.md           # In-depth architectural & pipeline specifications
├── README.md                 # Project documentation & user guide
├── src/
│   ├── cli.py                # Command-Line Interface & argument parser
│   ├── core/                 # Core engine & processing components
│   │   ├── ai_models.py      # Facade for models.py
│   │   ├── batch.py          # Batch queue orchestrator, dry-run & manifest handling
│   │   ├── batch_manager.py  # Facade for batch.py
│   │   ├── detector.py       # Facade for hardware.py
│   │   ├── hardware.py       # Hardware, NVENC & external-tool detection
│   │   ├── metadata_provider.py # Resilient TVmaze episode title enrichment
│   │   ├── models.py         # AI models catalog, aspect-ratio & dimension calculations
│   │   ├── pipeline.py       # Facade for video_pipeline.py
│   │   ├── presets.py        # Quality & speed profile definitions
│   │   ├── probe.py          # FFprobe stream inspector & rational framerate analyzer
│   │   ├── process.py        # Process tree management, Windows process groups & cancellation
│   │   ├── scene_namer.py    # Facade for scene_renamer.py
│   │   ├── scene_renamer.py  # Scene/Warez naming conventions & privacy filters
│   │   ├── storage.py        # Atomic file moves, PID lease locking & safe cleanup
│   │   ├── validator.py      # Contract-based verification (resolution, duration, audio)
│   │   └── video_pipeline.py # FFmpeg frame extraction, Vulkan inference & NVENC encoding
│   └── ui/                   # Modern desktop user interface
│       ├── app_qt.py         # GUI entry point & QApplication lifecycle
│       ├── main_window.py    # Responsive QSplitter layout & queue table management
│       ├── theme.py          # Dark theme palette, style tokens & QSS styles
│       ├── workers.py        # Asynchronous QThread workers (batch, scan, preview)
│       └── dialogs/          # Modal inspection & diagnostic dialogs
│           ├── comparison.py # Side-by-side Before/After frame viewer with zoom
│           ├── doctor.py     # System environment diagnostic modal
│           ├── logs.py       # Live color-coded log viewer with HTML escaping
│           └── preflight.py  # Preflight validation & dry-run mapping plan
├── tests/                    # 94 automated unit, integration, and UI tests
└── tools/                    # Local Vulkan AI binaries (Real-ESRGAN, RealCUGAN)
```

---

## 🧪 Testing & Verification

Run the full automated test suite (94 tests):
```powershell
python -m pytest -v
```

All 94 tests validate audio preservation, contract integrity, CLI argument parsing, dry-run mappings, process cancellation safety, 1080p UI responsiveness, and privacy filters.

---

## 🔒 Security & Privacy

- **100% Offline Processing**: No video content or telemetry is ever transmitted over the network.
- **Privacy Scrubbing**: All output files are stripped of internal processing markers and metadata tags via `-map_metadata -1`.
- **Diagnostic Redaction**: `export-diagnostics` automatically masks user paths (`<USER_HOME>`) to prevent leaking private paths.
- **Subprocess Isolation**: Subprocesses are spawned in separate process groups and cleanly terminated via process tree kills upon cancellation.

