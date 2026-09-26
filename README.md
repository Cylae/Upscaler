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
- **Python**: Python 3.10, 3.11, or 3.12 (64-bit)
- **FFmpeg & FFprobe**: Must be installed and accessible in system `PATH` (or specified in environment).
- **GPU**: NVIDIA GeForce GTX 10-series or newer (RTX 20/30/40 recommended) with Vulkan support.
  - *Note: CPU encoding fallback (`libx265` / `libx264`) is supported if NVENC is unavailable.*
- **AI Upscaler Binaries**: Vulkan ncnn executables (`realesrgan-ncnn-vulkan.exe` and `realcugan-ncnn-vulkan.exe`) placed in `tools/realesrgan/` and `tools/realcugan/`.

---

## 🚀 Quick Start

### 1. Desktop GUI
Double-click:
```cmd
launch.bat
```
or run from PowerShell:
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

```text
Upscaler/
├── launch.bat                # Windows launcher & environment bootstrap
├── main.py                   # Unified application entry point (GUI / CLI / Doctor)
├── requirements.txt          # Production dependencies (PyQt6, Pillow)
├── requirements-dev.txt      # Development dependencies (pytest, pytest-cov)
├── pytest.ini                # Pytest configuration
├── src/
│   ├── cli.py                # Command-Line Interface & argument parser
│   ├── core/
│   │   ├── ai_models.py      # AI models catalog, aspect-ratio & dimension calculations
│   │   ├── batch_manager.py  # Queue management, state machine, manifests & preflight planning
│   │   ├── detector.py       # Hardware, NVENC & external-tool detection & redacted diagnostics
│   │   ├── metadata_provider.py # Optional, resilient TVmaze metadata client
│   │   ├── presets.py        # Quality & speed profile definitions
│   │   ├── scene_renamer.py  # Scene naming conventions & privacy filters
│   │   └── video_pipeline.py # FFmpeg lifecycle, Vulkan inference, encoding, validation & atomic locking
│   └── ui/
│       └── app_qt.py         # PyQt6 desktop interface, workers, preflight & preview dialogs
├── tests/                    # Automated unit, integration, and UI test suite
└── README.md
```

---

## 🧪 Testing & Verification

Run the full automated test suite:
```powershell
python -m pytest -v
```

Run test suite with test coverage:
```powershell
python -m pytest --cov=src -v
```

---

## 🔒 Security & Privacy

- All generated media files are thoroughly scrubbed of internal process markers and internal tool paths.
- Container metadata is stripped using `-map_metadata -1` to prevent tag leakage.
- Diagnostic export automatically redacts user directory paths to protect personal privacy when troubleshooting.
- Temporary files, locks, and test outputs are excluded from version control via `.gitignore`.
