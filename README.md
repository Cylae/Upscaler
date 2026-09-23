# Video Enhancer 4K — Animation & Cartoon Edition

High-performance Windows application for batch upscaling and enhancing 2D animation and cartoon videos (e.g., *Happy Tree Friends*) to **4K UHD** (or custom resolutions).

Built for **exceptional visual fidelity** without artifacts, fully accelerated on **NVIDIA RTX GPUs (Vulkan + NVENC)**, and adhering strictly to the standardized release conventions of the **Scene / Warez** community.

---

## ⚡ Key Features

- **100% Local Offline Storage & Top-Tier Cartoon AI Models**:
  - **Zero Cloud / Completely Offline**: All AI models are pre-packaged and verified locally on disk inside `tools/`. No cloud dependencies, internet downloads, or API keys required.
  - **Real-ESRGAN Anime 6B (`realesrgan-x4plus-anime`)**: Gold standard for 2D animation. Sharpens linework, eliminates heavy compression macroblocks, and restores blurred SD/480p outlines.
  - **Real-ESRGAN AnimeVideo V3 (`realesr-animevideov3`)**: Peak video inference throughput (~37–40 fps on RTX 3070), maintaining temporal consistency and frame-to-frame smoothness across scenes.
  - **Real-ESRGAN General High-Detail (`realesrgan-x4plus`)**: Deep 16-block RRDB architecture for photorealistic detail and complex backgrounds.
  - **RealCUGAN Pro / SE (`models-pro`, `models-se`)**: Specialized deep U-Net architectures for cartoon/anime illustration with fine denoise control.
  - **Auto-Selection**: Inspects source resolution (480p, 720p, etc.) to apply the optimal model and scale without manual tuning.
- **High-Performance Multithreading & Multi-Worker Concurrency**:
  - **Engine Multithreading**: Fine-tuned `-j load:proc:save` thread spec (auto-tuned to `4:4:4` on multi-core CPUs) delivering **over 2.1x speedup** on Vulkan inference.
  - **Multi-Worker Parallel Processing**: Run up to 2–3 video pipelines concurrently. Overlaps CPU frame extraction and NVENC hardware encoding of Episode N+1 with GPU inference of Episode N, eliminating idle bottlenecks on RTX GPUs.
  - **Optimized Frame I/O Pipeline**: High-speed frame extraction and loading via JPEG (`-qscale:v 1`), cutting disk I/O and frame decode time by 2x–3x compared to raw PNG with zero visible loss on 2D line art.
- **Hardware Accelerated Pipeline**:
  - Neural inference via **Vulkan FP16** on NVIDIA GPUs (e.g., RTX 3070).
  - Ultra-fast hardware encoding via **NVIDIA NVENC** (`hevc_nvenc` H.265 or `h264_nvenc` H.264).
  - High-grade Lanczos scaling, quality preset P7, and Constant Quality (`-cq 19`) rate control for transparent visuals and compact file sizes.
  - Automated CPU fallback (`libx265` / `libx264`) when hardware encoders are unavailable.
- **Scene / Warez Release Compliance**:
  - Replaces spaces with dots (`.`).
  - Standardizes season and episode tags (`S01E01`, `S02E25`, `TV.S01E01`, `S00E01`).
  - Strips forbidden characters (`!`, `'`, `?`, `:`, `,`).
  - Standard tags: `Resolution.Source.Audio.Codec-Group` (e.g., `2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4`).
  - **Strict Anonymity**: No internal tooling or model references in output filenames or video metadata (`-map_metadata -1`).
- **Batch Processing & Resume Capability**:
  - Instant indexing of large video libraries (e.g., all 227 episodes of *Happy Tree Friends*).
  - Smart resume: already-completed episodes are automatically recognized and skipped to prevent duplicate rendering.
  - Lossless audio stream copy (`-c:a copy`) with audio-video sync preservation.

---

## 🚀 Quick Start

### Option 1: Desktop GUI (1-Click)
Double-click:
```text
launch.bat
```
or run from your terminal:
```powershell
python main.py
```

On first launch, `launch.bat` automatically sets up the local `.venv`, installs dependencies from `requirements.txt`, and opens the GUI.

### Option 2: Command-Line Interface (CLI)
Run batch jobs directly in headless mode:
```powershell
python main.py --cli --source "Z:\DL\Happy Tree Friends" --output "Z:\DL\Happy Tree Friends (2160p)" --resolution 4k --codec hevc_nvenc --workers 2 --threads 4:4:4 --frame-format jpg
```

### System Diagnostic:
Run the built-in diagnostic tool to verify GPU, NVENC, and portable engines:
```powershell
launch.bat --doctor
```
or
```powershell
python main.py --doctor
```

---

## 🛠️ Project Structure

```text
Upscaler/
├── launch.bat                # 1-click Windows launcher
├── main.py                   # Main entry point (GUI / CLI / Doctor)
├── requirements.txt          # Python dependencies
├── src/
│   ├── core/
│   │   ├── ai_models.py      # Model catalog & automatic selector
│   │   ├── batch_manager.py  # Queue management & resume logic
│   │   ├── detector.py       # Hardware & encoder detection
│   │   ├── scene_renamer.py  # Warez / Scene naming engine & privacy filter
│   │   └── video_pipeline.py # End-to-end video pipeline (FFmpeg + Vulkan + NVENC)
│   └── ui/
│       └── app_qt.py         # Modern PyQt6 dark mode interface
└── README.md
```

---

## 🔒 Privacy & Clean Output

- All generated media files are thoroughly scrubbed of internal process markers.
- Metadata is stripped using `-map_metadata -1` to prevent tag leakage.
- Testing scripts, test renders, and temporary files are strictly excluded from git tracking via `.gitignore`.
