# Video Enhancer 4K — Animation & Cartoon Edition

High-performance Windows application for batch upscaling and enhancing 2D animation and cartoon videos (e.g., *Happy Tree Friends*) to **4K UHD** (or custom resolutions).

Built for **exceptional visual fidelity** without artifacts, fully accelerated on **NVIDIA RTX GPUs (Vulkan + NVENC)**, and adhering strictly to the standardized release conventions of the **Scene / Warez** community.

---

## ⚡ Key Features

- **Specialized for 2D Animation & Cartoons**:
  - **Real-ESRGAN Anime 6B**: Contour sharpening, line reconstruction, and complete removal of compression macroblocks without generating artificial film grain.
  - **Real-ESRGAN AnimeVideo V3**: High-speed video model maintaining temporal consistency across frames.
  - **RealCUGAN Pro / SE**: Neural network optimized for anime and cartoon illustration with fine denoise control.
  - **Auto-Selection**: Inspects source resolution (480p, 720p, etc.) to apply the optimal model and scale.
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
python main.py --cli --source "Z:\DL\Happy Tree Friends" --output "Z:\DL\Happy Tree Friends (2160p)" --resolution 4k --codec hevc_nvenc
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
