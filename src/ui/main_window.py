"""Main application window for Video Enhancer 4K."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo, get_base_dir, get_hardware_info
from src.core.models import (
    AVAILABLE_MODELS,
    check_local_model_availability,
    recommend_model_for_video,
)
from src.core.pipeline import VideoUpscalerPipeline
from src.core.presets import ENHANCEMENT_PRESETS, get_preset
from src.core.probe import probe_video
from src.core.scene_namer import (
    SceneNamingConfig,
    format_output_name,
    format_scene_name,
    get_scene_relative_output_path,
)
from src.core.storage import cleanup_stale_temp_directories
from src.ui.dialogs.comparison import FrameComparisonDialog
from src.ui.dialogs.doctor import DoctorDialog
from src.ui.dialogs.logs import LogViewerDialog
from src.ui.dialogs.preflight import PreflightDialog
from src.ui.theme import DARK_STYLESHEET
from src.ui.workers import (
    FramePreviewWorker,
    ModelHintWorker,
    ScanWorker,
    UpscaleWorker,
)


class UpscalerMainWindow(QMainWindow):
    """Main window for batch upscaling, quality presets, and queue management."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancer 4K — Animation & Cartoon Edition")
        self.resize(1400, 860)
        self.setMinimumSize(960, 660)
        self.setAcceptDrops(True)

        self.hardware = get_hardware_info()
        self.batch_manager = BatchManager(self.hardware)
        self.worker: Optional[UpscaleWorker] = None
        self._scan_worker: Optional[ScanWorker] = None
        self._current_filter_mode = "all"
        self._log_buffer: list = []
        self._last_cli_cmd: str = ""
        self._log_viewer: Optional[LogViewerDialog] = None
        self._hint_worker: Optional[ModelHintWorker] = None
        self._frame_worker: Optional[FramePreviewWorker] = None
        self.settings = QSettings("CYLAE", "VideoEnhancer4K")

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        last_src = self.settings.value("last_source", "")
        last_out = self.settings.value("last_output", "")
        if last_src and os.path.exists(str(last_src)):
            self.txt_source.setText(str(last_src))
            if last_out:
                self.txt_output.setText(str(last_out))
            else:
                self.txt_output.setText(str(last_src) + " (2160p)")
            QTimer.singleShot(150, self.on_scan_clicked)
        else:
            default_dir = r"Z:\DL\Happy Tree Friends"
            if os.path.isdir(default_dir):
                self.txt_source.setText(default_dir)
                self.txt_output.setText(r"Z:\DL\Happy Tree Friends (2160p)")
                QTimer.singleShot(150, self.on_scan_clicked)

        try:
            cleanup_stale_temp_directories()
        except Exception:
            pass

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        self.main_layout = root_layout
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 1. HEADER BAR (full width) ──────────────────────────────────────
        header = QFrame()
        header.setObjectName("headerPanel")
        header.setFixedHeight(62)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 0, 16, 0)
        h_layout.setSpacing(12)

        # Brand logo badge
        badge = QLabel("4K")
        badge.setFixedSize(36, 36)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #7c3aed);"
            "border: 1px solid #4f46e5; border-radius: 9px; color: #ffffff;"
            "font-size: 13px; font-weight: 900; letter-spacing: 0.5px;"
        )
        h_layout.addWidget(badge)
        h_layout.addSpacing(4)

        # Title
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        lbl_title = QLabel("Video Enhancer 4K")
        lbl_title.setStyleSheet(
            "font-size: 15px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.3px;"
        )
        lbl_sub = QLabel("Animation & Cartoon Edition  ·  Local AI Upscaler")
        lbl_sub.setStyleSheet("font-size: 11px; color: #64748b; font-weight: 500;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        h_layout.addLayout(title_box)

        h_layout.addStretch()

        # Status chips (always visible — no hiding logic needed in new layout)
        chips_layout = QHBoxLayout()
        chips_layout.setSpacing(8)

        self.lbl_gpu = QLabel(f"⚡ {self.hardware.gpu_name}  {self.hardware.gpu_vram_gb} GB")
        self.lbl_gpu.setStyleSheet(
            "color: #34d399; font-weight: 700; font-size: 11px; padding: 4px 12px;"
            "background-color: rgba(52, 211, 153, 0.10); border: 1px solid rgba(52, 211, 153, 0.25);"
            "border-radius: 14px;"
        )
        chips_layout.addWidget(self.lbl_gpu)

        nvenc_text = "NVENC H.265" if self.hardware.has_nvenc_hevc else "CPU x265"
        self.lbl_encoder = QLabel(f"🎞 {nvenc_text}")
        self.lbl_encoder.setStyleSheet(
            "color: #60a5fa; font-weight: 700; font-size: 11px; padding: 4px 12px;"
            "background-color: rgba(96, 165, 250, 0.10); border: 1px solid rgba(96, 165, 250, 0.25);"
            "border-radius: 14px;"
        )
        chips_layout.addWidget(self.lbl_encoder)

        self.lbl_audio = QLabel("🔊 Lossless Audio")
        self.lbl_audio.setStyleSheet(
            "color: #a78bfa; font-weight: 700; font-size: 11px; padding: 4px 12px;"
            "background-color: rgba(167, 139, 250, 0.10); border: 1px solid rgba(167, 139, 250, 0.25);"
            "border-radius: 14px;"
        )
        chips_layout.addWidget(self.lbl_audio)
        h_layout.addLayout(chips_layout)

        h_layout.addSpacing(6)

        btn_doctor = QPushButton("🩺 Doctor")
        btn_doctor.setToolTip("Run hardware diagnostics and verify tool availability.")
        btn_doctor.setStyleSheet(
            "padding: 5px 12px; font-size: 11px; font-weight: 600;"
            "border-radius: 12px; background-color: #1c1c2a;"
            "border: 1px solid #252535; color: #94a3b8;"
        )
        btn_doctor.clicked.connect(self.show_doctor_dialog)
        h_layout.addWidget(btn_doctor)

        self.btn_open_logs = QPushButton("📋 Logs (0)")
        self.btn_open_logs.setToolTip(
            "Open the live engine log viewer. Shows FFmpeg, Vulkan AI, and audio pipeline commands."
        )
        self.btn_open_logs.setStyleSheet(
            "padding: 5px 12px; font-size: 11px; font-weight: 600;"
            "background-color: #1c1c2a; border: 1px solid #252535;"
            "border-radius: 12px; color: #64748b;"
        )
        self.btn_open_logs.clicked.connect(self._open_log_viewer)
        h_layout.addWidget(self.btn_open_logs)

        root_layout.addWidget(header)

        # ── 2. MIDDLE ZONE: sidebar | queue (resizable splitter) ───────────
        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.setObjectName("mainSplitter")
        middle.setHandleWidth(2)
        middle.setChildrenCollapsible(False)
        self.main_splitter = middle

        # ── 2a. LEFT SIDEBAR ───────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(520)
        sidebar_outer = QVBoxLayout(sidebar)
        sidebar_outer.setContentsMargins(0, 0, 0, 0)
        sidebar_outer.setSpacing(0)

        # Scrollable inner area
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("background-color: transparent; border: none;")

        sidebar_inner = QWidget()
        sidebar_inner.setStyleSheet("background-color: transparent;")
        s_layout = QVBoxLayout(sidebar_inner)
        s_layout.setContentsMargins(12, 12, 10, 12)
        s_layout.setSpacing(8)

        # ── Section: Source & Output ─────────────────────────────────────
        grp_io = QGroupBox("Source & Output")
        grp_io.setObjectName("sideCard")
        io_layout = QVBoxLayout(grp_io)
        io_layout.setContentsMargins(10, 16, 10, 10)
        io_layout.setSpacing(8)

        # Source row
        lbl_src = QLabel("Source folder / files")
        lbl_src.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        io_layout.addWidget(lbl_src)
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Drag & drop or browse...")
        self.txt_source.setToolTip(
            "Path to the input directory or files. Drag-and-drop video files or folders directly here."
        )
        self.txt_source.setClearButtonEnabled(True)
        io_layout.addWidget(self.txt_source)

        src_btn_row = QHBoxLayout()
        src_btn_row.setSpacing(6)
        btn_add_files = QPushButton("➕ Files")
        btn_add_files.setToolTip("Select one or more individual video files to add to the queue.")
        btn_add_files.clicked.connect(self.on_add_files_clicked)
        src_btn_row.addWidget(btn_add_files)
        btn_browse_src = QPushButton("📁 Folder")
        btn_browse_src.setToolTip("Browse and select the input video folder.")
        btn_browse_src.clicked.connect(self.on_browse_source)
        src_btn_row.addWidget(btn_browse_src)
        self.btn_scan = QPushButton("🔍 Scan")
        self.btn_scan.setObjectName("btnScan")
        self.btn_scan.setToolTip("Scan source directory and index all video files into the queue.")
        self.btn_scan.clicked.connect(self.on_scan_clicked)
        src_btn_row.addWidget(self.btn_scan)
        io_layout.addLayout(src_btn_row)

        # Output row
        lbl_out = QLabel("Output folder")
        lbl_out.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 4px;")
        io_layout.addWidget(lbl_out)
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Destination for 4K videos...")
        self.txt_output.setToolTip(
            "Path to the target directory where enhanced 4K files will be saved."
        )
        self.txt_output.setClearButtonEnabled(True)
        io_layout.addWidget(self.txt_output)

        out_btn_row = QHBoxLayout()
        out_btn_row.setSpacing(6)
        btn_browse_out = QPushButton("📁 Browse")
        btn_browse_out.setToolTip("Browse and select the destination directory.")
        btn_browse_out.clicked.connect(self.on_browse_output)
        out_btn_row.addWidget(btn_browse_out)
        btn_open_out = QPushButton("📂 Open")
        btn_open_out.setToolTip("Open destination directory in Windows File Explorer.")
        btn_open_out.clicked.connect(self.on_open_output)
        out_btn_row.addWidget(btn_open_out)
        out_btn_row.addStretch()
        io_layout.addLayout(out_btn_row)

        s_layout.addWidget(grp_io)

        # ── Section: Quality Preset ──────────────────────────────────────
        grp_preset = QGroupBox("Quality Preset")
        grp_preset.setObjectName("sideCard")
        preset_layout = QVBoxLayout(grp_preset)
        preset_layout.setContentsMargins(10, 16, 10, 10)
        preset_layout.setSpacing(8)

        self.cb_preset = QComboBox()
        self.cb_preset.setToolTip(
            "1-click curated enhancement profile. Auto-configures resolution, AI model, CQ, and NVENC."
        )
        for p_id, p_obj in ENHANCEMENT_PRESETS.items():
            self.cb_preset.addItem(f"{p_obj.icon}  {p_obj.name}", p_id)
        self.cb_preset.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.cb_preset)

        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 0px; line-height: 1.3;")
        preset_layout.addWidget(self.lbl_preset_desc)
        s_layout.addWidget(grp_preset)

        # ── Section: Encoding ────────────────────────────────────────────
        grp_enc = QGroupBox("Encoding")
        grp_enc.setObjectName("sideCard")
        enc_layout = QVBoxLayout(grp_enc)
        enc_layout.setContentsMargins(10, 16, 10, 10)
        enc_layout.setSpacing(8)

        lbl_res = QLabel("Resolution")
        lbl_res.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        enc_layout.addWidget(lbl_res)
        self.cb_resolution = QComboBox()
        self.cb_resolution.setToolTip(
            "Target upscaling resolution. 4K UHD (3840x2160) provides pristine line clarity."
        )
        self.cb_resolution.addItems(
            ["4K UHD (3840×2160)", "1440p QHD (2560×1440)", "1080p FHD (1920×1080)", "720p HD (1280×720)"]
        )
        self.cb_resolution.currentIndexChanged.connect(self.on_resolution_changed)
        enc_layout.addWidget(self.cb_resolution)

        lbl_codec = QLabel("Video Codec")
        lbl_codec.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        enc_layout.addWidget(lbl_codec)
        self.cb_codec = QComboBox()
        self.cb_codec.setToolTip(
            "Video encoding engine. NVENC Hardware provides ultra-fast GPU encoding with spatial/temporal AQ."
        )
        self.cb_codec.addItems(
            ["H.265 (HEVC NVENC Hardware)", "H.264 (NVENC Hardware)", "H.265 (CPU x265)"]
        )
        self.cb_codec.currentIndexChanged.connect(self.on_settings_changed)
        enc_layout.addWidget(self.cb_codec)
        s_layout.addWidget(grp_enc)

        # ── Section: AI Model ────────────────────────────────────────────
        grp_ai = QGroupBox("AI Upscale Model")
        grp_ai.setObjectName("sideCard")
        ai_layout = QVBoxLayout(grp_ai)
        ai_layout.setContentsMargins(10, 16, 10, 10)
        ai_layout.setSpacing(8)

        self.cb_model = QComboBox()
        self.cb_model.setToolTip("Neural network model used for line reconstruction and upscaling.")
        for m_id, m_spec in AVAILABLE_MODELS.items():
            self.cb_model.addItem(m_spec.name, m_id)
        self.cb_model.currentIndexChanged.connect(self.on_model_changed)
        ai_layout.addWidget(self.cb_model)

        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 0px; line-height: 1.3;")
        ai_layout.addWidget(self.lbl_model_info)
        s_layout.addWidget(grp_ai)

        # ── Section: Naming ──────────────────────────────────────────────
        grp_naming = QGroupBox("File Naming")
        grp_naming.setObjectName("sideCard")
        naming_layout = QVBoxLayout(grp_naming)
        naming_layout.setContentsMargins(10, 16, 10, 10)
        naming_layout.setSpacing(8)

        self.cb_naming_mode = QComboBox()
        self.cb_naming_mode.setToolTip("File naming convention for enhanced output videos.")
        self.cb_naming_mode.addItem("🏷️ Clean  (e.g. Video_4K.mp4)", "clean")
        self.cb_naming_mode.addItem("🎬 Scene Release  (Show.S01E01…-CYLAE.mp4)", "scene")
        self.cb_naming_mode.addItem("📄 Original Name  (e.g. Video.mp4)", "original")
        self.cb_naming_mode.currentIndexChanged.connect(self.on_settings_changed)
        naming_layout.addWidget(self.cb_naming_mode)

        self.lbl_preview_name = QLabel("")
        self.lbl_preview_name.setWordWrap(True)
        self.lbl_preview_name.setToolTip(
            "Standardized Scene release filename preview generated dynamically based on active settings."
        )
        self.lbl_preview_name.setStyleSheet(
            "background-color: #090912; border: 1px solid #1e1e2e; border-radius: 6px;"
            "padding: 6px 10px; color: #4f8fff;"
            "font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 10px;"
        )
        naming_layout.addWidget(self.lbl_preview_name)
        s_layout.addWidget(grp_naming)

        # ── Section: Options ─────────────────────────────────────────────
        grp_opts = QGroupBox("Options")
        grp_opts.setObjectName("sideCard")
        opts_layout = QVBoxLayout(grp_opts)
        opts_layout.setContentsMargins(10, 16, 10, 10)
        opts_layout.setSpacing(6)

        self.chk_preserve_folders = QCheckBox("Preserve subfolder structure")
        self.chk_preserve_folders.setToolTip(
            "Replicate source subfolder hierarchy (e.g. Season 01/) inside the output folder."
        )
        self.chk_preserve_folders.setChecked(True)
        opts_layout.addWidget(self.chk_preserve_folders)

        self.chk_overwrite = QCheckBox("Overwrite existing files")
        self.chk_overwrite.setToolTip(
            "Re-process and overwrite files that already exist in the output directory."
        )
        self.chk_overwrite.setChecked(False)
        opts_layout.addWidget(self.chk_overwrite)
        s_layout.addWidget(grp_opts)

        # ── Expert toggle button ─────────────────────────────────────────
        self.btn_toggle_expert = QPushButton("⚙️  Advanced & Expert Controls ▼")
        self.btn_toggle_expert.setObjectName("btnToggleExpert")
        self.btn_toggle_expert.setToolTip(
            "Show/hide fine-tuning controls: CQ factor, NVENC preset, threads, tile size, and audio modes."
        )
        self.btn_toggle_expert.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_toggle_expert.clicked.connect(self.toggle_expert_panel)
        s_layout.addWidget(self.btn_toggle_expert)

        # ── Expert Panel ─────────────────────────────────────────────────
        self.grp_expert = QGroupBox("⚙️ Advanced & Expert Controls")
        self.grp_expert.setObjectName("sideCard")
        expert_layout = QVBoxLayout(self.grp_expert)
        expert_layout.setContentsMargins(10, 16, 10, 10)
        expert_layout.setSpacing(8)

        # CQ Slider
        lbl_cq = QLabel("CQ Factor  (14 = lossless → 28 = draft)")
        lbl_cq.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        expert_layout.addWidget(lbl_cq)
        cq_row = QHBoxLayout()
        self.slider_cq = QSlider(Qt.Orientation.Horizontal)
        self.slider_cq.setRange(14, 28)
        self.slider_cq.setValue(19)
        self.slider_cq.setToolTip(
            "Constant Quality factor (14-28). Lower CQ delivers higher bitrate. 19 is visually lossless for 4K."
        )
        self.slider_cq.valueChanged.connect(self.on_cq_slider_changed)
        cq_row.addWidget(self.slider_cq, 1)
        self.lbl_cq_val = QLabel("CQ 19")
        self.lbl_cq_val.setFixedWidth(52)
        self.lbl_cq_val.setStyleSheet("color: #60a5fa; font-weight: 700; font-size: 11px; text-align: right;")
        cq_row.addWidget(self.lbl_cq_val)
        expert_layout.addLayout(cq_row)

        # NVENC Preset
        lbl_nvenc = QLabel("NVENC Preset")
        lbl_nvenc.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_nvenc)
        self.cb_nvenc_preset = QComboBox()
        self.cb_nvenc_preset.setToolTip(
            "Hardware NVENC speed vs compression trade-off. p7 = highest quality/compression."
        )
        self.cb_nvenc_preset.addItems([
            "p7 (Slowest / Max Compression)",
            "p6 (High Quality)",
            "p5 (Medium Quality)",
            "p4 (Fast Draft)",
            "p1 (Fastest / Low Overhead)",
        ])
        self.cb_nvenc_preset.currentIndexChanged.connect(self.on_expert_setting_changed)
        expert_layout.addWidget(self.cb_nvenc_preset)

        # Audio Mode
        lbl_audio_mode = QLabel("Soundtrack")
        lbl_audio_mode.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_audio_mode)
        self.cb_audio_mode = QComboBox()
        self.cb_audio_mode.setToolTip(
            "Audio stream management. Lossless Stream Copy remuxes the original soundtrack with zero loss."
        )
        self.cb_audio_mode.addItem("Lossless Stream Copy (-c:a copy)", "copy")
        self.cb_audio_mode.addItem("Re-encode AAC 192k (Transparent)", "aac_192k")
        self.cb_audio_mode.addItem("Re-encode AAC 320k (Audiophile)", "aac_320k")
        self.cb_audio_mode.addItem("Lossless FLAC 24-bit", "flac")
        self.cb_audio_mode.currentIndexChanged.connect(self.on_expert_setting_changed)
        expert_layout.addWidget(self.cb_audio_mode)

        # Workers & Threads
        lbl_workers = QLabel("Workers")
        lbl_workers.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_workers)
        self.cb_workers = QComboBox()
        self.cb_workers.setToolTip(
            "Number of episodes processed simultaneously. 2 workers saturates GPU on RTX 3070."
        )
        self.cb_workers.addItems(
            ["2 Workers (Parallel — Recommended)", "1 Worker (Sequential)", "3 Workers (Max Throughput)"]
        )
        expert_layout.addWidget(self.cb_workers)

        lbl_threads = QLabel("Vulkan Threads")
        lbl_threads.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_threads)
        self.cb_threads = QComboBox()
        self.cb_threads.setToolTip(
            "Vulkan worker threads (load:proc:save). Auto exploits high core count CPUs."
        )
        cpu_cores = os.cpu_count() or 4
        self.cb_threads.addItems(
            [f"Auto ({cpu_cores} Cores — 4:4:4)", "4:4:4 (Max Multi-Core)", "2:4:2 (Balanced)", "1:2:2 (Default)"]
        )
        expert_layout.addWidget(self.cb_threads)

        # Frame Format & Tile Size
        lbl_fmt = QLabel("Frame Format")
        lbl_fmt.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_fmt)
        self.cb_frame_fmt = QComboBox()
        self.cb_frame_fmt.setToolTip(
            "Intermediate frame format. Fast JPEG is 2x-3x faster disk I/O; PNG is lossless."
        )
        self.cb_frame_fmt.addItems(["Fast JPEG (2x–3x Faster I/O)", "Lossless PNG"])
        expert_layout.addWidget(self.cb_frame_fmt)

        lbl_tile = QLabel("Tile Size")
        lbl_tile.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_tile)
        self.cb_tile_size = QComboBox()
        self.cb_tile_size.setToolTip(
            "Tile split size. 0 (Auto) uses entire frame — best for 8 GB+ VRAM."
        )
        self.cb_tile_size.addItem("0 — Auto / Full VRAM", 0)
        self.cb_tile_size.addItem("200 px  (Low VRAM)", 200)
        self.cb_tile_size.addItem("300 px", 300)
        self.cb_tile_size.addItem("400 px", 400)
        self.cb_tile_size.addItem("600 px", 600)
        expert_layout.addWidget(self.cb_tile_size)

        # Denoise & TTA
        lbl_denoise = QLabel("RealCUGAN Denoise Level")
        lbl_denoise.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; margin-top: 2px;")
        expert_layout.addWidget(lbl_denoise)
        self.cb_denoise = QComboBox()
        self.cb_denoise.setToolTip("RealCUGAN noise reduction filter. Suppresses ringing and compression artifacts.")
        self.cb_denoise.addItem("None (-1)", -1)
        self.cb_denoise.addItem("Conservative (0)", 0)
        self.cb_denoise.addItem("Standard (1)", 1)
        self.cb_denoise.addItem("Aggressive (2)", 2)
        self.cb_denoise.addItem("Ultra Clean (3)", 3)
        expert_layout.addWidget(self.cb_denoise)

        self.chk_tta = QCheckBox("Enable 8-pass TTA (slower, best edges)")
        self.chk_tta.setToolTip(
            "Test-Time Augmentation: 8 rotations/flips ensemble for maximum edge quality."
        )
        expert_layout.addWidget(self.chk_tta)

        # Scene release naming
        lbl_release = QLabel("Scene Release Tags")
        lbl_release.setStyleSheet(
            "color: #64748b; font-size: 11px; font-weight: 600; margin-top: 6px;"
        )
        expert_layout.addWidget(lbl_release)

        show_row = QHBoxLayout()
        show_row.setSpacing(6)
        lbl_show = QLabel("Show:")
        lbl_show.setFixedWidth(38)
        show_row.addWidget(lbl_show)
        self.txt_show_name = QLineEdit("Happy.Tree.Friends")
        self.txt_show_name.setToolTip("Show title with dots for standard Scene naming.")
        self.txt_show_name.textChanged.connect(self.on_settings_changed)
        show_row.addWidget(self.txt_show_name, 1)
        expert_layout.addLayout(show_row)

        grp_row = QHBoxLayout()
        grp_row.setSpacing(6)
        lbl_grp = QLabel("Group:")
        lbl_grp.setFixedWidth(38)
        grp_row.addWidget(lbl_grp)
        self.txt_group = QLineEdit("CYLAE")
        self.txt_group.setToolTip("Scene release group tag (e.g. CYLAE).")
        self.txt_group.textChanged.connect(self.on_settings_changed)
        grp_row.addWidget(self.txt_group)
        expert_layout.addLayout(grp_row)

        src_tag_row = QHBoxLayout()
        src_tag_row.setSpacing(6)
        self.cb_source = QComboBox()
        self.cb_source.setToolTip("Source media type tag (e.g. WEB-DL, WEBRip).")
        self.cb_source.addItems(["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"])
        self.cb_source.currentIndexChanged.connect(self.on_settings_changed)
        src_tag_row.addWidget(self.cb_source)
        self.cb_audio_tag = QComboBox()
        self.cb_audio_tag.setToolTip("Standard audio codec release tag (e.g. AAC2.0, FLAC2.0).")
        self.cb_audio_tag.addItems(["AAC2.0", "FLAC2.0", "DDP5.1"])
        self.cb_audio_tag.currentIndexChanged.connect(self.on_settings_changed)
        src_tag_row.addWidget(self.cb_audio_tag)
        expert_layout.addLayout(src_tag_row)

        # Live console
        lbl_con_title = QLabel("🖥 Live CLI Log")
        lbl_con_title.setStyleSheet(
            "font-weight: 700; color: #4f8fff; font-size: 11px; margin-top: 6px;"
        )
        expert_layout.addWidget(lbl_con_title)

        con_btn_row = QHBoxLayout()
        con_btn_row.setSpacing(6)
        self.btn_copy_cli = QPushButton("📋 Copy CMD")
        self.btn_copy_cli.setToolTip(
            "Copy the most recently executed FFmpeg or AI upscaler CLI command to clipboard."
        )
        self.btn_copy_cli.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.btn_copy_cli.clicked.connect(self.on_copy_cli_clicked)
        con_btn_row.addWidget(self.btn_copy_cli)
        self.btn_clear_console = QPushButton("🧹 Clear")
        self.btn_clear_console.setToolTip("Clear all engine logs.")
        self.btn_clear_console.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.btn_clear_console.clicked.connect(self.clear_all_logs)
        con_btn_row.addWidget(self.btn_clear_console)
        con_btn_row.addStretch()
        expert_layout.addLayout(con_btn_row)

        self.txt_console = QTextEdit()
        self.txt_console.setObjectName("txtConsole")
        self.txt_console.setReadOnly(True)
        self.txt_console.setFixedHeight(90)
        self.txt_console.setPlaceholderText("Engine commands will appear here...")
        expert_layout.addWidget(self.txt_console)

        self.grp_expert.setVisible(False)
        s_layout.addWidget(self.grp_expert)

        s_layout.addStretch()

        sidebar_inner.setLayout(s_layout)
        sidebar_scroll.setWidget(sidebar_inner)
        sidebar_outer.addWidget(sidebar_scroll)
        middle.addWidget(sidebar)

        # ── 2b. RIGHT AREA: queue ─────────────────────────────────────────
        right_area = QWidget()
        right_area.setMinimumWidth(550)
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(12, 10, 14, 10)
        right_layout.setSpacing(8)

        # Stat / filter pill bar
        stat_bar = QHBoxLayout()
        stat_bar.setSpacing(6)

        self.btn_stat_all = QPushButton("All: 0")
        self.btn_stat_all.setProperty("class", "statPill")
        self.btn_stat_all.setCheckable(True)
        self.btn_stat_all.setChecked(True)
        self.btn_stat_all.setToolTip("Show all indexed episodes.")
        self.btn_stat_all.clicked.connect(lambda: self.set_filter_mode("all"))
        stat_bar.addWidget(self.btn_stat_all)

        self.btn_stat_ready = QPushButton("Ready: 0")
        self.btn_stat_ready.setProperty("class", "statPill")
        self.btn_stat_ready.setCheckable(True)
        self.btn_stat_ready.setToolTip("Filter only episodes pending processing.")
        self.btn_stat_ready.clicked.connect(lambda: self.set_filter_mode("ready"))
        stat_bar.addWidget(self.btn_stat_ready)

        self.btn_stat_completed = QPushButton("Done: 0")
        self.btn_stat_completed.setProperty("class", "statPill")
        self.btn_stat_completed.setCheckable(True)
        self.btn_stat_completed.setToolTip("Filter successfully upscaled episodes.")
        self.btn_stat_completed.clicked.connect(lambda: self.set_filter_mode("completed"))
        stat_bar.addWidget(self.btn_stat_completed)

        self.btn_stat_skipped = QPushButton("Skipped: 0")
        self.btn_stat_skipped.setProperty("class", "statPill")
        self.btn_stat_skipped.setCheckable(True)
        self.btn_stat_skipped.setToolTip("Filter already converted / skipped episodes.")
        self.btn_stat_skipped.clicked.connect(lambda: self.set_filter_mode("skipped"))
        stat_bar.addWidget(self.btn_stat_skipped)

        self.btn_stat_failed = QPushButton("Errors: 0")
        self.btn_stat_failed.setProperty("class", "statPill")
        self.btn_stat_failed.setCheckable(True)
        self.btn_stat_failed.setToolTip("Filter episodes that encountered an error during processing.")
        self.btn_stat_failed.clicked.connect(lambda: self.set_filter_mode("failed"))
        stat_bar.addWidget(self.btn_stat_failed)

        stat_bar.addStretch()

        self.lbl_search = QLabel("🔍")
        stat_bar.addWidget(self.lbl_search)
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter by name or season...")
        self.txt_filter.setToolTip("Filter queue by episode name, season, or filename.")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setMaximumWidth(240)
        self.txt_filter.textChanged.connect(self.on_filter_changed)
        stat_bar.addWidget(self.txt_filter)

        self.btn_clear_queue = QPushButton("🧹 Clear")
        self.btn_clear_queue.setToolTip("Clear all items currently in the processing queue.")
        self.btn_clear_queue.clicked.connect(self.on_clear_queue_clicked)
        stat_bar.addWidget(self.btn_clear_queue)

        right_layout.addLayout(stat_bar)

        # Queue table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#", "Original File", "Size", "Target Scene Name", "Status", "Detail / Progress"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 48)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 85)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 115)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(5, 260)
        header.setMinimumSectionSize(45)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Align specific headers
        for col_idx, align in (
            (0, Qt.AlignmentFlag.AlignCenter),
            (2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            (4, Qt.AlignmentFlag.AlignCenter),
        ):
            hdr_item = self.table.horizontalHeaderItem(col_idx)
            if hdr_item:
                hdr_item.setTextAlignment(align)

        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_table_context_menu)
        self.table.itemDoubleClicked.connect(self.on_table_double_click)
        right_layout.addWidget(self.table, 1)

        middle.addWidget(right_area)
        middle.setStretchFactor(0, 0)
        middle.setStretchFactor(1, 1)
        middle.setSizes([380, 1060])
        root_layout.addWidget(middle, 1)

        # ── 3. BOTTOM BAR (full width) ──────────────────────────────────────
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        b_layout = QVBoxLayout(bottom_bar)
        b_layout.setContentsMargins(18, 10, 18, 12)
        b_layout.setSpacing(8)

        # Status row
        status_row = QHBoxLayout()
        self.lbl_overall_status = QLabel("Ready to process.")
        self.lbl_overall_status.setStyleSheet(
            "font-weight: 600; color: #e2e8f0; font-size: 13px;"
        )
        status_row.addWidget(self.lbl_overall_status)
        status_row.addStretch()

        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet(
            "color: #60a5fa; font-weight: 700; font-size: 12px; padding: 4px 12px;"
            "background-color: rgba(96, 165, 250, 0.10); border: 1px solid rgba(96, 165, 250, 0.20);"
            "border-radius: 10px;"
        )
        self.lbl_eta.setVisible(False)
        status_row.addWidget(self.lbl_eta)
        b_layout.addLayout(status_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        b_layout.addWidget(self.progress_bar)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_compare_frame = QPushButton("🔍 Compare Frame")
        self.btn_compare_frame.setObjectName("btnCompareFrame")
        self.btn_compare_frame.setToolTip(
            "Generate a side-by-side Before/After on a single frame — no full video render needed."
        )
        self.btn_compare_frame.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_compare_frame.clicked.connect(self.on_compare_frame_clicked)
        self.btn_compare_frame.setMinimumHeight(38)
        btn_row.addWidget(self.btn_compare_frame)

        clip_box = QHBoxLayout()
        clip_box.setSpacing(4)
        lbl_clip = QLabel("Clip:")
        lbl_clip.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        clip_box.addWidget(lbl_clip)
        self.spin_preview_dur = QSpinBox()
        self.spin_preview_dur.setRange(5, 300)
        self.spin_preview_dur.setValue(30)
        self.spin_preview_dur.setSuffix(" s")
        self.spin_preview_dur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spin_preview_dur.setToolTip(
            "Duration of the upscaled preview clip in seconds. The clip starts 10% into the video."
        )
        self.spin_preview_dur.setFixedWidth(82)
        self.spin_preview_dur.setMinimumHeight(38)
        clip_box.addWidget(self.spin_preview_dur)
        btn_row.addLayout(clip_box)

        self.btn_preview_clip = QPushButton("🎬 Preview Clip")
        self.btn_preview_clip.setObjectName("btnPreviewClip")
        self.btn_preview_clip.setToolTip(
            "Upscale a short clip (default 30 s) from 10% into the video. "
            "Saved with a [Preview] suffix — never overwrites the full encode."
        )
        self.btn_preview_clip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_preview_clip.clicked.connect(self.on_preview_clip_clicked)
        self.btn_preview_clip.setMinimumHeight(38)
        btn_row.addWidget(self.btn_preview_clip)

        self.btn_preview = QPushButton("⚡ Test 1 Episode")
        self.btn_preview.setObjectName("btnPreview")
        self.btn_preview.setToolTip(
            "Process 1 full episode to verify visual quality, sync, and encoding speed before full batch."
        )
        self.btn_preview.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_preview.clicked.connect(self.on_test_single_clicked)
        self.btn_preview.setMinimumHeight(38)
        btn_row.addWidget(self.btn_preview)

        # Subtle vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #252535; max-height: 22px;")
        btn_row.addWidget(sep)

        self.btn_preflight = QPushButton("📋 Preflight")
        self.btn_preflight.setObjectName("btnPreflight")
        self.btn_preflight.setToolTip(
            "View preflight verification, disk space requirements, warnings, and dry-run mappings."
        )
        self.btn_preflight.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_preflight.clicked.connect(self.on_preflight_clicked)
        self.btn_preflight.setMinimumHeight(38)
        btn_row.addWidget(self.btn_preflight)

        self.btn_retry_failed = QPushButton("🔄 Retry Failed")
        self.btn_retry_failed.setObjectName("btnRetryFailed")
        self.btn_retry_failed.setToolTip(
            "Reset failed or cancelled episodes back to Pending for re-processing."
        )
        self.btn_retry_failed.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_retry_failed.clicked.connect(self.on_retry_failed_clicked)
        self.btn_retry_failed.setMinimumHeight(38)
        btn_row.addWidget(self.btn_retry_failed)

        btn_row.addStretch(1)

        self.btn_start = QPushButton("▶  START BATCH PROCESSING")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setToolTip("Launch concurrent batch processing for all pending episodes.")
        self.btn_start.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_start.setMinimumHeight(38)
        self.btn_start.setMinimumWidth(230)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setToolTip("Gracefully stop active processing jobs.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setMinimumWidth(90)
        btn_row.addWidget(self.btn_stop)

        b_layout.addLayout(btn_row)
        root_layout.addWidget(bottom_bar)

        # Finalise sidebar config
        self.on_preset_changed()

    def on_model_changed(self):
        m_id = self.cb_model.currentData() or "auto"
        spec = AVAILABLE_MODELS.get(m_id)
        if spec:
            self.lbl_model_info.setText(f"ℹ️ {spec.name}: {spec.description} ({spec.recommended_for})")
        if m_id == "realcugan-pro":
            self.lbl_model_info.setText(
                "ℹ️ RealCUGAN Pro: the installed bundle supports ×2/×3 and denoise levels "
                "-1, 0, and 3. Unavailable levels are adjusted automatically."
            )
            if hasattr(self, "cb_denoise"):
                self.cb_denoise.setToolTip(
                    "RealCUGAN Pro installed: levels -1, 0 and 3. Levels 1 and 2 are "
                    "automatically rounded to the nearest available weight."
                )
        self.update_scene_preview()

    def resizeEvent(self, event):
        # Sidebar layout handles space naturally — no chip hiding, no margin gymnastics.
        # Compact label shortening only applies to the bottom action bar at very small widths.
        width = event.size().width()
        if hasattr(self, "btn_start"):
            compact = width < 1100
            if hasattr(self, "btn_compare_frame"):
                self.btn_compare_frame.setText("🔍 Frame" if compact else "🔍 Compare Frame")
            if hasattr(self, "btn_preview_clip"):
                self.btn_preview_clip.setText("🎬 Clip" if compact else "🎬 Preview Clip")
            if hasattr(self, "btn_preview"):
                self.btn_preview.setText("⚡ Test" if compact else "⚡ Test 1 Episode")
            if hasattr(self, "btn_preflight"):
                self.btn_preflight.setText("📋" if compact else "📋 Preflight")
            if hasattr(self, "btn_retry_failed"):
                self.btn_retry_failed.setText("🔄" if compact else "🔄 Retry Failed")
            self.btn_start.setText("▶ START" if compact else "▶  START BATCH PROCESSING")
            self.btn_stop.setText("⏹" if compact else "⏹ STOP")
        super().resizeEvent(event)

    def _update_auto_model_hint(self, queue: List[QueueItem]):
        if (self.cb_model.currentData() or "auto") != "auto" or not queue:
            return
        target_preset = ("4k", "1440p", "1080p", "720p")[self.cb_resolution.currentIndex()]
        self._hint_worker = ModelHintWorker(queue[0].source_path, target_preset)
        self._hint_worker.hint_ready.connect(self._on_hint_ready)
        self._hint_worker.start()

    def _on_hint_ready(self, text: str, reason: str):
        self.lbl_model_info.setText(text)
        if reason:
            self.lbl_model_info.setToolTip(reason)

    def toggle_expert_panel(self):
        vis = not self.grp_expert.isVisible()
        self.grp_expert.setVisible(vis)
        if vis:
            self.btn_toggle_expert.setText("⚙️  Hide Advanced & Expert Controls ▲")
        else:
            self.btn_toggle_expert.setText("⚙️  Show Advanced & Expert Controls ▼")

    def on_preset_changed(self):
        p_id = self.cb_preset.currentData()
        if not p_id:
            return
        preset = get_preset(p_id)
        if p_id != "custom":
            self._updating_preset = True
            try:
                for i in range(self.cb_resolution.count()):
                    if preset.target_preset in self.cb_resolution.itemText(i).lower():
                        self.cb_resolution.setCurrentIndex(i)
                        break

                for i in range(self.cb_codec.count()):
                    txt = self.cb_codec.itemText(i).lower()
                    if "hevc" in preset.video_codec and "h.265 (hevc" in txt:
                        self.cb_codec.setCurrentIndex(i)
                        break
                    elif "h264" in preset.video_codec and "h.264" in txt:
                        self.cb_codec.setCurrentIndex(i)
                        break
                    elif "x265" in preset.video_codec and "cpu" in txt:
                        self.cb_codec.setCurrentIndex(i)
                        break

                for i in range(self.cb_model.count()):
                    if self.cb_model.itemData(i) == preset.model_id:
                        self.cb_model.setCurrentIndex(i)
                        break

                self.slider_cq.setValue(preset.cq_level)
                self.lbl_cq_val.setText(f"CQ {preset.cq_level}")

                for i in range(self.cb_nvenc_preset.count()):
                    if self.cb_nvenc_preset.itemText(i).startswith(preset.nvenc_preset):
                        self.cb_nvenc_preset.setCurrentIndex(i)
                        break

                for i in range(self.cb_audio_mode.count()):
                    if self.cb_audio_mode.itemData(i) == preset.audio_mode:
                        self.cb_audio_mode.setCurrentIndex(i)
                        break

                self.chk_tta.setChecked(preset.enable_tta)

                for i in range(self.cb_denoise.count()):
                    if self.cb_denoise.itemData(i) == preset.denoise_level:
                        self.cb_denoise.setCurrentIndex(i)
                        break

                w_idx = 0 if preset.workers == 2 else (1 if preset.workers == 1 else 2)
                self.cb_workers.setCurrentIndex(w_idx)

                f_idx = 0 if preset.frame_format == "jpg" else 1
                self.cb_frame_fmt.setCurrentIndex(f_idx)

            finally:
                self._updating_preset = False

        self.lbl_preset_desc.setText(f"ℹ️ {preset.description} ({preset.recommended_for})")
        self.update_scene_preview()

    def on_cq_slider_changed(self, val: int):
        self.lbl_cq_val.setText(f"CQ {val}")
        if not getattr(self, '_updating_preset', False):
            idx = self.cb_preset.findData("custom")
            if idx >= 0 and self.cb_preset.currentIndex() != idx:
                self.cb_preset.setCurrentIndex(idx)
        self.update_scene_preview()

    def on_expert_setting_changed(self):
        if not getattr(self, '_updating_preset', False):
            idx = self.cb_preset.findData("custom")
            if idx >= 0 and self.cb_preset.currentIndex() != idx:
                self.cb_preset.setCurrentIndex(idx)
        self.update_scene_preview()

    def update_scene_preview(self):
        if hasattr(self, 'lbl_preview_name') and hasattr(self, 'cb_resolution'):
            cfg = self.get_current_scene_config()
            sample_name = format_output_name("Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.avi", config=cfg)
            self.lbl_preview_name.setText(f"Sample: {sample_name}")

        if hasattr(self, 'lbl_audio') and hasattr(self, 'cb_audio_mode'):
            mode = self.cb_audio_mode.currentData() or "copy"
            labels = {
                "copy": "🔊 Audio: Direct Copy",
                "aac_192k": "🔊 Audio: AAC 192k",
                "aac_320k": "🔊 Audio: AAC 320k",
                "flac": "🔊 Audio: FLAC 24-bit",
            }
            tooltips = {
                "copy": "Lossless Stream Copy: Original soundtrack is preserved 1:1 without re-encoding.",
                "aac_192k": "Re-encoded to high quality AAC 192 kbps stereo.",
                "aac_320k": "Re-encoded to transparent AAC 320 kbps stereo.",
                "flac": "Lossless 24-bit FLAC audio re-encode.",
            }
            self.lbl_audio.setText(labels.get(mode, "🔊 Audio: Direct Copy"))
            self.lbl_audio.setToolTip(tooltips.get(mode, "Audio stream settings"))

    def on_log_received(self, msg: str):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        timestamped = f"[{timestamp}]  {msg}"
        self._last_cli_cmd = msg
        self._log_buffer.append(timestamped)
        if len(self._log_buffer) > 5000:
            del self._log_buffer[:-5000]

        count = len(self._log_buffer)
        self.btn_open_logs.setText(f"📋 Logs ({count})")
        if count > 0:
            self.btn_open_logs.setStyleSheet(
                "padding: 4px 12px; font-size: 11px; font-weight: 600;"
                "background-color: rgba(10, 132, 255, 0.15); border: 1px solid #0a84ff; border-radius: 12px; color: #0a84ff;"
            )

        if self._log_viewer and self._log_viewer.isVisible():
            self._log_viewer.append_line(timestamped)

        if hasattr(self, 'txt_console'):
            self.txt_console.append(msg)
            doc = self.txt_console.document()
            if doc.blockCount() > 250:
                cursor = self.txt_console.textCursor()
                cursor.movePosition(cursor.MoveOperation.Start)
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()

    def clear_all_logs(self):
        self._log_buffer.clear()
        if hasattr(self, 'txt_console'):
            self.txt_console.clear()
        self.btn_open_logs.setText("📋 Logs (0)")
        self.btn_open_logs.setStyleSheet(
            "padding: 4px 12px; font-size: 11px; font-weight: 600;"
            "background-color: #2c2c2e; border: 1px solid #3a3a3c; border-radius: 12px; color: #aeaeb2;"
        )
        if self._log_viewer and self._log_viewer.isVisible():
            self._log_viewer.txt_log.clear()
            self._log_viewer._update_count()

    def _open_log_viewer(self):
        if self._log_viewer is None or not self._log_viewer.isVisible():
            self._log_viewer = LogViewerDialog(self._log_buffer, parent=self)
            self._log_viewer.show()
        else:
            self._log_viewer.raise_()
            self._log_viewer.activateWindow()

    def on_copy_cli_clicked(self):
        if hasattr(self, '_last_cli_cmd') and self._last_cli_cmd:
            QApplication.clipboard().setText(self._last_cli_cmd)
            self.btn_copy_cli.setText("✔ Copied!")
            QTimer.singleShot(1500, lambda: self.btn_copy_cli.setText("📋 Copy Last CLI Command"))

    def show_doctor_dialog(self):
        dlg = DoctorDialog(self)
        dlg.exec()

    def set_filter_mode(self, mode: str):
        self._current_filter_mode = mode
        self.btn_stat_all.setChecked(mode == "all")
        self.btn_stat_ready.setChecked(mode == "ready")
        self.btn_stat_completed.setChecked(mode == "completed")
        self.btn_stat_skipped.setChecked(mode == "skipped")
        if hasattr(self, 'btn_stat_failed'):
            self.btn_stat_failed.setChecked(mode == "failed")
        self.apply_table_filters()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        file_paths = []
        folder_paths = []
        for u in urls:
            p = u.toLocalFile()
            if os.path.isfile(p):
                file_paths.append(p)
            elif os.path.isdir(p):
                folder_paths.append(p)

        cfg = self.get_current_scene_config()
        out_dir = self.txt_output.text().strip()

        if file_paths:
            first_dir = os.path.dirname(file_paths[0])
            if not self.txt_source.text().strip():
                self.txt_source.setText(first_dir)
            if not out_dir:
                tag = self._get_output_tag_for_current_resolution()
                out_dir = os.path.join(first_dir, f"({tag})")
                self.txt_output.setText(out_dir)
            self.batch_manager.add_files(file_paths, out_dir, cfg)
            self._populate_table(self.batch_manager.queue)
            self._update_auto_model_hint(self.batch_manager.queue)
            self.settings.setValue("last_source", first_dir)
            self.settings.setValue("last_output", out_dir)
        elif folder_paths:
            p = folder_paths[0]
            self.txt_source.setText(p)
            if not out_dir:
                tag = self._get_output_tag_for_current_resolution()
                out_dir = f"{p} ({tag})"
                self.txt_output.setText(out_dir)
            self.settings.setValue("last_source", p)
            self.settings.setValue("last_output", out_dir)
            self.on_scan_clicked()

    def on_browse_source(self):
        default_dir = self.txt_source.text() or str(self.settings.value("last_source", "")) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Source Directory", default_dir)
        if folder:
            self.txt_source.setText(folder)
            tag = self._get_output_tag_for_current_resolution()
            out_folder = f"{folder} ({tag})"
            self.txt_output.setText(out_folder)
            self.settings.setValue("last_source", folder)
            self.settings.setValue("last_output", out_folder)
            self.on_scan_clicked()

    def on_add_files_clicked(self):
        default_dir = self.txt_source.text() or str(self.settings.value("last_source", "")) or os.path.expanduser("~")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files to Upscale",
            default_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.m4v);;All Files (*.*)"
        )
        if files:
            first_dir = os.path.dirname(files[0])
            if not self.txt_source.text().strip():
                self.txt_source.setText(first_dir)
            out_dir = self.txt_output.text().strip()
            if not out_dir:
                tag = self._get_output_tag_for_current_resolution()
                out_dir = os.path.join(first_dir, f"({tag})")
                self.txt_output.setText(out_dir)
            cfg = self.get_current_scene_config()
            self.batch_manager.add_files(files, out_dir, cfg)
            self._populate_table(self.batch_manager.queue)
            self._update_auto_model_hint(self.batch_manager.queue)
            self.settings.setValue("last_source", first_dir)
            self.settings.setValue("last_output", out_dir)

    def on_clear_queue_clicked(self):
        self.batch_manager.clear_queue()
        self._populate_table(self.batch_manager.queue)
        self.lbl_overall_status.setText("Queue cleared.")

    def on_remove_item_clicked(self, target_idx: int):
        self.batch_manager.remove_items_by_indices([target_idx])
        self._populate_table(self.batch_manager.queue)
        self.lbl_overall_status.setText(f"Episode #{target_idx} removed from queue.")

    def on_browse_output(self):
        default_dir = self.txt_output.text() or str(self.settings.value("last_output", "")) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", default_dir)
        if folder:
            self.txt_output.setText(folder)
            self.settings.setValue("last_output", folder)
            self.on_settings_changed()

    def on_open_output(self):
        out_dir = self.txt_output.text().strip()
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            os.makedirs(out_dir, exist_ok=True)
            os.startfile(out_dir)

    def on_compare_frame_clicked(self, target_idx: Optional[int] = None):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue. Please add files or scan a folder first.")
            return

        target_item = None
        if target_idx is not None:
            for it in self.batch_manager.queue:
                if it.index == target_idx:
                    target_item = it
                    break
        else:
            selected_rows = self.table.selectedIndexes()
            if selected_rows:
                r = selected_rows[0].row()
                target_item = self._get_queue_item_from_row(r)

        if target_item is None:
            for it in self.batch_manager.queue:
                if it.status == EpisodeStatus.PENDING:
                    target_item = it
                    break
        if target_item is None and self.batch_manager.queue:
            target_item = self.batch_manager.queue[0]

        if target_item is None:
            self.lbl_overall_status.setText("No episodes available for comparison.")
            return

        res_idx = self.cb_resolution.currentIndex()
        target_preset = ("4k", "1440p", "1080p", "720p")[res_idx]
        model_id = self.cb_model.currentData() or "auto"

        if hasattr(self, "btn_compare_frame"):
            self.btn_compare_frame.setEnabled(False)
        self.lbl_overall_status.setText(f"🔍 Generating Before / After comparison frame for {os.path.basename(target_item.source_path)}...")

        pipeline = VideoUpscalerPipeline(self.hardware)
        self._frame_worker = FramePreviewWorker(
            pipeline=pipeline,
            video_path=target_item.source_path,
            model_id=model_id,
            target_preset=target_preset
        )
        self._frame_worker.preview_ready.connect(self._on_frame_preview_ready)
        self._frame_worker.start()

    def _on_frame_preview_ready(self, ok: bool, orig_path: str, upscaled_path: str, info: dict):
        if hasattr(self, "btn_compare_frame"):
            self.btn_compare_frame.setEnabled(True)
        if not ok:
            err = info.get("error", "Frame preview failed")
            self.lbl_overall_status.setText(f"Preview error: {err}")
            QMessageBox.warning(self, "Preview Error", f"Could not generate visual comparison:\n{err}")
            return

        self.lbl_overall_status.setText("Visual Before / After comparison ready.")
        dlg = FrameComparisonDialog(orig_path, upscaled_path, info, parent=self)
        dlg.exec()

    def on_table_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if not item:
            return
        row = item.row()
        q_item = self._get_queue_item_from_row(row)
        if not q_item:
            return

        menu = QMenu(self)

        act_compare = QAction("🔍 Compare Before / After (Single Frame)", self)
        act_compare.triggered.connect(lambda: self.on_compare_frame_clicked(target_idx=q_item.index))
        menu.addAction(act_compare)

        act_process_single = QAction("⚡ Process Only This Episode (Preview)", self)
        act_process_single.triggered.connect(lambda: self.start_processing(target_indices=[q_item.index]))
        menu.addAction(act_process_single)

        menu.addSeparator()

        act_remove = QAction("❌ Remove from Queue", self)
        act_remove.triggered.connect(lambda: self.on_remove_item_clicked(q_item.index))
        menu.addAction(act_remove)

        menu.addSeparator()

        act_play_src = QAction("🎬 Play Source Video", self)
        act_play_src.triggered.connect(lambda: self.play_file(q_item.source_path))
        menu.addAction(act_play_src)

        if os.path.isfile(q_item.target_full_path):
            act_play_out = QAction("▶️ Play Converted Video", self)
            act_play_out.triggered.connect(lambda: self.play_file(q_item.target_full_path))
            menu.addAction(act_play_out)

        menu.addSeparator()

        act_reveal_src = QAction("📂 Reveal in File Explorer", self)
        act_reveal_src.triggered.connect(lambda: self.reveal_file(q_item.source_path))
        menu.addAction(act_reveal_src)

        act_copy_name = QAction("📋 Copy Target File Name", self)
        act_copy_name.triggered.connect(lambda: QApplication.clipboard().setText(q_item.target_scene_name))
        menu.addAction(act_copy_name)

        act_copy_path = QAction("📋 Copy Full Path", self)
        act_copy_path.triggered.connect(lambda: QApplication.clipboard().setText(q_item.source_path))
        menu.addAction(act_copy_path)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def on_table_double_click(self, item: QTableWidgetItem):
        row = item.row()
        q_item = self._get_queue_item_from_row(row)
        if q_item:
            if os.path.isfile(q_item.target_full_path):
                self.play_file(q_item.target_full_path)
            else:
                self.reveal_file(q_item.source_path)

    def play_file(self, filepath: str):
        if os.path.exists(filepath):
            try:
                os.startfile(filepath)
            except Exception as e:
                self.lbl_overall_status.setText(f"Could not open file: {e}")

    def reveal_file(self, filepath: str):
        if os.path.exists(filepath):
            try:
                subprocess.Popen(f'explorer /select,"{os.path.normpath(filepath)}"')
            except Exception:
                os.startfile(os.path.dirname(filepath))

    def get_current_scene_config(self) -> SceneNamingConfig:
        res_idx = self.cb_resolution.currentIndex()
        res_tag = "2160p" if res_idx == 0 else ("1440p" if res_idx == 1 else ("1080p" if res_idx == 2 else "720p"))

        codec_idx = self.cb_codec.currentIndex()
        codec_tag = "H.265" if codec_idx in (0, 2) else "H.264"

        show_name = self.txt_show_name.text().strip() if hasattr(self, 'txt_show_name') and self.txt_show_name.text().strip() else "Happy.Tree.Friends"
        group = self.txt_group.text().strip() if hasattr(self, 'txt_group') and self.txt_group.text().strip() else "CYLAE"
        source = self.cb_source.currentText().strip() if hasattr(self, 'cb_source') else "WEB-DL"

        if hasattr(self, 'cb_audio_tag'):
            audio_tag = self.cb_audio_tag.currentText().strip() or "AAC2.0"
        else:
            audio_mode = self.cb_audio_mode.currentData() if hasattr(self, 'cb_audio_mode') else "copy"
            audio_tag = "FLAC2.0" if audio_mode == "flac" else "AAC2.0"

        naming_mode = "clean"
        if hasattr(self, 'cb_naming_mode'):
            naming_mode = self.cb_naming_mode.currentData() or "clean"

        return SceneNamingConfig(
            naming_mode=naming_mode,
            show_name=show_name,
            resolution_tag=res_tag,
            source_tag=source,
            audio_tag=audio_tag,
            codec_tag=codec_tag,
            group_tag=group,
            container=".mp4"
        )

    def _get_output_tag_for_current_resolution(self) -> str:
        idx = self.cb_resolution.currentIndex() if hasattr(self, 'cb_resolution') else 0
        return "2160p" if idx == 0 else ("1440p" if idx == 1 else ("1080p" if idx == 2 else "720p"))

    def on_resolution_changed(self):
        old_out = self.txt_output.text().strip()
        new_tag = self._get_output_tag_for_current_resolution()
        for tag in ("(2160p)", "(1440p)", "(1080p)", "(720p)"):
            if old_out.endswith(tag):
                updated_out = old_out[:-len(tag)] + f"({new_tag})"
                self.txt_output.setText(updated_out)
                self.settings.setValue("last_output", updated_out)
                break
        self.on_settings_changed()

    def on_scan_clicked(self):
        src = self.txt_source.text().strip()
        out = self.txt_output.text().strip()
        if not src or not os.path.exists(src):
            self.lbl_overall_status.setText("Error: Source directory does not exist.")
            return

        if not out:
            tag = self._get_output_tag_for_current_resolution()
            out = f"{src} ({tag})"
            self.txt_output.setText(out)

        cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()

        self.lbl_overall_status.setText("Scanning folder in background...")
        if hasattr(self, 'btn_scan'):
            self.btn_scan.setEnabled(False)

        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.terminate()
            self._scan_worker.wait(500)

        self._scan_worker = ScanWorker(self.batch_manager, src, out, cfg, preserve_folders)
        self._scan_worker.scan_complete.connect(self._on_scan_complete)
        self._scan_worker.scan_error.connect(self._on_scan_error)
        self._scan_worker.start()

    def _on_scan_complete(self, queue: List[QueueItem]):
        if hasattr(self, 'btn_scan'):
            self.btn_scan.setEnabled(True)
        self._populate_table(queue)
        self._update_auto_model_hint(queue)

    def _on_scan_error(self, err_msg: str):
        if hasattr(self, 'btn_scan'):
            self.btn_scan.setEnabled(True)
        self.lbl_overall_status.setText(f"Scan failed: {err_msg}")

    def on_settings_changed(self):
        """Called when a setting (codec, resolution, audio, etc.) changes.

        Only refreshes the scene name preview label — does NOT re-scan the source
        folder.  A re-scan wipes queue progress and must be initiated explicitly
        by the user via the Scan Folder button or a browse/drop action.
        """
        self.update_scene_preview()

    def _do_settings_changed(self):
        """Retained for backward-compatibility; delegates to update_scene_preview only."""
        self.update_scene_preview()

    def on_preflight_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("Queue is empty. Please scan a folder first.")
            return

        out_dir = self.txt_output.text().strip()
        res_idx = self.cb_resolution.currentIndex()
        target_preset = "4k" if res_idx == 0 else ("1440p" if res_idx == 1 else ("1080p" if res_idx == 2 else "720p"))
        codec_idx = self.cb_codec.currentIndex()
        video_codec = "hevc_nvenc" if codec_idx == 0 else ("h264_nvenc" if codec_idx == 1 else "libx265")
        model_id = self.cb_model.currentData() or "auto"
        cq_level = self.slider_cq.value()
        scene_cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()

        preflight_data = self.batch_manager.get_preflight_summary(
            output_dir=out_dir,
            target_preset=target_preset,
            video_codec=video_codec,
            model_id=model_id,
            cq_level=cq_level
        )
        dry_run_data = self.batch_manager.plan_dry_run(
            output_dir=out_dir,
            scene_config=scene_cfg,
            preserve_folders=preserve_folders
        )

        dlg = PreflightDialog(preflight_data, dry_run_data, parent=self)
        dlg.exec()

    def on_retry_failed_clicked(self):
        retried = self.batch_manager.retry_failed_items()
        if retried > 0:
            self._populate_table(self.batch_manager.queue)
            self.lbl_overall_status.setText(f"Retried {retried} failed/cancelled episode(s).")
        else:
            self.lbl_overall_status.setText("No failed or cancelled episodes to retry.")

    def _find_row_by_item_id(self, item_id: str) -> int:
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == item_id:
                return r
        return -1

    def _get_queue_item_from_row(self, row: int) -> Optional[QueueItem]:
        if 0 <= row < self.table.rowCount():
            it = self.table.item(row, 0)
            if it:
                item_id = it.data(Qt.ItemDataRole.UserRole)
                if item_id:
                    for q in self.batch_manager.queue:
                        if q.id == item_id:
                            return q
        return None

    def _populate_table(self, queue: List[QueueItem]):
        self.table.setRowCount(len(queue))
        skipped_count = 0
        completed_count = 0

        for r, item in enumerate(queue):
            it_idx = QTableWidgetItem(str(item.index))
            it_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_idx.setData(Qt.ItemDataRole.UserRole, item.id)
            it_idx.setForeground(QColor("#64748b"))
            self.table.setItem(r, 0, it_idx)

            it_src = QTableWidgetItem(item.relative_source_path)
            it_src.setToolTip(f"Source: {item.source_path}\nSize: {item.file_size_mb:.1f} MB")
            self.table.setItem(r, 1, it_src)

            it_sz = QTableWidgetItem(f"{item.file_size_mb:.1f} MB")
            it_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it_sz.setForeground(QColor("#94a3b8"))
            self.table.setItem(r, 2, it_sz)

            it_scene = QTableWidgetItem(item.target_scene_name)
            it_scene.setToolTip(f"Target: {item.target_full_path}")
            it_scene.setForeground(QColor("#38bdf8"))
            self.table.setItem(r, 3, it_scene)

            it_status = QTableWidgetItem(self._status_badge_text(item.status))
            it_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(it_status, item.status)
            self.table.setItem(r, 4, it_status)

            if item.status == EpisodeStatus.SKIPPED:
                skipped_count += 1
            elif item.status == EpisodeStatus.COMPLETED:
                completed_count += 1

            it_detail = QTableWidgetItem(item.status_detail or "Ready")
            it_detail.setForeground(QColor("#94a3b8"))
            self.table.setItem(r, 5, it_detail)

        total = len(queue)
        failed_count = sum(1 for it in queue if it.status == EpisodeStatus.FAILED)
        ready = total - skipped_count - completed_count - failed_count

        self.btn_stat_all.setText(f"All: {total}")
        self.btn_stat_ready.setText(f"Ready: {ready}")
        self.btn_stat_completed.setText(f"Done: {completed_count}")
        self.btn_stat_skipped.setText(f"Skipped: {skipped_count}")
        if hasattr(self, 'btn_stat_failed'):
            self.btn_stat_failed.setText(f"Errors: {failed_count}")
            if failed_count > 0:
                self.btn_stat_failed.setProperty("class", "statPillError")
            else:
                self.btn_stat_failed.setProperty("class", "statPill")
            self.btn_stat_failed.style().unpolish(self.btn_stat_failed)
            self.btn_stat_failed.style().polish(self.btn_stat_failed)

        self.lbl_overall_status.setText(f"Indexed {total} episodes ({ready} ready to process).")
        self.apply_table_filters()

    def _status_badge_text(self, status: EpisodeStatus) -> str:
        if status == EpisodeStatus.PENDING:
            return "● Pending"
        elif status == EpisodeStatus.PROCESSING:
            return "▶ Processing"
        elif status == EpisodeStatus.COMPLETED:
            return "✔ Completed"
        elif status == EpisodeStatus.SKIPPED:
            return "⏭ Skipped"
        elif status == EpisodeStatus.FAILED:
            return "✖ Error"
        elif status == EpisodeStatus.CANCELLED:
            return "⏹ Cancelled"
        return status.value

    def _style_status_item(self, item: QTableWidgetItem, status: EpisodeStatus):
        if status == EpisodeStatus.COMPLETED:
            item.setForeground(QColor("#10b981"))
        elif status == EpisodeStatus.PROCESSING:
            item.setForeground(QColor("#f59e0b"))
        elif status == EpisodeStatus.SKIPPED:
            item.setForeground(QColor("#64748b"))
        elif status == EpisodeStatus.FAILED:
            item.setForeground(QColor("#ef4444"))
        elif status == EpisodeStatus.CANCELLED:
            item.setForeground(QColor("#f97316"))
        else:
            item.setForeground(QColor("#cbd5e1"))

    def on_filter_changed(self, text: str):
        self.apply_table_filters()

    def apply_table_filters(self):
        search_text = self.txt_filter.text().lower().strip()
        mode = self._current_filter_mode

        for r in range(self.table.rowCount()):
            text_match = True
            if search_text:
                text_match = False
                for col in (1, 3):
                    item = self.table.item(r, col)
                    if item and search_text in item.text().lower():
                        text_match = True
                        break

            status_match = True
            q_item = self._get_queue_item_from_row(r)
            if q_item:
                q_status = q_item.status
                if mode == "ready":
                    status_match = (q_status == EpisodeStatus.PENDING)
                elif mode == "completed":
                    status_match = (q_status == EpisodeStatus.COMPLETED)
                elif mode == "skipped":
                    status_match = (q_status == EpisodeStatus.SKIPPED)
                elif mode == "failed":
                    status_match = (q_status == EpisodeStatus.FAILED)
            elif mode != "all":
                status_match = False

            self.table.setRowHidden(r, not (text_match and status_match))

    def on_start_clicked(self):
        self.start_processing()

    def on_test_single_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue. Please scan a folder first.")
            return

        target_idx = None
        for it in self.batch_manager.queue:
            if it.status == EpisodeStatus.PENDING:
                target_idx = it.index
                break
        if target_idx is None:
            target_idx = self.batch_manager.queue[0].index

        self.start_processing(target_indices=[target_idx])

    def on_preview_clip_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue. Please scan a folder first.")
            return

        target_idx = None
        for it in self.batch_manager.queue:
            if it.status == EpisodeStatus.PENDING:
                target_idx = it.index
                break
        if target_idx is None:
            target_idx = self.batch_manager.queue[0].index

        dur = float(self.spin_preview_dur.value())
        self.start_processing(target_indices=[target_idx], preview_duration_s=dur)

    def start_processing(self, target_indices: Optional[List[int]] = None, preview_duration_s: Optional[float] = None):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue.")
            return

        out_dir = self.txt_output.text().strip()
        base_dir = self.txt_source.text().strip()
        if not out_dir:
            self.lbl_overall_status.setText("Please specify an output folder.")
            return

        res_idx = self.cb_resolution.currentIndex()
        target_preset = "4k" if res_idx == 0 else ("1440p" if res_idx == 1 else ("1080p" if res_idx == 2 else "720p"))

        codec_idx = self.cb_codec.currentIndex()
        video_codec = "hevc_nvenc" if codec_idx == 0 else ("h264_nvenc" if codec_idx == 1 else "libx265")

        model_id = self.cb_model.currentData() or "auto"
        if model_id == "auto":
            try:
                first_item = next((it for it in self.batch_manager.queue if not target_indices or it.index in target_indices), None)
                if first_item:
                    meta = probe_video(first_item.source_path)
                    rec = recommend_model_for_video(meta.width, meta.height, target_preset)
                    self.on_log_received(f"[AI MODEL AUTO] {rec.model.name} ×{rec.scale} ({rec.reason})")
            except Exception:
                pass
        else:
            m_spec = AVAILABLE_MODELS.get(model_id)
            if m_spec:
                self.on_log_received(f"[AI MODEL] {m_spec.name} ({m_spec.description})")
        scene_cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()
        overwrite = self.chk_overwrite.isChecked()

        w_idx = self.cb_workers.currentIndex()
        max_workers = 2 if w_idx == 0 else (1 if w_idx == 1 else 3)
        if target_indices and len(target_indices) == 1:
            max_workers = 1

        t_idx = self.cb_threads.currentIndex()
        threads = "auto" if t_idx == 0 else ("4:4:4" if t_idx == 1 else ("2:4:2" if t_idx == 2 else "1:2:2"))

        f_idx = self.cb_frame_fmt.currentIndex()
        frame_format = "jpg" if f_idx == 0 else "png"

        cq_level = self.slider_cq.value()
        nvenc_preset = self.cb_nvenc_preset.currentText().split()[0]
        audio_mode = self.cb_audio_mode.currentData() or "copy"
        tile_size = int(self.cb_tile_size.currentData() or 0)
        denoise_level = int(self.cb_denoise.currentData() if self.cb_denoise.currentData() is not None else -1)
        enable_tta = self.chk_tta.isChecked()

        try:
            preflight = self.batch_manager.get_preflight_summary(
                output_dir=out_dir,
                target_preset=target_preset,
                video_codec=video_codec,
                model_id=model_id,
                cq_level=cq_level
            )
            for warn in preflight.get("warnings", []):
                self.on_log_received(f"[PREFLIGHT WARNING] ⚠️ {warn}")
        except Exception:
            pass

        self.btn_start.setEnabled(False)
        if hasattr(self, 'btn_preview'):
            self.btn_preview.setEnabled(False)
        if hasattr(self, 'btn_preview_clip'):
            self.btn_preview_clip.setEnabled(False)
        self.btn_stop.setEnabled(True)

        if preview_duration_s is not None:
            dur_label = int(preview_duration_s)
            self.lbl_overall_status.setText(
                f"🎬 Rendering {dur_label}s preview clip from episode #{target_indices[0] if target_indices else '?'}..."
            )
        elif target_indices and len(target_indices) == 1:
            self.lbl_overall_status.setText(f"⚡ Testing episode #{target_indices[0]} (Full)...")
        else:
            self.lbl_overall_status.setText(f"Starting batch processing ({max_workers} concurrent workers · CQ {cq_level} · Audio: {audio_mode.upper()})...")

        self.worker = UpscaleWorker(
            batch_manager=self.batch_manager,
            output_dir=out_dir,
            base_dir=base_dir,
            target_preset=target_preset,
            video_codec=video_codec,
            model_id=model_id,
            scene_config=scene_cfg,
            preserve_folders=preserve_folders,
            overwrite=overwrite,
            threads=threads,
            frame_format=frame_format,
            max_workers=max_workers,
            cq_level=cq_level,
            nvenc_preset=nvenc_preset,
            enable_tta=enable_tta,
            denoise_level=denoise_level,
            audio_mode=audio_mode,
            tile_size=tile_size,
            target_indices=target_indices,
            preview_duration_s=preview_duration_s,
        )
        self.worker.item_updated.connect(self.on_item_updated)
        self.worker.batch_updated.connect(self.on_batch_updated)
        self.worker.log_signal.connect(self.on_log_received)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_stop_clicked(self):
        self.lbl_overall_status.setText("Stopping active workers...")
        self.batch_manager.stop()
        self.btn_stop.setEnabled(False)

    def on_item_updated(self, item: QueueItem):
        row = self._find_row_by_item_id(item.id)
        if row == -1:
            row = item.index - 1
        if 0 <= row < self.table.rowCount():
            it_status = self.table.item(row, 4)
            if it_status:
                it_status.setText(self._status_badge_text(item.status))
                self._style_status_item(it_status, item.status)

            it_detail = self.table.item(row, 5)
            if it_detail:
                it_detail.setText(item.status_detail)

            if item.status == EpisodeStatus.PROCESSING and it_status:
                self.table.scrollToItem(it_status)

        completed_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.COMPLETED)
        skipped_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.SKIPPED)
        failed_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.FAILED)
        ready_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.PENDING)

        self.btn_stat_ready.setText(f"Ready: {ready_count}")
        self.btn_stat_completed.setText(f"Completed: {completed_count}")
        self.btn_stat_skipped.setText(f"Skipped: {skipped_count}")
        if hasattr(self, 'btn_stat_failed'):
            self.btn_stat_failed.setText(f"Errors: {failed_count}")
            if failed_count > 0:
                self.btn_stat_failed.setProperty("class", "statPillError")
            else:
                self.btn_stat_failed.setProperty("class", "statPill")
            self.btn_stat_failed.style().unpolish(self.btn_stat_failed)
            self.btn_stat_failed.style().polish(self.btn_stat_failed)

    def on_batch_updated(self, current: int, total: int, pct: float, status_str: str, eta_str: str = ""):
        self.progress_bar.setValue(int(pct))
        self.lbl_overall_status.setText(status_str)
        if eta_str:
            self.lbl_eta.setText(eta_str)
            self.lbl_eta.setVisible(True)
        else:
            self.lbl_eta.setVisible(False)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        if hasattr(self, 'btn_preview'):
            self.btn_preview.setEnabled(True)
        if hasattr(self, 'btn_preview_clip'):
            self.btn_preview_clip.setEnabled(True)
        self.btn_stop.setEnabled(False)
        failed_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.FAILED)
        cancelled_count = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.CANCELLED)
        if cancelled_count:
            self.lbl_overall_status.setText(f"Processing stopped ({cancelled_count} cancelled).")
        elif failed_count:
            self.lbl_overall_status.setText(f"Processing finished with {failed_count} error(s).")
        else:
            self.lbl_overall_status.setText("Processing finished successfully.")
        self.lbl_eta.setText("")
        self.lbl_eta.setVisible(False)
        if not cancelled_count:
            self.progress_bar.setValue(100)
        try:
            QApplication.beep()
        except Exception:
            pass

    def closeEvent(self, event):
        """Ensures all background workers and external processes terminate cleanly on exit."""
        if hasattr(self, "batch_manager") and self.batch_manager:
            self.batch_manager.stop()

        if hasattr(self, "_scan_worker") and self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.terminate()
            self._scan_worker.wait(500)

        if hasattr(self, "worker") and self.worker and self.worker.isRunning():
            self.worker.wait(1500)

        super().closeEvent(event)


def run_gui():
    app = QApplication(sys.argv)
    window = UpscalerMainWindow()
    window.show()
    return app.exec()
