"""Main application window for Video Enhancer."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QCursor, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import get_hardware_info
from src.core.models import (
    AVAILABLE_MODELS,
    get_model_display_name,
    recommend_model_for_video,
)
from src.core.pipeline import VideoUpscalerPipeline
from src.core.presets import ENHANCEMENT_PRESETS, RESOLUTION_LABELS, RESOLUTION_TAGS, get_preset
from src.core.probe import probe_video
from src.core.scene_namer import (
    SceneNamingConfig,
    format_output_name,
)
from src.core.storage import cleanup_stale_temp_directories
from src.ui.dialogs.comparison import FrameComparisonDialog
from src.ui.dialogs.doctor import DoctorDialog
from src.ui.dialogs.logs import LogViewerDialog
from src.ui.dialogs.preflight import PreflightDialog
from src.ui.theme import (
    DARK_STYLESHEET,
    UTILITY_STYLESHEET,
    apply_dark_titlebar,
    setup_combobox,
)
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
        self.setWindowTitle("AnimaScale - Video processing")
        self.resize(1400, 860)
        self.setMinimumSize(960, 660)
        self.setAcceptDrops(True)

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        icon_path = os.path.join(assets_dir, "app_icon.ico" if os.name == "nt" else "app_icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

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
        self._operation_active = False
        self._queue_items_by_id: Dict[str, QueueItem] = {}
        self._table_rows_by_id: Dict[str, int] = {}
        self._displayed_statuses: Dict[str, EpisodeStatus] = {}
        self._status_counts: Counter = Counter()
        self.settings = QSettings("CYLAE", "VideoEnhancer4K")

        self._initializing_ui = True
        self._init_ui()
        self._initializing_ui = False
        self._setup_all_comboboxes()

        app_inst = QApplication.instance()
        if app_inst:
            if app_inst.style().objectName() != "fusion":
                app_inst.setStyle("Fusion")
            if app_inst.styleSheet() != DARK_STYLESHEET:
                app_inst.setStyleSheet(DARK_STYLESHEET)
        self.setStyleSheet(UTILITY_STYLESHEET)

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
        """Keep paths, actions and parameter categories directly accessible."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        self.main_layout = root
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        status = QHBoxLayout()
        self.lbl_gpu = QLabel(f"GPU: {self.hardware.gpu_name} ({self.hardware.gpu_vram_gb:g} GB)")
        self.lbl_encoder = QLabel("NVENC available" if self.hardware.nvenc_available else "CPU encoding")
        self.lbl_audio = QLabel("Audio: Direct Copy")
        status.addWidget(self.lbl_gpu, 1)
        status.addWidget(self.lbl_encoder)
        status.addWidget(self.lbl_audio)
        self.btn_doctor = QPushButton("Diagnostics")
        self.btn_doctor.clicked.connect(self.show_doctor_dialog)
        status.addWidget(self.btn_doctor)
        self.btn_open_logs = QPushButton("Logs (0)")
        self.btn_open_logs.setToolTip("Open, filter, copy or export the engine logs. Ctrl+L")
        self.btn_open_logs.setShortcut("Ctrl+L")
        self.btn_open_logs.clicked.connect(self._open_log_viewer)
        status.addWidget(self.btn_open_logs)
        root.addLayout(status)

        paths = QGridLayout()
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Source folder or files")
        self.txt_source.setToolTip("Source directory. Use Add files for individual videos, or drop them onto the window.")
        self.txt_source.setClearButtonEnabled(True)
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Output folder")
        self.txt_output.setToolTip("Destination used for preflight planning and processing.")
        self.txt_output.setClearButtonEnabled(True)
        for row, title, field in ((0, "Source", self.txt_source), (1, "Output", self.txt_output)):
            label = QLabel(title)
            label.setBuddy(field)
            paths.addWidget(label, row, 0)
            paths.addWidget(field, row, 1)
        self.btn_add_files = QPushButton("Add files")
        self.btn_add_files.setShortcut("Ctrl+O")
        self.btn_add_files.clicked.connect(self.on_add_files_clicked)
        self.btn_browse_source = QPushButton("Source folder")
        self.btn_browse_source.clicked.connect(self.on_browse_source)
        self.btn_scan = QPushButton("Scan (F5)")
        self.btn_scan.setObjectName("btnScan")
        self.btn_scan.setToolTip("Index videos in the source folder. F5")
        self.btn_scan.setShortcut("F5")
        self.btn_scan.clicked.connect(self.on_scan_clicked)
        self.btn_browse_output = QPushButton("Choose output")
        self.btn_browse_output.clicked.connect(self.on_browse_output)
        self.btn_open_output = QPushButton("Open output")
        self.btn_open_output.clicked.connect(self.on_open_output)
        paths.addWidget(self.btn_add_files, 0, 2)
        paths.addWidget(self.btn_browse_source, 0, 3)
        paths.addWidget(self.btn_scan, 0, 4)
        paths.addWidget(self.btn_browse_output, 1, 2)
        paths.addWidget(self.btn_open_output, 1, 3, 1, 2)
        paths.setColumnStretch(1, 1)
        root.addLayout(paths)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        settings = QWidget()
        self.parameter_panel = settings
        settings.setMinimumWidth(350)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)
        preset_row = QFormLayout()
        self.cb_preset = QComboBox()
        for preset_id, preset in ENHANCEMENT_PRESETS.items():
            self.cb_preset.addItem(preset.name, preset_id)
        self.cb_preset.setToolTip("Apply a complete starting profile. Manual processing changes select Custom.")
        preset_row.addRow("Profile", self.cb_preset)
        settings_layout.addLayout(preset_row)
        self.lbl_preset_desc = QLabel()
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        settings_layout.addWidget(self.lbl_preset_desc)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setObjectName("settingsTabs")
        self.settings_tabs.setUsesScrollButtons(True)
        settings_layout.addWidget(self.settings_tabs, 1)

        def page(title):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            contents = QWidget()
            form = QFormLayout(contents)
            form.setContentsMargins(8, 8, 8, 8)
            form.setSpacing(6)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            scroll.setWidget(contents)
            self.settings_tabs.addTab(scroll, title)
            return form

        def combo(items):
            control = QComboBox()
            control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            control.setMinimumContentsLength(10)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            for text, data in items:
                control.addItem(text, data)
            return control

        video = page("Video")
        self.cb_resolution = combo([(label, value) for value, label in RESOLUTION_LABELS.items()])
        self.cb_resolution.setToolTip("Target resolution. Portrait orientation and display aspect ratio are preserved.")
        video.addRow("Resolution", self.cb_resolution)
        self.cb_codec = combo([
            ("H.265 (HEVC / NVENC)", "hevc_nvenc"), ("H.264 (NVENC)", "h264_nvenc"),
            ("H.265 (CPU / libx265)", "libx265"), ("H.264 (CPU / libx264)", "libx264"),
        ])
        self.cb_codec.setToolTip("Select the video encoder explicitly; CPU H.264 and H.265 are available.")
        video.addRow("Encoder", self.cb_codec)
        self.cb_model = combo([(spec.name, model_id) for model_id, spec in AVAILABLE_MODELS.items()])
        self.cb_model.setToolTip("Model for the batch. The queue context menu can override it per video.")
        video.addRow("AI model", self.cb_model)
        quality = QWidget()
        quality_row = QHBoxLayout(quality)
        quality_row.setContentsMargins(0, 0, 0, 0)
        self.slider_cq = QSlider(Qt.Orientation.Horizontal)
        self.slider_cq.setRange(14, 28)
        self.slider_cq.setValue(19)
        self.spin_cq = QSpinBox()
        self.spin_cq.setRange(14, 28)
        self.spin_cq.setValue(19)
        self.spin_cq.setToolTip("CQ for NVENC / CRF for CPU encoders. Lower values request higher quality.")
        self.slider_cq.setToolTip(self.spin_cq.toolTip())
        quality_row.addWidget(self.slider_cq, 1)
        quality_row.addWidget(self.spin_cq)
        self.lbl_cq_val = QLabel("CQ 19")
        self.lbl_cq_val.setVisible(False)
        video.addRow("CQ / CRF", quality)
        self.cb_nvenc_preset = combo([(f"p{number}" + (" - highest compression" if number == 7 else ""), f"p{number}") for number in range(7, 0, -1)])
        self.cb_nvenc_preset.setToolTip("All NVENC presets p1 through p7. p7 spends more work on compression.")
        video.addRow("NVENC preset", self.cb_nvenc_preset)
        self.cb_denoise = combo([(f"{level}", level) for level in (-1, 0, 1, 2, 3)])
        self.cb_denoise.setToolTip("RealCUGAN denoise level. -1 disables denoising; available weights depend on the installed model.")
        video.addRow("RealCUGAN denoise", self.cb_denoise)
        self.chk_tta = QCheckBox("8-pass TTA")
        self.chk_tta.setToolTip("Run the model's test-time augmentation ensemble. Also applied to frame previews.")
        video.addRow("Augmentation", self.chk_tta)
        self.cb_frame_fmt = combo([("JPEG (lossy)", "jpg"), ("Lossless PNG", "png")])
        self.cb_frame_fmt.setToolTip("PNG preserves decoded pixels. JPEG uses less temporary storage but is lossy.")
        video.addRow("Intermediate frames", self.cb_frame_fmt)
        self.lbl_model_info = QLabel()
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        video.addRow(self.lbl_model_info)

        audio = page("Audio")
        self.cb_audio_mode = combo([
            ("Copy original audio", "copy"), ("AAC 192 kbps", "aac_192k"),
            ("AAC 320 kbps", "aac_320k"), ("FLAC (lossless re-encode)", "flac"),
        ])
        self.cb_audio_mode.setToolTip("Copy avoids audio re-encoding; AAC is lossy, FLAC is lossless.")
        audio.addRow("Soundtrack", self.cb_audio_mode)
        audio_note = QLabel("The audio release tag is a filename label; it does not choose the audio encoder. Set filename tags in Naming.")
        audio_note.setWordWrap(True)
        audio_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        audio.addRow(audio_note)

        engine = page("Engine")
        self.cb_workers = combo([("2 Workers", 2), ("1 Worker", 1), ("3 Workers", 3), ("4 Workers", 4)])
        self.cb_workers.setToolTip("Concurrent video jobs. More workers can compete for GPU memory; 8K profiles use one.")
        engine.addRow("Concurrent videos", self.cb_workers)
        self.cb_threads = combo([("Auto", "auto"), ("4:4:4", "4:4:4"), ("2:4:2", "2:4:2"), ("1:2:2", "1:2:2")])
        self.cb_threads.setEditable(True)
        self.cb_threads.setToolTip("Auto or a custom LOAD:PROCESS:SAVE thread count, e.g. 3:2:4.")
        engine.addRow("Vulkan threads", self.cb_threads)
        self.cb_tile_size = combo([("Auto", 0)] + [(str(value), value) for value in (64, 100, 128, 200, 300, 400, 600)])
        self.cb_tile_size.setEditable(True)
        self.cb_tile_size.setToolTip("Auto (0), or a custom tile edge of at least 32 pixels. Smaller tiles use less VRAM.")
        engine.addRow("Tile size (px)", self.cb_tile_size)
        self.chk_overwrite = QCheckBox("Reprocess existing outputs")
        self.chk_overwrite.setToolTip("When off, only outputs validated against the current target can be skipped.")
        engine.addRow("Overwrite", self.chk_overwrite)

        naming = page("Naming")
        self.cb_naming_mode = combo([("Clean", "clean"), ("Scene Release", "scene"), ("Original Name", "original")])
        self.cb_naming_mode.setToolTip("Choose how output filenames are generated.")
        naming.addRow("Filename mode", self.cb_naming_mode)
        self.cb_container = combo([("MP4", ".mp4"), ("Matroska (MKV)", ".mkv")])
        self.cb_container.setToolTip("Output container. Matroska supports a broader range of audio codecs.")
        naming.addRow("Container", self.cb_container)
        self.txt_custom_suffix = QLineEdit()
        self.txt_custom_suffix.setToolTip("Optional suffix used by the existing filename formatter.")
        naming.addRow("Custom suffix", self.txt_custom_suffix)
        self.txt_show_name = QLineEdit("Happy.Tree.Friends")
        self.txt_show_name.setToolTip("Show title used by Scene Release naming.")
        naming.addRow("Show", self.txt_show_name)
        self.txt_group = QLineEdit("CYLAE")
        self.txt_group.setToolTip("Release group used by Scene Release naming.")
        naming.addRow("Group", self.txt_group)
        self.cb_source = combo([(text, text) for text in ("WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER")])
        self.cb_source.setToolTip("Source tag used in Scene Release filenames.")
        naming.addRow("Source tag", self.cb_source)
        self.cb_audio_tag = combo([(text, text) for text in ("AAC2.0", "FLAC2.0", "DDP5.1")])
        self.cb_audio_tag.setToolTip("Filename label only; use Audio to choose actual encoding.")
        naming.addRow("Audio tag", self.cb_audio_tag)
        self.chk_preserve_folders = QCheckBox("Preserve source subfolders")
        self.chk_preserve_folders.setChecked(True)
        self.chk_preserve_folders.setToolTip("Keep the source directory structure below the destination.")
        naming.addRow("Folders", self.chk_preserve_folders)
        self.lbl_preview_name = QLabel()
        self.lbl_preview_name.setWordWrap(True)
        self.lbl_preview_name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_preview_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        naming.addRow("Example output", self.lbl_preview_name)
        self.main_splitter.addWidget(settings)

        queue_panel = QWidget()
        queue_panel.setMinimumWidth(430)
        queue_layout = QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        filters = QHBoxLayout()
        for suffix, label, mode in (("all", "All", "all"), ("ready", "Ready", "ready"),
                                    ("completed", "Completed", "completed"), ("skipped", "Skipped", "skipped"),
                                    ("failed", "Errors", "failed")):
            button = QPushButton(f"{label}: 0")
            button.setCheckable(True)
            button.setChecked(mode == "all")
            button.clicked.connect(lambda checked=False, mode=mode: self.set_filter_mode(mode))
            setattr(self, "btn_stat_" + suffix, button)
            filters.addWidget(button)
        queue_layout.addLayout(filters)
        search = QHBoxLayout()
        self.lbl_search = QLabel("Filter")
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filename, output name or model")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setToolTip("Filter the queue without modifying it.")
        self.txt_filter.textChanged.connect(self.on_filter_changed)
        search.addWidget(self.lbl_search)
        search.addWidget(self.txt_filter, 1)
        self.btn_remove_selected = QPushButton("Remove selected")
        self.btn_remove_selected.setToolTip("Remove selected queue entries; no media file is deleted.")
        self.btn_remove_selected.clicked.connect(self.on_remove_selected_clicked)
        search.addWidget(self.btn_remove_selected)
        self.btn_clear_queue = QPushButton("Clear queue")
        self.btn_clear_queue.setToolTip("Remove queue entries; source and output files are kept.")
        self.btn_clear_queue.clicked.connect(self.on_clear_queue_clicked)
        search.addWidget(self.btn_clear_queue)
        queue_layout.addLayout(search)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "Original File", "Size", "Output File", "Model", "Status", "Detail / Progress"])
        header = self.table.horizontalHeader()
        for column, width in ((0, 40), (2, 70), (4, 120), (5, 105)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        for column in (1, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(column, 220)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(6, 220)
        header.setMinimumSectionSize(45)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_table_context_menu)
        self.table.itemDoubleClicked.connect(self.on_table_double_click)
        queue_layout.addWidget(self.table, 1)
        logs = QHBoxLayout()
        logs.addWidget(QLabel("Live engine log"))
        self.btn_copy_cli = QPushButton("Copy last command")
        self.btn_copy_cli.setToolTip("Copy the last engine command to the clipboard.")
        self.btn_copy_cli.clicked.connect(self.on_copy_cli_clicked)
        self.btn_clear_console = QPushButton("Clear logs")
        self.btn_clear_console.clicked.connect(self.clear_all_logs)
        logs.addStretch()
        logs.addWidget(self.btn_copy_cli)
        logs.addWidget(self.btn_clear_console)
        queue_layout.addLayout(logs)
        self.txt_console = QTextEdit()
        self.txt_console.setObjectName("txtConsole")
        self.txt_console.setReadOnly(True)
        self.txt_console.document().setMaximumBlockCount(250)
        self.txt_console.setMaximumHeight(100)
        self.txt_console.setPlaceholderText("Commands and errors appear here; full logs are available above.")
        queue_layout.addWidget(self.txt_console)
        self.main_splitter.addWidget(queue_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([420, 960])
        root.addWidget(self.main_splitter, 1)

        self.lbl_overall_status = QLabel("Ready to process.")
        self.lbl_overall_status.setWordWrap(True)
        self.lbl_eta = QLabel()
        self.lbl_eta.setVisible(False)
        progress = QHBoxLayout()
        progress.addWidget(self.lbl_overall_status, 1)
        progress.addWidget(self.lbl_eta)
        root.addLayout(progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)

        previews = QHBoxLayout()
        self.btn_compare_frame = QPushButton("Compare Frame")
        self.btn_compare_frame.setObjectName("btnCompareFrame")
        self.btn_compare_frame.setToolTip(
            "PNG comparison using the selected video's model and current AI parameters. "
            "Use Preview Clip to also judge final encoding and audio."
        )
        self.btn_compare_frame.clicked.connect(lambda: self.on_compare_frame_clicked())
        previews.addWidget(self.btn_compare_frame)
        self.spin_preview_dur = QSpinBox()
        self.spin_preview_dur.setRange(5, 300)
        self.spin_preview_dur.setValue(30)
        self.spin_preview_dur.setSuffix(" s")
        self.spin_preview_dur.setMinimumWidth(80)
        self.spin_preview_dur.setToolTip("Short clip duration; starts 10% into the selected video.")
        previews.addWidget(self.spin_preview_dur)
        self.btn_preview_clip = QPushButton("Preview Clip")
        self.btn_preview_clip.setObjectName("btnPreviewClip")
        self.btn_preview_clip.setToolTip("Render a short clip from the selected video without replacing a full output.")
        self.btn_preview_clip.clicked.connect(self.on_preview_clip_clicked)
        previews.addWidget(self.btn_preview_clip)
        self.btn_preview = QPushButton("Test 1 Episode")
        self.btn_preview.setObjectName("btnPreview")
        self.btn_preview.setToolTip("Process the selected full video to check quality before starting the whole batch.")
        self.btn_preview.clicked.connect(self.on_test_single_clicked)
        previews.addWidget(self.btn_preview)
        previews.addStretch()
        root.addLayout(previews)
        actions = QHBoxLayout()
        self.btn_preflight = QPushButton("Preflight / Output plan (F6)")
        self.btn_preflight.setObjectName("btnPreflight")
        self.btn_preflight.setShortcut("F6")
        self.btn_preflight.setToolTip("Recalculate destinations and validate existing outputs using the current settings.")
        self.btn_preflight.clicked.connect(self.on_preflight_clicked)
        actions.addWidget(self.btn_preflight)
        self.btn_retry_failed = QPushButton("Retry failed")
        self.btn_retry_failed.setObjectName("btnRetryFailed")
        self.btn_retry_failed.clicked.connect(self.on_retry_failed_clicked)
        actions.addWidget(self.btn_retry_failed)
        actions.addStretch()
        self.btn_start = QPushButton("Start batch (Ctrl+Enter)")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setShortcut("Ctrl+Return")
        self.btn_start.setToolTip("Start all pending videos using the current visible parameters.")
        self.btn_start.clicked.connect(self.on_start_clicked)
        actions.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop (Ctrl+.)")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setShortcut("Ctrl+.")
        self.btn_stop.setToolTip("Cancel active processing jobs.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        actions.addWidget(self.btn_stop)
        root.addLayout(actions)

        self.cb_preset.currentIndexChanged.connect(self.on_preset_changed)
        self.cb_resolution.currentIndexChanged.connect(self.on_resolution_changed)
        self.cb_model.currentIndexChanged.connect(self.on_model_changed)
        self.slider_cq.valueChanged.connect(self.spin_cq.setValue)
        self.spin_cq.valueChanged.connect(self.slider_cq.setValue)
        self.slider_cq.valueChanged.connect(self.on_cq_slider_changed)
        for control in (self.cb_codec, self.cb_nvenc_preset, self.cb_audio_mode, self.cb_workers,
                        self.cb_frame_fmt, self.cb_denoise):
            control.currentIndexChanged.connect(self.on_expert_setting_changed)
        self.cb_threads.currentTextChanged.connect(self.on_expert_setting_changed)
        self.cb_tile_size.currentTextChanged.connect(self.on_expert_setting_changed)
        self.chk_tta.toggled.connect(self.on_expert_setting_changed)
        self.chk_overwrite.toggled.connect(self.on_settings_changed)
        self.chk_preserve_folders.toggled.connect(self.on_settings_changed)
        for control in (self.cb_naming_mode, self.cb_source, self.cb_audio_tag, self.cb_container):
            control.currentIndexChanged.connect(self.on_settings_changed)
        self.txt_group.textChanged.connect(self.on_settings_changed)
        self.txt_show_name.textChanged.connect(self.on_settings_changed)
        self.txt_custom_suffix.textChanged.connect(self.on_settings_changed)
        self.on_preset_changed()
        self.on_model_changed()

    def on_model_changed(self):
        self._mark_custom_profile()
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
        self._update_model_column_in_table()

    def _update_auto_model_hint(self, queue: List[QueueItem]):
        if (self.cb_model.currentData() or "auto") != "auto" or not queue:
            return
        if self._hint_worker and self._hint_worker.isRunning():
            return
        target_preset = self._get_target_preset()
        self._hint_worker = ModelHintWorker(queue[0].source_path, target_preset)
        self._hint_worker.hint_ready.connect(self._on_hint_ready)
        self._hint_worker.start()

    def _on_hint_ready(self, text: str, reason: str):
        if (self.cb_model.currentData() or "auto") != "auto":
            return
        self.lbl_model_info.setText(text)
        if reason:
            self.lbl_model_info.setToolTip(reason)

    def on_preset_changed(self):
        preset_id = self.cb_preset.currentData()
        if not preset_id:
            return
        preset = get_preset(preset_id)
        if preset_id != "custom":
            self._updating_preset = True
            try:
                for control, value in (
                    (self.cb_resolution, preset.target_preset), (self.cb_codec, preset.video_codec),
                    (self.cb_model, preset.model_id), (self.cb_nvenc_preset, preset.nvenc_preset),
                    (self.cb_audio_mode, preset.audio_mode), (self.cb_denoise, preset.denoise_level),
                    (self.cb_workers, preset.workers), (self.cb_frame_fmt, preset.frame_format),
                ):
                    index = control.findData(value)
                    if index >= 0:
                        control.setCurrentIndex(index)
                self.slider_cq.setValue(preset.cq_level)
                self.chk_tta.setChecked(preset.enable_tta)
            finally:
                self._updating_preset = False
        self.lbl_preset_desc.setText(preset.description)
        self.update_scene_preview()

    def _mark_custom_profile(self):
        if getattr(self, "_updating_preset", False) or getattr(self, "_initializing_ui", False):
            return
        index = self.cb_preset.findData("custom")
        if index >= 0 and self.cb_preset.currentIndex() != index:
            self.cb_preset.setCurrentIndex(index)

    def _get_video_codec(self) -> str:
        return self.cb_codec.currentData() or "hevc_nvenc"

    @staticmethod
    def _editable_choice(control):
        if control.currentText() == control.itemText(control.currentIndex()):
            return control.currentData()
        return control.currentText().strip()

    def _get_thread_spec(self) -> str:
        value = str(self._editable_choice(self.cb_threads)).strip()
        if value.casefold() == "auto":
            return "auto"
        if not re.fullmatch(r"[0-9]+:[0-9]+:[0-9]+", value) or any(int(part) < 1 for part in value.split(":")):
            raise ValueError("Vulkan threads must be Auto or positive LOAD:PROCESS:SAVE counts (e.g. 3:2:4).")
        return value

    def _get_tile_size(self) -> int:
        raw_value = self._editable_choice(self.cb_tile_size)
        if str(raw_value).strip().casefold() == "auto":
            return 0
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError("Tile size must be Auto (0) or an integer of at least 32 pixels.") from None
        if value != 0 and value < 32:
            raise ValueError("Tile size must be Auto (0) or at least 32 pixels.")
        return value

    def _refresh_output_plan(self):
        if self.batch_manager.is_running:
            return
        base = self.txt_source.text().strip()
        if os.path.isfile(base):
            base = os.path.dirname(base)
        self.batch_manager.refresh_output_plan(
            self.txt_output.text().strip(), base or None,
            self.get_current_scene_config(), self.chk_preserve_folders.isChecked(),
            target_preset=self._get_target_preset(), overwrite=self.chk_overwrite.isChecked(),
        )
        self._populate_table(self.batch_manager.queue)

    def on_cq_slider_changed(self, val: int):
        self.lbl_cq_val.setText(f"CQ {val}")
        self._mark_custom_profile()
        self.update_scene_preview()

    def on_expert_setting_changed(self):
        self._mark_custom_profile()
        self.on_settings_changed()

    def update_scene_preview(self):
        if hasattr(self, 'lbl_preview_name') and hasattr(self, 'cb_resolution'):
            cfg = self.get_current_scene_config()
            sample_name = format_output_name("Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.avi", config=cfg)
            self.lbl_preview_name.setText(f"Sample: {sample_name}")

        if hasattr(self, 'lbl_audio') and hasattr(self, 'cb_audio_mode'):
            mode = self.cb_audio_mode.currentData() or "copy"
            labels = {
                "copy": "Audio: Direct Copy",
                "aac_192k": "Audio: AAC 192k",
                "aac_320k": "Audio: AAC 320k",
                "flac": "Audio: FLAC",
            }
            tooltips = {
                "copy": "Lossless Stream Copy: Original soundtrack is preserved 1:1 without re-encoding.",
                "aac_192k": "Re-encoded to AAC at 192 kbps.",
                "aac_320k": "Re-encoded to AAC at 320 kbps.",
                "flac": "Lossless FLAC audio re-encode.",
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
        self.btn_open_logs.setText(f"Logs ({count})")

        if self._log_viewer and self._log_viewer.isVisible():
            self._log_viewer.append_line(timestamped, store=False)

        if hasattr(self, 'txt_console'):
            self.txt_console.append(msg)

    def clear_all_logs(self):
        self._log_buffer.clear()
        if hasattr(self, 'txt_console'):
            self.txt_console.clear()
        self.btn_open_logs.setText("Logs (0)")
        if self._log_viewer and self._log_viewer.isVisible():
            self._log_viewer.txt_log.clear()
            self._log_viewer._update_count()

    def _open_log_viewer(self):
        if self._log_viewer is None:
            self._log_viewer = LogViewerDialog(self._log_buffer, parent=self)
            self._log_viewer.show()
        elif not self._log_viewer.isVisible():
            self._log_viewer._reload_all()
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
        if self._operation_active:
            event.ignore()
            return
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
        if self._operation_active or self.batch_manager.is_running:
            return
        self.batch_manager.clear_queue()
        self._populate_table(self.batch_manager.queue)
        self.lbl_overall_status.setText("Queue cleared.")

    def on_remove_item_clicked(self, target_idx: int):
        if self._operation_active or self.batch_manager.is_running:
            return
        self.batch_manager.remove_items_by_indices([target_idx])
        self._populate_table(self.batch_manager.queue)
        self.lbl_overall_status.setText(f"Episode #{target_idx} removed from queue.")

    def on_remove_selected_clicked(self):
        if self._operation_active or self.batch_manager.is_running:
            return
        indices = {item.index for row in self.table.selectionModel().selectedRows()
                   if (item := self._get_queue_item_from_row(row.row())) is not None}
        self.batch_manager.remove_items_by_indices(indices)
        self._populate_table(self.batch_manager.queue)
        self.lbl_overall_status.setText(f"Removed {len(indices)} selected queue entries.")

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
        if self._operation_active:
            self.lbl_overall_status.setText("Stop the current operation before starting a frame comparison.")
            return
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

        target_preset = self._get_target_preset()
        model_id = target_item.model if target_item.model and target_item.model != "auto" else self.cb_model.currentData() or "auto"
        # Progress stores a display label in older queue items; resolve it like batch jobs do.
        if model_id not in AVAILABLE_MODELS:
            model_id = next((key for key, spec in AVAILABLE_MODELS.items()
                             if model_id in (spec.name, spec.internal_name, spec.name.split(" (")[0])), "auto")
        try:
            threads = self._get_thread_spec()
            tile_size = self._get_tile_size()
        except ValueError as exc:
            self.settings_tabs.setCurrentIndex(2)
            self.lbl_overall_status.setText(str(exc))
            return

        if hasattr(self, "btn_compare_frame"):
            self.btn_compare_frame.setEnabled(False)
        self.lbl_overall_status.setText(f"🔍 Generating Before / After comparison frame for {os.path.basename(target_item.source_path)}...")

        pipeline = VideoUpscalerPipeline(self.hardware)
        self._frame_worker = FramePreviewWorker(
            pipeline=pipeline,
            video_path=target_item.source_path,
            model_id=model_id,
            target_preset=target_preset,
            threads=threads,
            tile_size=tile_size,
            denoise_level=int(self.cb_denoise.currentData()),
            enable_tta=self.chk_tta.isChecked(),
        )
        self._frame_worker.preview_ready.connect(self._on_frame_preview_ready)
        self._set_processing_controls(True)
        self._frame_worker.start()

    def _on_frame_preview_ready(self, ok: bool, orig_path: str, upscaled_path: str, info: dict):
        self._set_processing_controls(False)
        if self._frame_worker and self._frame_worker.pipeline._is_cancelled:
            self.lbl_overall_status.setText("Frame preview cancelled.")
            return
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

        selected_indexes = self.table.selectedIndexes()
        selected_rows = sorted(list({idx.row() for idx in selected_indexes})) if selected_indexes else [row]
        if row not in selected_rows:
            selected_rows = [row]

        model_menu_title = f"🤖 Set AI Model for {len(selected_rows)} Selected Episode(s)" if len(selected_rows) > 1 else "🤖 Set AI Model for this file"
        model_menu = menu.addMenu(model_menu_title)

        act_auto = QAction("Auto Selection (Recommended)", self)
        act_auto.triggered.connect(lambda: self._set_model_for_rows(selected_rows, ""))
        model_menu.addAction(act_auto)
        model_menu.addSeparator()

        for m_id, spec in AVAILABLE_MODELS.items():
            if m_id == "auto":
                continue
            act_m = QAction(spec.name, self)
            act_m.triggered.connect(lambda checked=False, mid=m_id: self._set_model_for_rows(selected_rows, mid))
            model_menu.addAction(act_m)

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
        if item.column() == 4:
            self._show_model_selection_menu(row)
            return

        q_item = self._get_queue_item_from_row(row)
        if q_item:
            if os.path.isfile(q_item.target_full_path):
                self.play_file(q_item.target_full_path)
            else:
                self.reveal_file(q_item.source_path)

    def _show_model_selection_menu(self, row: int):
        q_item = self._get_queue_item_from_row(row)
        if not q_item:
            return

        menu = QMenu(self)
        title_act = menu.addAction(f"AI Model for Episode #{q_item.index}:")
        title_act.setEnabled(False)
        menu.addSeparator()

        act_auto = QAction("Auto Selection (Recommended)", self)
        act_auto.triggered.connect(lambda: self._set_model_for_rows([row], ""))
        menu.addAction(act_auto)
        menu.addSeparator()

        for m_id, spec in AVAILABLE_MODELS.items():
            if m_id == "auto":
                continue
            act_m = QAction(spec.name, self)
            act_m.triggered.connect(lambda checked=False, mid=m_id: self._set_model_for_rows([row], mid))
            menu.addAction(act_m)

        menu.exec(QCursor.pos())

    def _set_model_for_rows(self, rows: List[int], model_id: str):
        if self._operation_active:
            self.lbl_overall_status.setText("Stop processing before changing per-video models.")
            return
        count = 0
        for r in rows:
            q_item = self._get_queue_item_from_row(r)
            if q_item:
                q_item.model = model_id
                q_item.resolved_model = ""
                count += 1
        self._update_model_column_in_table()
        disp = get_model_display_name(model_id) if model_id else "Auto Selection"
        self.lbl_overall_status.setText(f"Updated AI model to '{disp}' for {count} episode(s).")

    @staticmethod
    def _infer_resolution_hint(filepath: str) -> Optional[Tuple[int, int]]:
        """Quickly infers standard resolution dimensions from filename tokens if present."""
        fname = os.path.basename(filepath).lower()
        if any(tag in fname for tag in ("8k", "4320p")):
            return (7680, 4320)
        if any(tag in fname for tag in ("4k", "2160p", "uhd")):
            return (3840, 2160)
        if any(tag in fname for tag in ("1440p", "2k", "qhd")):
            return (2560, 1440)
        if any(tag in fname for tag in ("1080p", "1080i", "fhd")):
            return (1920, 1080)
        if any(tag in fname for tag in ("720p", "hd")):
            return (1280, 720)
        if any(tag in fname for tag in ("576p", "576i", "480p", "480i", "sd", "dvd")):
            return (640, 480)
        return None

    def _get_model_display_for_item(self, item: QueueItem) -> str:
        """Returns the model label to display in the queue table for a given item."""
        if getattr(item, "model", ""):
            return get_model_display_name(item.model)

        cb_val = self.cb_model.currentData() if hasattr(self, "cb_model") else "auto"
        if cb_val and cb_val != "auto":
            return get_model_display_name(cb_val)

        target_preset = self._get_target_preset()

        if getattr(item, "resolution_str", ""):
            try:
                parts = item.resolution_str.lower().split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    rec = recommend_model_for_video(w, h, target_preset)
                    return f"Auto ({get_model_display_name(rec.model.id)})"
            except Exception:
                pass

        hint = self._infer_resolution_hint(item.source_path)
        if hint:
            w, h = hint
            rec = recommend_model_for_video(w, h, target_preset)
            return f"Auto ({get_model_display_name(rec.model.id)})"

        return "Auto (Anime 6B)"

    def _get_model_tooltip_for_item(self, item: QueueItem) -> str:
        display = self._get_model_display_for_item(item)
        raw_model = getattr(item, "model", "") or (self.cb_model.currentData() if hasattr(self, "cb_model") else "auto")
        spec = None
        for m_id, s in AVAILABLE_MODELS.items():
            if m_id == raw_model or s.name.startswith(display) or s.id in display.lower() or display in s.name:
                spec = s
                break
        desc = spec.description if spec else "Neural network upscaling engine."
        rec_for = f"\nBest suited for: {spec.recommended_for}" if spec and spec.recommended_for else ""
        last_run = f"\nLast run: {item.resolved_model}" if item.resolved_model else ""
        return (
            f"AI Model: {display}\n"
            f"{desc}{rec_for}{last_run}\n"
            "Double-click or right-click to customize model for this episode."
        )

    def _update_model_column_in_table(self):
        """Refreshes the Model column across all visible rows."""
        for r in range(self.table.rowCount()):
            q_item = self._get_queue_item_from_row(r)
            if q_item:
                it_model = self.table.item(r, 4)
                if it_model:
                    it_model.setText(self._get_model_display_for_item(q_item))
                    it_model.setToolTip(self._get_model_tooltip_for_item(q_item))

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
        res_tag = self._get_output_tag_for_current_resolution()

        codec_tag = "H.265" if self._get_video_codec() in ("hevc_nvenc", "libx265") else "H.264"

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
            container=self.cb_container.currentData(),
            custom_suffix=self.txt_custom_suffix.text().strip(),
        )

    def _get_target_preset(self) -> str:
        if hasattr(self, "cb_resolution"):
            return self.cb_resolution.currentData() or "4k"
        return "4k"

    def _get_output_tag_for_current_resolution(self) -> str:
        return RESOLUTION_TAGS[self._get_target_preset()]

    def on_resolution_changed(self):
        self._mark_custom_profile()
        old_out = self.txt_output.text().strip()
        new_tag = self._get_output_tag_for_current_resolution()
        for resolution_tag in RESOLUTION_TAGS.values():
            tag = f"({resolution_tag})"
            if old_out.endswith(tag):
                updated_out = old_out[:-len(tag)] + f"({new_tag})"
                self.txt_output.setText(updated_out)
                self.settings.setValue("last_output", updated_out)
                break
        self.on_settings_changed()

    def on_scan_clicked(self):
        if self._operation_active:
            return
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
        self._update_model_column_in_table()

    def _do_settings_changed(self):
        """Retained for backward-compatibility; delegates to update_scene_preview only."""
        self.update_scene_preview()

    def on_preflight_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("Queue is empty. Please scan a folder first.")
            return

        out_dir = self.txt_output.text().strip()
        if not out_dir:
            self.lbl_overall_status.setText("Please specify an output folder.")
            return
        target_preset = self._get_target_preset()
        video_codec = self._get_video_codec()
        model_id = self.cb_model.currentData() or "auto"
        cq_level = self.slider_cq.value()
        scene_cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()

        self._refresh_output_plan()
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
        row = self._table_rows_by_id.get(item_id, -1)
        if 0 <= row < self.table.rowCount():
            it = self.table.item(row, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == item_id:
                return row
        # Keep identity correct if rows have been reordered by the view.
        self._table_rows_by_id.clear()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                self._table_rows_by_id[it.data(Qt.ItemDataRole.UserRole)] = r
        return self._table_rows_by_id.get(item_id, -1)

    def _get_queue_item_from_row(self, row: int) -> Optional[QueueItem]:
        if 0 <= row < self.table.rowCount():
            it = self.table.item(row, 0)
            if it:
                item_id = it.data(Qt.ItemDataRole.UserRole)
                if item_id:
                    return self._queue_items_by_id.get(item_id)
        return None

    def _populate_table(self, queue: List[QueueItem]):
        self._queue_items_by_id = {item.id: item for item in queue}
        self._table_rows_by_id = {item.id: row for row, item in enumerate(queue)}
        self._displayed_statuses = {item.id: item.status for item in queue}
        self._status_counts = Counter(self._displayed_statuses.values())
        updates_enabled = self.table.updatesEnabled()
        self.table.setUpdatesEnabled(False)
        try:
            self._populate_table_rows(queue)
        finally:
            self.table.setUpdatesEnabled(updates_enabled)
        self._resize_queue_columns()

    def _populate_table_rows(self, queue: List[QueueItem]):
        self.table.setRowCount(len(queue))

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

            model_disp = self._get_model_display_for_item(item)
            it_model = QTableWidgetItem(model_disp)
            it_model.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it_model.setForeground(QColor("#a78bfa"))
            it_model.setToolTip(self._get_model_tooltip_for_item(item))
            self.table.setItem(r, 4, it_model)

            it_status = QTableWidgetItem(self._status_badge_text(item.status))
            it_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(it_status, item.status)
            self.table.setItem(r, 5, it_status)

            it_detail = QTableWidgetItem(item.status_detail or "Ready")
            it_detail.setForeground(QColor("#94a3b8"))
            self.table.setItem(r, 6, it_detail)

        total = len(queue)
        self.btn_stat_all.setText(f"All: {total}")
        self._update_queue_stats()
        ready = self._status_counts[EpisodeStatus.PENDING]
        self.lbl_overall_status.setText(f"Indexed {total} episodes ({ready} ready to process).")
        self.apply_table_filters()

    def _update_queue_stats(self):
        self.btn_stat_ready.setText(f"Ready: {self._status_counts[EpisodeStatus.PENDING]}")
        self.btn_stat_completed.setText(f"Completed: {self._status_counts[EpisodeStatus.COMPLETED]}")
        self.btn_stat_skipped.setText(f"Skipped: {self._status_counts[EpisodeStatus.SKIPPED]}")
        if hasattr(self, 'btn_stat_failed'):
            failed_count = self._status_counts[EpisodeStatus.FAILED]
            self.btn_stat_failed.setText(f"Errors: {failed_count}")
            style_class = "statPillError" if failed_count else "statPill"
            if self.btn_stat_failed.property("class") != style_class:
                self.btn_stat_failed.setProperty("class", style_class)
                self.btn_stat_failed.style().unpolish(self.btn_stat_failed)
                self.btn_stat_failed.style().polish(self.btn_stat_failed)

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
            self._apply_row_filter(r, search_text, mode)

    def _apply_row_filter(self, row: int, search_text: str, mode: str):
        text_match = not search_text or any(
            (cell := self.table.item(row, col)) is not None and search_text in cell.text().lower()
            for col in (1, 3, 4)
        )
        expected_status = {
            "ready": EpisodeStatus.PENDING,
            "completed": EpisodeStatus.COMPLETED,
            "skipped": EpisodeStatus.SKIPPED,
            "failed": EpisodeStatus.FAILED,
        }.get(mode)
        q_item = self._get_queue_item_from_row(row)
        status_match = expected_status is None or (q_item is not None and q_item.status == expected_status)
        self.table.setRowHidden(row, not (text_match and status_match))

    def on_start_clicked(self):
        self.start_processing()

    def on_test_single_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue. Please scan a folder first.")
            return

        target_idx = self._selected_item_index()
        for it in self.batch_manager.queue:
            if target_idx is None and it.status == EpisodeStatus.PENDING:
                target_idx = it.index
                break
        if target_idx is None:
            target_idx = self.batch_manager.queue[0].index

        self.start_processing(target_indices=[target_idx])

    def on_preview_clip_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue. Please scan a folder first.")
            return

        target_idx = self._selected_item_index()
        for it in self.batch_manager.queue:
            if target_idx is None and it.status == EpisodeStatus.PENDING:
                target_idx = it.index
                break
        if target_idx is None:
            target_idx = self.batch_manager.queue[0].index

        dur = float(self.spin_preview_dur.value())
        self.start_processing(target_indices=[target_idx], preview_duration_s=dur)

    def _selected_item_index(self) -> Optional[int]:
        selected = self.table.selectionModel().selectedRows()
        if selected:
            item = self._get_queue_item_from_row(selected[0].row())
            if item:
                return item.index
        return None

    def start_processing(self, target_indices: Optional[List[int]] = None, preview_duration_s: Optional[float] = None):
        if self._operation_active or (self.worker and self.worker.isRunning()):
            self.lbl_overall_status.setText("Processing is already running. Stop it before starting another batch.")
            return
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue.")
            return

        out_dir = self.txt_output.text().strip()
        base_dir = self.txt_source.text().strip()
        if os.path.isfile(base_dir):
            base_dir = os.path.dirname(base_dir)
        if not out_dir:
            self.lbl_overall_status.setText("Please specify an output folder.")
            return

        target_preset = self._get_target_preset()

        video_codec = self._get_video_codec()

        try:
            threads = self._get_thread_spec()
            tile_size = self._get_tile_size()
        except ValueError as exc:
            self.settings_tabs.setCurrentIndex(2)
            self.lbl_overall_status.setText(str(exc))
            return
        self._refresh_output_plan()

        model_id = self.cb_model.currentData() or "auto"
        if model_id == "auto":
            try:
                first_item = next((it for it in self.batch_manager.queue if not target_indices or it.index in target_indices), None)
                if first_item:
                    meta = probe_video(first_item.source_path)
                    rec = recommend_model_for_video(meta.display_width, meta.display_height, target_preset)
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

        max_workers = int(self.cb_workers.currentData())
        if target_indices and len(target_indices) == 1:
            max_workers = 1

        frame_format = self.cb_frame_fmt.currentData()

        cq_level = self.slider_cq.value()
        nvenc_preset = self.cb_nvenc_preset.currentData()
        audio_mode = self.cb_audio_mode.currentData() or "copy"
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
        if hasattr(self, 'btn_compare_frame'):
            self.btn_compare_frame.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_processing_controls(True)

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
        if self._frame_worker and self._frame_worker.isRunning():
            self._frame_worker.pipeline.cancel()
        self.btn_stop.setEnabled(False)

    def on_item_updated(self, item: QueueItem):
        row = self._find_row_by_item_id(item.id)
        if row == -1:
            # Ignore a queued signal belonging to an item that has since been removed.
            return
        self._queue_items_by_id[item.id] = item
        previous_status = self._displayed_statuses.get(item.id)
        status_changed = previous_status != item.status
        if status_changed:
            if previous_status is not None:
                self._status_counts[previous_status] -= 1
            self._status_counts[item.status] += 1
            self._displayed_statuses[item.id] = item.status
            self._update_queue_stats()
        if 0 <= row < self.table.rowCount():
            it_model = self.table.item(row, 4)
            if it_model:
                it_model.setText(self._get_model_display_for_item(item))
                it_model.setToolTip(self._get_model_tooltip_for_item(item))

            it_status = self.table.item(row, 5)
            if it_status and status_changed:
                it_status.setText(self._status_badge_text(item.status))
                self._style_status_item(it_status, item.status)

            it_detail = self.table.item(row, 6)
            if it_detail:
                it_detail.setText(item.status_detail)

            if item.status == EpisodeStatus.PROCESSING and it_status:
                self.table.scrollToItem(it_status)
            if status_changed or self.txt_filter.text().strip():
                self._apply_row_filter(row, self.txt_filter.text().lower().strip(), self._current_filter_mode)

    def on_batch_updated(self, current: int, total: int, pct: float, status_str: str, eta_str: str = ""):
        self.progress_bar.setValue(int(pct))
        self.lbl_overall_status.setText(status_str)
        if eta_str:
            self.lbl_eta.setText(eta_str)
            self.lbl_eta.setVisible(True)
        else:
            self.lbl_eta.setVisible(False)

    def on_finished(self):
        self._set_processing_controls(False)
        self.btn_start.setEnabled(True)
        if hasattr(self, 'btn_preview'):
            self.btn_preview.setEnabled(True)
        if hasattr(self, 'btn_preview_clip'):
            self.btn_preview_clip.setEnabled(True)
        if hasattr(self, 'btn_compare_frame'):
            self.btn_compare_frame.setEnabled(True)
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

        if self._frame_worker and self._frame_worker.isRunning():
            self._frame_worker.pipeline.cancel()
            if not self._frame_worker.wait(3000):
                self.lbl_overall_status.setText("Stopping frame preview. Close again once it has finished.")
                event.ignore()
                return

        if hasattr(self, "_scan_worker") and self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.terminate()
            self._scan_worker.wait(500)

        if hasattr(self, "worker") and self.worker and self.worker.isRunning():
            self.worker.wait(1500)

        super().closeEvent(event)

    def _setup_all_comboboxes(self):
        """Ensures all QComboBox instances have styled item delegates and solid backgrounds."""
        for cb in self.findChildren(QComboBox):
            setup_combobox(cb)
        for control_type in (QComboBox, QSpinBox, QSlider):
            for control in self.findChildren(control_type):
                control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                control.installEventFilter(self)

    def _toggle_maximize(self):
        """Toggles between maximized and normal window state."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def showEvent(self, event):
        super().showEvent(event)
        if QApplication.platformName() != "offscreen":
            apply_dark_titlebar(self)
        QTimer.singleShot(0, self._resize_queue_columns)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_queue_columns()

    def _resize_queue_columns(self):
        if hasattr(self, "table"):
            available = self.table.viewport().width()
            detail = max(180, min(360, available // 4))
            self.table.setColumnWidth(6, detail)
            filename = max(220, (available - (40 + 70 + 120 + 105 + detail)) // 2)
            for column in (1, 3):
                self.table.setColumnWidth(column, filename)

    def _set_processing_controls(self, busy: bool):
        self._operation_active = busy
        # Leave the tabs navigable so every running parameter remains inspectable.
        for control_type in (QComboBox, QSpinBox, QSlider, QCheckBox, QLineEdit):
            for control in self.parameter_panel.findChildren(control_type):
                control.setEnabled(not busy)
        for control in (self.txt_source, self.txt_output, self.spin_preview_dur):
            control.setEnabled(not busy)
        for control in (self.btn_add_files, self.btn_browse_source, self.btn_browse_output,
                        self.btn_scan, self.btn_clear_queue, self.btn_remove_selected,
                        self.btn_retry_failed, self.btn_preflight, self.btn_start,
                        self.btn_compare_frame, self.btn_preview, self.btn_preview_clip):
            control.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def eventFilter(self, watched, event):
        # Scrolling a settings page must not silently edit the value under the pointer.
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def run_gui():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("CYLAE.AnimaScale.4K.1.0")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    icon_path = os.path.join(assets_dir, "app_icon.ico" if os.name == "nt" else "app_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = UpscalerMainWindow()
    window.show()
    return app.exec()
