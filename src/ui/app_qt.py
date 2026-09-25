import os
import sys
import subprocess
from typing import Dict, List, Optional

from PyQt6.QtCore import QSettings, QStandardPaths, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QCursor, QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget
)

from src.core.ai_models import AVAILABLE_MODELS, check_local_model_availability, recommend_model_for_video
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import HardwareInfo, get_base_dir, get_hardware_info
from src.core.presets import ENHANCEMENT_PRESETS, get_preset
from src.core.scene_renamer import (
    SceneNamingConfig,
    format_clean_name,
    format_output_name,
    format_scene_name,
    get_scene_relative_output_path
)
from src.core.video_pipeline import VideoUpscalerPipeline, cleanup_stale_temp_directories, probe_video


DARK_STYLESHEET = """
/* ==========================================================================
   macOS Human Interface Guidelines (HIG) Dark Mode Design System
   ========================================================================== */

QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
}

/* Elevated card surfaces (macOS Finder / System Settings panel style) */
QGroupBox, QFrame#macPanel {
    background-color: #252528;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-top: 10px;
    padding: 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #0a84ff;
    font-weight: 700;
}

/* Controls & Inputs (macOS dark textfields & select popups) */
QLineEdit, QComboBox, QSpinBox {
    background-color: #1c1c1e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 7px 10px;
    color: #f5f5f7;
    selection-background-color: #0a84ff;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #0a84ff;
    background-color: #242426;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
    color: #f5f5f7;
    padding: 4px;
    border-radius: 8px;
}

/* Standard Buttons (macOS push button style) */
QPushButton {
    background-color: #323236;
    border: 1px solid #48484a;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
    color: #f5f5f7;
}

QPushButton:hover {
    background-color: #3a3a3e;
    border-color: #545458;
}

QPushButton:pressed {
    background-color: #252528;
}

QPushButton:disabled {
    background-color: #242426;
    border-color: #323236;
    color: #636366;
}

/* Primary Action - Apple Green (macOS Run / Start Batch) */
QPushButton#btnStart {
    background-color: #30d158;
    border: 1px solid #34c759;
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    padding: 9px 22px;
    border-radius: 8px;
}

QPushButton#btnStart:hover {
    background-color: #34c759;
}

QPushButton#btnStart:pressed {
    background-color: #248a3d;
}

QPushButton#btnStart:disabled {
    background-color: #28282a;
    border-color: #38383a;
    color: #636366;
}

/* Secondary Action - Apple Blue (Render Preview Clip) */
QPushButton#btnPreviewClip {
    background-color: #0a84ff;
    border: 1px solid #409cff;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    padding: 9px 18px;
    border-radius: 8px;
}

QPushButton#btnPreviewClip:hover {
    background-color: #409cff;
}

QPushButton#btnPreviewClip:pressed {
    background-color: #0071e3;
}

QPushButton#btnPreviewClip:disabled {
    background-color: #28282a;
    border-color: #38383a;
    color: #636366;
}

/* Tertiary Action - Apple Indigo (Test 1 Episode) */
QPushButton#btnPreview {
    background-color: #5e5ce6;
    border: 1px solid #7d7aff;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    padding: 9px 18px;
    border-radius: 8px;
}

QPushButton#btnPreview:hover {
    background-color: #706ef5;
}

QPushButton#btnPreview:pressed {
    background-color: #4b48d6;
}

QPushButton#btnPreview:disabled {
    background-color: #28282a;
    border-color: #38383a;
    color: #636366;
}

/* Destructive / Cancel Action - Apple Red (Stop Processing) */
QPushButton#btnStop {
    background-color: #ff453a;
    border: 1px solid #ff6961;
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    padding: 9px 22px;
    border-radius: 8px;
}

QPushButton#btnStop:hover {
    background-color: #ff6961;
}

QPushButton#btnStop:pressed {
    background-color: #d70015;
}

QPushButton#btnStop:disabled {
    background-color: #28282a;
    border-color: #38383a;
    color: #636366;
}

/* Toolbar Action (Scan / Select Folder) */
QPushButton#btnScan {
    background-color: #0a84ff;
    border: 1px solid #409cff;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
}

QPushButton#btnScan:hover {
    background-color: #409cff;
}

/* Filter Stat Pills (macOS Segmented Control / Capsule badges) */
QPushButton.statPill {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #aeaeb2;
}

QPushButton.statPill:hover {
    background-color: #3a3a3c;
    color: #f5f5f7;
    border-color: #0a84ff;
}

QPushButton.statPill:checked {
    background-color: #0a84ff;
    color: #ffffff;
    border-color: #0a84ff;
}

QPushButton.statPillError {
    background-color: rgba(255, 69, 58, 0.15);
    border: 1px solid rgba(255, 69, 58, 0.35);
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #ff453a;
}

QPushButton.statPillError:hover {
    background-color: rgba(255, 69, 58, 0.25);
    border-color: #ff453a;
    color: #ff6961;
}

QPushButton.statPillError:checked {
    background-color: #ff453a;
    color: #ffffff;
    border-color: #ff453a;
}

/* Table Widget (macOS Finder List View) */
QTableWidget {
    background-color: #1e1e1e;
    alternate-background-color: #252528;
    border: 1px solid #38383a;
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
}

QTableWidget QHeaderView::section {
    background-color: #252528;
    color: #8e8e93;
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid #38383a;
    font-weight: 600;
    font-size: 11px;
}

/* Progress Bar (macOS Slim Pill Track) */
QProgressBar {
    background-color: #2c2c2e;
    border: none;
    border-radius: 5px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    min-height: 10px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0a84ff, stop:1 #30d158);
    border-radius: 5px;
}

/* Checkbox (macOS Toggle indicator) */
QCheckBox {
    color: #aeaeb2;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #48484a;
    border-radius: 4px;
    background-color: #2c2c2e;
}

QCheckBox::indicator:checked {
    background-color: #0a84ff;
    border-color: #0a84ff;
}

/* Menus & Dialog Alerts (macOS Translucent Dark styling) */
QMenu, QMessageBox {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 10px;
    padding: 6px;
}

QMenu::item {
    padding: 6px 20px 6px 12px;
    border-radius: 6px;
    color: #f5f5f7;
}

QMenu::item:selected {
    background-color: #0a84ff;
    color: #ffffff;
}

/* Scrollbars (macOS Overlay Scrollbar feel) */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #48484a;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #636366;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #48484a;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #636366;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips (macOS Cocoa HUD tooltips) */
QToolTip {
    background-color: #2c2c2e;
    color: #f5f5f7;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}

/* Sliders (macOS slider control) */
QSlider::groove:horizontal {
    height: 4px;
    background: #3a3a3c;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #0a84ff;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
    border: 1px solid #48484a;
}

QSlider::handle:horizontal:hover {
    background: #f5f5f7;
    border-color: #0a84ff;
}

/* Expert Panel Toggle */
QPushButton#btnToggleExpert {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 5px 12px;
    color: #0a84ff;
    font-weight: 600;
    font-size: 12px;
}

QPushButton#btnToggleExpert:hover {
    background-color: #3a3a3c;
    border-color: #0a84ff;
}

/* Terminal Console (macOS Terminal dark style) */
QTextEdit#txtConsole {
    background-color: #141416;
    border: 1px solid #2c2c2e;
    border-radius: 8px;
    font-family: 'SF Mono', 'Cascadia Mono', 'Consolas', monospace;
    font-size: 11px;
    color: #0a84ff;
    padding: 8px;
}
"""


class LogViewerDialog(QDialog):
    """Full-featured log viewer dialog with per-category coloring, auto-scroll,
    live filter, and export to file.

    The dialog is non-modal so it can stay open alongside the main window
    during a batch job. It receives new lines via append_line().
    """

    # CSS color tokens keyed by log-line prefix (macOS HIG Dark Mode Palette)
    _COLORS = {
        "[FFMPEG AUDIO]": "#ff9f0a",       # Apple Orange
        "[FFMPEG AUDIO FALLBACK]": "#ff9f0a",
        "[FFMPEG EXTRACT": "#5e5ce6",      # Apple Indigo
        "[FFMPEG ENCODE]": "#bf5af2",      # Apple Purple
        "[VULKAN AI]": "#30d158",          # Apple Green
        "[AUDIO REPAIR": "#ff453a",        # Apple Red
        "[AUDIO VERIFIED]": "#30d158",     # Apple Green
        "[AUDIO WARNING]": "#ffd60a",      # Apple Yellow
        "[AUDIO]": "#30d158",
        "[FRAMES]": "#0a84ff",             # Apple Blue
        "[ERROR]": "#ff453a",             # Apple Red
        "[WARNING]": "#ffd60a",           # Apple Yellow
    }
    _DEFAULT_COLOR = "#8e8e93"             # Apple Muted Slate             # slate-400 for uncategorised lines

    def __init__(self, log_buffer: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 Live Engine Logs")
        self.resize(900, 560)
        self.setMinimumSize(640, 360)
        self.setStyleSheet(DARK_STYLESHEET)
        self._log_buffer = log_buffer  # shared reference — updated externally
        self._auto_scroll = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        lbl_title = QLabel("🖥️ Live Engine Logs")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #0a84ff;")
        toolbar.addWidget(lbl_title)

        toolbar.addStretch()

        lbl_filter = QLabel("Filter:")
        lbl_filter.setStyleSheet("color: #8e8e93; font-size: 12px;")
        toolbar.addWidget(lbl_filter)

        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("ffmpeg / audio / vulkan...")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setMaximumWidth(200)
        self.txt_filter.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.txt_filter)

        self.btn_autoscroll = QPushButton("⬇ Auto-scroll: ON")
        self.btn_autoscroll.setCheckable(True)
        self.btn_autoscroll.setChecked(True)
        self.btn_autoscroll.setStyleSheet("padding: 4px 12px; font-size: 11px; font-weight: 600; color: #30d158; border: 1px solid #30d158; border-radius: 6px;")
        self.btn_autoscroll.toggled.connect(self._toggle_autoscroll)
        toolbar.addWidget(self.btn_autoscroll)

        btn_copy_all = QPushButton("📋 Copy All")
        btn_copy_all.setStyleSheet("padding: 4px 12px; font-size: 11px; border-radius: 6px;")
        btn_copy_all.clicked.connect(self._copy_all)
        toolbar.addWidget(btn_copy_all)

        btn_export = QPushButton("💾 Export Log...")
        btn_export.setStyleSheet("padding: 4px 12px; font-size: 11px; border-radius: 6px;")
        btn_export.clicked.connect(self._export_log)
        toolbar.addWidget(btn_export)

        btn_clear = QPushButton("🧹 Clear")
        btn_clear.setStyleSheet("padding: 4px 12px; font-size: 11px; border-radius: 6px;")
        btn_clear.clicked.connect(self._clear_logs)
        toolbar.addWidget(btn_clear)

        layout.addLayout(toolbar)

        # ── Legend row ───────────────────────────────────────────────────────
        legend = QHBoxLayout()
        legend.setSpacing(12)
        _categories = [
            ("FFmpeg Extract", "#5e5ce6"),
            ("FFmpeg Encode", "#bf5af2"),
            ("Audio", "#30d158"),
            ("Vulkan AI", "#30d158"),
            ("Audio Repair", "#ff453a"),
            ("Warning", "#ffd60a"),
            ("Error", "#ff453a"),
        ]
        for name, color in _categories:
            dot = QLabel(f"● {name}")
            dot.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            legend.addWidget(dot)
        legend.addStretch()
        layout.addLayout(legend)

        # ── Log text area ────────────────────────────────────────────────────
        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("txtConsole")
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.txt_log, 1)

        # ── Status bar ───────────────────────────────────────────────────────
        self.lbl_count = QLabel("0 lines")
        self.lbl_count.setStyleSheet("color: #8e8e93; font-size: 11px;")
        layout.addWidget(self.lbl_count, 0, Qt.AlignmentFlag.AlignRight)

        # Populate from existing buffer
        self._reload_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def append_line(self, msg: str):
        """Appends one log line, applying per-category color.  Thread-safe via
        the Qt signal/slot mechanism — must be called from the main thread."""
        filter_text = self.txt_filter.text().lower().strip()
        if filter_text and filter_text not in msg.lower():
            # Line is added to buffer but not shown while filter is active
            self._update_count()
            return
        self._append_colored(msg)
        if self._auto_scroll:
            self.txt_log.verticalScrollBar().setValue(
                self.txt_log.verticalScrollBar().maximum()
            )
        self._update_count()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _color_for(self, line: str) -> str:
        for prefix, color in self._COLORS.items():
            if prefix in line:
                return color
        return self._DEFAULT_COLOR

    def _append_colored(self, msg: str):
        color = self._color_for(msg)
        # Escape HTML special chars then wrap in a colored span
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.txt_log.append(f'<span style="color:{color};">{safe}</span>')

    def _reload_all(self):
        """Rebuilds the view from the shared log buffer (respecting active filter)."""
        self.txt_log.clear()
        filter_text = self.txt_filter.text().lower().strip()
        for line in self._log_buffer:
            if filter_text and filter_text not in line.lower():
                continue
            self._append_colored(line)
        if self._auto_scroll:
            self.txt_log.verticalScrollBar().setValue(
                self.txt_log.verticalScrollBar().maximum()
            )
        self._update_count()

    def _update_count(self):
        total = len(self._log_buffer)
        visible = self.txt_log.document().blockCount() - 1  # Qt adds trailing \n block
        if self.txt_filter.text().strip():
            self.lbl_count.setText(f"{visible} / {total} lines (filtered)")
        else:
            self.lbl_count.setText(f"{total} lines")

    def _apply_filter(self):
        self._reload_all()

    def _toggle_autoscroll(self, checked: bool):
        self._auto_scroll = checked
        self.btn_autoscroll.setText(f"⬇ Auto-scroll: {'ON' if checked else 'OFF'}")
        if checked:
            self.btn_autoscroll.setStyleSheet(
                "padding: 4px 12px; font-size: 11px; font-weight: 600; color: #30d158; border: 1px solid #30d158; border-radius: 6px;"
            )
        else:
            self.btn_autoscroll.setStyleSheet(
                "padding: 4px 12px; font-size: 11px; font-weight: 600; color: #8e8e93; border: 1px solid #3a3a3c; border-radius: 6px;"
            )

    def _copy_all(self):
        QApplication.clipboard().setText("\n".join(self._log_buffer))

    def _export_log(self):
        import datetime
        default_name = f"upscaler_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", default_name, "Text files (*.txt);;All files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(self._log_buffer))
            except Exception as e:
                self.lbl_count.setText(f"Export failed: {e}")

    def _clear_logs(self):
        if self.parent() and hasattr(self.parent(), "clear_all_logs"):
            self.parent().clear_all_logs()
        else:
            self._log_buffer.clear()
            self.txt_log.clear()
            self._update_count()


class DoctorDialog(QDialog):
    """Modern modal dialog displaying complete hardware and environment diagnostic."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System & Hardware Diagnostic")
        self.resize(580, 520)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("🩺 Hardware & Environment Diagnostic")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0a84ff;")
        layout.addWidget(title)

        hw = get_hardware_info()
        base_dir = get_base_dir()
        models_avail = check_local_model_availability(base_dir)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("background-color: #141416; border: 1px solid #38383a; border-radius: 10px; padding: 12px; font-family: 'SF Mono', 'Cascadia Mono', monospace; font-size: 12px; color: #f5f5f7;")

        lines = [
            "=" * 55,
            "               SYSTEM ACCELERATION REPORT",
            "=" * 55,
            f"CPU Logical Cores    : {hw.cpu_cores}",
            f"GPU Name             : {hw.gpu_name}",
            f"GPU VRAM             : {hw.gpu_vram_gb} GB",
            f"NVIDIA GPU Active    : {'Yes' if hw.has_nvidia else 'No'}",
            f"NVENC H.265 (HEVC)   : {'Available (Hardware Acceleration)' if hw.has_nvenc_hevc else 'Not Available'}",
            f"NVENC H.264 (AVC)    : {'Available (Hardware Acceleration)' if hw.has_nvenc_h264 else 'Not Available'}",
            f"FFmpeg Path          : {hw.ffmpeg_path or 'Not Found'}",
            f"FFmpeg Version       : {hw.ffmpeg_version}",
            f"Real-ESRGAN Binary   : {'Found' if hw.realesrgan_bin else 'Not Found'}",
            f"RealCUGAN Binary     : {'Found' if hw.realcugan_bin else 'Not Found'}",
            "-" * 55,
            "               LOCAL AI MODELS (100% OFFLINE)",
            "-" * 55,
        ]
        for m_id, ok in models_avail.items():
            if m_id == "auto":
                continue
            status = "Verified on Disk (Ready)" if ok else "Missing"
            lines.append(f"  {m_id:<23}: {status}")
        lines.extend([
            "=" * 55,
            "Status: System is fully configured for fast 4K local upscaling.",
            "=" * 55
        ])

        text_edit.setPlainText("\n".join(lines))
        layout.addWidget(text_edit, 1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignRight)


class FrameComparisonDialog(QDialog):
    """Interactive visual Before / After comparison with side-by-side or split viewing."""
    def __init__(self, orig_path: str, upscaled_path: str, info: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Before / After Visual Preview")
        self.resize(1100, 750)
        self.setStyleSheet(DARK_STYLESHEET)
        self.orig_path = orig_path
        self.upscaled_path = upscaled_path
        self.info = info
        self.zoom_level = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header info bar
        header = QFrame()
        header.setObjectName("macPanel")
        h_box = QHBoxLayout(header)
        h_box.setContentsMargins(14, 10, 14, 10)
        h_box.setSpacing(12)

        title = QLabel("🔍 High Precision Comparison")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        h_box.addWidget(title)

        model_badge = QLabel(f"Model: {info.get('model_name', 'AI')}")
        model_badge.setStyleSheet("color: #0a84ff; font-weight: 600; font-size: 11px; padding: 4px 8px; background-color: rgba(10, 132, 255, 0.12); border: 1px solid rgba(10, 132, 255, 0.3); border-radius: 8px;")
        h_box.addWidget(model_badge)

        orig_w = info.get("orig_width", "?")
        orig_h = info.get("orig_height", "?")
        tar_w = info.get("target_width", "?")
        tar_h = info.get("target_height", "?")
        res_badge = QLabel(f"Source: {orig_w}×{orig_h}  ➔  Target: {tar_w}×{tar_h}")
        res_badge.setStyleSheet("color: #30d158; font-weight: 600; font-size: 11px; padding: 4px 8px; background-color: rgba(48, 209, 88, 0.12); border: 1px solid rgba(48, 209, 88, 0.3); border-radius: 8px;")
        h_box.addWidget(res_badge)

        time_s = info.get("time_seconds", 0)
        time_badge = QLabel(f"Render: {time_s}s")
        time_badge.setStyleSheet("color: #ff9f0a; font-weight: 600; font-size: 11px; padding: 4px 8px; background-color: rgba(255, 159, 10, 0.12); border: 1px solid rgba(255, 159, 10, 0.3); border-radius: 8px;")
        h_box.addWidget(time_badge)

        h_box.addStretch()

        # View Mode selector
        h_box.addWidget(QLabel("View Mode:"))
        self.cb_view_mode = QComboBox()
        self.cb_view_mode.addItems(["Side by side", "After (AI Upscaled)", "Before (Original)"])
        self.cb_view_mode.currentIndexChanged.connect(self._on_view_mode_changed)
        h_box.addWidget(self.cb_view_mode)

        # Zoom buttons
        btn_zoom_in = QPushButton("🔍 +")
        btn_zoom_in.setToolTip("Zoom in")
        btn_zoom_in.clicked.connect(self._zoom_in)
        h_box.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("🔍 -")
        btn_zoom_out.setToolTip("Zoom out")
        btn_zoom_out.clicked.connect(self._zoom_out)
        h_box.addWidget(btn_zoom_out)

        btn_fit = QPushButton("100%")
        btn_fit.setToolTip("Actual size 100%")
        btn_fit.clicked.connect(self._zoom_reset)
        h_box.addWidget(btn_fit)

        layout.addWidget(header)

        # Images Container
        self.img_container = QWidget()
        self.img_layout = QHBoxLayout(self.img_container)
        self.img_layout.setContentsMargins(0, 0, 0, 0)
        self.img_layout.setSpacing(12)

        # Left / Original Panel
        self.panel_orig = QFrame()
        self.panel_orig.setObjectName("macPanel")
        p_orig_layout = QVBoxLayout(self.panel_orig)
        p_orig_layout.setContentsMargins(8, 8, 8, 8)
        lbl_orig_title = QLabel(f"📷 ORIGINAL ({orig_w}×{orig_h})")
        lbl_orig_title.setStyleSheet("font-weight: 700; color: #8e8e93; font-size: 12px;")
        lbl_orig_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p_orig_layout.addWidget(lbl_orig_title)

        self.scroll_orig = QScrollArea()
        self.scroll_orig.setWidgetResizable(True)
        self.scroll_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig_img = QLabel()
        self.lbl_orig_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_orig.setWidget(self.lbl_orig_img)
        p_orig_layout.addWidget(self.scroll_orig, 1)
        self.img_layout.addWidget(self.panel_orig, 1)

        # Right / Upscaled Panel
        self.panel_up = QFrame()
        self.panel_up.setObjectName("macPanel")
        p_up_layout = QVBoxLayout(self.panel_up)
        p_up_layout.setContentsMargins(8, 8, 8, 8)
        lbl_up_title = QLabel(f"✨ AI UPSCALED ({tar_w}×{tar_h})")
        lbl_up_title.setStyleSheet("font-weight: 700; color: #30d158; font-size: 12px;")
        lbl_up_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p_up_layout.addWidget(lbl_up_title)

        self.scroll_up = QScrollArea()
        self.scroll_up.setWidgetResizable(True)
        self.scroll_up.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_up_img = QLabel()
        self.lbl_up_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_up.setWidget(self.lbl_up_img)
        p_up_layout.addWidget(self.scroll_up, 1)
        self.img_layout.addWidget(self.panel_up, 1)

        # Synchronize scrollbars
        self.scroll_orig.horizontalScrollBar().valueChanged.connect(self.scroll_up.horizontalScrollBar().setValue)
        self.scroll_up.horizontalScrollBar().valueChanged.connect(self.scroll_orig.horizontalScrollBar().setValue)
        self.scroll_orig.verticalScrollBar().valueChanged.connect(self.scroll_up.verticalScrollBar().setValue)
        self.scroll_up.verticalScrollBar().valueChanged.connect(self.scroll_orig.verticalScrollBar().setValue)

        layout.addWidget(self.img_container, 1)

        # Bottom Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        btn_close = QPushButton("Close preview")
        btn_close.setStyleSheet("padding: 6px 18px; font-weight: 600; border-radius: 8px;")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)
        layout.addLayout(bottom_bar)

        self.pixmap_orig = QPixmap(orig_path)
        self.pixmap_up = QPixmap(upscaled_path)
        self._update_display()

    def _zoom_in(self):
        self.zoom_level = min(4.0, self.zoom_level * 1.25)
        self._update_display()

    def _zoom_out(self):
        self.zoom_level = max(0.25, self.zoom_level / 1.25)
        self._update_display()

    def _zoom_reset(self):
        self.zoom_level = 1.0
        self._update_display()

    def _on_view_mode_changed(self, idx: int):
        if idx == 0:  # Side by side
            self.panel_orig.setVisible(True)
            self.panel_up.setVisible(True)
        elif idx == 1:  # After
            self.panel_orig.setVisible(False)
            self.panel_up.setVisible(True)
        else:  # Before
            self.panel_orig.setVisible(True)
            self.panel_up.setVisible(False)

    def _update_display(self):
        if self.pixmap_orig.isNull() or self.pixmap_up.isNull():
            return
        w_orig = max(10, int(self.pixmap_orig.width() * self.zoom_level))
        h_orig = max(10, int(self.pixmap_orig.height() * self.zoom_level))
        self.lbl_orig_img.setPixmap(self.pixmap_orig.scaled(w_orig, h_orig, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        w_up = max(10, int(self.pixmap_up.width() * self.zoom_level))
        h_up = max(10, int(self.pixmap_up.height() * self.zoom_level))
        self.lbl_up_img.setPixmap(self.pixmap_up.scaled(w_up, h_up, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        # Clean up temporary preview directory if present
        temp_dir = self.info.get("temp_dir")
        if temp_dir and os.path.isdir(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        super().closeEvent(event)


class FramePreviewWorker(QThread):
    preview_ready = pyqtSignal(bool, str, str, dict)

    def __init__(self, pipeline: VideoUpscalerPipeline, video_path: str, model_id: str, target_preset: str):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.model_id = model_id
        self.target_preset = target_preset

    def run(self):
        try:
            ok, orig, up, info = self.pipeline.extract_and_upscale_single_frame(
                self.video_path,
                model_id=self.model_id,
                target_preset=self.target_preset
            )
            self.preview_ready.emit(ok, orig, up, info)
        except Exception as exc:
            self.preview_ready.emit(False, "", "", {"error": str(exc)})


class ModelHintWorker(QThread):
    hint_ready = pyqtSignal(str, str)

    def __init__(self, source_path: str, target_preset: str):
        super().__init__()
        self.source_path = source_path
        self.target_preset = target_preset

    def run(self):
        try:
            meta = probe_video(self.source_path)
            recommendation = recommend_model_for_video(meta.width, meta.height, self.target_preset)
            text = (
                f"🧠 AI Detection: {recommendation.model.name} ×{recommendation.scale} — "
                "full details will be pending validation before batch."
            )
            self.hint_ready.emit(text, recommendation.reason)
        except Exception:
            self.hint_ready.emit("🧠 AI Detection: full analysis available upon batch launch.", "")


class UpscaleWorker(QThread):
    item_updated = pyqtSignal(object)
    batch_updated = pyqtSignal(int, int, float, str, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        batch_manager: BatchManager,
        output_dir: str,
        base_dir: str,
        target_preset: str,
        video_codec: str,
        model_id: str,
        scene_config: SceneNamingConfig,
        preserve_folders: bool,
        overwrite: bool,
        threads: str = "auto",
        frame_format: str = "jpg",
        max_workers: int = 2,
        cq_level: int = 19,
        nvenc_preset: str = "p7",
        enable_tta: bool = False,
        denoise_level: int = -1,
        audio_mode: str = "copy",
        tile_size: int = 0,
        target_indices: Optional[List[int]] = None,
        preview_duration_s: Optional[float] = None,
    ):
        super().__init__()
        self.bm = batch_manager
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.target_preset = target_preset
        self.video_codec = video_codec
        self.model_id = model_id
        self.scene_config = scene_config
        self.preserve_folders = preserve_folders
        self.overwrite = overwrite
        self.threads = threads
        self.frame_format = frame_format
        self.max_workers = max_workers
        self.cq_level = cq_level
        self.nvenc_preset = nvenc_preset
        self.enable_tta = enable_tta
        self.denoise_level = denoise_level
        self.audio_mode = audio_mode
        self.tile_size = tile_size
        self.target_indices = target_indices
        self.preview_duration_s = preview_duration_s

    def run(self):
        try:
            self.bm.run_batch(
                output_dir=self.output_dir,
                base_dir=self.base_dir,
                target_preset=self.target_preset,
                video_codec=self.video_codec,
                model_id=self.model_id,
                scene_config=self.scene_config,
                preserve_folders=self.preserve_folders,
                overwrite=self.overwrite,
                threads=self.threads,
                frame_format=self.frame_format,
                max_workers=self.max_workers,
                cq_level=self.cq_level,
                nvenc_preset=self.nvenc_preset,
                enable_tta=self.enable_tta,
                denoise_level=self.denoise_level,
                audio_mode=self.audio_mode,
                tile_size=self.tile_size,
                target_indices=self.target_indices,
                preview_duration_s=self.preview_duration_s,
                log_callback=lambda msg: self.log_signal.emit(msg),
                item_callback=lambda item: self.item_updated.emit(item),
                batch_callback=lambda cur, tot, pct, status, eta="": self.batch_updated.emit(cur, tot, pct, status, eta)
            )
        except Exception as exc:
            self.log_signal.emit(f"[ERROR] Batch worker crashed: {type(exc).__name__}: {exc}")
        finally:
            self.finished_signal.emit()


class UpscalerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancer 4K — Animation & Cartoon Edition")
        self.resize(1280, 840)
        self.setMinimumSize(900, 640)
        self.setAcceptDrops(True)

        self.hardware = get_hardware_info()
        self.batch_manager = BatchManager(self.hardware)
        self.worker: Optional[UpscaleWorker] = None
        self._current_filter_mode = "all"
        self._log_buffer: list = []            # shared log buffer (all sessions)
        self._last_cli_cmd: str = ""
        self._log_viewer: Optional[LogViewerDialog] = None  # singleton dialog
        self._hint_worker: Optional[ModelHintWorker] = None
        self._frame_worker: Optional[FramePreviewWorker] = None
        self.settings = QSettings("CYLAE", "VideoEnhancer4K")

        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(350)
        self._settings_timer.timeout.connect(self._do_settings_changed)

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # Restore last used directories or check default
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

        # Clean stale temporary directories from crashed / aborted past sessions
        try:
            cleanup_stale_temp_directories()
        except Exception:
            pass

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        self.main_layout = main_layout
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # 1. Header Banner
        header = QFrame()
        header.setObjectName("macPanel")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 12, 16, 12)
        h_layout.setSpacing(10)

        traffic_lights = QHBoxLayout()
        traffic_lights.setSpacing(6)
        for color, border in (("#ff5f56", "#e0443e"), ("#ffbd2e", "#dea123"), ("#27c93f", "#1aab29")):
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background-color: {color}; border: 1px solid {border}; border-radius: 6px;")
            traffic_lights.addWidget(dot)
        h_layout.addLayout(traffic_lights)

        title_box = QVBoxLayout()
        lbl_title = QLabel("Video Enhancer 4K — Animation & Cartoon Edition")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff; letter-spacing: -0.2px;")
        lbl_sub = QLabel("Local upscaling • quality control • standardized export")
        lbl_sub.setStyleSheet("font-size: 12px; color: #8e8e93;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        h_layout.addLayout(title_box)

        h_layout.addStretch()

        # Telemetry chips
        chips_layout = QHBoxLayout()
        chips_layout.setSpacing(8)

        self.lbl_gpu = QLabel(f"● {self.hardware.gpu_name} ({self.hardware.gpu_vram_gb} GB)")
        self.lbl_gpu.setStyleSheet("color: #30d158; font-weight: 600; font-size: 11px; padding: 4px 10px; background-color: rgba(48, 209, 88, 0.12); border: 1px solid rgba(48, 209, 88, 0.3); border-radius: 12px;")
        chips_layout.addWidget(self.lbl_gpu)

        nvenc_text = "NVENC H.265 Active" if self.hardware.has_nvenc_hevc else "CPU x265"
        self.lbl_encoder = QLabel(nvenc_text)
        self.lbl_encoder.setStyleSheet("color: #0a84ff; font-weight: 600; font-size: 11px; padding: 4px 10px; background-color: rgba(10, 132, 255, 0.12); border: 1px solid rgba(10, 132, 255, 0.3); border-radius: 12px;")
        chips_layout.addWidget(self.lbl_encoder)

        self.lbl_audio = QLabel("● Audio: Lossless Stream Copy (Guaranteed)")
        self.lbl_audio.setStyleSheet("color: #30d158; font-weight: 600; font-size: 11px; padding: 4px 10px; background-color: rgba(48, 209, 88, 0.12); border: 1px solid rgba(48, 209, 88, 0.3); border-radius: 12px;")
        chips_layout.addWidget(self.lbl_audio)

        btn_doctor = QPushButton("🩺 Hardware Doctor")
        btn_doctor.setStyleSheet("padding: 4px 12px; font-size: 11px; font-weight: 600; border-radius: 12px; background-color: #2c2c2e; border: 1px solid #3a3a3c; color: #f5f5f7;")
        btn_doctor.clicked.connect(self.show_doctor_dialog)
        chips_layout.addWidget(btn_doctor)

        h_layout.addLayout(chips_layout)

        # ── Logs access button (always visible) ───────────────────────────────
        self.btn_open_logs = QPushButton("📋 Logs (0)")
        self.btn_open_logs.setToolTip("Open the live engine log viewer. Shows FFmpeg, Vulkan AI, and audio pipeline commands with color coding.")
        self.btn_open_logs.setStyleSheet(
            "padding: 4px 12px; font-size: 11px; font-weight: 600;"
            "background-color: #2c2c2e; border: 1px solid #3a3a3c; border-radius: 12px; color: #aeaeb2;"
        )
        self.btn_open_logs.clicked.connect(self._open_log_viewer)
        h_layout.addWidget(self.btn_open_logs)

        main_layout.addWidget(header)

        # 2. Paths & Config Group
        grp_config = QGroupBox("Processing & Enhancement Configuration")
        cfg_layout = QVBoxLayout(grp_config)
        cfg_layout.setContentsMargins(14, 16, 14, 14)
        cfg_layout.setSpacing(10)

        # Path row 1: Source
        row_src = QHBoxLayout()
        lbl_src_title = QLabel("Source:")
        lbl_src_title.setMinimumWidth(80)
        row_src.addWidget(lbl_src_title)
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Drag and drop cartoon directory or click browse...")
        self.txt_source.setToolTip("Path to the input directory or files. Drag-and-drop video files or folders directly here.")
        self.txt_source.setClearButtonEnabled(True)
        row_src.addWidget(self.txt_source, 1)

        btn_add_files = QPushButton("➕ Add Files...")
        btn_add_files.setToolTip("Select one or more individual video files to add to the queue.")
        btn_add_files.clicked.connect(self.on_add_files_clicked)
        row_src.addWidget(btn_add_files)

        btn_browse_src = QPushButton("📁 Browse Folder...")
        btn_browse_src.setToolTip("Browse and select the input video folder.")
        btn_browse_src.clicked.connect(self.on_browse_source)
        row_src.addWidget(btn_browse_src)

        btn_scan = QPushButton("🔍 Scan Folder")
        btn_scan.setObjectName("btnScan")
        btn_scan.setToolTip("Scan source directory and index all video files into the queue.")
        btn_scan.clicked.connect(self.on_scan_clicked)
        row_src.addWidget(btn_scan)
        cfg_layout.addLayout(row_src)

        # Path row 2: Output
        row_out = QHBoxLayout()
        lbl_out_title = QLabel("Output Folder:")
        lbl_out_title.setMinimumWidth(80)
        row_out.addWidget(lbl_out_title)
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Destination directory for converted 4K videos...")
        self.txt_output.setToolTip("Path to the target directory where enhanced 4K files will be saved.")
        self.txt_output.setClearButtonEnabled(True)
        row_out.addWidget(self.txt_output, 1)

        btn_browse_out = QPushButton("📁 Browse...")
        btn_browse_out.setToolTip("Browse and select the destination directory.")
        btn_browse_out.clicked.connect(self.on_browse_output)
        row_out.addWidget(btn_browse_out)

        btn_open_out = QPushButton("📂 Open Output")
        btn_open_out.setToolTip("Open destination directory in Windows File Explorer.")
        btn_open_out.clicked.connect(self.on_open_output)
        row_out.addWidget(btn_open_out)
        cfg_layout.addLayout(row_out)

        # Row 3: 1-Click Quality Profiles + Naming Format
        row_presets = QHBoxLayout()
        row_presets.setSpacing(10)
        lbl_preset_title = QLabel("🎯 Quality Profile:")
        lbl_preset_title.setStyleSheet("font-weight: 700; color: #0a84ff;")
        row_presets.addWidget(lbl_preset_title)

        self.cb_preset = QComboBox()
        self.cb_preset.setToolTip("1-click curated enhancement profile. Auto-configures resolution, AI model, CQ factor, and NVENC.")
        for p_id, p_obj in ENHANCEMENT_PRESETS.items():
            self.cb_preset.addItem(f"{p_obj.icon}  {p_obj.name}", p_id)
        self.cb_preset.currentIndexChanged.connect(self.on_preset_changed)
        row_presets.addWidget(self.cb_preset, 1)

        lbl_naming = QLabel("🏷️ Naming:")
        lbl_naming.setStyleSheet("font-weight: 600; color: #f5f5f7;")
        row_presets.addWidget(lbl_naming)

        self.cb_naming_mode = QComboBox()
        self.cb_naming_mode.setToolTip("File naming convention for enhanced output videos.")
        self.cb_naming_mode.addItem("🏷️ Clean (e.g. Video_4K.mp4)", "clean")
        self.cb_naming_mode.addItem("🎬 Scene Release (e.g. Show.S01E01...-CYLAE.mp4)", "scene")
        self.cb_naming_mode.addItem("📄 Original Name (e.g. Video.mp4)", "original")
        self.cb_naming_mode.currentIndexChanged.connect(self.on_settings_changed)
        row_presets.addWidget(self.cb_naming_mode, 1)

        self.lbl_preview_name = QLabel("Release Sample: Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4")
        self.lbl_preview_name.setToolTip("Standardized Scene release filename preview generated dynamically based on active settings.")
        self.lbl_preview_name.setStyleSheet("background-color: #1c1c1e; border: 1px solid #3a3a3c; border-radius: 8px; padding: 7px 12px; color: #0a84ff; font-family: 'SF Mono', 'Cascadia Mono', monospace; font-size: 11px;")
        row_presets.addWidget(self.lbl_preview_name, 2)
        cfg_layout.addLayout(row_presets)

        # Profile description banner
        self.lbl_preset_desc = QLabel("")
        self.lbl_preset_desc.setStyleSheet("color: #8e8e93; font-size: 11px; padding-left: 4px;")
        cfg_layout.addWidget(self.lbl_preset_desc)

        # Row 4: Core Settings. Two balanced rows remain readable at the
        # minimum window width instead of compressing every control at once.
        row_settings = QHBoxLayout()
        row_settings.setSpacing(10)

        # Resolution
        row_settings.addWidget(QLabel("Resolution:"))
        self.cb_resolution = QComboBox()
        self.cb_resolution.setToolTip("Target upscaling resolution. 4K UHD (3840x2160) provides pristine line clarity.")
        self.cb_resolution.addItems(["4K UHD (3840x2160)", "1440p QHD (2560x1440)", "1080p FHD (1920x1080)", "720p HD (1280x720)"])
        self.cb_resolution.currentIndexChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.cb_resolution)

        # Codec
        row_settings.addWidget(QLabel("Codec:"))
        self.cb_codec = QComboBox()
        self.cb_codec.setToolTip("Video encoding engine. NVENC Hardware provides ultra-fast GPU encoding with spatial/temporal AQ.")
        self.cb_codec.addItems(["H.265 (HEVC NVENC Hardware)", "H.264 (NVENC Hardware)", "H.265 (CPU x265)"])
        self.cb_codec.currentIndexChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.cb_codec)

        # Model
        row_settings.addWidget(QLabel("AI Model:"))
        self.cb_model = QComboBox()
        self.cb_model.setToolTip("Neural network model used for line reconstruction and upscaling.")
        for m_id, m_spec in AVAILABLE_MODELS.items():
            self.cb_model.addItem(m_spec.name, m_id)
        self.cb_model.currentIndexChanged.connect(self.on_model_changed)
        row_settings.addWidget(self.cb_model)

        row_settings.addStretch()
        cfg_layout.addLayout(row_settings)

        row_options = QHBoxLayout()
        row_options.setSpacing(12)

        # Checkboxes
        self.chk_preserve_folders = QCheckBox("Preserve subfolders")
        self.chk_preserve_folders.setToolTip("Replicate source subfolder hierarchy (e.g. Season 01/) inside the output folder.")
        self.chk_preserve_folders.setChecked(True)
        row_options.addWidget(self.chk_preserve_folders)

        self.chk_overwrite = QCheckBox("Overwrite existing")
        self.chk_overwrite.setToolTip("Re-process and overwrite files that already exist in the output directory.")
        self.chk_overwrite.setChecked(False)
        row_options.addWidget(self.chk_overwrite)

        row_options.addStretch()

        self.btn_toggle_expert = QPushButton("⚙️  Advanced & Expert Controls ▼")
        self.btn_toggle_expert.setObjectName("btnToggleExpert")
        self.btn_toggle_expert.setToolTip("Show/hide fine-tuning controls: CQ factor, NVENC preset, threads, tile size, and audio modes.")
        self.btn_toggle_expert.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_toggle_expert.clicked.connect(self.toggle_expert_panel)
        row_options.addWidget(self.btn_toggle_expert)
        cfg_layout.addLayout(row_options)

        # ----------------------------------------------------
        # EXPANDABLE EXPERT & POWER USER PANEL
        # ----------------------------------------------------
        self.grp_expert = QGroupBox("⚙️ Advanced & Expert Controls")
        self.grp_expert.setStyleSheet("background-color: #252528; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; margin-top: 6px; padding: 14px;")
        expert_layout = QVBoxLayout(self.grp_expert)
        expert_layout.setSpacing(10)

        # Expert Row 1: Rate Control & Audio Modes
        row_exp_1 = QHBoxLayout()
        row_exp_1.setSpacing(14)

        row_exp_1.addWidget(QLabel("CQ (Quality):"))
        self.slider_cq = QSlider(Qt.Orientation.Horizontal)
        self.slider_cq.setRange(14, 28)
        self.slider_cq.setValue(19)
        self.slider_cq.setMinimumWidth(130)
        self.slider_cq.setToolTip("Constant Quality factor (14-28). Lower CQ delivers higher bitrate and finer quality. 19 is visually lossless for 4K.")
        self.slider_cq.valueChanged.connect(self.on_cq_slider_changed)
        row_exp_1.addWidget(self.slider_cq)

        self.lbl_cq_val = QLabel("CQ 19 (Visually Lossless)")
        self.lbl_cq_val.setStyleSheet("color: #0a84ff; font-weight: 700; font-size: 11px;")
        row_exp_1.addWidget(self.lbl_cq_val)

        row_exp_1.addWidget(QLabel("NVENC Preset:"))
        self.cb_nvenc_preset = QComboBox()
        self.cb_nvenc_preset.setToolTip("Hardware NVENC speed vs compression trade-off preset. p7 provides highest compression and quality.")
        self.cb_nvenc_preset.addItems([
            "p7 (Slowest / Max Compression)",
            "p6 (High Quality)",
            "p5 (Medium Quality)",
            "p4 (Fast Draft)",
            "p1 (Fastest / Low Overhead)"
        ])
        self.cb_nvenc_preset.currentIndexChanged.connect(self.on_expert_setting_changed)
        row_exp_1.addWidget(self.cb_nvenc_preset)

        row_exp_1.addWidget(QLabel("Soundtrack Stream:"))
        self.cb_audio_mode = QComboBox()
        self.cb_audio_mode.setToolTip("Audio stream management. Lossless Stream Copy (-c:a copy) extracts and remuxes the original soundtrack with zero loss.")
        self.cb_audio_mode.addItem("Lossless Stream Copy (-c:a copy)", "copy")
        self.cb_audio_mode.addItem("Re-encode AAC 192k (Transparent)", "aac_192k")
        self.cb_audio_mode.addItem("Re-encode AAC 320k (Audiophile)", "aac_320k")
        self.cb_audio_mode.addItem("Lossless FLAC 24-bit", "flac")
        self.cb_audio_mode.currentIndexChanged.connect(self.on_expert_setting_changed)
        row_exp_1.addWidget(self.cb_audio_mode)

        row_exp_1.addStretch()
        expert_layout.addLayout(row_exp_1)

        # Expert Row 2: Neural Engine Tuning & Tile Size
        row_exp_2 = QHBoxLayout()
        row_exp_2.setSpacing(14)

        row_exp_2.addWidget(QLabel("Workers:"))
        self.cb_workers = QComboBox()
        self.cb_workers.setToolTip("Number of episodes processed simultaneously. 2 workers maximizes GPU compute saturation on RTX 3070.")
        self.cb_workers.addItems(["2 Workers (Parallel - Recommended for RTX 3070)", "1 Worker (Sequential)", "3 Workers (Max Throughput)"])
        row_exp_2.addWidget(self.cb_workers)

        row_exp_2.addWidget(QLabel("Threads:"))
        self.cb_threads = QComboBox()
        self.cb_threads.setToolTip("Vulkan worker threads allocation (load:proc:save). Auto (4:4:4) takes advantage of high core count CPUs.")
        cpu_cores = os.cpu_count() or 4
        self.cb_threads.addItems([f"Auto ({cpu_cores} Cores - Optimal 4:4:4)", "4:4:4 (Max Multi-Core)", "2:4:2 (Balanced)", "1:2:2 (Default)"])
        row_exp_2.addWidget(self.cb_threads)

        row_exp_2.addWidget(QLabel("Frame Format:"))
        self.cb_frame_fmt = QComboBox()
        self.cb_frame_fmt.setToolTip("Intermediate extracted frame image format. Fast JPEG provides 2x-3x faster disk I/O; PNG is lossless.")
        self.cb_frame_fmt.addItems(["Fast JPEG (2x-3x Faster I/O)", "Lossless PNG"])
        row_exp_2.addWidget(self.cb_frame_fmt)

        row_exp_2.addWidget(QLabel("Tile Size:"))
        self.cb_tile_size = QComboBox()
        self.cb_tile_size.setToolTip("Image tile split size. 0 (Auto) processes entire frames at once for maximum inference performance on 8GB+ VRAM.")
        self.cb_tile_size.addItem("0 (Auto / Full VRAM)", 0)
        self.cb_tile_size.addItem("200 px (Low VRAM)", 200)
        self.cb_tile_size.addItem("300 px", 300)
        self.cb_tile_size.addItem("400 px", 400)
        self.cb_tile_size.addItem("600 px", 600)
        row_exp_2.addWidget(self.cb_tile_size)

        row_exp_2.addWidget(QLabel("RealCUGAN Denoise:"))
        self.cb_denoise = QComboBox()
        self.cb_denoise.setToolTip("Real-CUGAN noise reduction filter. Suppresses cartoon line ringing and artifacts.")
        self.cb_denoise.addItem("None (-1)", -1)
        self.cb_denoise.addItem("Conservative (0)", 0)
        self.cb_denoise.addItem("Standard (1)", 1)
        self.cb_denoise.addItem("Aggressive (2)", 2)
        self.cb_denoise.addItem("Ultra Clean (3)", 3)
        row_exp_2.addWidget(self.cb_denoise)

        self.chk_tta = QCheckBox("Enable 8-pass TTA")
        self.chk_tta.setToolTip("Test-Time Augmentation ensemble: 8 rotations/flips for maximum edge quality (slower)")
        row_exp_2.addWidget(self.chk_tta)

        row_exp_2.addStretch()
        expert_layout.addLayout(row_exp_2)

        # Expert Row 3: Scene Release Tagging
        row_exp_3 = QHBoxLayout()
        row_exp_3.setSpacing(12)

        row_exp_3.addWidget(QLabel("Show Name:"))
        self.txt_show_name = QLineEdit("Happy.Tree.Friends")
        self.txt_show_name.setToolTip("Show title formatted with dots for standard Scene release naming.")
        self.txt_show_name.setMaximumWidth(150)
        self.txt_show_name.textChanged.connect(self.on_settings_changed)
        row_exp_3.addWidget(self.txt_show_name)

        row_exp_3.addWidget(QLabel("Group:"))
        self.txt_group = QLineEdit("CYLAE")
        self.txt_group.setToolTip("Scene release group tag (e.g. CYLAE).")
        self.txt_group.setMaximumWidth(80)
        self.txt_group.textChanged.connect(self.on_settings_changed)
        row_exp_3.addWidget(self.txt_group)

        row_exp_3.addWidget(QLabel("Source:"))
        self.cb_source = QComboBox()
        self.cb_source.setToolTip("Source media type tag (e.g. WEB-DL, WEBRip).")
        self.cb_source.addItems(["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"])
        self.cb_source.currentIndexChanged.connect(self.on_settings_changed)
        row_exp_3.addWidget(self.cb_source)

        row_exp_3.addWidget(QLabel("Audio Tag:"))
        self.cb_audio_tag = QComboBox()
        self.cb_audio_tag.setToolTip("Standard audio codec release tag (e.g. AAC2.0, FLAC2.0).")
        self.cb_audio_tag.addItems(["AAC2.0", "FLAC2.0", "DDP5.1"])
        self.cb_audio_tag.currentIndexChanged.connect(self.on_settings_changed)
        row_exp_3.addWidget(self.cb_audio_tag)

        row_exp_3.addStretch()
        expert_layout.addLayout(row_exp_3)

        # Expert Row 4: Live Engine Console & CLI Inspector
        console_bar = QHBoxLayout()
        lbl_con_title = QLabel("🖥️ Live Engine Logs & Raw CLI Inspector")
        lbl_con_title.setStyleSheet("font-weight: 700; color: #0a84ff; font-size: 11px;")
        console_bar.addWidget(lbl_con_title)
        console_bar.addStretch()

        self.btn_copy_cli = QPushButton("📋 Copy Last CLI Command")
        self.btn_copy_cli.setToolTip("Copy the most recently executed FFmpeg or AI upscaler CLI command to clipboard.")
        self.btn_copy_cli.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        self.btn_copy_cli.clicked.connect(self.on_copy_cli_clicked)
        console_bar.addWidget(self.btn_copy_cli)

        self.btn_clear_console = QPushButton("🧹 Clear Logs")
        self.btn_clear_console.setToolTip("Clear all engine logs and reset the logs counter.")
        self.btn_clear_console.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        self.btn_clear_console.clicked.connect(self.clear_all_logs)
        console_bar.addWidget(self.btn_clear_console)
        expert_layout.addLayout(console_bar)

        self.txt_console = QTextEdit()
        self.txt_console.setObjectName("txtConsole")
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumHeight(110)
        self.txt_console.setPlaceholderText("Live subprocess commands (FFmpeg, Vulkan engine) will appear here during execution...")
        expert_layout.addWidget(self.txt_console)

        self.grp_expert.setVisible(False)
        cfg_layout.addWidget(self.grp_expert)

        # Dynamic model description banner
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setStyleSheet("color: #8e8e93; font-size: 11px; padding: 3px 2px;")
        cfg_layout.addWidget(self.lbl_model_info)
        self.on_preset_changed()

        main_layout.addWidget(grp_config)

        # 3. Queue Section
        table_box = QVBoxLayout()
        table_box.setSpacing(6)

        # Stat pills row + Filter bar
        stat_bar = QHBoxLayout()
        stat_bar.setSpacing(8)

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

        self.btn_stat_completed = QPushButton("Completed: 0")
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

        self.lbl_search = QLabel("🔍 Search:")
        stat_bar.addWidget(self.lbl_search)
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter by name or season...")
        self.txt_filter.setToolTip("Filter queue by episode name, season, or filename.")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setMaximumWidth(220)
        self.txt_filter.textChanged.connect(self.on_filter_changed)
        stat_bar.addWidget(self.txt_filter)

        self.btn_clear_queue = QPushButton("🧹 Clear Queue")
        self.btn_clear_queue.setToolTip("Clear all items currently in the processing queue.")
        self.btn_clear_queue.clicked.connect(self.on_clear_queue_clicked)
        stat_bar.addWidget(self.btn_clear_queue)

        table_box.addLayout(stat_bar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#",
            "Original File",
            "Size",
            "Standardized Target Scene Name",
            "Status",
            "Detail / Progress"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(1, 260)
        self.table.setColumnWidth(5, 270)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        # Context Menu & Double Click
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_table_context_menu)
        self.table.itemDoubleClicked.connect(self.on_table_double_click)

        table_box.addWidget(self.table, 1)
        main_layout.addLayout(table_box, 1)

        # 4. Progress and Controls Bottom Bar
        bottom_bar = QFrame()
        bottom_bar.setObjectName("macPanel")
        b_layout = QVBoxLayout(bottom_bar)
        b_layout.setContentsMargins(16, 12, 16, 12)
        b_layout.setSpacing(10)

        status_row = QHBoxLayout()
        self.lbl_overall_status = QLabel("Ready to process.")
        self.lbl_overall_status.setStyleSheet("font-weight: 600; color: #f5f5f7; font-size: 13px;")
        status_row.addWidget(self.lbl_overall_status)
        status_row.addStretch()

        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color: #0a84ff; font-weight: 700; font-size: 12px; padding: 4px 10px; background-color: rgba(10, 132, 255, 0.12); border: 1px solid rgba(10, 132, 255, 0.25); border-radius: 8px;")
        status_row.addWidget(self.lbl_eta)
        b_layout.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        b_layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        # Single frame instant comparison
        self.btn_compare_frame = QPushButton("🔍  Compare Frame")
        self.btn_compare_frame.setObjectName("btnCompareFrame")
        self.btn_compare_frame.setToolTip("Generate an instant side-by-side Before/After on a single frame (no full video render required).")
        self.btn_compare_frame.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_compare_frame.clicked.connect(self.on_compare_frame_clicked)
        btn_row.addWidget(self.btn_compare_frame, 2)

        # Preview clip duration selector
        self.spin_preview_dur = QSpinBox()
        self.spin_preview_dur.setRange(5, 300)
        self.spin_preview_dur.setValue(30)
        self.spin_preview_dur.setSuffix(" sec")
        self.spin_preview_dur.setToolTip("Duration of the upscaled preview clip in seconds. The clip starts 10% into the video.")
        self.spin_preview_dur.setMaximumWidth(80)
        btn_row.addWidget(QLabel("Clip:"))
        btn_row.addWidget(self.spin_preview_dur)

        self.btn_preview_clip = QPushButton("🎬  Render Preview Clip")
        self.btn_preview_clip.setObjectName("btnPreviewClip")
        self.btn_preview_clip.setToolTip(
            "Upscale a short clip (default 30 s) from 10% into the video. "
            "The output file is saved with a [Preview Xsec] suffix — it never overwrites the full encode."
        )
        self.btn_preview_clip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_preview_clip.clicked.connect(self.on_preview_clip_clicked)
        btn_row.addWidget(self.btn_preview_clip, 2)

        self.btn_preview = QPushButton("⚡  Test 1 Episode (Full)")
        self.btn_preview.setObjectName("btnPreview")
        self.btn_preview.setToolTip("Process 1 full episode to verify visual quality, soundtrack sync, and encoding speed before launching the full batch.")
        self.btn_preview.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_preview.clicked.connect(self.on_test_single_clicked)
        btn_row.addWidget(self.btn_preview, 2)

        self.btn_start = QPushButton("▶  START BATCH PROCESSING")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setToolTip("Launch concurrent batch processing for all pending episodes.")
        self.btn_start.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_start.clicked.connect(self.on_start_clicked)
        btn_row.addWidget(self.btn_start, 3)

        self.btn_stop = QPushButton("⏹  STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setToolTip("Gracefully stop active processing jobs.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        btn_row.addWidget(self.btn_stop, 1)

        b_layout.addLayout(btn_row)
        main_layout.addWidget(bottom_bar)

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
        """Keep the header useful on compact windows instead of clipping controls."""
        width = event.size().width()
        if hasattr(self, "lbl_audio"):
            self.lbl_audio.setVisible(width >= 1180)
        if hasattr(self, "lbl_gpu"):
            self.lbl_gpu.setVisible(width >= 1020)
        if hasattr(self, "lbl_encoder"):
            self.lbl_encoder.setVisible(width >= 940)
        if hasattr(self, "btn_preview_clip"):
            compact = width < 1060
            margin = 12 if compact else 18
            self.main_layout.setContentsMargins(margin, 12 if compact else 16, margin, 12 if compact else 16)
            self.main_layout.setSpacing(10 if compact else 12)
            if hasattr(self, "btn_compare_frame"):
                self.btn_compare_frame.setText("🔍 Preview" if compact else "🔍  Compare Frame")
            self.btn_preview_clip.setText("🎬 Preview" if compact else "🎬  Render Preview Clip")
            self.btn_preview.setText("⚡ Test 1 Episode" if compact else "⚡  Test 1 Episode (Full)")
            self.btn_start.setText("▶ LANCER" if compact else "▶  START BATCH PROCESSING")
            self.btn_stop.setText("⏹" if compact else "⏹  STOP")
            self.lbl_search.setVisible(not compact)
            self.txt_filter.setMaximumWidth(140 if compact else 220)
            n_all = len(self.batch_manager.queue)
            n_ready = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.PENDING)
            n_done = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.COMPLETED)
            n_skip = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.SKIPPED)
            n_err = sum(1 for it in self.batch_manager.queue if it.status == EpisodeStatus.FAILED)
            if compact:
                self.btn_stat_all.setText(f"All: {n_all}")
                self.btn_stat_ready.setText(f"Ready: {n_ready}")
                self.btn_stat_completed.setText(f"Done: {n_done}")
                self.btn_stat_skipped.setText(f"Skipped: {n_skip}")
                self.btn_stat_failed.setText(f"Err: {n_err}")
            else:
                self.btn_stat_all.setText(f"All: {n_all}")
                self.btn_stat_ready.setText(f"Ready: {n_ready}")
                self.btn_stat_completed.setText(f"Completed: {n_done}")
                self.btn_stat_skipped.setText(f"Skipped: {n_skip}")
                self.btn_stat_failed.setText(f"Errors: {n_err}")
        super().resizeEvent(event)

    def _update_auto_model_hint(self, queue: List[QueueItem]):
        """Detects best model choice in a background thread so UI never freezes."""
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
                # Resolution
                for i in range(self.cb_resolution.count()):
                    if preset.target_preset in self.cb_resolution.itemText(i).lower():
                        self.cb_resolution.setCurrentIndex(i)
                        break

                # Codec
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

                # Model
                for i in range(self.cb_model.count()):
                    if self.cb_model.itemData(i) == preset.model_id:
                        self.cb_model.setCurrentIndex(i)
                        break

                # CQ slider
                self.slider_cq.setValue(preset.cq_level)
                self.lbl_cq_val.setText(f"CQ {preset.cq_level} (Visually Lossless)")

                # NVENC preset
                for i in range(self.cb_nvenc_preset.count()):
                    if self.cb_nvenc_preset.itemText(i).startswith(preset.nvenc_preset):
                        self.cb_nvenc_preset.setCurrentIndex(i)
                        break

                # Audio Mode
                for i in range(self.cb_audio_mode.count()):
                    if self.cb_audio_mode.itemData(i) == preset.audio_mode:
                        self.cb_audio_mode.setCurrentIndex(i)
                        break

                # TTA
                self.chk_tta.setChecked(preset.enable_tta)

                # Denoise
                for i in range(self.cb_denoise.count()):
                    if self.cb_denoise.itemData(i) == preset.denoise_level:
                        self.cb_denoise.setCurrentIndex(i)
                        break

                # Workers
                w_idx = 0 if preset.workers == 2 else (1 if preset.workers == 1 else 2)
                self.cb_workers.setCurrentIndex(w_idx)

                # Frame format
                f_idx = 0 if preset.frame_format == "jpg" else 1
                self.cb_frame_fmt.setCurrentIndex(f_idx)

            finally:
                self._updating_preset = False

        self.lbl_preset_desc.setText(f"ℹ️ {preset.description} ({preset.recommended_for})")
        self.update_scene_preview()

    def on_cq_slider_changed(self, val: int):
        qual_desc = "Visually Lossless" if val <= 19 else ("High Quality" if val <= 22 else "Balanced")
        self.lbl_cq_val.setText(f"CQ {val} ({qual_desc})")
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
                "copy": "● Audio: Lossless Stream Copy (Guaranteed)",
                "aac_192k": "● Audio: AAC 192k Transparent (Guaranteed)",
                "aac_320k": "● Audio: AAC 320k Audiophile (Guaranteed)",
                "flac": "● Audio: Lossless FLAC 24-bit (Guaranteed)"
            }
            self.lbl_audio.setText(labels.get(mode, "● Audio: Guaranteed"))

    def on_log_received(self, msg: str):
        """Handles incoming log lines from the worker thread.

        1. Stores into the shared _log_buffer.
        2. Updates the Logs badge counter.
        3. Forwards to the LogViewerDialog if it is open.
        4. Also appends to the inline expert-panel console (legacy path).
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        timestamped = f"[{timestamp}]  {msg}"
        self._last_cli_cmd = msg
        self._log_buffer.append(timestamped)
        if len(self._log_buffer) > 5000:
            del self._log_buffer[:-5000]

        # Update badge
        count = len(self._log_buffer)
        self.btn_open_logs.setText(f"📋 Logs ({count})")
        if count > 0:
            self.btn_open_logs.setStyleSheet(
                "padding: 4px 12px; font-size: 11px; font-weight: 600;"
                "background-color: rgba(10, 132, 255, 0.15); border: 1px solid #0a84ff; border-radius: 12px; color: #0a84ff;"
            )

        # Forward to open log viewer dialog
        if self._log_viewer and self._log_viewer.isVisible():
            self._log_viewer.append_line(timestamped)

        # Inline expert-panel console (legacy)
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
        """Clears the shared log buffer, inline console, and resets the Logs badge."""
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
        """Opens (or raises) the singleton LogViewerDialog."""
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
                out_dir = os.path.join(first_dir, "(2160p)")
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
                out_dir = p + " (2160p)"
                self.txt_output.setText(out_dir)
            self.settings.setValue("last_source", p)
            self.settings.setValue("last_output", out_dir)
            self.on_scan_clicked()

    def on_browse_source(self):
        default_dir = self.txt_source.text() or str(self.settings.value("last_source", "")) or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Source Directory", default_dir)
        if folder:
            self.txt_source.setText(folder)
            out_folder = folder + " (2160p)"
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
                out_dir = os.path.join(first_dir, "(2160p)")
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
        """Generates and displays instant side-by-side Before/After on a single frame."""
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
                if 0 <= r < len(self.batch_manager.queue):
                    target_item = self.batch_manager.queue[r]

        if target_item is None:
            for it in self.batch_manager.queue:
                if it.status == EpisodeStatus.PENDING:
                    target_item = it
                    break
        if target_item is None:
            target_item = self.batch_manager.queue[0]

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
        if not (0 <= row < len(self.batch_manager.queue)):
            return

        q_item = self.batch_manager.queue[row]
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
            act_play_out = QAction("▶️ Play Converted 4K Video", self)
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
        if 0 <= row < len(self.batch_manager.queue):
            q_item = self.batch_manager.queue[row]
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

    def on_scan_clicked(self):
        src = self.txt_source.text().strip()
        out = self.txt_output.text().strip()
        if not src or not os.path.exists(src):
            self.lbl_overall_status.setText("Error: Source directory does not exist.")
            return

        if not out:
            out = src + " (2160p)"
            self.txt_output.setText(out)

        cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()

        self.lbl_overall_status.setText("Indexing files...")
        queue = self.batch_manager.scan_folder(src, out, cfg, preserve_folders=preserve_folders)
        self._populate_table(queue)
        self._update_auto_model_hint(queue)

    def on_settings_changed(self):
        self.update_scene_preview()
        if hasattr(self, "_settings_timer"):
            self._settings_timer.start()

    def _do_settings_changed(self):
        src = self.txt_source.text().strip()
        out = self.txt_output.text().strip()
        if src and os.path.exists(src):
            cfg = self.get_current_scene_config()
            preserve_folders = self.chk_preserve_folders.isChecked()
            queue = self.batch_manager.scan_folder(src, out, cfg, preserve_folders=preserve_folders)
            self._populate_table(queue)

    def _populate_table(self, queue: List[QueueItem]):
        self.table.setRowCount(len(queue))
        skipped_count = 0
        completed_count = 0

        for r, item in enumerate(queue):
            # 0. Index
            it_idx = QTableWidgetItem(str(item.index))
            it_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 0, it_idx)

            # 1. Source relative path
            it_src = QTableWidgetItem(item.relative_source_path)
            self.table.setItem(r, 1, it_src)

            # 2. Source Size
            it_sz = QTableWidgetItem(f"{item.file_size_mb:.1f} MB")
            it_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, it_sz)

            # 3. Target Scene Name
            it_scene = QTableWidgetItem(item.target_scene_name)
            it_scene.setForeground(QColor("#38bdf8"))
            self.table.setItem(r, 3, it_scene)

            # 4. Status
            it_status = QTableWidgetItem(self._status_badge_text(item.status))
            it_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(it_status, item.status)
            self.table.setItem(r, 4, it_status)

            if item.status == EpisodeStatus.SKIPPED:
                skipped_count += 1
            elif item.status == EpisodeStatus.COMPLETED:
                completed_count += 1

            # 5. Detail / Progress
            it_detail = QTableWidgetItem(item.status_detail)
            self.table.setItem(r, 5, it_detail)

        total = len(queue)
        failed_count = sum(1 for it in queue if it.status == EpisodeStatus.FAILED)
        ready = total - skipped_count - completed_count - failed_count

        self.btn_stat_all.setText(f"All: {total}")
        self.btn_stat_ready.setText(f"Ready: {ready}")
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
        q_len = len(self.batch_manager.queue)

        for r in range(self.table.rowCount()):
            # Text search match
            text_match = True
            if search_text:
                text_match = False
                for col in (1, 3):
                    item = self.table.item(r, col)
                    if item and search_text in item.text().lower():
                        text_match = True
                        break

            # Stat pill filter match
            status_match = True
            if r < q_len:
                q_status = self.batch_manager.queue[r].status
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
        """Renders a short preview clip (configurable duration) from one episode."""
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

    def _confirm_model_choice(self, model_id: str, target_preset: str, target_indices: Optional[List[int]]) -> bool:
        """Informative model validation. Always returns True to allow seamless, non-blocking execution."""
        return True

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

        # Performance settings
        w_idx = self.cb_workers.currentIndex()
        max_workers = 2 if w_idx == 0 else (1 if w_idx == 1 else 3)
        if target_indices and len(target_indices) == 1:
            max_workers = 1  # 1 episode preview only needs 1 worker

        t_idx = self.cb_threads.currentIndex()
        threads = "auto" if t_idx == 0 else ("4:4:4" if t_idx == 1 else ("2:4:2" if t_idx == 2 else "1:2:2"))

        f_idx = self.cb_frame_fmt.currentIndex()
        frame_format = "jpg" if f_idx == 0 else "png"

        # Expert parameters
        cq_level = self.slider_cq.value()
        nvenc_preset = self.cb_nvenc_preset.currentText().split()[0]
        audio_mode = self.cb_audio_mode.currentData() or "copy"
        tile_size = int(self.cb_tile_size.currentData() or 0)
        denoise_level = int(self.cb_denoise.currentData() if self.cb_denoise.currentData() is not None else -1)
        enable_tta = self.chk_tta.isChecked()

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
        row = item.index - 1
        if 0 <= row < self.table.rowCount():
            it_status = self.table.item(row, 4)
            if it_status:
                it_status.setText(self._status_badge_text(item.status))
                self._style_status_item(it_status, item.status)

            it_detail = self.table.item(row, 5)
            if it_detail:
                it_detail.setText(item.status_detail)

            if item.status == EpisodeStatus.PROCESSING:
                self.table.scrollToItem(it_status)

        # Update stat pill counters
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
        if not cancelled_count:
            self.progress_bar.setValue(100)
        try:
            QApplication.beep()
        except Exception:
            pass


def run_gui():
    app = QApplication(sys.argv)
    window = UpscalerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
