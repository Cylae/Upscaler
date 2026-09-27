"""Live log viewer dialog with color-coding and export capabilities."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.ui.theme import DARK_STYLESHEET


class LogViewerDialog(QDialog):
    """Full-featured log viewer dialog with per-category coloring, auto-scroll,
    live filter, and export to file.
    """

    _COLORS: Dict[str, str] = {
        "[FFMPEG AUDIO]": "#ff9f0a",
        "[FFMPEG AUDIO FALLBACK]": "#ff9f0a",
        "[FFMPEG EXTRACT": "#5e5ce6",
        "[FFMPEG ENCODE]": "#bf5af2",
        "[VULKAN AI]": "#30d158",
        "[AUDIO REPAIR": "#ff453a",
        "[AUDIO VERIFIED]": "#30d158",
        "[AUDIO WARNING]": "#ffd60a",
        "[AUDIO]": "#30d158",
        "[FRAMES]": "#0a84ff",
        "[ERROR]": "#ff453a",
        "[WARNING]": "#ffd60a",
    }
    _DEFAULT_COLOR = "#8e8e93"

    def __init__(self, log_buffer: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📋 Live Engine Logs")
        self.resize(900, 560)
        self.setMinimumSize(640, 360)
        self.setStyleSheet(DARK_STYLESHEET)
        self._log_buffer = log_buffer if log_buffer is not None else []
        self._auto_scroll = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # Toolbar
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

        # Legend row
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

        # Log text area
        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("txtConsole")
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.txt_log, 1)

        # Status bar
        self.lbl_count = QLabel("0 lines")
        self.lbl_count.setStyleSheet("color: #8e8e93; font-size: 11px;")
        layout.addWidget(self.lbl_count, 0, Qt.AlignmentFlag.AlignRight)

        self._reload_all()

    def append_line(self, msg: str):
        color = self._color_for(msg)
        html_line = f"<span style='color:{color}; font-family: monospace;'>{msg}</span>"
        flt = self.txt_filter.text().strip().lower()
        if not flt or flt in msg.lower():
            self.txt_log.append(html_line)
            if self._auto_scroll:
                sb = self.txt_log.verticalScrollBar()
                sb.setValue(sb.maximum())
        self._update_count()

    def _color_for(self, msg: str) -> str:
        for prefix, color in self._COLORS.items():
            if msg.startswith(prefix):
                return color
        return self._DEFAULT_COLOR

    def _reload_all(self):
        flt = self.txt_filter.text().strip().lower()
        lines = []
        for line in self._log_buffer:
            if not flt or flt in line.lower():
                color = self._color_for(line)
                lines.append(f"<span style='color:{color}; font-family: monospace;'>{line}</span>")
        self.txt_log.setHtml("<br>".join(lines))
        if self._auto_scroll:
            sb = self.txt_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        self._update_count()

    def _apply_filter(self):
        self._reload_all()

    def _toggle_autoscroll(self, enabled: bool):
        self._auto_scroll = enabled
        if enabled:
            self.btn_autoscroll.setText("⬇ Auto-scroll: ON")
            self.btn_autoscroll.setStyleSheet("padding: 4px 12px; font-size: 11px; font-weight: 600; color: #30d158; border: 1px solid #30d158; border-radius: 6px;")
            sb = self.txt_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        else:
            self.btn_autoscroll.setText("⬇ Auto-scroll: OFF")
            self.btn_autoscroll.setStyleSheet("padding: 4px 12px; font-size: 11px; font-weight: 600; color: #8e8e93; border: 1px solid #8e8e93; border-radius: 6px;")

    def _copy_all(self):
        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(self._log_buffer))

    def _export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Log", "upscaler_engine.log", "Log Files (*.log *.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(self._log_buffer))
                self.lbl_count.setText(f"Exported {len(self._log_buffer)} lines to {os.path.basename(path)}")
            except Exception as e:
                self.lbl_count.setText(f"Export failed: {e}")

    def _clear_logs(self):
        if self.parent() and hasattr(self.parent(), "clear_all_logs"):
            self.parent().clear_all_logs()
        else:
            self._log_buffer.clear()
            self.txt_log.clear()
            self._update_count()

    def _update_count(self):
        self.lbl_count.setText(f"{len(self._log_buffer)} lines")
