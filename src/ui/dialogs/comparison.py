"""Before / After visual frame comparison dialog with zoom and view mode controls."""

from __future__ import annotations

import os
from typing import Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.theme import DARK_STYLESHEET


class FrameComparisonDialog(QDialog):
    """Interactive visual Before / After comparison with side-by-side or single viewing."""

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

        # Synchronize scrollbars proportionally with re-entrancy guard
        self._syncing_scroll = False

        def _sync_scroll(src_sb, dst_sb):
            if self._syncing_scroll:
                return
            src_max = src_sb.maximum()
            dst_max = dst_sb.maximum()
            if src_max > 0 and dst_max > 0:
                self._syncing_scroll = True
                try:
                    ratio = src_sb.value() / src_max
                    dst_sb.setValue(int(ratio * dst_max))
                finally:
                    self._syncing_scroll = False

        self.scroll_orig.horizontalScrollBar().valueChanged.connect(
            lambda _: _sync_scroll(self.scroll_orig.horizontalScrollBar(), self.scroll_up.horizontalScrollBar())
        )
        self.scroll_up.horizontalScrollBar().valueChanged.connect(
            lambda _: _sync_scroll(self.scroll_up.horizontalScrollBar(), self.scroll_orig.horizontalScrollBar())
        )
        self.scroll_orig.verticalScrollBar().valueChanged.connect(
            lambda _: _sync_scroll(self.scroll_orig.verticalScrollBar(), self.scroll_up.verticalScrollBar())
        )
        self.scroll_up.verticalScrollBar().valueChanged.connect(
            lambda _: _sync_scroll(self.scroll_up.verticalScrollBar(), self.scroll_orig.verticalScrollBar())
        )

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
        if self.pixmap_orig.isNull():
            self.lbl_orig_img.setText("Original frame unavailable")
        else:
            w_orig = max(10, int(self.pixmap_orig.width() * self.zoom_level))
            h_orig = max(10, int(self.pixmap_orig.height() * self.zoom_level))
            self.lbl_orig_img.setPixmap(
                self.pixmap_orig.scaled(
                    w_orig, h_orig, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )

        if self.pixmap_up.isNull():
            self.lbl_up_img.setText("Upscaled frame unavailable")
        else:
            w_up = max(10, int(self.pixmap_up.width() * self.zoom_level))
            h_up = max(10, int(self.pixmap_up.height() * self.zoom_level))
            self.lbl_up_img.setPixmap(
                self.pixmap_up.scaled(
                    w_up, h_up, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )

    def closeEvent(self, event):
        temp_dir = self.info.get("temp_dir")
        if temp_dir and os.path.isdir(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        super().closeEvent(event)
