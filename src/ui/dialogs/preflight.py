"""Preflight check & output plan dialog with collision warnings and dry-run table."""

from __future__ import annotations

from typing import Dict, List

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.ui.theme import DARK_STYLESHEET, apply_dark_titlebar


class PreflightDialog(QDialog):
    """Displays comprehensive preflight summary before starting batch processing."""

    def __init__(self, preflight_data: Dict, dry_run_data: List[Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preflight Check & Output Plan")
        self.resize(780, 560)
        self.setStyleSheet(DARK_STYLESHEET)
        apply_dark_titlebar(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("📋 Preflight Summary & Output Plan")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #0a84ff;")
        layout.addWidget(title)

        # Warnings block if any
        warnings = preflight_data.get("warnings", [])
        if warnings:
            warn_frame = QFrame()
            warn_frame.setStyleSheet("background-color: rgba(255, 69, 58, 0.12); border: 1px solid #ff453a; border-radius: 8px; padding: 10px;")
            warn_layout = QVBoxLayout(warn_frame)
            warn_layout.setContentsMargins(8, 8, 8, 8)
            warn_title = QLabel("⚠️ Preflight Warnings:")
            warn_title.setStyleSheet("font-weight: 700; color: #ff453a; font-size: 12px;")
            warn_layout.addWidget(warn_title)
            for w in warnings:
                lbl = QLabel(f"• {w}")
                lbl.setStyleSheet("color: #ff9f0a; font-size: 11px;")
                warn_layout.addWidget(lbl)
            layout.addWidget(warn_frame)

        # Summary Grid
        info_frame = QFrame()
        info_frame.setObjectName("macPanel")
        info_grid = QHBoxLayout(info_frame)
        info_grid.setContentsMargins(14, 12, 14, 12)

        col1 = QVBoxLayout()
        col1.addWidget(QLabel(f"<b>Total Queue:</b> {preflight_data.get('total_items', 0)} files"))
        col1.addWidget(QLabel(f"<b>Pending:</b> {preflight_data.get('pending_items', 0)} files"))
        col1.addWidget(QLabel(f"<b>Already Present:</b> {preflight_data.get('skipped_items', 0)} files"))
        info_grid.addLayout(col1)

        col2 = QVBoxLayout()
        col2.addWidget(QLabel(f"<b>Target:</b> {str(preflight_data.get('target_preset', '')).upper()} (CQ {preflight_data.get('cq_level', 19)})"))
        col2.addWidget(QLabel(f"<b>Encoder:</b> {preflight_data.get('video_codec', '')}"))
        col2.addWidget(QLabel(f"<b>Hardware:</b> {preflight_data.get('gpu_name', '')}"))
        info_grid.addLayout(col2)

        col3 = QVBoxLayout()
        col3.addWidget(QLabel(f"<b>Source Size:</b> {preflight_data.get('total_source_size_mb', 0):.1f} MB"))
        col3.addWidget(QLabel(f"<b>Est. Output:</b> ~{preflight_data.get('estimated_output_mb', 0):.1f} MB"))
        col3.addWidget(QLabel(f"<b>Target Free:</b> {preflight_data.get('free_disk_mb', 0):.1f} MB"))
        info_grid.addLayout(col3)

        layout.addWidget(info_frame)

        # Dry Run Table
        lbl_table = QLabel("Output Mapping Plan (Dry Run):")
        lbl_table.setStyleSheet("font-weight: 600; color: #f5f5f7; font-size: 12px;")
        layout.addWidget(lbl_table)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["#", "Source File", "Target Scene Name", "Model", "Status"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        table.setRowCount(len(dry_run_data))

        for r, item in enumerate(dry_run_data):
            table.setItem(r, 0, QTableWidgetItem(str(item.get("index", r + 1))))
            table.setItem(r, 1, QTableWidgetItem(item.get("relative_source", "")))
            it_target = QTableWidgetItem(item.get("target_name", ""))
            if item.get("collision"):
                it_target.setForeground(QColor("#ff453a"))
            else:
                it_target.setForeground(QColor("#38bdf8"))
            table.setItem(r, 2, it_target)
            it_model = QTableWidgetItem(str(item.get("model", "Auto")))
            it_model.setForeground(QColor("#a78bfa"))
            table.setItem(r, 3, it_model)
            st_text = "Already Present (Skip)" if item.get("already_exists") else "Ready"
            if item.get("collision"):
                st_text = "COLLISION!"
            it_st = QTableWidgetItem(st_text)
            table.setItem(r, 4, it_st)

        layout.addWidget(table, 1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)
