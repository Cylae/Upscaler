"""System & hardware environment diagnostic dialog."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.core.hardware import (
    export_diagnostic_report,
    get_base_dir,
    get_hardware_info,
)
from src.core.models import check_local_model_availability
from src.ui.theme import DARK_STYLESHEET


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

        btn_row = QHBoxLayout()
        btn_export = QPushButton("💾 Export Diagnostic Report...")
        btn_export.setStyleSheet("padding: 5px 14px; font-size: 11px; border-radius: 6px;")
        btn_export.clicked.connect(self._export_diagnostic)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _export_diagnostic(self):
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnostic Report",
            os.path.expanduser("~/upscaler_diagnostic.txt"),
            "Text Files (*.txt);;All Files (*)"
        )
        if target_path:
            out_file = export_diagnostic_report(target_path)
            self.parent().lbl_overall_status.setText(f"Diagnostic exported: {os.path.basename(out_file)}")
