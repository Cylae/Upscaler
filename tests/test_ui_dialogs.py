import os
from unittest.mock import patch, MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QHeaderView

from src.ui.dialogs.comparison import FrameComparisonDialog
from src.ui.dialogs.doctor import DoctorDialog
from src.ui.dialogs.logs import LogViewerDialog
from src.ui.dialogs.preflight import PreflightDialog


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_preflight_dialog_columns(qapp):
    preflight_data = {
        "total_items": 2,
        "pending_items": 2,
        "skipped_items": 0,
        "target_preset": "4K",
        "video_codec": "hevc_nvenc",
        "gpu_name": "NVIDIA RTX",
        "total_source_size_mb": 100.0,
        "estimated_output_mb": 250.0,
        "free_disk_mb": 5000.0,
        "warnings": [],
    }
    dry_run_data = [
        {
            "index": 1,
            "relative_source": "episodes/s01e01.mp4",
            "target_name": "s01e01_4k.mp4",
            "collision": False,
            "already_exists": False,
        }
    ]
    dlg = PreflightDialog(preflight_data, dry_run_data)
    dlg.show()
    qapp.processEvents()

    # Find the table in the dialog
    from PyQt6.QtWidgets import QTableWidget
    table = dlg.findChild(QTableWidget)
    assert table is not None
    assert table.rowCount() == 1
    assert table.horizontalHeader().sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    assert table.horizontalHeader().sectionResizeMode(2) == QHeaderView.ResizeMode.Stretch
    dlg.close()


def test_doctor_dialog(qapp, tmp_path):
    dlg = DoctorDialog(parent=None)
    dlg.show()
    qapp.processEvents()

    # Test export without parent
    out_file = tmp_path / "doctor_export.txt"
    with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(out_file), "Text Files (*.txt)")):
        dlg._export_diagnostic()
    assert out_file.exists()
    assert "SYSTEM ENVIRONMENT DIAGNOSTIC" in out_file.read_text(encoding="utf-8")
    dlg.close()


def test_doctor_dialog_with_parent_status(qapp, tmp_path):
    from PyQt6.QtWidgets import QWidget, QLabel
    parent = QWidget()
    parent.lbl_overall_status = QLabel()
    dlg = DoctorDialog(parent=parent)
    out_file = tmp_path / "doctor_export2.txt"
    with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(out_file), "Text Files (*.txt)")):
        dlg._export_diagnostic()
    assert "Diagnostic exported:" in parent.lbl_overall_status.text()
    dlg.close()


def test_log_viewer_dialog(qapp, tmp_path):
    initial_logs = [
        "[FFMPEG ENCODE] frame=100 fps=35.2 q=19.0",
        "[ERROR] Test error with <tags> & arrows -> ok",
    ]
    dlg = LogViewerDialog(log_buffer=initial_logs)
    dlg.show()
    qapp.processEvents()

    assert dlg.lbl_count.text() == "2 lines"

    # Append line with special characters
    dlg.append_line("[WARNING] Stream <0:0> -> <0:1> mismatch")
    assert dlg.lbl_count.text() == "3 lines"

    # Test filter
    dlg.txt_filter.setText("error")
    assert "<span" in dlg.txt_log.toHtml()
    assert "&lt;tags&gt;" in dlg.txt_log.toHtml()

    # Test autoscroll toggle
    dlg.btn_autoscroll.setChecked(False)
    assert dlg._auto_scroll is False
    dlg.btn_autoscroll.setChecked(True)
    assert dlg._auto_scroll is True

    # Test export
    export_path = tmp_path / "exported.log"
    with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(export_path), "Log Files")):
        dlg._export_log()
    assert export_path.exists()
    assert "[FFMPEG ENCODE]" in export_path.read_text(encoding="utf-8")

    dlg.close()


def test_frame_comparison_dialog(qapp, tmp_path):
    info = {
        "model_name": "RealCUGAN Pro",
        "orig_width": 640,
        "orig_height": 480,
        "target_width": 1920,
        "target_height": 1080,
        "time_seconds": 1.5,
    }
    # Test with non-existent or null pixmaps
    dlg = FrameComparisonDialog("nonexistent_orig.png", "nonexistent_up.png", info)
    dlg.show()
    qapp.processEvents()

    assert dlg.lbl_orig_img.text() == "Original frame unavailable"
    assert dlg.lbl_up_img.text() == "Upscaled frame unavailable"

    # Zoom controls
    dlg._zoom_in()
    assert dlg.zoom_level == 1.25
    dlg._zoom_out()
    assert dlg.zoom_level == 1.0
    dlg._zoom_reset()
    assert dlg.zoom_level == 1.0

    # View modes
    dlg._on_view_mode_changed(0)  # Side by side
    assert dlg.panel_orig.isVisible() is True
    assert dlg.panel_up.isVisible() is True

    dlg._on_view_mode_changed(1)  # After
    assert dlg.panel_orig.isVisible() is False
    assert dlg.panel_up.isVisible() is True

    dlg._on_view_mode_changed(2)  # Before
    assert dlg.panel_orig.isVisible() is True
    assert dlg.panel_up.isVisible() is False

    dlg.close()
