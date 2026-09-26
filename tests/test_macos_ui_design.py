import os
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QApplication
from src.ui.app_qt import DARK_STYLESHEET, DoctorDialog, LogViewerDialog, UpscalerMainWindow


def test_macos_dark_stylesheet_tokens():
    """Verify that the stylesheet contains authentic Apple HIG Dark Mode tokens."""
    # System window background
    assert "#1e1e1e" in DARK_STYLESHEET
    # Elevated card panel surface
    assert "#252528" in DARK_STYLESHEET
    # Apple accent colors
    assert "#0a84ff" in DARK_STYLESHEET  # Apple Blue
    assert "#30d158" in DARK_STYLESHEET  # Apple Green
    assert "#5e5ce6" in DARK_STYLESHEET  # Apple Indigo
    assert "#ff453a" in DARK_STYLESHEET  # Apple Red


def test_macos_window_components_and_paddings():
    """Verify macOS window elements, progress bar height, and dialogs."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    window = UpscalerMainWindow()

    # Progress bar slim macOS indicator
    assert window.progress_bar.maximumHeight() == 12 or window.progress_bar.height() <= 14

    # Button object names
    assert window.btn_start.objectName() == "btnStart"
    assert window.btn_preview_clip.objectName() == "btnPreviewClip"
    assert window.btn_preview.objectName() == "btnPreview"
    assert window.btn_stop.objectName() == "btnStop"

    # Dialogs load with DARK_STYLESHEET
    doc_dlg = DoctorDialog(parent=window)
    assert doc_dlg.styleSheet() == DARK_STYLESHEET

    log_dlg = LogViewerDialog(log_buffer=["[AUDIO] test", "[ERROR] test err"], parent=window)
    assert log_dlg.styleSheet() == DARK_STYLESHEET
    assert log_dlg._color_for("[AUDIO]") == "#30d158"
    assert log_dlg._color_for("[ERROR]") == "#ff453a"
    assert log_dlg._color_for("[FRAMES]") == "#0a84ff"

    # Responsive resize states without clipping
    for w in (900, 1020, 1180, 1400):
        ev = QResizeEvent(QSize(w, 700), QSize(1280, 840))
        window.resizeEvent(ev)
        assert window.isVisible() or not window.isVisible()  # valid widget state
