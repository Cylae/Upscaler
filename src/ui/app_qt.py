"""Qt Graphical User Interface for Video Enhancer 4K.

Serves as the clean entry point and facade, re-exporting the main window,
dialogs, workers, and styles from modular UI components.
"""

from __future__ import annotations

import sys

from src.ui.dialogs.comparison import FrameComparisonDialog
from src.ui.dialogs.doctor import DoctorDialog
from src.ui.dialogs.logs import LogViewerDialog
from src.ui.dialogs.preflight import PreflightDialog
from src.ui.main_window import UpscalerMainWindow, run_gui
from src.ui.theme import DARK_STYLESHEET
from src.ui.workers import (
    FramePreviewWorker,
    ModelHintWorker,
    ScanWorker,
    UpscaleWorker,
)

__all__ = [
    "DARK_STYLESHEET",
    "DoctorDialog",
    "FrameComparisonDialog",
    "FramePreviewWorker",
    "LogViewerDialog",
    "ModelHintWorker",
    "PreflightDialog",
    "ScanWorker",
    "UpscaleWorker",
    "UpscalerMainWindow",
    "run_gui",
]

if __name__ == "__main__":
    sys.exit(run_gui())
