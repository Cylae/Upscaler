"""Modal dialogs for diagnostics, logs, preflight checks, and visual comparisons."""

from __future__ import annotations

from src.ui.dialogs.comparison import FrameComparisonDialog
from src.ui.dialogs.doctor import DoctorDialog
from src.ui.dialogs.logs import LogViewerDialog
from src.ui.dialogs.preflight import PreflightDialog

__all__ = [
    "FrameComparisonDialog",
    "DoctorDialog",
    "LogViewerDialog",
    "PreflightDialog",
]
