"""Keep ordinary test runs independent of the user's desktop settings and media."""

import os

# This must be set before test modules import QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt6.QtCore import QSettings, QTimer


@pytest.fixture(autouse=True)
def isolate_desktop_settings_and_startup_scan(tmp_path, monkeypatch):
    from src.ui import main_window

    def isolated_settings(*_args, **_kwargs):
        settings = QSettings(str(tmp_path / "desktop-settings.ini"), QSettings.Format.IniFormat)
        settings.setFallbacksEnabled(False)
        return settings

    original_single_shot = QTimer.singleShot

    def safe_single_shot(*args):
        callback = args[-1] if args else None
        if getattr(callback, "__name__", "") == "on_scan_clicked":
            return None
        return original_single_shot(*args)

    monkeypatch.setattr(main_window, "QSettings", isolated_settings)
    monkeypatch.setattr(main_window.QTimer, "singleShot", safe_single_shot)
    # Startup housekeeping must not remove the developer's abandoned job data.
    # Storage cleanup itself remains covered with temporary directories.
    monkeypatch.setattr(main_window, "cleanup_stale_temp_directories", lambda: 0)
