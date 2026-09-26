import os
import inspect
import pytest
from unittest.mock import MagicMock, patch
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import HardwareInfo
from src.core.scene_renamer import SceneNamingConfig
from src.core.video_pipeline import UpscaleJobConfig, VideoUpscalerPipeline


def test_batch_manager_selective_target_indices():
    """Verify that specifying target_indices only executes targeted episodes."""
    hw = HardwareInfo(gpu_name="RTX 3070", gpu_vram_gb=8.0, cpu_cores=32)
    bm = BatchManager(hw)

    item1 = QueueItem(
        index=1,
        source_path="fake/S01E01.mp4",
        relative_source_path="S01E01.mp4",
        target_scene_name="Test.S01E01.mp4",
        target_full_path="fake/out/Test.S01E01.mp4",
        status=EpisodeStatus.PENDING
    )
    item2 = QueueItem(
        index=2,
        source_path="fake/S01E02.mp4",
        relative_source_path="S01E02.mp4",
        target_scene_name="Test.S01E02.mp4",
        target_full_path="fake/out/Test.S01E02.mp4",
        status=EpisodeStatus.PENDING
    )
    bm.queue = [item1, item2]

    def mock_process_video(self, job_config, log_callback=None, progress_callback=None):
        return True, "Completed successfully"

    with patch.object(VideoUpscalerPipeline, "process_video", new=mock_process_video):
        # Target only index 2
        bm.run_batch(
            output_dir="fake/out",
            base_dir="fake",
            target_preset="4k",
            video_codec="hevc_nvenc",
            model_id="auto",
            scene_config=SceneNamingConfig(),
            target_indices=[2],
            max_workers=1
        )

    # Item 1 must remain PENDING because it was not in target_indices
    assert item1.status == EpisodeStatus.PENDING
    # Item 2 was targeted, so it was processed and marked COMPLETED
    assert item2.status == EpisodeStatus.COMPLETED


def test_nvenc_spatial_and_temporal_aq():
    """Verify hardware encoding command generation includes spatial & temporal AQ."""
    # The NVENC parameters now live in _build_encode_cmd (extracted for testability).
    # Inspect both that builder and process_video to cover the full contract.
    src_builder = inspect.getsource(VideoUpscalerPipeline._build_encode_cmd)
    src_process = inspect.getsource(VideoUpscalerPipeline.process_video)
    combined = src_builder + src_process

    assert "-spatial-aq" in combined
    assert "-temporal-aq" in combined
    assert "-b_ref_mode" in combined


def test_qt_ui_preview_and_error_pills():
    """Verify UI components (Preview button, error stat pill, filter modes) without GUI crash."""
    from PyQt6.QtWidgets import QApplication
    from src.ui.app_qt import UpscalerMainWindow

    # Ensure offscreen QApplication
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    window = UpscalerMainWindow()

    # Check preview button
    assert hasattr(window, "btn_preview")
    assert "Test 1 Episode" in window.btn_preview.text()
    assert window.btn_preview.toolTip() != ""

    # Check error stat pill
    assert hasattr(window, "btn_stat_failed")
    assert "Errors: 0" in window.btn_stat_failed.text()

    # Check filter mode switching
    window.set_filter_mode("failed")
    assert window._current_filter_mode == "failed"
    assert window.btn_stat_failed.isChecked()

    window.set_filter_mode("all")
    assert window._current_filter_mode == "all"
    assert window.btn_stat_all.isChecked()

    # Check comprehensive tooltips
    assert window.txt_source.toolTip() != ""
    assert window.txt_output.toolTip() != ""
    assert window.cb_preset.toolTip() != ""
    assert window.slider_cq.toolTip() != ""
    assert window.cb_audio_mode.toolTip() != ""


def test_ui_audit_fixes():
    """Verify bugfixes: distinct objectNames, log buffer capping, clear_all_logs, debounce timer."""
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent
    from PyQt6.QtWidgets import QApplication
    from src.ui.app_qt import UpscalerMainWindow

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    window = UpscalerMainWindow()

    # 1. Distinct objectNames for CSS targeting
    assert window.btn_preview_clip.objectName() == "btnPreviewClip"
    assert window.btn_preview.objectName() == "btnPreview"
    assert window.btn_preview_clip.objectName() != window.btn_preview.objectName()

    # 2. Debounce timer exists and is configured
    assert hasattr(window, "_settings_timer")
    assert window._settings_timer.isSingleShot()
    assert window._settings_timer.interval() == 350

    # 3. Log reception and clear_all_logs
    window.on_log_received("FFmpeg extraction test message")
    assert len(window._log_buffer) == 1
    assert "Logs (1)" in window.btn_open_logs.text()

    window.clear_all_logs()
    assert len(window._log_buffer) == 0
    assert "Logs (0)" in window.btn_open_logs.text()
    assert window.txt_console.toPlainText() == ""

    # 4. Log buffer max capacity cap (5000 lines)
    for i in range(5050):
        window.on_log_received(f"line {i}")
    assert len(window._log_buffer) == 5000

    # 5. Compact resize displays numbers
    window.resizeEvent(QResizeEvent(QSize(900, 600), QSize(1280, 800)))
    assert "All: 0" in window.btn_stat_all.text()
    assert "Ready: 0" in window.btn_stat_ready.text()


def test_frame_comparison_dialog_and_controls():
    """Verify FrameComparisonDialog instantiation, zoom, view modes and new controls."""
    from PyQt6.QtWidgets import QApplication
    from src.ui.app_qt import DARK_STYLESHEET, FrameComparisonDialog, UpscalerMainWindow

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    # 1. Window controls
    win = UpscalerMainWindow()
    assert hasattr(win, "btn_compare_frame")
    assert win.btn_compare_frame.objectName() == "btnCompareFrame"
    assert hasattr(win, "btn_clear_queue")
    assert hasattr(win, "cb_naming_mode")
    assert win.cb_naming_mode.count() >= 3

    # 2. Frame comparison dialog
    info = {
        "model_name": "Real-CUGAN Pro",
        "orig_width": 640,
        "orig_height": 480,
        "target_width": 2880,
        "target_height": 2160,
        "time_seconds": 0.42
    }
    dlg = FrameComparisonDialog("nonexistent_orig.png", "nonexistent_up.png", info, parent=win)
    assert dlg.styleSheet() == DARK_STYLESHEET
    assert dlg.zoom_level == 1.0

    # Test zoom controls
    dlg._zoom_in()
    assert dlg.zoom_level > 1.0
    dlg._zoom_reset()
    assert dlg.zoom_level == 1.0
    dlg._zoom_out()
    assert dlg.zoom_level < 1.0

    # Test view modes
    dlg._on_view_mode_changed(1)  # After only
    assert dlg.panel_orig.isHidden()
    assert not dlg.panel_up.isHidden()

    dlg._on_view_mode_changed(2)  # Before only
    assert not dlg.panel_orig.isHidden()
    assert dlg.panel_up.isHidden()

    dlg._on_view_mode_changed(0)  # Side-by-side
    assert not dlg.panel_orig.isHidden()
    assert not dlg.panel_up.isHidden()

