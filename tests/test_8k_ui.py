"""8K UI routing, using isolated settings and mocked media workers."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.batch import QueueItem
from src.core.hardware import HardwareInfo
from src.ui.main_window import UpscalerMainWindow


@pytest.fixture
def window(monkeypatch):
    app = QApplication.instance() or QApplication(["8k-ui-tests"])
    settings = {}
    memory_settings = MagicMock()
    memory_settings.value.side_effect = lambda key, default="": settings.get(key, default)
    memory_settings.setValue.side_effect = lambda key, value: settings.__setitem__(key, value)
    monkeypatch.setattr("src.ui.main_window.QSettings", lambda *args: memory_settings)
    monkeypatch.setattr("src.ui.main_window.get_hardware_info", lambda: HardwareInfo())
    monkeypatch.setattr("src.ui.main_window.cleanup_stale_temp_directories", lambda: None)
    monkeypatch.setattr(UpscalerMainWindow, "on_scan_clicked", lambda self: None)
    win = UpscalerMainWindow()
    yield win
    win.close()
    app.processEvents()


def _queue_sample(window):
    window.batch_manager.queue = [QueueItem(
        index=1, source_path="sample/input.mp4", relative_source_path="input.mp4",
        target_scene_name="input_8K.mp4", target_full_path="sample/output/input_8K.mp4",
    )]
    window.txt_source.setText("sample")
    window.txt_output.setText("sample/output")


def test_realistic_8k_preset_updates_controls_and_output_names(window):
    window.txt_output.setText("sample/output (2160p)")
    window.cb_preset.setCurrentIndex(window.cb_preset.findData("realistic_8k"))

    assert window.cb_resolution.currentData() == "8k"
    assert window.cb_model.currentData() == "realesrgan-x4plus"
    assert window.slider_cq.value() == 18
    assert window.cb_workers.currentText().startswith("1 Worker")
    assert window.cb_frame_fmt.currentText() == "Lossless PNG"
    assert window.chk_tta.isChecked() is False
    assert window.txt_output.text() == "sample/output (4320p)"
    assert window.get_current_scene_config().resolution_tag == "4320p"
    assert "_8K.mp4" in window.lbl_preview_name.text()

    window.cb_resolution.setCurrentIndex(window.cb_resolution.findData("1080p"))
    assert window.txt_output.text() == "sample/output (1080p)"


def test_8k_target_reaches_preflight_batch_and_clip_preview(window, monkeypatch):
    window.cb_preset.setCurrentIndex(window.cb_preset.findData("realistic_8k"))
    _queue_sample(window)
    summary = MagicMock(return_value={"warnings": []})
    window.batch_manager.get_preflight_summary = summary
    window.batch_manager.plan_dry_run = MagicMock(return_value=[])
    dialog = MagicMock()
    monkeypatch.setattr("src.ui.main_window.PreflightDialog", dialog)
    worker = MagicMock()
    monkeypatch.setattr("src.ui.main_window.UpscaleWorker", worker)

    window.on_preflight_clicked()
    assert summary.call_args.kwargs["target_preset"] == "8k"
    dialog.return_value.exec.assert_called_once()

    window.start_processing(target_indices=[1], preview_duration_s=5.0)
    config = worker.call_args.kwargs
    assert config["target_preset"] == "8k"
    assert config["scene_config"].resolution_tag == "4320p"
    assert config["model_id"] == "realesrgan-x4plus"
    assert config["frame_format"] == "png"
    assert config["max_workers"] == 1
    assert config["cq_level"] == 18
    assert config["enable_tta"] is False
    assert config["threads"] == "auto"
    assert config["preview_duration_s"] == 5.0
    worker.return_value.start.assert_called_once()


def test_8k_target_reaches_single_frame_and_model_hint_workers(window, monkeypatch):
    window.cb_resolution.setCurrentIndex(window.cb_resolution.findData("8k"))
    _queue_sample(window)
    frame_worker = MagicMock()
    hint_worker = MagicMock()
    monkeypatch.setattr("src.ui.main_window.FramePreviewWorker", frame_worker)
    monkeypatch.setattr("src.ui.main_window.ModelHintWorker", hint_worker)

    window.on_compare_frame_clicked(target_idx=1)
    assert frame_worker.call_args.kwargs["target_preset"] == "8k"
    frame_worker.return_value.start.assert_called_once()

    window._update_auto_model_hint(window.batch_manager.queue)
    assert hint_worker.call_args.args[1] == "8k"
    hint_worker.return_value.start.assert_called_once()


def test_8k_filename_resolution_hint_precedes_uhd_match(window):
    assert window._infer_resolution_hint("sample.8k.uhd.mp4") == (7680, 4320)
