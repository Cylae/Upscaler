"""Direct settings access and exact option forwarding for the functional UI."""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QScrollArea, QTabWidget

from src.core.batch import QueueItem
from src.core.hardware import HardwareInfo
from src.ui import main_window


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "get_hardware_info", lambda: HardwareInfo())
    monkeypatch.setattr(main_window, "apply_dark_titlebar", lambda *_args: True)
    win = main_window.UpscalerMainWindow()
    yield win
    win.close()


def choose(combo, value):
    index = combo.findData(value)
    assert index >= 0, f"Setting does not expose {value!r}"
    combo.setCurrentIndex(index)


def reveal_control(window, control, qapp):
    for tabs in window.findChildren(QTabWidget):
        for index in range(tabs.count()):
            page = tabs.widget(index)
            if page is control or page.isAncestorOf(control):
                tabs.setCurrentIndex(index)
                break
    qapp.processEvents()
    for scroll in window.findChildren(QScrollArea):
        page = scroll.widget()
        if page and page.isAncestorOf(control):
            scroll.ensureWidgetVisible(control)
    qapp.processEvents()


def test_all_settings_are_directly_reachable_in_compact_window(window, qapp):
    window.resize(1024, 768)
    window.show()
    qapp.processEvents()
    assert window.findChildren(QTabWidget), "Settings should have direct tab access"
    settings = (
        "cb_preset", "cb_resolution", "cb_codec", "cb_model", "slider_cq", "spin_cq",
        "cb_nvenc_preset", "cb_denoise", "cb_audio_mode", "cb_workers",
        "cb_threads", "cb_frame_fmt", "cb_tile_size", "chk_tta",
        "cb_naming_mode", "txt_show_name", "txt_group", "cb_source",
        "cb_audio_tag", "cb_container", "txt_custom_suffix", "chk_preserve_folders", "chk_overwrite",
    )
    for name in settings:
        control = getattr(window, name)
        reveal_control(window, control, qapp)
        assert control.isVisible(), f"{name} still requires an extra hidden-panel toggle"
        assert control.width() > 0 and control.height() > 0
        center = control.mapTo(window, control.rect().center())
        assert window.rect().contains(center), f"{name} is outside the compact window"
        for scroll in window.findChildren(QScrollArea):
            if scroll.widget() and scroll.widget().isAncestorOf(control):
                assert control.width() <= scroll.viewport().width(), f"{name} overflows horizontally"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("cb_codec", "libx264"),
        ("cb_nvenc_preset", "p2"),
        ("cb_nvenc_preset", "p3"),
        ("cb_workers", 4),
        ("cb_resolution", "8k"),
        ("cb_model", "realcugan-pro"),
        ("cb_frame_fmt", "png"),
        ("cb_audio_mode", "flac"),
        ("cb_denoise", 2),
    ],
)
def test_manual_processing_option_marks_preset_custom(window, attribute, value):
    choose(window.cb_preset, "cartoon_4k_master")
    choose(getattr(window, attribute), value)
    assert window.cb_preset.currentData() == "custom"


def test_speed_preset_selects_its_actual_nvenc_preset(window):
    choose(window.cb_preset, "fast_draft")
    assert window.cb_preset.currentData() == "fast_draft"
    assert window.cb_nvenc_preset.currentData() == "p3"
    assert window.slider_cq.value() == 22


def test_quality_can_be_entered_numerically_and_stays_in_sync(window):
    window.spin_cq.setValue(17)
    assert window.slider_cq.value() == 17
    assert window.cb_preset.currentData() == "custom"
    window.slider_cq.setValue(24)
    assert window.spin_cq.value() == 24
    choose(window.cb_preset, "realistic_8k")
    assert window.spin_cq.value() == window.slider_cq.value() == 18


@pytest.mark.parametrize(
    ("attribute", "text", "helper", "expected"),
    [("cb_threads", "3:5:3", "_get_thread_spec", "3:5:3"), ("cb_tile_size", "384", "_get_tile_size", 384)],
)
def test_manual_engine_values_override_previously_selected_option(window, attribute, text, helper, expected):
    choose(window.cb_preset, "cartoon_4k_master")
    combo = getattr(window, attribute)
    assert combo.isEditable()
    combo.setCurrentIndex(1)
    combo.setEditText(text)
    combo.lineEdit().editingFinished.emit()
    assert getattr(window, helper)() == expected
    assert window.cb_preset.currentData() == "custom"


def test_data_based_processing_options_reach_worker_and_preflight(window, monkeypatch):
    window.batch_manager.queue = [QueueItem(1, "C:/input.mp4", "input.mp4", "output.mp4", "C:/out/output.mp4")]
    window.txt_source.setText("C:/input")
    window.txt_output.setText("C:/out")
    choose(window.cb_resolution, "8k")
    choose(window.cb_codec, "libx264")
    choose(window.cb_model, "realcugan-pro")
    choose(window.cb_nvenc_preset, "p2")
    choose(window.cb_workers, 4)
    choose(window.cb_frame_fmt, "png")
    choose(window.cb_audio_mode, "flac")
    choose(window.cb_denoise, 2)
    window.cb_threads.setCurrentIndex(1)
    window.cb_threads.setEditText("3:5:3")
    window.cb_tile_size.setCurrentIndex(1)
    window.cb_tile_size.setEditText("384")
    window.chk_tta.setChecked(True)
    window.chk_overwrite.setChecked(True)
    window.chk_preserve_folders.setChecked(False)
    window.slider_cq.setValue(17)

    worker = MagicMock()
    worker.isRunning.return_value = False
    factory = MagicMock(return_value=worker)
    preflight = MagicMock(return_value={"warnings": []})
    monkeypatch.setattr(main_window, "UpscaleWorker", factory)
    monkeypatch.setattr(window.batch_manager, "get_preflight_summary", preflight)
    window.start_processing()

    options = factory.call_args.kwargs
    expected = {
        "target_preset": "8k", "video_codec": "libx264", "model_id": "realcugan-pro",
        "nvenc_preset": "p2", "max_workers": 4, "frame_format": "png", "audio_mode": "flac",
        "denoise_level": 2, "enable_tta": True, "overwrite": True, "preserve_folders": False,
        "cq_level": 17, "threads": "3:5:3", "tile_size": 384,
    }
    for name, value in expected.items():
        assert options[name] == value
    for name in ("target_preset", "video_codec", "model_id", "cq_level"):
        assert preflight.call_args.kwargs[name] == expected[name]
    assert options["scene_config"].codec_tag == "H.264"
    worker.start.assert_called_once()


def test_preflight_button_uses_selected_cpu_h264_and_quality(window, monkeypatch):
    window.batch_manager.queue = [QueueItem(1, "C:/input.mp4", "input.mp4", "output.mp4", "C:/out/output.mp4")]
    window.txt_output.setText("C:/out")
    choose(window.cb_codec, "libx264")
    choose(window.cb_model, "realesrgan-x4plus")
    choose(window.cb_resolution, "1080p")
    window.slider_cq.setValue(20)
    preflight = MagicMock(return_value={"warnings": []})
    dialog = MagicMock()
    monkeypatch.setattr(window.batch_manager, "get_preflight_summary", preflight)
    monkeypatch.setattr(window.batch_manager, "plan_dry_run", MagicMock(return_value={}))
    monkeypatch.setattr(main_window, "PreflightDialog", MagicMock(return_value=dialog))
    window.on_preflight_clicked()
    assert preflight.call_args.kwargs["video_codec"] == "libx264"
    assert preflight.call_args.kwargs["target_preset"] == "1080p"
    assert preflight.call_args.kwargs["model_id"] == "realesrgan-x4plus"
    assert preflight.call_args.kwargs["cq_level"] == 20
    dialog.exec.assert_called_once()


def test_test_and_clip_actions_use_selected_episode(window, monkeypatch):
    queue = [QueueItem(index, f"C:/input{index}.mp4", f"input{index}.mp4", "output.mp4", "C:/out/output.mp4")
             for index in (1, 2)]
    window.batch_manager.queue = queue
    window._populate_table(queue)
    window.table.selectRow(1)
    start = MagicMock()
    monkeypatch.setattr(window, "start_processing", start)
    window.on_test_single_clicked()
    start.assert_called_once_with(target_indices=[2])
    start.reset_mock()
    window.spin_preview_dur.setValue(45)
    window.on_preview_clip_clicked()
    start.assert_called_once_with(target_indices=[2], preview_duration_s=45.0)


def test_frame_comparison_uses_selected_episode_override_and_engine_settings(window, monkeypatch):
    choose(window.cb_model, "realesr-animevideov3")
    choose(window.cb_resolution, "8k")
    choose(window.cb_denoise, 2)
    window.chk_tta.setChecked(True)
    window.cb_threads.setEditText("3:5:3")
    window.cb_tile_size.setEditText("384")
    queue = [QueueItem(index, f"C:/input{index}.mp4", f"input{index}.mp4", "output.mp4", "C:/out/output.mp4")
             for index in (1, 2)]
    queue[1].model = "realcugan-pro"
    window.batch_manager.queue = queue
    window._populate_table(queue)
    window.table.selectRow(1)
    pipeline = object()
    preview_worker = MagicMock()
    preview_worker.isRunning.return_value = False
    factory = MagicMock(return_value=preview_worker)
    monkeypatch.setattr(main_window, "VideoUpscalerPipeline", MagicMock(return_value=pipeline))
    monkeypatch.setattr(main_window, "FramePreviewWorker", factory)
    window.on_compare_frame_clicked()
    factory.assert_called_once_with(
        pipeline=pipeline, video_path="C:/input2.mp4", model_id="realcugan-pro", target_preset="8k",
        threads="3:5:3", tile_size=384, denoise_level=2, enable_tta=True,
    )
    preview_worker.start.assert_called_once()


def test_mkv_and_custom_suffix_reach_config_and_output_plan(window, tmp_path):
    source = str(tmp_path / "input.mp4")
    output_dir = str(tmp_path / "output")
    window.txt_source.setText(str(tmp_path))
    window.txt_output.setText(output_dir)
    window.batch_manager.queue = [QueueItem(1, source, "input.mp4", "input_4K.mp4", str(tmp_path / "old.mp4"))]
    choose(window.cb_naming_mode, "clean")
    choose(window.cb_container, ".mkv")
    window.txt_custom_suffix.setText("review")
    config = window.get_current_scene_config()
    assert config.container == ".mkv"
    assert config.custom_suffix == "review"
    window._refresh_output_plan()
    assert window.batch_manager.queue[0].target_scene_name == "input_review.mkv"
    assert window.batch_manager.queue[0].target_full_path.endswith("input_review.mkv")
    assert "input_review.mkv" in window.table.item(0, 3).text()


@pytest.mark.parametrize(("attribute", "text", "message"), [("cb_threads", "0:2:1", "threads"), ("cb_tile_size", "16", "tile")])
def test_invalid_manual_engine_values_do_not_start_workers(window, monkeypatch, attribute, text, message):
    window.batch_manager.queue = [QueueItem(1, "C:/input.mp4", "input.mp4", "output.mp4", "C:/out/output.mp4")]
    window.txt_output.setText("C:/out")
    getattr(window, attribute).setEditText(text)
    factory = MagicMock()
    monkeypatch.setattr(main_window, "UpscaleWorker", factory)
    window.start_processing()
    factory.assert_not_called()
    assert message in window.lbl_overall_status.text().casefold()
    assert "engine" in window.settings_tabs.tabText(window.settings_tabs.currentIndex()).casefold()


def test_processing_controls_remain_inspectable_and_are_restored(window):
    window._set_processing_controls(True)
    assert window.settings_tabs.isEnabled()
    window.settings_tabs.setCurrentIndex(2)
    assert window.settings_tabs.currentIndex() == 2
    assert not window.cb_workers.isEnabled()
    assert not window.btn_remove_selected.isEnabled()
    assert not window.btn_start.isEnabled()
    assert window.btn_stop.isEnabled()
    assert window.txt_filter.isEnabled()
    window._set_processing_controls(False)
    assert window.cb_workers.isEnabled()
    assert window.btn_start.isEnabled()
    assert not window.btn_stop.isEnabled()


def test_remove_selected_keeps_other_queue_identities_and_source_files(window, tmp_path):
    sources = [tmp_path / f"input{index}.mp4" for index in (1, 2)]
    for source in sources:
        source.write_bytes(b"source")
    queue = [QueueItem(index, str(source), source.name, "output.mp4", "out/output.mp4")
             for index, source in enumerate(sources, 1)]
    window.batch_manager.queue = queue
    window._populate_table(queue)
    window.table.selectRow(1)
    window.on_remove_selected_clicked()
    assert window.batch_manager.queue == [queue[0]]
    assert all(source.is_file() for source in sources)
    assert "1 selected" in window.lbl_overall_status.text()


def test_stop_cancels_active_frame_preview(window):
    preview = MagicMock()
    preview.isRunning.return_value = True
    window._frame_worker = preview
    window.on_stop_clicked()
    preview.pipeline.cancel.assert_called_once()
    preview.isRunning.return_value = False


@pytest.mark.parametrize("attribute", ["cb_resolution", "cb_codec", "cb_workers", "cb_nvenc_preset", "cb_threads", "cb_tile_size", "slider_cq", "spin_cq"])
def test_scrolling_parameters_does_not_accidentally_change_values(window, qapp, attribute):
    window.show()
    control = getattr(window, attribute)
    reveal_control(window, control, qapp)
    if hasattr(control, "currentIndex"):
        control.setCurrentIndex(1)
    window.table.setFocus()
    control.clearFocus()
    before = control.currentIndex() if hasattr(control, "currentIndex") else control.value()
    center = control.rect().center()
    event = QWheelEvent(
        QPointF(center), QPointF(control.mapToGlobal(center)), QPoint(), QPoint(0, -120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
    )
    QApplication.sendEvent(control, event)
    after = control.currentIndex() if hasattr(control, "currentIndex") else control.value()
    assert after == before
