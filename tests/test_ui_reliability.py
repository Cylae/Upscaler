import os
import sys
import pytest
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.core.batch_manager import BatchManager, QueueItem, EpisodeStatus
from src.core.detector import HardwareInfo
from src.ui.app_qt import UpscalerMainWindow

# Ensure offscreen Qt platform for headless CI / testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0]])
    return app


def test_ui_table_identity_and_filtering(qapp):
    """Verify table row lookup uses stable item ID and handles filtering properly."""
    window = UpscalerMainWindow()
    
    # Create two test items
    item1 = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mp4",
        relative_source_path="s01e01.mp4",
        target_scene_name="Show.S01E01.mp4",
        target_full_path="C:/out/Show.S01E01.mp4",
        file_size_mb=25.0,
        status=EpisodeStatus.COMPLETED
    )
    item2 = QueueItem(
        index=2,
        source_path="C:/vids/s01e02.mp4",
        relative_source_path="s01e02.mp4",
        target_scene_name="Show.S01E02.mp4",
        target_full_path="C:/out/Show.S01E02.mp4",
        file_size_mb=30.0,
        status=EpisodeStatus.FAILED,
        status_detail="Encoder error"
    )

    window.batch_manager.queue = [item1, item2]
    window._populate_table(window.batch_manager.queue)

    assert window.table.rowCount() == 2

    # Verify stable ID stored in table
    row0_id = window.table.item(0, 0).data(Qt.ItemDataRole.UserRole)
    row1_id = window.table.item(1, 0).data(Qt.ItemDataRole.UserRole)
    assert row0_id == item1.id
    assert row1_id == item2.id

    # Test _get_queue_item_from_row
    assert window._get_queue_item_from_row(0).id == item1.id
    assert window._get_queue_item_from_row(1).id == item2.id

    # Test _find_row_by_item_id
    assert window._find_row_by_item_id(item1.id) == 0
    assert window._find_row_by_item_id(item2.id) == 1
    assert window._find_row_by_item_id("non-existent-id") == -1

    # Test filtering by text
    window.txt_filter.setText("s01e02")
    window.apply_table_filters()
    assert window.table.isRowHidden(0) is True
    assert window.table.isRowHidden(1) is False

    # Reset filter
    window.txt_filter.setText("")
    window.apply_table_filters()
    assert window.table.isRowHidden(0) is False
    assert window.table.isRowHidden(1) is False

    # Test retry failed
    window.on_retry_failed_clicked()
    assert item2.status == EpisodeStatus.PENDING
    assert "Retried 1 failed/cancelled" in window.lbl_overall_status.text()


def test_ui_dynamic_resolution_suffix(qapp):
    """Verify changing resolution updates output folder tag dynamically."""
    window = UpscalerMainWindow()
    window.txt_source.setText("C:/Videos")
    window.txt_output.setText("C:/Videos (2160p)")

    # Change to 1080p (index 2)
    window.cb_resolution.setCurrentIndex(2)
    assert window.txt_output.text() == "C:/Videos (1080p)"

    # Change to 1440p (index 1)
    window.cb_resolution.setCurrentIndex(1)
    assert window.txt_output.text() == "C:/Videos (1440p)"

    # Change to 720p (index 3)
    window.cb_resolution.setCurrentIndex(3)
    assert window.txt_output.text() == "C:/Videos (720p)"

    # Change back to 4K (index 0)
    window.cb_resolution.setCurrentIndex(0)
    assert window.txt_output.text() == "C:/Videos (2160p)"


def test_ui_preflight_summary_and_dialog_execution(qapp, monkeypatch):
    """Verify on_preflight_clicked constructs preflight data and opens PreflightDialog without AttributeError."""
    window = UpscalerMainWindow()
    item = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mp4",
        relative_source_path="s01e01.mp4",
        target_scene_name="Show.S01E01.mp4",
        target_full_path="C:/out/Show.S01E01.mp4",
        file_size_mb=25.0,
        status=EpisodeStatus.PENDING
    )
    window.batch_manager.queue = [item]
    window.txt_source.setText("C:/vids")
    window.txt_output.setText("C:/out")

    dialog_opened = False
    def mock_exec(self):
        nonlocal dialog_opened
        dialog_opened = True
        return 0

    monkeypatch.setattr("src.ui.app_qt.PreflightDialog.exec", mock_exec)
    window.on_preflight_clicked()
    assert dialog_opened is True


def test_ui_1080p_layout_and_column_distribution(qapp):
    """Verify that on a 1920x1080 screen, the UI elements and columns are properly proportioned."""
    from PyQt6.QtWidgets import QHeaderView, QSplitter
    window = UpscalerMainWindow()
    window.resize(1920, 1080)
    window.show()
    qapp.processEvents()

    # Verify splitter exists and has proper configuration
    assert hasattr(window, "main_splitter")
    assert isinstance(window.main_splitter, QSplitter)
    assert window.main_splitter.childrenCollapsible() is False

    # Verify table column configuration
    header = window.table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(4) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(5) == QHeaderView.ResizeMode.Fixed
    assert header.sectionResizeMode(6) == QHeaderView.ResizeMode.Interactive

    # Populate sample items and verify column widths are balanced
    item1 = QueueItem(
        index=1,
        source_path="C:/vids/s01e01 - Spin Fun Knowin Ya.mp4",
        relative_source_path="s01e01 - Spin Fun Knowin Ya.mp4",
        target_scene_name="s01e01 - Spin Fun Knowin Ya_4K.mp4",
        target_full_path="C:/out/s01e01 - Spin Fun Knowin Ya_4K.mp4",
        file_size_mb=8.9,
        status=EpisodeStatus.PENDING
    )
    window.batch_manager.queue = [item1]
    window._populate_table(window.batch_manager.queue)
    qapp.processEvents()

    # Column 1 and Column 3 should both receive generous width (>300px) on 1080p
    col1_w = window.table.columnWidth(1)
    col3_w = window.table.columnWidth(3)
    assert col1_w >= 300, f"Original File column width too small: {col1_w}"
    assert col3_w >= 300, f"Target Scene Name column width too small: {col3_w}"

    # Verify preview duration spinbox has sufficient width for "30 s"
    assert window.spin_preview_dur.width() >= 80

    # Verify lbl_eta is hidden by default (no ugly empty border bubble)
    assert window.lbl_eta.isVisible() is False

    # Verify lbl_eta shows when eta_str provided, and hides on finish
    window.on_batch_updated(1, 10, 10.0, "Processing...", eta_str="ETA: 02:45")
    assert window.lbl_eta.isVisible() is True
    assert window.lbl_eta.text() == "ETA: 02:45"

    window.on_finished()
    assert window.lbl_eta.isVisible() is False

    # Verify audio chip is concise
    window.update_scene_preview()
    assert len(window.lbl_audio.text()) < 30


def test_ui_close_event_clean_shutdown(qapp):
    """Verify closeEvent safely requests batch stop and cleans up workers without exception."""
    from PyQt6.QtGui import QCloseEvent
    window = UpscalerMainWindow()
    window.show()
    qapp.processEvents()

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_compare_frame_button_lifecycle(qapp):
    """Verify btn_compare_frame is disabled during batch execution and re-enabled on finish."""
    window = UpscalerMainWindow()
    assert window.btn_compare_frame.isEnabled() is True

    # Simulate batch launch
    item = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mp4",
        relative_source_path="s01e01.mp4",
        target_scene_name="Show.S01E01.mp4",
        target_full_path="C:/out/Show.S01E01.mp4",
        file_size_mb=25.0,
        status=EpisodeStatus.PENDING
    )
    window.batch_manager.queue = [item]
    window.txt_source.setText("C:/vids")
    window.txt_output.setText("C:/out")

    with patch("src.ui.main_window.UpscaleWorker.start"):
        window.start_processing()
        assert window.btn_compare_frame.isEnabled() is False

    window.on_finished()
    assert window.btn_compare_frame.isEnabled() is True


def test_frame_comparison_dialog_done_cleans_temp(qapp, tmp_path):
    """Verify FrameComparisonDialog.done cleans up temporary preview folder."""
    from src.ui.dialogs.comparison import FrameComparisonDialog

    preview_dir = tmp_path / "preview_temp_123"
    preview_dir.mkdir()
    frame1 = preview_dir / "orig.png"
    frame2 = preview_dir / "up.png"
    frame1.write_bytes(b"\x00" * 100)
    frame2.write_bytes(b"\x00" * 100)

    dlg = FrameComparisonDialog(
        orig_path=str(frame1),
        upscaled_path=str(frame2),
        info={"temp_dir": str(preview_dir)},
    )
    assert preview_dir.exists()

    # Calling done(0) simulates closing dialog via accept or reject
    dlg.done(0)
    assert not preview_dir.exists()


def test_video_pipeline_respects_custom_ffmpeg_bin():
    """Verify VideoUpscalerPipeline commands use custom ffmpeg binary path from HardwareInfo."""
    from src.core.video_pipeline import VideoUpscalerPipeline
    from src.core.probe import VideoMetadata

    custom_hw = HardwareInfo(
        gpu_name="Test GPU",
        gpu_vram_gb=8.0,
        ffmpeg_path="C:/Custom/bin/ffmpeg.exe",
    )
    pipeline = VideoUpscalerPipeline(custom_hw)

    extract_cmd = pipeline._build_extract_frames_cmd(
        input_file="in.mp4",
        frames_in_dir="frames_in",
        frame_ext="jpg",
        use_hwdecode=False,
    )
    assert extract_cmd[0] == "C:/Custom/bin/ffmpeg.exe"

    dummy_meta = VideoMetadata(
        filepath="in.mp4",
        width=1280, height=720, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="h264", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    encode_cmd = pipeline._build_encode_cmd(
        frames_out_dir="frames_out",
        frame_ext="jpg",
        meta=dummy_meta,
        target_w=3840,
        target_h=2160,
        chosen_encoder="libx265",
        cq_val="19",
        p_val="fast",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mp4",
    )
    assert encode_cmd[0] == "C:/Custom/bin/ffmpeg.exe"


def test_ui_combobox_delegates_and_frameless_controls(qapp):
    """Verify that all QComboBoxes have QStyledItemDelegate and frameless controls exist."""
    from PyQt6.QtWidgets import QComboBox, QStyledItemDelegate
    from src.ui.theme import setup_combobox, apply_dark_titlebar

    win = UpscalerMainWindow()
    combos = win.findChildren(QComboBox)
    assert len(combos) >= 10, f"Expected at least 10 comboboxes, found {len(combos)}"

    for cb in combos:
        assert isinstance(cb.itemDelegate(), QStyledItemDelegate), f"Combobox {cb} does not have QStyledItemDelegate"
        assert cb.view().autoFillBackground() is True

    # Test window control buttons
    assert hasattr(win, "btn_win_min")
    assert hasattr(win, "btn_win_max")
    assert hasattr(win, "btn_win_close")

    # Test maximize toggle
    assert not win.isMaximized()
    win._toggle_maximize()
    assert win.isMaximized()
    win._toggle_maximize()
    assert not win.isMaximized()

    # Test helper functions
    test_cb = QComboBox()
    setup_combobox(test_cb)
    assert isinstance(test_cb.itemDelegate(), QStyledItemDelegate)
    assert test_cb.view().autoFillBackground() is True
    # Test titlebar helper does not crash
    apply_dark_titlebar(win)


def test_ui_table_shows_model_per_file(qapp):
    """Verify table displays AI model used per file and updates dynamically."""
    window = UpscalerMainWindow()

    item1 = QueueItem(
        index=1,
        source_path="C:/vids/s01e01_sd_480p.mp4",
        relative_source_path="s01e01_sd_480p.mp4",
        target_scene_name="Show.S01E01.mp4",
        target_full_path="C:/out/Show.S01E01.mp4",
        file_size_mb=25.0,
        resolution_str="640x480",
        status=EpisodeStatus.PENDING,
    )
    item2 = QueueItem(
        index=2,
        source_path="C:/vids/s01e02_hd_720p.mp4",
        relative_source_path="s01e02_hd_720p.mp4",
        target_scene_name="Show.S01E02.mp4",
        target_full_path="C:/out/Show.S01E02.mp4",
        file_size_mb=50.0,
        resolution_str="1280x720",
        status=EpisodeStatus.PENDING,
        model="realcugan-pro",
    )

    window.batch_manager.queue = [item1, item2]
    window._populate_table(window.batch_manager.queue)

    assert window.table.columnCount() == 7
    assert window.table.horizontalHeaderItem(4).text() == "Model"

    # item1 in auto mode with 480p resolution -> Anime 6B
    row0_model = window.table.item(0, 4).text()
    assert "Anime 6B" in row0_model or "Auto" in row0_model

    # item2 has explicit model override "realcugan-pro"
    row1_model = window.table.item(1, 4).text()
    assert "RealCUGAN Pro" in row1_model

    # Changing model dropdown in UI updates the table
    for i in range(window.cb_model.count()):
        if "AnimeVideo V3" in window.cb_model.itemText(i):
            window.cb_model.setCurrentIndex(i)
            break
    qapp.processEvents()
    # item1 (no explicit override) now reflects the chosen model
    assert "AnimeVideo V3" in window.table.item(0, 4).text()
    # item2 (explicit override) preserves RealCUGAN Pro
    assert "RealCUGAN Pro" in window.table.item(1, 4).text()

    # on_item_updated updates model in table
    item1.model = "realesrgan-x4plus-anime"
    window.on_item_updated(item1)
    assert "Anime 6B" in window.table.item(0, 4).text()

    # Search filter includes Model column
    window.txt_filter.setText("RealCUGAN")
    window.apply_table_filters()
    assert window.table.isRowHidden(0) is True
    assert window.table.isRowHidden(1) is False


