import os
import sys
import pytest
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
    assert header.sectionResizeMode(5) == QHeaderView.ResizeMode.Interactive

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

