"""Queue and log workloads stay bounded without losing UI state or records."""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.core.batch import EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo
from src.ui import main_window
from src.ui.dialogs.logs import LogViewerDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr(main_window, "get_hardware_info", lambda: HardwareInfo())
    win = main_window.UpscalerMainWindow()
    yield win
    win.close()


def make_item(index, status=EpisodeStatus.PENDING):
    name = f"episode{index:04}.mp4"
    return QueueItem(index, f"C:/vids/{name}", name, name, f"C:/out/{name}", status=status)


def test_filters_and_progress_do_not_rescan_queue(window):
    class TrackedQueue(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    queue = TrackedQueue(make_item(index) for index in range(1, 201))
    window.batch_manager.queue = queue
    window._populate_table(queue)
    queue.iterations = 0

    window.txt_filter.setText("episode")
    window.set_filter_mode("ready")
    last = queue[-1]
    last.status = EpisodeStatus.PROCESSING
    for progress in range(20):
        last.status_detail = f"Upscaling {progress}%"
        window.on_item_updated(last)

    assert queue.iterations == 0
    assert window.table.isRowHidden(199)
    assert window.table.item(199, 6).text() == "Upscaling 19%"
    assert window.btn_stat_ready.text() == "Ready: 199"


def test_status_counters_follow_completion_failure_cancellation_and_retry(window):
    queue = [make_item(1), make_item(2), make_item(3, EpisodeStatus.SKIPPED)]
    window.batch_manager.queue = queue
    window._populate_table(queue)

    for status in (EpisodeStatus.PROCESSING, EpisodeStatus.FAILED, EpisodeStatus.FAILED):
        queue[0].status = status
        window.on_item_updated(queue[0])
    queue[1].status = EpisodeStatus.CANCELLED
    window.on_item_updated(queue[1])
    assert window.btn_stat_ready.text() == "Ready: 0"
    assert window.btn_stat_failed.text() == "Errors: 1"
    assert window.btn_stat_skipped.text() == "Skipped: 1"

    window.on_retry_failed_clicked()
    assert window.btn_stat_ready.text() == "Ready: 2"
    assert window.btn_stat_failed.text() == "Errors: 0"
    queue[0].status = EpisodeStatus.COMPLETED
    window.on_item_updated(queue[0])
    window.on_item_updated(queue[0])
    assert window.btn_stat_completed.text() == "Completed: 1"
    assert window.btn_stat_ready.text() == "Ready: 1"


def test_queue_identity_survives_reordering_and_stale_worker_update(window):
    queue = [make_item(1), make_item(2)]
    window.batch_manager.queue = queue
    window._populate_table(queue)
    window.table.sortItems(1, Qt.SortOrder.DescendingOrder)
    queue[0].status = EpisodeStatus.FAILED
    queue[0].status_detail = "Encoder failure"
    window.on_item_updated(queue[0])
    assert window._find_row_by_item_id(queue[0].id) == 1
    assert window._get_queue_item_from_row(1) is queue[0]
    assert window.table.item(1, 6).text() == "Encoder failure"

    removed = queue.pop(0)
    window._populate_table(queue)
    window.on_item_updated(removed)
    assert window.table.item(0, 5).text() == "● Pending"
    assert window.btn_stat_failed.text() == "Errors: 0"


def test_live_logs_are_stored_once_and_dialog_is_reused(window):
    window._open_log_viewer()
    dialog = window._log_viewer
    for index in range(300):
        window.on_log_received(f"[VULKAN AI] event {index}")

    assert len(window._log_buffer) == 300
    assert dialog.lbl_count.text() == "300 lines"
    assert dialog.txt_log.toPlainText().count("event 299") == 1
    assert window.txt_console.document().blockCount() == 250
    assert "event 0\n" not in window.txt_console.toPlainText()
    assert dialog._color_for(window._log_buffer[-1]) == "#30d158"

    dialog.close()
    window.on_log_received("[ERROR] while hidden")
    window._open_log_viewer()
    assert window._log_viewer is dialog
    assert len(window.findChildren(LogViewerDialog)) == 1
    assert "while hidden" in dialog.txt_log.toPlainText()
    window.clear_all_logs()
    assert dialog.txt_log.toPlainText() == ""
    assert dialog.lbl_count.text() == "0 lines"


def test_log_reload_and_append_keep_document_bounded_without_dropping_export_data(qapp):
    logs = [f"[AUDIO] line {index}" for index in range(5000)]
    dialog = LogViewerDialog(logs)
    dialog.append_line("[WARNING] final <frame> & audio")
    assert len(logs) == 5001
    assert dialog.txt_log.document().blockCount() == 5000
    assert "line 0\n" not in dialog.txt_log.toPlainText()
    assert "final <frame> & audio" in dialog.txt_log.toPlainText()
    dialog.txt_filter.setText("final")
    assert dialog.txt_log.toPlainText().strip() == "[WARNING] final <frame> & audio"
    dialog._copy_all()
    assert QApplication.clipboard().text().startswith("[AUDIO] line 0\n")
    dialog.close()
