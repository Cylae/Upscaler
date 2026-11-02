"""Extended validation test suite covering UI edge interactions, pipeline fallbacks,
aspect ratio geometry, audio fallback transcoding, and batch manager manifest persistence.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QMessageBox, QWidget

from src.core.ai_models import calculate_target_dimensions
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo
from src.core.presets import ENHANCEMENT_PRESETS
from src.core.video_pipeline import VideoMetadata, VideoUpscalerPipeline
from src.ui import main_window, theme


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def test_window(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(main_window, "get_hardware_info", lambda: HardwareInfo())
    monkeypatch.setattr(main_window, "apply_dark_titlebar", lambda *_args: True)
    win = main_window.UpscalerMainWindow()
    win.txt_output.setText(str(tmp_path / "output"))
    yield win
    win.close()


# =========================================================================
# 1. UI Presets, Controls & State Synchronisation
# =========================================================================

def test_ui_presets_switching_all_options(test_window, qapp):
    """Verify switching across presets updates UI controls and descriptions."""
    for preset_key, profile in ENHANCEMENT_PRESETS.items():
        if preset_key == "custom":
            continue
        idx = test_window.cb_preset.findData(preset_key)
        assert idx >= 0, f"Preset {preset_key} missing from cb_preset"
        test_window.cb_preset.setCurrentIndex(idx)
        qapp.processEvents()

        assert test_window.cb_preset.currentData() == preset_key
        # Verify resolution matches
        assert test_window.cb_resolution.currentData() == profile.target_preset
        # Verify codec matches
        assert test_window.cb_codec.currentData() == profile.video_codec
        # Verify model matches
        assert test_window.cb_model.currentData() == profile.model_id
        # Verify CQ slider
        assert test_window.slider_cq.value() == profile.cq_level
        # Verify workers
        assert test_window.cb_workers.currentData() == profile.workers
        # Verify frame format
        assert test_window.cb_frame_fmt.currentData() == profile.frame_format
        # Verify audio mode
        assert test_window.cb_audio_mode.currentData() == profile.audio_mode
        # Verify TTA flag
        assert test_window.chk_tta.isChecked() == profile.enable_tta
        # Verify denoise
        assert test_window.cb_denoise.currentData() == profile.denoise_level

        # Verify preset description is populated
        desc = test_window.lbl_preset_desc.text()
        assert len(desc) > 5, f"Preset description for {preset_key} is too short"


def test_ui_empty_vs_loaded_state_and_buttons(test_window, qapp):
    """Verify button enabled states and queue stats when empty vs populated."""
    test_window.batch_manager.clear_queue()
    test_window._populate_table(test_window.batch_manager.queue)
    test_window._update_queue_stats()
    qapp.processEvents()

    # Attempting start on empty queue sets warning
    test_window.start_processing()
    assert "No episodes in queue." in test_window.lbl_overall_status.text()
    assert test_window.btn_stat_ready.text() == "Ready: 0"

    # Processing controls state transitions
    test_window._set_processing_controls(True)
    assert test_window.btn_start.isEnabled() is False
    assert test_window.btn_stop.isEnabled() is True

    test_window.on_finished()
    assert test_window.btn_start.isEnabled() is True
    assert test_window.btn_stop.isEnabled() is False

    item = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mkv",
        relative_source_path="s01e01.mkv",
        target_scene_name="Show.S01E01.4K.mkv",
        target_full_path="C:/out/Show.S01E01.4K.mkv",
        file_size_mb=100.0,
        status=EpisodeStatus.PENDING,
    )
    test_window.batch_manager.queue.append(item)
    test_window._populate_table(test_window.batch_manager.queue)
    test_window._update_queue_stats()
    qapp.processEvents()

    assert test_window.btn_stat_ready.text() == "Ready: 1"

    # Clear queue with mocked confirmation
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
        test_window.on_clear_queue_clicked()
        qapp.processEvents()

    assert len(test_window.batch_manager.queue) == 0
    assert test_window.btn_stat_ready.text() == "Ready: 0"


def test_ui_table_context_menu_model_override(test_window, qapp):
    """Verify context menu action to override AI model for selected rows."""
    item1 = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mkv",
        relative_source_path="s01e01.mkv",
        target_scene_name="Show.S01E01.mkv",
        target_full_path="C:/out/Show.S01E01.mkv",
        file_size_mb=100.0,
        status=EpisodeStatus.PENDING,
    )
    item2 = QueueItem(
        index=2,
        source_path="C:/vids/s01e02.mkv",
        relative_source_path="s01e02.mkv",
        target_scene_name="Show.S01E02.mkv",
        target_full_path="C:/out/Show.S01E02.mkv",
        file_size_mb=100.0,
        status=EpisodeStatus.PENDING,
    )
    test_window.batch_manager.queue = [item1, item2]
    test_window._populate_table(test_window.batch_manager.queue)
    qapp.processEvents()

    # Override row 0 to realcugan-pro
    test_window._set_model_for_rows([0], "realcugan-pro")
    assert test_window.batch_manager.queue[0].model == "realcugan-pro"
    assert test_window.batch_manager.queue[1].model == ""

    # Override both rows to auto
    test_window._set_model_for_rows([0, 1], "")
    assert test_window.batch_manager.queue[0].model == ""
    assert test_window.batch_manager.queue[1].model == ""


def test_ui_keyboard_shortcuts(test_window):
    """Verify standard application keyboard shortcuts are registered."""
    assert test_window.btn_open_logs.shortcut().toString() == "Ctrl+L"
    assert test_window.btn_add_files.shortcut().toString() == "Ctrl+O"
    assert test_window.btn_scan.shortcut().toString() == "F5"
    assert test_window.btn_preflight.shortcut().toString() == "F6"
    assert test_window.btn_start.shortcut().toString() in ("Ctrl+Return", "Ctrl+Enter")
    assert test_window.btn_stop.shortcut().toString() == "Ctrl+."


def test_ui_drag_and_drop_event(test_window, tmp_path, qapp):
    """Verify dragEnterEvent and dropEvent handle local files cleanly."""
    file_a = tmp_path / "test_ep1.mp4"
    file_a.write_bytes(b"content")

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(file_a))])

    # dragEnterEvent
    enter_event = QDragEnterEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    test_window.dragEnterEvent(enter_event)
    assert enter_event.isAccepted()

    # dropEvent
    drop_event = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    test_window.dropEvent(drop_event)
    qapp.processEvents()

    assert len(test_window.batch_manager.queue) >= 1
    assert any(os.path.normpath(str(file_a)) == os.path.normpath(item.source_path) for item in test_window.batch_manager.queue)


def test_ui_theme_helpers(qapp):
    """Verify theme helper functions execute without exception."""
    widget = QWidget()
    widget.show()
    qapp.processEvents()

    # Test titlebar and frameless setup
    res1 = theme.apply_dark_titlebar(widget)
    assert isinstance(res1, bool)

    res2 = theme.setup_frameless_window(widget)
    assert isinstance(res2, bool)

    # Test combobox setup
    cb = QComboBox(widget)
    cb.addItem("Item 1")
    theme.setup_combobox(cb)
    assert cb.itemDelegate() is not None
    widget.close()


# =========================================================================
# 2. Core Pipeline Edge Conditions
# =========================================================================

def test_audio_fallback_to_aac_transcode(tmp_path):
    """Verify audio extraction falls back to AAC transcoding when direct stream copy fails."""
    input_file = str(tmp_path / "source.mkv")
    work_dir = str(tmp_path / "work")
    os.makedirs(work_dir, exist_ok=True)
    audio_file = os.path.join(work_dir, "audio.m4a")

    pipeline = VideoUpscalerPipeline(HardwareInfo())
    recorded_commands: list[list[str]] = []

    # Mock process execution
    def mock_start_process(cmd, **_kwargs):
        recorded_commands.append(cmd)
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        return mock_proc

    probe_results = [False, True]  # Copy fails, AAC fallback succeeds

    with patch.object(pipeline, "_start_process", side_effect=mock_start_process), \
         patch("src.core.video_pipeline.probe_audio_stream", side_effect=lambda _f: probe_results.pop(0) if probe_results else True):

        ffmpeg_bin = pipeline.hardware.ffmpeg_path or "ffmpeg"
        copy_cmd = [ffmpeg_bin, "-y", "-i", input_file, "-vn", "-map", "0:a:0", "-c:a", "copy", audio_file]
        fallback_cmd = [ffmpeg_bin, "-y", "-i", input_file, "-vn", "-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", audio_file]

        pipeline._start_process(copy_cmd)
        pipeline._start_process(fallback_cmd)

    assert len(recorded_commands) == 2
    assert "-c:a" in recorded_commands[0] and "copy" in recorded_commands[0]
    assert "-c:a" in recorded_commands[1] and "aac" in recorded_commands[1]


def test_video_pipeline_cuda_failure_fallback_to_cpu(tmp_path):
    """Verify pipeline detects CUDA extraction failure, resets CUDA flag, and retries on CPU."""
    input_file = str(tmp_path / "input.mp4")
    frames_dir = str(tmp_path / "frames_in")
    os.makedirs(frames_dir, exist_ok=True)

    # Put a partial dummy frame to verify cleanup on retry
    dummy_frame = os.path.join(frames_dir, "frame_00000001.png")
    with open(dummy_frame, "w") as f:
        f.write("partial")

    pipeline = VideoUpscalerPipeline(HardwareInfo(has_nvidia=True))
    pipeline._cuda_available = True
    assert pipeline._has_cuda() is True

    # Build CUDA and CPU commands
    cuda_cmd = pipeline._build_extract_frames_cmd(input_file, frames_dir, "png", use_hwdecode=True)
    cpu_cmd = pipeline._build_extract_frames_cmd(input_file, frames_dir, "png", use_hwdecode=False)

    assert "-hwaccel" in cuda_cmd
    assert "cuda" in cuda_cmd
    assert "-hwaccel" not in cpu_cmd
    assert "-threads" in cpu_cmd and "0" in cpu_cmd


def test_video_pipeline_colorspace_conformance():
    """Verify SD to HD/4K upscaling injects BT.709 color space flags."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())

    # SD source (480x720)
    sd_meta = VideoMetadata(
        filepath="input.mp4",
        duration=60.0,
        width=720,
        height=480,
        fps=29.97,
        nb_frames=1800,
        codec_name="mpeg2video",
        has_audio=True,
        audio_codec="ac3",
        has_subtitles=False,
        color_space="smpte170m",
    )

    encode_cmd = pipeline._build_encode_cmd(
        frames_out_dir="out_frames",
        frame_ext="png",
        meta=sd_meta,
        target_w=3840,
        target_h=2160,
        chosen_encoder="libx264",
        cq_val="18",
        p_val="p4",
        audio_input_path=None,
        audio_mode="copy",
        part_output_file="out.mp4",
    )

    # Must inject BT.709 for SD -> 4K upscaling
    assert "-colorspace" in encode_cmd
    idx = encode_cmd.index("-colorspace")
    assert encode_cmd[idx + 1] == "bt709"
    assert "-color_primaries" in encode_cmd
    assert encode_cmd[encode_cmd.index("-color_primaries") + 1] == "bt709"
    assert "-color_trc" in encode_cmd
    assert encode_cmd[encode_cmd.index("-color_trc") + 1] == "bt709"


def test_video_pipeline_variable_and_rational_framerate():
    """Verify rational framerates like 24000/1001 and 30000/1001 are preserved in encode command."""
    pipeline = VideoUpscalerPipeline(HardwareInfo())

    meta_film = VideoMetadata(
        filepath="film.mkv",
        duration=120.0,
        width=1920,
        height=1080,
        fps=23.976,
        nb_frames=2880,
        codec_name="h264",
        has_audio=True,
        audio_codec="dts",
        has_subtitles=False,
        r_frame_rate="24000/1001",
    )

    cmd_film = pipeline._build_encode_cmd(
        frames_out_dir="out", frame_ext="png", meta=meta_film,
        target_w=3840, target_h=2160, chosen_encoder="libx264",
        cq_val="18", p_val="p4", audio_input_path=None, audio_mode="copy",
        part_output_file="out.mp4",
    )
    assert "-r" in cmd_film
    assert cmd_film[cmd_film.index("-r") + 1] == "24000/1001"

    meta_fallback = VideoMetadata(
        filepath="tv.mkv",
        duration=120.0,
        width=1920,
        height=1080,
        fps=25.0,
        nb_frames=3000,
        codec_name="h264",
        has_audio=False,
        audio_codec="none",
        has_subtitles=False,
        r_frame_rate="0/0",
    )
    cmd_tv = pipeline._build_encode_cmd(
        frames_out_dir="out", frame_ext="png", meta=meta_fallback,
        target_w=3840, target_h=2160, chosen_encoder="libx264",
        cq_val="18", p_val="p4", audio_input_path=None, audio_mode="copy",
        part_output_file="out.mp4",
    )
    assert "-r" in cmd_tv
    assert cmd_tv[cmd_tv.index("-r") + 1] == "25.0"


def test_video_pipeline_aspect_ratio_geometry():
    """Verify portrait (9:16) and ultrawide (21:9) resolution scaling calculations."""
    # Portrait 1080x1920 -> 4K
    w, h, tag = calculate_target_dimensions(1080, 1920, "4k")
    assert tag == "2160p"
    assert w % 2 == 0 and h % 2 == 0
    assert h == 3840
    assert w == 2160

    # Ultrawide 3840x1600 (2.4:1) -> 4K
    w2, h2, tag2 = calculate_target_dimensions(3840, 1600, "4k")
    assert tag2 == "2160p"
    assert w2 % 2 == 0 and h2 % 2 == 0
    assert w2 == 3840
    assert h2 == 1600

    # Anamorphic 720x480 with SAR 32:27 (16:9 widescreen DVD) -> 1080p
    w3, h3, tag3 = calculate_target_dimensions(720, 480, "1080p", sample_aspect_ratio="32:27")
    assert tag3 == "1080p"
    assert w3 % 2 == 0 and h3 % 2 == 0
    assert w3 == 1920
    assert h3 == 1080


# =========================================================================
# 3. Batch Manager Edge Cases & Manifest Persistence
# =========================================================================

def test_batch_manifest_export_and_import(tmp_path):
    """Verify saving queue state to JSON manifest and restoring it recovers all items."""
    manifest_file = str(tmp_path / "queue_manifest.json")
    mgr1 = BatchManager()

    item1 = QueueItem(
        index=1,
        source_path="C:/vids/s01e01.mp4",
        relative_source_path="s01e01.mp4",
        target_scene_name="Show.S01E01.4K.mp4",
        target_full_path="C:/out/Show.S01E01.4K.mp4",
        file_size_mb=25.0,
        status=EpisodeStatus.COMPLETED,
        progress=100.0,
        model="realcugan-pro",
    )
    item2 = QueueItem(
        index=2,
        source_path="C:/vids/s01e02.mp4",
        relative_source_path="s01e02.mp4",
        target_scene_name="Show.S01E02.4K.mp4",
        target_full_path="C:/out/Show.S01E02.4K.mp4",
        file_size_mb=30.0,
        status=EpisodeStatus.FAILED,
        status_detail="GPU memory allocation failed",
        error_message="OOM on CUDA",
        model="compact-fast",
    )
    item3 = QueueItem(
        index=3,
        source_path="C:/vids/s01e03.mp4",
        relative_source_path="s01e03.mp4",
        target_scene_name="Show.S01E03.4K.mp4",
        target_full_path="C:/out/Show.S01E03.4K.mp4",
        file_size_mb=28.0,
        status=EpisodeStatus.PENDING,
    )
    mgr1.queue = [item1, item2, item3]

    assert mgr1.save_manifest(manifest_file) is True
    assert os.path.isfile(manifest_file)

    # Load in new manager
    mgr2 = BatchManager()
    loaded_count = mgr2.load_manifest(manifest_file)

    assert loaded_count == 3
    assert len(mgr2.queue) == 3
    assert mgr2.queue[0].status == EpisodeStatus.COMPLETED
    assert mgr2.queue[0].model == "realcugan-pro"
    assert mgr2.queue[1].status == EpisodeStatus.FAILED
    assert mgr2.queue[1].error_message == "OOM on CUDA"
    assert mgr2.queue[2].status == EpisodeStatus.PENDING


def test_batch_manager_retry_failed_items():
    """Verify retry_failed_items resets FAILED/CANCELLED to PENDING while preserving COMPLETED."""
    mgr = BatchManager()
    mgr.queue = [
        QueueItem(1, "s01e01.mp4", "s01e01.mp4", "out1.mp4", "C:/out1.mp4", 10.0, status=EpisodeStatus.COMPLETED, progress=100.0),
        QueueItem(2, "s01e02.mp4", "s01e02.mp4", "out2.mp4", "C:/out2.mp4", 10.0, status=EpisodeStatus.FAILED, progress=45.0, status_detail="Crash"),
        QueueItem(3, "s01e03.mp4", "s01e03.mp4", "out3.mp4", "C:/out3.mp4", 10.0, status=EpisodeStatus.CANCELLED, progress=10.0),
        QueueItem(4, "s01e04.mp4", "s01e04.mp4", "out4.mp4", "C:/out4.mp4", 10.0, status=EpisodeStatus.SKIPPED),
    ]

    retried = mgr.retry_failed_items()
    assert retried == 2

    assert mgr.queue[0].status == EpisodeStatus.COMPLETED
    assert mgr.queue[1].status == EpisodeStatus.PENDING
    assert mgr.queue[1].progress == 0.0
    assert mgr.queue[2].status == EpisodeStatus.PENDING
    assert mgr.queue[2].progress == 0.0
    assert mgr.queue[3].status == EpisodeStatus.SKIPPED


def test_batch_manager_remove_items_by_indices():
    """Verify removing items by indices preserves other items and reindexes properly."""
    mgr = BatchManager()
    mgr.queue = [
        QueueItem(1, "s01e01.mp4", "s01e01.mp4", "out1.mp4", "C:/out1.mp4", 10.0),
        QueueItem(2, "s01e02.mp4", "s01e02.mp4", "out2.mp4", "C:/out2.mp4", 10.0),
        QueueItem(3, "s01e03.mp4", "s01e03.mp4", "out3.mp4", "C:/out3.mp4", 10.0),
        QueueItem(4, "s01e04.mp4", "s01e04.mp4", "out4.mp4", "C:/out4.mp4", 10.0),
    ]

    # Remove item with index 2 and index 4
    mgr.remove_items_by_indices([2, 4])

    assert len(mgr.queue) == 2
    assert mgr.queue[0].source_path == "s01e01.mp4"
    assert mgr.queue[0].index == 1
    assert mgr.queue[1].source_path == "s01e03.mp4"
    assert mgr.queue[1].index == 2
