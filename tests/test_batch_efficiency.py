"""Batch overhead stays bounded without changing media jobs or validation."""

import concurrent.futures
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core import batch, scene_renamer
from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo
from src.core.scene_renamer import SceneNamingConfig


def _item(index, status=EpisodeStatus.PENDING):
    return QueueItem(
        index, f"source-{index}.mp4", f"source-{index}.mp4",
        f"output-{index}.mp4", f"output-{index}.mp4", status=status,
    )


@pytest.mark.parametrize("mode", ["scene", "clean", "original"])
@pytest.mark.parametrize("preserve_folders", [False, True])
def test_scan_formats_each_name_once_and_keeps_mapping(
    tmp_path, monkeypatch, mode, preserve_folders
):
    source_dir = tmp_path / "source"
    source = source_dir / "Season 01" / "S01E02 - Episode Title.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * 120_000)
    output_dir = tmp_path / "output"
    config = SceneNamingConfig(naming_mode=mode)
    expected_relative = scene_renamer.get_scene_relative_output_path(
        str(source), str(source_dir), config, preserve_folders=preserve_folders
    )
    name_formatter = Mock(wraps=scene_renamer.format_output_name)
    monkeypatch.setattr(scene_renamer, "format_output_name", name_formatter)
    monkeypatch.setattr(batch, "format_output_name", name_formatter)
    validator = Mock(side_effect=AssertionError("Missing outputs need no media probe"))
    monkeypatch.setattr(batch, "is_file_complete_and_valid", validator)
    monkeypatch.setattr(batch, "probe_video", Mock(side_effect=AssertionError("No existing output")))

    manager = BatchManager(HardwareInfo())
    items = manager.scan_folder(str(source_dir), str(output_dir), config, preserve_folders)

    assert len(items) == 1
    assert items[0].target_full_path == str(output_dir / expected_relative)
    assert items[0].target_scene_name == os.path.basename(expected_relative)
    assert items[0].file_size_mb == 0.1
    assert items[0].status == EpisodeStatus.PENDING
    name_formatter.assert_called_once_with(str(source), str(source_dir), config)
    validator.assert_not_called()


@pytest.mark.parametrize("method", ["scan_folder", "add_files"])
@pytest.mark.parametrize("valid", [False, True])
def test_existing_outputs_keep_source_audio_and_duration_validation(
    tmp_path, monkeypatch, method, valid
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = SceneNamingConfig(naming_mode="original")
    output = output_dir / "source.mp4"
    output.write_bytes(b"existing output" * 10_000)
    source_probe = Mock(return_value=SimpleNamespace(has_audio=True, duration=73.5))
    validator = Mock(return_value=valid)
    monkeypatch.setattr(batch, "probe_video", source_probe)
    monkeypatch.setattr(batch, "is_file_complete_and_valid", validator)
    manager = BatchManager(HardwareInfo())

    source_argument = [str(source)] if method == "add_files" else str(source)
    items = getattr(manager, method)(source_argument, str(output_dir), config)

    source_probe.assert_called_once_with(str(source), manager.hardware.ffprobe_path)
    validator.assert_called_once_with(str(output), expect_audio=True, expected_duration=73.5)
    assert items[0].status == (EpisodeStatus.SKIPPED if valid else EpisodeStatus.PENDING)


def test_progress_cost_does_not_grow_with_inactive_queue_size(monkeypatch):
    status_reads = [0]

    class CountedItem(QueueItem):
        def __getattribute__(self, name):
            if name == "status":
                status_reads[0] += 1
            return super().__getattribute__(name)

    class ProgressPipeline:
        def __init__(self, _hardware):
            pass

        def process_video(self, _job, progress_callback):
            for index in range(1, 201):
                progress_callback(index / 200, "Upscaling frames", {})
            return True, "complete"

    monkeypatch.setattr("src.core.batch_manager.VideoUpscalerPipeline", ProgressPipeline)
    manager = BatchManager(HardwareInfo())
    queue_size = 2_000
    manager.queue = [
        CountedItem(
            index, f"source-{index}.mp4", f"source-{index}.mp4",
            f"output-{index}.mp4", f"output-{index}.mp4",
            status=EpisodeStatus.PENDING if index == 1 else EpisodeStatus.SKIPPED,
        )
        for index in range(1, queue_size + 1)
    ]
    updates = []

    manager.run_batch("out", ".", batch_callback=lambda *args: updates.append(args))

    # A full queue scan per frame would cause over 400,000 status reads.
    assert status_reads[0] < queue_size * 3
    frame_updates = [update for update in updates if "Upscaling frames" in update[3]]
    assert len(frame_updates) == 200
    assert frame_updates[0][2] == pytest.approx((1 / 200) / queue_size * 100)
    assert frame_updates[-1][2] == pytest.approx(1 / queue_size * 100)
    assert updates[-1][:3] == (queue_size, queue_size, 100.0)


def test_cancellation_never_enqueues_the_whole_library(monkeypatch):
    manager = BatchManager(HardwareInfo())
    manager.queue = [_item(index) for index in range(1, 101)]
    started = []
    submitted = []
    cancellations = []
    both_started = threading.Event()
    start_lock = threading.Lock()
    real_executor = concurrent.futures.ThreadPoolExecutor

    class TrackingExecutor(real_executor):
        def submit(self, callback, item):
            submitted.append(item.index)
            return super().submit(callback, item)

    class CancelPipeline:
        def __init__(self, _hardware):
            pass

        def process_video(self, job, progress_callback):
            with start_lock:
                started.append(job.input_file)
                if len(started) == 2:
                    manager.stop()
                    both_started.set()
            assert both_started.wait(5), "Both workers should start concurrently"
            return False, "cancelled"

        def cancel(self):
            cancellations.append(self)

    monkeypatch.setattr(batch.concurrent.futures, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr("src.core.batch_manager.VideoUpscalerPipeline", CancelPipeline)

    manager.run_batch("out", ".", max_workers=2)

    assert len(submitted) == 2
    assert len(started) == 2
    assert len(cancellations) == 2
    assert [item.status for item in manager.queue[:2]] == [EpisodeStatus.CANCELLED] * 2
    assert all(item.status == EpisodeStatus.PENDING for item in manager.queue[2:])
    assert not manager.is_running
    assert not manager._active_pipelines


def test_bounded_scheduler_finishes_all_jobs_with_selected_quality_settings(monkeypatch):
    manager = BatchManager(HardwareInfo())
    manager.queue = [_item(index) for index in range(1, 31)]
    manager.queue[0].status = EpisodeStatus.SKIPPED
    jobs = []

    class CapturingPipeline:
        def __init__(self, _hardware):
            pass

        def process_video(self, job, progress_callback):
            jobs.append(job)
            progress_callback(0.75, "Encoding", {"file_eta_str": "1s"})
            return True, "complete"

    monkeypatch.setattr("src.core.batch_manager.VideoUpscalerPipeline", CapturingPipeline)
    selected = {
        "target_preset": "8k", "video_codec": "hevc_nvenc",
        "model_id": "realesrgan-x4plus", "threads": "2:4:2",
        "frame_format": "png", "cq_level": 18, "nvenc_preset": "p7",
        "enable_tta": True, "denoise_level": 2, "audio_mode": "copy",
        "tile_size": 128, "preserve_folders": True, "overwrite": True,
    }

    manager.run_batch("out", ".", max_workers=3, **selected)

    assert len(jobs) == len(manager.queue)
    assert len({job.input_file for job in jobs}) == len(manager.queue)
    assert all(item.status == EpisodeStatus.COMPLETED for item in manager.queue)
    assert not manager._active_pipelines
    for job in jobs:
        for option, value in selected.items():
            assert getattr(job, option) == value


def test_already_validated_queue_needs_no_worker_futures(monkeypatch):
    manager = BatchManager(HardwareInfo())
    manager.queue = [_item(index, EpisodeStatus.SKIPPED) for index in range(1, 501)]
    callbacks = []
    updates = []
    real_executor = concurrent.futures.ThreadPoolExecutor

    class NoJobsExecutor(real_executor):
        def submit(self, *_args, **_kwargs):
            raise AssertionError("Skipped files need no queued worker task")

    monkeypatch.setattr(batch.concurrent.futures, "ThreadPoolExecutor", NoJobsExecutor)

    manager.run_batch(
        "out", ".", max_workers=4,
        item_callback=lambda item: callbacks.append(item.index),
        batch_callback=lambda *args: updates.append(args),
    )

    assert callbacks == list(range(1, 501))
    assert updates[-1][:3] == (500, 500, 100.0)
    assert all(item.status == EpisodeStatus.SKIPPED for item in manager.queue)
