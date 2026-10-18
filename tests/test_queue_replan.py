"""Changing output settings replans queue rows without losing their identity."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.core import batch, validator
from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo
from src.core.models import calculate_target_dimensions
from src.core.probe import VideoMetadata
from src.core.scene_renamer import SceneNamingConfig, get_scene_relative_output_path


def _metadata(path, **changes):
    values = dict(
        filepath=str(path), width=1920, height=1080, fps=24, duration=120,
        nb_frames=2880, codec_name="h264", has_audio=True,
        audio_codec="aac", has_subtitles=False,
    )
    values.update(changes)
    return VideoMetadata(**values)


@pytest.fixture
def queued(tmp_path):
    source_dir = tmp_path / "source"
    source = source_dir / "Season 01" / "S01E02 - Title.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source placeholder")
    output_dir = tmp_path / "old-output"
    config = SceneNamingConfig(naming_mode="original")
    relative = get_scene_relative_output_path(str(source), str(source_dir), config)
    output = output_dir / relative
    output.parent.mkdir(parents=True)
    output.write_bytes(b"x" * 150_000)
    item = QueueItem(
        1, str(source), str(source.relative_to(source_dir)), output.name, str(output),
        status=EpisodeStatus.SKIPPED, progress=0.75, model="realesrgan-x4plus",
        file_eta="4s", error_message="old message",
    )
    manager = BatchManager(HardwareInfo())
    manager.queue = [item]
    return manager, item, source_dir, source, output_dir, output, config


def test_new_destination_cannot_skip_old_output_and_keeps_identity(queued, tmp_path, monkeypatch):
    manager, item, source_dir, source, _old_dir, _old_output, config = queued
    queue = manager.queue
    original_id = item.id
    original_model = item.model
    new_dir = tmp_path / "new-output"
    probe = Mock(side_effect=AssertionError("Missing destinations need no media probe"))
    monkeypatch.setattr(batch, "probe_video", probe)
    monkeypatch.setattr(manager, "scan_folder", Mock(side_effect=AssertionError("Do not rescan")))

    refreshed = manager.refresh_output_plan(str(new_dir), str(source_dir), config)

    assert refreshed is queue
    assert refreshed[0] is item
    assert item.id == original_id
    assert item.model == original_model
    assert item.source_path == str(source)
    assert item.target_full_path == str(new_dir / source.relative_to(source_dir))
    assert item.status == EpisodeStatus.PENDING
    assert item.progress == 0
    assert item.output_size_mb == 0
    assert item.file_eta == ""
    assert item.error_message == ""
    probe.assert_not_called()
    assert manager.plan_dry_run(str(new_dir), config)[0]["target_path"] == item.target_full_path


@pytest.mark.parametrize("invalid", ["geometry", "audio", "duration", "unreadable-source"])
def test_same_filename_cannot_skip_output_violating_new_contract(queued, monkeypatch, invalid):
    manager, item, source_dir, source, output_dir, output, config = queued
    item.status = EpisodeStatus.COMPLETED
    source_meta = _metadata(source)
    output_meta = replace(source_meta, filepath=str(output), width=7680, height=4320)
    if invalid == "geometry":
        output_meta = replace(output_meta, width=3840, height=2160)
    elif invalid == "audio":
        output_meta = replace(output_meta, has_audio=False)
    elif invalid == "duration":
        output_meta = replace(output_meta, duration=40)
    source_probe = Mock(return_value=source_meta)
    if invalid == "unreadable-source":
        source_probe.side_effect = ValueError("source metadata unavailable")
    monkeypatch.setattr(batch, "probe_video", source_probe)
    monkeypatch.setattr(validator, "_resolve_probe_video", lambda: lambda _path: output_meta)

    manager.refresh_output_plan(str(output_dir), str(source_dir), config, target_preset="8k")

    assert item.target_full_path == str(output)
    assert item.status == EpisodeStatus.PENDING
    assert item.progress == 0


@pytest.mark.parametrize(
    ("source_geometry", "expected_dimensions"),
    [({}, (7680, 4320)), ({"rotation": 90}, (4320, 7680)),
     ({"width": 720, "height": 576, "sample_aspect_ratio": "16:15"}, (5760, 4320))],
)
def test_new_valid_destination_checks_display_geometry_audio_and_duration(
    queued, tmp_path, monkeypatch, source_geometry, expected_dimensions
):
    manager, item, source_dir, source, _old_dir, _old_output, config = queued
    new_dir = tmp_path / "new-output"
    new_output = new_dir / source.relative_to(source_dir)
    new_output.parent.mkdir(parents=True)
    new_output.write_bytes(b"x" * 150_000)
    source_meta = _metadata(source, **source_geometry)
    output_meta = _metadata(
        new_output, width=expected_dimensions[0], height=expected_dimensions[1]
    )
    source_probe = Mock(return_value=source_meta)
    monkeypatch.setattr(batch, "probe_video", source_probe)
    monkeypatch.setattr(validator, "_resolve_probe_video", lambda: lambda _path: output_meta)
    validate = Mock(wraps=validator.is_file_complete_and_valid)
    monkeypatch.setattr(batch, "is_file_complete_and_valid", validate)

    manager.refresh_output_plan(str(new_dir), str(source_dir), config, target_preset="8k")

    source_probe.assert_called_once_with(str(source), manager.hardware.ffprobe_path)
    validate.assert_called_once_with(
        str(new_output), expect_audio=True, expected_width=expected_dimensions[0],
        expected_height=expected_dimensions[1], expected_duration=120,
    )
    assert item.status == EpisodeStatus.SKIPPED
    assert item.progress == 1
    assert item.output_size_mb == 0.1
    assert item.model == "realesrgan-x4plus"


def test_completed_same_valid_destination_preserves_completion(queued, monkeypatch):
    manager, item, source_dir, source, output_dir, output, config = queued
    item.status = EpisodeStatus.COMPLETED
    item.status_detail = "Completed in 15s"
    item.progress = 1.0
    source_meta = _metadata(source)
    width, height, _ = calculate_target_dimensions(source_meta.width, source_meta.height, "4k")
    output_meta = _metadata(output, width=width, height=height)
    monkeypatch.setattr(batch, "probe_video", lambda *_args: source_meta)
    monkeypatch.setattr(validator, "_resolve_probe_video", lambda: lambda _path: output_meta)

    manager.refresh_output_plan(str(output_dir), str(source_dir), config)

    assert item.status == EpisodeStatus.COMPLETED
    assert item.status_detail == "Completed in 15s"
    assert item.progress == 1


@pytest.mark.parametrize("status", [EpisodeStatus.COMPLETED, EpisodeStatus.SKIPPED])
def test_overwrite_requeues_without_probing_destination(queued, monkeypatch, status):
    manager, item, source_dir, _source, output_dir, _output, config = queued
    item.status = status
    source_probe = Mock(side_effect=AssertionError("Overwrite needs no resume validation"))
    monkeypatch.setattr(batch, "probe_video", source_probe)

    manager.refresh_output_plan(str(output_dir), str(source_dir), config, overwrite=True)

    assert item.status == EpisodeStatus.PENDING
    assert item.progress == 0
    source_probe.assert_not_called()


@pytest.mark.parametrize("running", [False, True])
def test_running_jobs_are_untouched(queued, tmp_path, monkeypatch, running):
    manager, item, source_dir, _source, _output_dir, _output, config = queued
    manager._is_running = running
    if not running:
        item.status = EpisodeStatus.PROCESSING
    old_values = vars(item).copy()
    source_probe = Mock(side_effect=AssertionError("Running jobs must not be reprobed"))
    monkeypatch.setattr(batch, "probe_video", source_probe)

    manager.refresh_output_plan(str(tmp_path / "new-output"), str(source_dir), config)

    assert vars(item) == old_values
    source_probe.assert_not_called()


@pytest.mark.parametrize("mode", ["scene", "clean", "original"])
@pytest.mark.parametrize("preserve_folders", [False, True])
def test_replanning_uses_current_naming_and_folder_options(queued, tmp_path, mode, preserve_folders):
    manager, item, source_dir, source, _output_dir, _output, _config = queued
    output_dir = tmp_path / "changed-output"
    config = SceneNamingConfig(naming_mode=mode, resolution_tag="4320p", group_tag="NEW")
    expected_relative = get_scene_relative_output_path(
        str(source), str(source_dir), config, preserve_folders=preserve_folders
    )

    manager.refresh_output_plan(
        str(output_dir), str(source_dir), config, preserve_folders,
        target_preset="8k",
    )

    assert item.target_full_path == str(output_dir / expected_relative)
    assert item.target_scene_name == (output_dir / expected_relative).name
    assert item.status == EpisodeStatus.PENDING


@pytest.mark.parametrize("status", [EpisodeStatus.FAILED, EpisodeStatus.CANCELLED])
def test_unchanged_failed_jobs_keep_retry_state(queued, monkeypatch, status):
    manager, item, source_dir, _source, output_dir, _output, config = queued
    item.status = status
    item.status_detail = "User action needed"
    monkeypatch.setattr(batch, "probe_video", Mock(side_effect=ValueError("bad source")))

    manager.refresh_output_plan(str(output_dir), str(source_dir), config)

    assert item.status == status
    assert item.status_detail == "User action needed"


def test_direct_file_input_uses_its_parent_as_naming_base(queued, tmp_path):
    manager, item, _source_dir, source, _output_dir, _output, _config = queued
    output_dir = tmp_path / "changed-output"
    config = SceneNamingConfig(naming_mode="scene")
    expected = get_scene_relative_output_path(str(source), str(source.parent), config)

    manager.refresh_output_plan(str(output_dir), str(source), config)

    assert item.target_full_path == str(output_dir / expected)


@pytest.mark.parametrize("workers", [1, 2])
def test_existing_full_output_does_not_skip_a_requested_clip_preview(queued, monkeypatch, workers):
    from src.core.video_pipeline import VideoUpscalerPipeline

    manager, item, _source_dir, _source, output_dir, _output, config = queued
    item.status = EpisodeStatus.SKIPPED
    render = Mock(return_value=(True, "rendered_preview.mp4"))
    monkeypatch.setattr(VideoUpscalerPipeline, "process_video", render)
    manager.run_batch(str(output_dir), str(_source_dir), scene_config=config,
                      preview_duration_s=5.0, max_workers=workers)
    render.assert_called_once()
    assert render.call_args.args[0].preview_duration_s == 5.0
    assert item.status == EpisodeStatus.PENDING
    assert "Preview completed" in item.status_detail


@pytest.mark.parametrize("override", ["", "realcugan-pro"])
def test_progress_model_information_never_becomes_a_per_file_override(queued, monkeypatch, override, tmp_path):
    from src.core.video_pipeline import VideoUpscalerPipeline

    manager, item, source_dir, _source, output_dir, _output, config = queued
    item.model = override
    seen = []

    def render(self, job, progress_callback):
        seen.append(job.model_id)
        progress_callback(0.5, "Selected model: Reported Model (Scale 4x)", {"model_name": "Reported Model"})
        return True, "completed.mp4"

    monkeypatch.setattr(VideoUpscalerPipeline, "process_video", render)
    for global_model in ("realesr-animevideov3", "realesrgan-x4plus"):
        manager.run_batch(str(output_dir), str(source_dir), scene_config=config,
                          model_id=global_model, overwrite=True)
    assert seen == ([override, override] if override else ["realesr-animevideov3", "realesrgan-x4plus"])
    assert item.model == override
    assert item.resolved_model == "Reported Model"
    manifest = tmp_path / "queue.json"
    assert manager.save_manifest(str(manifest))
    restored = BatchManager(manager.hardware)
    assert restored.load_manifest(str(manifest)) == 1
    assert restored.queue[0].model == override
    assert restored.queue[0].resolved_model == "Reported Model"
