"""A frame comparison must exercise the inference settings selected for the batch."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.detector import HardwareInfo
from src.core.models import AVAILABLE_MODELS
from src.core.probe import VideoMetadata
from src.core.video_pipeline import UpscaleJobConfig, VideoUpscalerPipeline
from src.ui.workers import FramePreviewWorker


@pytest.fixture
def preview_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    esrgan = tmp_path / "realesrgan" / "upscale.exe"
    cugan = tmp_path / "realcugan" / "upscale.exe"
    for binary in (esrgan, cugan):
        binary.parent.mkdir()
        binary.write_bytes(b"binary")
    (esrgan.parent / "models").mkdir()
    for folder in ("models-pro", "models-se"):
        (cugan.parent / folder).mkdir()
    hardware = HardwareInfo(
        gpu_vram_gb=8.0,
        ffmpeg_path="preview-ffmpeg",
        realesrgan_bin=str(esrgan),
        realcugan_bin=str(cugan),
    )
    pipeline = VideoUpscalerPipeline(hardware)
    metadata = VideoMetadata(str(source), 640, 360, 24, 10, 240, "h264", False, "", False)
    monkeypatch.setattr("src.core.video_pipeline.probe_video", lambda *_args: metadata)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        output = command[command.index("-o") + 1] if "-o" in command else command[-1]
        Path(output).write_bytes(b"generated frame")
        return 0

    monkeypatch.setattr(pipeline, "_run_tracked", run)
    return pipeline, str(source), commands


@pytest.mark.parametrize("model_id", ["realesrgan-x4plus-anime", "realcugan-pro"])
@pytest.mark.parametrize("enable_tta", [False, True])
def test_preview_honors_manual_threads_tile_and_tta_for_both_engines(preview_pipeline, model_id, enable_tta):
    pipeline, source, commands = preview_pipeline
    ok, original, upscaled, info = pipeline.extract_and_upscale_single_frame(
        source, model_id=model_id, target_preset="720p",
        threads="2:3:2", tile_size=300, denoise_level=2, enable_tta=enable_tta,
    )

    assert ok and Path(original).is_file() and Path(upscaled).is_file()
    assert len(commands) == 3
    ai_command = commands[1]
    assert ai_command[ai_command.index("-j") + 1] == "2:3:2"
    assert ai_command[ai_command.index("-t") + 1] == "300"
    assert ("-x" in ai_command) == enable_tta
    assert ai_command[ai_command.index("-f") + 1] == "png"
    if model_id == "realcugan-pro":
        assert ai_command[ai_command.index("-n") + 1] == "2"
    else:
        assert ai_command[ai_command.index("-n") + 1] == AVAILABLE_MODELS[model_id].internal_name
    assert (info["target_width"], info["target_height"]) == (1280, 720)
    assert "scale=1280:720:flags=lanczos,setsar=1" in commands[2]


@pytest.mark.parametrize("model_id", ["realesrgan-x4plus-anime", "realcugan-pro"])
def test_preview_legacy_arguments_keep_automatic_defaults(preview_pipeline, monkeypatch, model_id):
    pipeline, source, commands = preview_pipeline
    threads = Mock(return_value="1:2:1")
    tile = Mock(return_value=200)
    monkeypatch.setattr("src.core.video_pipeline.get_optimal_thread_spec", threads)
    monkeypatch.setattr("src.core.video_pipeline._auto_tile_size", tile)

    # Offset remains a valid fourth positional argument for existing callers.
    result = pipeline.extract_and_upscale_single_frame(source, model_id, "720p", 2.0)

    assert result[0]
    threads.assert_called_once_with("auto", gpu_vram_gb=8.0, model=AVAILABLE_MODELS[model_id])
    tile.assert_called_once_with(8.0, model=AVAILABLE_MODELS[model_id])
    assert commands[0][commands[0].index("-ss") + 1] == "2.000"
    assert commands[1][commands[1].index("-j") + 1] == "1:2:1"
    assert commands[1][commands[1].index("-t") + 1] == "200"
    assert "-x" not in commands[1]
    if model_id == "realcugan-pro":
        assert commands[1][commands[1].index("-n") + 1] == "-1"


def test_manual_preview_tile_does_not_invoke_automatic_tuning(preview_pipeline, monkeypatch):
    pipeline, source, _commands = preview_pipeline
    tile = Mock(side_effect=AssertionError("Manual tile must be respected"))
    monkeypatch.setattr("src.core.video_pipeline._auto_tile_size", tile)

    assert pipeline.extract_and_upscale_single_frame(
        source, model_id="realcugan-pro", target_preset="720p", tile_size=400,
    )[0]
    tile.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [{}, {"threads": "4:4:4", "tile_size": 600, "denoise_level": 3, "enable_tta": True}],
)
def test_worker_forwards_selected_preview_settings_and_emits_result(options):
    result = (True, "original.png", "upscaled.png", {"target_width": 3840})
    pipeline = SimpleNamespace(extract_and_upscale_single_frame=Mock(return_value=result))
    worker = FramePreviewWorker(pipeline, "source.mp4", "realcugan-pro", "4k", **options)
    received = []
    worker.preview_ready.connect(lambda *args: received.append(args))

    worker.run()

    expected = {"threads": "auto", "tile_size": 0, "denoise_level": -1, "enable_tta": False}
    expected.update(options)
    pipeline.extract_and_upscale_single_frame.assert_called_once_with(
        "source.mp4", model_id="realcugan-pro", target_preset="4k", **expected,
    )
    assert received == [result]


def test_preview_worker_reports_settings_or_engine_error():
    pipeline = SimpleNamespace(extract_and_upscale_single_frame=Mock(side_effect=ValueError("Invalid model options")))
    worker = FramePreviewWorker(pipeline, "source.mp4", "auto", "4k", enable_tta=True)
    received = []
    worker.preview_ready.connect(lambda *args: received.append(args))

    worker.run()

    assert received == [(False, "", "", {"error": "Invalid model options"})]


def test_cancelled_frame_preview_does_no_work(preview_pipeline):
    pipeline, source, commands = preview_pipeline
    pipeline.cancel()
    result = pipeline.extract_and_upscale_single_frame(source)
    assert not result[0]
    assert "Cancelled" in result[3]["error"]
    assert commands == []


def test_preview_launch_failure_cleans_intermediate_directory(preview_pipeline, monkeypatch):
    pipeline, source, _commands = preview_pipeline
    monkeypatch.setattr(pipeline, "_run_tracked", Mock(side_effect=OSError("engine unavailable")))
    result = pipeline.extract_and_upscale_single_frame(source)
    assert not result[0]
    assert "engine unavailable" in result[3]["error"]
    assert not list((Path(source).parent / ".temp_upscaler").glob("preview_*"))


@pytest.mark.parametrize("model_id,attribute", [
    ("realesr-animevideov3", "realesrgan_bin"), ("realcugan-pro", "realcugan_bin"),
])
@pytest.mark.parametrize("stale", [None, "removed-engine.exe"])
def test_preview_rediscovers_engine_in_existing_hardware_snapshot(
    preview_pipeline, monkeypatch, model_id, attribute, stale,
):
    pipeline, source, commands = preview_pipeline
    paths = pipeline.hardware.realesrgan_bin, pipeline.hardware.realcugan_bin
    expected = getattr(pipeline.hardware, attribute)
    shared_hardware = pipeline.hardware
    setattr(shared_hardware, attribute, stale)
    monkeypatch.setattr("src.core.hardware.detect_ai_binaries", lambda _base: paths)

    result = pipeline.extract_and_upscale_single_frame(source, model_id, "720p")

    assert result[0], result[3]
    assert pipeline.hardware is shared_hardware
    assert getattr(shared_hardware, attribute) == expected
    assert commands[1][0] == expected
    if model_id == "realesr-animevideov3":
        assert commands[1][commands[1].index("-n") + 1] == AVAILABLE_MODELS[model_id].internal_name
    else:
        assert commands[1][commands[1].index("-m") + 1].endswith("models-pro")


@pytest.mark.parametrize("model_id", ["realesr-animevideov3", "realcugan-pro"])
@pytest.mark.parametrize("operation", ["frame", "video"])
def test_missing_engine_fails_before_extraction_or_encoder_probe(
    preview_pipeline, monkeypatch, tmp_path, model_id, operation,
):
    pipeline, source, commands = preview_pipeline
    pipeline.hardware.realesrgan_bin = None
    pipeline.hardware.realcugan_bin = None
    monkeypatch.setattr("src.core.hardware.get_base_dir", lambda: str(tmp_path / "no-engines"))
    monkeypatch.setattr("src.core.hardware.shutil.which", lambda _name: None)
    encoder = Mock(side_effect=AssertionError("No encoder probe before engine discovery"))
    monkeypatch.setattr(pipeline, "_resolve_encoder", encoder)
    start = Mock(side_effect=AssertionError("No extraction without the chosen AI engine"))
    monkeypatch.setattr(pipeline, "_start_process", start)

    if operation == "frame":
        ok, _original, _upscaled, info = pipeline.extract_and_upscale_single_frame(source, model_id, "720p")
        error = info["error"]
    else:
        ok, error = pipeline.process_video(UpscaleJobConfig(
            source, str(tmp_path / "out"), model_id=model_id, overwrite=True,
        ))

    assert not ok
    expected_engine = AVAILABLE_MODELS[model_id].engine.value
    assert f"{expected_engine}-ncnn-vulkan" in error
    assert str(tmp_path / "no-engines" / "tools" / expected_engine) in error
    assert "None" not in error
    assert not commands
    encoder.assert_not_called()
    start.assert_not_called()
    assert not list(tmp_path.rglob("preview_*"))
    assert not list(tmp_path.rglob("job_*"))
