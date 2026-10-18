"""Performance changes must preserve decoded pixels and pipeline safeguards."""

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from src.core.hardware import HardwareInfo
from src.core.probe import VideoMetadata, probe_video
from src.core.video_pipeline import (
    UpscaleJobConfig,
    VideoUpscalerPipeline,
    _complete_frame_sequence,
    _list_nonempty_frames,
)


def test_frame_inventory_ignores_empty_files_directories_and_other_formats(tmp_path):
    for name in ("frame_00000002.png", "frame_00000001.png", "other.png", "frame_00000003.jpg"):
        (tmp_path / name).write_bytes(b"frame")
    (tmp_path / "frame_00000003.png").touch()
    (tmp_path / "frame_00000004.png").mkdir()

    assert _list_nonempty_frames(str(tmp_path), "png") == [
        "frame_00000001.png", "frame_00000002.png",
    ]


@pytest.mark.parametrize("names,expected,complete", [
    (["frame_00000001.png", "frame_00000002.png"], 2, True),
    (["frame_00000001.png", "frame_00000003.png"], 2, False),
    (["frame_00000000.png", "frame_00000001.png"], 2, False),
    (["frame_00000001.png"], 2, False),
    (["frame_00000001.png", "frame_backup.png"], 2, False),
])
def test_same_frame_count_cannot_hide_missing_or_misnumbered_images(names, expected, complete):
    assert _complete_frame_sequence(names, "png", expected) is complete


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
def test_faster_png_extraction_is_pixel_exact_at_8_and_10_bits(tmp_path, pixel_format):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is needed for decoded-pixel equivalence")
    source = tmp_path / "source.mkv"
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=72x40:rate=4:duration=1", "-pix_fmt", pixel_format,
        "-c:v", "ffv1", str(source),
    ], check=True, capture_output=True, timeout=30)
    meta = probe_video(str(source))
    pipeline = VideoUpscalerPipeline(HardwareInfo(ffmpeg_path=ffmpeg))
    decoded = []
    for mode in ("default", "fast_lossless"):
        frames = tmp_path / mode
        frames.mkdir()
        cmd = pipeline._build_extract_frames_cmd(str(source), str(frames), "png", False, meta=meta)
        if mode == "default":
            position = cmd.index("-compression_level")
            del cmd[position:position + 2]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        hashes = subprocess.check_output([
            ffmpeg, "-v", "error", "-i", str(frames / "frame_%08d.png"),
            "-f", "framemd5", "-",
        ], text=True, timeout=30)
        decoded.append(hashes)

    assert decoded[0] == decoded[1]


def test_fast_inference_does_not_flood_progress_callbacks(monkeypatch):
    import src.core.video_pipeline as pipeline_module

    names = [f"frame_{index:08d}.png" for index in range(1, 1001)]
    source = VideoMetadata("input.mp4", 1280, 720, 24, 1000 / 24, 1000, "h264", False, "", False)
    output = VideoMetadata("output.mp4", 1280, 720, 24, 1000 / 24, 1000, "h264", False, "", False)
    pipeline = VideoUpscalerPipeline(HardwareInfo(realesrgan_bin="engine.exe"))
    proc = MagicMock(returncode=0)
    proc.communicate.return_value = (None, "")
    proc.stdout = iter(f"in -> {name} done" for name in names)
    monkeypatch.setattr(pipeline_module.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(pipeline_module.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(pipeline_module.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(pipeline_module.os.path, "getsize", lambda path: 1_000_000)
    monkeypatch.setattr(pipeline_module.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module.os, "listdir", lambda path: names)
    monkeypatch.setattr(pipeline_module, "_list_nonempty_frames", lambda *a: names)
    monkeypatch.setattr(pipeline_module, "probe_video", MagicMock(side_effect=[source, output]))
    monkeypatch.setattr(pipeline_module, "register_active_job_dir", lambda path: None)
    monkeypatch.setattr(pipeline_module, "unregister_active_job_dir", lambda path: None)
    monkeypatch.setattr(pipeline_module, "safe_atomic_move", lambda *a: None)
    monkeypatch.setattr(pipeline_module.shutil, "rmtree", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: 100.0)
    reports = []

    success, _ = pipeline.process_video(
        UpscaleJobConfig("input.mp4", "output", target_preset="720p", video_codec="libx264",
                         model_id="realesr-animevideov3", frame_format="png", overwrite=True),
        lambda ratio, message, extra: reports.append((ratio, extra)),
    )

    assert success
    inference = [extra for _, extra in reports if extra.get("stage") == "upscaling"]
    assert len(inference) == 2
    assert inference[0]["completed_frames"] == 0
    assert inference[-1]["completed_frames"] == len(names)
    assert reports[-1][0] == 1.0
