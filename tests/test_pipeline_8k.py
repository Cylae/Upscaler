"""Encoder and display-geometry contracts, using synthetic inputs only."""

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from src.core.hardware import HardwareInfo
from src.core.models import AVAILABLE_MODELS
from src.core.probe import VideoMetadata, probe_video
from src.core.video_pipeline import (
    UpscaleJobConfig, VideoUpscalerPipeline, _select_model_scale,
)


def metadata(**updates):
    values = dict(filepath="synthetic.mp4", width=1080, height=1920,
                  fps=24, duration=1, nb_frames=24, codec_name="h264",
                  has_audio=False, audio_codec="", has_subtitles=False,
                  r_frame_rate="24/1")
    values.update(updates)
    return VideoMetadata(**values)


def test_preflight_uses_actual_geometry_and_encoding_options(monkeypatch):
    pipeline = VideoUpscalerPipeline(HardwareInfo(ffmpeg_path="test-ffmpeg"))
    proc = MagicMock(returncode=0)
    proc.communicate.return_value = (None, "")
    start = MagicMock(return_value=proc)
    monkeypatch.setattr(pipeline, "_start_process", start)

    assert pipeline._check_nvenc_target(metadata(), 4320, 7680, "hevc_nvenc", "18", "p7") == (True, "")
    command = start.call_args.args[0]
    assert command[0] == "test-ffmpeg"
    assert "color=c=black:s=4320x7680:r=24/1" in command
    assert command[command.index("-c:v") + 1] == "hevc_nvenc"
    assert command[command.index("-cq") + 1] == "18"
    assert command[command.index("-preset") + 1] == "p7"
    assert command[command.index("-frames:v") + 1] == "1"
    assert "-spatial-aq" in command and "-temporal-aq" in command
    assert command[-3:] == ["-f", "null", "-"]
    assert "frame_" not in " ".join(command)
    proc.communicate.assert_called_once_with(timeout=30)


@pytest.mark.parametrize("encoder,fallback", [("hevc_nvenc", "libx265"), ("h264_nvenc", "libx264")])
def test_unsupported_nvenc_falls_back_before_processing(monkeypatch, encoder, fallback):
    pipeline = VideoUpscalerPipeline(HardwareInfo(has_nvenc_hevc=True, has_nvenc_h264=True))
    check = MagicMock(return_value=(False, "Unsupported dimensions"))
    monkeypatch.setattr(pipeline, "_check_nvenc_target", check)
    logs = []
    job = UpscaleJobConfig("synthetic.mp4", "output", video_codec=encoder)
    assert pipeline._resolve_encoder(job, metadata(), 4320, 7680, logs.append) == fallback
    assert "Unsupported dimensions" in logs[0]
    check.assert_called_once()


def test_supported_nvenc_stays_on_gpu_and_cpu_selection_skips_probe(monkeypatch):
    pipeline = VideoUpscalerPipeline(HardwareInfo(has_nvenc_hevc=True))
    check = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(pipeline, "_check_nvenc_target", check)
    job = UpscaleJobConfig("synthetic.mp4", "output")
    assert pipeline._resolve_encoder(job, metadata(), 4320, 7680, lambda _: None) == "hevc_nvenc"
    job.video_codec = "libx265"
    assert pipeline._resolve_encoder(job, metadata(), 4320, 7680, lambda _: None) == "libx265"
    assert check.call_count == 1


def test_preflight_timeout_kills_and_unregisters_process(monkeypatch):
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    proc = MagicMock()
    proc.communicate.side_effect = [subprocess.TimeoutExpired("ffmpeg", 30), (None, "")]
    monkeypatch.setattr(pipeline, "_start_process", lambda *a, **k: proc)
    pipeline._active_processes.add(proc)
    kill = MagicMock()
    monkeypatch.setattr("src.core.video_pipeline.safe_kill_process_tree", kill)

    supported, reason = pipeline._check_nvenc_target(metadata(), 4320, 7680, "hevc_nvenc", "18", "p7")
    assert not supported and "timed out" in reason
    kill.assert_called_once_with(proc)
    assert not pipeline._active_processes


def test_cancellation_during_preflight_never_falls_back(monkeypatch):
    pipeline = VideoUpscalerPipeline(HardwareInfo(has_nvenc_hevc=True))
    def cancelled(*args):
        pipeline._is_cancelled = True
        return False, "cancelled"
    monkeypatch.setattr(pipeline, "_check_nvenc_target", cancelled)
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline._resolve_encoder(UpscaleJobConfig("synthetic.mp4", "output"), metadata(), 4320, 7680, lambda _: None)


@pytest.mark.parametrize("required,expected", [(2, 2), (3, 3), (4, 4), (8, 4)])
def test_preview_and_render_share_native_scale_selection(required, expected):
    assert _select_model_scale(AVAILABLE_MODELS["realesr-animevideov3"], required) == expected


def test_encode_outputs_square_pixels():
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    cmd = pipeline._build_encode_cmd("frames", "png", metadata(), 4320, 7680,
                                     "libx265", "18", "p7", None, "copy", "synthetic.mp4")
    assert cmd[cmd.index("-vf") + 1] == "scale=4320:7680:flags=lanczos,setsar=1"


def test_rotated_anamorphic_extraction_matches_reported_geometry(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic extraction")
    source = tmp_path / "source.mp4"
    rotated = tmp_path / "rotated.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=72x72:rate=2:duration=1", "-vf", "setsar=16/9",
                    "-c:v", "libx264", str(source)], check=True, capture_output=True, timeout=30)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-display_rotation:v:0", "90", "-i",
                    str(source), "-c", "copy", str(rotated)], check=True, capture_output=True, timeout=30)
    meta = probe_video(str(rotated))
    frames = tmp_path / "frames"
    frames.mkdir()
    pipeline = VideoUpscalerPipeline(HardwareInfo(ffmpeg_path=ffmpeg))
    cmd = pipeline._build_extract_frames_cmd(str(rotated), str(frames), "png", False, meta=meta)
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    frame_meta = probe_video(str(frames / "frame_00000001.png"))
    assert (frame_meta.width, frame_meta.height) == (72, 128)
    assert frame_meta.sample_aspect_ratio == "1:1"
