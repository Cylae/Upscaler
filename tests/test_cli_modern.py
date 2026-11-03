"""Unit and integration test suite for the next-generation unified AnimaScale CLI.

Validates argument parsing, preset mapping, backend selection, recursive directory discovery,
dry-run planning, telemetry formatting, signal handling, and main entrypoint routing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import main as entrypoint
from src.core.video_pipeline import VideoMetadata
from src.engine.cli import (
    build_parser,
    format_eta_str,
    get_vram_telemetry_string,
    run_cli,
)


# -----------------------------------------------------------------------------
# 1. Telemetry and Utility Formatting Tests
# -----------------------------------------------------------------------------

def test_format_eta_str_handles_various_intervals():
    assert format_eta_str(0) == "00:00"
    assert format_eta_str(-10) == "00:00"
    assert format_eta_str(5) == "00:05"
    assert format_eta_str(65) == "01:05"
    assert format_eta_str(3665) == "01:01:05"


def test_vram_telemetry_string_with_mocked_hardware():
    with patch("src.core.hardware.query_gpu_vram_status", return_value=(4.0, 8.0)):
        telemetry = get_vram_telemetry_string()
        assert "4.0/8.0 GB" in telemetry
        assert "50%" in telemetry

    with patch("src.core.hardware.query_gpu_vram_status", return_value=(0.0, 0.0)):
        telemetry = get_vram_telemetry_string()
        assert telemetry == "N/A"


# -----------------------------------------------------------------------------
# 2. Argument Parser and Validation Tests
# -----------------------------------------------------------------------------

def test_build_parser_options():
    parser = build_parser()
    args = parser.parse_args(["-i", "input.mp4", "-o", "out.mp4", "-b", "directml", "--scale", "4", "-c", "hevc_nvenc", "-a", "aac"])
    assert args.input_flag == "input.mp4"
    assert args.output == "out.mp4"
    assert args.backend == "directml"
    assert args.scale == 4
    assert args.codec == "hevc_nvenc"
    assert args.audio_mode == "aac"


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--cq", "10"],
        ["--cq", "35"],
        ["--workers", "0"],
        ["--workers", "8"],
        ["--tile-size", "-10"],
        ["--tile-size", "16"],
        ["--limit", "-5"],
        ["--threads", "invalid_threads"],
        ["--threads", "0:2:1"],
        [],  # Missing source
    ],
)
def test_modern_cli_validation_errors(invalid_args):
    with pytest.raises(SystemExit) as exc_info:
        run_cli(invalid_args)
    assert exc_info.value.code == 2


# -----------------------------------------------------------------------------
# 3. Discovery, Listing, and Diagnostics Routes
# -----------------------------------------------------------------------------

def test_cli_list_presets_includes_modern_and_legacy(capsys):
    ret = run_cli(["--list-presets"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "AVAILABLE QUALITY PROFILES" in captured.out
    assert "anime_4k_master" in captured.out
    assert "photo_realistic_4k" in captured.out
    assert "balanced_fast" in captured.out
    assert "ultrawide" in captured.out
    assert "cartoon_4k_master" in captured.out


def test_cli_list_models(capsys):
    ret = run_cli(["--list-models"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "AVAILABLE NEURAL UPSCALING MODELS" in captured.out
    assert "realesrgan-x4plus-anime" in captured.out
    assert "realesr-animevideov3" in captured.out


def test_cli_diagnostics_route(capsys):
    with patch("src.engine.cli._get_print_doctor_report") as mock_doctor:
        ret = run_cli(["--diagnostics"])
        assert ret == 0
        mock_doctor.return_value.assert_called_once()


def test_cli_export_diagnostics_route(tmp_path, capsys):
    out_file = tmp_path / "diagnostics.txt"
    ret = run_cli(["--export-diagnostics", str(out_file)])
    assert ret == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "SYSTEM ENVIRONMENT DIAGNOSTIC" in content


# -----------------------------------------------------------------------------
# 4. Preset Resolution and Modern Flags
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("preset_arg", "expected_target", "expected_model"),
    [
        ("anime_4k_master", "4k", "auto"),
        ("Anime 4K Master", "4k", "auto"),
        ("photo_realistic_4k", "4k", "realesrgan-x4plus"),
        ("Photo Realistic 4K", "4k", "realesrgan-x4plus"),
        ("balanced_fast", "4k", "realesr-animevideov3"),
        ("Balanced Fast", "4k", "realesr-animevideov3"),
        ("ultrawide", "1440p", "auto"),
    ],
)
def test_modern_presets_reach_batch_configuration(tmp_path, preset_arg, expected_target, expected_model):
    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Episode_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

    manager = MagicMock()
    manager.queue = [MagicMock(index=1, status=MagicMock(value="Pending"), file_size_mb=1.0, relative_source_path="Episode_01.mp4", target_scene_name="out.mp4")]
    manager.scan_folder.return_value = manager.queue
    manager.get_preflight_summary.return_value = {
        "total_items": 1,
        "pending_items": 1,
        "skipped_items": 0,
        "total_source_size_mb": 1.0,
        "estimated_output_mb": 2.5,
        "free_disk_mb": 5000.0,
        "warnings": [],
    }

    dummy_meta = VideoMetadata(
        filepath=str(video_file),
        width=1280,
        height=720,
        fps=24.0,
        duration=10.0,
        nb_frames=240,
        codec_name="h264",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False,
    )

    with (
        patch("src.engine.cli._get_batch_manager_cls", return_value=lambda _hw: manager),
        patch("src.engine.cli._get_probe_video", return_value=lambda _f: dummy_meta),
    ):
        ret = run_cli(["-i", str(src_dir), "-p", preset_arg, "-y"])
        assert ret == 0
        kwargs = manager.run_batch.call_args.kwargs
        assert kwargs["target_preset"] == expected_target
        assert kwargs["model_id"] == expected_model


def test_cli_dry_run_with_custom_backend_and_scale(tmp_path, capsys):
    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Test_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

    dummy_meta = VideoMetadata(
        filepath=str(video_file),
        width=1920,
        height=1080,
        fps=30.0,
        duration=5.0,
        nb_frames=150,
        codec_name="h264",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False,
    )

    with patch("src.engine.cli._get_probe_video", return_value=lambda _f: dummy_meta):
        ret = run_cli([
            "-i", str(src_dir),
            "--backend", "directml",
            "--scale", "4",
            "--codec", "hevc_nvenc",
            "--audio", "aac",
            "--dry-run",
            "-y",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DRY-RUN OUTPUT PLAN" in captured.out
        assert "DIRECTML" in captured.out
        assert "Scale Factor: 4x" in captured.out
        assert "Audio Mode  : AAC" in captured.out


# -----------------------------------------------------------------------------
# 5. Signal Handling and Cancellation
# -----------------------------------------------------------------------------

def test_cli_sigint_handling(tmp_path, capsys):
    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Episode_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

    manager = MagicMock()
    manager.queue = [MagicMock(index=1, status=MagicMock(value="Pending"), file_size_mb=1.0, relative_source_path="Episode_01.mp4", target_scene_name="out.mp4")]
    manager.scan_folder.return_value = manager.queue
    manager.get_preflight_summary.return_value = {
        "total_items": 1,
        "pending_items": 1,
        "skipped_items": 0,
        "total_source_size_mb": 1.0,
        "estimated_output_mb": 2.5,
        "free_disk_mb": 5000.0,
        "warnings": [],
    }

    # Simulate KeyboardInterrupt during batch run
    manager.run_batch.side_effect = KeyboardInterrupt()

    with patch("src.engine.cli._get_batch_manager_cls", return_value=lambda _hw: manager):
        ret = run_cli(["-i", str(src_dir), "-y"])
        assert ret == 130
        manager.stop.assert_called_once()
        assert "Batch cancelled by user." in capsys.readouterr().out


# -----------------------------------------------------------------------------
# 6. Main Entrypoint Integration and Routing
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "modern_flag",
    [
        ["--input", "test.mp4"],
        ["-i", "test.mp4"],
        ["--backend", "directml"],
        ["-b", "vulkan"],
        ["--scale", "4"],
        ["--audio", "aac"],
        ["-a", "copy"],
        ["--diagnostics"],
        ["--recursive"],
        ["--list-models"],
    ],
)
def test_main_routes_modern_cli_flags_without_gui(monkeypatch, modern_flag):
    probe_gui = MagicMock(side_effect=AssertionError("GUI dependency check should not occur for CLI routes"))
    monkeypatch.setattr("importlib.util.find_spec", probe_gui)

    mock_cli = MagicMock(return_value=0)
    monkeypatch.setattr("src.cli.run_cli", mock_cli)
    monkeypatch.setattr(sys, "argv", ["main.py", *modern_flag])

    assert entrypoint.main() == 0
    mock_cli.assert_called_once_with()
    probe_gui.assert_not_called()
