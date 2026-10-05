import io
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from src.cli import run_cli
from src.core.video_pipeline import VideoMetadata


def test_cli_list_presets(capsys):
    ret = run_cli(["--list-presets"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "AVAILABLE QUALITY PROFILES" in captured.out
    assert "cartoon_4k_master" in captured.out


def test_cli_doctor(capsys):
    with patch("src.cli.print_doctor_report") as mock_doctor:
        ret = run_cli(["--doctor"])
        assert ret == 0
        mock_doctor.assert_called_once()


def test_cli_export_diagnostics_stdout(capsys):
    ret = run_cli(["--export-diagnostics"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SYSTEM ENVIRONMENT DIAGNOSTIC" in captured.out


def test_cli_export_diagnostics_file(tmp_path, capsys):
    out_file = tmp_path / "diag.txt"
    ret = run_cli(["--export-diagnostics", str(out_file)])
    assert ret == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "SYSTEM ENVIRONMENT DIAGNOSTIC" in content


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--cq", "10"],
        ["--cq", "35"],
        ["--workers", "0"],
        ["--workers", "10"],
        ["--tile-size", "-10"],
        ["--tile-size", "16"],
        ["--limit", "-5"],
        ["--threads", "invalid_format"],
        ["--threads", "0:4:2"],
        [],  # missing source
    ],
)
def test_cli_validation_errors(invalid_args):
    with pytest.raises(SystemExit) as exc_info:
        run_cli(invalid_args)
    assert exc_info.value.code == 2


def test_cli_source_not_found(capsys):
    non_existent = "non_existent_folder_xyz_12345"
    ret = run_cli(["--source", non_existent])
    assert ret == 1
    captured = capsys.readouterr()
    assert f"Error: Source path '{non_existent}' does not exist." in captured.out


def test_cli_empty_folder(tmp_path, capsys):
    empty_dir = tmp_path / "empty_videos"
    empty_dir.mkdir()
    ret = run_cli(["--source", str(empty_dir)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "No video files found." in captured.out


def test_cli_dry_run_with_files(tmp_path, capsys):
    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Episode_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

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

    with patch("src.cli.probe_video", return_value=dummy_meta):
        ret = run_cli([
            "--source", str(src_dir),
            "--dry-run",
            "--model", "realcugan-pro",
            "--yes",
        ])
        assert ret == 0
        captured = capsys.readouterr()
        assert "DRY-RUN OUTPUT PLAN" in captured.out
        assert "Dry run complete: 1 item(s) mapped" in captured.out


def test_cli_non_interactive_without_yes(tmp_path, capsys):
    src_dir = tmp_path / "episodes"
    src_dir.mkdir()
    video_file = src_dir / "Episode_01.mp4"
    video_file.write_bytes(b"\x00" * 1024)

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

    with patch("src.cli.probe_video", return_value=dummy_meta):
        with patch("sys.stdin.isatty", return_value=False):
            ret = run_cli([
                "--source", str(src_dir),
                "--model", "realcugan-pro",
            ])
            assert ret == 2
            captured = capsys.readouterr()
            assert "Error: model validation requires an interactive terminal" in captured.out
