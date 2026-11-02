from unittest.mock import MagicMock, patch

import pytest

from src.core.pipeline import UpscaleJobConfig
from src.core.probe import VideoMetadata
from src.core.validator import (
    is_file_complete_and_valid,
    validate_encoded_video,
    validate_job_config,
)


def test_is_file_complete_and_valid(tmp_path):
    # Non-existent
    assert is_file_complete_and_valid("non_existent_file.mp4") is False

    # Tiny file (< 100KB)
    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"x" * 50_000)
    assert is_file_complete_and_valid(str(tiny)) is False

    large = tmp_path / "complete.mp4"
    large.write_bytes(b"x" * 150_000)

    # Corrupt probe
    with patch("src.core.validator._resolve_probe_video", return_value=MagicMock(side_effect=RuntimeError("Corrupt moov atom"))):
        assert is_file_complete_and_valid(str(large)) is False

    # Zero duration or invalid dimensions
    bad_meta = VideoMetadata(
        filepath=str(large), width=0, height=1080, fps=24, duration=0.2, nb_frames=0,
        codec_name="h264", has_audio=False, audio_codec="", has_subtitles=False
    )
    with patch("src.core.validator._resolve_probe_video", return_value=lambda path: bad_meta):
        assert is_file_complete_and_valid(str(large)) is False

    # Truncated moov atom (nb_frames <= 0)
    zero_frames = VideoMetadata(
        filepath=str(large), width=1920, height=1080, fps=24, duration=10.0, nb_frames=0,
        codec_name="h264", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    with patch("src.core.validator._resolve_probe_video", return_value=lambda path: zero_frames):
        assert is_file_complete_and_valid(str(large)) is False

    # Expected dimensions mismatch
    valid_meta = VideoMetadata(
        filepath=str(large), width=1920, height=1080, fps=24, duration=100.0, nb_frames=2400,
        codec_name="hevc", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    with patch("src.core.validator._resolve_probe_video", return_value=lambda path: valid_meta):
        assert is_file_complete_and_valid(str(large), expected_width=3840, expected_height=2160) is False
        # Expected duration mismatch
        assert is_file_complete_and_valid(str(large), expected_duration=50.0) is False
        # Expected audio missing
        assert is_file_complete_and_valid(str(large), expect_audio=True) is True

        silent_meta = VideoMetadata(
            filepath=str(large), width=1920, height=1080, fps=24, duration=100.0, nb_frames=2400,
            codec_name="hevc", has_audio=False, audio_codec="", has_subtitles=False
        )
        with patch("src.core.validator._resolve_probe_video", return_value=lambda path: silent_meta):
            assert is_file_complete_and_valid(str(large), expect_audio=True) is False

        # Perfect match
        assert is_file_complete_and_valid(
            str(large),
            expect_audio=True,
            expected_width=1920,
            expected_height=1080,
            expected_duration=100.5
        ) is True


def test_validate_encoded_video(tmp_path):
    # Non-existent
    ok, err = validate_encoded_video("missing.mp4", 1920, 1080)
    assert ok is False
    assert "did not create an output file" in err

    # Empty file
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    ok, err = validate_encoded_video(str(empty), 1920, 1080)
    assert ok is False

    valid_file = tmp_path / "valid.mp4"
    valid_file.write_bytes(b"x" * 200)

    # Resolution mismatch
    mismatch_meta = VideoMetadata(
        filepath=str(valid_file), width=1280, height=720, fps=24, duration=10, nb_frames=240,
        codec_name="h264", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    ok, err = validate_encoded_video(str(valid_file), 1920, 1080, metadata=mismatch_meta)
    assert ok is False
    assert "Final output resolution is 1280x720, expected 1920x1080" in err

    # Missing audio
    silent_meta = VideoMetadata(
        filepath=str(valid_file), width=1920, height=1080, fps=24, duration=10, nb_frames=240,
        codec_name="h264", has_audio=False, audio_codec="", has_subtitles=False
    )
    ok, err = validate_encoded_video(str(valid_file), 1920, 1080, expect_audio=True, metadata=silent_meta)
    assert ok is False
    assert "missing its expected audio stream" in err

    # Valid
    ok, err = validate_encoded_video(str(valid_file), 1920, 1080, expect_audio=False, metadata=silent_meta)
    assert ok is True
    assert err == ""


def test_validate_job_config(tmp_path):
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"dummy")

    valid_job = UpscaleJobConfig(
        input_file=str(input_file),
        output_dir=str(tmp_path / "out"),
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="auto",
        cq_level=19,
        nvenc_preset="p7",
        denoise_level=-1,
        audio_mode="copy",
        tile_size=0,
        threads="auto",
        frame_format="jpg",
    )
    # Valid job passes without error
    validate_job_config(valid_job)

    # Missing input
    with pytest.raises(ValueError, match="Input video does not exist"):
        validate_job_config(UpscaleJobConfig(input_file="non_existent.mp4", output_dir="out"))

    # Missing output dir
    with pytest.raises(ValueError, match="Output directory is required"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir=""))

    # Unsupported preset
    with pytest.raises(ValueError, match="Unsupported target resolution"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", target_preset="16k"))

    # Unsupported codec
    with pytest.raises(ValueError, match="Unsupported video codec"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", video_codec="vp9"))

    # Unknown model
    with pytest.raises(ValueError, match="Unknown AI model"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", model_id="fantasy-model"))

    # Unsupported frame format
    with pytest.raises(ValueError, match="Unsupported frame format"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", frame_format="bmp"))

    # CQ level out of range
    with pytest.raises(ValueError, match="CQ level must be between 14 and 28"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", cq_level=10))

    # NVENC preset invalid
    with pytest.raises(ValueError, match="Unsupported NVENC preset"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", nvenc_preset="p8"))

    # RealCUGAN denoise invalid
    with pytest.raises(ValueError, match="RealCUGAN denoise level"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", denoise_level=5))

    # Audio mode invalid
    with pytest.raises(ValueError, match="Unsupported audio mode"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", audio_mode="opus"))

    # Tile size < 32 and != 0
    with pytest.raises(ValueError, match="Tile size must be 0"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", tile_size=16))

    # Threads invalid format
    with pytest.raises(ValueError, match="Threads must be 'auto' or LOAD:PROC:SAVE"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", threads="2:0:2"))

    # Preview duration <= 0
    with pytest.raises(ValueError, match="Preview duration must be greater than zero"):
        validate_job_config(UpscaleJobConfig(input_file=str(input_file), output_dir="out", preview_duration_s=0.0))
