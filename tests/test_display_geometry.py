"""Display geometry regressions using mocked metadata and generated test patterns."""

import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from src.core.models import calculate_target_dimensions, recommend_model_for_video
from src.core.probe import VideoMetadata, probe_video


@pytest.mark.parametrize(
    "width,height,preset,expected",
    [
        (1080, 1920, "8k", (4320, 7680, "4320p")),
        (1080, 1920, "4320p", (4320, 7680, "4320p")),
        (1080, 1920, "4k", (2160, 3840, "2160p")),
        (1920, 1080, "8k", (7680, 4320, "4320p")),
        (720, 720, "8k", (4320, 4320, "4320p")),
        (640, 480, "4k", (2880, 2160, "2160p")),
        (640, 480, "8k", (5760, 4320, "4320p")),
        (480, 640, "8k", (4320, 5760, "4320p")),
        (1920, 800, "4k", (3840, 1600, "2160p")),
        (800, 1920, "4k", (1600, 3840, "2160p")),
        (720, 480, "2x", (1440, 960, "960p")),
    ],
)
def test_target_dimensions_fit_display_orientation(width, height, preset, expected):
    assert calculate_target_dimensions(width, height, preset) == expected


@pytest.mark.parametrize("rotation", [-270, -90, 90, 270, 450])
def test_square_coded_anamorphic_portrait_uses_display_aspect(rotation):
    # Coded pixels are square-shaped; SAR and rotation make the display 9:16.
    assert calculate_target_dimensions(
        720, 720, "8k", sample_aspect_ratio="16:9", rotation=rotation,
    ) == (4320, 7680, "4320p")


def test_anamorphic_landscape_and_rotated_native_geometry():
    assert calculate_target_dimensions(
        720, 576, "4k", sample_aspect_ratio="64:45",
    ) == (3840, 2160, "2160p")
    assert calculate_target_dimensions(
        1920, 1080, "4k", rotation=90,
    ) == (2160, 3840, "2160p")
    assert calculate_target_dimensions(
        720, 576, "2x", sample_aspect_ratio="16:15", rotation=-90,
    ) == (1152, 1536, "1536p")


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (1280, 720)])
def test_8k_auto_recommendation_uses_available_native_four_times_scale(width, height):
    recommendation = recommend_model_for_video(width, height, "8k")
    assert recommendation.model.id == "realesr-animevideov3"
    assert recommendation.scale == 4
    if (width, height) == (1280, 720):
        assert "final resize" in recommendation.reason


@pytest.mark.parametrize("width,height,scale", [(1280, 720, 3), (1920, 1080, 2), (1080, 1920, 2)])
def test_4k_auto_recommendation_keeps_existing_native_scales(width, height, scale):
    assert recommend_model_for_video(width, height, "4k").scale == scale


@pytest.mark.parametrize("sar", [None, "N/A", "0:1", "1:0", "-1:1", "nan", "inf"])
@pytest.mark.parametrize("rotation", [None, "N/A", "90 degrees", float("nan"), float("inf")])
def test_malformed_display_metadata_falls_back_to_square_pixels(sar, rotation):
    assert calculate_target_dimensions(
        1920, 1080, "4k", sample_aspect_ratio=sar, rotation=rotation,
    ) == (3840, 2160, "2160p")


def _video_stream(**updates):
    stream = {
        "codec_type": "video",
        "width": 720,
        "height": 720,
        "avg_frame_rate": "25/1",
        "duration": "1",
        "nb_frames": "25",
        "codec_name": "h264",
    }
    stream.update(updates)
    return stream


def _probe_json(streams):
    with patch("src.core.probe.subprocess.check_output", return_value=json.dumps({"streams": streams})):
        return probe_video("nonexistent-synthetic-geometry.mp4")


def test_probe_ignores_cover_art_and_retains_display_metadata():
    meta = _probe_json([
        _video_stream(width=512, height=512, disposition={"attached_pic": 1}, codec_name="mjpeg"),
        _video_stream(
            sample_aspect_ratio="16:9",
            tags={"rotate": "180"},
            side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90}],
        ),
    ])
    assert (meta.width, meta.height) == (720, 720)
    assert meta.codec_name == "h264"
    assert meta.sample_aspect_ratio == "16:9"
    assert meta.rotation == -90
    assert meta.display_width == 720
    assert meta.display_height == 1280


@pytest.mark.parametrize("disposition", [{"attached_pic": 1}, {"attached_pic": "1"}])
def test_probe_rejects_cover_art_without_a_video(disposition):
    with pytest.raises(ValueError, match="No video stream"):
        _probe_json([_video_stream(disposition=disposition)])


def test_probe_rotation_tag_fallback_and_invalid_sar():
    meta = _probe_json([_video_stream(
        sample_aspect_ratio="0:1", tags={"rotate": "90"},
        side_data_list=[{"rotation": "N/A"}],
    )])
    assert meta.sample_aspect_ratio == "1:1"
    assert meta.rotation == 90


def test_metadata_defaults_preserve_existing_constructor():
    meta = VideoMetadata("synthetic.mp4", 1920, 1080, 25, 1, 25, "h264", False, "", False)
    assert meta.sample_aspect_ratio == "1:1"
    assert meta.rotation == 0
    assert (meta.display_width, meta.display_height) == (1920, 1080)


def test_generated_square_coded_anamorphic_portrait(tmp_path):
    """Probe a real synthetic MP4 instead of relying only on mocked JSON."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and FFprobe are required for the generated-media regression")
    coded = tmp_path / "coded-square.mp4"
    rotated = tmp_path / "anamorphic-portrait.mp4"
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi",
        "-i", "testsrc2=size=128x128:rate=2:duration=1",
        "-vf", "setsar=16/9", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(coded),
    ], check=True, capture_output=True, timeout=30)
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", str(coded),
        "-c", "copy", str(rotated),
    ], check=True, capture_output=True, timeout=30)
    meta = probe_video(str(rotated), ffprobe)
    assert (meta.width, meta.height) == (128, 128)
    assert meta.sample_aspect_ratio == "16:9"
    assert abs(meta.rotation) == 90
    assert meta.display_width / meta.display_height == pytest.approx(9 / 16)
    assert calculate_target_dimensions(
        meta.width, meta.height, "8k",
        sample_aspect_ratio=meta.sample_aspect_ratio, rotation=meta.rotation,
    ) == (4320, 7680, "4320p")
