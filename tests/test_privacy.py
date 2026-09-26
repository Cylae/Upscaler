import json
import os
import shutil
import subprocess

import pytest

from src.core.metadata_provider import sanitize_metadata_text
from src.core.scene_renamer import (
    SceneNamingConfig,
    find_forbidden_output_markers,
    format_scene_name,
)
from src.core.video_pipeline import clean_metadata_args, find_forbidden_media_metadata


@pytest.mark.parametrize("value", ["AI", "A.I.", "IA", "I.A.", "artificial intelligence", "intelligence artificielle", "Real-ESRGAN", "RealCUGAN", "upscaled"])
def test_forbidden_markers_are_detected(value):
    assert find_forbidden_output_markers(value)


@pytest.mark.parametrize("value", ["Happy Trails", "Maia", "Gaia", "CYLAE", "Train Ride"])
def test_innocent_substrings_are_not_detected(value):
    assert not find_forbidden_output_markers(value)


def test_all_configurable_filename_fields_are_scrubbed():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, "AI Renders", "s01e01 - Real-ESRGAN Upscaled.mp4")
    config = SceneNamingConfig(
        show_name="AI Happy Tree Friends",
        source_tag="AI-WEB-DL",
        audio_tag="RealCUGAN-AAC",
        group_tag="AI-UPSCALER",
    )
    output = format_scene_name(path, base, config)
    assert not find_forbidden_output_markers(output)


def test_remote_metadata_text_is_plain_and_scrubbed():
    value = sanitize_metadata_text("<p>AI enhanced by <b>Real-ESRGAN</b> — synopsis</p>")
    assert "<" not in value
    assert not find_forbidden_output_markers(value)


def test_ffmpeg_metadata_cleanup_is_enforced(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg/FFprobe indisponible")

    output = tmp_path / "clean.mp4"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=0.2",
        "-c:v", "libx264",
        "-metadata", "title=AI UPSCALED",
        "-metadata:s:v:0", "comment=Real-ESRGAN",
    ]
    command.extend(clean_metadata_args(include_video=True))
    command.append(str(output))
    subprocess.run(command, check=True)

    assert not find_forbidden_media_metadata(str(output), ffprobe)
    raw = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(output)],
        text=True,
    )
    tags = json.loads(raw)
    assert not find_forbidden_output_markers(json.dumps(tags))
