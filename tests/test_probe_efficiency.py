"""Probe reuse must preserve every stream/tag and reject stale file snapshots."""

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from src.core import probe


@pytest.fixture(autouse=True)
def empty_probe_cache():
    probe.clear_probe_cache()
    yield
    probe.clear_probe_cache()


def media_document(*, width=1920, title="Clean title"):
    return json.dumps({
        "streams": [
            {
                "codec_type": "video", "codec_name": "h264", "width": width,
                "height": 1080, "avg_frame_rate": "24000/1001", "nb_frames": "240",
                "color_space": "bt709", "sample_aspect_ratio": "1:1",
                "tags": {"title": "Video"},
            },
            {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "fra"}},
            {"codec_type": "audio", "codec_name": "flac", "tags": {"language": "eng"}},
            {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"title": title}},
        ],
        "format": {"duration": "10.01", "tags": {"title": "Container title"}},
    })


def test_video_audio_and_tag_audits_share_one_complete_inspection(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    inspect = Mock(return_value=media_document(title="Real-ESRGAN"))
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)

    video = probe.probe_video(str(path))
    audio = probe.probe_audio_streams(str(path))
    clean, markers, error = probe.verify_clean_media_metadata(str(path))

    assert video.width == 1920 and video.audio_stream_count == 2
    assert video.has_subtitles and video.color_space == "bt709"
    assert [track["codec_name"] for track in audio] == ["aac", "flac"]
    assert not clean and markers and not error
    assert inspect.call_count == 1


def test_audio_first_inspection_remains_complete_and_audio_only_files_work(tmp_path, monkeypatch):
    path = tmp_path / "sound.mka"
    path.write_bytes(b"media")
    inspect = Mock(return_value='{"streams": [{"codec_type": "audio", "codec_name": "flac"}]}')
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)

    assert probe.probe_audio_streams(str(path))[0]["codec_name"] == "flac"
    assert probe.verify_clean_media_metadata(str(path)) == (True, [], "")
    with pytest.raises(ValueError, match="No video stream"):
        probe.probe_video(str(path))
    assert inspect.call_count == 1
    assert "-show_format" in inspect.call_args.args[0]


def test_returned_video_and_nested_audio_metadata_cannot_poison_cache(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    inspect = Mock(return_value=media_document())
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)

    video = probe.probe_video(str(path))
    video.width = 1
    tracks = probe.probe_audio_streams(str(path))
    tracks[0]["tags"]["language"] = "AI"
    tracks.pop()

    assert probe.probe_video(str(path)).width == 1920
    tracks = probe.probe_audio_streams(str(path))
    assert len(tracks) == 2 and tracks[0]["tags"]["language"] == "fra"
    assert probe.verify_clean_media_metadata(str(path)) == (True, [], "")
    assert inspect.call_count == 1


def test_same_size_replacement_with_preserved_mtime_is_reprobed(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"old")
    previous = path.stat()
    inspect = Mock(side_effect=[media_document(width=1920), media_document(width=3840)])
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)
    assert probe.probe_video(str(path)).width == 1920

    replacement = tmp_path / "replacement.mkv"
    replacement.write_bytes(b"new")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    os.replace(replacement, path)

    assert probe.probe_video(str(path)).width == 3840
    assert inspect.call_count == 2


def test_submicrosecond_mtime_change_is_reprobed(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"old")
    stamp = 1_700_000_000_000_000_000
    os.utime(path, ns=(stamp, stamp))
    inspect = Mock(side_effect=[media_document(width=1920), media_document(width=3840)])
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)
    assert probe.probe_video(str(path)).width == 1920

    os.utime(path, ns=(stamp, stamp + 100))
    assert probe.probe_video(str(path)).width == 3840
    assert inspect.call_count == 2


def test_probe_executables_have_separate_snapshots(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    first_bin = tmp_path / "ffprobe-one.exe"
    second_bin = tmp_path / "ffprobe-two.exe"
    first_bin.write_bytes(b"one")
    second_bin.write_bytes(b"two")
    inspect = Mock(side_effect=[media_document(width=1920), media_document(width=3840)])
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)

    assert probe.probe_video(str(path), str(first_bin)).width == 1920
    assert probe.probe_video(str(path), str(second_bin)).width == 3840
    assert inspect.call_count == 2


def test_concurrent_video_audio_and_tag_requests_launch_one_probe(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    started = threading.Event()
    release = threading.Event()

    def inspect(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        return media_document()

    inspect_mock = Mock(side_effect=inspect)
    monkeypatch.setattr(probe.subprocess, "check_output", inspect_mock)
    with ThreadPoolExecutor(max_workers=6) as pool:
        video = pool.submit(probe.probe_video, str(path))
        assert started.wait(5)
        audio = pool.submit(probe.probe_audio_streams, str(path))
        tags = pool.submit(probe.verify_clean_media_metadata, str(path))
        more_video = pool.submit(probe.probe_video, str(path))
        release.set()
        assert video.result(timeout=5).audio_stream_count == 2
        assert len(audio.result(timeout=5)) == 2
        assert tags.result(timeout=5) == (True, [], "")
        assert more_video.result(timeout=5).width == 1920
    assert inspect_mock.call_count == 1


def test_failed_probe_does_not_cache_audio_absence_or_successful_audit(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    inspect = Mock(side_effect=[subprocess.TimeoutExpired("ffprobe", 1), media_document(title="AI")])
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)

    assert probe.probe_audio_streams(str(path)) == []
    clean, markers, error = probe.verify_clean_media_metadata(str(path))
    assert not clean and markers and not error
    assert inspect.call_count == 2


def test_file_changing_during_probe_is_retried_before_validation(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"old")
    calls = []

    def inspect(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            path.write_bytes(b"new content")
            return media_document(width=1920)
        return media_document(width=3840)

    monkeypatch.setattr(probe.subprocess, "check_output", inspect)
    assert probe.probe_video(str(path)).width == 3840
    assert probe.probe_video(str(path)).width == 3840
    assert len(calls) == 2


def test_continuously_changing_file_fails_validation_instead_of_reusing_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")

    def inspect(*_args, **_kwargs):
        with path.open("ab") as media:
            media.write(b"x")
        return media_document()

    monkeypatch.setattr(probe.subprocess, "check_output", inspect)
    with pytest.raises(RuntimeError, match="changed while ffprobe"):
        probe.probe_video(str(path))
    assert not probe._PROBE_CACHE


def test_cache_evicts_old_snapshots_and_clear_always_forces_inspection(tmp_path, monkeypatch):
    paths = [tmp_path / f"movie-{index}.mkv" for index in range(3)]
    for path in paths:
        path.write_bytes(b"media")
    inspect = Mock(return_value=media_document())
    monkeypatch.setattr(probe.subprocess, "check_output", inspect)
    monkeypatch.setattr(probe, "_PROBE_CACHE_LIMIT", 2)

    probe.probe_video(str(paths[0]))
    probe.probe_video(str(paths[1]))
    probe.probe_video(str(paths[0]))  # Keep the first file recently used.
    probe.probe_video(str(paths[2]))
    probe.probe_video(str(paths[0]))
    assert inspect.call_count == 3
    probe.probe_video(str(paths[1]))  # The older file was evicted.
    assert inspect.call_count == 4
    probe.clear_probe_cache()
    probe.probe_video(str(paths[1]))
    assert inspect.call_count == 5


def test_cache_clear_during_active_probe_prevents_old_snapshot_republication(tmp_path, monkeypatch):
    path = tmp_path / "movie.mkv"
    path.write_bytes(b"media")
    started = threading.Event()
    release = threading.Event()

    def inspect(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        return media_document()

    inspect_mock = Mock(side_effect=inspect)
    monkeypatch.setattr(probe.subprocess, "check_output", inspect_mock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(probe.probe_video, str(path))
        assert started.wait(5)
        probe.clear_probe_cache()
        release.set()
        assert future.result(timeout=5).width == 1920
    probe.probe_video(str(path))
    assert inspect_mock.call_count == 2
