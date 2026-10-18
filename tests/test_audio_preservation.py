import os
from unittest.mock import MagicMock, patch
from src.core.video_pipeline import (
    VideoMetadata,
    UpscaleJobConfig,
    is_file_complete_and_valid,
    clean_metadata_args
)


def test_audio_preservation_flags_in_config():
    cfg_copy = UpscaleJobConfig(input_file="test.mp4", output_dir="out", audio_mode="copy")
    assert cfg_copy.audio_mode == "copy"

    cfg_aac = UpscaleJobConfig(input_file="test.mp4", output_dir="out", audio_mode="aac_192k")
    assert cfg_aac.audio_mode == "aac_192k"

    cfg_audiophile = UpscaleJobConfig(input_file="test.mp4", output_dir="out", audio_mode="aac_320k")
    assert cfg_audiophile.audio_mode == "aac_320k"

    cfg_flac = UpscaleJobConfig(input_file="test.mp4", output_dir="out", audio_mode="flac")
    assert cfg_flac.audio_mode == "flac"


def test_is_file_complete_and_valid_with_audio_expectation():
    # Test with mocked probe_video
    fake_meta_with_audio = VideoMetadata(
        filepath="dummy.mp4",
        width=3840,
        height=2160,
        fps=24.0,
        duration=120.0,
        nb_frames=2880,
        codec_name="hevc",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False
    )
    fake_meta_silent = VideoMetadata(
        filepath="dummy_silent.mp4",
        width=3840,
        height=2160,
        fps=24.0,
        duration=120.0,
        nb_frames=2880,
        codec_name="hevc",
        has_audio=False,
        audio_codec="",
        has_subtitles=False
    )

    with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=500_000):
        with patch("src.core.video_pipeline.probe_video", return_value=fake_meta_with_audio):
            # If expect_audio is True and file has audio -> True
            assert is_file_complete_and_valid("dummy.mp4", expect_audio=True) is True
            assert is_file_complete_and_valid("dummy.mp4", expect_audio=False) is True

        with patch("src.core.video_pipeline.probe_video", return_value=fake_meta_silent):
            # If expect_audio is True and file lacks audio -> MUST BE FALSE! (Guarantees re-rendering with sound)
            assert is_file_complete_and_valid("dummy_silent.mp4", expect_audio=True) is False
            # If expect_audio is False, silent is acceptable
            assert is_file_complete_and_valid("dummy_silent.mp4", expect_audio=False) is True


def test_probe_audio_stream():
    from src.core.video_pipeline import probe_audio_stream

    # Non-existent or too small file
    assert probe_audio_stream("non_existent_file.m4a") is False

    # Mock ffprobe returning streams with audio
    with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=10_000):
        with patch("subprocess.check_output", return_value='{"streams": [{"codec_type": "audio", "codec_name": "aac"}]}'):
            assert probe_audio_stream("valid_audio.m4a") is True

        with patch("subprocess.check_output", return_value='{"streams": []}'):
            assert probe_audio_stream("no_audio_stream.m4a") is False


def test_audio_integrity_check_failure_aborts_output():
    """Verify that if an output file is missing audio and repair fails, the pipeline aborts."""
    from src.core.video_pipeline import VideoUpscalerPipeline, VideoMetadata, UpscaleJobConfig
    from src.core.detector import HardwareInfo

    fake_hw = HardwareInfo(
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=8.0,
        has_nvidia=True,
        has_nvenc_hevc=True,
        has_nvenc_h264=True,
        realesrgan_bin="tools/realesrgan/realesrgan-ncnn-vulkan.exe",
        realcugan_bin="tools/realcugan/realcugan-ncnn-vulkan.exe"
    )
    pipeline = VideoUpscalerPipeline(fake_hw)

    meta_in = VideoMetadata(
        filepath="input.mp4",
        width=1280, height=720, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="h264", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    # Output file metadata lacks audio
    meta_out_silent = VideoMetadata(
        filepath="out_part.mp4",
        width=3840, height=2160, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="hevc", has_audio=False, audio_codec="", has_subtitles=False
    )

    job = UpscaleJobConfig(input_file="input.mp4", output_dir="dummy_out", overwrite=True)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = iter([])
    mock_proc.communicate.return_value = ("", "")
    mock_proc.wait = MagicMock()

    with patch("src.core.video_pipeline.probe_video", side_effect=[meta_in, meta_out_silent, meta_out_silent, meta_out_silent]):
        with patch("os.makedirs"), patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=True), patch("os.path.getsize", return_value=1_000_000):
            with patch("subprocess.Popen", return_value=mock_proc):
                with patch("src.core.video_pipeline.probe_audio_stream", return_value=True):
                    # Patch os.listdir at the module level so worker threads see it too
                    with patch("src.core.video_pipeline.os.listdir", return_value=["frame_00000001.jpg"]):
                        success, msg = pipeline.process_video(job)
                        # Must fail because soundtrack could not be preserved
                        assert success is False
                        assert "Audio integrity check failed" in msg


def test_audio_integrity_repair_pass_restores_audio():
    """Verify that when initial encode misses audio, repair remux pass recovers it and succeeds."""
    from src.core.video_pipeline import VideoUpscalerPipeline, VideoMetadata, UpscaleJobConfig
    from src.core.detector import HardwareInfo

    fake_hw = HardwareInfo(
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=8.0,
        has_nvidia=True,
        has_nvenc_hevc=True,
        has_nvenc_h264=True,
        realesrgan_bin="tools/realesrgan/realesrgan-ncnn-vulkan.exe",
        realcugan_bin="tools/realcugan/realcugan-ncnn-vulkan.exe"
    )
    pipeline = VideoUpscalerPipeline(fake_hw)

    meta_in = VideoMetadata(
        filepath="input.mp4",
        width=1280, height=720, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="h264", has_audio=True, audio_codec="aac", has_subtitles=False
    )
    meta_out_silent = VideoMetadata(
        filepath="out_part.mp4",
        width=3840, height=2160, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="hevc", has_audio=False, audio_codec="", has_subtitles=False
    )
    meta_out_fixed = VideoMetadata(
        filepath="out_part.mp4.fixed.mp4",
        width=3840, height=2160, fps=24.0, duration=10.0, nb_frames=240,
        codec_name="hevc", has_audio=True, audio_codec="aac", has_subtitles=False
    )

    job = UpscaleJobConfig(input_file="input.mp4", output_dir="dummy_out", overwrite=True)

    # 1. probe_video input -> meta_in
    # 2. probe_video part_output_file -> meta_out_silent (triggers repair!)
    # 3. probe_video repair_file -> meta_out_fixed (repair worked!)
    # 4. probe_video final part_output_file check -> meta_out_fixed (verified!)
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = iter([])
    mock_proc.communicate.return_value = ("", "")
    mock_proc.wait = MagicMock()

    with patch("src.core.video_pipeline.probe_video", side_effect=[meta_in, meta_out_silent, meta_out_fixed, meta_out_fixed]):
        with patch("os.makedirs"), patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=True), patch("os.path.getsize", return_value=1_000_000):
            with patch("subprocess.Popen", return_value=mock_proc):
                with patch("src.core.video_pipeline.probe_audio_stream", return_value=True):
                    with patch("src.core.video_pipeline.os.listdir", return_value=["frame_00000001.jpg"]):
                        with patch("os.replace"), patch("shutil.move"):
                            success, final_path = pipeline.process_video(job)
                            assert success is True
                            assert "dummy_out" in final_path
