import os
from src.core.video_pipeline import UpscaleJobConfig, is_file_complete_and_valid, clean_metadata_args
from src.core.scene_renamer import SceneNamingConfig


def test_upscale_job_config_expert_fields():
    cfg = UpscaleJobConfig(
        input_file="input.mp4",
        output_dir="output",
        cq_level=16,
        nvenc_preset="p7",
        enable_tta=True,
        denoise_level=2,
        audio_mode="aac_320k",
        tile_size=400
    )
    assert cfg.cq_level == 16
    assert cfg.nvenc_preset == "p7"
    assert cfg.enable_tta is True
    assert cfg.denoise_level == 2
    assert cfg.audio_mode == "aac_320k"
    assert cfg.tile_size == 400


def test_is_file_complete_non_existent():
    assert is_file_complete_and_valid("non_existent_file_path.mp4", expect_audio=True) is False
    assert is_file_complete_and_valid("non_existent_file_path.mp4", expect_audio=False) is False


def test_clean_metadata_args():
    args = clean_metadata_args(include_video=True)
    assert "-map_metadata" in args
    assert "-1" in args
    for item in args:
        assert "ai" not in item.lower()
        assert "waifu" not in item.lower()
