from src.core.presets import ENHANCEMENT_PRESETS, get_preset, EnhancementPreset


def test_presets_exist():
    assert "cartoon_4k_master" in ENHANCEMENT_PRESETS
    assert "fast_draft" in ENHANCEMENT_PRESETS
    assert "archival" in ENHANCEMENT_PRESETS
    assert "lightweight_1080p" in ENHANCEMENT_PRESETS
    assert "clean_denoise" in ENHANCEMENT_PRESETS
    assert "custom" in ENHANCEMENT_PRESETS


def test_cartoon_4k_master_defaults():
    preset = get_preset("cartoon_4k_master")
    assert preset.target_preset == "4k"
    assert preset.video_codec == "hevc_nvenc"
    assert preset.cq_level == 19
    assert preset.nvenc_preset == "p7"
    assert preset.audio_mode == "copy"
    assert preset.workers == 2


def test_archival_defaults():
    preset = get_preset("archival")
    assert preset.enable_tta is True
    assert preset.cq_level <= 16
    assert preset.audio_mode == "copy"


def test_clean_denoise_defaults():
    preset = get_preset("clean_denoise")
    assert preset.denoise_level >= 0
    assert preset.model_id == "realcugan-pro"


def test_fallback_preset():
    preset = get_preset("non_existent_preset_id")
    assert preset.id == "cartoon_4k_master"
