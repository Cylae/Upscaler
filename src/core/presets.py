from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class EnhancementPreset:
    id: str
    name: str
    icon: str
    description: str
    recommended_for: str
    target_preset: str = "4k"  # "4k", "1440p", "1080p", "720p"
    video_codec: str = "hevc_nvenc"  # "hevc_nvenc", "h264_nvenc", "libx265"
    model_id: str = "auto"
    cq_level: int = 19  # Constant Quality: 14 (near lossless) to 28 (compact)
    nvenc_preset: str = "p7"  # "p1" to "p7"
    enable_tta: bool = False  # 8-pass test-time augmentation
    denoise_level: int = -1  # -1 (none), 0, 1, 2, 3
    audio_mode: str = "copy"  # "copy", "aac_192k", "aac_320k", "flac"
    workers: int = 2
    frame_format: str = "jpg"


ENHANCEMENT_PRESETS: Dict[str, EnhancementPreset] = {
    "cartoon_4k_master": EnhancementPreset(
        id="cartoon_4k_master",
        name="Cartoon 4K Master (Recommended)",
        icon="🎬",
        description="Pristine vector outlines, clean color fills, high NVENC compression, and zero artifacting.",
        recommended_for="Best balance of visual quality, pristine lines, and speed for 2D animation.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="auto",
        cq_level=19,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg"
    ),
    "fast_draft": EnhancementPreset(
        id="fast_draft",
        name="Maximum Speed Draft (40+ fps)",
        icon="🚀",
        description="High-speed turnaround with Real-ESRGAN AnimeVideo V3 and maximum-speed NVENC P3 encoding.",
        recommended_for="Fast previewing or rapid batch processing of large season collections.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="realesr-animevideov3",
        cq_level=22,
        nvenc_preset="p3",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg"
    ),
    "archival": EnhancementPreset(
        id="archival",
        name="Archival Preservation Master",
        icon="🏆",
        description="Maximum mathematical fidelity using Anime 6B model, 8-pass TTA ensemble, and CQ 16.",
        recommended_for="Master copies, reference archiving, and maximum line reconstruction precision.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="realesrgan-x4plus-anime",
        cq_level=16,
        nvenc_preset="p7",
        enable_tta=True,
        denoise_level=-1,
        audio_mode="copy",
        workers=1,
        frame_format="jpg"
    ),
    "lightweight_1080p": EnhancementPreset(
        id="lightweight_1080p",
        name="Lightweight 1080p FHD",
        icon="📱",
        description="Sharp 1080p upscale with high compression efficiency, optimal for mobile devices or streaming.",
        recommended_for="Plex, Jellyfin, mobile devices, and low-storage environments.",
        target_preset="1080p",
        video_codec="hevc_nvenc",
        model_id="realesr-animevideov3",
        cq_level=20,
        nvenc_preset="p5",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=3,
        frame_format="jpg"
    ),
    "clean_denoise": EnhancementPreset(
        id="clean_denoise",
        name="Legacy DVD / Denoise Restoration",
        icon="🖌️",
        description="Cleans macroblocking, compression ringing, and noise with RealCUGAN neural filters.",
        recommended_for="Heavily compressed web rips, legacy 480p DVD sources, and noisy scans.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="realcugan-pro",
        cq_level=19,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg"
    ),
    "custom": EnhancementPreset(
        id="custom",
        name="Custom / Manual Expert Tuning",
        icon="⚙️",
        description="Unrestricted manual configuration of all rate control, neural engine, and audio parameters.",
        recommended_for="Power users and compressionists who want granular manual control.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="auto",
        cq_level=19,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg"
    )
}


def get_preset(preset_id: str) -> EnhancementPreset:
    """Returns the requested preset or defaults to cartoon_4k_master."""
    return ENHANCEMENT_PRESETS.get(preset_id, ENHANCEMENT_PRESETS["cartoon_4k_master"])
