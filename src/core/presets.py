from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered to preserve the existing desktop defaults and resolution choices.
RESOLUTION_LABELS = {
    "4k": "4K UHD (3840×2160)",
    "1440p": "1440p QHD (2560×1440)",
    "1080p": "1080p FHD (1920×1080)",
    "720p": "720p HD (1280×720)",
    "8k": "8K UHD (7680×4320 / Portrait)",
    "ultrawide": "Ultrawide 21:9 (3440×1440 / 5120×2160)",
}
RESOLUTION_TAGS = {
    "8k": "4320p",
    "4k": "2160p",
    "1440p": "1440p",
    "1080p": "1080p",
    "720p": "720p",
    "ultrawide": "1440p",
}


@dataclass(frozen=True)
class EnhancementPreset:
    id: str
    name: str
    icon: str
    description: str
    recommended_for: str
    target_preset: str = "4k"  # "8k", "4k", "1440p", "1080p", "720p"
    video_codec: str = "hevc_nvenc"  # "hevc_nvenc", "h264_nvenc", "libx265"
    model_id: str = "auto"
    cq_level: int = 19  # Constant Quality: 14 (near lossless) to 28 (compact)
    nvenc_preset: str = "p7"  # "p1" to "p7"
    enable_tta: bool = False  # 8-pass test-time augmentation
    denoise_level: int = -1  # -1 (none), 0, 1, 2, 3
    audio_mode: str = "copy"  # "copy", "aac_192k", "aac_320k", "flac"
    workers: int = 2
    frame_format: str = "jpg"


ENHANCEMENT_PRESETS: dict[str, EnhancementPreset] = {
    "anime_4k_master": EnhancementPreset(
        id="anime_4k_master",
        name="Anime 4K Master",
        icon="🎬",
        description="Pristine vector outlines, clean color fills, high NVENC compression, and zero artifacting.",
        recommended_for="Best balance of visual quality, pristine lines, and speed for 2D animation and cartoons.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="auto",
        cq_level=19,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg",
    ),
    "photo_realistic_4k": EnhancementPreset(
        id="photo_realistic_4k",
        name="Photo Realistic 4K",
        icon="📷",
        description="General Real-ESRGAN x4, lossless PNG intermediates, and HEVC CQ 18 for photorealistic footage.",
        recommended_for="Live-action footage, photorealistic video, and 3D CGI sequences.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="realesrgan-x4plus",
        cq_level=18,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=1,
        frame_format="png",
    ),
    "balanced_fast": EnhancementPreset(
        id="balanced_fast",
        name="Balanced Fast (40+ fps)",
        icon="🚀",
        description="High-speed turnaround with Real-ESRGAN AnimeVideo V3 and maximum-speed NVENC P3 encoding.",
        recommended_for="Fast previewing or rapid batch processing of large collections.",
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="realesr-animevideov3",
        cq_level=21,
        nvenc_preset="p3",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg",
    ),
    "ultrawide": EnhancementPreset(
        id="ultrawide",
        name="Ultrawide 21:9 Master",
        icon="🖥️",
        description="Aspect ratio preserved for 21:9 and 32:9 ultrawide cinema formats (3440x1440p / 5120x2160p).",
        recommended_for="Cinema widescreen and ultrawide monitor media.",
        target_preset="1440p",
        video_codec="hevc_nvenc",
        model_id="auto",
        cq_level=19,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=2,
        frame_format="jpg",
    ),
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
        frame_format="jpg",
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
        frame_format="jpg",
    ),
    "realistic_8k": EnhancementPreset(
        id="realistic_8k",
        name="Realistic Video 8K",
        icon="📷",
        description="General Real-ESRGAN x4, lossless PNG intermediates, and HEVC CQ 18 for realistic footage.",
        recommended_for="Live-action footage, including portrait video. One worker keeps GPU memory available for inference.",
        target_preset="8k",
        video_codec="hevc_nvenc",
        model_id="realesrgan-x4plus",
        cq_level=18,
        nvenc_preset="p7",
        enable_tta=False,
        denoise_level=-1,
        audio_mode="copy",
        workers=1,
        frame_format="png",
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
        frame_format="png",
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
        frame_format="jpg",
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
        denoise_level=2,
        audio_mode="copy",
        workers=2,
        frame_format="jpg",
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
        frame_format="jpg",
    ),
}


def get_preset(preset_id: str) -> EnhancementPreset:
    """Returns the requested preset or defaults to cartoon_4k_master."""
    if not preset_id:
        return ENHANCEMENT_PRESETS["cartoon_4k_master"]
    if preset_id in ENHANCEMENT_PRESETS:
        return ENHANCEMENT_PRESETS[preset_id]

    norm = re.sub(r"[\s\-_]+", "", preset_id.lower())
    for pid, p in ENHANCEMENT_PRESETS.items():
        if re.sub(r"[\s\-_]+", "", pid.lower()) == norm:
            return p
        if re.sub(r"[\s\-_]+", "", p.name.lower()) == norm:
            return p

    alias_map = {
        "anime": "anime_4k_master",
        "anime4k": "anime_4k_master",
        "photo": "photo_realistic_4k",
        "photorealistic": "photo_realistic_4k",
        "fast": "balanced_fast",
        "draft": "fast_draft",
        "uw": "ultrawide",
        "wide": "ultrawide",
        "cartoon": "cartoon_4k_master",
    }
    if norm in alias_map and alias_map[norm] in ENHANCEMENT_PRESETS:
        return ENHANCEMENT_PRESETS[alias_map[norm]]

    return ENHANCEMENT_PRESETS["cartoon_4k_master"]
