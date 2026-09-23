from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModelEngine(str, Enum):
    REALESRGAN = "realesrgan"
    REALCUGAN = "realcugan"


@dataclass
class AIModelSpec:
    id: str
    name: str
    engine: ModelEngine
    description: str
    supported_scales: List[int]
    default_scale: int
    recommended_for: str
    internal_name: str  # Parameter name passed to binary


# Curated catalog of models for Cartoons / 2D Animation
AVAILABLE_MODELS: Dict[str, AIModelSpec] = {
    "auto": AIModelSpec(
        id="auto",
        name="Auto Selection (Recommended)",
        engine=ModelEngine.REALESRGAN,
        description="Automatically selects the best model based on source resolution and artifact profile.",
        supported_scales=[2, 3, 4],
        default_scale=4,
        recommended_for="All episode types (720p, 480p, TV, Web)",
        internal_name="auto"
    ),
    "realesrgan-x4plus-anime": AIModelSpec(
        id="realesrgan-x4plus-anime",
        name="Real-ESRGAN Anime 6B (Ultimate Quality)",
        engine=ModelEngine.REALESRGAN,
        description="Gold standard for 2D animation. Sharpens linework and completely eliminates compression macroblocks.",
        supported_scales=[4],
        default_scale=4,
        recommended_for="480p / SD videos or cartoons with blurred outlines",
        internal_name="realesrgan-x4plus-anime"
    ),
    "realesr-animevideov3": AIModelSpec(
        id="realesr-animevideov3",
        name="Real-ESRGAN AnimeVideo V3 (Fast & Smooth)",
        engine=ModelEngine.REALESRGAN,
        description="High-performance video model ensuring optimal temporal coherence and ~2x faster rendering.",
        supported_scales=[2, 3, 4],
        default_scale=3,
        recommended_for="720p to 4K upscaling (exact 3x native scaling)",
        internal_name="realesr-animevideov3"
    ),
    "realcugan-pro": AIModelSpec(
        id="realcugan-pro",
        name="RealCUGAN Pro (Advanced Restoration)",
        engine=ModelEngine.REALCUGAN,
        description="Specialized model for anime and cartoon illustration with fine denoise control.",
        supported_scales=[2, 3, 4],
        default_scale=2,
        recommended_for="Anime and cartoons with residual noise",
        internal_name="models-pro"
    ),
    "realcugan-se": AIModelSpec(
        id="realcugan-se",
        name="RealCUGAN SE (Speed & Efficiency)",
        engine=ModelEngine.REALCUGAN,
        description="Lightweight RealCUGAN variant for high-throughput batch rendering.",
        supported_scales=[2, 3, 4],
        default_scale=2,
        recommended_for="High-volume batches",
        internal_name="models-se"
    ),
    "realesrgan-x4plus": AIModelSpec(
        id="realesrgan-x4plus",
        name="Real-ESRGAN General (High Detail)",
        engine=ModelEngine.REALESRGAN,
        description="Deep 16-block RRDB architecture for photorealistic textures and complex background art.",
        supported_scales=[4],
        default_scale=4,
        recommended_for="Complex backgrounds and mixed animation styles",
        internal_name="realesrgan-x4plus"
    )
}


def check_local_model_availability(base_dir: Optional[str] = None) -> Dict[str, bool]:
    """
    Verifies that all AI models are physically present on local storage.
    Returns a dictionary mapping model_id to boolean availability.
    """
    import os
    if base_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(current_dir))
    else:
        root = base_dir

    r_models = os.path.join(root, "tools", "realesrgan", "models")
    c_pro = os.path.join(root, "tools", "realcugan", "models-pro")
    c_se = os.path.join(root, "tools", "realcugan", "models-se")

    availability = {
        "auto": True,
        "realesrgan-x4plus-anime": os.path.isfile(os.path.join(r_models, "realesrgan-x4plus-anime.bin")),
        "realesr-animevideov3": (
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x2.bin")) and
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x3.bin")) and
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x4.bin"))
        ),
        "realesrgan-x4plus": os.path.isfile(os.path.join(r_models, "realesrgan-x4plus.bin")),
        "realcugan-pro": (
            os.path.isdir(c_pro) and
            os.path.isfile(os.path.join(c_pro, "up2x-conservative.bin"))
        ),
        "realcugan-se": (
            os.path.isdir(c_se) and
            os.path.isfile(os.path.join(c_se, "up2x-conservative.bin"))
        )
    }
    return availability



def choose_best_model_for_video(width: int, height: int) -> Tuple[AIModelSpec, int]:
    """
    Automatically selects the optimal model and scale factor based on video dimensions:
    - 720p (1280x720): 1280 * 3 = 3840, 720 * 3 = 2160 -> Exact native 4K UHD.
      AnimeVideoV3 in scale 3 is ~2x faster, preserves clean line sharpness,
      and matches the 4K pixel grid.
    - 480p (640x480 or 720x480): RealESRGAN Anime 6B (scale 4) produces the crispest linework
      and eliminates severe compression artifacts, then scales cleanly to 4K.
    """
    if width <= 960 or height <= 576:
        # SD resolution (480p / 576p)
        return AVAILABLE_MODELS["realesrgan-x4plus-anime"], 4
    elif width <= 1366 and height <= 768:
        # HD 720p resolution
        return AVAILABLE_MODELS["realesr-animevideov3"], 3
    else:
        # 1080p or other resolution
        return AVAILABLE_MODELS["realesr-animevideov3"], 2


def calculate_target_dimensions(
    orig_w: int,
    orig_h: int,
    target_preset: str = "4k"
) -> Tuple[int, int, str]:
    """
    Calculates target width and height maintaining original aspect ratio.
    Supported presets: '4k' (2160p), '1440p', '1080p', '720p'.
    Returns (width, height, tag).
    """
    aspect = orig_w / max(1, orig_h)
    is_4_3 = 1.25 <= aspect <= 1.45

    if target_preset == "4k":
        target_h = 2160
        target_w = 2880 if is_4_3 else 3840
        tag = "2160p"
    elif target_preset == "1440p":
        target_h = 1440
        target_w = 1920 if is_4_3 else 2560
        tag = "1440p"
    elif target_preset == "1080p":
        target_h = 1080
        target_w = 1440 if is_4_3 else 1920
        tag = "1080p"
    elif target_preset == "720p":
        target_h = 720
        target_w = 960 if is_4_3 else 1280
        tag = "720p"
    else:
        # Default 4K
        target_h = 2160
        target_w = 2880 if is_4_3 else 3840
        tag = "2160p"

    # Make dimensions even (required by H.264/H.265 encoders)
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    return target_w, target_h, tag
