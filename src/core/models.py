"""AI Model specifications, resolution calculators, and automatic recommendation logic.

Supports Real-ESRGAN and RealCUGAN Vulkan ncnn architectures for 2D animation
and high-detail upscaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ModelEngine(str, Enum):
    REALESRGAN = "realesrgan"
    REALCUGAN = "realcugan"
    # Legacy alias
    REAL_ESRGAN = "realesrgan"
    REAL_CUGAN = "realcugan"


@dataclass
class AIModelSpec:
    id: str
    name: str
    engine: ModelEngine
    description: str
    supported_scales: List[int]
    default_scale: int
    recommended_for: str = ""
    internal_name: str = ""
    supports_denoise: bool = False
    vram_profile: str = "medium"
    weights_path: str = ""

    def __post_init__(self):
        if not self.internal_name:
            object.__setattr__(self, "internal_name", self.id)


@dataclass(frozen=True)
class ModelRecommendation:
    """Automatic model decision together with an explanation for the user."""
    model: AIModelSpec
    scale: int
    reason: str


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
        internal_name="auto",
    ),
    "realesrgan-x4plus-anime": AIModelSpec(
        id="realesrgan-x4plus-anime",
        name="Real-ESRGAN Anime 6B (Ultimate Quality)",
        engine=ModelEngine.REALESRGAN,
        description="Gold standard for 2D animation. Sharpens linework and completely eliminates compression macroblocks.",
        supported_scales=[4],
        default_scale=4,
        recommended_for="480p / SD videos or cartoons with blurred outlines",
        internal_name="realesrgan-x4plus-anime",
        vram_profile="medium",
        weights_path="tools/realesrgan/models",
    ),
    "realesr-animevideov3": AIModelSpec(
        id="realesr-animevideov3",
        name="Real-ESRGAN AnimeVideo V3 (Fast & Smooth)",
        engine=ModelEngine.REALESRGAN,
        description="High-performance video model ensuring optimal temporal coherence and ~2x faster rendering.",
        supported_scales=[2, 3, 4],
        default_scale=3,
        recommended_for="720p to 4K upscaling (exact 3x native scaling)",
        internal_name="realesr-animevideov3",
        vram_profile="low",
        weights_path="tools/realesrgan/models",
    ),
    "realcugan-pro": AIModelSpec(
        id="realcugan-pro",
        name="RealCUGAN Pro (High Detail & Denoise)",
        engine=ModelEngine.REALCUGAN,
        description="Deep convolutional model capable of cleaning severe compression artifacts and ringing noise.",
        supported_scales=[2, 3],
        default_scale=2,
        recommended_for="Restoring low-quality web encodes or noisy DVD masters",
        internal_name="models-pro",
        supports_denoise=True,
        vram_profile="high",
        weights_path="tools/realcugan/models-pro",
    ),
    "realcugan-se": AIModelSpec(
        id="realcugan-se",
        name="RealCUGAN SE (Fast Denoise)",
        engine=ModelEngine.REALCUGAN,
        description="Lightweight version of RealCUGAN for fast artifact removal.",
        supported_scales=[2, 3, 4],
        default_scale=2,
        recommended_for="Batch processing with moderate denoising needs",
        internal_name="models-se",
        supports_denoise=True,
        vram_profile="medium",
        weights_path="tools/realcugan/models-se",
    ),
    "realesrgan-x4plus": AIModelSpec(
        id="realesrgan-x4plus",
        name="Real-ESRGAN General (Photorealistic / 3D)",
        engine=ModelEngine.REALESRGAN,
        description="General purpose model for real-life footage or 3D CGI animation with realistic textures.",
        supported_scales=[4],
        default_scale=4,
        recommended_for="3D CGI sequences or live action elements",
        internal_name="realesrgan-x4plus",
        vram_profile="high",
        weights_path="tools/realesrgan/models",
    ),
    "realesrnet-x4plus": AIModelSpec(
        id="realesrnet-x4plus",
        name="Real-ESRNet (Pure Upscale / No Hallucination)",
        engine=ModelEngine.REALESRGAN,
        description="Non-adversarial model that upscales without inventing speculative high-frequency textures.",
        supported_scales=[4],
        default_scale=4,
        recommended_for="Faithful resolution increase without artistic alteration",
        internal_name="realesrnet-x4plus",
        vram_profile="medium",
        weights_path="tools/realesrgan/models",
    ),
}


def check_local_model_availability(base_dir: Optional[str] = None) -> Dict[str, bool]:
    """Verifies that all AI models are physically present on local storage."""
    if base_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(current_dir))
    else:
        root = base_dir

    r_models = os.path.join(root, "tools", "realesrgan", "models")
    c_pro = os.path.join(root, "tools", "realcugan", "models-pro")
    c_se = os.path.join(root, "tools", "realcugan", "models-se")

    return {
        "auto": True,
        "realesrgan-x4plus-anime": os.path.isfile(os.path.join(r_models, "realesrgan-x4plus-anime.bin")),
        "realesr-animevideov3": (
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x2.bin")) and
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x3.bin")) and
            os.path.isfile(os.path.join(r_models, "realesr-animevideov3-x4.bin"))
        ),
        "realesrgan-x4plus": os.path.isfile(os.path.join(r_models, "realesrgan-x4plus.bin")),
        "realesrnet-x4plus": os.path.isfile(os.path.join(r_models, "realesrnet-x4plus.bin")),
        "realcugan-pro": (
            os.path.isdir(c_pro) and
            os.path.isfile(os.path.join(c_pro, "up2x-conservative.bin"))
        ),
        "realcugan-se": (
            os.path.isdir(c_se) and
            os.path.isfile(os.path.join(c_se, "up2x-conservative.bin"))
        ),
    }


def recommend_model_for_video(
    width: int,
    height: int,
    target_preset: str = "4k",
) -> ModelRecommendation:
    """Recommends an animation model from source geometry and requested output."""
    target_w, target_h, _ = calculate_target_dimensions(width, height, target_preset)
    required_scale = max(target_w / max(1, width), target_h / max(1, height))
    source_label = f"{width}×{height}"

    if width <= 960 or height <= 576:
        return ModelRecommendation(
            AVAILABLE_MODELS["realesrgan-x4plus-anime"],
            4,
            f"SD source ({source_label}): significant upscale to {target_w}×{target_h}. "
            "Real-ESRGAN Anime 6B is favored for reconstructing contours and reducing compression artifacts.",
        )

    if width <= 1366 and height <= 768:
        scale = 3 if required_scale > 2 else 2
        return ModelRecommendation(
            AVAILABLE_MODELS["realesr-animevideov3"],
            scale,
            f"HD source ({source_label}): {scale}× scale neatly covers the {target_w}×{target_h} target. "
            "AnimeVideo V3 is chosen to preserve temporal consistency and limit flickering.",
        )

    return ModelRecommendation(
        AVAILABLE_MODELS["realesr-animevideov3"],
        2,
        f"{source_label} source already detailed: a moderate upscale before resizing to "
        f"{target_w}×{target_h} avoids over-sharpening. AnimeVideo V3 preserves stability between frames.",
    )


def choose_best_model_for_video(width: int, height: int, target_preset: str = "4k") -> Tuple[AIModelSpec, int]:
    """Backward-compatible shorthand for recommend_model_for_video."""
    rec = recommend_model_for_video(width, height, target_preset)
    return rec.model, rec.scale


def get_model_display_name(model_id_or_name: str, include_scale: bool = False, scale: Optional[int] = None) -> str:
    """Returns a concise, human-friendly model label for UI lists and tables."""
    if not model_id_or_name:
        return "Auto"

    clean_str = str(model_id_or_name).strip()
    if clean_str.lower() in ("auto", "auto selection", "auto selection (recommended)"):
        return "Auto"

    if clean_str in AVAILABLE_MODELS:
        spec = AVAILABLE_MODELS[clean_str]
        base_name = spec.name.split(" (")[0]
        if include_scale or scale:
            s = scale or spec.default_scale
            return f"{base_name} ({s}x)"
        return base_name

    for m_id, spec in AVAILABLE_MODELS.items():
        if m_id == "auto":
            continue
        spec_base = spec.name.split(" (")[0]
        if clean_str in (spec.name, spec_base, spec.internal_name):
            if include_scale or scale:
                s = scale or spec.default_scale
                return f"{spec_base} ({s}x)"
            return spec_base
        if clean_str.lower() in spec.name.lower() or spec.internal_name.lower() in clean_str.lower():
            if include_scale or scale:
                s = scale or spec.default_scale
                return f"{spec_base} ({s}x)"
            return spec_base

    return clean_str.split(" (")[0]


def calculate_target_dimensions(
    orig_w: int,
    orig_h: int,
    target_preset: str = "4k",
) -> Tuple[int, int, str]:
    """Calculates target output resolution while strictly preserving original aspect ratio.

    Supported presets: '4k' (2160p), '1440p', '1080p', '720p', '2x', '4x'.
    Returns (width, height, tag).
    """
    orig_w = max(2, orig_w)
    orig_h = max(2, orig_h)
    aspect = orig_w / float(orig_h)

    preset_clean = (target_preset or "4k").lower().strip()

    if preset_clean in ("2x", "x2"):
        target_w = orig_w * 2
        target_h = orig_h * 2
        tag = f"{target_h}p"
    elif preset_clean in ("4x", "x4"):
        target_w = orig_w * 4
        target_h = orig_h * 4
        tag = f"{target_h}p"
    else:
        if preset_clean in ("8k", "4320p"):
            base_h, max_w, tag = 4320, 7680, "4320p"
        elif preset_clean in ("4k", "2160p", "uhd"):
            base_h, max_w, tag = 2160, 3840, "2160p"
        elif preset_clean in ("1440p", "qhd", "2k"):
            base_h, max_w, tag = 1440, 2560, "1440p"
        elif preset_clean in ("1080p", "fhd"):
            base_h, max_w, tag = 1080, 1920, "1080p"
        elif preset_clean in ("720p", "hd"):
            base_h, max_w, tag = 720, 1280, "720p"
        else:
            base_h, max_w, tag = 2160, 3840, "2160p"

        if aspect > (16.0 / 9.0 + 0.05):
            target_w = max_w
            target_h = int(round(target_w / aspect))
        elif aspect < 1.0:
            target_h = base_h
            target_w = int(round(target_h * aspect))
        else:
            target_h = base_h
            target_w = int(round(target_h * aspect))

    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))

    return target_w, target_h, tag


def is_heavy_model(model: Optional[AIModelSpec | str] = None) -> bool:
    """Returns True if the model uses heavy architecture (like RRDBNet or RealCUGAN)
    requiring conservative VRAM tile sizes and worker concurrency to avoid Vulkan OOM.
    """
    if model is None:
        return False
    if isinstance(model, AIModelSpec):
        m_id = (model.id or model.internal_name or "").lower()
        profile = getattr(model, "vram_profile", "medium")
    else:
        m_id = str(model).lower()
        spec = AVAILABLE_MODELS.get(m_id)
        profile = getattr(spec, "vram_profile", "medium") if spec else "medium"

    # AnimeVideo V3 is lightweight
    if "animevideov3" in m_id or "animevideo" in m_id:
        return False

    if any(k in m_id for k in ("x4plus", "realesrnet", "models-pro", "models-se")):
        return True

    return profile in ("high", "medium")


def auto_tile_size(
    gpu_vram_gb: float = 0.0,
    model: Optional[AIModelSpec | str] = None,
    *args,
    **kwargs
) -> int:
    """Computes optimal tile_size for ncnn-vulkan upscalers based on available GPU VRAM.

    Larger tiles reduce kernel-launch overhead but require proportionally more VRAM.
    Heavy models (RRDBNet / RealCUGAN) automatically scale down tile size to prevent
    GPU VRAM exhaustion and silent Vulkan crashes.
    """
    effective_vram = gpu_vram_gb
    if "vram_gb" in kwargs:
        effective_vram = kwargs["vram_gb"]
    elif len(args) >= 2:
        effective_vram = args[1]
    elif len(args) == 1 and not isinstance(gpu_vram_gb, (int, float)):
        effective_vram = args[0]

    chosen_model = model or kwargs.get("model")

    try:
        val = float(effective_vram)
    except (TypeError, ValueError):
        val = 0.0

    if is_heavy_model(chosen_model):
        if val >= 16.0:
            return 400
        elif val >= 10.0:
            return 300
        elif val >= 6.0:
            return 200  # Safe and highly performant for 6GB-8GB GPUs (e.g. RTX 3070)
        elif val >= 4.0:
            return 100
        else:
            return 64
    else:
        if val >= 10.0:
            return 0    # No tiling needed — entire frame fits in VRAM
        elif val >= 6.0:
            return 400  # Large tiles, minimal overhead
        elif val >= 4.0:
            return 200  # Medium tiles
        elif val >= 2.0:
            return 100  # Small tiles for low-VRAM GPUs
        else:
            return 64   # Conservative fallback


def get_optimal_thread_spec(
    threads_param: str = "auto",
    gpu_vram_gb: float = 0.0,
    vram_gb: Optional[float] = None,
    cpu_cores: Optional[int] = None,
    model: Optional[AIModelSpec | str] = None,
    *args,
    **kwargs
) -> str:
    """Calculates optimal load:proc:save thread count based on CPU cores and GPU VRAM."""
    if isinstance(threads_param, int) and isinstance(gpu_vram_gb, (int, float)):
        # Called as (cpu_cores, vram_gb)
        cpu_count = threads_param
        effective_vram = float(gpu_vram_gb)
        threads_param = "auto"
    else:
        effective_vram = vram_gb if vram_gb is not None else (gpu_vram_gb if isinstance(gpu_vram_gb, (int, float)) else 0.0)
        cpu_count = cpu_cores if cpu_cores is not None else (os.cpu_count() or 4)

    if threads_param and threads_param != "auto":
        return str(threads_param)

    chosen_model = model or kwargs.get("model")
    heavy = is_heavy_model(chosen_model)

    # Determine proc threads based on available VRAM and model architecture
    if heavy:
        if effective_vram >= 16.0:
            proc = 6
        elif effective_vram >= 10.0:
            proc = 4
        elif effective_vram >= 6.0:
            proc = 4  # 4 proc threads with tile 200 fits comfortably within 8GB VRAM
        else:
            proc = 2
    else:
        if effective_vram >= 10.0:
            proc = 8
        elif effective_vram >= 6.0:
            proc = 6
        elif effective_vram >= 4.0:
            proc = 4
        else:
            proc = 4

    # Determine load/save threads based on CPU core count
    if cpu_count >= 32:
        load_save = 12
    elif cpu_count >= 16:
        load_save = 8
    elif cpu_count >= 8:
        load_save = 3
    elif cpu_count >= 4:
        load_save = 2
    else:
        load_save = 1

    return f"{load_save}:{proc}:{load_save}"


def resolve_realcugan_options(
    models_dir: str,
    requested_scale: int = 2,
    requested_denoise: int = -1,
    *args,
    **kwargs
) -> Tuple[int, int, str]:
    """Finds the best valid scale, denoise level, and model subfolder for RealCUGAN."""
    if isinstance(models_dir, int):
        # Called as resolve_realcugan_options(scale, denoise, sync_mode)
        scale = models_dir
        denoise = requested_scale if isinstance(requested_scale, int) else -1
        sync_mode = str(requested_denoise) if requested_denoise != -1 else "auto"
        valid_scale = scale if scale in (2, 3, 4) else 2
        valid_denoise = denoise if denoise in (-1, 0, 1, 2, 3) else -1
        return valid_scale, valid_denoise, sync_mode

    mdir = str(models_dir)
    req_scale = requested_scale if isinstance(requested_scale, int) else 2
    req_denoise = requested_denoise if isinstance(requested_denoise, int) else -1

    base_name = os.path.basename(os.path.normpath(mdir))
    parent_dir = os.path.dirname(os.path.normpath(mdir))

    has_pro = (
        (base_name == "models-pro" and os.path.isdir(mdir))
        or os.path.isdir(os.path.join(mdir, "models-pro"))
        or os.path.isdir(os.path.join(parent_dir, "models-pro"))
    )

    if base_name == "models-se":
        model_name = "models-se"
    elif req_scale == 2 and has_pro:
        model_name = "models-pro"
    else:
        model_name = "models-se"

    if model_name == "models-pro":
        valid_denoise = {-1, 0, 2, 3}
    elif req_scale in (3, 4):
        valid_denoise = {-1, 0, 3}
    else:
        valid_denoise = {-1, 0, 1, 2, 3}

    actual_denoise = req_denoise if req_denoise in valid_denoise else -1
    actual_scale = req_scale if req_scale in (2, 3, 4) else 2
    return actual_scale, actual_denoise, model_name
