import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


FORBIDDEN_PATTERNS = [
    r"(?<![a-zA-Z0-9])a\.i\.(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])i\.a\.(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])ai(?![a-zA-Z0-9])",
    r"(?<![a-zA-Z0-9])ia(?![a-zA-Z0-9])",
    r"\bartificial\s+intelligence\b",
    r"\bintelligence\s+artificielle\b",
    r"\breal[\-_.]?esrgan\b",
    r"\breal[\-_.]?cugan\b",
    # These are tooling fingerprints. Match them even when embedded in a
    # user-supplied group/show tag (e.g. "MyUpscaler"), where word boundaries
    # would otherwise leak the internal workflow into a release name.
    r"upscaled",
    r"upscaler",
    r"upscaling",
]
FORBIDDEN_REGEX = re.compile("|".join(FORBIDDEN_PATTERNS), re.IGNORECASE)


def find_forbidden_output_markers(s: str) -> List[str]:
    """Finds any forbidden AI / upscaler reference markers in text."""
    if not s:
        return []
    return [m.group(0) for m in FORBIDDEN_REGEX.finditer(s)]


def strip_forbidden_output_markers(s: str) -> str:
    """Strips any forbidden AI or upscaler tokens from text."""
    if not s:
        return ""
    return FORBIDDEN_REGEX.sub(" ", s)


@dataclass
class SceneNamingConfig:
    naming_mode: str = "scene"  # "scene", "clean" (e.g. Original_4K.mp4), "original"
    show_name: str = "Happy.Tree.Friends"
    resolution_tag: str = "2160p"
    source_tag: str = "WEB-DL"
    audio_tag: str = "AAC2.0"
    codec_tag: str = "H.265"
    group_tag: str = "CYLAE"
    container: str = ".mp4"  # or .mkv
    custom_suffix: str = ""


def sanitize_title_word(word: str) -> str:
    """Clean individual word for scene compliance."""
    cleaned = re.sub(r"[\'\"’`\^\~\*]", "", word)
    if not cleaned:
        return ""
    if cleaned.isupper():
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def clean_scene_text(text: str) -> str:
    """Sanitizes text by stripping forbidden tokens, replacing separators and removing invalid characters."""
    if not text:
        return ""

    text = strip_forbidden_output_markers(text)
    tokens = re.split(r"[\s._\-\(\)\[\]\{\}\:\;\,\!\?\/\\\|\+\=\&\#\@]+", text)
    clean_words = [sanitize_title_word(w) for w in tokens if w]
    return ".".join(clean_words)


def parse_episode_metadata(filepath: str, base_dir: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Analyzes filepath and extracts:
    - sub_category (e.g. 'TV', '2003.Smoochies', 'Irregular', '')
    - season (e.g. 'S01', 'S02', 'S00')
    - episode (e.g. 'E01', 'E25')
    - episode_title (e.g. 'Spin.Fun.Knowin.Ya')
    """
    rel_path = os.path.relpath(filepath, base_dir) if base_dir else os.path.basename(filepath)
    parts = rel_path.split(os.sep)
    filename = os.path.splitext(parts[-1])[0]

    folder = parts[0] if len(parts) > 1 else ""
    sub_folder = parts[1] if len(parts) > 2 else ""

    sub_category = ""
    season = "S01"
    episode = "E01"
    title_raw = ""

    # Regex patterns
    m_s_e = re.search(r"s(\d+)e(\d+)\s*[-_]?\s*(.*)", filename, re.I)
    m_leading_num = re.search(r"^(\d+)\s*[-_]?\s*(.*)", filename)
    m_season_folder = re.search(r"Season\s*(\d+)", folder, re.I)
    m_tv_folder = re.search(r"2006\s*TV", folder, re.I)
    m_others_folder = re.search(r"Others", folder, re.I)

    if m_s_e:
        s_num = int(m_s_e.group(1))
        e_num = int(m_s_e.group(2))
        season = f"S{s_num:02d}"
        episode = f"E{e_num:02d}"
        title_raw = m_s_e.group(3)
    elif m_tv_folder and m_leading_num:
        sub_category = "TV"
        season = "S01"
        e_num = int(m_leading_num.group(1))
        episode = f"E{e_num:02d}"
        title_raw = m_leading_num.group(2)
    elif m_others_folder:
        category_name = sub_folder or "Special"
        sub_category = clean_scene_text(category_name)
        season = "S00"
        if m_leading_num:
            e_num = int(m_leading_num.group(1))
            episode = f"E{e_num:02d}"
            title_raw = m_leading_num.group(2)
        else:
            episode = "E01"
            title_raw = filename
    elif m_leading_num:
        s_num = int(m_season_folder.group(1)) if m_season_folder else 1
        season = f"S{s_num:02d}"
        e_num = int(m_leading_num.group(1))
        episode = f"E{e_num:02d}"
        title_raw = m_leading_num.group(2)
    else:
        # Fallback
        s_num = int(m_season_folder.group(1)) if m_season_folder else 1
        season = f"S{s_num:02d}"
        episode = "E01"
        title_raw = filename

    clean_title = clean_scene_text(title_raw)
    return sub_category, season, episode, clean_title


def format_scene_name(
    filepath: str,
    base_dir: Optional[str] = None,
    config: Optional[SceneNamingConfig] = None
) -> str:
    """Generates the full Warez/Scene release filename for a video."""
    cfg = config or SceneNamingConfig()
    sub_category, season, episode, clean_title = parse_episode_metadata(filepath, base_dir)

    clean_show = clean_scene_text(cfg.show_name) or "Happy.Tree.Friends"
    clean_sub = clean_scene_text(sub_category)

    # Scrub forbidden markers while preserving valid dashes/dots in tags
    clean_src = strip_forbidden_output_markers(cfg.source_tag).strip().replace(" ", "")
    clean_aud = strip_forbidden_output_markers(cfg.audio_tag).strip().replace(" ", "")
    clean_cod = strip_forbidden_output_markers(cfg.codec_tag).strip().replace(" ", "")
    clean_grp = strip_forbidden_output_markers(cfg.group_tag).strip().replace(" ", "")

    elements = [clean_show]

    if clean_sub:
        elements.append(clean_sub)

    elements.append(f"{season}{episode}")

    if clean_title:
        elements.append(clean_title)

    if cfg.resolution_tag:
        elements.append(clean_scene_text(cfg.resolution_tag))
    if clean_src:
        elements.append(clean_src)
    if clean_aud:
        elements.append(clean_aud)
    if clean_cod:
        elements.append(clean_cod)

    group_part = f"-{clean_grp}" if clean_grp else ""
    ext = cfg.container if cfg.container.startswith(".") else f".{cfg.container}"

    filename = ".".join([e for e in elements if e]) + group_part + ext

    # Safety check: if forbidden markers still detected, clean thoroughly
    if find_forbidden_output_markers(filename):
        filename = clean_scene_text(filename) + ext
    return filename


def format_clean_name(
    filepath: str,
    resolution_tag: str = "2160p",
    container: str = ".mp4",
    custom_suffix: str = ""
) -> str:
    """Formats a clean, human-friendly output filename without scene jargon: VideoName_4K.mp4."""
    stem, orig_ext = os.path.splitext(os.path.basename(filepath))
    clean_stem = strip_forbidden_output_markers(stem).strip()
    trimmed = re.sub(r'[\s._-]+(?:2160p|1440p|1080p|720p|480p|4k|uhd)$', '', clean_stem, flags=re.IGNORECASE).rstrip(' ._-')
    if trimmed:
        clean_stem = trimmed
    if not clean_stem:
        clean_stem = "video"

    ext = container if (container and container.startswith(".")) else f".{container or orig_ext.lstrip('.') or 'mp4'}"
    
    if custom_suffix:
        tag = custom_suffix if custom_suffix.startswith("_") else f"_{custom_suffix}"
    else:
        r_low = resolution_tag.strip().lower()
        if r_low in ("2160p", "4k", "uhd"):
            res = "4K"
        elif r_low in ("1440p", "2k", "qhd"):
            res = "1440p"
        elif r_low in ("1080p", "fhd"):
            res = "1080p"
        elif r_low in ("720p", "hd"):
            res = "720p"
        elif r_low in ("480p", "sd"):
            res = "480p"
        else:
            res = resolution_tag.strip()
        tag = f"_{res}" if res else "_4K"

    clean_filename = f"{clean_stem}{tag}{ext}"
    if find_forbidden_output_markers(clean_filename):
        clean_filename = clean_scene_text(clean_stem) + f"{tag}{ext}"
    return clean_filename


def format_output_name(
    filepath: str,
    base_dir: Optional[str] = None,
    config: Optional[SceneNamingConfig] = None
) -> str:
    """Formats output filename based on selected naming mode (clean, scene, or original)."""
    cfg = config or SceneNamingConfig()
    mode = getattr(cfg, "naming_mode", "scene")

    if mode == "clean":
        return format_clean_name(filepath, cfg.resolution_tag, cfg.container, cfg.custom_suffix)
    elif mode == "original":
        stem, orig_ext = os.path.splitext(os.path.basename(filepath))
        ext = cfg.container if cfg.container else orig_ext
        return f"{stem}{ext}"
    else:
        return format_scene_name(filepath, base_dir, cfg)


def get_scene_relative_output_path(
    filepath: str,
    base_dir: Optional[str] = None,
    config: Optional[SceneNamingConfig] = None,
    preserve_folders: bool = True
) -> str:
    """Returns the relative path under the output directory (preserving folder structure if desired)."""
    cfg = config or SceneNamingConfig()
    mode = getattr(cfg, "naming_mode", "scene")
    target_name = format_output_name(filepath, base_dir, cfg)

    if not preserve_folders or not base_dir:
        return target_name

    rel_path = os.path.relpath(filepath, base_dir)
    folder = os.path.dirname(rel_path)
    if folder:
        if mode == "scene":
            cleaned_folder_parts = [clean_scene_text(p) for p in folder.split(os.sep) if clean_scene_text(p)]
        else:
            cleaned_folder_parts = [p for p in folder.split(os.sep) if p]
        if cleaned_folder_parts:
            return os.path.join(*cleaned_folder_parts, target_name)
    return target_name
