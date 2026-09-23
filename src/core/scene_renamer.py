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
    r"\bupscaled\b",
    r"\bupscaler\b",
    r"\bupscaling\b",
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
    show_name: str = "Happy.Tree.Friends"
    resolution_tag: str = "2160p"
    source_tag: str = "WEB-DL"
    audio_tag: str = "AAC2.0"
    codec_tag: str = "H.265"
    group_tag: str = "CYLAE"
    container: str = ".mp4"  # or .mkv


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


def get_scene_relative_output_path(
    filepath: str,
    base_dir: Optional[str] = None,
    config: Optional[SceneNamingConfig] = None,
    preserve_folders: bool = True
) -> str:
    """Returns the relative path under the output directory (preserving season folder structure if desired)."""
    scene_name = format_scene_name(filepath, base_dir, config)
    if not preserve_folders or not base_dir:
        return scene_name

    rel_path = os.path.relpath(filepath, base_dir)
    folder = os.path.dirname(rel_path)
    if folder:
        cleaned_folder_parts = [clean_scene_text(p) for p in folder.split(os.sep) if clean_scene_text(p)]
        return os.path.join(*cleaned_folder_parts, scene_name)
    return scene_name
