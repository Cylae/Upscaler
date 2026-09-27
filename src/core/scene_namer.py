"""Warez / Scene naming conventions and privacy metadata scrubber.

Re-exports and extends scene renaming logic from scene_renamer.
"""

from __future__ import annotations

from src.core.scene_renamer import (
    FORBIDDEN_PATTERNS,
    FORBIDDEN_REGEX,
    SceneNamingConfig,
    clean_scene_text,
    find_forbidden_output_markers,
    format_clean_name,
    format_output_name,
    format_scene_name,
    get_scene_relative_output_path,
    parse_episode_metadata,
    sanitize_title_word,
    strip_forbidden_output_markers,
)

# Friendly alias
scrub_text_of_internal_markers = strip_forbidden_output_markers
FORBIDDEN_MARKERS = FORBIDDEN_PATTERNS

__all__ = [
    "FORBIDDEN_PATTERNS",
    "FORBIDDEN_REGEX",
    "FORBIDDEN_MARKERS",
    "SceneNamingConfig",
    "clean_scene_text",
    "find_forbidden_output_markers",
    "format_clean_name",
    "format_output_name",
    "format_scene_name",
    "get_scene_relative_output_path",
    "parse_episode_metadata",
    "sanitize_title_word",
    "strip_forbidden_output_markers",
    "scrub_text_of_internal_markers",
]
