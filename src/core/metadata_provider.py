import difflib
import hashlib
import html
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.scene_renamer import (
    find_forbidden_output_markers,
    strip_forbidden_output_markers,
)


TVMAZE_API_ROOT = "https://api.tvmaze.com"
TVMAZE_SITE_ROOT = "https://www.tvmaze.com"


class MetadataLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShowCandidate:
    id: int
    name: str
    premiered: str = ""
    network: str = ""
    country: str = ""
    url: str = ""
    episode_count: int = 0

    @property
    def display_name(self) -> str:
        year = self.premiered[:4] if self.premiered else "année inconnue"
        detail = " · ".join(part for part in (year, self.network, self.country) if part)
        return f"{self.name} — {detail}" if detail else self.name


@dataclass(frozen=True)
class SafePublicMetadata:
    title: str
    show: str
    season_number: int
    episode_number: int
    episode_id: str
    airdate: str = ""
    genres: Tuple[str, ...] = ()
    description: str = ""
    network: str = ""
    source_url: str = ""

    def as_ffmpeg_tags(self) -> Dict[str, str]:
        tags = {
            "title": self.title,
            "show": self.show,
            "season_number": str(self.season_number),
            "episode_sort": str(self.episode_number),
            "episode_id": self.episode_id,
            "date": self.airdate,
            "genre": ", ".join(self.genres),
            "description": self.description,
            "synopsis": self.description,
            "publisher": self.network,
        }
        return {key: value for key, value in tags.items() if value}


def sanitize_metadata_text(value: object, max_length: int = 2048) -> str:
    """Converts remote text to a small, plain and privacy-safe tag value."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_forbidden_output_markers(text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:max_length].rstrip()
    if find_forbidden_output_markers(text):
        return ""
    return text


def _normalized_title(value: str) -> str:
    value = sanitize_metadata_text(value, 300).casefold()
    value = re.sub(r"^still\s+alive\s*#?\d+\s*[:\-]\s*", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def _default_cache_dir() -> str:
    local_data = os.environ.get("LOCALAPPDATA")
    if local_data:
        return os.path.join(local_data, "Cylae", "VideoEnhancer", "cache", "tvmaze")
    return os.path.join(tempfile.gettempdir(), "CylaeVideoEnhancer", "cache", "tvmaze")


class TVMazeClient:
    """Small TVmaze client with an app-local cache and safe episode matching."""

    def __init__(self, cache_dir: Optional[str] = None, timeout: float = 8.0, cache_ttl: int = 86_400):
        self.cache_dir = cache_dir or _default_cache_dir()
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._memory: Dict[int, dict] = {}

    def _cache_path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, digest + ".json")

    def _read_cache(self, key: str, allow_stale: bool = False):
        path = self._cache_path(key)
        try:
            age = time.time() - os.path.getmtime(path)
            if age > self.cache_ttl and not allow_stale:
                return None
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            return None

    def _write_cache(self, key: str, payload) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        destination = self._cache_path(key)
        temporary = f"{destination}.{os.getpid()}.{threading.get_ident()}.part"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temporary, destination)
        finally:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass

    def _request_json(self, path: str, params: Optional[Dict[str, object]] = None, cache_key: str = ""):
        if cache_key:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        query = urllib.parse.urlencode(params or {})
        url = f"{TVMAZE_API_ROOT}{path}" + (f"?{query}" if query else "")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Cylae-VideoEnhancer/1.0 (desktop metadata client)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            stale = self._read_cache(cache_key, allow_stale=True) if cache_key else None
            if stale is not None:
                return stale
            raise MetadataLookupError(f"TVmaze indisponible: {exc}") from exc
        if cache_key:
            try:
                self._write_cache(cache_key, payload)
            except OSError:
                # Metadata remains usable when the cache directory is read-only.
                pass
        return payload

    def search_shows(self, query: str) -> List[ShowCandidate]:
        cleaned_query = sanitize_metadata_text(query, 200)
        if not cleaned_query:
            return []
        payload = self._request_json(
            "/search/shows",
            {"q": cleaned_query},
            cache_key=f"search:{cleaned_query.casefold()}",
        )
        candidates: List[ShowCandidate] = []
        for result in payload[:20]:
            show = result.get("show") or {}
            network_data = show.get("network") or show.get("webChannel") or {}
            country_data = network_data.get("country") or {}
            try:
                show_id = int(show.get("id"))
            except (TypeError, ValueError):
                continue
            candidates.append(
                ShowCandidate(
                    id=show_id,
                    name=sanitize_metadata_text(show.get("name"), 200),
                    premiered=sanitize_metadata_text(show.get("premiered"), 20),
                    network=sanitize_metadata_text(network_data.get("name"), 120),
                    country=sanitize_metadata_text(country_data.get("name"), 80),
                    url=sanitize_metadata_text(show.get("url"), 500),
                )
            )
        return candidates

    def get_show_bundle(self, show_id: int) -> dict:
        show_id = int(show_id)
        if show_id in self._memory:
            return self._memory[show_id]

        cache_key = f"show:{show_id}:episodes"
        cached = self._read_cache(cache_key)
        if cached is not None:
            self._memory[show_id] = cached
            return cached

        show = self._request_json(f"/shows/{show_id}")
        episodes = self._request_json(f"/shows/{show_id}/episodes", {"specials": 1})
        bundle = {"show": show, "episodes": episodes}
        try:
            self._write_cache(cache_key, bundle)
        except OSError:
            pass
        self._memory[show_id] = bundle
        return bundle

    def get_episode_metadata(
        self,
        show_id: int,
        season_number: int,
        episode_number: int,
        local_title: str = "",
    ) -> Optional[SafePublicMetadata]:
        bundle = self.get_show_bundle(show_id)
        episodes = bundle.get("episodes") or []
        local_key = _normalized_title(local_title)

        exact_number = [
            episode for episode in episodes
            if episode.get("season") == season_number and episode.get("number") == episode_number
        ]
        candidates = exact_number

        if local_key:
            # Numbering differs between several Happy Tree Friends releases;
            # title agreement is required before trusting an S/E match.
            agreed = [
                episode for episode in exact_number
                if difflib.SequenceMatcher(None, local_key, _normalized_title(episode.get("name", ""))).ratio() >= 0.78
            ]
            if agreed:
                candidates = agreed
            else:
                scored = []
                for episode in episodes:
                    remote_key = _normalized_title(episode.get("name", ""))
                    if not remote_key:
                        continue
                    score = difflib.SequenceMatcher(None, local_key, remote_key).ratio()
                    if score >= 0.90:
                        scored.append((score, episode))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
                    candidates = [scored[0][1]]
                else:
                    return None

        if len(candidates) != 1:
            return None

        episode = candidates[0]
        show = bundle.get("show") or {}
        network_data = show.get("network") or show.get("webChannel") or {}
        remote_season = int(episode.get("season") or season_number)
        remote_episode = int(episode.get("number") or episode_number)
        title = sanitize_metadata_text(episode.get("name"), 300)
        show_name = sanitize_metadata_text(show.get("name"), 300)
        if not title or not show_name:
            return None

        return SafePublicMetadata(
            title=title,
            show=show_name,
            season_number=remote_season,
            episode_number=remote_episode,
            episode_id=f"S{remote_season:02d}E{remote_episode:02d}",
            airdate=sanitize_metadata_text(episode.get("airdate"), 20),
            genres=tuple(
                genre for genre in (sanitize_metadata_text(item, 80) for item in show.get("genres") or []) if genre
            ),
            description=sanitize_metadata_text(episode.get("summary"), 2048),
            network=sanitize_metadata_text(network_data.get("name"), 120),
            source_url=sanitize_metadata_text(episode.get("url"), 500),
        )
