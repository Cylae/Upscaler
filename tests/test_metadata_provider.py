import json
import time
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from src.core.metadata_provider import (
    MetadataLookupError,
    SafePublicMetadata,
    ShowCandidate,
    TVMazeClient,
    _normalized_title,
    sanitize_metadata_text,
)


def test_show_candidate_display_name():
    candidate_full = ShowCandidate(
        id=123,
        name="Happy Tree Friends",
        premiered="2006-10-02",
        network="G4",
        country="United States",
    )
    assert candidate_full.display_name == "Happy Tree Friends — 2006 · G4 · United States"

    candidate_minimal = ShowCandidate(id=456, name="Minimal Show")
    assert candidate_minimal.display_name == "Minimal Show — unknown year"


def test_safe_public_metadata_as_ffmpeg_tags():
    meta = SafePublicMetadata(
        title="Spin Fun Knowin' Ya!",
        show="Happy Tree Friends",
        season_number=1,
        episode_number=1,
        episode_id="S01E01",
        airdate="1999-12-24",
        genres=("Animation", "Comedy"),
        description="Toothy and Lumpy play on a merry-go-round.",
        network="Mondo Media",
        source_url="https://www.tvmaze.com/episodes/123",
    )
    tags = meta.as_ffmpeg_tags()
    assert tags["title"] == "Spin Fun Knowin' Ya!"
    assert tags["show"] == "Happy Tree Friends"
    assert tags["season_number"] == "1"
    assert tags["episode_sort"] == "1"
    assert tags["episode_id"] == "S01E01"
    assert tags["date"] == "1999-12-24"
    assert tags["genre"] == "Animation, Comedy"
    assert tags["publisher"] == "Mondo Media"
    assert "source_url" not in tags  # internal field


def test_sanitize_metadata_text():
    raw_html = "<b>Lumpy &amp; Friends</b>"
    sanitized = sanitize_metadata_text(raw_html)
    assert sanitized == "Lumpy & Friends"

    control_chars = "Hello\x00World\x1f\x7f!"
    assert sanitize_metadata_text(control_chars) == "Hello World !"

    none_val = sanitize_metadata_text(None)
    assert none_val == ""


def test_normalized_title():
    assert _normalized_title("Still Alive #01: Spin Fun") == "spinfun"
    assert _normalized_title("Out on a Limb (Part 1)") == "outonalimbpart1"
    assert _normalized_title("Special / Episode!") == "specialepisode"


def test_tvmaze_client_cache(tmp_path):
    cache_dir = tmp_path / "tvmaze_cache"
    client = TVMazeClient(cache_dir=str(cache_dir), cache_ttl=10)

    # Cache miss
    assert client._read_cache("test_key") is None

    # Write and read back
    data = {"hello": "world"}
    client._write_cache("test_key", data)
    assert client._read_cache("test_key") == data

    # Expired cache without allow_stale
    time.sleep(0.05)
    short_ttl_client = TVMazeClient(cache_dir=str(cache_dir), cache_ttl=0)
    assert short_ttl_client._read_cache("test_key", allow_stale=False) is None
    assert short_ttl_client._read_cache("test_key", allow_stale=True) == data


def test_tvmaze_client_search_shows(tmp_path):
    client = TVMazeClient(cache_dir=str(tmp_path))

    # Empty query
    assert client.search_shows("") == []

    mock_response = [
        {
            "show": {
                "id": 100,
                "name": "Happy Tree Friends",
                "premiered": "2006-10-02",
                "network": {"name": "G4", "country": {"name": "USA"}},
                "url": "https://www.tvmaze.com/shows/100",
            }
        },
        {
            "show": {
                "id": None,  # corrupted id
                "name": "Broken Show",
            }
        },
    ]

    mock_cm = MagicMock()
    mock_cm.read.return_value = json.dumps(mock_response).encode("utf-8")
    mock_cm.__enter__.return_value = mock_cm

    with patch("urllib.request.urlopen", return_value=mock_cm):
        results = client.search_shows("Happy Tree Friends")
        assert len(results) == 1
        assert results[0].id == 100
        assert results[0].name == "Happy Tree Friends"
        assert results[0].network == "G4"
        assert results[0].country == "USA"


def test_tvmaze_client_get_show_bundle_and_episodes(tmp_path):
    client = TVMazeClient(cache_dir=str(tmp_path))

    mock_show = {
        "id": 100,
        "name": "Happy Tree Friends",
        "genres": ["Animation", "Comedy"],
        "network": {"name": "Mondo Media"},
    }
    mock_episodes = [
        {
            "id": 1001,
            "season": 1,
            "number": 1,
            "name": "Spin Fun Knowin' Ya!",
            "airdate": "1999-12-24",
            "summary": "<p>First pilot episode.</p>",
            "url": "https://www.tvmaze.com/episodes/1001",
        },
        {
            "id": 1002,
            "season": 1,
            "number": 2,
            "name": "House Warming",
            "airdate": "1999-12-31",
            "summary": "<p>Second episode.</p>",
            "url": "https://www.tvmaze.com/episodes/1002",
        },
    ]

    def mock_urlopen_side_effect(req, timeout=None):
        cm = MagicMock()
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/episodes" in url:
            cm.read.return_value = json.dumps(mock_episodes).encode("utf-8")
        else:
            cm.read.return_value = json.dumps(mock_show).encode("utf-8")
        cm.__enter__.return_value = cm
        return cm

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect):
        bundle = client.get_show_bundle(100)
        assert bundle["show"]["name"] == "Happy Tree Friends"
        assert len(bundle["episodes"]) == 2

        # Exact match S01E01
        ep_meta = client.get_episode_metadata(100, 1, 1, local_title="Spin Fun Knowin Ya")
        assert ep_meta is not None
        assert ep_meta.title == "Spin Fun Knowin' Ya!"
        assert ep_meta.season_number == 1
        assert ep_meta.episode_number == 1
        assert ep_meta.description == "First pilot episode."
        assert ep_meta.genres == ("Animation", "Comedy")

        # Fuzzy title match fallback when numbering differs
        fuzzy_meta = client.get_episode_metadata(100, 2, 99, local_title="House Warming")
        assert fuzzy_meta is not None
        assert fuzzy_meta.title == "House Warming"
        assert fuzzy_meta.episode_number == 2

        # Unmatched returns None
        unmatched = client.get_episode_metadata(100, 5, 5, local_title="Completely Unknown")
        assert unmatched is None


def test_tvmaze_client_network_error_and_stale_cache(tmp_path):
    client = TVMazeClient(cache_dir=str(tmp_path), cache_ttl=3600)
    cache_key = "show:999:episodes"
    cached_bundle = {"show": {"id": 999, "name": "Cached Show"}, "episodes": []}
    client._write_cache(cache_key, cached_bundle)

    # Simulate network failure
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        # Fresh cache or stale cache hit
        res = client._request_json("/shows/999", cache_key=cache_key)
        assert res == cached_bundle

        # Missing cache key raises MetadataLookupError
        with pytest.raises(MetadataLookupError):
            client._request_json("/shows/888", cache_key="non_existent")
