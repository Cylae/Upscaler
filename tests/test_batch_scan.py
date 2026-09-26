import os
import pytest
from src.core.batch_manager import BatchManager, EpisodeStatus
from src.core.scene_renamer import SceneNamingConfig


def test_batch_scan_sample_episodes():
    """Tests indexing on real episode files without triggering full upscaling renders."""
    source_dir = r"Z:\DL\Happy Tree Friends"
    if not os.path.isdir(source_dir):
        pytest.skip("Source directory not accessible")

    bm = BatchManager()
    cfg = SceneNamingConfig(group_tag="CYLAE")
    out_dir = r"Z:\DL\Happy Tree Friends (2160p)"

    queue = bm.scan_folder(source_dir, out_dir, cfg, preserve_folders=True)
    assert len(queue) > 0

    # Validate first 20 episodes
    sample = queue[:20]
    for item in sample:
        assert item.target_scene_name.startswith("Happy.Tree.Friends.")
        assert item.target_scene_name.endswith(".mp4")
        assert "2160p" in item.target_scene_name
        assert "WEB-DL" in item.target_scene_name
        assert "-CYLAE" in item.target_scene_name
        # Guarantee no AI/IA in output name
        assert "AI" not in item.target_scene_name
        assert "IA" not in item.target_scene_name
