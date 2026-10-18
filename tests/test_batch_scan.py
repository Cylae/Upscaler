import os
import pytest
from src.core.batch_manager import BatchManager, EpisodeStatus
from src.core.hardware import HardwareInfo
from src.core.scene_renamer import SceneNamingConfig


@pytest.mark.parametrize("preserve_folders", [True, False])
@pytest.mark.parametrize("resolution_tag", ["2160p", "4320p"])
def test_batch_scan_sample_episodes(tmp_path, monkeypatch, preserve_folders, resolution_tag):
    """Index generated placeholders and verify Scene names and relative folders."""
    source_dir = tmp_path / "source"
    out_dir = tmp_path / "output"
    episodes = [
        ("Season 01", "S01E01 - First Steps.mp4", "S01E01.First.Steps"),
        ("Season 02", "S02E03 - Summer Night.MKV", "S02E03.Summer.Night"),
    ]
    for folder, filename, _ in episodes:
        source = source_dir / folder / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"placeholder media; never decoded")
    (source_dir / "notes.txt").write_text("Not a video", encoding="utf-8")

    def no_output_exists(filepath, **_kwargs):
        assert os.path.commonpath([str(out_dir), os.path.abspath(filepath)]) == str(out_dir)
        return False

    monkeypatch.setattr("src.core.batch.is_file_complete_and_valid", no_output_exists)
    bm = BatchManager(HardwareInfo())
    cfg = SceneNamingConfig(show_name="Sample.Show", group_tag="EXAMPLE", resolution_tag=resolution_tag)

    queue = bm.scan_folder(str(source_dir), str(out_dir), cfg, preserve_folders=preserve_folders)
    assert len(queue) == len(episodes)
    for index, (item, (folder, filename, episode_tag)) in enumerate(zip(queue, episodes), start=1):
        expected_name = f"Sample.Show.{episode_tag}.{resolution_tag}.WEB-DL.AAC2.0.H.265-EXAMPLE.mp4"
        assert item.index == index
        assert item.source_path == str(source_dir / folder / filename)
        assert item.relative_source_path == os.path.join(folder, filename)
        assert item.target_scene_name == expected_name
        expected_folder = folder.replace(" ", ".") if preserve_folders else ""
        assert item.target_full_path == os.path.join(str(out_dir), expected_folder, expected_name)
        assert item.status == EpisodeStatus.PENDING
