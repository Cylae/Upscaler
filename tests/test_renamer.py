import os
import pytest
from src.core.scene_renamer import (
    SceneNamingConfig,
    clean_scene_text,
    format_scene_name,
    get_scene_relative_output_path,
    parse_episode_metadata
)


def test_clean_scene_text():
    assert clean_scene_text("Spin Fun Knowin' Ya!") == "Spin.Fun.Knowin.Ya"
    assert clean_scene_text("You're Bakin' Me Crazy!") == "Youre.Bakin.Me.Crazy"
    assert clean_scene_text("Happy Trails (Part 2_ Jumping The Shark)") == "Happy.Trails.Part.2.Jumping.The.Shark"
    assert clean_scene_text("Who's to Flame_") == "Whos.To.Flame"
    assert clean_scene_text("Aw, Shucks!") == "Aw.Shucks"


def test_parse_episode_metadata_season1():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"1999 - 2001 Season 1\s01e01 - Spin Fun Knowin' Ya!.mp4")
    sub_cat, season, episode, title = parse_episode_metadata(path, base)
    assert sub_cat == ""
    assert season == "S01"
    assert episode == "E01"
    assert title == "Spin.Fun.Knowin.Ya"


def test_parse_episode_metadata_season2():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"2002 - 2005 Season 2\s02e26 - Suck It Up!.mp4")
    sub_cat, season, episode, title = parse_episode_metadata(path, base)
    assert sub_cat == ""
    assert season == "S02"
    assert episode == "E26"
    assert title == "Suck.It.Up"


def test_parse_episode_metadata_tv():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"2006 TV\01 - The Wrong Side of the Tracks.mp4")
    sub_cat, season, episode, title = parse_episode_metadata(path, base)
    assert sub_cat == "TV"
    assert season == "S01"
    assert episode == "E01"
    assert title == "The.Wrong.Side.Of.The.Tracks"


def test_parse_episode_metadata_others_smoochies():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"Others\2003 Smoochies\01 - Cuddles' Pet Smoochie.mp4")
    sub_cat, season, episode, title = parse_episode_metadata(path, base)
    assert sub_cat == "2003.Smoochies"
    assert season == "S00"
    assert episode == "E01"
    assert title == "Cuddles.Pet.Smoochie"


def test_format_scene_name_custom_config():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"1999 - 2001 Season 1\s01e01 - Spin Fun Knowin' Ya!.mp4")
    
    cfg = SceneNamingConfig(
        show_name="Happy.Tree.Friends",
        resolution_tag="2160p",
        source_tag="WEB-DL",
        audio_tag="AAC2.0",
        codec_tag="H.265",
        group_tag="CYLAE",
        container=".mp4"
    )
    result = format_scene_name(path, base, cfg)
    assert result == "Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4"
    assert "AI" not in result
    assert "IA" not in result

    # MKV test
    cfg.container = ".mkv"
    cfg.codec_tag = "x265"
    assert format_scene_name(path, base, cfg) == "Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.2160p.WEB-DL.AAC2.0.x265-CYLAE.mkv"


def test_get_scene_relative_output_path():
    base = r"Z:\DL\Happy Tree Friends"
    path = os.path.join(base, r"1999 - 2001 Season 1\s01e01 - Spin Fun Knowin' Ya!.mp4")
    cfg = SceneNamingConfig(group_tag="CYLAE")
    rel_out = get_scene_relative_output_path(path, base, cfg, preserve_folders=True)
    expected_folder = os.path.join("1999.2001.Season.1", "Happy.Tree.Friends.S01E01.Spin.Fun.Knowin.Ya.2160p.WEB-DL.AAC2.0.H.265-CYLAE.mp4")
    assert rel_out == expected_folder
    assert "AI" not in rel_out
    assert "IA" not in rel_out


def test_format_clean_and_original_naming_modes():
    from src.core.scene_renamer import format_clean_name, format_output_name

    # Arbitrary cartoon / video file
    video_path = r"C:\Anime\DragonBall_Episode_01.mkv"

    # 1. Clean mode
    cfg_clean = SceneNamingConfig(naming_mode="clean", resolution_tag="2160p", container=".mp4")
    out_clean = format_output_name(video_path, config=cfg_clean)
    assert out_clean == "DragonBall_Episode_01_4K.mp4"
    assert "AI" not in out_clean
    assert "CYLAE" not in out_clean

    # 2. Original mode
    cfg_orig = SceneNamingConfig(naming_mode="original", container=".mp4")
    out_orig = format_output_name(video_path, config=cfg_orig)
    assert out_orig == "DragonBall_Episode_01.mp4"

    # 3. Direct format_clean_name
    direct = format_clean_name("My Great Cartoon.avi", resolution_tag="1080p", container=".mp4")
    assert direct == "My Great Cartoon_1080p.mp4"
