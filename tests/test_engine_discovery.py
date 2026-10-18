"""AI engines are rediscovered without replacing settings or probing hardware."""

import os
from unittest.mock import Mock

import pytest

from src.core import hardware
from src.core.hardware import HardwareInfo
from src.core.models import ModelEngine


ENGINES = ["realesrgan", "realcugan", ModelEngine.REALESRGAN, ModelEngine.REALCUGAN]


def engine_name(engine):
    return engine.value if isinstance(engine, ModelEngine) else engine


def binary_name(engine):
    return f"{engine_name(engine)}-ncnn-vulkan" + (".exe" if os.name == "nt" else "")


def write_binary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test executable; never execute")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "get_base_dir", lambda: str(tmp_path))
    which = Mock(return_value=None)
    monkeypatch.setattr(hardware.shutil, "which", which)
    # Recovery must only check engine files, never rerun GPU/FFmpeg detection.
    for name in ("detect_nvidia_gpu", "detect_ffmpeg"):
        monkeypatch.setattr(hardware, name, Mock(side_effect=AssertionError(f"Unexpected {name}")))
    for name in ("check_output", "run"):
        monkeypatch.setattr(hardware.subprocess, name, Mock(side_effect=AssertionError("Unexpected subprocess")))
    return tmp_path, which


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("stale", [False, True], ids=["missing", "stale"])
def test_resolves_local_engine_and_updates_only_selected_shared_field(discovery, engine, stale):
    base, which = discovery
    selected = engine_name(engine)
    other = "realcugan" if selected == "realesrgan" else "realesrgan"
    selected_field, other_field = f"{selected}_bin", f"{other}_bin"
    custom_other = write_binary(base / "custom" / binary_name(other))
    info = HardwareInfo(**{selected_field: str(base / "removed.exe") if stale else None, other_field: custom_other})
    shared_info = info
    expected = write_binary(base / "tools" / selected / binary_name(engine))
    # Finding both portable engines must not replace the other custom path.
    write_binary(base / "tools" / other / binary_name(other))

    result = hardware.resolve_ai_binary(info, engine)

    assert result == expected
    assert getattr(shared_info, selected_field) == expected
    assert getattr(shared_info, other_field) == custom_other
    which.assert_not_called()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("stale", [False, True], ids=["missing", "stale"])
def test_resolves_engine_from_path_when_local_binary_is_unavailable(discovery, engine, stale):
    base, which = discovery
    field = f"{engine_name(engine)}_bin"
    expected = write_binary(base / "path-bin" / binary_name(engine))
    which.side_effect = lambda command: expected if command == binary_name(engine) else None
    info = HardwareInfo(**{field: str(base / "removed.exe") if stale else None})

    assert hardware.resolve_ai_binary(info, engine) == expected
    assert getattr(info, field) == expected
    assert any(call.args == (binary_name(engine),) for call in which.call_args_list)


@pytest.mark.parametrize("engine", ENGINES)
def test_valid_custom_engine_is_preserved_without_any_discovery(discovery, monkeypatch, engine):
    base, which = discovery
    selected = engine_name(engine)
    custom = write_binary(base / "custom engine" / binary_name(engine))
    write_binary(base / "tools" / selected / binary_name(engine))
    info = HardwareInfo(**{f"{selected}_bin": custom})
    detect = Mock(side_effect=AssertionError("Valid custom paths must not be replaced"))
    monkeypatch.setattr(hardware, "detect_ai_binaries", detect)
    monkeypatch.setattr(hardware, "get_base_dir", Mock(side_effect=AssertionError("No discovery needed")))

    assert hardware.resolve_ai_binary(info, engine) == custom
    assert getattr(info, f"{selected}_bin") == custom
    detect.assert_not_called()
    which.assert_not_called()


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("stale", [False, True], ids=["missing", "stale"])
def test_missing_engine_error_explains_exact_installation_and_allows_late_retry(discovery, engine, stale):
    base, _which = discovery
    selected = engine_name(engine)
    field = f"{selected}_bin"
    previous = str(base / "removed.exe") if stale else None
    info = HardwareInfo(**{field: previous})
    expected_path = base / "tools" / selected / binary_name(engine)

    with pytest.raises(FileNotFoundError) as error:
        hardware.resolve_ai_binary(info, engine)

    message = str(error.value)
    assert str(expected_path) in message
    assert ("Real-ESRGAN" if selected == "realesrgan" else "RealCUGAN") in message
    assert "model" in message.casefold()
    assert "PATH" in message
    assert getattr(info, field) == previous
    if stale:
        assert previous in message

    # The user installs the engine after a failed attempt in the same session.
    expected = write_binary(expected_path)
    assert hardware.resolve_ai_binary(info, engine) == expected
    assert getattr(info, field) == expected


@pytest.mark.parametrize("engine", ["realesrgan", "realcugan"])
def test_disappearing_cached_engine_is_replaced_by_new_path_installation(discovery, engine):
    base, which = discovery
    field = f"{engine}_bin"
    local = base / "tools" / engine / binary_name(engine)
    info = HardwareInfo(**{field: write_binary(local)})
    assert hardware.resolve_ai_binary(info, engine) == str(local)
    local.unlink()
    replacement = write_binary(base / "replacement" / binary_name(engine))
    which.side_effect = lambda command: replacement if command == binary_name(engine) else None

    assert hardware.resolve_ai_binary(info, engine) == replacement
    assert getattr(info, field) == replacement


@pytest.mark.parametrize("engine", ["realesrgan", "realcugan"])
def test_nonexistent_path_discovery_result_is_not_accepted(discovery, engine):
    base, which = discovery
    which.return_value = str(base / "missing-binary.exe")
    with pytest.raises(FileNotFoundError):
        hardware.resolve_ai_binary(HardwareInfo(), engine)
