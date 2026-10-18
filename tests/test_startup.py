"""Keep GUI startup usable from Python installations without desktop packages."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

import main as entrypoint


def _project_python():
    project_dir = Path(entrypoint.__file__).resolve().parent
    return project_dir / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )


def _mock_dependencies(monkeypatch, missing):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name in missing else object(),
    )


def test_gui_runs_in_current_interpreter_when_dependencies_are_available(monkeypatch):
    _mock_dependencies(monkeypatch, set())
    gui_module = ModuleType("src.ui.app_qt")
    gui_module.run_gui = Mock(return_value=27)
    monkeypatch.setitem(sys.modules, "src.ui.app_qt", gui_module)
    child = Mock()
    monkeypatch.setattr(subprocess, "run", child)

    assert entrypoint._run_gui() == 27

    gui_module.run_gui.assert_called_once_with()
    child.assert_not_called()


def test_missing_gui_dependencies_reuse_project_python_and_preserve_arguments(
    monkeypatch, tmp_path, capsys
):
    _mock_dependencies(monkeypatch, {"PyQt6"})
    monkeypatch.setattr(Path, "is_file", lambda path: path == _project_python())
    monkeypatch.setattr(sys, "executable", str(tmp_path / "system python" / "python.exe"))
    arguments = ["--custom-gui-option", "a path with spaces", "--another=value"]
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    child = Mock(return_value=subprocess.CompletedProcess([], 19))
    monkeypatch.setattr(subprocess, "run", child)

    assert entrypoint._run_gui() == 19

    child.assert_called_once_with(
        [str(_project_python()), str(Path(entrypoint.__file__).resolve()), *arguments],
        check=False,
    )
    assert ".venv" in capsys.readouterr().err


@pytest.mark.parametrize("missing", [{"PyQt6"}, {"PIL"}, {"PyQt6", "PIL"}])
def test_unavailable_project_python_reports_missing_packages_and_install_command(
    monkeypatch, tmp_path, capsys, missing
):
    _mock_dependencies(monkeypatch, missing)
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    current_python = str(tmp_path / "system python" / "python.exe")
    monkeypatch.setattr(sys, "executable", current_python)
    child = Mock()
    monkeypatch.setattr(subprocess, "run", child)

    assert entrypoint._run_gui() == 1

    child.assert_not_called()
    error = capsys.readouterr().err
    for package in missing:
        assert package in error
    assert "launch.bat" in error
    requirements = Path(entrypoint.__file__).resolve().parent / "requirements.txt"
    assert f'"{current_python}" -m pip install -r "{requirements}"' in error


def test_project_python_with_missing_dependency_does_not_restart_itself(
    monkeypatch, capsys
):
    _mock_dependencies(monkeypatch, {"PyQt6"})
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(sys, "executable", str(_project_python()))
    child = Mock()
    monkeypatch.setattr(subprocess, "run", child)

    assert entrypoint._run_gui() == 1

    child.assert_not_called()
    assert "PyQt6" in capsys.readouterr().err


def test_unlaunchable_project_python_still_provides_dependency_guidance(
    monkeypatch, tmp_path, capsys
):
    _mock_dependencies(monkeypatch, {"PyQt6"})
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    child = Mock(side_effect=OSError("interpreter unavailable"))
    monkeypatch.setattr(subprocess, "run", child)

    assert entrypoint._run_gui() == 1

    child.assert_called_once()
    error = capsys.readouterr().err
    assert "interpreter unavailable" in error
    assert "PyQt6" in error
    assert "launch.bat" in error
    assert "-m pip install -r" in error


@pytest.mark.parametrize(
    ("arguments", "expected_status"),
    [(["--list-presets"], 23), (["--cli", "--list-presets"], 23), (["--help"], 0)],
)
def test_cli_routes_do_not_probe_gui_dependencies(monkeypatch, arguments, expected_status):
    probe = Mock(side_effect=AssertionError("CLI should not probe GUI dependencies"))
    monkeypatch.setattr(importlib.util, "find_spec", probe)
    cli_module = ModuleType("src.cli")
    cli_module.run_cli = Mock(return_value=23)
    monkeypatch.setitem(sys.modules, "src.cli", cli_module)
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])

    assert entrypoint.main() == expected_status

    cli_module.run_cli.assert_called_once_with()
    probe.assert_not_called()
    assert "--cli" not in sys.argv


def test_doctor_route_does_not_probe_gui_dependencies(monkeypatch):
    probe = Mock(side_effect=AssertionError("Doctor should not probe GUI dependencies"))
    monkeypatch.setattr(importlib.util, "find_spec", probe)
    detector_module = ModuleType("src.core.detector")
    detector_module.print_doctor_report = Mock()
    monkeypatch.setitem(sys.modules, "src.core.detector", detector_module)
    monkeypatch.setattr(sys, "argv", ["main.py", "--doctor"])

    assert entrypoint.main() == 0

    detector_module.print_doctor_report.assert_called_once_with()
    probe.assert_not_called()
