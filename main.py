import importlib.util
import os
from pathlib import Path
import subprocess
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_gui():
    """Reuse the project's environment when launched with a bare Python."""
    missing = [name for name in ("PyQt6", "PIL") if importlib.util.find_spec(name) is None]
    if missing:
        project_dir = Path(__file__).resolve().parent
        venv_python = project_dir / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        current_python = os.path.normcase(os.path.abspath(sys.executable))
        if venv_python.is_file() and os.path.normcase(os.path.abspath(venv_python)) != current_python:
            print("Using the project's .venv for GUI dependencies...", file=sys.stderr, flush=True)
            try:
                return subprocess.run(
                    [str(venv_python), str(project_dir / "main.py"), *sys.argv[1:]],
                    check=False,
                ).returncode
            except OSError as exc:
                print(f"Unable to start the project's .venv: {exc}", file=sys.stderr)

        shell_prefix = "& " if os.name == "nt" else ""
        print(
            f"Missing GUI dependencies: {', '.join(missing)}.\n"
            "Run .\\launch.bat on Windows to set up the project environment, or install with:\n"
            f'  {shell_prefix}"{sys.executable}" -m pip install -r "{project_dir / "requirements.txt"}"',
            file=sys.stderr,
        )
        return 1

    from src.ui.app_qt import run_gui
    return run_gui()


def main():
    if "--doctor" in sys.argv:
        from src.core.detector import print_doctor_report
        print_doctor_report()
        return 0

    if "--help" in sys.argv or "-h" in sys.argv:
        from src.cli import run_cli
        run_cli()
        return 0

    cli_triggers = {
        "--cli", "--source", "-s", "--output", "-o", "--preset", "-p",
        "--resolution", "-r", "--codec", "-c", "--model", "-m",
        "--dry-run", "--export-diagnostics", "--list-presets",
        "--manifest", "--retry-failed"
    }
    if any(arg in sys.argv for arg in cli_triggers):
        if "--cli" in sys.argv:
            sys.argv.remove("--cli")
        from src.cli import run_cli
        return run_cli()
    else:
        return _run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
