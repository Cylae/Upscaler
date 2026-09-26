import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
        from src.ui.app_qt import run_gui
        return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
