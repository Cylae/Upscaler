import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if "--doctor" in sys.argv:
        from src.core.detector import print_doctor_report
        print_doctor_report()
        return

    if "--help" in sys.argv or "-h" in sys.argv:
        from src.cli import run_cli
        run_cli()
        return

    if "--cli" in sys.argv:
        if "--cli" in sys.argv:
            sys.argv.remove("--cli")
        from src.cli import run_cli
        return run_cli()
    else:
        from src.ui.app_qt import run_gui
        return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
