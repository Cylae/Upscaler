# Validation & Verification Ledger — Apollo Engineering Audit

## 1. Environment & Hardware Profile
- **Host OS**: Windows 11 (build info: win32)
- **Shell**: PowerShell 5.1 / pwsh
- **Python Runtime**: Python 3.12.10 (tags/v3.12.10:07e8494, 64-bit)
- **Qt Toolkit**: PyQt6 6.7.0 (Qt 6.7.3)
- **Image Processing**: Pillow 12.3.0
- **Test Harness**: pytest 8.2.0, pytest-cov 7.1.0

## 2. Baseline Test Execution
- **Command**: `python -m pytest`
- **Result**: `379 passed in 103.98s`
- **Exit Code**: 0

## 3. Baseline Coverage Breakdown
- **Command**: `python -m pytest --cov=src --cov-report=term-missing`
- **Result**:
  - `src/__init__.py`: 100% (0/0)
  - `src/cli.py`: 86% (179/209)
  - `src/core/__init__.py`: 100% (0/0)
  - `src/core/ai_models.py`: 100% (3/3)
  - `src/core/batch.py`: 89% (431/482)
  - `src/core/batch_manager.py`: 100% (4/4)
  - `src/core/detector.py`: 100% (3/3)
  - `src/core/hardware.py`: 86% (138/161)
  - `src/core/metadata_provider.py`: 34% (65/191)
  - `src/core/models.py`: 78% (199/256)
  - `src/core/pipeline.py`: 71% (5/7)
  - `src/core/presets.py`: 100% (24/24)
  - `src/core/probe.py`: 91% (220/241)
  - `src/core/process.py`: 53% (40/75)
  - `src/core/scene_namer.py`: 100% (5/5)
  - `src/core/scene_renamer.py`: 87% (167/191)
  - `src/core/storage.py`: 65% (74/114)
  - `src/core/validator.py`: 68% (56/82)
  - `src/core/video_pipeline.py`: 68% (506/745)
  - `src/ui/__init__.py`: 100% (0/0)
  - `src/ui/app_qt.py`: 92% (11/12)
  - `src/ui/dialogs/__init__.py`: 100% (6/6)
  - `src/ui/dialogs/comparison.py`: 89% (151/169)
  - `src/ui/dialogs/doctor.py`: 98% (53/54)
  - `src/ui/dialogs/logs.py`: 94% (131/139)
  - `src/ui/dialogs/preflight.py`: 98% (85/87)
  - `src/ui/main_window.py`: 74% (1030/1387)
  - `src/ui/theme.py`: 73% (61/84)
  - `src/ui/workers.py`: 71% (63/89)
  - **TOTAL**: 77% (3710 / 4820 statements covered)

## 4. Static Analysis & Type Checking Baseline
- **Ruff**: `ruff check .` -> Found 443 errors (269 fixable with `--fix`)
- **Mypy**: `mypy src --ignore-missing-imports` -> Found 38 errors across 11 files

