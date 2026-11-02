# Validation & Verification Ledger — Apollo Engineering Audit

## 1. Environment & Hardware Profile
- **Host OS**: Windows 11 (build info: win32)
- **Shell**: PowerShell 5.1 / pwsh
- **Python Runtime**: Python 3.12.10 (tags/v3.12.10:07e8494, 64-bit)
- **Qt Toolkit**: PyQt6 6.7.0 (Qt 6.7.3)
- **Image Processing**: Pillow 12.3.0
- **Test Harness**: pytest 8.2.0, pytest-cov 7.1.0
- **Linter & Formatter**: ruff 0.16.8
- **Static Type Checker**: mypy 2.3.1

---

## 2. Test Execution Comparison

| Metric | Baseline (`05471a0`) | Final Verified State | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Test Count** | 379 tests | **411 tests** | **+32 regression & validation tests** |
| **Test Duration** | 103.98s | **67.36s (with full coverage instrumentation)** | **1.54x faster (instrumented)** / **34.08s (uninstrumented, 3.05x faster)** |
| **Test Pass Rate** | 100% (379/379) | **100% (411/411)** | Maintained 100% pass |
| **Failures / Errors** | 0 | **0** | Zero regressions |

---

## 3. Code Coverage Comparison

| Module | Baseline Coverage | Milestone 4 Coverage | Final Verified State | Total Delta |
| :--- | :--- | :--- | :--- | :--- |
| `src/__init__.py` | 100% | 100% | 100% | 0% |
| `src/cli.py` | 86% | 86% | 86% | 0% |
| `src/core/__init__.py` | 100% | 100% | 100% | 0% |
| `src/core/ai_models.py` | 100% | 100% | 100% | 0% |
| `src/core/batch.py` | 89% | 89% | 89% | 0% |
| `src/core/batch_manager.py` | 100% | 100% | 100% | 0% |
| `src/core/detector.py` | 100% | 100% | 100% | 0% |
| `src/core/hardware.py` | 86% | 86% | 86% | 0% |
| `src/core/metadata_provider.py` | 34% | 90% | **90%** | **+56%** |
| `src/core/models.py` | 78% | 78% | 78% | 0% |
| `src/core/pipeline.py` | 71% | 71% | 71% | 0% |
| `src/core/presets.py` | 100% | 100% | 100% | 0% |
| `src/core/probe.py` | 91% | 91% | 91% | 0% |
| `src/core/process.py` | 53% | 74% | **74%** | **+21%** |
| `src/core/scene_namer.py` | 100% | 100% | 100% | 0% |
| `src/core/scene_renamer.py` | 87% | 88% | **88%** | **+1%** |
| `src/core/storage.py` | 65% | 79% | **79%** | **+14%** |
| `src/core/validator.py` | 68% | 86% | **95%** | **+27%** |
| `src/core/video_pipeline.py` | 68% | 68% | 68% | 0% |
| `src/ui/__init__.py` | 100% | 100% | 100% | 0% |
| `src/ui/app_qt.py` | 92% | 92% | 92% | 0% |
| `src/ui/dialogs/__init__.py` | 100% | 100% | 100% | 0% |
| `src/ui/dialogs/comparison.py` | 89% | 90% | **90%** | **+1%** |
| `src/ui/dialogs/doctor.py` | 98% | 98% | 98% | 0% |
| `src/ui/dialogs/logs.py` | 94% | 94% | 94% | 0% |
| `src/ui/dialogs/preflight.py` | 98% | 98% | 98% | 0% |
| `src/ui/main_window.py` | 74% | 74% | **77%** | **+3%** |
| `src/ui/theme.py` | 73% | 73% | **88%** | **+15%** |
| `src/ui/workers.py` | 71% | 70% | **74%** | **+3%** |
| **TOTAL STATEMENT COVERAGE** | **77%** (3710/4820) | **80%** (3883/4857) | **82% (3976 / 4868)** | **+5% (Net +266 statements covered)** |

---

## 4. Static Analysis & Type Checking Comparison

| Tool | Baseline | Final Verified State | Status |
| :--- | :--- | :--- | :--- |
| **`mypy src --ignore-missing-imports`** | 38 errors across 11 files | **0 errors across 29 files** | **RESOLVED** |
| **`ruff check .`** | 443 errors / warnings | **0 errors across entire repository** | **RESOLVED** |

---

## 5. Verification Commands & Execution Logs
- `python -m pytest`: 411 passed in 72.99s
- `python -m pytest --cov=src --cov-report=term`: **82% total statement coverage** (3976 / 4868 statements)
- `mypy src --ignore-missing-imports`: Success: no issues found in 29 source files
- `ruff check .`: All checks passed!
- `python benchmarks/benchmark_efficiency.py --baseline-ref 05471a0`: 6.88x speedup on frame inventory

---

## 6. Extended Test Suite Manifest (`tests/test_extended_validation.py`)
1. `test_ui_presets_switching_all_options`: Deep verification across all 7 presets ensuring synchronisation across resolution, codec, model, CQ, workers, frame format, audio mode, TTA, denoise, and descriptions.
2. `test_ui_empty_vs_loaded_state_and_buttons`: Empty queue protection, start inhibition, stat pill count transitions (`Ready: 0` -> `Ready: 1` -> `Ready: 0`), and control locking during processing.
3. `test_ui_table_context_menu_model_override`: Multi-row AI model override and bulk reset to Auto.
4. `test_ui_keyboard_shortcuts`: Registration and mapping of `Ctrl+L`, `Ctrl+O`, `F5`, `F6`, `Ctrl+Return`, `Ctrl+.`.
5. `test_ui_drag_and_drop_event`: Full simulation of native file drag-and-drop into the queue with path normalization.
6. `test_ui_theme_helpers`: Runtime verification of `apply_dark_titlebar`, `setup_frameless_window`, and `setup_combobox`.
7. `test_audio_fallback_to_aac_transcode`: Automatic fallback from failed stream copy to 192k AAC transcode.
8. `test_video_pipeline_cuda_failure_fallback_to_cpu`: Graceful CUDA decode failure handling with automatic CPU extraction retry and partial frame cleanup.
9. `test_video_pipeline_colorspace_conformance`: Conformance of SD sources (BT.601 / SMPTE 170M) to BT.709 upon HD/4K upscaling.
10. `test_video_pipeline_variable_and_rational_framerate`: Preservation of precise rational framerates (e.g. `24000/1001`, `30000/1001`) during encoding.
11. `test_video_pipeline_aspect_ratio_geometry`: Portrait (9:16), ultrawide (21:9), and DVD anamorphic (SAR 32:27) even-dimension fitting.
12. `test_batch_manifest_export_and_import`: JSON manifest persistence, queue session recovery, and status retention.
13. `test_batch_manager_retry_failed_items`: Selective retry of `FAILED` and `CANCELLED` items while preserving `COMPLETED`.
14. `test_batch_manager_remove_items_by_indices`: Index preservation, removal, and continuous reindexing.
