# AI engine discovery follow-up — 2026-10-18

The reported frame-comparison dialog showed `Real-ESRGAN engine not found: None`. Both portable engines are present on this host and a fresh `py main.py --doctor` detects them. The GUI retains its initial `HardwareInfo` object for the session, while preview and video jobs previously trusted that snapshot without rechecking a missing executable. An initially negative detection therefore remained negative after installation. Refreshing the global cache also leaves already-retained objects unchanged.

This failure mechanism reproduces the exact reported error against revision `a5d0aff`. The screenshot alone does not establish why the original session's initial detection returned no path.

## Correction

- Resolve the selected engine before frame extraction or encoder checks. Recheck local portable tools and PATH only when its cached path is absent or stale.
- Update the selected path on the shared hardware object. Preserve valid caller-provided paths, the other engine's configuration, selected model and inference/encoding options.
- If the selected engine is still missing, give its expected executable path and installation guidance before creating a processing workspace or running extraction. The normal output directory may already have been created by video naming/resume checks.
- Keep GPU and FFmpeg capability detection cached; recovery performs file checks and executable lookup, without expensive capability subprocesses.

## Verification

- **379 tests plus one subtest passed in 62.30 seconds** in the project environment. This includes 32 isolated discovery cases and eight new preview/video integration cases. Discovery covers both engines as strings and `ModelEngine` values, missing/stale/local/PATH/custom paths, a retry after installation in the same session, and replacement after an executable disappears.
- Independent review found a formatting defect with `str` enums in the missing-engine diagnostic; normalizing the engine name fixed it before final verification. The independent reviewer then passed 102 related tests and found no further blocking issue in this change.
- The real Vulkan check reproduced the old error with a missing cached path, then rendered four recovered previews (two engines, missing and stale paths). Each recovered preview's decoded RGB pixels matched the old implementation with a valid path exactly, at 960×720.
- Complete synthetic video encodes for both engines also recovered from missing cached paths. Each matched its valid-path baseline across all four decoded video frames and 47 decoded audio blocks. All completed operations left zero tracked child processes. [Raw render results](engine_recovery_2026-10-18.json).
- Critical Ruff checks (`E9,F63,F7,F82`), byte compilation, dependency consistency and Git whitespace checks passed. No production dependency or render-quality setting changed.
- A temporary source copy and empty virtual environment also passed **379 tests plus one subtest in 44.30 seconds**, with no broken dependencies. It installed the documented development requirements, excluded ignored portable engines and user media, and used the available system FFmpeg/FFprobe. [Clean-environment results](engine_recovery_clean_environment_2026-10-18.json).

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_engine_discovery.py tests/test_frame_preview_settings.py
.\.venv\Scripts\python.exe benchmarks/verify_engine_recovery.py --baseline a5d0aff --output REPORTS/engine_recovery_2026-10-18.json
```

The last command requires the installed portable engines/models, FFmpeg/FFprobe and a Vulkan GPU; it uses and deletes only generated temporary media. These runtime checks used this Windows RTX 3070 host and small synthetic inputs. They establish discovery recovery and output equivalence for those cases, without claiming equivalent validation on other hardware or on every possible input video.
