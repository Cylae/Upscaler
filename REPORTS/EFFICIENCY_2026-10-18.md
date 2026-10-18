# Efficiency and quality verification — 2026-10-18

## Scope and project model

Upscaler is a Windows Python/PyQt desktop application and CLI. FFprobe analyzes local media, FFmpeg extracts frames/audio, Vulkan ncnn models enhance frames, and FFmpeg encodes the selected target before validation and atomic promotion. The task was to reduce processing overhead without lowering selected render quality, then push the verified changes. The preceding startup fix is included in this delivery.

Strategy: **keep the Python/Qt/FFmpeg/Vulkan architecture and refactor measured hot paths**. Native tools perform the heavy decoding, inference and encoding work. A Python-to-Rust/C++ or UI-framework rewrite would not remove the observed redundant probes or queue scans and would add migration risk without measured benefit. No dependency, public job option, model, scale, TTA, denoise, codec quality setting, pixel format or audio selection was reduced. No production subsystem was removed.

## Baseline and changes

The performance reference is Git revision `2c03d35`. The development test baseline, including the preceding startup regressions, passed 226 tests plus one subtest in 10.84 seconds. Timing differences between test runs are not used as a performance claim.

| Component | Confirmed cost or defect | Change |
|---|---|---|
| Batch | Every progress event scans the entire selected queue; all worker futures are queued immediately | Sum active-worker progress and keep at most 1–4 futures submitted; completed jobs refill slots; skip validated rows without futures |
| File scan | Duplicate filename generation and unnecessary absent-output validation | Compute relative output once, reuse its basename, avoid probing absent destinations |
| FFprobe | Separate video/audio/tag subprocesses; unbounded caches and weak file identity | Share a bounded complete snapshot, coalesce concurrent reads, invalidate replacements/changed files, isolate mutable return data |
| PNG extraction | CPU spent compressing disposable frames | Use compression level 1 with unchanged decoded pixels; larger temporary storage is an explicit consequence |
| Frame inventory | A separate file-size lookup per output | Reuse directory-entry metadata and reject missing/misnumbered sequences, including equal-count substitutions |
| UI | Quadratic queue lookup/filtering; repeated full statistics scans and style updates | Stable-ID indexes, incremental counts, row-local filtering, paint suppression while populating |
| Logs | Shared list appended twice while viewer is open; repeated style work | Store each line once, reuse dialog, bound displayed documents, preserve escaping/filter/export behavior |
| Cancellation | Stop can occur between child creation and registration; job entry erases a prior stop | Protect spawn/registration with the same lock; retain stop requests; join extraction threads before workspace cleanup |
| Startup | Bare global Python lacks installed GUI packages; batch echo parentheses break parsing | Reuse project `.venv`, give actionable setup errors, check Qt/Pillow, repair batch syntax |

Elapsed-time and progress calculations use monotonic clocks. Progress reports are bounded to four intermediate updates per second per inference job, with completion always delivered. No inference frame is skipped.

## Reproducible measurements

Run from the project environment:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_efficiency.py --baseline-ref 2c03d35 --rounds 3 --render
```

Measurements are medians of three sequential runs on this Windows machine (32 logical CPU cores, RTX 3070, Python 3.14.7, FFmpeg 9.0.1). Inputs and desktop settings are synthetic/temporary. Raw samples are in [performance_2026-10-18.json](performance_2026-10-18.json).

| Workload | Before | After | Measured factor |
|---|---:|---:|---:|
| 10,000-item batch, 500 synthetic progress events, including queue setup | 271.79 ms | 24.25 ms | 11.2× |
| Inventory 10,000 nonempty frame files | 166.30 ms | 19.80 ms | 8.4× |
| Filter 1,000 UI rows, 10 passes | 130.17 ms | 19.07 ms | 6.8× |
| 200 item updates with 1,000 UI rows | 246.82 ms | 2.99 ms | 82.6× |
| 500 visible log events | 67.28 ms | 58.84 ms | 1.14×; records corrected from 1,000 to 500 |
| Cold video + audio + tag inspections of one media file | 130.43 ms | 59.84 ms | 2.18×; subprocesses 3 → 1 |
| Extract 24 detailed synthetic 1080p PNG frames | 344.01 ms | 277.06 ms | 1.24× |

These are component workloads, **not full-episode throughput promises**. A flat 720p animation sample showed no meaningful PNG speed improvement in exploratory measurement. The detailed PNG sample grew from 92,791,901 to 132,195,673 temporary bytes (42.5%); the 24 decoded frames matched exactly. PNG compression/prediction options are documented by [FFmpeg](https://ffmpeg.org/ffmpeg-codecs.html#png).

An additional complete render used the installed Real-ESRGAN AnimeVideo V3 engine, CPU decoding, unchanged fixed tiles/threads, lossless PNG intermediates and libx264 at the existing quality settings. It produced a synthetic one-second 720p output. Before/after decoded hashes matched for all **4 video frames and 47 audio blocks**. Observed runtimes were 1.159 and 1.103 seconds; a single tiny run is insufficient to infer a general rendering speedup. Independent extraction tests also match decoded frame hashes at 8-bit and 10-bit source depth.

## Audits, failure tests and maintainability

Phase II rechecked cache freshness, UI identity after row sorting/removal, shared log ownership, stream/tag availability and unchanged job configuration. The new tests cover equal-size file replacement with preserved timestamps, submicrosecond timestamp changes, differing FFprobe executables, cache eviction/clearing, concurrent reads, errors and files changing during inspection. Cache failure never becomes a successful audit.

Phase III reproduced the cancellation race using a blocked process-spawn factory and a simultaneous stop. Both process implementations now register the child before cancellation takes its snapshot; regressions verify the child is killed and later spawns are rejected. An already cancelled job now performs no media probing. The existing tests continue to cover leases, atomic replacement, audio repair, encoder failures and privacy handling. Missing or misnumbered output frames now fail even when the total file count matches.

Review retained the existing facade APIs and avoided a framework migration. Non-obvious ownership, cache validity, scheduler bounds and cancellation reuse rules are documented in `ARCHITECTURE.md`; repeatable measurements live in one benchmark script. No linter suppressions, skipped failures, deleted tests or weakened quality assertions were used. The two pre-existing mocked audio tests were adapted to the new inventory boundary; their audio-preservation assertions remain intact.

## Verification and requirement traceability

| Requirement / area | Verification and evidence | Result / confidence |
|---|---|---|
| Preserve selected render parameters | Batch job-capture regressions compare all selected quality options; encoder/preset/geometry suites | Proven by tests |
| Preserve rendered pixels/audio for changed I/O | Real Vulkan pipeline differential hashes; PNG 8-/10-bit differential tests; raw benchmark JSON | Supported by differential validation for these fixtures |
| Faster orchestration and indexing | Repeatable before/after component benchmark above | Supported by measurement |
| Complete output and safe failure | Frame-sequence, audio repair, dimensions, existing atomic-storage and cancellation tests | Proven by tests for covered cases |
| GUI correctness | Status/filter/sort/stale-signal/log regression tests and isolated offscreen startup | Proven by tests |
| Startup and Windows launcher | Startup dependency/argv/exit-code tests; actual `launch.bat --list-presets`; bare-Python fallback GUI smoke | Proven by runtime verification |
| Clean installation | Fresh temporary source copy, empty venv, install from `requirements-dev.txt`, full suite and offscreen GUI | [Clean-environment evidence](clean_environment_2026-10-18.json) |
| Python dependencies | `pip check`; pip-audit of development and fresh environments | No broken requirements or known package vulnerabilities in these audits; [installed](dependency_audit_2026-10-18.json), [fresh](clean_dependency_audit_2026-10-18.json) |
| Syntax and critical static checks | `python -m compileall -q main.py src benchmarks tests`; `uvx ruff check main.py src benchmarks tests --select E9,F63,F7,F82` | Passed; full style/type/security analysis not claimed |
| Repository text secret patterns | Private-key headers and common token signatures; values never printed | [Pattern scan](secret_scan_2026-10-18.json); not a comprehensive secret audit |
| Documentation | README setup/testing/benchmark commands and architecture ownership/limits synchronized | Supported by review |

The fresh environment uses PyQt6 6.11.0 / Qt 6.11.2, Pillow 13.0.0 and pytest 9.1.1. It passed **269 tests plus one subtest**; its offscreen GUI opened and exited normally. The development environment additionally exercised local Vulkan inference; portable model binaries are intentionally ignored by Git and were not copied into the fresh environment.

## Remaining risks and limits

- GPU inference, encoder speed, VRAM, source complexity and storage still dominate long jobs. Models, hardware concurrency defaults and encoder quality presets were not weakened to inflate speed measurements. No general 4K/8K throughput multiplier or universal visual equivalence is claimed.
- Faster lossless PNG compression uses more temporary storage on some inputs. Preflight remains explicitly an estimate of **final output only**, not a guaranteed temporary-space budget. JPEG and final lossy encoding retain their existing characteristics. PNG is needed for lossless intermediate images.
- Sources/outputs and filenames remain untrusted boundaries. Argument-list subprocess execution, escaped log HTML, validation and atomic output promotion were retained. Cache freshness uses filesystem identity/timestamps; metadata inspection is not a full bitstream decode or cryptographic integrity audit.
- pip-audit covers reported Python-package vulnerabilities, not a complete audit of bundled Qt, FFmpeg, GPU drivers or portable ncnn executables. No native-engine security audit or GPU driver upgrade was performed.
- Requirements retain their existing lower bounds instead of introducing an unverified universal lock. Exact tested versions are recorded. Only this Windows host was exercised; other operating systems and hardware remain unverified here.
- Manifest serialization and single-frame preview process management retain their existing implementations. They were not established as material performance bottlenecks in this task. A broader persistence or native-engine redesign would require separate evidence.

No new material actionable finding remained in the final review of the changed paths after these remediations. This statement does not assert software perfection or exhaustive coverage.
