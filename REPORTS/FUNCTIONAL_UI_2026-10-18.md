# Functional-interface follow-up — 2026-10-18

The user prioritized a plain, usable processing interface over decoration. The implementation replaces the long sidebar and collapsed expert panel with four direct parameter tabs, keeps paths/actions/logs visible, restores native window controls, and removes the duplicate resize handler that resized the status column instead of the detail column. No render-quality default was reduced.

## Functional changes and evidence

| Requirement / defect | Implementation | Verification |
|---|---|---|
| Direct parameter access | Video, Audio, Engine, Naming tabs; scrollable forms; numeric CQ/CRF entry synchronized with slider | Compact-window reachability tests and visual inspection at 1024×768 and 1400×900 with Windows Segoe UI loaded for offscreen rendering |
| Complete supported choices | CPU H.264/H.265, NVENC p1–p7, 1–4 workers, editable positive thread counts and tile sizes, MP4/MKV and suffix | Data-based choice/worker/preflight/config regressions |
| Truthful profiles | Every manual processing change selects Custom; profile application uses item data | Manual-edit tests; fast_draft now actually selects its declared p3 instead of silently retaining p7 |
| Preview uses selected video and inference settings | Frame, clip and full-video test use selected row; frame comparison honors per-item model, threads, tile, denoise and TTA | Worker/engine-command regressions for both engines; real Vulkan checks below |
| Global model remains effective after a run | Store last-run model information separately from explicit per-file overrides, including manifest round trips | Two successive jobs with different global models and an explicit-override counterpart |
| Current destinations and resume state | Recalculate output mappings before preflight/start; validate geometry, audio and duration; preserve queue identity/overrides | Replacement destination, resolution, portrait/SAR geometry, overwrite, invalid-output and running-job cases |
| Preview existing full outputs | Clip requests bypass the full-output SKIPPED shortcut | Single-/multi-worker clip regressions |
| Prevent accidental edits | Unfocused wheel events do not change settings; invalid manual entries display an error before worker creation | Wheel and invalid-input tests |
| Safe active-job UI | Lock parameter changes and queue mutations while leaving tabs/logs/filter inspectable; Stop also cancels frame-preview processes | Busy-state, queue-removal, stop, preview failure/cleanup tests |
| Usable small-window table | Resizable filename columns with readable minima and horizontal scroll | Layout regression at 1920×1080; visual inspection at 1024×768 |

The default cartoon profile retains its previous model selection, CQ, p7, audio mode, frame format and concurrency. Choosing fast_draft now honors its existing declared p3; this is an explicit profile selection, not an automatic quality reduction. A frame preview is a PNG inference comparison; use Preview Clip to also inspect final lossy encoding and audio.

## Runtime and audit results

- The full development suite passed **337 tests plus one subtest** before the final review changes. The final fresh environment passed **339 tests plus one subtest in 57.50 seconds**, and the four-tab native-frame GUI smoke test passed. Exact versions and output are recorded in the clean-environment evidence below.
- Two real one-frame previews were generated from a synthetic 64×48 FFV1 source using installed Real-ESRGAN AnimeVideo V3 and RealCUGAN. Both used explicit `2:2:2` threads, tile 100, denoise 0 and enabled TTA, produced the expected 960×720 geometry, and left zero tracked processes. [Raw results](functional_preview_2026-10-18.json).
- Critical Ruff checks (`E9,F63,F7,F82`), Python byte compilation and Git whitespace checks passed. No production dependency was added.
- The original layout tests were updated to assert native minimize/maximize/close flags and resizable filename columns instead of custom titlebar buttons/stretch-only columns. Render-quality and worker-option assertions were retained.
- Review found and fixed missing p3, CPU H.264/worker choices, stale profile labels, stale output mappings, preview-option omissions, mis-targeted test/clip buttons, skipped clip previews, progress information becoming an unintended per-file model override, and a preview process-tracking gap. The initial overflowing combo size hint was corrected and rechecked.

## Reproduction and limits

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_functional.py tests/test_frame_preview_settings.py tests/test_queue_replan.py
```

See [functional_clean_environment_2026-10-18.json](functional_clean_environment_2026-10-18.json) for the final fresh-install test result. The clean environment excludes ignored portable Vulkan binaries; installed-engine previews were checked separately on this Windows RTX 3070 host. Other operating systems/hardware were not exercised here.

Metadata validation can establish geometry, duration and audio presence, not the provenance or subjective quality of an existing render. Use Reprocess existing outputs to deliberately replace a valid same-name output after a model/quality change. Installed RealCUGAN weight/scale compatibility resolution remains in effect. Preflight revalidation runs before launch, not on every text edit, to avoid repeated probing while typing.
