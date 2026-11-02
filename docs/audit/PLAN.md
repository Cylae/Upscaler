# Apollo-Standard Engineering Execution Plan — Upscaler

## 1. Mission Context
- **Target Repository**: `c:\Users\Cylae\Documents\GitHub\Upscaler`
- **Active Branch**: `main` (commit SHA `05471a0`, clean working tree)
- **Mandate**: 100% file review, full defect remediation, strict invariant enforcement, hardened falsifiable testing, bottleneck optimization, and Apollo-standard verifiable handoff in French.
- **Current Baseline**:
  - Python: 3.12.10 (x64)
  - Pytest: 379 passed in 98.45s (77% statement coverage, 1110 missed / 4820 statements)
  - Ruff: 443 linter alerts (imports unorganized, unused imports, B023 variable binding, S102, etc.)
  - Mypy: 38 static typing errors in 11 files (union attr, type mismatches, variable shadowing)

---

## 2. Milestone Structure & Dependency Ordering

### Milestone 1: Reconnaissance, Architecture Mapping & Baseline Capture
- [x] Baseline git status, branch, toolchain inspection (Python 3.12, PyQt6 6.7.0, Pillow 12.3.0, pytest 8.2.0, ruff 0.16.8, mypy 2.3.1).
- [x] Initial full test suite execution & statement coverage baseline (379 tests passing, 77% coverage).
- [x] Initial static analysis and type checking baseline (443 ruff alerts, 38 mypy errors).
- [ ] Initialize `docs/audit/` ledgers (`PLAN.md`, `REVIEW_MATRIX.md`, `FINDINGS.md`, `VALIDATION.md`, `PERFORMANCE.md`, `HANDOFF.md`).
- [ ] Dispatch parallel reconnaissance sub-agents across all 4 repository partitions to achieve 100% full file reading and invariant extraction.
- [ ] Populate `REVIEW_MATRIX.md` with every first-party file, line counts, hazard tags, and verified contracts.

### Milestone 2: Critical Security, Identity, Data-Corruption & State-Safety Remediation
- [ ] Audit and harden input validation, path sanitization, and shell argument safety in `src/cli.py`, `src/core/storage.py`, `src/core/process.py`, `src/core/video_pipeline.py`.
- [ ] Eliminate formula injection vectors or unescaped paths in CSV/logs/export diagnostics.
- [ ] Verify atomic file write, temporary file cleanup, PID lock leasing, and crash-resilience in `src/core/storage.py`.
- [ ] Fix variable shadowing in `src/core/batch.py` (`parts` string/list scope collision).
- [ ] Hardened Windows process tree termination (`taskkill`) and deadlock prevention on pipe reads in `src/core/process.py`.

### Milestone 3: Exhaustive Module-by-Module Defect Remediation
- [ ] Resolve all 38 mypy static typing errors across `src/core/batch.py`, `src/core/models.py`, `src/core/metadata_provider.py`, `src/core/probe.py`, `src/core/video_pipeline.py`, `src/core/pipeline.py`, `src/ui/main_window.py`, `src/ui/dialogs/*`, `src/cli.py`.
- [ ] Resolve all ruff linter issues (import sorting, unused imports, lambda/loop variable bindings, exception formatting).
- [ ] Verify and harden TVmaze metadata enrichment in `src/core/metadata_provider.py` (proper null handling, type coercion, offline fallback).
- [ ] Fix dynamic attribute bindings (`btn_stat_*`) in `src/ui/main_window.py` to provide explicit type-safe properties.
- [ ] Resolve any dead CLI flags, UI options, or contradictory defaults uncovered during spec extraction.

### Milestone 4: Falsifiable Test Suite, Action-Sink Replay, Fuzzing & Gate Hardening
- [ ] Expand unit, boundary, and property tests to address the 1110 missed statements in `src/` (raise coverage from 77% toward 90%+).
- [ ] Implement falsifiable regression tests for every finding (`F-xxx`).
- [ ] Add adversarial fuzzing / boundary tests for corrupted media files, NaN/inf dimensions, invalid rational framerates, path traversals, missing audio streams, interrupted sub-processes.
- [ ] Ensure all new tests fail when the fix is reverted (mutation proof).

### Milestone 5: Profiling, Bottleneck Optimization & Resource Soak Verification
- [ ] Profile metadata probe LRU caching, batch scheduling loop, and frame extraction/encoding pipelines.
- [ ] Verify resource soak (file descriptor, memory, child process leaks) during repetitive batch operations.
- [ ] Run benchmarks in `benchmarks/` and record before/after performance numbers in `PERFORMANCE.md`.

### Milestone 6: Packaging, Clean-Install, CI Matrix & Documentation Synchronization
- [ ] Test clean virtual environment bootstrap and CLI/GUI entry points.
- [ ] Ensure `launch.bat`, `main.py`, `README.md`, and `ARCHITECTURE.md` are in 100% synchrony with actual behavior.
- [ ] Verify all documentation claims, CLI flags, and UI controls reflect reality.

### Milestone 7: Dual Independent Red-Team Adversarial Audit & Final Delivery
- [ ] Pass 1: Fresh-context Red-Team audit searching for unhandled edge cases, race conditions, or regression risks.
- [ ] Pass 2: Clean-context diff and contract audit against baseline.
- [ ] Final handoff report delivered in French adhering to all Apollo-Standard engineering gates.

