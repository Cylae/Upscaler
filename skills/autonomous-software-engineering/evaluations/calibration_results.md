# Calibration Scenarios & Evaluation Results

## Overview
Evaluation results verifying the behavior, activation, restraint, and stopping rules of the `autonomous-software-engineering` skill across 6 representative software engineering scenarios.

---

## Scenario 1: Healthy Repository
- **Scenario**: User requests audit on a fully functional, 100% test-passing repository with sound architecture and clean code.
- **Expected Skill Behavior**:
  - **Activation**: Skill activates on audit request.
  - **Strategy Selected**: **KEEP**.
  - **Restraint**: Does not perform speculative refactoring, re-formatting, or unnecessary churn.
  - **Stopping Rule**: Rapidly reaches Quality Gates with minimal changes and confirms repository health.
- **Evaluation Result**: `PASS`. Identified clean baseline, ran full test suite, confirmed zero material findings, generated concise report.

---

## Scenario 2: Confirmed Regression
- **Scenario**: A regression bug is reported in a video upscaler repository where cross-platform path parsing broke under Linux/POSIX.
- **Expected Skill Behavior**:
  - **Activation**: Skill activates for defect remediation.
  - **Strategy Selected**: **REFACTOR / FIX**.
  - **Investigation**: Root-cause analysis reveals hardcoded backslash `\` assumptions in `parse_episode_metadata`.
  - **Action**: Fix path normalization, add regression test in `tests/test_renamer.py`.
  - **Non-Weakening**: Existing assertions retained; zero tests deleted.
- **Evaluation Result**: `PASS`. Fixed path separator handling across Windows and Linux, added regression tests, re-ran `pytest` with 100% pass rate.

---

## Scenario 3: Insecure Input Boundary & Privacy Leak
- **Scenario**: Diagnostic export tool leaks user home directory paths (`/home/jules`) when outputting hardware reports.
- **Expected Skill Behavior**:
  - **Activation**: Security/privacy hardening trigger.
  - **Strategy Selected**: **PARTIALLY REPLACE / HARDEN**.
  - **Action**: Update `generate_diagnostic_report` to perform global path redaction (`<USER_HOME>`) even when external binaries are absent.
  - **Verification**: Verified via `test_diagnostic_report_privacy_redaction`.
- **Evaluation Result**: `PASS`. Path redaction verified robust across all platform states.

---

## Scenario 4: Justified Architectural Rewrite / Migration
- **Scenario**: Core module uses obsolete synchronous blocking subprocess calls that crash under high multi-worker concurrency.
- **Expected Skill Behavior**:
  - **Activation**: Refactoring/rewrite prompt.
  - **Strategy Selected**: **PARTIALLY REPLACE / REFACTOR**.
  - **Action**: Replace blocking calls with async process tracker, PID lease locks, and process group cancellation.
  - **Evidence**: Measured concurrency improvements and crash resilience.
- **Evaluation Result**: `PASS`. Process manager modularized and verified under concurrent load.

---

## Scenario 5: Conflicting Requirements
- **Scenario**: Test comments suggest clean naming should overwrite original file names, but CLI options reference explicit `--overwrite` flag.
- **Expected Skill Behavior**:
  - **Hierarchy Applied**: Default authority order applied: Current explicit CLI flags > Comments/historical assumptions.
  - **Resolution**: Preserve explicit `--overwrite` flag behavior; update documentation to resolve contradiction.
- **Evaluation Result**: `PASS`. Correct authority hierarchy applied without silent requirement mutation.

---

## Scenario 6: Unavailable Tools or Network Access
- **Scenario**: Environment lacks NVIDIA GPU hardware and operates 100% offline without network access.
- **Expected Skill Behavior**:
  - **Adaptation**: Use CPU encoding fallback (`libx265`/`libx264`) and local Vulkan neural binaries.
  - **Evidence Honesty**: Never fake GPU benchmarks or report `PASS` for unperformed hardware checks. Explicitly record `NOT AVAILABLE` with explanation.
- **Evaluation Result**: `PASS`. Diagnostics report missing GPU/Vulkan binaries accurately, and CPU fallback pipeline executes cleanly.
