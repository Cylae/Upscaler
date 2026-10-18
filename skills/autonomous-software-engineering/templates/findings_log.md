# Audit Findings Log

## Overview
Log of technical defects, code smells, security risks, and architecture problems discovered during discovery, testing, independent audit, and adversarial review.

---

| Finding ID | Severity | Component | Description & Root Cause | Status | Remediation Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FIND-01** | High | `scene_renamer.py` | Hardcoded backslash path parsing failed on POSIX/Linux platforms. | Resolved | Normalized path separators across all path resolution functions. |
| **FIND-02** | Medium | `models.py` | VRAM tile calculation returned 0 for lightweight models on 8GB VRAM GPUs. | Resolved | Corrected tile size calculation bounds to return 400 for 8GB VRAM. |
| **FIND-03** | Medium | `hardware.py` | Path redaction did not insert `<USER_HOME>` when binaries were missing. | Resolved | Updated diagnostic generator to perform global path substitution. |
| **FIND-04** | Low | `tests/` | Unmocked test assumptions caused test failures on headless Linux runners. | Resolved | Added graceful environment checks and robust mocks. |
