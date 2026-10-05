# Reference: Testing & Verification Standards

## Overview
Verification must be automated, multi-layered, and evidence-driven. Passing tests alone do not establish completion—tests must meaningfully assert the system's behavioral invariants and contract specifications.

---

## The Non-Weakening Rule
Never achieve a passing result by:
- Weakening assertions or widening tolerances without explicit requirement justification.
- Deleting or commenting out failing tests.
- Mocking away the core logic or behavior being verified.
- Suppressing, ignoring, or silently swallowing errors.
- Changing test assertions to match incorrect or buggy implementation behavior.

If a test is flawed (e.g. asserts obsolete or incorrect requirements), document the defect, update the test to reflect authoritative specifications, and explain the change in the audit report.

---

## Multi-Layered Verification Hierarchy

### 1. Unit Tests
- Verify individual functions, classes, and isolated pure logic.
- Fast execution, zero external side effects, high code coverage.

### 2. Integration Tests
- Verify interactions between modules, storage layers, subprocess wrappers, and file systems.
- Test actual file I/O, serialization, and pipeline orchestration.

### 3. Contract & Boundary Tests
- Verify compliance with public API schemas, CLI flag specifications, and external data formats.
- Test boundary conditions (e.g., zero-length files, maximum values, unusual aspect ratios, Unicode strings).

### 4. Negative & Hostile Input Tests
- Exercise malformed inputs, truncated files, invalid paths, missing environment variables, and permission errors.
- Ensure graceful error handling and informative user-facing diagnostics.

### 5. Regression & Bug-Fix Tests
- Every fix for a confirmed defect must be accompanied by a regression test reproducing the original issue and verifying the fix.

---

## Verification Evidence Recording
Capture actual execution commands and verbatim terminal output for inclusion in the final delivery matrix:
```
Requirement ➔ Implementation ➔ Verification Test ➔ Recorded Evidence
```
