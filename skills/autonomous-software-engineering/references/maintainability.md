# Reference: Long-Term Maintainability & Documentation

## Overview
A project is complete only when an unfamiliar, competent developer can understand, build, test, debug, extend, deploy, and maintain the repository without needing access to the original author or conversation history.

---

## Code Quality & Architecture Rules

### 1. Separation of Concerns
- Keep business logic, hardware detection, pipeline execution, storage I/O, CLI parsing, and UI presentation cleanly decoupled into distinct modules.
- Avoid global mutable state; pass configuration explicitly.

### 2. Error Handling & Diagnostics
- Use explicit, descriptive exception types.
- Provide actionable error messages that explain what failed, why, and how to resolve it.
- Ensure logging is informative, contextual, and clean.

### 3. Cross-Platform Compatibility
- Standardize path manipulation using standard library functions (`os.path.join`, `os.path.relpath`, `os.path.basename`) and normalize path separators (`\` vs `/`).
- Avoid platform-specific shell commands or unhandled platform dependencies.

### 4. Dependency Hygiene
- Keep dependencies minimal, justified, and pinned in environment files (`requirements.txt`, `package.json`, etc.).
- Avoid introducing unnecessary heavy frameworks or mandatory toolchains for small projects.

---

## Documentation Deliverables

### 1. Operational Documentation (`README.md`)
- Clear description of purpose, features, system prerequisites, and quick-start guides (CLI and GUI).
- Comprehensive options reference table covering all arguments and presets.
- System diagnostic instructions (`--doctor` / `--export-diagnostics`).

### 2. Architectural Specification (`ARCHITECTURE.md`)
- Detailed pipeline diagrams, module responsibilities, concurrency model, and data flow.
- Safety guarantees (atomic replacement, active job locks, privacy scrubbing).

### 3. Final Delivery Report (`REPORTS/FINAL_DELIVERY_REPORT.md`)
- Comprehensive audit report detailing what changed, why, verification evidence, and remaining limitations.
