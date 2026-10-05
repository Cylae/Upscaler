# Reference: Architecture Evaluation & Strategy Selection

## Overview
Architecture evaluation determines whether the existing design can reliably support the project's requirements, or whether restructuring is needed. The agent must choose and formally justify one of five core change strategies.

---

## The 5 Change Strategies

### 1. KEEP
- **When to Choose**: The architecture is sound, modular, well-tested, and meets all performance, security, and maintainability requirements. Bugs are localized to individual function or module logic.
- **Action**: Preserve existing structure; apply surgical defect fixes.

### 2. REFACTOR
- **When to Choose**: The overall design is viable, but code quality, internal abstractions, naming consistency, module boundaries, or testability are degraded.
- **Action**: Restructure internal implementation without altering public interfaces or external behavior.

### 3. PARTIALLY REPLACE
- **When to Choose**: Specific components, subsystems, or dependencies are obsolete, insecure, unmaintainable, or fundamentally broken, while the surrounding system is healthy.
- **Action**: Isolate the broken component behind an interface, build a replacement, and swap it cleanly with full test verification.

### 4. INCREMENTALLY MIGRATE
- **When to Choose**: The system requires a major technology, framework, or paradigm shift, but a clean-slate rewrite carries excessive risk or breaks too many active integrations.
- **Action**: Establish an abstraction bridge, migrate modules incrementally, and verify compatibility at each step.

### 5. FULLY REWRITE
- **When to Choose**: The codebase suffers from fundamental design failure, pervasive security vulnerability, extreme debt, or structural unmaintainability where incremental fixes cost more than rebuilding.
- **Action**: Reconstruct the system clean-slate within the workspace while strictly preserving public contracts, behavioral requirements, and data guarantees.

---

## Evaluation Criteria Matrix

| Criterion | Keep | Refactor | Partial Replace | Incremental Migrate | Full Rewrite |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Architecture** | Sound | Sound | Mostly Sound | Mixed | Unsalvageable |
| **Defect Scope** | Isolated | Localized | Subsystem | Systemic | Pervasive |
| **Testability** | High | Low | Moderate | Moderate | Non-existent |
| **Contract Stability**| Unchanged | Unchanged | Unchanged | Preserved | Preserved |
| **Risk / Effort Ratio**| Lowest | Low | Medium | High | High (Justified) |

---

## Strategy Justification Requirement
Document the selected strategy in `REPORTS/FINAL_DELIVERY_REPORT.md` with:
1. Technical rationale based on baseline measurements.
2. Trade-off analysis.
3. Mitigation plan for preservation of behavioral invariants and consumer contracts.
