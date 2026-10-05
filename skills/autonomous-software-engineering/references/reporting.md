# Reference: Reporting Standards & Audit Structures

## Overview
Reporting communicates the technical trajectory, baseline metrics, strategy justification, defect remediation, audit findings, and verification matrix to project maintainers. Reporting should be scaled to project complexity—comprehensive for major engineering tasks, concise for small projects.

---

## Final Delivery Report Structure

For substantial engagements, the report (`REPORTS/FINAL_DELIVERY_REPORT.md`) must contain:

1. **Executive Summary**: Core objective, strategy executed, and overall outcome.
2. **Project Purpose & Invariants**: Scope, target contracts, public APIs, and behavioral invariants.
3. **Initial Baseline vs. Final Metrics**: Measured before/after comparisons for test pass rates, build status, performance, and code quality.
4. **Strategy & Architectural Decisions**: Justification for chosen strategy (Keep, Refactor, Partial Replace, Incremental Migrate, Full Rewrite) and trade-off analysis.
5. **Defect Investigation & Remediation**: Root-cause analysis for each identified bug, showing original defect, root cause, code fix, and regression test.
6. **Independent Quality & Adversarial Audit**: Findings from quality code reviews, security checks, and failure injection audits.
7. **Requirement ➔ Implementation ➔ Verification ➔ Evidence Matrix**: Complete traceability matrix linking each requirement to code changes, tests, and actual command execution output.
8. **Remaining Risks & Limitations**: Explicit disclosure of unperformed checks (e.g. hardware-dependent Vulkan GPU tests on headless servers), environment limitations, or future maintainer recommendations.
