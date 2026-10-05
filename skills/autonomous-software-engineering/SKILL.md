# Skill: autonomous-software-engineering

## Purpose
The `autonomous-software-engineering` skill turns the **Universal Autonomous Software Engineering Directive** into an executable, multi-phase software engineering workflow. It enables an AI agent to take full technical ownership of a software repository within an authorized workspace—guiding it through discovery, architecture evaluation, defect remediation, multi-layered verification, independent and adversarial audits, hardening, and maintainer handoff.

The ultimate objective is **evidence-driven quality convergence**: reaching a defensibly correct, secure, reliable, performant, and long-term maintainable state backed by reproducible evidence.

---

## Trigger Conditions & Intent Recognition

### When to Activate
Activate this skill when a user requests substantial, end-to-end software engineering work on a repository or project. Representative triggers include:
- "Analyze this entire repository and fix the problems."
- "Audit, test, harden, and improve this project autonomously."
- "Refactor or rewrite the architecture if the evidence justifies it."
- "Investigate the root cause, add regression coverage, and validate the fix."
- "Take this project through implementation, independent review, and final verification."
- Substantial batch requests involving refactoring, architecture migration, CI/CD repair, performance optimization, security hardening, or maintainability overhaul.

### Restraint & Non-Activation
Do **not** activate the full engineering workflow for:
- Passing mentions of "testing," "security," or "optimization" in general informational Q&A.
- Simple single-file code explanation or minor documentation lookup.
- Requests restricted to explaining how a feature works without modifying or auditing the workspace.

---

## Authority Boundaries & Autonomy

### Technical Authority
Within the authorized project workspace, the agent possesses implementation authority to:
- Create, modify, delete, move, rename, split, merge, and reorganize files and directories.
- Change dependencies, build tools, frameworks, system configuration, CI/CD pipelines, and project architecture.
- Execute any strategy (**KEEP, REFACTOR, PARTIALLY REPLACE, INCREMENTALLY MIGRATE, or FULLY REWRITE**) when justified by evidence.

### Authority Constraints
- **Implementation freedom does not equal requirement freedom.** Preserve legitimate business logic, public contracts, API compatibility, network protocols, security guarantees, and data integrity.
- Before replacing or deleting major components, identify downstream consumers, extract behavioral invariants and test fixtures, and ensure complete replacement coverage.
- **Human Input Boundary**: Execute necessary, justified work autonomously without repeatedly asking permission for intermediate actions. Request human input **only** when essential decisions require missing product intent, business policy, secret credentials, or authorization for irreversible external actions (e.g. pushing to production remotes or destroying external infrastructure).

---

## Hierarchy of Truth & Engineering Priorities

### Default Authority Order
When requirements, tests, or implementations conflict, resolve ambiguities using this default hierarchy:
1. **Current explicit requirements**
2. **Authoritative references & specifications**
3. **Expected input/output examples & fixtures**
4. **External contracts & API specs**
5. **Acceptance tests**
6. **Clearly intended user-visible behavior**
7. **Implementation code**
8. **Comments & inline documentation**
9. **Historical assumptions & legacy conventions**

### Engineering Priorities
Decisions must align with the following priority hierarchy:
1. **Correctness**
2. **Data Integrity**
3. **Security**
4. **Reliability**
5. **Testability**
6. **Maintainability**
7. **Performance**
8. **Operability**
9. **Developer Experience**
10. **Cosmetics**

Fix root causes rather than symptoms. Avoid speculative refactoring, fashionable stack changes, or unnecessary abstraction.

---

## Executable Engineering Workflow

The workflow consists of 12 sequential phases. Detail for each phase is provided in supporting manuals under `references/`.

```
Discovery ➔ Requirements ➔ Baseline ➔ Architecture Strategy ➔ Implementation
    ➔ Testing ➔ Independent Audit ➔ Remediation ➔ Adversarial Audit
    ➔ Hardening ➔ Reproducibility Checks ➔ Documentation & Delivery
```

### Phase 1: Discovery & Intent Reconstruction
- Inspect source code, test suites, documentation, build tooling, dependencies, git history, and runtime configuration.
- Reconstruct the project's legitimate purpose, public contracts, invariants, and critical user journeys.
- See: `references/discovery.md`

### Phase 2: Requirements & Invariant Formalization
- Formulate explicit system requirements and invariants.
- Document expected input/output contracts, error modes, performance bounds, and platform compatibility constraints.
- Template: `templates/requirements.md`

### Phase 3: Baseline & Environment Diagnostics
- Run build commands, test suites, static analyzers, linters, and runtime diagnostics.
- Record initial pass/fail baselines, code coverage, performance benchmarks, and environmental gaps.

### Phase 4: Architecture & Strategy Evaluation
- Evaluate current architecture against requirements.
- Choose and formally justify one of five strategies:
  - **KEEP**: Existing architecture is sound; apply targeted defect fixes.
  - **REFACTOR**: Modular layout is viable; improve internal structure and testability.
  - **PARTIALLY REPLACE**: Specific components are broken/obsolete; replace subsystems cleanly.
  - **INCREMENTALLY MIGRATE**: Architecture requires transition; migrate module by module.
  - **FULLY REWRITE**: Core design is fundamentally flawed, insecure, or unmaintainable; execute clean-slate rebuild.
- See: `references/architecture.md`

### Phase 5: Implementation & Root-Cause Remediation
- Execute code and configuration changes according to the chosen strategy.
- Investigate root causes for every confirmed bug; search for equivalent defect patterns across the entire codebase.
- Maintain atomic file safety, process tree isolation, and crash resilience.

### Phase 6: Verification & Test-Driven Validation
- Write and run automated verification across appropriate layers (unit, integration, end-to-end, boundary, regression).
- **Non-Weakening Rule**: Never achieve a green pass by deleting tests, weakening assertions, mocking away core behavior, or suppressing legitimate errors.
- See: `references/testing.md`

### Phase 7: Independent Quality & Code Audit
- Conduct a separate, critical review of the implementation against clean architecture and maintainability standards.
- Check code structure, naming consistency, error handling, logging, and dependency hygiene.
- Log findings in `templates/findings_log.md`.

### Phase 8: Remediation & Hardening
- Address all findings identified in the independent quality audit.
- Refine code, eliminate technical debt, and ensure error messages are informative and actionable.

### Phase 9: Adversarial Security & Reliability Audit
- Audit input boundaries, subprocess isolation, credential safety, resource leaks, concurrency, and failure paths.
- Exercise hostile inputs, unexpected terminations, missing tools, and network disconnects.
- See: `references/security_and_reliability.md`

### Phase 10: Reproducibility Checks
- Verify clean builds and execution in isolated or fresh environments.
- Confirm build scripts, setup documentation, and lock files work deterministically.

### Phase 11: Documentation & Maintainer Handoff
- Update `README.md`, `ARCHITECTURE.md`, API docs, and deployment guides so an unfamiliar developer can operate and maintain the project.
- Template: `templates/before_after_comparison.md`

### Phase 12: Final Delivery Report & Quality Convergence
- Produce a detailed delivery report matching project complexity.
- Template: `templates/verification_matrix.md`

---

## Quality Gates & Stopping Rule

### Quality Gates
To claim completion, the project must satisfy all quality gates:
1. **Verification Gate**: 100% of automated tests pass cleanly in the target environment.
2. **Contract Gate**: All public contracts, APIs, CLI flags, and data invariants are verified intact or explicitly migrated.
3. **Security & Privacy Gate**: No unhandled injection boundaries, path traversals, secret leaks, or insecure temporary files.
4. **Audit Gate**: All material actionable findings from independent and adversarial reviews are remediated.
5. **Documentation Gate**: Build, run, test, and maintenance instructions are up to date and verified.

### Convergence Stopping Rule
Iterate through the **Audit → Find → Prioritize → Fix → Test → Measure → Re-audit** loop while material actionable findings remain whose remediation provides positive net engineering value. Stop when quality gates are met and further iteration yields negligible engineering benefit.

---

## Supporting References & Helper Tools

- **Discovery Manual**: `references/discovery.md`
- **Architecture Strategy Manual**: `references/architecture.md`
- **Testing & Verification Manual**: `references/testing.md`
- **Security & Reliability Audit Manual**: `references/security_and_reliability.md`
- **Maintainability & Documentation Manual**: `references/maintainability.md`
- **Reporting Standards**: `references/reporting.md`
- **Validation Script**: `python3 skills/autonomous-software-engineering/scripts/validate_skill.py`
- **Evaluation Calibrations**: `evaluations/calibration_results.md`
