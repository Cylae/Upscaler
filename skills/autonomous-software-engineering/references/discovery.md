# Reference: Discovery & Intent Reconstruction

## Overview
The Discovery phase establishes a comprehensive model of the project before any code is modified. The goal is to reconstruct the true product intent, architectural boundaries, system invariants, public contracts, and execution baseline.

---

## Key Discovery Steps

### 1. Codebase Inventory
Inspect the workspace structure using filesystem traversal tools:
- Locate core source code, entry points, configuration files, build manifests (`package.json`, `requirements.txt`, `Cargo.toml`, `go.mod`, `pom.xml`, etc.).
- Identify test files, integration test fixtures, mock data, and test utilities.
- Map CI/CD workflows (`.github/workflows`, `.gitlab-ci.yml`), deployment scripts, and container configurations (`Dockerfile`, `docker-compose.yml`).
- Read documentation files (`README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `AGENTS.md`).

### 2. Public Contract & Invariant Reconstruction
Document the external surface of the software:
- **APIs & Protocols**: Endpoints, request/response schemas, gRPC protos, GraphQL schemas.
- **CLI Tools**: Command-line arguments, environment variables, exit codes, stdin/stdout/stderr conventions.
- **Libraries**: Public exports, package signatures, interface definitions, backward compatibility guarantees.
- **Data & File System Invariants**: File naming conventions, database schemas, migration formats, atomic write guarantees, lock file mechanisms.

### 3. Environment & Toolchain Diagnostics
Inventory available local tools:
- Compiler/interpreter versions (`python --version`, `node --version`, `go version`, `gcc --version`).
- Build systems and runners (`pytest`, `npm test`, `cargo test`, `make`, `maven`).
- System utilities (`ffmpeg`, `nvidia-smi`, `docker`, `git`).
- Note unavailable tools, missing hardware acceleration (e.g. GPU), or network access restrictions.

### 4. Git History & Context Analysis
- Review recent commits, commit messages, open PRs, or branch history to understand recent work, past bug fixes, and intended features.
- Distinguish legacy workarounds or accidental behavior from intentional requirements.

---

## Output Artifact
Summarize discovery findings in `requirements.md` (using `templates/requirements.md`) before proceeding to baseline execution.
