# Reference: Security, Reliability & Privacy Auditing

## Overview
Security, reliability, and data integrity must be designed into the engineering workflow. All software modifications must be audited against common failure modes, input boundary vulnerabilities, and operational edge cases.

---

## Security Audit Checklist

### 1. Input Boundary Hardening
- **Path Traversal**: Validate and sanitize all user-supplied paths. Prevent `../` manipulation from escaping authorized directories.
- **Command Injection**: Avoid shell string execution (`shell=True` in Python subprocesses, or unescaped shell commands). Use explicit argument arrays.
- **Format String & Template Injection**: Sanitize user inputs rendered into template strings, logs, or UI layouts.

### 2. Privacy & Secret Protection
- **Path Masking**: Automatically redact user home directories (`<USER_HOME>`) and sensitive paths in generated logs, error messages, or exported diagnostic reports.
- **Secret Hygiene**: Ensure no passwords, API keys, credentials, or personal identifiers are hardcoded, logged, or committed to version control.
- **Container / Metadata Scrubbing**: Strip internal process markers, machine names, and temporary tags from output files (`-map_metadata -1` or equivalent).

### 3. File System & Data Safety
- **Atomic Operations**: Render outputs to temporary files in isolated job directories and promote them via atomic replacement (`os.replace` or safe atomic move).
- **Lease Locks & Active Job Protection**: Use PID-based lease lock files (`.job_lock`) and active job tracking to prevent stale directory cleanup from deleting active processes.
- **Cross-Filesystem Safety**: Ensure staging/moving logic falls back safely to staged copies when crossing mount points or drive boundaries.

---

## Reliability Audit Checklist

### 1. Resource & Process Lifecycle
- **Subprocess Isolation**: Spawn child processes in separate process groups (`CREATE_NEW_PROCESS_GROUP` on Windows, process groups on POSIX) and ensure clean process tree termination upon cancellation.
- **Resource Leaks**: Ensure open file handles, memory allocations, subprocesses, and temporary directories are cleaned up in `finally` blocks or via context managers.
- **Vulkan / GPU Fallbacks**: Catch hardware allocation failures (OOM) and retry gracefully with conservative tile sizes and reduced worker threads.

### 2. Failure Injection & Resilience
- **Missing External Tools**: Ensure graceful detection and informative diagnostics when dependencies (e.g. FFmpeg, Vulkan binaries) are absent.
- **Missing Soundtracks / Audio Repair**: Check media integrity contracts (e.g. verifying audio stream presence) and trigger automated repair remux passes when audio is lost during video transcoding.
- **Interrupted Operations**: Verify system state remains consistent if process is aborted mid-execution.
