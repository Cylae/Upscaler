"""Filesystem storage, atomic operations, PID lease locks, and temporary directory management.

Protects active renders against mid-process deletion and ensures cross-filesystem
atomic output file replacement.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from typing import Dict, Optional, Set

_active_job_dirs: Set[str] = set()
_active_job_dirs_lock = threading.Lock()


def register_active_job_dir(work_dir: str, job_info: Optional[Dict] = None) -> None:
    """Registers an active work directory to protect it from cleanup."""
    norm_path = os.path.normcase(os.path.abspath(work_dir))
    with _active_job_dirs_lock:
        _active_job_dirs.add(norm_path)
    create_job_lock(work_dir, job_info)


def create_job_lock(work_dir: str, job_info: Optional[Dict] = None) -> str:
    """Writes a PID-based lease lock file (.job_lock) inside the working directory."""
    os.makedirs(work_dir, exist_ok=True)
    lock_file = os.path.join(work_dir, ".job_lock")
    payload = {
        "pid": os.getpid(),
        "created_at": time.time(),
        "last_heartbeat": time.time(),
        "info": job_info or {}
    }
    try:
        with open(lock_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
    return lock_file


def unregister_active_job_dir(work_dir: str) -> None:
    """Unregisters an active work directory when work is finished."""
    norm_path = os.path.normcase(os.path.abspath(work_dir))
    with _active_job_dirs_lock:
        _active_job_dirs.discard(norm_path)


def is_process_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is actively running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            STILL_ACTIVE = 259
            return exit_code.value == STILL_ACTIVE
        except Exception:
            return True  # Conservative: assume alive if error occurs
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def cleanup_stale_temp_directories(
    root_dir: Optional[str] = None,
    max_age_seconds: int = 86400,
    max_age_hours: Optional[float] = None
) -> int:
    """Scans and removes stale .temp_upscaler directories left over from terminated sessions.

    Safely protects active jobs:
    1. Checks in-memory active job directories for the current process.
    2. Checks PID-based lease lock files (.job_lock) to avoid deleting directories
       of concurrent processes currently performing work.
    3. Uses a conservative default max age (24 hours instead of 15 minutes).
    """
    if max_age_hours is not None:
        max_age_seconds = int(max_age_hours * 3600)

    cleaned = 0
    search_dirs = [root_dir] if root_dir else []
    try:
        from src.core.hardware import get_base_dir
        search_dirs.append(get_base_dir())
    except Exception:
        pass

    for s_dir in search_dirs:
        if not s_dir or not os.path.isdir(s_dir):
            continue
        temp_dir = os.path.join(s_dir, ".temp_upscaler")
        if os.path.isdir(temp_dir):
            try:
                for entry in os.listdir(temp_dir):
                    sub = os.path.join(temp_dir, entry)
                    if not os.path.isdir(sub):
                        continue

                    # 1. Skip if registered as active in current process
                    norm_sub = os.path.normcase(os.path.abspath(sub))
                    with _active_job_dirs_lock:
                        if norm_sub in _active_job_dirs:
                            continue

                    # 2. Check lock file for active foreign processes
                    lock_file = os.path.join(sub, ".job_lock")
                    if os.path.isfile(lock_file):
                        try:
                            with open(lock_file, "r", encoding="utf-8") as f:
                                lock_data = json.load(f)
                            owner_pid = lock_data.get("pid")
                            if owner_pid and is_process_alive(int(owner_pid)):
                                continue
                        except Exception:
                            pass

                    # 3. Check modification age
                    try:
                        mtime = os.path.getmtime(sub)
                        if time.time() - mtime >= max_age_seconds:
                            shutil.rmtree(sub, ignore_errors=True)
                            cleaned += 1
                    except Exception:
                        pass
            except Exception:
                pass
    return cleaned


def safe_atomic_move(src: str, dst: str) -> None:
    """Atomically moves src to dst, with safe fallback for cross-filesystem moves.

    If src and dst live on different volumes/drives (WinError 17 on Windows or
    EXDEV errno 18 on POSIX), copies to a temporary file on the destination
    directory first, then performs atomic replace.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as exc:
        if exc.errno in (17, 18) or (hasattr(exc, "winerror") and exc.winerror == 17):
            temp_dst = dst + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
            try:
                shutil.copy2(src, temp_dst)
                os.replace(temp_dst, dst)
                try:
                    os.remove(src)
                except OSError:
                    pass
            finally:
                if os.path.exists(temp_dst):
                    try:
                        os.remove(temp_dst)
                    except OSError:
                        pass
        else:
            raise
