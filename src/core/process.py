"""Subprocess lifecycle management, process tree tracking, and controlled cancellation.

Ensures child processes (FFmpeg, Vulkan engines) run in separate process groups
and are terminated cleanly on cancellation without leaving orphaned processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading


def safe_kill_process_tree(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Terminates an entire subprocess tree safely and completely, waiting for file locks to release."""
    if proc.poll() is not None:
        return
    pid = getattr(proc, "pid", None)
    if os.name == "nt" and pid:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except Exception:
            pass
    elif hasattr(os, "killpg") and hasattr(os, "getpgid") and pid:
        try:
            getpgid = getattr(os, "getpgid")
            killpg = getattr(os, "killpg")
            pgid = getpgid(pid)
            sigkill = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15))
            killpg(pgid, sigkill)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError, TypeError):
        pass


class ProcessTracker:
    """Tracks active subprocesses and manages atomic cancellation across pipeline stages."""

    def __init__(self) -> None:
        self._is_cancelled = False
        self._current_process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancel(self) -> None:
        """Signals cancellation and terminates all active child processes."""
        self._is_cancelled = True
        with self._process_lock:
            processes = list(self._active_processes)
        for proc in processes:
            safe_kill_process_tree(proc)

    def reset_cancel(self) -> None:
        self._is_cancelled = False
        self._current_process = None

    def start_process(self, cmd: list[str], **popen_kwargs) -> subprocess.Popen:
        """Spawns a child process in a new process group and tracks it."""
        if self._is_cancelled:
            raise RuntimeError("Operation cancelled by user.")
        if os.name == "nt":
            creationflags = popen_kwargs.get("creationflags", 0)
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
            popen_kwargs["creationflags"] = creationflags
        else:
            if "preexec_fn" not in popen_kwargs and "start_new_session" not in popen_kwargs:
                popen_kwargs["start_new_session"] = True
        with self._process_lock:
            if self._is_cancelled:
                raise RuntimeError("Operation cancelled by user.")
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self._active_processes.add(proc)
        return proc

    def forget_process(self, proc: subprocess.Popen) -> None:
        with self._process_lock:
            self._active_processes.discard(proc)

    def wait_for_process(self) -> int:
        proc = self._current_process
        if proc is None:
            return 0
        while True:
            try:
                proc.wait(timeout=0.05)
                return proc.returncode if proc.returncode is not None else 0
            except subprocess.TimeoutExpired:
                if self._is_cancelled:
                    safe_kill_process_tree(proc)
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
                    return -1

    def run_tracked(self, cmd: list[str], **popen_kwargs) -> int:
        proc = self.start_process(cmd, **popen_kwargs)
        self._current_process = proc
        try:
            return self.wait_for_process()
        finally:
            self.forget_process(proc)
            if self._current_process is proc:
                self._current_process = None
