import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.process import ProcessTracker, safe_kill_process_tree
from src.core.storage import (
    _active_job_dirs,
    cleanup_stale_temp_directories,
    create_job_lock,
    is_process_alive,
    register_active_job_dir,
    safe_atomic_move,
    unregister_active_job_dir,
)


def test_safe_kill_process_tree():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 99999

    safe_kill_process_tree(mock_proc)
    mock_proc.kill.assert_called()

    # Already terminated process: poll() is not None
    mock_proc.reset_mock()
    mock_proc.poll.return_value = 0
    safe_kill_process_tree(mock_proc)
    mock_proc.kill.assert_not_called()


def test_process_tracker_lifecycle():
    tracker = ProcessTracker()
    assert not tracker.is_cancelled

    # Cancel
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 12345

    with patch("subprocess.Popen", return_value=mock_proc):
        proc = tracker.start_process(["test_cmd"])
        assert proc is mock_proc
        assert mock_proc in tracker._active_processes

        tracker.cancel()
        assert tracker.is_cancelled
        mock_proc.kill.assert_called()

    # Starting a process after cancel raises
    with pytest.raises(RuntimeError, match="Operation cancelled by user"):
        tracker.start_process(["cmd"])

    # Reset cancel
    tracker.reset_cancel()
    assert not tracker.is_cancelled


def test_process_tracker_wait_and_run_tracked():
    tracker = ProcessTracker()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.returncode = 0
    mock_proc.wait.return_value = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        code = tracker.run_tracked(["test_cmd"])
        assert code == 0
        assert tracker._current_process is None
        assert mock_proc not in tracker._active_processes


def test_storage_active_job_registration(tmp_path):
    job_dir = str(tmp_path / "active_job")
    register_active_job_dir(job_dir, {"item": 1})
    norm = os.path.normcase(os.path.abspath(job_dir))
    assert norm in _active_job_dirs

    lock_file = os.path.join(job_dir, ".job_lock")
    assert os.path.isfile(lock_file)
    with open(lock_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pid"] == os.getpid()

    unregister_active_job_dir(job_dir)
    assert norm not in _active_job_dirs


def test_is_process_alive():
    assert is_process_alive(os.getpid()) is True
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_cleanup_stale_temp_directories(tmp_path):
    temp_root = tmp_path / "output_root"
    temp_dir = temp_root / ".temp_upscaler"
    temp_dir.mkdir(parents=True)

    # 1. Stale folder (expired)
    stale_sub = temp_dir / "job_stale"
    stale_sub.mkdir()
    (stale_sub / "frame.jpg").write_text("x")
    # Set mtime back 2 days
    past = time.time() - 200_000
    os.utime(str(stale_sub), (past, past))

    # 2. Active registered folder
    active_sub = temp_dir / "job_active"
    active_sub.mkdir()
    register_active_job_dir(str(active_sub))

    # 3. Active locked folder (owned by our live PID)
    locked_sub = temp_dir / "job_locked"
    create_job_lock(str(locked_sub), {"status": "running"})

    cleaned = cleanup_stale_temp_directories(str(temp_root), max_age_seconds=86400)
    assert cleaned == 1
    assert not stale_sub.exists()
    assert active_sub.exists()
    assert locked_sub.exists()

    unregister_active_job_dir(str(active_sub))


def test_safe_atomic_move_cross_device(tmp_path):
    src = tmp_path / "source.mp4"
    src.write_text("video_data")
    dst = tmp_path / "final.mp4"

    # Regular atomic move
    safe_atomic_move(str(src), str(dst))
    assert dst.read_text() == "video_data"
    assert not src.exists()

    # Cross-device move simulation (EXDEV errno 18)
    src2 = tmp_path / "source2.mp4"
    src2.write_text("video_data_2")
    dst2 = tmp_path / "final2.mp4"

    real_replace = os.replace
    call_count = [0]

    def mock_replace(s, d):
        call_count[0] += 1
        if call_count[0] == 1:
            err = OSError("Cross-device link")
            err.errno = 18
            raise err
        return real_replace(s, d)

    with patch("os.replace", side_effect=mock_replace):
        safe_atomic_move(str(src2), str(dst2))
        assert dst2.read_text() == "video_data_2"
        assert not src2.exists()
