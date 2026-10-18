"""A stop request cannot miss a child spawned concurrently or be reset by a job."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from unittest.mock import Mock

import pytest

from src.core.hardware import HardwareInfo
from src.core.process import ProcessTracker
from src.core.video_pipeline import UpscaleJobConfig, VideoUpscalerPipeline


@pytest.mark.parametrize("kind", ["pipeline", "tracker"])
def test_cancellation_during_spawn_reaches_the_new_child(monkeypatch, kind):
    tracker = VideoUpscalerPipeline(HardwareInfo()) if kind == "pipeline" else ProcessTracker()
    start = tracker._start_process if kind == "pipeline" else tracker.start_process
    spawning = threading.Event()
    finish_spawn = threading.Event()
    child = Mock(pid=None)
    child.poll.return_value = None

    def popen(*_args, **_kwargs):
        spawning.set()
        assert finish_spawn.wait(5)
        return child

    monkeypatch.setattr("subprocess.Popen", popen)
    with ThreadPoolExecutor(max_workers=2) as pool:
        started = pool.submit(start, ["test-engine"])
        assert spawning.wait(5)
        stopped = pool.submit(tracker.cancel)
        deadline = time.monotonic() + 5
        while not tracker._is_cancelled and time.monotonic() < deadline:
            time.sleep(0.001)
        try:
            assert tracker._is_cancelled
        finally:
            finish_spawn.set()
        assert started.result(timeout=5) is child
        stopped.result(timeout=5)

    child.kill.assert_called_once_with()
    with pytest.raises(RuntimeError, match="cancelled"):
        start(["another-engine"])


def test_cancelled_pipeline_does_not_restart_when_job_enters(monkeypatch):
    pipeline = VideoUpscalerPipeline(HardwareInfo())
    pipeline.cancel()
    probe = Mock(side_effect=AssertionError("A stopped job must do no media work"))
    monkeypatch.setattr("src.core.video_pipeline.probe_video", probe)

    assert pipeline.process_video(UpscaleJobConfig("input.mp4", "output")) == (False, "Cancelled by user.")
    probe.assert_not_called()
    pipeline.reset_cancel()
    assert not pipeline._is_cancelled
