"""Comprehensive Concurrency, Race Condition, and Pipeline Stress Test Suite.

Attacks AnimaScale's concurrency model across:
1. Rapid cancellation cycles under active demuxing, inference, and muxing:
   - Verifies zero orphaned child processes (FFmpeg, Vulkan binaries).
   - Verifies zero leaked file handles / pipe deadlocks.
   - Verifies zero lingering .part_* or .lock files.
2. High multi-worker concurrency (4 to 8 parallel workers):
   - Parallel batch workers competing for disk files, output directories, and process locks.
   - Contention during atomic output file promotion.
3. Background stale directory cleanup concurrency:
   - Verifies cleanup routines never delete directories actively in use by workers.
   - Validates reference-counted directory lease locks and hierarchical directory protection.
4. Active queue mutations under heavy load:
   - Dynamic queue reordering while workers are running.
   - Queue pausing and resuming without deadlocks.
   - Stop while paused.
   - Queue clearing and item removal while workers are actively processing.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.batch import BatchManager, EpisodeStatus, QueueItem
from src.core.hardware import HardwareInfo
from src.core.probe import VideoMetadata
from src.core.process import ProcessTracker
from src.core.storage import (
    _active_job_counts,
    cleanup_stale_temp_directories,
    create_job_lock,
    is_dir_active,
    register_active_job_dir,
    safe_atomic_move,
    unregister_active_job_dir,
)
from src.engine.media_stream import (
    MuxerConfig,
    StreamPipelineController,
    VideoFrame,
    VideoStreamDemuxer,
    VideoStreamMuxer,
)
from src.engine.vulkan_backend import VulkanConfig, VulkanNCNNBackend
from src.ui.workers import UpscaleWorker


# ==============================================================================
# Helper Fixtures & Mock Utilities
# ==============================================================================

@pytest.fixture
def dummy_meta():
    return VideoMetadata(
        filepath="test.mp4",
        width=1280,
        height=720,
        fps=24.0,
        duration=5.0,
        nb_frames=120,
        codec_name="h264",
        has_audio=True,
        audio_codec="aac",
        has_subtitles=False,
    )


# ==============================================================================
# GROUP 1: RAPID CANCELLATION CYCLES (DEMUXING, INFERENCE, MUXING)
# ==============================================================================

def test_rapid_cancellation_during_active_demuxing(tmp_path, dummy_meta):
    """Aborting while demuxer process is actively streaming uncompressed frames."""
    dummy_file = tmp_path / "stream_in.mp4"
    dummy_file.write_bytes(b"\x00" * 4096)

    tracker = ProcessTracker()
    with patch("src.engine.media_stream.probe_video", return_value=dummy_meta):
        demuxer = VideoStreamDemuxer(
            str(dummy_file),
            ffmpeg_bin="ffmpeg",
            process_tracker=tracker,
        )

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.pid = 99912
    fake_frame_bytes = demuxer.frame_bytes
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.read.return_value = b"\x10" * fake_frame_bytes
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline.return_value = b""

    with patch("subprocess.Popen", return_value=mock_proc):
        demuxer.start()
        assert demuxer.is_running is True

        # Read 2 frames while active
        f1 = demuxer.read_frame()
        assert f1 is not None

        # Violently close / cancel demuxer
        demuxer.close()

        assert demuxer.is_running is False
        assert mock_proc.stdout.close.called
        assert mock_proc.stderr.close.called
        mock_proc.kill.assert_called()
        assert len(tracker._active_processes) == 0


def test_rapid_cancellation_during_active_inference(tmp_path, dummy_meta):
    """Aborting while neural inference worker is mid-computation."""
    dummy_file = tmp_path / "infer_in.mp4"
    dummy_file.write_bytes(b"\x00" * 4096)
    out_file = tmp_path / "infer_out.mp4"

    with patch("src.engine.media_stream.probe_video", return_value=dummy_meta):
        demuxer = VideoStreamDemuxer(str(dummy_file))

        mux_cfg = MuxerConfig(
            target_w=1280,
            target_h=720,
            fps=24.0,
            encoder="libx265",
            source_path=str(dummy_file),
        )
        muxer = VideoStreamMuxer(str(out_file), config=mux_cfg)

    inference_entered = threading.Event()
    inference_cancelled = threading.Event()

    def slow_inference(frame: VideoFrame) -> VideoFrame:
        inference_entered.set()
        # Wait until test signals cancellation
        inference_cancelled.wait(timeout=3.0)
        return frame

    controller = StreamPipelineController(
        demuxer=demuxer,
        muxer=muxer,
        inference_fn=slow_inference,
        queue_size=4,
    )

    # Mock demuxer and muxer subprocesses
    mock_demux_proc = MagicMock(spec=subprocess.Popen, pid=1111)
    mock_demux_proc.poll.return_value = None
    mock_demux_proc.stdout = MagicMock()
    mock_demux_proc.stdout.read.return_value = b"\x00" * demuxer.frame_bytes
    mock_demux_proc.stderr = MagicMock()
    mock_demux_proc.stderr.readline.return_value = b""

    mock_mux_proc = MagicMock(spec=subprocess.Popen, pid=2222)
    mock_mux_proc.poll.return_value = None
    mock_mux_proc.stdin = MagicMock()
    mock_mux_proc.stdin.write.return_value = None
    mock_mux_proc.stderr = MagicMock()
    mock_mux_proc.stderr.readline.return_value = b""

    with patch.object(demuxer, "_process", mock_demux_proc), \
         patch.object(muxer, "_process", mock_mux_proc):

        def run_thread():
            try:
                controller.run()
            except Exception:
                pass

        t = threading.Thread(target=run_thread, daemon=True)
        t.start()

        # Wait for inference thread to enter
        assert inference_entered.wait(timeout=2.0)

        # Trigger violent cancellation
        controller.cancel()
        inference_cancelled.set()
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert controller.is_cancelled is True
        mock_demux_proc.kill.assert_called()
        mock_mux_proc.kill.assert_called()


def test_rapid_cancellation_during_active_muxing_cleans_part_and_lock_files(tmp_path):
    """Aborting while muxer is actively writing: zero lingering .part_* or .lock files."""
    dest_file = tmp_path / "out_video.mp4"

    cfg = MuxerConfig(
        target_w=640,
        target_h=480,
        fps=30.0,
        encoder="libx265",
    )
    muxer = VideoStreamMuxer(str(dest_file), config=cfg)

    # Ensure .part_* and .lock file exist on disk
    part_file = muxer.part_path
    lock_file = muxer.lock_path
    assert os.path.exists(lock_file)

    # Create dummy part file as if FFmpeg started writing
    with open(part_file, "wb") as f:
        f.write(b"\x00" * 1024 * 100)
    assert os.path.exists(part_file)

    mock_proc = MagicMock(spec=subprocess.Popen, pid=3333)
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write.return_value = None
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.readline.return_value = b""
    muxer._process = mock_proc

    # Abort muxer
    muxer.close()

    # Verify zero lingering files
    assert not os.path.exists(part_file)
    assert not os.path.exists(lock_file)
    assert mock_proc.kill.called
    assert not is_dir_active(str(tmp_path))


def test_twenty_rapid_back_to_back_cancellation_cycles(tmp_path, dummy_meta):
    """Stress test: 20 rapid back-to-back violent cancellations leave zero leaked locks or part files."""
    out_dir = tmp_path / "rapid_cycles"
    out_dir.mkdir()

    for i in range(20):
        dest_file = out_dir / f"test_{i}.mp4"
        cfg = MuxerConfig(
            target_w=1280,
            target_h=720,
            fps=24.0,
            encoder="libx265",
        )
        muxer = VideoStreamMuxer(str(dest_file), config=cfg)

        # Mock writing
        with open(muxer.part_path, "wb") as f:
            f.write(b"interrupted data" * 50)

        mock_proc = MagicMock(spec=subprocess.Popen, pid=5000 + i)
        mock_proc.poll.return_value = None
        muxer._process = mock_proc

        # Immediate abort
        muxer.close()

    # Scan directory: verify zero .lock and zero .part_* files remain
    files_remaining = list(out_dir.iterdir())
    locks = [f.name for f in files_remaining if f.name.endswith(".lock")]
    parts = [f.name for f in files_remaining if ".part_" in f.name]

    assert len(locks) == 0, f"Found leaked lock files: {locks}"
    assert len(parts) == 0, f"Found leaked part files: {parts}"
    assert not is_dir_active(str(out_dir))


def test_vulkan_ncnn_backend_cancellation_under_load(tmp_path):
    """Simulates Vulkan NCNN binary cancellation, ensuring process tree is terminated cleanly."""
    backend = VulkanNCNNBackend.__new__(VulkanNCNNBackend)
    backend.engine_type = "realesrgan"
    backend.config = VulkanConfig()

    mock_proc = MagicMock(spec=subprocess.Popen, pid=8888)
    mock_proc.poll.return_value = None
    backend._active_process = mock_proc

    # Cancel execution
    backend.cancel()

    assert backend._active_process is None
    assert mock_proc.kill.called


# ==============================================================================
# GROUP 2: HIGH MULTI-WORKER CONCURRENCY (4 TO 8 PARALLEL WORKERS)
# ==============================================================================

def test_eight_parallel_workers_batch_execution(tmp_path):
    """8 parallel workers executing queue items competing for locks and callbacks."""
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()

    bm = BatchManager(HardwareInfo())

    # Build queue with 16 items
    for i in range(1, 17):
        src_path = str(tmp_path / f"source_{i}.mp4")
        with open(src_path, "wb") as f:
            f.write(b"\x00" * 1024)
        item = QueueItem(
            index=i,
            source_path=src_path,
            relative_source_path=f"source_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=str(out_dir / f"out_{i}.mp4"),
            file_size_mb=1.0,
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    active_concurrency_lock = threading.Lock()
    active_concurrency = [0]
    peak_concurrency = [0]

    # Mock pipeline that tracks active concurrency
    class ConcurrencyTrackingPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            with active_concurrency_lock:
                active_concurrency[0] += 1
                if active_concurrency[0] > peak_concurrency[0]:
                    peak_concurrency[0] = active_concurrency[0]

            # Simulate work
            time.sleep(0.04)
            if progress_callback:
                progress_callback(0.5, "Upscaling...", {"file_eta_str": "00:01"})
            time.sleep(0.04)

            # Create destination file to complete promotion contract
            with open(job_config.output_dir + "/" + os.path.basename(job_config.input_file), "wb") as f:
                f.write(b"finished")

            with active_concurrency_lock:
                active_concurrency[0] -= 1

            return True, "Completed successfully"

    with patch("src.core.batch.VideoUpscalerPipeline", ConcurrencyTrackingPipeline):
        bm.run_batch(
            output_dir=str(out_dir),
            base_dir=str(tmp_path),
            max_workers=8,
            overwrite=True,
        )

    # Verify peak concurrency reached multi-worker level (4 to 8 workers)
    assert peak_concurrency[0] >= 4, f"Expected concurrency >= 4, got {peak_concurrency[0]}"
    assert all(it.status == EpisodeStatus.COMPLETED for it in bm.queue)
    assert bm.is_running is False


def test_parallel_output_promotion_file_locks_contention(tmp_path):
    """8 parallel threads concurrently promoting files via safe_atomic_move in the same folder."""
    out_dir = tmp_path / "contention_out"
    out_dir.mkdir()

    num_threads = 8

    def worker_promotion(worker_id: int):
        src_file = str(tmp_path / f"temp_{worker_id}.dat")
        with open(src_file, "wb") as f:
            f.write(f"content from worker {worker_id}".encode("utf-8"))

        dst_file = str(out_dir / f"final_{worker_id}.dat")
        safe_atomic_move(src_file, dst_file)
        return dst_file

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker_promotion, i) for i in range(num_threads)]
        results = [f.result(timeout=5.0) for f in futures]

    assert len(results) == num_threads
    for i, dst in enumerate(results):
        assert os.path.exists(dst)
        with open(dst, "rb") as f:
            assert f.read() == f"content from worker {i}".encode("utf-8")

    # Verify zero temporary files left in destination folder
    leftovers = [f.name for f in out_dir.iterdir() if ".tmp_" in f.name]
    assert len(leftovers) == 0, f"Found leftover temporary files: {leftovers}"


def test_parallel_workers_muxer_disk_isolation(tmp_path):
    """8 parallel workers concurrently using VideoStreamMuxer in the same target directory."""
    out_dir = tmp_path / "shared_mux_dir"
    out_dir.mkdir()

    num_workers = 8

    def mux_worker(worker_id: int):
        dest_file = out_dir / f"mux_out_{worker_id}.mp4"
        cfg = MuxerConfig(target_w=320, target_h=240, fps=24.0, encoder="libx265")
        muxer = VideoStreamMuxer(str(dest_file), config=cfg)

        # Verify unique part and lock files
        assert os.path.exists(muxer.lock_path)
        with open(muxer.part_path, "wb") as f:
            f.write(b"encoded video bytes" * 100)

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.poll.return_value = 0
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline.return_value = b""
        muxer._process = mock_proc

        # Promote
        final = muxer.finalize()
        return final

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(mux_worker, i) for i in range(num_workers)]
        results = [f.result(timeout=10.0) for f in futures]

    assert len(results) == num_workers
    for res in results:
        assert os.path.exists(res)

    # Verify zero .part_* or .lock files remain
    files = list(out_dir.iterdir())
    locks = [f.name for f in files if f.name.endswith(".lock")]
    parts = [f.name for f in files if ".part_" in f.name]
    assert len(locks) == 0, f"Lingering locks: {locks}"
    assert len(parts) == 0, f"Lingering parts: {parts}"
    assert not is_dir_active(str(out_dir))


# ==============================================================================
# GROUP 3: BACKGROUND STALE DIRECTORY CLEANUP CONCURRENCY
# ==============================================================================

def test_active_workers_immune_to_background_stale_cleanup(tmp_path):
    """Cleanup routines must never delete directories actively used by running workers."""
    base_dir = tmp_path / "render_root"
    base_dir.mkdir()
    temp_root = base_dir / ".temp_upscaler"
    temp_root.mkdir()

    # Worker A and Worker B directories
    worker_a_dir = str(temp_root / "upscale_worker_A")
    worker_b_dir = str(temp_root / "upscale_worker_B")
    stale_dead_dir = str(temp_root / "upscale_abandoned_dead")

    os.makedirs(worker_a_dir)
    os.makedirs(worker_b_dir)
    os.makedirs(stale_dead_dir)

    # Register Worker A and Worker B as active
    register_active_job_dir(worker_a_dir)
    register_active_job_dir(worker_b_dir)

    # Create stale dead directory with a non-existent PID (PID 99999999)
    create_job_lock(stale_dead_dir, pid=99999999)
    # Set dead owner lock file modification time to 1 hour ago
    lock_file = os.path.join(stale_dead_dir, ".job_lock")
    old_time = time.time() - 3600
    os.utime(lock_file, (old_time, old_time))
    os.utime(stale_dead_dir, (old_time, old_time))

    cleanup_stop = threading.Event()

    def aggressive_cleanup_loop():
        while not cleanup_stop.is_set():
            cleanup_stale_temp_directories(root_dir=str(base_dir), max_age_seconds=0)
            time.sleep(0.005)

    cleanup_thread = threading.Thread(target=aggressive_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # Workers do work for 200 ms while cleanup sweeps continuously
    for _ in range(20):
        assert os.path.exists(worker_a_dir), "Worker A active directory was deleted by cleanup!"
        assert os.path.exists(worker_b_dir), "Worker B active directory was deleted by cleanup!"
        time.sleep(0.01)

    # Worker A finishes and unregisters
    unregister_active_job_dir(worker_a_dir)
    time.sleep(0.05)

    # Worker B is STILL RUNNING -> Worker B must NOT be touched
    assert os.path.exists(worker_b_dir), "Worker B was deleted after Worker A unregistered!"

    # Finish Worker B
    unregister_active_job_dir(worker_b_dir)
    cleanup_stop.set()
    cleanup_thread.join(timeout=2.0)

    # Verify dead directory was cleaned up
    assert not os.path.exists(stale_dead_dir), "Dead stale directory was not cleaned up!"


def test_reference_counted_multisession_directory_lease(tmp_path):
    """Reference counted registry keeps shared directories protected until all workers unregister."""
    shared_dir = str(tmp_path / "shared_workspace")
    os.makedirs(shared_dir)

    # Worker 1 registers
    register_active_job_dir(shared_dir)
    assert is_dir_active(shared_dir) is True
    assert _active_job_counts[os.path.normcase(os.path.abspath(shared_dir))] == 1

    # Worker 2 registers same directory
    register_active_job_dir(shared_dir)
    assert _active_job_counts[os.path.normcase(os.path.abspath(shared_dir))] == 2

    # Worker 1 unregisters
    unregister_active_job_dir(shared_dir)
    # MUST STILL BE ACTIVE for Worker 2
    assert is_dir_active(shared_dir) is True
    assert _active_job_counts[os.path.normcase(os.path.abspath(shared_dir))] == 1

    # Worker 2 unregisters
    unregister_active_job_dir(shared_dir)
    assert is_dir_active(shared_dir) is False
    assert os.path.normcase(os.path.abspath(shared_dir)) not in _active_job_counts
    # Verify .job_lock was cleanly removed
    assert not os.path.exists(os.path.join(shared_dir, ".job_lock"))


def test_parent_child_directory_hierarchy_cleanup_protection(tmp_path):
    """Hierarchy protection: child of active directory and parent of active directory are protected."""
    parent_dir = str(tmp_path / "parent_work")
    child_dir = str(tmp_path / "parent_work" / "child_job")
    os.makedirs(child_dir, exist_ok=True)

    register_active_job_dir(child_dir)

    # Checking parent must report active because its child is actively working
    assert is_dir_active(parent_dir) is True
    assert is_dir_active(child_dir) is True

    unregister_active_job_dir(child_dir)
    assert is_dir_active(parent_dir) is False
    assert is_dir_active(child_dir) is False


# ==============================================================================
# GROUP 4: ACTIVE QUEUE MUTATIONS UNDER HEAVY LOAD
# ==============================================================================

def test_queue_reordering_while_workers_actively_processing(tmp_path):
    """Reordering the queue dynamically while workers are running picks up newly prioritized items."""
    bm = BatchManager(HardwareInfo())
    processed_order: list[int] = []
    order_lock = threading.Lock()

    # Build queue with 8 items
    for i in range(1, 9):
        item = QueueItem(
            index=i,
            source_path=f"file_{i}.mp4",
            relative_source_path=f"file_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=f"out_{i}.mp4",
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    worker_1_running = threading.Event()
    allow_worker_1_to_finish = threading.Event()

    class DynamicOrderMockPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            idx = int(job_config.input_file.split("_")[1].split(".")[0])
            if idx == 1:
                worker_1_running.set()
                allow_worker_1_to_finish.wait(timeout=3.0)
            time.sleep(0.02)
            with order_lock:
                processed_order.append(idx)
            return True, "Done"

    with patch("src.core.batch.VideoUpscalerPipeline", DynamicOrderMockPipeline):
        def run_batch():
            bm.run_batch(
                output_dir=str(tmp_path),
                base_dir=str(tmp_path),
                max_workers=2,
            )

        t = threading.Thread(target=run_batch, daemon=True)
        t.start()

        # Wait until worker 1 starts executing item 1
        assert worker_1_running.wait(timeout=2.0)

        # Dynamically move Item 8 to the very top (index 2 right after item 1)
        # Using reorder_queue: prioritize [8, 7, 2, 3, 4, 5, 6]
        bm.reorder_queue([8, 7, 2, 3, 4, 5, 6])

        # Release worker 1
        allow_worker_1_to_finish.set()
        t.join(timeout=5.0)

        assert not t.is_alive()
        # Item 8 MUST have been processed before items 3, 4, 5, 6
        assert 8 in processed_order
        idx_8 = processed_order.index(8)
        idx_6 = processed_order.index(6)
        assert idx_8 < idx_6, f"Item 8 was not prioritized! Processed order: {processed_order}"


def test_queue_pause_and_resume_under_multi_worker_load(tmp_path):
    """Pausing queue prevents new jobs from dispatching; resuming restarts dispatch seamlessly."""
    bm = BatchManager(HardwareInfo())
    processed_count = [0]
    count_lock = threading.Lock()

    for i in range(1, 9):
        item = QueueItem(
            index=i,
            source_path=f"file_{i}.mp4",
            relative_source_path=f"file_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=f"out_{i}.mp4",
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    first_batch_started = threading.Event()

    class PauseTestPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            first_batch_started.set()
            time.sleep(0.05)
            with count_lock:
                processed_count[0] += 1
            return True, "Done"

    with patch("src.core.batch.VideoUpscalerPipeline", PauseTestPipeline):
        def run():
            bm.run_batch(
                output_dir=str(tmp_path),
                base_dir=str(tmp_path),
                max_workers=2,
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert first_batch_started.wait(timeout=2.0)

        # Pause the queue
        bm.pause()
        assert bm.is_paused is True

        # Wait for currently running workers (at most 2) to settle
        time.sleep(0.2)
        with count_lock:
            count_at_pause = processed_count[0]

        # Verify no new jobs are starting while paused
        time.sleep(0.15)
        with count_lock:
            assert processed_count[0] == count_at_pause, "New jobs started while queue was paused!"

        # Resume the queue
        bm.resume()
        assert bm.is_paused is False

        t.join(timeout=5.0)
        assert not t.is_alive()
        assert processed_count[0] == 8
        assert all(it.status == EpisodeStatus.COMPLETED for it in bm.queue)


def test_queue_pause_then_stop_deadlock_freedom(tmp_path):
    """Calling stop() while queue is paused must never deadlock and exit immediately."""
    bm = BatchManager(HardwareInfo())

    for i in range(1, 5):
        item = QueueItem(
            index=i,
            source_path=f"file_{i}.mp4",
            relative_source_path=f"file_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=f"out_{i}.mp4",
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    started = threading.Event()

    class SlowPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            started.set()
            time.sleep(0.5)
            return True, "Done"

    with patch("src.core.batch.VideoUpscalerPipeline", SlowPipeline):
        def run():
            bm.run_batch(str(tmp_path), str(tmp_path), max_workers=2)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert started.wait(timeout=2.0)
        bm.pause()
        assert bm.is_paused is True

        # Stop while paused
        t_stop_start = time.monotonic()
        bm.stop()
        t.join(timeout=2.0)

        t_elapsed = time.monotonic() - t_stop_start
        assert not t.is_alive(), "Batch worker deadlocked when stopped while paused!"
        assert t_elapsed < 2.0
        assert bm.is_running is False


def test_queue_clearing_while_workers_actively_processing(tmp_path):
    """Clearing queue while workers are running completes active jobs and skips remaining."""
    bm = BatchManager(HardwareInfo())

    for i in range(1, 9):
        item = QueueItem(
            index=i,
            source_path=f"file_{i}.mp4",
            relative_source_path=f"file_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=f"out_{i}.mp4",
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    job_started = threading.Event()
    items_processed: list[str] = []
    lock = threading.Lock()

    class ClearTestPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            job_started.set()
            time.sleep(0.06)
            with lock:
                items_processed.append(job_config.input_file)
            return True, "Done"

    with patch("src.core.batch.VideoUpscalerPipeline", ClearTestPipeline):
        def run():
            bm.run_batch(str(tmp_path), str(tmp_path), max_workers=2)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert job_started.wait(timeout=2.0)

        # Dynamically clear queue while jobs are running
        bm.clear_queue()
        assert len(bm.queue) == 0

        t.join(timeout=4.0)
        assert not t.is_alive()
        # Only the actively claimed jobs (at most 2) finished; the remaining 6 were cleared
        assert len(items_processed) <= 2


def test_dynamic_item_removal_during_active_execution(tmp_path):
    """Removing items by indices while batch is running prevents them from executing."""
    bm = BatchManager(HardwareInfo())

    for i in range(1, 7):
        item = QueueItem(
            index=i,
            source_path=f"file_{i}.mp4",
            relative_source_path=f"file_{i}.mp4",
            target_scene_name=f"out_{i}.mp4",
            target_full_path=f"out_{i}.mp4",
            status=EpisodeStatus.PENDING,
        )
        bm.queue.append(item)

    worker_running = threading.Event()
    executed_indices: list[int] = []
    lock = threading.Lock()

    class RemovalPipeline:
        def __init__(self, hw=None):
            pass

        def cancel(self):
            pass

        def process_video(self, job_config, progress_callback=None):
            idx = int(job_config.input_file.split("_")[1].split(".")[0])
            worker_running.set()
            time.sleep(0.05)
            with lock:
                executed_indices.append(idx)
            return True, "Done"

    with patch("src.core.batch.VideoUpscalerPipeline", RemovalPipeline):
        def run():
            bm.run_batch(str(tmp_path), str(tmp_path), max_workers=2)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        assert worker_running.wait(timeout=2.0)

        # Remove items 4 and 5 dynamically
        bm.remove_items_by_indices({4, 5})

        t.join(timeout=4.0)
        assert not t.is_alive()

        with lock:
            assert 4 not in executed_indices, "Removed item 4 was executed!"
            assert 5 not in executed_indices, "Removed item 5 was executed!"


def test_upscale_worker_pause_resume_stop_controls(tmp_path):
    """UpscaleWorker QThread wrapper forwards pause, resume, and stop controls to BatchManager."""
    bm = BatchManager(HardwareInfo())
    worker = UpscaleWorker(
        batch_manager=bm,
        output_dir=str(tmp_path),
        base_dir=str(tmp_path),
        target_preset="4k",
        video_codec="hevc_nvenc",
        model_id="auto",
        scene_config=None,
        preserve_folders=False,
        overwrite=True,
        max_workers=4,
    )

    assert worker.is_paused is False
    worker.pause()
    assert worker.is_paused is True
    assert bm.is_paused is True

    worker.resume()
    assert worker.is_paused is False
    assert bm.is_paused is False

    worker.stop()
    assert bm._stop_requested is True
