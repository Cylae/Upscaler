"""Compare orchestration and lossless I/O against a Git revision on synthetic data.

Run from the repository with its configured Python. Optional --render exercises
the installed Vulkan engine and compares decoded final video and audio hashes.
No user media or desktop settings are read or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from src.core import batch, probe, video_pipeline
from src.core.hardware import HardwareInfo, get_hardware_info


def baseline_module(ref, path):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True, encoding="utf-8")
    name = "benchmark_baseline_" + Path(path).stem
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def timed(callback, rounds):
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - start)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def comparison(before, after):
    return {"before": before, "after": after,
            "speedup": before["median_seconds"] / after["median_seconds"]}


def frame_hashes(ffmpeg, path, audio=False):
    args = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0" if audio else "0:v:0"]
    result = subprocess.check_output(args + ["-f", "framemd5", "-"], text=True, timeout=120)
    return [line for line in result.splitlines() if not line.startswith("#")]


def media_benchmarks(old_probe, old_pipeline, scratch, rounds, render):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        return {"status": "NOT AVAILABLE: FFmpeg and FFprobe are required"}
    source = scratch / "source.mkv"
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=1920x1080:rate=24:duration=1,noise=alls=20:allf=t+u:all_seed=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "ffv1", "-c:a", "flac", str(source),
    ], check=True, capture_output=True, timeout=120)
    measurements = []
    snapshots = []
    for module in (old_probe, probe):
        def inspect_media():
            module.clear_probe_cache()
            video = module.probe_video(str(source))
            tracks = module.probe_audio_streams(str(source))
            clean = module.verify_clean_media_metadata(str(source))
            return (video.width, video.height, video.fps, video.duration,
                    [track["codec_name"] for track in tracks], clean)
        snapshots.append(inspect_media())
        measurements.append(timed(inspect_media, rounds))
    assert snapshots[0] == snapshots[1]
    results = {"probe_video_audio_tags": comparison(*measurements)}
    measurements = []
    sequences = []
    for label, module in (("before", old_pipeline), ("after", video_pipeline)):
        folder = scratch / label
        folder.mkdir()
        pipeline = module.VideoUpscalerPipeline(HardwareInfo(ffmpeg_path=ffmpeg))
        meta = probe.probe_video(str(source))
        cmd = pipeline._build_extract_frames_cmd(str(source), str(folder), "png", False, meta=meta)
        measurements.append(timed(lambda: subprocess.run(cmd, check=True, capture_output=True, timeout=120), rounds))
        measurements[-1]["intermediate_bytes"] = sum(path.stat().st_size for path in folder.iterdir())
        sequences.append(frame_hashes(ffmpeg, folder / "frame_%08d.png"))
    assert sequences[0] == sequences[1]
    results["png_detailed_1080p"] = comparison(*measurements)
    results["png_detailed_1080p"]["identical_decoded_frames"] = len(sequences[0])
    if render:
        hw = get_hardware_info()
        if not hw.realesrgan_bin:
            results["render"] = {"status": "NOT AVAILABLE: local Real-ESRGAN required"}
            return results
        tiny = scratch / "tiny.mkv"
        subprocess.run([
            ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=4:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v", "ffv1", "-c:a", "aac", str(tiny),
        ], check=True, capture_output=True, timeout=30)
        rendered = []
        for label, module in (("before", old_pipeline), ("after", video_pipeline)):
            pipeline = module.VideoUpscalerPipeline(hw)
            # CPU decode makes the pixel comparison independent of NVDEC availability.
            pipeline._cuda_available = False
            job = module.UpscaleJobConfig(
                str(tiny), str(scratch / (label + "_render")), target_preset="720p",
                video_codec="libx264", model_id="realesr-animevideov3", frame_format="png",
                threads="2:2:2", tile_size=200, overwrite=True,
            )
            start = time.perf_counter()
            success, output = pipeline.process_video(job)
            if not success:
                raise RuntimeError(f"{label} full render failed: {output}")
            rendered.append({"seconds": time.perf_counter() - start,
                             "video": frame_hashes(ffmpeg, output), "audio": frame_hashes(ffmpeg, output, audio=True)})
        assert rendered[0]["video"] == rendered[1]["video"]
        assert rendered[0]["audio"] == rendered[1]["audio"]
        results["render"] = {"before_seconds": rendered[0]["seconds"], "after_seconds": rendered[1]["seconds"],
                             "identical_video_frames": len(rendered[0]["video"]),
                             "identical_audio_blocks": len(rendered[0]["audio"]),
                             "note": "One tiny synthetic render; not a general GPU-throughput benchmark."}
    return results


def batch_benchmark(old_batch, rounds):
    class SyntheticPipeline:
        def __init__(self, _hardware):
            pass

        def process_video(self, _job, progress_callback):
            for number in range(1, 501):
                progress_callback(number / 500, "Synthetic progress", {})
            return True, "complete"

    measurements = []
    for module in (old_batch, batch):
        def run():
            manager = module.BatchManager(HardwareInfo())
            manager.queue = [module.QueueItem(
                index, "source.mp4", "source.mp4", "output.mp4", "output.mp4",
                status=module.EpisodeStatus.PENDING if index == 1 else module.EpisodeStatus.SKIPPED,
            ) for index in range(1, 10_001)]
            with patch("src.core.batch_manager.VideoUpscalerPipeline", SyntheticPipeline):
                manager.run_batch("unused", ".", batch_callback=lambda *args: None)
            assert manager.queue[0].status == module.EpisodeStatus.COMPLETED
        measurements.append(timed(run, rounds))
    return comparison(*measurements)


def inventory_benchmark(scratch, rounds):
    folder = scratch / "inventory"
    folder.mkdir()
    for index in range(1, 10_001):
        (folder / f"frame_{index:08d}.png").write_bytes(b"frame")

    def previous_inventory():
        return sorted(name for name in os.listdir(folder)
                      if name.startswith("frame_") and name.endswith(".png")
                      and os.path.getsize(folder / name) > 0)

    def current_inventory():
        return video_pipeline._list_nonempty_frames(str(folder), "png")

    assert previous_inventory() == current_inventory()
    return comparison(timed(previous_inventory, rounds), timed(current_inventory, rounds))


def ui_benchmarks(old_ui, scratch, rounds):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    from src.ui import main_window
    app = QApplication.instance() or QApplication([])
    results = {}
    for label, module in (("before", old_ui), ("after", main_window)):
        def settings(*_args):
            result = QSettings(str(scratch / (label + ".ini")), QSettings.Format.IniFormat)
            result.setFallbacksEnabled(False)
            return result
        with patch.object(module, "get_hardware_info", lambda: HardwareInfo()), \
             patch.object(module, "QSettings", settings), \
             patch.object(module, "cleanup_stale_temp_directories", lambda: 0), \
             patch.object(module.UpscalerMainWindow, "on_scan_clicked", lambda self: None):
            window = module.UpscalerMainWindow()
            queue = [batch.QueueItem(index, f"episode{index}.mp4", f"episode{index}.mp4",
                                    f"output{index}.mp4", f"output{index}.mp4") for index in range(1, 1001)]
            window.batch_manager.queue = queue
            window._populate_table(queue)
            # Old and new modules share the public enum; use the UI's normal status.
            queue[-1].status = batch.EpisodeStatus.PROCESSING
            results.setdefault("filter_1000_rows_10_passes", {})[label] = timed(
                lambda: [window.apply_table_filters() for _ in range(10)], rounds)
            results.setdefault("progress_1000_rows_200_events", {})[label] = timed(
                lambda: [window.on_item_updated(queue[-1]) for _ in range(200)], rounds)
            window._open_log_viewer()
            def log_messages():
                window.clear_all_logs()
                for index in range(500):
                    window.on_log_received(f"[VULKAN AI] synthetic event {index}")
            results.setdefault("live_logs_500_events", {})[label] = timed(log_messages, rounds)
            results["live_logs_500_events"][label]["stored_records"] = len(window._log_buffer)
            window.close()
            window.deleteLater()
            app.processEvents()
    for metric, values in results.items():
        values["speedup"] = values["before"]["median_seconds"] / values["after"]["median_seconds"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="2c03d35")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    old_probe = baseline_module(args.baseline_ref, "src/core/probe.py")
    old_pipeline = baseline_module(args.baseline_ref, "src/core/video_pipeline.py")
    old_batch = baseline_module(args.baseline_ref, "src/core/batch.py")
    old_ui = baseline_module(args.baseline_ref, "src/ui/main_window.py")
    # Baseline logs must also use the previous dialog implementation.
    old_ui.LogViewerDialog = baseline_module(args.baseline_ref, "src/ui/dialogs/logs.py").LogViewerDialog
    with tempfile.TemporaryDirectory(prefix="upscaler-efficiency-") as directory:
        scratch = Path(directory)
        results = {"baseline_ref": args.baseline_ref, "rounds": args.rounds,
                   "python": sys.version.split()[0], "cpu_cores": os.cpu_count(),
                   "batch_10000_items_500_progress_events": batch_benchmark(old_batch, args.rounds),
                   "frame_inventory_10000_files": inventory_benchmark(scratch, args.rounds),
                   "ui": ui_benchmarks(old_ui, scratch, args.rounds),
                   "media": media_benchmarks(old_probe, old_pipeline, scratch, args.rounds, args.render)}
    serialized = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
