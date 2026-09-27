"""Batch queue management, concurrency scheduling, state persistence, and preflight planning.

Manages video job state transitions, thread-safe queue updates, multi-worker
parallel execution, session recovery manifests, and dry-run collision detection.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.hardware import HardwareInfo, get_hardware_info
from src.core.models import calculate_target_dimensions
from src.core.pipeline import UpscaleJobConfig, VideoUpscalerPipeline
from src.core.probe import format_eta, probe_video
from src.core.scene_namer import (
    SceneNamingConfig,
    format_output_name,
    format_scene_name,
    get_scene_relative_output_path,
)
from src.core.storage import cleanup_stale_temp_directories
from src.core.validator import is_file_complete_and_valid


class EpisodeStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    SKIPPED = "Already Present"
    FAILED = "Error"
    CANCELLED = "Cancelled"


@dataclass
class QueueItem:
    index: int
    source_path: str
    relative_source_path: str
    target_scene_name: str
    target_full_path: str
    file_size_mb: float = 0.0
    output_size_mb: float = 0.0
    status: EpisodeStatus = EpisodeStatus.PENDING
    status_detail: str = "Ready"
    progress: float = 0.0
    resolution_str: str = ""
    duration_str: str = ""
    file_eta: str = ""
    error_message: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class BatchManager:
    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or get_hardware_info()
        self.queue: List[QueueItem] = []
        self._is_running = False
        self._stop_requested = False
        self._active_pipelines: Set[VideoUpscalerPipeline] = set()
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def scan_folder(
        self,
        folder_path: str,
        output_dir: str,
        scene_config: Optional[SceneNamingConfig] = None,
        preserve_folders: bool = True
    ) -> List[QueueItem]:
        """Scans a directory recursively for supported video files and builds the batch queue."""
        self.queue.clear()
        cfg = scene_config or SceneNamingConfig()

        cleanup_stale_temp_directories(output_dir)

        valid_exts = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".flv", ".ts", ".wmv"}
        found_files: List[str] = []

        if os.path.isfile(folder_path):
            found_files.append(folder_path)
            base_dir = os.path.dirname(folder_path)
        else:
            base_dir = folder_path
            output_abs = os.path.normcase(os.path.abspath(output_dir))
            for root, dirs, files in os.walk(folder_path):
                dirs[:] = sorted(
                    directory for directory in dirs
                    if directory != ".temp_upscaler"
                    and os.path.normcase(os.path.abspath(os.path.join(root, directory))) != output_abs
                )
                for f in sorted(files, key=str.casefold):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        found_files.append(os.path.join(root, f))

        for idx, fpath in enumerate(found_files, start=1):
            rel_path = os.path.relpath(fpath, base_dir)
            target_name = format_output_name(fpath, base_dir, cfg)
            rel_out = get_scene_relative_output_path(fpath, base_dir, cfg, preserve_folders=preserve_folders)
            target_full = os.path.join(output_dir, rel_out)

            size_mb = round(os.path.getsize(fpath) / (1024**2), 1) if os.path.exists(fpath) else 0.0

            status = EpisodeStatus.PENDING
            status_detail = "Ready"
            out_size_mb = 0.0
            expect_audio = False
            expected_dur = None
            if os.path.isfile(target_full):
                try:
                    src_meta = probe_video(fpath, self.hardware.ffprobe_path)
                    expect_audio = src_meta.has_audio
                    expected_dur = src_meta.duration
                except Exception:
                    expect_audio = False
            if is_file_complete_and_valid(target_full, expect_audio=expect_audio, expected_duration=expected_dur):
                status = EpisodeStatus.SKIPPED
                status_detail = "Already present (will be skipped)"
                out_size_mb = round(os.path.getsize(target_full) / (1024**2), 1)

            item = QueueItem(
                index=idx,
                source_path=fpath,
                relative_source_path=rel_path,
                target_scene_name=target_name,
                target_full_path=target_full,
                file_size_mb=size_mb,
                output_size_mb=out_size_mb,
                status=status,
                status_detail=status_detail,
            )
            self.queue.append(item)

        return self.queue

    def add_files(
        self,
        file_paths: List[str],
        output_dir: str,
        scene_config: Optional[SceneNamingConfig] = None,
        preserve_folders: bool = False
    ) -> List[QueueItem]:
        """Appends individual files to the queue without clearing existing items."""
        cfg = scene_config or SceneNamingConfig()
        valid_exts = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".flv", ".ts", ".wmv"}
        existing_sources = {it.source_path for it in self.queue}

        start_idx = len(self.queue) + 1
        for fpath in file_paths:
            if fpath in existing_sources or not os.path.isfile(fpath):
                continue
            if os.path.splitext(fpath)[1].lower() not in valid_exts:
                continue

            base_dir = os.path.dirname(fpath)
            rel_path = os.path.basename(fpath)
            target_name = format_output_name(fpath, base_dir, cfg)
            rel_out = get_scene_relative_output_path(fpath, base_dir, cfg, preserve_folders=preserve_folders)
            target_full = os.path.join(output_dir, rel_out)

            size_mb = round(os.path.getsize(fpath) / (1024**2), 1) if os.path.exists(fpath) else 0.0

            status = EpisodeStatus.PENDING
            status_detail = "Ready"
            out_size_mb = 0.0
            expect_audio = False
            expected_dur = None
            if os.path.isfile(target_full):
                try:
                    src_meta = probe_video(fpath, self.hardware.ffprobe_path)
                    expect_audio = src_meta.has_audio
                    expected_dur = src_meta.duration
                except Exception:
                    expect_audio = False
            if is_file_complete_and_valid(target_full, expect_audio=expect_audio, expected_duration=expected_dur):
                status = EpisodeStatus.SKIPPED
                status_detail = "Already present (will be skipped)"
                out_size_mb = round(os.path.getsize(target_full) / (1024**2), 1)

            item = QueueItem(
                index=start_idx,
                source_path=fpath,
                relative_source_path=rel_path,
                target_scene_name=target_name,
                target_full_path=target_full,
                file_size_mb=size_mb,
                output_size_mb=out_size_mb,
                status=status,
                status_detail=status_detail,
            )
            self.queue.append(item)
            existing_sources.add(fpath)
            start_idx += 1

        self.reindex_queue()
        return self.queue

    def reindex_queue(self):
        """Re-indexes queue items 1..N sequentially."""
        for idx, it in enumerate(self.queue, start=1):
            it.index = idx

    def remove_items_by_indices(self, indices: Set[int]):
        """Removes queue items whose indices are in the set and reindexes."""
        self.queue = [it for it in self.queue if it.index not in indices]
        self.reindex_queue()

    def clear_queue(self):
        """Clears all items in queue."""
        self.queue.clear()

    def retry_failed_items(self) -> int:
        """Resets failed or cancelled items back to PENDING so they can be re-run."""
        count = 0
        for item in self.queue:
            if item.status in (EpisodeStatus.FAILED, EpisodeStatus.CANCELLED):
                item.status = EpisodeStatus.PENDING
                item.progress = 0.0
                item.status_detail = "Ready (Retry)"
                item.error_message = ""
                item.file_eta = ""
                count += 1
        return count

    def save_manifest(self, filepath: str) -> bool:
        """Saves current queue state to a JSON manifest for session recovery."""
        try:
            manifest_data = {
                "version": 1,
                "timestamp": time.time(),
                "items": [
                    {
                        "id": item.id,
                        "index": item.index,
                        "source_path": item.source_path,
                        "relative_source_path": item.relative_source_path,
                        "target_scene_name": item.target_scene_name,
                        "target_full_path": item.target_full_path,
                        "resolution_str": item.resolution_str,
                        "duration_str": item.duration_str,
                        "status": item.status.value,
                        "progress": item.progress,
                        "status_detail": item.status_detail,
                        "file_size_mb": item.file_size_mb,
                        "output_size_mb": item.output_size_mb,
                        "error_message": item.error_message,
                    }
                    for item in self.queue
                ]
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            return True
        except Exception:
            return False

    def load_manifest(self, filepath: str) -> int:
        """Restores queue items from a saved JSON manifest. Returns count of loaded items."""
        if not os.path.isfile(filepath):
            return 0
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_items = data.get("items", [])
        self.queue.clear()
        for raw in raw_items:
            status_val = raw.get("status", "Pending")
            try:
                status = EpisodeStatus(status_val)
            except ValueError:
                status = EpisodeStatus.PENDING

            item = QueueItem(
                index=raw.get("index", len(self.queue) + 1),
                source_path=raw.get("source_path", ""),
                relative_source_path=raw.get("relative_source_path", ""),
                target_scene_name=raw.get("target_scene_name", ""),
                target_full_path=raw.get("target_full_path", ""),
                resolution_str=raw.get("resolution_str", ""),
                duration_str=raw.get("duration_str", ""),
                status=status,
                progress=raw.get("progress", 0.0),
                status_detail=raw.get("status_detail", "Pending"),
                file_size_mb=raw.get("file_size_mb", 0.0),
                output_size_mb=raw.get("output_size_mb", 0.0),
                error_message=raw.get("error_message", ""),
                id=raw.get("id", uuid.uuid4().hex),
            )
            self.queue.append(item)
        self.reindex_queue()
        return len(self.queue)

    def plan_dry_run(
        self,
        output_dir: str,
        scene_config: Optional[SceneNamingConfig] = None,
        preserve_folders: bool = True
    ) -> List[Dict[str, Any]]:
        """Generates dry-run source-to-output mapping without modifying files."""
        results: List[Dict[str, Any]] = []
        target_counts: Dict[str, int] = {}
        for item in self.queue:
            target_counts[item.target_full_path] = target_counts.get(item.target_full_path, 0) + 1

        for item in self.queue:
            collision = target_counts[item.target_full_path] > 1

            results.append({
                "index": item.index,
                "id": item.id,
                "source_path": item.source_path,
                "relative_source": item.relative_source_path,
                "target_name": item.target_scene_name,
                "target_path": item.target_full_path,
                "status": item.status.value,
                "source_size_mb": item.file_size_mb,
                "already_exists": os.path.isfile(item.target_full_path),
                "collision": collision,
            })
        return results

    def get_preflight_summary(
        self,
        output_dir: str,
        target_preset: str = "4k",
        video_codec: str = "hevc_nvenc",
        model_id: str = "auto",
        cq_level: int = 19
    ) -> Dict[str, Any]:
        """Generates a complete preflight summary before starting batch processing."""
        total_items = len(self.queue)
        pending_items = sum(1 for it in self.queue if it.status == EpisodeStatus.PENDING)
        skipped_items = sum(1 for it in self.queue if it.status == EpisodeStatus.SKIPPED)
        failed_items = sum(1 for it in self.queue if it.status == EpisodeStatus.FAILED)
        cancelled_items = sum(1 for it in self.queue if it.status == EpisodeStatus.CANCELLED)
        completed_items = sum(1 for it in self.queue if it.status == EpisodeStatus.COMPLETED)

        total_source_size_mb = sum(it.file_size_mb for it in self.queue)
        mult = 2.5 if target_preset == "4k" else (1.8 if target_preset == "1440p" else 1.2)
        pending_source_size_mb = sum(it.file_size_mb for it in self.queue if it.status == EpisodeStatus.PENDING)
        estimated_output_mb = round(pending_source_size_mb * mult, 1)

        free_disk_mb = 0.0
        try:
            target_check_dir = output_dir if os.path.isdir(output_dir) else os.path.dirname(os.path.abspath(output_dir))
            if os.path.isdir(target_check_dir):
                usage = shutil.disk_usage(target_check_dir)
                free_disk_mb = round(usage.free / (1024**2), 1)
        except Exception:
            pass

        warnings: List[str] = []
        if total_items == 0:
            warnings.append("Queue is empty. Please add source videos.")
        if pending_items > 0 and free_disk_mb > 0 and free_disk_mb < estimated_output_mb:
            warnings.append(
                f"Low disk space on target volume: {free_disk_mb:.0f} MB free, "
                f"estimated {estimated_output_mb:.0f} MB needed."
            )

        target_paths: Dict[str, List[str]] = {}
        for it in self.queue:
            target_paths.setdefault(it.target_full_path, []).append(it.source_path)
        for target, sources in target_paths.items():
            if len(sources) > 1:
                warnings.append(
                    f"Destination collision detected: {len(sources)} files map to '{os.path.basename(target)}'."
                )

        if "hevc_nvenc" in video_codec and not getattr(self.hardware, "has_nvenc_hevc", False):
            warnings.append(f"NVENC HEVC codec requested ('{video_codec}') but HEVC NVENC is not available on this system. Software fallback will be used.")
        elif "h264_nvenc" in video_codec and not getattr(self.hardware, "has_nvenc_h264", False):
            warnings.append(f"NVENC H.264 codec requested ('{video_codec}') but H.264 NVENC is not available on this system. Software fallback will be used.")
        elif "nvenc" in video_codec and not getattr(self.hardware, "nvenc_available", False):
            warnings.append(f"NVENC codec requested ('{video_codec}') but NVENC hardware acceleration is not available on this system. Software fallback will be used.")

        return {
            "total_items": total_items,
            "pending_items": pending_items,
            "skipped_items": skipped_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "cancelled_items": cancelled_items,
            "total_source_size_mb": round(total_source_size_mb, 1),
            "estimated_output_mb": estimated_output_mb,
            "free_disk_mb": free_disk_mb,
            "target_preset": target_preset,
            "video_codec": video_codec,
            "model_id": model_id,
            "cq_level": cq_level,
            "gpu_name": self.hardware.gpu_name or "CPU / Generic",
            "warnings": warnings,
        }

    def stop(self):
        """Requests cancellation of all active pipelines and stops the batch."""
        self._stop_requested = True
        with self._lock:
            for pipeline in list(self._active_pipelines):
                pipeline.cancel()

    def run_batch(
        self,
        output_dir: str,
        base_dir: str,
        target_preset: str = "4k",
        video_codec: str = "hevc_nvenc",
        model_id: str = "auto",
        scene_config: Optional[SceneNamingConfig] = None,
        preserve_folders: bool = True,
        overwrite: bool = False,
        threads: str = "auto",
        frame_format: str = "jpg",
        max_workers: int = 1,
        cq_level: int = 19,
        nvenc_preset: str = "p7",
        enable_tta: bool = False,
        denoise_level: int = -1,
        audio_mode: str = "copy",
        tile_size: int = 0,
        target_indices: Optional[List[int]] = None,
        preview_duration_s: Optional[float] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        item_callback: Optional[Callable[[QueueItem], None]] = None,
        batch_callback: Optional[Callable[..., None]] = None
    ):
        """Executes batch processing on queue with optional multi-worker concurrency."""
        self._is_running = True
        self._stop_requested = False

        if target_indices:
            items_to_run = [item for item in self.queue if item.index in target_indices]
        else:
            items_to_run = list(self.queue)

        total_items = len(items_to_run)
        completed_items = [0]
        t_batch_start = time.time()

        def process_single_item(item: QueueItem):
            if self._stop_requested:
                item.status = EpisodeStatus.CANCELLED
                item.status_detail = "Cancelled by user"
                item.file_eta = ""
                with self._lock:
                    completed_items[0] += 1
                if item_callback:
                    item_callback(item)
                return

            if item.status == EpisodeStatus.SKIPPED and not overwrite:
                with self._lock:
                    completed_items[0] += 1
                    pct = (completed_items[0] / total_items) * 100.0 if total_items > 0 else 0.0
                if item_callback:
                    item_callback(item)
                if batch_callback:
                    try:
                        batch_callback(completed_items[0], total_items, pct, f"Episode {item.index}/{total_items} already present", "")
                    except TypeError:
                        batch_callback(completed_items[0], total_items, pct, f"Episode {item.index}/{total_items} already present")
                return

            item.status = EpisodeStatus.PROCESSING
            item.progress = 0.05
            item.status_detail = "Starting..."
            item.file_eta = "Calculating..."
            if item_callback:
                item_callback(item)

            pipeline_cls = VideoUpscalerPipeline
            job_cls = UpscaleJobConfig
            try:
                import sys
                bm = sys.modules.get("src.core.batch_manager")
                if bm and hasattr(bm, "VideoUpscalerPipeline"):
                    pipeline_cls = bm.VideoUpscalerPipeline
                if bm and hasattr(bm, "UpscaleJobConfig"):
                    job_cls = bm.UpscaleJobConfig
            except Exception:
                pass

            pipeline = pipeline_cls(self.hardware)
            with self._lock:
                self._active_pipelines.add(pipeline)

            job = job_cls(
                input_file=item.source_path,
                output_dir=output_dir,
                base_dir=base_dir,
                target_preset=target_preset,
                video_codec=video_codec,
                model_id=model_id,
                scene_config=scene_config,
                preserve_folders=preserve_folders,
                overwrite=overwrite,
                threads=threads,
                frame_format=frame_format,
                cq_level=cq_level,
                nvenc_preset=nvenc_preset,
                enable_tta=enable_tta,
                denoise_level=denoise_level,
                audio_mode=audio_mode,
                tile_size=tile_size,
                log_callback=log_callback,
                preview_duration_s=preview_duration_s,
            )

            def on_progress(ratio: float, status_text: str, extra: Dict):
                item.progress = ratio
                item.status_detail = status_text
                item.file_eta = extra.get("file_eta_str", "")

                if item_callback:
                    item_callback(item)

                if batch_callback:
                    with self._lock:
                        cur_completed = completed_items[0]
                        active_progress = sum(
                            it.progress for it in items_to_run
                            if it.status == EpisodeStatus.PROCESSING
                        )
                        effective_done = cur_completed + active_progress

                    pct = (effective_done / total_items) * 100.0 if total_items > 0 else 0.0
                    batch_elapsed = max(0.1, time.time() - t_batch_start)

                    batch_eta_str = ""
                    if effective_done >= 0.05 and total_items > 0:
                        rate = effective_done / batch_elapsed
                        rem_files = max(0.0, total_items - effective_done)
                        batch_eta_sec = rem_files / max(0.0001, rate)
                        batch_eta_str = format_eta(batch_eta_sec)

                    parts = []
                    if item.file_eta and item.file_eta not in ("--", "Calculating..."):
                        parts.append(f"File ETA: {item.file_eta}")
                    elif item.file_eta == "Calculating...":
                        parts.append("File ETA: calculating...")
                    if batch_eta_str and batch_eta_str != "--":
                        parts.append(f"Batch ETA: {batch_eta_str}")
                    combined_eta = "  |  ".join(parts)

                    try:
                        batch_callback(cur_completed, total_items, pct, f"Episode {item.index}/{total_items}: {status_text}", combined_eta)
                    except TypeError:
                        batch_callback(cur_completed, total_items, pct, f"Episode {item.index}/{total_items}: {status_text}")

            try:
                success, msg = pipeline.process_video(job, progress_callback=on_progress)
            except Exception as exc:
                success, msg = False, f"Unexpected pipeline error: {type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self._active_pipelines.discard(pipeline)

            if success:
                if "Already completed" in msg:
                    item.status = EpisodeStatus.SKIPPED
                    item.status_detail = "Already present"
                elif preview_duration_s is not None:
                    item.status = EpisodeStatus.PENDING
                    item.status_detail = f"Preview completed: {os.path.basename(msg)}"
                else:
                    item.status = EpisodeStatus.COMPLETED
                    if not item.status_detail.startswith("Completed in"):
                        item.status_detail = "Completed successfully"
                    if os.path.exists(item.target_full_path):
                        item.output_size_mb = round(os.path.getsize(item.target_full_path) / (1024**2), 1)
                item.progress = 1.0
                item.file_eta = ""
                with self._lock:
                    completed_items[0] += 1
            else:
                item.file_eta = ""
                if self._stop_requested:
                    item.status = EpisodeStatus.CANCELLED
                    item.status_detail = "Cancelled"
                else:
                    item.status = EpisodeStatus.FAILED
                    item.status_detail = f"Error: {msg}"
                    item.error_message = msg
                with self._lock:
                    completed_items[0] += 1

            if item_callback:
                item_callback(item)

            if batch_callback:
                with self._lock:
                    cur_completed = completed_items[0]
                pct = (cur_completed / total_items) * 100.0 if total_items > 0 else 0.0
                batch_elapsed = max(0.1, time.time() - t_batch_start)
                batch_eta_str = ""
                if cur_completed > 0 and total_items > 0:
                    rate = cur_completed / batch_elapsed
                    rem_files = max(0, total_items - cur_completed)
                    batch_eta_sec = rem_files / max(0.0001, rate)
                    batch_eta_str = f"Batch ETA: {format_eta(batch_eta_sec)}"
                try:
                    batch_callback(cur_completed, total_items, pct, f"Overall progress: {cur_completed}/{total_items} completed", batch_eta_str)
                except TypeError:
                    batch_callback(cur_completed, total_items, pct, f"Overall progress: {cur_completed}/{total_items} completed")

        try:
            effective_workers = max(1, min(max_workers, 4))
            if effective_workers == 1:
                for item in items_to_run:
                    if self._stop_requested:
                        break
                    process_single_item(item)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = [executor.submit(process_single_item, item) for item in items_to_run]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                        if self._stop_requested:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
        finally:
            self._is_running = False
