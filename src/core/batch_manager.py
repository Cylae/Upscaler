import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from src.core.ai_models import AVAILABLE_MODELS, choose_best_model_for_video
from src.core.detector import HardwareInfo, get_hardware_info
from src.core.scene_renamer import (
    SceneNamingConfig,
    format_scene_name,
    get_scene_relative_output_path
)
from src.core.video_pipeline import (
    UpscaleJobConfig,
    VideoMetadata,
    VideoUpscalerPipeline,
    format_eta,
    is_file_complete_and_valid,
    probe_video
)


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
    resolution_str: str = ""
    duration_str: str = ""
    status: EpisodeStatus = EpisodeStatus.PENDING
    progress: float = 0.0
    status_detail: str = "Pending"
    file_size_mb: float = 0.0
    output_size_mb: float = 0.0
    error_message: str = ""
    file_eta: str = ""


class BatchManager:
    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or get_hardware_info()
        self.queue: List[QueueItem] = []
        self._is_running = False
        self._stop_requested = False
        self._lock = threading.Lock()
        self._active_pipelines: Set[VideoUpscalerPipeline] = set()

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
        """Scans input folder recursively and builds queue with Scene names."""
        self.queue.clear()
        cfg = scene_config or SceneNamingConfig()

        valid_exts = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".wmv"}
        found_files: List[str] = []

        if os.path.isfile(folder_path):
            found_files.append(folder_path)
            base_dir = os.path.dirname(folder_path)
        else:
            base_dir = folder_path
            for root, _, files in os.walk(folder_path):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        found_files.append(os.path.join(root, f))

        for idx, fpath in enumerate(found_files, start=1):
            rel_path = os.path.relpath(fpath, base_dir)
            scene_name = format_scene_name(fpath, base_dir, cfg)
            rel_out = get_scene_relative_output_path(fpath, base_dir, cfg, preserve_folders=preserve_folders)
            target_full = os.path.join(output_dir, rel_out)

            size_mb = round(os.path.getsize(fpath) / (1024**2), 1) if os.path.exists(fpath) else 0.0

            # Pre-check if already completed (including audio track verification)
            status = EpisodeStatus.PENDING
            status_detail = "Ready"
            out_size_mb = 0.0
            if is_file_complete_and_valid(target_full, expect_audio=True):
                status = EpisodeStatus.SKIPPED
                status_detail = "Already present (will be skipped)"
                out_size_mb = round(os.path.getsize(target_full) / (1024**2), 1)

            item = QueueItem(
                index=idx,
                source_path=fpath,
                relative_source_path=rel_path,
                target_scene_name=scene_name,
                target_full_path=target_full,
                file_size_mb=size_mb,
                output_size_mb=out_size_mb,
                status=status,
                status_detail=status_detail
            )
            self.queue.append(item)

        return self.queue

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
        batch_callback: Optional[Callable[[int, int, float, str], None]] = None
    ):
        """
        Executes batch processing on queue with optional multi-worker concurrency.
        Supports selective processing via target_indices (e.g. single episode quick test).
        Calls item_callback when an item updates,
        and batch_callback(completed_count, total_items, overall_pct, overall_status).
        """
        self._is_running = True
        self._stop_requested = False

        if target_indices:
            items_to_run = [item for item in self.queue if item.index in target_indices]
        else:
            items_to_run = list(self.queue)

        total_items = len(items_to_run)
        completed_items = [0]  # mutable container for closure
        t_batch_start = time.time()

        def process_single_item(item: QueueItem):
            if self._stop_requested:
                item.status = EpisodeStatus.CANCELLED
                item.status_detail = "Cancelled by user"
                item.file_eta = ""
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

            pipeline = VideoUpscalerPipeline(self.hardware)
            with self._lock:
                self._active_pipelines.add(pipeline)

            job = UpscaleJobConfig(
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
                        active_progress = sum(it.progress for it in self.queue if it.status == EpisodeStatus.PROCESSING)
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
            finally:
                with self._lock:
                    self._active_pipelines.discard(pipeline)

            if success:
                if "Already completed" in msg:
                    item.status = EpisodeStatus.SKIPPED
                    item.status_detail = "Already present"
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

        # Execute using ThreadPoolExecutor for concurrency
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
                    if self._stop_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

        self._is_running = False
