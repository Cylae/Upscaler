import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

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


class BatchManager:
    def __init__(self, hardware: Optional[HardwareInfo] = None):
        self.hardware = hardware or get_hardware_info()
        self.pipeline = VideoUpscalerPipeline(self.hardware)
        self.queue: List[QueueItem] = []
        self._is_running = False
        self._stop_requested = False

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

            # Pre-check if already completed
            status = EpisodeStatus.PENDING
            status_detail = "Ready"
            out_size_mb = 0.0
            if is_file_complete_and_valid(target_full):
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
        """Requests cancellation of current job and stops batch."""
        self._stop_requested = True
        self.pipeline.cancel()

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
        item_callback: Optional[Callable[[QueueItem], None]] = None,
        batch_callback: Optional[Callable[[int, int, float, str], None]] = None
    ):
        """
        Executes batch processing on queue.
        Calls item_callback when an item updates,
        and batch_callback(current_idx, total_items, overall_pct, overall_status).
        """
        self._is_running = True
        self._stop_requested = False

        total_items = len(self.queue)
        completed_count = sum(1 for item in self.queue if item.status == EpisodeStatus.SKIPPED)

        t_start_batch = time.time()

        for idx, item in enumerate(self.queue, start=1):
            if self._stop_requested:
                item.status = EpisodeStatus.CANCELLED
                item.status_detail = "Cancelled by user"
                if item_callback:
                    item_callback(item)
                break

            # If already skipped and not overwriting, move to next
            if item.status == EpisodeStatus.SKIPPED and not overwrite:
                if item_callback:
                    item_callback(item)
                if batch_callback:
                    overall_pct = (idx / total_items) * 100.0
                    batch_callback(idx, total_items, overall_pct, f"Episode {idx}/{total_items} already present")
                continue

            item.status = EpisodeStatus.PROCESSING
            item.progress = 0.05
            item.status_detail = "Starting..."
            if item_callback:
                item_callback(item)

            job = UpscaleJobConfig(
                input_file=item.source_path,
                output_dir=output_dir,
                base_dir=base_dir,
                target_preset=target_preset,
                video_codec=video_codec,
                model_id=model_id,
                scene_config=scene_config,
                preserve_folders=preserve_folders,
                overwrite=overwrite
            )

            def on_progress(ratio: float, status_text: str, extra: Dict):
                item.progress = ratio
                item.status_detail = status_text
                if item_callback:
                    item_callback(item)
                if batch_callback:
                    overall_pct = ((idx - 1 + ratio) / total_items) * 100.0
                    batch_callback(idx, total_items, overall_pct, f"Episode {idx}/{total_items}: {status_text}")

            success, msg = self.pipeline.process_video(job, progress_callback=on_progress)

            if success:
                if "Already completed" in msg:
                    item.status = EpisodeStatus.SKIPPED
                    item.status_detail = "Already present"
                else:
                    item.status = EpisodeStatus.COMPLETED
                    item.status_detail = "Completed successfully"
                    if os.path.exists(item.target_full_path):
                        item.output_size_mb = round(os.path.getsize(item.target_full_path) / (1024**2), 1)
                item.progress = 1.0
                completed_count += 1
            else:
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
                overall_pct = (idx / total_items) * 100.0
                batch_callback(idx, total_items, overall_pct, f"Overall progress: {idx}/{total_items}")

        self._is_running = False
