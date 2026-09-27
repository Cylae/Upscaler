"""Background QThread workers for non-blocking UI operations."""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.batch import BatchManager
from src.core.models import recommend_model_for_video
from src.core.pipeline import VideoUpscalerPipeline
from src.core.probe import probe_video
from src.core.scene_namer import SceneNamingConfig


class FramePreviewWorker(QThread):
    """Generates a Before/After comparison for a single extracted frame in the background."""
    preview_ready = pyqtSignal(bool, str, str, dict)

    def __init__(self, pipeline: VideoUpscalerPipeline, video_path: str, model_id: str, target_preset: str):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.model_id = model_id
        self.target_preset = target_preset

    def run(self):
        try:
            ok, orig, up, info = self.pipeline.extract_and_upscale_single_frame(
                self.video_path,
                model_id=self.model_id,
                target_preset=self.target_preset
            )
            self.preview_ready.emit(ok, orig, up, info)
        except Exception as exc:
            self.preview_ready.emit(False, "", "", {"error": str(exc)})


class ModelHintWorker(QThread):
    """Probes video in background to suggest the optimal AI model and scale factor."""
    hint_ready = pyqtSignal(str, str)

    def __init__(self, source_path: str, target_preset: str):
        super().__init__()
        self.source_path = source_path
        self.target_preset = target_preset

    def run(self):
        try:
            meta = probe_video(self.source_path)
            recommendation = recommend_model_for_video(meta.width, meta.height, self.target_preset)
            text = (
                f"🧠 AI Detection: {recommendation.model.name} ×{recommendation.scale} — "
                "full details will be pending validation before batch."
            )
            self.hint_ready.emit(text, recommendation.reason)
        except Exception:
            self.hint_ready.emit("🧠 AI Detection: full analysis available upon batch launch.", "")


class ScanWorker(QThread):
    """Scans directories asynchronously for media files and builds queue."""
    scan_complete = pyqtSignal(list)
    scan_error = pyqtSignal(str)

    def __init__(
        self,
        batch_manager: BatchManager,
        folder_path: str,
        output_dir: str,
        scene_config: SceneNamingConfig,
        preserve_folders: bool
    ):
        super().__init__()
        self.bm = batch_manager
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.scene_config = scene_config
        self.preserve_folders = preserve_folders

    def run(self):
        try:
            queue = self.bm.scan_folder(
                self.folder_path,
                self.output_dir,
                self.scene_config,
                preserve_folders=self.preserve_folders
            )
            self.scan_complete.emit(queue)
        except Exception as exc:
            self.scan_error.emit(str(exc))


class UpscaleWorker(QThread):
    """Executes batch video upscaling job in a dedicated background worker thread."""
    item_updated = pyqtSignal(object)
    batch_updated = pyqtSignal(int, int, float, str, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        batch_manager: BatchManager,
        output_dir: str,
        base_dir: str,
        target_preset: str,
        video_codec: str,
        model_id: str,
        scene_config: SceneNamingConfig,
        preserve_folders: bool,
        overwrite: bool,
        threads: str = "auto",
        frame_format: str = "jpg",
        max_workers: int = 2,
        cq_level: int = 19,
        nvenc_preset: str = "p7",
        enable_tta: bool = False,
        denoise_level: int = -1,
        audio_mode: str = "copy",
        tile_size: int = 0,
        target_indices: Optional[List[int]] = None,
        preview_duration_s: Optional[float] = None,
    ):
        super().__init__()
        self.bm = batch_manager
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.target_preset = target_preset
        self.video_codec = video_codec
        self.model_id = model_id
        self.scene_config = scene_config
        self.preserve_folders = preserve_folders
        self.overwrite = overwrite
        self.threads = threads
        self.frame_format = frame_format
        self.max_workers = max_workers
        self.cq_level = cq_level
        self.nvenc_preset = nvenc_preset
        self.enable_tta = enable_tta
        self.denoise_level = denoise_level
        self.audio_mode = audio_mode
        self.tile_size = tile_size
        self.target_indices = target_indices
        self.preview_duration_s = preview_duration_s

    def run(self):
        try:
            self.bm.run_batch(
                output_dir=self.output_dir,
                base_dir=self.base_dir,
                target_preset=self.target_preset,
                video_codec=self.video_codec,
                model_id=self.model_id,
                scene_config=self.scene_config,
                preserve_folders=self.preserve_folders,
                overwrite=self.overwrite,
                threads=self.threads,
                frame_format=self.frame_format,
                max_workers=self.max_workers,
                cq_level=self.cq_level,
                nvenc_preset=self.nvenc_preset,
                enable_tta=self.enable_tta,
                denoise_level=self.denoise_level,
                audio_mode=self.audio_mode,
                tile_size=self.tile_size,
                target_indices=self.target_indices,
                preview_duration_s=self.preview_duration_s,
                log_callback=lambda msg: self.log_signal.emit(msg),
                item_callback=lambda item: self.item_updated.emit(item),
                batch_callback=lambda cur, tot, pct, status, eta="": self.batch_updated.emit(cur, tot, pct, status, eta)
            )
        except Exception as exc:
            self.log_signal.emit(f"[ERROR] Batch worker crashed: {type(exc).__name__}: {exc}")
        finally:
            self.finished_signal.emit()
