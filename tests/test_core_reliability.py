import unittest
from unittest.mock import patch

from src.core.ai_models import recommend_model_for_video
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import HardwareInfo
from src.core.scene_renamer import SceneNamingConfig, format_scene_name
from src.core.video_pipeline import validate_encoded_video


class CoreReliabilityTests(unittest.TestCase):
    def test_recommendation_matches_source_resolution(self):
        self.assertEqual(
            recommend_model_for_video(640, 480, "4k").model.id,
            "realesrgan-x4plus-anime",
        )
        recommendation = recommend_model_for_video(1280, 720, "4k")
        self.assertEqual(recommendation.model.id, "realesr-animevideov3")
        self.assertEqual(recommendation.scale, 3)
        self.assertTrue(recommendation.reason)

    def test_scene_name_removes_internal_markers(self):
        name = format_scene_name(
            "S01E01 AI Upscaled.mkv",
            config=SceneNamingConfig(show_name="My AI Show", group_tag="MyUpscaler"),
        )
        self.assertNotIn("ai", name.lower())
        self.assertNotIn("upscaler", name.lower())

    def test_pipeline_exception_is_reported_as_a_file_failure(self):
        manager = BatchManager(HardwareInfo())
        item = QueueItem(1, "input.mp4", "input.mp4", "output.mp4", "output.mp4")
        manager.queue = [item]

        class ExplodingPipeline:
            def __init__(self, _hardware):
                pass

            def process_video(self, _job, progress_callback=None):
                raise RuntimeError("engine unavailable")

            def cancel(self):
                pass

        with patch("src.core.batch_manager.VideoUpscalerPipeline", ExplodingPipeline):
            manager.run_batch("out", ".", max_workers=1)

        self.assertFalse(manager.is_running)
        self.assertEqual(item.status, EpisodeStatus.FAILED)
        self.assertIn("engine unavailable", item.error_message)

    def test_encoded_output_without_video_stream_is_rejected(self):
        with self.subTest("missing output"):
            valid, message = validate_encoded_video("does-not-exist.mp4", 3840, 2160)
            self.assertFalse(valid)
            self.assertIn("did not create", message)


    def test_aspect_ratio_preservation_all_formats(self):
        from src.core.ai_models import calculate_target_dimensions

        # Standard 16:9
        self.assertEqual(calculate_target_dimensions(1920, 1080, "4k"), (3840, 2160, "2160p"))
        # Classic 4:3 pillarbox
        self.assertEqual(calculate_target_dimensions(640, 480, "4k"), (2880, 2160, "2160p"))
        # Ultrawide Cinemascope 21:9 (1920x800) -> should NOT stretch to 3840x2160!
        w_21_9, h_21_9, _ = calculate_target_dimensions(1920, 800, "4k")
        self.assertEqual(w_21_9, 3840)
        self.assertEqual(h_21_9, 1600)
        self.assertEqual(w_21_9 % 2, 0)
        self.assertEqual(h_21_9 % 2, 0)
        # Vertical video 9:16 (1080x1920)
        w_vert, h_vert, _ = calculate_target_dimensions(1080, 1920, "4k")
        self.assertEqual(h_vert, 2160)
        self.assertEqual(w_vert, 1214)
        self.assertEqual(w_vert % 2, 0)
        # Square 1:1 (720x720)
        self.assertEqual(calculate_target_dimensions(720, 720, "4k"), (2160, 2160, "2160p"))

    def test_batch_manager_add_files_and_clear_queue(self):
        import tempfile
        manager = BatchManager(HardwareInfo())
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f2:
            f1.write(b"video1")
            f2.write(b"video2")
            p1, p2 = f1.name, f2.name

        try:
            cfg = SceneNamingConfig(naming_mode="clean")
            added = manager.add_files([p1, p2], "C:/Output", cfg)
            self.assertEqual(len(added), 2)
            self.assertEqual(len(manager.queue), 2)
            self.assertEqual(manager.queue[0].index, 1)
            self.assertEqual(manager.queue[1].index, 2)
            self.assertTrue(manager.queue[0].target_scene_name.endswith("_4K.mp4"))

            # Remove first item
            manager.remove_items_by_indices([1])
            self.assertEqual(len(manager.queue), 1)
            self.assertEqual(manager.queue[0].index, 1)  # re-indexed to 1

            # Clear queue
            manager.clear_queue()
            self.assertEqual(len(manager.queue), 0)
        finally:
            import os
            for p in (p1, p2):
                if os.path.exists(p):
                    os.remove(p)

    def test_heavy_model_tile_and_thread_safety(self):
        from src.core.models import is_heavy_model, auto_tile_size, get_optimal_thread_spec, AVAILABLE_MODELS

        # 1. Model weight detection
        self.assertTrue(is_heavy_model("realesrgan-x4plus-anime"))
        self.assertTrue(is_heavy_model(AVAILABLE_MODELS["realesrgan-x4plus-anime"]))
        self.assertTrue(is_heavy_model("realesrgan-x4plus"))
        self.assertFalse(is_heavy_model("realesr-animevideov3"))
        self.assertFalse(is_heavy_model(AVAILABLE_MODELS["realesr-animevideov3"]))

        # 2. Tile size safety for 8GB GPU (e.g. RTX 3070)
        tile_heavy = auto_tile_size(gpu_vram_gb=8.0, model="realesrgan-x4plus-anime")
        tile_light = auto_tile_size(gpu_vram_gb=8.0, model="realesr-animevideov3")
        self.assertEqual(tile_heavy, 200)
        self.assertEqual(tile_light, 400)

        # 3. Thread spec safety
        spec_heavy = get_optimal_thread_spec("auto", gpu_vram_gb=8.0, cpu_cores=8, model="realesrgan-x4plus-anime")
        spec_light = get_optimal_thread_spec("auto", gpu_vram_gb=8.0, cpu_cores=8, model="realesr-animevideov3")
        self.assertEqual(spec_heavy, "3:4:3")
        self.assertEqual(spec_light, "3:6:3")


if __name__ == "__main__":
    unittest.main()
