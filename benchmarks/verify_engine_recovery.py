"""Exercise stale engine discovery with real Vulkan renders on synthetic media.

Run with --baseline <revision> and optionally --output <report.json>.
Requires the portable engines/models, FFmpeg/FFprobe and a working Vulkan GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile

from benchmark_efficiency import baseline_module, frame_hashes
from PIL import Image

from src.core.hardware import get_hardware_info
from src.core import video_pipeline


def pixels(path):
    with Image.open(path) as image:
        return image.size, hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def verify(baseline):
    previous = baseline_module(baseline, "src/core/video_pipeline.py")
    hardware = get_hardware_info()
    results = {
        "baseline": baseline,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "gpu": hardware.gpu_name, "ffmpeg": hardware.ffmpeg_version},
        "preview": [], "video": [],
    }
    with tempfile.TemporaryDirectory(prefix="upscaler_engine_recovery_") as scratch:
        root = Path(scratch)
        source = root / "source.mkv"
        subprocess.run([
            hardware.ffmpeg_path or "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=4:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v", "ffv1", "-c:a", "aac", str(source),
        ], check=True, capture_output=True, timeout=30)
        for model, field in (
            ("realesr-animevideov3", "realesrgan_bin"), ("realcugan-pro", "realcugan_bin"),
        ):
            if not getattr(hardware, field):
                raise RuntimeError(f"Install {field} before running this check")
            options = dict(threads="2:2:2", tile_size=100, denoise_level=0, enable_tta=True)
            old = previous.VideoUpscalerPipeline(replace(hardware, **{field: None}))
            failed = old.extract_and_upscale_single_frame(str(source), model, "720p", **options)
            assert not failed[0] and "not found" in failed[3]["error"], failed
            control = previous.VideoUpscalerPipeline(replace(hardware))
            reference = control.extract_and_upscale_single_frame(str(source), model, "720p", **options)
            assert reference[0], reference[3]
            expected_pixels = pixels(reference[2])
            for stale in (None, str(root / "removed-engine.exe")):
                shared = replace(hardware, **{field: stale})
                pipeline = video_pipeline.VideoUpscalerPipeline(shared)
                result = pipeline.extract_and_upscale_single_frame(str(source), model, "720p", **options)
                assert result[0], result[3]
                assert pixels(result[2]) == expected_pixels
                assert getattr(shared, field) == getattr(hardware, field)
                assert not pipeline._active_processes
                results["preview"].append({
                    "model": model, "cached_path": "None" if stale is None else "removed",
                    "baseline_missing_path_error": failed[3]["error"],
                    "success": True, "size": expected_pixels[0],
                    "pixels_equal_to_baseline_with_valid_path": True,
                    "pixel_sha256": expected_pixels[1], "active_processes_after": 0,
                })

            # Compare complete encodes with sound, not just image previews.
            outputs = []
            for index, module in enumerate((previous, video_pipeline)):
                info = replace(hardware, **({field: None} if index else {}))
                pipeline = module.VideoUpscalerPipeline(info)
                pipeline._cuda_available = False
                job = module.UpscaleJobConfig(
                    str(source), str(root / f"{model}_{index}"), model_id=model,
                    target_preset="720p", video_codec="libx264", frame_format="png",
                    audio_mode="copy", overwrite=True, **options,
                )
                ok, output = pipeline.process_video(job)
                assert ok, output
                assert not pipeline._active_processes
                outputs.append(output)
            video = [frame_hashes(hardware.ffmpeg_path, path) for path in outputs]
            audio = [frame_hashes(hardware.ffmpeg_path, path, audio=True) for path in outputs]
            assert video[0] == video[1] and audio[0] == audio[1]
            results["video"].append({
                "model": model, "cached_path": "None", "success": True,
                "equal_decoded_video_frames": len(video[0]),
                "equal_decoded_audio_blocks": len(audio[0]), "active_processes_after": 0,
            })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.baseline), indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
