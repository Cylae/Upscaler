import argparse
import os
import sys
import time

from src.core.ai_models import AVAILABLE_MODELS
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import get_hardware_info, print_doctor_report
from src.core.presets import ENHANCEMENT_PRESETS, get_preset
from src.core.scene_renamer import SceneNamingConfig


def run_cli():
    parser = argparse.ArgumentParser(
        description="4K Video Enhancer for Cartoons & 2D Animation (Scene / Warez Naming)"
    )
    parser.add_argument("--source", "-s", default=r"Z:\DL\Happy Tree Friends", help="Source folder or file path")
    parser.add_argument("--output", "-o", default=r"Z:\DL\Happy Tree Friends (2160p)", help="Output directory")
    parser.add_argument("--preset", "-p", choices=list(ENHANCEMENT_PRESETS.keys()), default=None, help="Enhancement profile preset")
    parser.add_argument("--resolution", "-r", choices=["4k", "1440p", "1080p", "720p"], default=None, help="Target resolution preset")
    parser.add_argument("--codec", "-c", choices=["hevc_nvenc", "h264_nvenc", "libx265", "libx264"], default=None, help="Video codec")
    parser.add_argument("--model", "-m", choices=list(AVAILABLE_MODELS.keys()), default=None, help="Upscaling model")
    parser.add_argument("--cq", type=int, default=None, help="Constant Quality rate control (14-28, default: 19)")
    parser.add_argument("--nvenc-preset", choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], default=None, help="NVENC performance/quality preset")
    parser.add_argument("--tta", action="store_true", help="Enable 8-pass Test-Time Augmentation ensemble inference")
    parser.add_argument("--denoise", type=int, choices=[-1, 0, 1, 2, 3], default=None, help="RealCUGAN denoise level (-1 to 3)")
    parser.add_argument("--audio-mode", choices=["copy", "aac_192k", "aac_320k", "flac"], default=None, help="Audio handling mode (default: copy)")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size in pixels (0 = auto / full VRAM)")
    parser.add_argument("--group", "-g", default="CYLAE", help="Scene release group tag (e.g. CYLAE)")
    parser.add_argument("--source-tag", default="WEB-DL", choices=["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"], help="Scene source tag")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to process (0 = all)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of concurrent workers (default: 2)")
    parser.add_argument("--threads", "-t", default="auto", help="Engine load:proc:save threads (e.g. auto, 4:4:4, 2:4:2)")
    parser.add_argument("--frame-format", "-f", choices=["jpg", "png"], default=None, help="Frame I/O format (default: jpg, 2x-3x faster)")
    parser.add_argument("--doctor", action="store_true", help="Run hardware and environment diagnostic")

    args = parser.parse_args()

    if args.doctor:
        print_doctor_report()
        sys.exit(0)

    # Resolve preset baseline
    chosen_preset = get_preset(args.preset or "cartoon_4k_master")
    res_val = args.resolution or chosen_preset.target_preset
    codec_val = args.codec or chosen_preset.video_codec
    model_val = args.model or chosen_preset.model_id
    cq_val = args.cq if args.cq is not None else chosen_preset.cq_level
    p_val = args.nvenc_preset or chosen_preset.nvenc_preset
    tta_val = args.tta or chosen_preset.enable_tta
    denoise_val = args.denoise if args.denoise is not None else chosen_preset.denoise_level
    audio_val = args.audio_mode or chosen_preset.audio_mode
    workers_val = args.workers if args.workers is not None else chosen_preset.workers
    fmt_val = args.frame_format or chosen_preset.frame_format

    hw = get_hardware_info()
    print("=" * 70)
    print("  VIDEO ENHANCER 4K — ANIMATION & CARTOON EDITION")
    print(f"  GPU         : {hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM)")
    print(f"  Encoder     : {'NVIDIA NVENC Hardware Active' if hw.has_nvenc_hevc else 'CPU (libx265)'}")
    print(f"  Preset      : {chosen_preset.name}")
    print(f"  Audio Mode  : {audio_val.upper()} (Preserved)")
    print("=" * 70)

    if not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        sys.exit(1)

    scene_cfg = SceneNamingConfig(
        show_name="Happy.Tree.Friends",
        resolution_tag="2160p" if res_val == "4k" else res_val,
        source_tag=args.source_tag,
        audio_tag="AAC2.0",
        codec_tag="H.265" if "hevc" in codec_val or "x265" in codec_val else "H.264",
        group_tag=args.group,
        container=".mp4"
    )

    bm = BatchManager(hw)
    print(f"\nScanning folder: {args.source} ...")
    queue = bm.scan_folder(args.source, args.output, scene_cfg, preserve_folders=True)

    if not queue:
        print("No video files found.")
        sys.exit(0)

    if args.limit > 0:
        queue = queue[:args.limit]
        bm.queue = queue

    print(f"Found {len(queue)} episode(s) to process.")
    print(f"Destination folder: {args.output}\n")

    def on_item_updated(item: QueueItem):
        if item.status == EpisodeStatus.PROCESSING:
            print(f"\r[{item.index}/{len(queue)}] {item.relative_source_path} : {item.status_detail}   ", end="", flush=True)
        elif item.status == EpisodeStatus.COMPLETED:
            print(f"\r[{item.index}/{len(queue)}] {item.target_scene_name} -> {item.status_detail} ({item.output_size_mb} MB)          \n")
        elif item.status == EpisodeStatus.SKIPPED:
            print(f"[{item.index}/{len(queue)}] {item.target_scene_name} -> Already present (skipped)")
        elif item.status == EpisodeStatus.FAILED:
            print(f"\n[{item.index}/{len(queue)}] ERROR: {item.error_message}")

    t0 = time.time()
    bm.run_batch(
        output_dir=args.output,
        base_dir=args.source if os.path.isdir(args.source) else os.path.dirname(args.source),
        target_preset=res_val,
        video_codec=codec_val,
        model_id=model_val,
        scene_config=scene_cfg,
        preserve_folders=True,
        overwrite=args.overwrite,
        threads=args.threads,
        frame_format=fmt_val,
        max_workers=workers_val,
        cq_level=cq_val,
        nvenc_preset=p_val,
        enable_tta=tta_val,
        denoise_level=denoise_val,
        audio_mode=audio_val,
        tile_size=args.tile_size,
        item_callback=on_item_updated
    )

    total_time = round(time.time() - t0, 1)
    print(f"\nProcessing finished in {total_time}s.")


if __name__ == "__main__":
    run_cli()
