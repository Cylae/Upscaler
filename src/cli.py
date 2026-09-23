import argparse
import os
import sys
import time

from src.core.ai_models import AVAILABLE_MODELS
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import get_hardware_info, print_doctor_report
from src.core.scene_renamer import SceneNamingConfig


def run_cli():
    parser = argparse.ArgumentParser(
        description="4K Video Enhancer for Cartoons & 2D Animation (Scene / Warez Naming)"
    )
    parser.add_argument("--source", "-s", default=r"Z:\DL\Happy Tree Friends", help="Source folder or file path")
    parser.add_argument("--output", "-o", default=r"Z:\DL\Happy Tree Friends (2160p)", help="Output directory")
    parser.add_argument("--resolution", "-r", choices=["4k", "1440p", "1080p", "720p"], default="4k", help="Target resolution preset")
    parser.add_argument("--codec", "-c", choices=["hevc_nvenc", "h264_nvenc", "libx265", "libx264"], default="hevc_nvenc", help="Video codec")
    parser.add_argument("--model", "-m", choices=list(AVAILABLE_MODELS.keys()), default="auto", help="Upscaling model")
    parser.add_argument("--group", "-g", default="CYLAE", help="Scene release group tag (e.g. CYLAE)")
    parser.add_argument("--source-tag", default="WEB-DL", choices=["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"], help="Scene source tag")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to process (0 = all)")
    parser.add_argument("--doctor", action="store_true", help="Run hardware and environment diagnostic")

    args = parser.parse_args()

    if args.doctor:
        print_doctor_report()
        sys.exit(0)

    hw = get_hardware_info()
    print("=" * 70)
    print("  VIDEO ENHANCER 4K — ANIMATION & CARTOON EDITION")
    print(f"  GPU      : {hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM)")
    print(f"  Encoder  : {'NVIDIA NVENC Hardware Active' if hw.has_nvenc_hevc else 'CPU (libx265)'}")
    print("=" * 70)

    if not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        sys.exit(1)

    scene_cfg = SceneNamingConfig(
        show_name="Happy.Tree.Friends",
        resolution_tag="2160p" if args.resolution == "4k" else args.resolution,
        source_tag=args.source_tag,
        audio_tag="AAC2.0",
        codec_tag="H.265" if "hevc" in args.codec or "x265" in args.codec else "H.264",
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
            print(f"\r[{item.index}/{len(queue)}] {item.relative_source_path} : {item.status_detail}", end="", flush=True)
        elif item.status == EpisodeStatus.COMPLETED:
            print(f"\r[{item.index}/{len(queue)}] {item.target_scene_name} -> Completed ({item.output_size_mb} MB)")
        elif item.status == EpisodeStatus.SKIPPED:
            print(f"[{item.index}/{len(queue)}] {item.target_scene_name} -> Already present (skipped)")
        elif item.status == EpisodeStatus.FAILED:
            print(f"\n[{item.index}/{len(queue)}] ERROR: {item.error_message}")

    t0 = time.time()
    bm.run_batch(
        output_dir=args.output,
        base_dir=args.source if os.path.isdir(args.source) else os.path.dirname(args.source),
        target_preset=args.resolution,
        video_codec=args.codec,
        model_id=args.model,
        scene_config=scene_cfg,
        preserve_folders=True,
        overwrite=args.overwrite,
        item_callback=on_item_updated
    )

    total_time = round(time.time() - t0, 1)
    print(f"\nProcessing finished in {total_time}s.")


if __name__ == "__main__":
    run_cli()
