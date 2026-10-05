import argparse
import os
import re
import sys
import time

if __name__ == "__main__" or "src" not in sys.modules:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.core.ai_models import AVAILABLE_MODELS, recommend_model_for_video
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import export_diagnostic_report, get_hardware_info, print_doctor_report
from src.core.presets import ENHANCEMENT_PRESETS, get_preset
from src.core.scene_renamer import SceneNamingConfig
from src.core.video_pipeline import probe_video


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="4K Video Enhancer for Cartoons & 2D Animation (Scene / Warez Naming)"
    )
    parser.add_argument("--source", "-s", help="Source folder or video file path")
    parser.add_argument("--output", "-o", help="Output directory (default: '<source> (2160p)')")
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
    parser.add_argument("--show-name", default="Happy Tree Friends", help="Show name used in generated release filenames")
    parser.add_argument("--source-tag", default="WEB-DL", choices=["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"], help="Scene source tag")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes to process (0 = all)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of concurrent workers (default: 2)")
    parser.add_argument("--threads", "-t", default="auto", help="Engine load:proc:save threads (e.g. auto, 4:4:4, 2:4:2)")
    parser.add_argument("--frame-format", "-f", choices=["jpg", "png"], default=None, help="Frame I/O format (default: jpg, 2x-3x faster)")
    parser.add_argument("--doctor", action="store_true", help="Run hardware and environment diagnostic")
    parser.add_argument("--export-diagnostics", nargs="?", const="", default=None, help="Export diagnostic report to file (or stdout)")
    parser.add_argument("--manifest", help="Path to save or restore batch manifest JSON file")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed or cancelled items from previous batch")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept the displayed AI-model choice without an interactive prompt")
    parser.add_argument("--dry-run", action="store_true", help="Scan and display the planned batch and mappings without converting files")
    parser.add_argument("--list-presets", action="store_true", help="List the available quality profiles and exit")

    args = parser.parse_args(argv)

    if args.cq is not None and not 14 <= args.cq <= 28:
        parser.error("--cq must be between 14 and 28")
    if args.workers is not None and not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    if args.tile_size != 0 and args.tile_size < 32:
        parser.error("--tile-size must be 0 (auto) or at least 32 pixels")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive number")
    if args.threads != "auto":
        match = re.fullmatch(r"(\d+):(\d+):(\d+)", args.threads)
        if not match or any(int(value) < 1 for value in match.groups()):
            parser.error("--threads must be 'auto' or LOAD:PROC:SAVE with positive integers")

    if args.export_diagnostics is not None:
        out_f = args.export_diagnostics if args.export_diagnostics else None
        report_text = export_diagnostic_report(out_f, redact_paths=True)
        if out_f:
            print(f"Diagnostic report exported to: {out_f}")
        else:
            print(report_text)
        return 0

    if args.doctor:
        print_doctor_report()
        return 0

    if args.list_presets:
        print("AVAILABLE QUALITY PROFILES")
        for preset_id, preset in ENHANCEMENT_PRESETS.items():
            print(f"- {preset_id:<22} {preset.icon} {preset.name}")
            print(f"  {preset.description}")
        return 0

    if not args.source and not (args.manifest and os.path.isfile(args.manifest)):
        parser.error("--source is required unless --doctor, --list-presets, or --manifest is used")

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

    if not args.output and args.source:
        output_tag = "2160p" if res_val == "4k" else res_val
        if os.path.isdir(args.source):
            args.output = f"{args.source} ({output_tag})"
        else:
            source_parent = os.path.dirname(os.path.abspath(args.source))
            source_stem = os.path.splitext(os.path.basename(args.source))[0]
            args.output = os.path.join(source_parent, f"{source_stem} ({output_tag})")

    hw = get_hardware_info()
    print("=" * 70)
    print("  VIDEO ENHANCER 4K — ANIMATION & CARTOON EDITION")
    print(f"  GPU         : {hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM)")
    print(f"  Encoder     : {'NVIDIA NVENC Hardware Active' if hw.has_nvenc_hevc else 'CPU (libx265)'}")
    print(f"  Preset      : {chosen_preset.name}")
    print(f"  Audio Mode  : {audio_val.upper()} (Preserved)")
    if args.source:
        print(f"  Source      : {args.source}")
    print(f"  Output      : {args.output}")
    print("=" * 70)

    if args.source and not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        return 1

    scene_cfg = SceneNamingConfig(
        show_name=args.show_name,
        resolution_tag="2160p" if res_val == "4k" else res_val,
        source_tag=args.source_tag,
        audio_tag="AAC2.0",
        codec_tag="H.265" if "hevc" in codec_val or "x265" in codec_val else "H.264",
        group_tag=args.group,
        container=".mp4"
    )

    bm = BatchManager(hw)
    if args.manifest and os.path.isfile(args.manifest):
        print(f"\nRestoring queue from manifest: {args.manifest} ...")
        bm.load_manifest(args.manifest)
        queue = bm.queue
        if not args.output and queue:
            args.output = os.path.dirname(os.path.abspath(queue[0].target_full_path))
    else:
        print(f"\nScanning folder: {args.source} ...")
        queue = bm.scan_folder(args.source, args.output, scene_cfg, preserve_folders=True)

    if not queue:
        print("No video files found.")
        return 0

    if args.retry_failed:
        retried = bm.retry_failed_items()
        print(f"Retried {retried} previously failed/cancelled item(s).")

    if args.limit > 0:
        queue = queue[:args.limit]
        bm.queue = queue

    print(f"Found {len(queue)} episode(s) in queue.")
    print(f"Destination folder: {args.output}\n")

    if model_val == "auto":
        recommendations = {}
        for item in queue:
            try:
                meta = probe_video(item.source_path)
                rec = recommend_model_for_video(meta.width, meta.height, res_val)
                key = (rec.model.name, rec.scale, rec.reason)
                recommendations[key] = recommendations.get(key, 0) + 1
            except Exception as exc:
                print(f"Warning: unable to analyse '{item.relative_source_path}': {exc}")

        print("AI MODEL AUTO-DETECTION")
        for (name, scale, reason), count in recommendations.items():
            print(f"- {name} ({scale}x) — {count} file(s)")
            print(f"  Reason: {reason}")
    else:
        model = AVAILABLE_MODELS[model_val]
        print(f"AI model selected manually: {model.name}")
        print(f"Reason: {model.description}")

    # Preflight Summary
    preflight = bm.get_preflight_summary(
        args.output,
        target_preset=res_val,
        video_codec=codec_val,
        model_id=model_val,
        cq_level=cq_val
    )

    print("\nPREFLIGHT SUMMARY")
    print(f"  Total items : {preflight['total_items']} ({preflight['pending_items']} pending, {preflight['skipped_items']} skipped)")
    print(f"  Target      : {res_val.upper()} · {codec_val} · CQ {cq_val}")
    print(f"  Source Size : {preflight['total_source_size_mb']} MB")
    print(f"  Est. Output : ~{preflight['estimated_output_mb']} MB")
    print(f"  Target Free : {preflight['free_disk_mb']} MB")
    print(f"  Processing  : {workers_val} worker(s) · threads {args.threads} · {fmt_val.upper()} frames")
    print(f"  Overwrite   : {'yes' if args.overwrite else 'no'}")

    if preflight["warnings"]:
        print("\n  PREFLIGHT WARNINGS:")
        for w in preflight["warnings"]:
            print(f"  [!] {w}")

    if args.dry_run:
        print("\nDRY-RUN OUTPUT PLAN")
        plan = bm.plan_dry_run(args.output, scene_cfg, preserve_folders=True)
        for p in plan:
            status_tag = "[SKIP]" if p["already_exists"] and not args.overwrite else "[READY]"
            coll_tag = " [COLLISION WARNING!]" if p["collision"] else ""
            print(f"  {status_tag} {p['relative_source']} ({p['source_size_mb']} MB) -> {p['target_name']}{coll_tag}")
        print(f"\nDry run complete: {len(plan)} item(s) mapped, no files were converted.")
        return 0

    if args.manifest and not os.path.isfile(args.manifest):
        bm.save_manifest(args.manifest)

    if not args.yes:
        if not sys.stdin.isatty():
            print("Error: model validation requires an interactive terminal. Re-run with --yes to accept the displayed choice.")
            return 2
        confirmation = input("Start batch with this AI model selection? [y/N]: ").strip().lower()
        if confirmation not in ("y", "yes", "o", "oui"):
            print("Batch cancelled: AI model selection was not validated.")
            return 0

    def on_item_updated(item: QueueItem):
        if item.status == EpisodeStatus.PROCESSING:
            print(f"\r[{item.index}/{len(queue)}] {item.relative_source_path} : {item.status_detail}   ", end="", flush=True)
        elif item.status == EpisodeStatus.COMPLETED:
            print(f"\r[{item.index}/{len(queue)}] {item.target_scene_name} -> {item.status_detail} ({item.output_size_mb} MB)          \n")
        elif item.status == EpisodeStatus.SKIPPED:
            print(f"[{item.index}/{len(queue)}] {item.target_scene_name} -> Already present (skipped)")
        elif item.status == EpisodeStatus.FAILED:
            print(f"\n[{item.index}/{len(queue)}] ERROR: {item.error_message}")
        elif item.status == EpisodeStatus.CANCELLED:
            print(f"\n[{item.index}/{len(queue)}] CANCELLED")

    t0 = time.time()
    effective_base_dir = (
        args.source
        if (args.source and os.path.isdir(args.source))
        else (
            os.path.dirname(os.path.abspath(args.source))
            if args.source
            else (os.path.dirname(os.path.abspath(queue[0].source_path)) if queue else os.getcwd())
        )
    )
    try:
        bm.run_batch(
            output_dir=args.output,
            base_dir=effective_base_dir,
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
    except KeyboardInterrupt:
        bm.stop()
        if args.manifest:
            bm.save_manifest(args.manifest)
        print("\nBatch cancelled by user.")
        return 130

    if args.manifest:
        bm.save_manifest(args.manifest)

    total_time = round(time.time() - t0, 1)
    failed = sum(1 for item in queue if item.status == EpisodeStatus.FAILED)
    cancelled = sum(1 for item in queue if item.status == EpisodeStatus.CANCELLED)
    completed = sum(1 for item in queue if item.status == EpisodeStatus.COMPLETED)
    skipped = sum(1 for item in queue if item.status == EpisodeStatus.SKIPPED)
    print(f"\nProcessing finished in {total_time}s: {completed} completed, {skipped} skipped, {failed} failed.")
    return 130 if cancelled else (1 if failed else 0)


if __name__ == "__main__":
    sys.exit(run_cli())
