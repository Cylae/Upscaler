"""High-Performance Command-Line Interface for AnimaScale.

Supports single files, batch directories, recursive matching, hardware diagnostics,
dry-run output planning, and real-time streaming telemetry with deterministic signal handling.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from typing import Any

# Dynamic dependency resolution to honor test patches on src.cli and standard engine components
def _get_dep(name: str, fallback_module: str, fallback_attr: str | None = None) -> Any:
    src_cli = sys.modules.get("src.cli")
    if src_cli is not None and hasattr(src_cli, name):
        return getattr(src_cli, name)
    import importlib
    mod = importlib.import_module(fallback_module)
    return getattr(mod, fallback_attr or name)


def _get_available_models() -> dict[str, Any]:
    return _get_dep("AVAILABLE_MODELS", "src.core.models")


def _get_recommend_model_for_video() -> Any:
    return _get_dep("recommend_model_for_video", "src.core.models")


def _get_batch_manager_cls() -> Any:
    return _get_dep("BatchManager", "src.core.batch")


def _get_episode_status_cls() -> Any:
    return _get_dep("EpisodeStatus", "src.core.batch")


def _get_hardware_info() -> Any:
    fn = _get_dep("get_hardware_info", "src.core.hardware")
    return fn()


def _get_print_doctor_report() -> Any:
    return _get_dep("print_doctor_report", "src.core.hardware")


def _get_export_diagnostic_report() -> Any:
    return _get_dep("export_diagnostic_report", "src.core.hardware")


def _get_presets() -> dict[str, Any]:
    return _get_dep("ENHANCEMENT_PRESETS", "src.core.presets")


def _get_preset_fn() -> Any:
    return _get_dep("get_preset", "src.core.presets")


def _get_resolution_tags() -> dict[str, str]:
    return _get_dep("RESOLUTION_TAGS", "src.core.presets")


def _get_scene_naming_config_cls() -> Any:
    return _get_dep("SceneNamingConfig", "src.core.scene_namer")


def _get_probe_video() -> Any:
    return _get_dep("probe_video", "src.core.probe")


def format_eta_str(seconds: float) -> str:
    """Formats seconds into HH:MM:SS or MM:SS."""
    if seconds <= 0 or seconds != seconds:  # NaN check
        return "00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_vram_telemetry_string() -> str:
    """Queries real-time GPU VRAM telemetry (used/total) via hardware probe."""
    try:
        from src.core.hardware import query_gpu_vram_status
        free_gb, total_gb = query_gpu_vram_status()
        if total_gb > 0:
            used_gb = max(0.0, total_gb - free_gb)
            pct = (used_gb / total_gb) * 100.0
            return f"{used_gb:.1f}/{total_gb:.1f} GB ({pct:.0f}%)"
    except Exception:
        pass
    return "N/A"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="animascale",
        description="AnimaScale 2.0 — Next-Gen Neural Video & Image Upscaling Engine (Zero-Disk Streaming & Vulkan/DirectML Architecture)",
    )

    # Inputs
    parser.add_argument(
        "positional_source",
        nargs="?",
        default=None,
        help="Input video/image file or directory (supports Windows drag-and-drop)",
    )
    parser.add_argument("--source", "-s", default=None, help="Source folder or video/image file path")
    parser.add_argument("--input", "-i", dest="input_flag", default=None, help="Alternative alias for --source")
    parser.add_argument(
        "--recursive", "-r", action="store_true", default=False, help="Recursively traverse directories for media files"
    )

    # Output
    parser.add_argument(
        "--output", "-o", default=None, help="Output directory or target filename (default: '<source> (<resolution>)')"
    )

    # Presets
    parser.add_argument(
        "--preset",
        "-p",
        default=None,
        help="Enhancement profile preset: 'anime_4k_master', 'photo_realistic_4k', 'balanced_fast', 'ultrawide', 'custom'",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available quality profiles and exit")

    # Engine Architecture & Backend
    parser.add_argument(
        "--backend",
        "-b",
        choices=["auto", "directml", "cuda", "tensorrt", "cpu", "vulkan"],
        default="auto",
        help="Neural inference backend (default: auto)",
    )
    parser.add_argument("--model", "-m", default=None, help="Upscaling neural model ID or path to .onnx model")
    parser.add_argument("--list-models", action="store_true", help="List available AI models and their capabilities")
    parser.add_argument(
        "--scale", type=int, choices=[2, 3, 4], default=None, help="Super-resolution scaling factor (2, 3, or 4)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile size in pixels (0 = auto-tuned to VRAM; minimum 32 pixels if manual)",
    )
    parser.add_argument(
        "--overlap", type=int, default=32, help="Feathered blending overlap in pixels (default: 32)"
    )

    # Encoders & Quality
    parser.add_argument(
        "--codec",
        "-c",
        choices=[
            "auto",
            "hevc_nvenc",
            "hevc_qsv",
            "hevc_amf",
            "libx265",
            "h264",
            "h264_nvenc",
            "libx264",
        ],
        default=None,
        help="Video encoder codec (default: auto)",
    )
    parser.add_argument("--cq", "--crf", dest="cq", type=int, default=None, help="Constant Quality rate control (14-28)")
    parser.add_argument(
        "--nvenc-preset",
        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        default=None,
        help="NVENC performance/quality preset",
    )
    parser.add_argument("--tta", action="store_true", help="Enable 8-pass Test-Time Augmentation ensemble inference")
    parser.add_argument(
        "--denoise", type=int, choices=[-1, 0, 1, 2, 3], default=None, help="RealCUGAN denoise level (-1 to 3)"
    )

    # Audio Handling
    parser.add_argument(
        "--audio",
        "-a",
        dest="audio_mode",
        choices=["copy", "aac", "aac_192k", "aac_320k", "flac", "none"],
        default=None,
        help="Audio handling mode: direct copy or transcode (default: copy)",
    )
    parser.add_argument(
        "--audio-mode",
        dest="audio_mode_alias",
        choices=["copy", "aac", "aac_192k", "aac_320k", "flac", "none"],
        default=None,
        help=argparse.SUPPRESS,
    )

    # Geometry & Resolution
    parser.add_argument(
        "--resolution",
        choices=["8k", "4k", "1440p", "1080p", "720p", "ultrawide"],
        default=None,
        help="Target resolution: '8k', '4k', '1440p', '1080p', '720p', 'ultrawide'",
    )

    # Diagnostics & Inspection
    parser.add_argument(
        "--diagnostics", "--doctor", dest="diagnostics", action="store_true", help="Run hardware and environment diagnostic"
    )
    parser.add_argument(
        "--export-diagnostics",
        nargs="?",
        const="",
        default=None,
        help="Export diagnostic report to file (or stdout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and display planned batch execution and mappings without converting files",
    )
    parser.add_argument(
        "--preview",
        type=float,
        default=None,
        help="Render only a short preview clip of N seconds (e.g. --preview 10.0)",
    )

    # Concurrency, Storage & Batch Control
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already existing files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items to process (0 = all)")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of concurrent workers (1-4)")
    parser.add_argument("--threads", "-t", default="auto", help="Engine load:proc:save threads (e.g. auto, 4:4:4)")
    parser.add_argument(
        "--frame-format", "-f", choices=["jpg", "png"], default=None, help="Frame intermediate format (default: jpg)"
    )
    parser.add_argument("--manifest", help="Path to save or restore batch manifest JSON file")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed or cancelled items from previous batch")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept AI model selection without interactive prompt")

    # Scene Naming
    parser.add_argument("--group", "-g", default="CYLAE", help="Scene release group tag (default: CYLAE)")
    parser.add_argument("--show-name", default="Happy Tree Friends", help="Show name used in generated release filenames")
    parser.add_argument(
        "--source-tag",
        default="WEB-DL",
        choices=["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"],
        help="Scene source tag",
    )

    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint with comprehensive option parsing, diagnostics, and deterministic execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Consolidate alias options
    if args.input_flag and not args.source:
        args.source = args.input_flag
    if not args.source and args.positional_source:
        args.source = args.positional_source
    if args.audio_mode_alias and not args.audio_mode:
        args.audio_mode = args.audio_mode_alias

    # Validate numeric and format constraints
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

    # Diagnostics mode
    if args.export_diagnostics is not None:
        out_f = args.export_diagnostics if args.export_diagnostics else None
        export_fn = _get_export_diagnostic_report()
        report_text = export_fn(out_f, redact_paths=True)
        if out_f:
            print(f"Diagnostic report exported to: {out_f}")
        else:
            print(report_text)
        return 0

    if args.diagnostics:
        doctor_fn = _get_print_doctor_report()
        doctor_fn()
        return 0

    # Preset listing
    presets_dict = _get_presets()
    if args.list_presets:
        print("AVAILABLE QUALITY PROFILES")
        for preset_id, preset in presets_dict.items():
            print(f"- {preset_id:<22} {preset.icon} {preset.name}")
            print(f"  {preset.description}")
        return 0

    # Model listing
    models_dict = _get_available_models()
    if args.list_models:
        print("AVAILABLE NEURAL UPSCALING MODELS")
        for m_id, model in models_dict.items():
            if m_id == "auto":
                continue
            scales = ", ".join(f"{s}x" for s in getattr(model, "supported_scales", [model.default_scale]))
            print(f"- {m_id:<24} {model.name} (Scales: {scales})")
            print(f"  {model.description}")
        return 0

    # Ensure source is provided unless restoring from manifest
    if not args.source and not (args.manifest and os.path.isfile(args.manifest)):
        parser.error("--source is required unless --doctor, --list-presets, or --manifest is used")

    # Resolve preset and fallback parameters
    preset_resolver = _get_preset_fn()
    chosen_preset = preset_resolver(args.preset or "cartoon_4k_master")
    res_val = args.resolution or chosen_preset.target_preset
    codec_val = args.codec or chosen_preset.video_codec
    if codec_val == "auto":
        codec_val = chosen_preset.video_codec
    elif codec_val == "h264":
        codec_val = "h264_nvenc"
    model_val = args.model or chosen_preset.model_id
    cq_val = args.cq if args.cq is not None else chosen_preset.cq_level
    p_val = args.nvenc_preset or chosen_preset.nvenc_preset
    tta_val = args.tta or chosen_preset.enable_tta
    denoise_val = args.denoise if args.denoise is not None else chosen_preset.denoise_level
    audio_val = args.audio_mode or chosen_preset.audio_mode
    workers_val = args.workers if args.workers is not None else chosen_preset.workers
    fmt_val = args.frame_format or chosen_preset.frame_format
    scale_val = args.scale or (4 if res_val in ("8k", "4k") else 2)

    resolution_tags = _get_resolution_tags()
    output_tag = resolution_tags.get(res_val, "2160p")

    if not args.output and args.source:
        if os.path.isdir(args.source):
            args.output = f"{args.source} ({output_tag})"
        else:
            source_parent = os.path.dirname(os.path.abspath(args.source))
            source_stem = os.path.splitext(os.path.basename(args.source))[0]
            args.output = os.path.join(source_parent, f"{source_stem} ({output_tag})")

    hw = _get_hardware_info()
    vram_telemetry = get_vram_telemetry_string()

    # Preflight Banner
    print("=" * 70)
    print("  ANIMASCALE 2.0 — HIGH-PERFORMANCE NEURAL MEDIA ENGINE")
    print(f"  GPU         : {hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM) [Live: {vram_telemetry}]")
    print(f"  Backend     : {args.backend.upper()}")
    print(f"  Encoder     : {codec_val.upper()}")
    print(f"  Preset      : {chosen_preset.name}")
    print(f"  Scale Factor: {scale_val}x (Target: {res_val.upper()})")
    print(f"  Audio Mode  : {audio_val.upper()} (Preserved)")
    if args.source:
        print(f"  Source      : {args.source}")
    print(f"  Output      : {args.output}")
    print("=" * 70)

    if args.source and not os.path.exists(args.source):
        print(f"Error: Source path '{args.source}' does not exist.")
        return 1

    scene_cfg_cls = _get_scene_naming_config_cls()
    scene_cfg = scene_cfg_cls(
        show_name=args.show_name,
        resolution_tag=output_tag,
        source_tag=args.source_tag,
        audio_tag="AAC2.0",
        codec_tag="H.265" if any(k in codec_val for k in ("hevc", "x265", "265")) else "H.264",
        group_tag=args.group,
        container=".mp4",
    )

    batch_mgr_cls = _get_batch_manager_cls()
    episode_status_cls = _get_episode_status_cls()
    probe_video_fn = _get_probe_video()

    bm = batch_mgr_cls(hw)

    # Signal Handling for Deterministic Instant Abort
    def _sig_handler(sig: int, frame: Any) -> None:
        print("\n\n[Cancellation] Interrupt signal received. Aborting all neural processes cleanly...", file=sys.stderr)
        try:
            bm.stop()
            if args.manifest:
                bm.save_manifest(args.manifest)
        except Exception:
            pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _sig_handler)
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass

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
        queue = queue[: args.limit]
        bm.queue = queue

    print(f"Found {len(queue)} episode(s) in queue.")
    print(f"Destination folder: {args.output}\n")

    # AI Model Selection & Telemetry
    if model_val == "auto":
        rec_fn = _get_recommend_model_for_video()
        recommendations: dict[tuple[str, int, str], int] = {}
        for item in queue:
            try:
                meta = probe_video_fn(item.source_path)
                rec = rec_fn(meta.display_width, meta.display_height, res_val)
                key = (rec.model.name, rec.scale, rec.reason)
                recommendations[key] = recommendations.get(key, 0) + 1
            except Exception as exc:
                print(f"Warning: unable to analyse '{item.relative_source_path}': {exc}")

        print("AI MODEL AUTO-DETECTION")
        for (name, scale, reason), count in recommendations.items():
            print(f"- {name} ({scale}x) — {count} file(s)")
            print(f"  Reason: {reason}")
    else:
        model = models_dict.get(model_val)
        model_name = getattr(model, "name", model_val)
        model_desc = getattr(model, "description", "Manual configuration")
        print(f"AI model selected manually: {model_name}")
        print(f"Reason: {model_desc}")

    # Preflight Summary
    preflight = bm.get_preflight_summary(
        args.output,
        target_preset=res_val,
        video_codec=codec_val,
        model_id=model_val,
        cq_level=cq_val,
    )

    print("\nPREFLIGHT SUMMARY")
    print(
        f"  Total items : {preflight['total_items']} ({preflight['pending_items']} pending, {preflight['skipped_items']} skipped)"
    )
    print(f"  Target      : {res_val.upper()} · {codec_val} · CQ {cq_val}")
    print(f"  Source Size : {preflight['total_source_size_mb']} MB")
    print(f"  Est. Output : ~{preflight['estimated_output_mb']} MB")
    print(f"  Target Free : {preflight['free_disk_mb']} MB")
    print(f"  Processing  : {workers_val} worker(s) · threads {args.threads} · {fmt_val.upper()} frames")
    print(f"  Overwrite   : {'yes' if args.overwrite else 'no'}")

    if preflight.get("warnings"):
        print("\n  PREFLIGHT WARNINGS:")
        for w in preflight["warnings"]:
            print(f"  [!] {w}")

    # Dry-Run Execution Plan
    if args.dry_run:
        print("\nDRY-RUN OUTPUT PLAN")
        plan = bm.plan_dry_run(args.output, scene_cfg, preserve_folders=True)
        for p in plan:
            status_tag = "[SKIP]" if p["already_exists"] and not args.overwrite else "[READY]"
            coll_tag = " [COLLISION WARNING!]" if p.get("collision") else ""
            print(f"  {status_tag} {p['relative_source']} ({p['source_size_mb']} MB) -> {p['target_name']}{coll_tag}")
        print(f"\nDry run complete: {len(plan)} item(s) mapped, no files were converted.")
        return 0

    if args.manifest and not os.path.isfile(args.manifest):
        bm.save_manifest(args.manifest)

    # Interactive confirmation
    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "Error: model validation requires an interactive terminal. Re-run with --yes to accept the displayed choice."
            )
            return 2
        confirmation = input("Start batch with this AI model selection? [y/N]: ").strip().lower()
        if confirmation not in ("y", "yes", "o", "oui"):
            print("Batch cancelled: AI model selection was not validated.")
            return 0

    # Real-Time Telemetry and Progress Reporting
    last_vram_check = 0.0
    cached_vram = get_vram_telemetry_string()

    def on_item_updated(item: Any) -> None:
        nonlocal last_vram_check, cached_vram
        now = time.monotonic()
        if now - last_vram_check >= 1.0:
            last_vram_check = now
            cached_vram = get_vram_telemetry_string()

        status = item.status
        if status == episode_status_cls.PROCESSING:
            vram_tag = f" · VRAM: {cached_vram}" if cached_vram != "N/A" else ""
            print(
                f"\r[{item.index}/{len(queue)}] {item.relative_source_path} : {item.status_detail}{vram_tag}   ",
                end="",
                flush=True,
            )
        elif status == episode_status_cls.COMPLETED:
            print(
                f"\r[{item.index}/{len(queue)}] {item.target_scene_name} -> {item.status_detail} ({item.output_size_mb} MB)          \n"
            )
        elif status == episode_status_cls.SKIPPED:
            print(f"[{item.index}/{len(queue)}] {item.target_scene_name} -> Already present (skipped)")
        elif status == episode_status_cls.FAILED:
            print(f"\n[{item.index}/{len(queue)}] ERROR: {item.error_message}")
        elif status == episode_status_cls.CANCELLED:
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
            preview_duration_s=args.preview,
            item_callback=on_item_updated,
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
    failed = sum(1 for item in queue if item.status == episode_status_cls.FAILED)
    cancelled = sum(1 for item in queue if item.status == episode_status_cls.CANCELLED)
    completed = sum(1 for item in queue if item.status == episode_status_cls.COMPLETED)
    skipped = sum(1 for item in queue if item.status == episode_status_cls.SKIPPED)

    print(f"\nProcessing finished in {total_time}s: {completed} completed, {skipped} skipped, {failed} failed.")
    return 130 if cancelled else (1 if failed else 0)


def main() -> int:
    """Console script entrypoint."""
    return run_cli()


if __name__ == "__main__":
    sys.exit(main())
