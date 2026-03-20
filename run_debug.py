import argparse
import logging
import os
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(levelname)s %(name)s: %(message)s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run jersey detection with debug-video output.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video-url", help="Remote video URL to process.")
    source_group.add_argument("--video-path", help="Local video path to process.")

    parser.add_argument(
        "--jersey-number",
        type=int,
        required=True,
        help="Target jersey number (0-99).",
    )
    parser.add_argument(
        "--jersey-color",
        required=True,
        help="Target jersey color name, for example blue or white.",
    )
    parser.add_argument(
        "--sport",
        required=True,
        choices=["basketball", "football", "lacrosse"],
        help="Sport-specific priors to use.",
    )
    parser.add_argument(
        "--position",
        default=None,
        help="Optional player position prior, for example guard or quarterback.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the annotated debug video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frame sampling rate used by the detection pipeline.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=int,
        default=60,
        help="Clip length cap for YouTube downloads and debug runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on processed frames.",
    )
    parser.add_argument(
        "--early-exit-consecutive",
        type=int,
        default=0,
        help="Disable or tune early-exit behavior for debugging.",
    )
    parser.add_argument(
        "--backend",
        choices=["public_reader_ensemble", "legacy_yolo"],
        default=None,
        help="Override the jersey-reader backend for this run.",
    )
    parser.add_argument(
        "--strategy",
        choices=["detection_first", "color_first"],
        default=None,
        help="Override the detection strategy for this run.",
    )
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> Path:
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["DEBUG_VIDEO_PATH"] = str(output_path)
    os.environ["YOUTUBE_CLIP_SECONDS"] = str(args.clip_seconds)
    os.environ["FPS"] = str(args.fps)
    os.environ["EARLY_EXIT_CONSECUTIVE"] = str(args.early_exit_consecutive)

    if args.max_frames is not None:
        os.environ["MAX_FRAMES"] = str(args.max_frames)
    else:
        os.environ["MAX_FRAMES"] = str(max(args.clip_seconds * args.fps, 1))

    if args.backend:
        os.environ["JERSEY_READER_BACKEND"] = args.backend
    if args.strategy:
        os.environ["DETECTION_STRATEGY"] = args.strategy

    return output_path


def main() -> int:
    args = _parse_args()
    output_path = _configure_environment(args)

    from app.services.detection_detector import clear_detector_cache
    from app.services.detection_pipeline import detect_jersey_in_frames
    from app.services.detection_runtime import PipelineSettings

    clear_detector_cache()

    settings = PipelineSettings()
    print(f"Debug video path: {settings.debug_video_path}")
    print(f"YouTube clip seconds: {settings.youtube_clip_seconds}")
    print(f"Strategy: {settings.detection_strategy}")
    print(f"Jersey reader backend: {settings.jersey_reader_backend}")
    print(f"YOLO model: {settings.yolo_model_source}")
    print(f"Person model: {settings.person_model_source}")
    print(f"FPS: {settings.fps}")
    print("Starting jersey detection debug run...")

    results = detect_jersey_in_frames(
        video_url=args.video_url,
        video_path=args.video_path,
        video_bytes=None,
        jersey_number=args.jersey_number,
        jersey_color=args.jersey_color,
        sport=args.sport,
        position=args.position,
        settings=settings,
    )

    print(f"\nDetections: {len(results)}")
    for result in results[:5]:
        print(f"  t={result['timestamp']:.2f}s  conf={result['confidence']:.2f}")

    if output_path.exists():
        size = output_path.stat().st_size
        print(f"\nDebug video created: {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
        print(f"Open it at: {output_path}")
    else:
        print(f"\nDebug video not created at: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
