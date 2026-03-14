import logging
import os
import sys

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(levelname)s %(name)s: %(message)s",
)

# Ensure debug video path is set
os.environ.setdefault("DEBUG_VIDEO_PATH", r"e:\playerJerseyIdentification\debug_output.mp4")
os.environ.setdefault("YOUTUBE_CLIP_SECONDS", "60")
os.environ.setdefault("MAX_FRAMES", "120")  # 60s * 2fps = 120 frames
os.environ.setdefault("EARLY_EXIT_CONSECUTIVE", "0")  # process all frames, no early exit

from app.services.detection_detector import clear_detector_cache
from app.services.detection_pipeline import detect_jersey_in_frames
from app.services.detection_runtime import PipelineSettings

# Clear cached detector so it reloads with dual-model config
clear_detector_cache()

settings = PipelineSettings()
print(f"Debug video path: {settings.debug_video_path}")
print(f"YouTube clip seconds: {settings.youtube_clip_seconds}")
print(f"Strategy: {settings.detection_strategy}")
print(f"YOLO model: {settings.yolo_model_source}")
print(f"Person model: {settings.person_model_source}")
print(f"FPS: {settings.fps}")
print("Starting detection on YouTube basketball video...")

results = detect_jersey_in_frames(
    video_url=None,
    video_path=r"e:\playerJerseyIdentification\test_download.mp4",
    video_bytes=None,
    jersey_number=2,
    jersey_color="white",
    sport="basketball",
    position="guard",
    settings=settings,
)

print(f"\nDetections: {len(results)}")
for r in results[:5]:
    print(f"  t={r['timestamp']:.2f}s  conf={r['confidence']:.2f}")

debug_path = r"e:\playerJerseyIdentification\debug_output.mp4"
if os.path.exists(debug_path):
    size = os.path.getsize(debug_path)
    print(f"\nDebug video created: {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
    print(f"Open it at: {debug_path}")
else:
    print("\nDebug video NOT created")
