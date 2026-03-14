# Pipeline config + dataclasses

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _env_optional_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped or default


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


def _default_ffmpeg_binary() -> str:
    env_value = _env_optional_str("FFMPEG_BINARY")
    if env_value:
        return env_value

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _default_ffprobe_binary() -> str:
    env_value = _env_optional_str("FFPROBE_BINARY")
    if env_value:
        return env_value
    system_ffprobe = shutil.which("ffprobe")
    return system_ffprobe or "ffprobe"


@dataclass(frozen=True)
class FrameRecord:
    index: int
    timestamp: float
    path: Path


@dataclass(frozen=True)
class InMemoryFrame:
    index: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    area: float

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return float(self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return float(self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class DigitDetection:
    digit: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class NumberCandidate:
    number: int
    confidence: float
    digits: tuple[DigitDetection, ...]
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    mask: "np.ndarray | None" = field(default=None, repr=False, compare=False)  # bbox-local seg mask (uint8 0/255)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class DetectedFrame:
    timestamp: float
    confidence: float
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    frame_w: float = 0.0
    frame_h: float = 0.0

    def to_dict(self) -> dict:
        # Calculate percentage coordinates (0-100%)
        def pct(val, dim):
            return round(100.0 * val / dim, 2) if dim else 0.0
        return {
            "timestamp": float(self.timestamp),
            "confidence": float(self.confidence),
            "bbox": {
                "x1": round(self.x1),
                "y1": round(self.y1),
                "x2": round(self.x2),
                "y2": round(self.y2),
                "x1_pct": pct(self.x1, self.frame_w),
                "y1_pct": pct(self.y1, self.frame_h),
                "x2_pct": pct(self.x2, self.frame_w),
                "y2_pct": pct(self.y2, self.frame_h),
            },
        }


@dataclass(frozen=True)
class PriorRegion:
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float


def _default_position_priors() -> dict[str, dict[str, PriorRegion]]:
    return {
        "football": {"quarterback": PriorRegion(0.50, 0.42, 0.24, 0.20)},
        "basketball": {"guard": PriorRegion(0.50, 0.54, 0.32, 0.28)},
        "lacrosse": {"midfielder": PriorRegion(0.50, 0.50, 0.35, 0.30)},
    }


@dataclass(frozen=True)
class PipelineSettings:
    fps: int = field(default_factory=lambda: _env_int("FPS", 2))
    frame_batch_size: int = field(default_factory=lambda: _env_int("FRAME_BATCH_SIZE", 50))
    max_frames: int | None = field(default_factory=lambda: _env_optional_int("MAX_FRAMES"))

    conf_threshold_export: float = field(
        default_factory=lambda: _env_float("CONF_THRESHOLD_EXPORT", 0.55)
    )
    conf_threshold_internal: float = field(
        default_factory=lambda: _env_float("CONF_THRESHOLD_INTERNAL", 0.55)
    )
    position_prior_weight: float = field(
        default_factory=lambda: _env_float("POSITION_PRIOR_WEIGHT", 0.10)
    )
    hsv_tolerance_h: int = field(default_factory=lambda: _env_int("HSV_TOLERANCE_H", 20))
    hsv_tolerance_sv: int = field(default_factory=lambda: _env_int("HSV_TOLERANCE_SV", 60))
    roi_min_area: int = field(default_factory=lambda: _env_int("ROI_MIN_AREA", 350))
    roi_padding_px: int = field(default_factory=lambda: _env_int("ROI_PADDING_PX", 8))
    roi_max_area_ratio: float = field(
        default_factory=lambda: _env_float("ROI_MAX_AREA_RATIO", 0.55)
    )
    max_rois_per_frame: int = field(default_factory=lambda: _env_int("MAX_ROIS_PER_FRAME", 12))

    detector_conf_threshold: float = field(
        default_factory=lambda: _env_float("DETECTOR_CONF_THRESHOLD", 0.35)
    )
    detector_iou_threshold: float = field(
        default_factory=lambda: _env_float("DETECTOR_IOU_THRESHOLD", 0.45)
    )
    min_person_crop_height: int = field(
        default_factory=lambda: _env_int("MIN_PERSON_CROP_HEIGHT", 80)
    )
    digit_max_gap_ratio: float = field(
        default_factory=lambda: _env_float("DIGIT_MAX_GAP_RATIO", 0.60)
    )
    yolo_model_source: str = field(
        default_factory=lambda: _env_optional_str("YOLO_MODEL_SOURCE", "app/model/jersey_number_yolo11m.pt")
        or "app/model/jersey_number_yolo11m.pt"
    )
    person_model_source: str = field(
        default_factory=lambda: _env_optional_str("PERSON_MODEL_SOURCE", "app/model/yolo26n-seg.pt")
        or "app/model/yolo26n-seg.pt"
    )
    yolo_device: str = field(default_factory=lambda: os.getenv("YOLO_DEVICE", "auto"))

    enable_person_fallback: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PERSON_FALLBACK", True)
    )
    person_fallback_min_conf: float = field(
        default_factory=lambda: _env_float("PERSON_FALLBACK_MIN_CONF", 0.35)
    )
    person_fallback_min_overlap: float = field(
        default_factory=lambda: _env_float("PERSON_FALLBACK_MIN_OVERLAP", 0.25)
    )
    person_fallback_area_scale: float = field(
        default_factory=lambda: _env_float("PERSON_FALLBACK_AREA_SCALE", 2200.0)
    )

    yt_dlp_binary: str = field(default_factory=lambda: os.getenv("YT_DLP_BINARY", "yt-dlp"))
    yt_dlp_js_runtimes: str = field(
        default_factory=lambda: _env_optional_str("YT_DLP_JS_RUNTIMES", "deno") or "deno"
    )
    youtube_clip_seconds: int | None = field(
        default_factory=lambda: _env_optional_int("YOUTUBE_CLIP_SECONDS")
    )

    pipeline_workers: int = field(
        default_factory=lambda: _env_int("PIPELINE_WORKERS", 4)
    )
    skip_similarity_threshold: float = field(
        default_factory=lambda: _env_float("SKIP_SIMILARITY_THRESHOLD", 0.97)
    )
    early_exit_consecutive: int = field(
        default_factory=lambda: _env_int("EARLY_EXIT_CONSECUTIVE", 0)
    )
    detection_strategy: str = field(
        default_factory=lambda: _env_optional_str("DETECTION_STRATEGY", "detection_first")
        or "detection_first"
    )

    debug_video_path: str | None = field(
        default_factory=lambda: _env_optional_str("DEBUG_VIDEO_PATH")
    )

    ffmpeg_binary: str = field(default_factory=_default_ffmpeg_binary)
    ffprobe_binary: str = field(default_factory=_default_ffprobe_binary)
    position_priors: dict[str, dict[str, PriorRegion]] = field(
        default_factory=_default_position_priors
    )

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("FPS must be greater than 0.")
        if self.frame_batch_size <= 0:
            raise ValueError("FRAME_BATCH_SIZE must be greater than 0.")
        if not (0.0 <= self.conf_threshold_internal <= 1.0):
            raise ValueError("CONF_THRESHOLD_INTERNAL must be in [0, 1].")
        if not (0.0 <= self.conf_threshold_export <= 1.0):
            raise ValueError("CONF_THRESHOLD_EXPORT must be in [0, 1].")
        if self.conf_threshold_export < self.conf_threshold_internal:
            raise ValueError("CONF_THRESHOLD_EXPORT must be >= CONF_THRESHOLD_INTERNAL.")
        if not (0.0 <= self.position_prior_weight <= 1.0):
            raise ValueError("POSITION_PRIOR_WEIGHT must be in [0, 1].")
        if self.roi_min_area < 0:
            raise ValueError("ROI_MIN_AREA cannot be negative.")
        if self.max_rois_per_frame <= 0:
            raise ValueError("MAX_ROIS_PER_FRAME must be greater than 0.")
        if not (0.0 < self.roi_max_area_ratio <= 1.0):
            raise ValueError("ROI_MAX_AREA_RATIO must be in (0, 1].")
        if self.youtube_clip_seconds is not None and self.youtube_clip_seconds <= 0:
            raise ValueError("YOUTUBE_CLIP_SECONDS must be greater than 0 when set.")
        if self.early_exit_consecutive < 0:
            raise ValueError("EARLY_EXIT_CONSECUTIVE cannot be negative.")
