# Main detection pipeline + helpers

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import webcolors

from app.services.detection_detector import get_or_create_detector
from app.services.detection_runtime import (
    DetectedFrame,
    InMemoryFrame,
    NumberCandidate,
    PersonBox,
    PipelineSettings,
    PriorRegion,
    ROI,
)
from app.services.detection_visualizer import DebugFrameData, write_debug_video

LOGGER = logging.getLogger(__name__)
SUPPORTED_SPORTS = {"basketball", "football", "lacrosse"}
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
NAMED_COLOR_OVERRIDES = {
    "royal blue": "#4169E1",
    "navy": "#000080",
    "maroon": "#800000",
    "scarlet": "#FF2400",
    "gold": "#FFD700",
}

# ── HSV range table for common jersey colours ──────────────────────────
# Each entry maps a normalised colour name to a list of
# (H_low, H_high, S_low, S_high, V_low, V_high) tuples.
# Multiple tuples allow wrapping hue ranges (e.g. red wraps 0/180).
# Ranges are intentionally generous to absorb camera white-balance,
# exposure and broadcast colour-grading variations.
COLOR_HSV_RANGES: dict[str, list[tuple[int, int, int, int, int, int]]] = {
    "white":       [(0, 179, 0,  60, 160, 255)],
    "black":       [(0, 179, 0,  80,   0,  70)],
    "grey":        [(0, 179, 0,  50,  70, 170)],
    "gray":        [(0, 179, 0,  50,  70, 170)],
    "red":         [(0, 10, 80, 255, 80, 255), (165, 179, 80, 255, 80, 255)],
    "scarlet":     [(0, 12, 100, 255, 100, 255), (165, 179, 100, 255, 100, 255)],
    "maroon":      [(0, 12, 60, 255, 30, 150), (165, 179, 60, 255, 30, 150)],
    "orange":      [(8, 25, 100, 255, 120, 255)],
    "yellow":      [(20, 40, 80, 255, 120, 255)],
    "gold":        [(18, 35, 80, 255, 100, 255)],
    "green":       [(35, 85, 50, 255, 50, 255)],
    "teal":        [(80, 100, 50, 255, 50, 255)],
    "cyan":        [(80, 105, 50, 255, 80, 255)],
    "blue":        [(95, 130, 60, 255, 50, 255)],
    "royal blue":  [(100, 130, 80, 255, 80, 255)],
    "navy":        [(95, 130, 50, 255, 20, 130)],
    "purple":      [(125, 155, 40, 255, 40, 255)],
    "violet":      [(125, 160, 50, 255, 60, 255)],
    "pink":        [(155, 175, 30, 180, 120, 255)],
    "magenta":     [(150, 170, 80, 255, 80, 255)],
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def _is_youtube_url(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def _safe_extension(path_like: str, fallback: str = ".mp4") -> str:
    ext = Path(path_like).suffix.lower()
    return ext if ext in SUPPORTED_EXTENSIONS else fallback


def _normalize_youtube_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "youtube.com" in host and parsed.path.startswith("/live/"):
        video_id = parsed.path.split("/live/", 1)[1].split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url


def _download_direct_video(video_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        video_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as sink:
            shutil.copyfileobj(response, sink)
    return destination


def _download_youtube_video(
    *,
    video_url: str,
    work_dir: Path,
    settings: PipelineSettings,
) -> Path:
    output_template = str(work_dir / "youtube_input.%(ext)s")
    normalized_url = _normalize_youtube_url(video_url)
    clip_seconds = settings.youtube_clip_seconds
    command = [
        settings.yt_dlp_binary,
        "--js-runtimes",
        settings.yt_dlp_js_runtimes,
    ]
    if clip_seconds is not None:
        command.extend([
            "--download-sections",
            f"*0-{clip_seconds}",
        ])
    command.extend([
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--concurrent-fragments",
        "4",
        "--ffmpeg-location",
        settings.ffmpeg_binary,
        "-f",
        "96/95/94/93/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        normalized_url,
    ])
    LOGGER.info("yt-dlp command: %s", " ".join(command))

    # yt-dlp can be slow on HLS streams, especially for full games.
    timeout_seconds = max(600, (clip_seconds * 10) if clip_seconds is not None else 3600)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"yt-dlp timed out after {timeout_seconds}s downloading {normalized_url}"
        )

    if result.stderr:
        LOGGER.debug("yt-dlp stderr: %s", result.stderr.strip()[:500])
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    candidates = sorted(
        work_dir.glob("youtube_input.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("yt-dlp succeeded but no output file was produced.")
    return candidates[0]


def _resolve_video_source(
    *,
    video_url: str | None,
    video_path: str | None,
    video_bytes: bytes | None,
    work_dir: Path,
    settings: PipelineSettings,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)

    if video_bytes is not None:
        target = work_dir / "input_buffer.mp4"
        target.write_bytes(video_bytes)
        return target

    if video_path:
        source = Path(video_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"video_path does not exist: {source}")
        return source

    if not video_url:
        raise ValueError("One of video_url, video_path, or video_bytes must be provided.")

    if _is_http_url(video_url):
        if _is_youtube_url(video_url):
            return _download_youtube_video(
                video_url=video_url,
                work_dir=work_dir,
                settings=settings,
            )
        ext = _safe_extension(urllib.parse.urlparse(video_url).path)
        return _download_direct_video(video_url, work_dir / f"input_download{ext}")

    source = Path(video_url).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"video_url path does not exist: {source}")
    return source


def _get_video_duration_seconds(video_path: Path, settings: PipelineSettings) -> float:
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return max(0.0, float(result.stdout.strip()))
    except Exception:
        pass

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    if fps <= 0.0:
        return 0.0
    return max(0.0, float(frame_count / fps))


def _get_video_dimensions(
    video_path: Path, settings: PipelineSettings
) -> tuple[int, int]:
    command = [
        settings.ffprobe_binary,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass

    capture = cv2.VideoCapture(str(video_path))
    if capture.isOpened():
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if w > 0 and h > 0:
            return w, h
    raise RuntimeError(f"Cannot determine video dimensions for {video_path}")


def _iter_frames_in_memory(
    *,
    video_path: Path,
    settings: PipelineSettings,
) -> Iterator[list[InMemoryFrame]]:
    """Pipe video through ffmpeg and yield frame batches (no temp files)."""
    width, height = _get_video_dimensions(video_path, settings)
    frame_size = width * height * 3  # BGR24

    command = [
        settings.ffmpeg_binary,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={settings.fps}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    max_frames = settings.max_frames
    index = 0
    batch: list[InMemoryFrame] = []

    try:
        while True:
            if max_frames is not None and index >= max_frames:
                break
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame_array = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            # frombuffer returns read-only, need a copy
            frame_array = frame_array.copy()
            batch.append(
                InMemoryFrame(
                    index=index,
                    timestamp=index / float(settings.fps),
                    image=frame_array,
                )
            )
            index += 1
            if len(batch) >= settings.frame_batch_size:
                yield batch
                batch = []
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()

    if batch:
        yield batch

    LOGGER.info("Decoded %s frames in-memory via ffmpeg pipe", index)



def _normalize_color_input(jersey_color: str) -> str:
    return re.sub(r"\s+", " ", jersey_color.strip().lower())


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return (
        int(normalized[0:2], 16),
        int(normalized[2:4], 16),
        int(normalized[4:6], 16),
    )


def _resolve_color_rgb(jersey_color: str) -> tuple[int, int, int]:
    color = _normalize_color_input(jersey_color)
    if color.startswith("#"):
        return _hex_to_rgb(color)
    if color in NAMED_COLOR_OVERRIDES:
        return _hex_to_rgb(NAMED_COLOR_OVERRIDES[color])
    try:
        return _hex_to_rgb(webcolors.name_to_hex(color))
    except ValueError as error:
        raise ValueError(f"Unsupported jersey_color value: {jersey_color}") from error


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _is_white_mode(jersey_color: str, hsv: tuple[int, int, int]) -> bool:
    normalized = _normalize_color_input(jersey_color)
    if "white" in normalized:
        return True
    _, saturation, value = hsv
    return saturation <= 60 and value >= 160


def _hue_ranges(hue: int, tolerance: int) -> list[tuple[int, int]]:
    low = hue - tolerance
    high = hue + tolerance
    if low >= 0 and high <= 179:
        return [(low, high)]
    if low < 0:
        return [(0, high), (179 + low, 179)]
    return [(0, high - 179), (low, 179)]


def _build_jersey_mask(
    frame_bgr: np.ndarray,
    jersey_color: str,
    settings: PipelineSettings,
    *,
    skip_morphology: bool = False,
) -> np.ndarray:
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # try HSV-range table first
    normalized = _normalize_color_input(jersey_color)
    hsv_ranges = COLOR_HSV_RANGES.get(normalized)
    if hsv_ranges is not None:
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in hsv_ranges:
            lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
    else:
        # fallback: single-point HSV + tolerance for hex colours
        target_rgb = _resolve_color_rgb(jersey_color)
        target_hsv = _rgb_to_hsv(target_rgb)

        if _is_white_mode(jersey_color, target_hsv):
            lower = np.array([0, 0, 160], dtype=np.uint8)
            upper = np.array([179, 60, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_frame, lower, upper)
        else:
            target_h, target_s, target_v = target_hsv
            ranges = _hue_ranges(target_h, settings.hsv_tolerance_h)
            sv_tolerance = max(settings.hsv_tolerance_sv, 60)
            lower_s = max(0, target_s - max(sv_tolerance, 140))
            lower_v = max(0, target_v - max((sv_tolerance * 3), 180))
            upper_s = min(255, target_s + sv_tolerance)
            upper_v = min(255, target_v + sv_tolerance)

            mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
            for range_low, range_high in ranges:
                lower = np.array([max(0, range_low), lower_s, lower_v], dtype=np.uint8)
                upper = np.array([min(179, range_high), upper_s, upper_v], dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))

    if not skip_morphology:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _extract_candidate_rois(mask: np.ndarray, *, settings: PipelineSettings) -> list[ROI]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = mask.shape[:2]
    frame_area = float(frame_h * frame_w)
    rois: list[ROI] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < settings.roi_min_area:
            continue
        if frame_area > 0.0 and area > frame_area * settings.roi_max_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - settings.roi_padding_px)
        y1 = max(0, y - settings.roi_padding_px)
        x2 = min(frame_w, x + w + settings.roi_padding_px)
        y2 = min(frame_h, y + h + settings.roi_padding_px)
        if x2 <= x1 or y2 <= y1:
            continue
        rois.append(ROI(x1=x1, y1=y1, x2=x2, y2=y2, area=area))

    rois.sort(key=lambda item: item.area, reverse=True)
    return rois


def _normalize(value: str) -> str:
    return value.strip().lower()


def _resolve_prior_region(
    *,
    sport: str,
    position: str | None,
    settings: PipelineSettings,
) -> PriorRegion | None:
    if not position:
        return None
    sport_key = _normalize(sport)
    position_key = _normalize(position)

    if sport_key in settings.position_priors:
        sport_priors = settings.position_priors[sport_key]
        if position_key in sport_priors:
            return sport_priors[position_key]

    for sport_priors in settings.position_priors.values():
        if position_key in sport_priors:
            return sport_priors[position_key]

    return None


def _compute_position_likelihood(
    *,
    sport: str,
    position: str | None,
    candidate: NumberCandidate,
    frame_width: int,
    frame_height: int,
    settings: PipelineSettings,
) -> float:
    region = _resolve_prior_region(sport=sport, position=position, settings=settings)
    if region is None:
        return 0.50
    if frame_width <= 0 or frame_height <= 0:
        return 0.50

    center_x = (candidate.x1 + candidate.x2) / 2.0
    center_y = (candidate.y1 + candidate.y2) / 2.0
    norm_x = center_x / float(frame_width)
    norm_y = center_y / float(frame_height)

    sigma_x = max(1e-3, region.sigma_x)
    sigma_y = max(1e-3, region.sigma_y)
    dx = (norm_x - region.center_x) / sigma_x
    dy = (norm_y - region.center_y) / sigma_y
    likelihood = math.exp(-0.5 * ((dx * dx) + (dy * dy)))
    return max(0.0, min(1.0, likelihood))


def _blend_confidence(
    *,
    base_confidence: float,
    prior_likelihood: float,
    settings: PipelineSettings,
) -> float:
    weight = _clamp01(settings.position_prior_weight)
    blended = (base_confidence * (1.0 - weight)) + (prior_likelihood * weight)
    return _clamp01(blended)


def _dedupe_frames(detections: list[DetectedFrame]) -> list[DetectedFrame]:
    by_timestamp: dict[float, DetectedFrame] = {}
    for item in detections:
        current = by_timestamp.get(item.timestamp)
        if current is None or item.confidence > current.confidence:
            by_timestamp[item.timestamp] = item
    return sorted(by_timestamp.values(), key=lambda item: item.timestamp)


def _validate_inputs(
    *,
    jersey_number: int,
    jersey_color: str,
    sport: str,
    position: str | None,
) -> None:
    if not (0 <= int(jersey_number) <= 99):
        raise ValueError("jersey_number must be in range 0..99.")
    if not str(jersey_color).strip():
        raise ValueError("jersey_color must be a non-empty string.")
    if str(sport).strip().lower() not in SUPPORTED_SPORTS:
        raise ValueError("sport must be one of: basketball, football, lacrosse.")


def _preprocess_frame(
    frame_image: np.ndarray,
    jersey_color: str,
    settings: PipelineSettings,
) -> list[ROI]:
    mask = _build_jersey_mask(frame_image, jersey_color, settings)
    rois = _extract_candidate_rois(mask, settings=settings)
    return rois[: settings.max_rois_per_frame] if rois else []


# temporal tricks

_SIMILARITY_THUMB_SIZE = 64


def _frame_to_thumb(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (_SIMILARITY_THUMB_SIZE, _SIMILARITY_THUMB_SIZE))


def _frame_similarity(thumb_a: np.ndarray, thumb_b: np.ndarray) -> float:
    """MSE-based similarity, 1.0 = identical."""
    diff = thumb_a.astype(np.float32) - thumb_b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    # Map MSE to similarity: MSE=0 -> 1.0, MSE=255^2 -> 0.0
    return max(0.0, 1.0 - mse / (255.0 * 255.0))


# detection-first helpers

def _compute_color_ratio_in_box(
    frame_bgr: np.ndarray,
    box: PersonBox,
    jersey_color: str,
    settings: PipelineSettings,
) -> float:
    """What % of the person's torso pixels match the target jersey colour."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(frame_w, int(box.x2))
    y2 = min(frame_h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    # jersey area = top 15%-70% of the person box (skip head + legs)
    box_h = y2 - y1
    torso_y1 = y1 + int(box_h * 0.15)
    torso_y2 = y1 + int(box_h * 0.70)
    torso_y1 = max(y1, torso_y1)
    torso_y2 = min(y2, torso_y2)
    if torso_y2 <= torso_y1:
        return 0.0

    torso_crop = frame_bgr[torso_y1:torso_y2, x1:x2]

    if box.mask is not None:
        # use seg mask so court floor doesn't contaminate the ratio
        local_torso_y1 = torso_y1 - y1
        local_torso_y2 = torso_y2 - y1
        seg_crop = box.mask[local_torso_y1:local_torso_y2, :]
        person_pixels = float(cv2.countNonZero(seg_crop))
        if person_pixels <= 0:
            return 0.0
        # seg mask gives clean edges, no morphology needed
        color_mask = _build_jersey_mask(torso_crop, jersey_color, settings, skip_morphology=True)
        combined = cv2.bitwise_and(color_mask, seg_crop)
        return float(cv2.countNonZero(combined)) / person_pixels
    else:
        color_mask = _build_jersey_mask(torso_crop, jersey_color, settings)
        total_pixels = float(color_mask.shape[0] * color_mask.shape[1])
        if total_pixels <= 0:
            return 0.0
        return float(cv2.countNonZero(color_mask)) / total_pixels


def _filter_persons_by_color(
    frame_bgr: np.ndarray,
    persons: list[PersonBox],
    jersey_color: str,
    settings: PipelineSettings,
    min_color_ratio: float = 0.15,
) -> list[PersonBox]:
    filtered: list[PersonBox] = []
    for person in persons:
        ratio = _compute_color_ratio_in_box(frame_bgr, person, jersey_color, settings)
        if ratio >= min_color_ratio:
            filtered.append(person)
    return filtered


def _color_filter_persons_for_frame(
    frame_bgr: np.ndarray,
    persons: list[PersonBox],
    jersey_color: str,
    settings: PipelineSettings,
    min_person_crop_height: int,
) -> tuple[list[PersonBox], dict[int, float]]:
    filtered_persons = [
        person for person in persons if (person.y2 - person.y1) >= min_person_crop_height
    ]
    if not filtered_persons:
        return [], {}

    color_ratios = {
        idx: _compute_color_ratio_in_box(frame_bgr, person, jersey_color, settings)
        for idx, person in enumerate(filtered_persons)
    }
    matching_persons = [
        person for idx, person in enumerate(filtered_persons)
        if color_ratios.get(idx, 0.0) >= 0.15
    ]
    return matching_persons, color_ratios


def _score_candidates(
    candidates: list[NumberCandidate],
    *,
    sport: str,
    position: str | None,
    frame_width: int,
    frame_height: int,
    settings: PipelineSettings,
) -> tuple[float, NumberCandidate] | None:
    best: tuple[float, NumberCandidate] | None = None
    for candidate in candidates:
        prior_score = _compute_position_likelihood(
            sport=sport,
            position=position,
            candidate=candidate,
            frame_width=frame_width,
            frame_height=frame_height,
            settings=settings,
        )
        blended = _blend_confidence(
            base_confidence=candidate.confidence,
            prior_likelihood=prior_score,
            settings=settings,
        )
        if blended < settings.conf_threshold_internal:
            continue
        if blended >= settings.conf_threshold_export:
            if best is None or blended > best[0]:
                best = (blended, candidate)
    return best


def detect_jersey_in_frames(
    *,
    video_url: str | None,
    video_path: str | None,
    video_bytes: bytes | None,
    jersey_number: int,
    jersey_color: str,
    sport: str,
    position: str | None = None,
    settings: PipelineSettings | None = None,
) -> list[dict[str, float]]:
    _validate_inputs(
        jersey_number=jersey_number,
        jersey_color=jersey_color,
        sport=sport,
        position=position,
    )

    settings = settings or PipelineSettings()
    settings.validate()
    started_at = time.perf_counter()
    detections: list[DetectedFrame] = []
    total_frames = 0
    debug_frames: list[DebugFrameData] = []
    collect_debug = bool(settings.debug_video_path)

    with TemporaryDirectory(prefix="layer1-") as temp_dir:
        work_dir = Path(temp_dir)
        video_file = _resolve_video_source(
            video_url=video_url,
            video_path=video_path,
            video_bytes=video_bytes,
            work_dir=work_dir,
            settings=settings,
        )
        duration_seconds = _get_video_duration_seconds(video_file, settings)
        detector = get_or_create_detector(settings)
        num_workers = max(1, min(settings.pipeline_workers, os.cpu_count() or 1))

        LOGGER.info(
            "Layer1 start duration=%.2fs fps=%s export_threshold=%.2f workers=%s strategy=%s",
            duration_seconds,
            settings.fps,
            settings.conf_threshold_export,
            num_workers,
            settings.detection_strategy,
        )
        if settings.youtube_clip_seconds is not None:
            LOGGER.info(
                "YouTube input clipping enabled for first %ss",
                settings.youtube_clip_seconds,
            )
        if settings.early_exit_consecutive > 0:
            LOGGER.info(
                "Early exit optimization enabled after %s consecutive detections",
                settings.early_exit_consecutive,
            )

        use_detection_first = (
            settings.detection_strategy.strip().lower() == "detection_first"
        )

        # temporal state
        prev_thumb: np.ndarray | None = None
        consecutive_detections = 0
        early_exit = False
        skipped_frames = 0

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for batch_index, batch in enumerate(
                _iter_frames_in_memory(video_path=video_file, settings=settings),
                start=1,
            ):
                if early_exit:
                    break

                batch_detection_count = 0
                valid_frames = [f for f in batch if f.image is not None]
                total_frames += len(valid_frames)

                if not valid_frames:
                    continue

                # Phase 5: skip near-identical frames
                frames_to_process: list[InMemoryFrame] = []
                for frame in valid_frames:
                    thumb = _frame_to_thumb(frame.image)
                    if (
                        prev_thumb is not None
                        and _frame_similarity(prev_thumb, thumb)
                        >= settings.skip_similarity_threshold
                    ):
                        skipped_frames += 1
                        continue
                    prev_thumb = thumb
                    frames_to_process.append(frame)

                if not frames_to_process:
                    continue

                frame_images = [f.image for f in frames_to_process]

                if use_detection_first:
                    # person seg -> colour filter -> number model
                    min_h = settings.min_person_crop_height
                    all_persons = detector.detect_persons_batch(frame_images)
                    color_filter_futures = [
                        pool.submit(
                            _color_filter_persons_for_frame,
                            frame_image,
                            persons,
                            jersey_color,
                            settings,
                            min_h,
                        )
                        for frame_image, persons in zip(frame_images, all_persons)
                    ]
                    color_filter_results = [
                        future.result() for future in color_filter_futures
                    ]

                    for frame, frame_image, persons, color_filter_result in zip(
                        frames_to_process, frame_images, all_persons, color_filter_results
                    ):
                        if early_exit:
                            break

        # Shared debug state for this frame
                        matching_persons: list[PersonBox] = []
                        color_ratios: dict[int, float] = {}
                        candidates: list[NumberCandidate] = []

                        matching_persons, color_ratios = color_filter_result
                        if not persons:
                            consecutive_detections = 0
                            if collect_debug:
                                debug_frames.append(DebugFrameData(
                                    timestamp=frame.timestamp,
                                    image=frame_image,
                                ))
                            continue

                        if not matching_persons:
                            consecutive_detections = 0
                            if collect_debug:
                                debug_frames.append(DebugFrameData(
                                    timestamp=frame.timestamp,
                                    image=frame_image,
                                    persons=persons,
                                    color_ratios=color_ratios,
                                ))
                            continue

                        # Step 3: jersey-number detection on matching crops
                        if (
                            detector.whole_number_detection_enabled
                            or detector.digit_detection_enabled
                        ):
                            candidates = detector.find_digits_in_person_crops(
                                frame_bgr=frame_image,
                                persons=matching_persons,
                                target_number=jersey_number,
                            )
                        else:
                            # Fallback: create candidates from color-matching persons
                            for p in matching_persons:
                                score = _clamp01(
                                    (0.50 * p.confidence) + 0.30
                                )
                                candidates.append(
                                    NumberCandidate(
                                        number=jersey_number,
                                        confidence=score,
                                        digits=tuple(),
                                        x1=p.x1, y1=p.y1,
                                        x2=p.x2, y2=p.y2,
                                    )
                                )

                        if not candidates:
                            consecutive_detections = 0
                            if collect_debug:
                                debug_frames.append(DebugFrameData(
                                    timestamp=frame.timestamp,
                                    image=frame_image,
                                    persons=persons,
                                    color_persons=matching_persons,
                                    color_ratios=color_ratios,
                                ))
                            continue

                        frame_h, frame_w = frame_image.shape[:2]
                        scored = _score_candidates(
                            candidates,
                            sport=sport,
                            position=position,
                            frame_width=frame_w,
                            frame_height=frame_h,
                            settings=settings,
                        )
                        frame_best = scored[0] if scored else None

                        if collect_debug:
                            debug_frames.append(DebugFrameData(
                                timestamp=frame.timestamp,
                                image=frame_image,
                                persons=persons,
                                color_persons=matching_persons,
                                candidates=candidates,
                                best_confidence=frame_best,
                                color_ratios=color_ratios,
                            ))

                        if scored is not None:
                            best_conf, best_cand = scored
                            detections.append(
                                DetectedFrame(
                                    timestamp=frame.timestamp,
                                    confidence=best_conf,
                                    x1=best_cand.x1,
                                    y1=best_cand.y1,
                                    x2=best_cand.x2,
                                    y2=best_cand.y2,
                                    frame_w=frame_w,
                                    frame_h=frame_h,
                                )
                            )
                            batch_detection_count += 1
                            consecutive_detections += 1
                            if (
                                settings.early_exit_consecutive > 0
                                and consecutive_detections
                                >= settings.early_exit_consecutive
                            ):
                                LOGGER.info(
                                    "Early exit after %s consecutive detections",
                                    consecutive_detections,
                                )
                                early_exit = True
                        else:
                            consecutive_detections = 0

                else:
                    # colour-first: HSV mask -> contours -> YOLO per ROI
                    rois_futures = [
                        pool.submit(_preprocess_frame, img, jersey_color, settings)
                        for img in frame_images
                    ]
                    rois_per_frame = [fut.result() for fut in rois_futures]

                    for frame, frame_image, rois in zip(
                        frames_to_process, frame_images, rois_per_frame
                    ):
                        if early_exit:
                            break

                        if not rois:
                            consecutive_detections = 0
                            if collect_debug:
                                debug_frames.append(DebugFrameData(
                                    timestamp=frame.timestamp,
                                    image=frame_image,
                                ))
                            continue

                        candidates = detector.find_target_matches(
                            frame_bgr=frame_image,
                            rois=rois,
                            target_number=jersey_number,
                        )
                        if not candidates:
                            consecutive_detections = 0
                            if collect_debug:
                                debug_frames.append(DebugFrameData(
                                    timestamp=frame.timestamp,
                                    image=frame_image,
                                    rois=rois,
                                ))
                            continue

                        frame_h, frame_w = frame_image.shape[:2]
                        scored = _score_candidates(
                            candidates,
                            sport=sport,
                            position=position,
                            frame_width=frame_w,
                            frame_height=frame_h,
                            settings=settings,
                        )
                        frame_best = scored[0] if scored else None

                        if collect_debug:
                            debug_frames.append(DebugFrameData(
                                timestamp=frame.timestamp,
                                image=frame_image,
                                rois=rois,
                                candidates=candidates,
                                best_confidence=frame_best,
                            ))

                        if scored is not None:
                            best_conf, best_cand = scored
                            detections.append(
                                DetectedFrame(
                                    timestamp=frame.timestamp,
                                    confidence=best_conf,
                                    x1=best_cand.x1,
                                    y1=best_cand.y1,
                                    x2=best_cand.x2,
                                    y2=best_cand.y2,
                                    frame_w=frame_w,
                                    frame_h=frame_h,
                                )
                            )
                            batch_detection_count += 1
                            consecutive_detections += 1
                            if (
                                settings.early_exit_consecutive > 0
                                and consecutive_detections
                                >= settings.early_exit_consecutive
                            ):
                                LOGGER.info(
                                    "Early exit after %s consecutive detections",
                                    consecutive_detections,
                                )
                                early_exit = True
                        else:
                            consecutive_detections = 0

                LOGGER.info(
                    "Processed batch=%s frames=%s (skipped=%s) detections=%s",
                    batch_index,
                    len(batch),
                    skipped_frames,
                    batch_detection_count,
                )

    stable_detections = _dedupe_frames(detections)
    elapsed_s = time.perf_counter() - started_at
    LOGGER.info(
        "Layer1 complete processed_frames=%s skipped=%s final_detections=%s elapsed=%.2fs",
        total_frames,
        skipped_frames,
        len(stable_detections),
        elapsed_s,
    )

    # write debug video if enabled
    if collect_debug and debug_frames:
        try:
            debug_path = write_debug_video(
                debug_frames,
                settings.debug_video_path,  # type: ignore[arg-type]
                fps=settings.fps,
            )
            LOGGER.info("Debug video written to %s", debug_path)
        except Exception as err:
            LOGGER.error("Failed to write debug video: %s", err)
        finally:
            # Free memory
            debug_frames.clear()

    return [item.to_dict() for item in stable_detections]
