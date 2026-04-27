"""
ByteTrack Detection Pipeline v1 — YOLOv8 + ByteTrack + EasyOCR

Replaces the multi-layer detection pipeline with a cleaner approach:
  1. YOLOv8n detects all people in each frame
  2. supervision ByteTrack assigns persistent track IDs
  3. EasyOCR reads jersey numbers from torso crops
  4. Play state detection (formation vs dead ball vs sideline)
  5. Clip assembly around confirmed plays with target jersey
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Lazy-loaded singletons to avoid model loading on import
_yolo_model = None
_ocr_reader = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        LOGGER.info("ByteTrack: loading YOLOv8n detector...")
        _yolo_model = YOLO("yolov8n.pt")
        LOGGER.info("ByteTrack: YOLOv8n loaded")
    return _yolo_model


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        LOGGER.info("ByteTrack: loading EasyOCR reader...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        LOGGER.info("ByteTrack: EasyOCR loaded")
    return _ocr_reader


def _free_models():
    """Free model memory after pipeline run."""
    global _yolo_model, _ocr_reader
    _yolo_model = None
    _ocr_reader = None
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def detect_people(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect people in a frame. Returns (xyxy_boxes, confidences)."""
    model = _get_yolo()
    results = model(frame, classes=[0], verbose=False, device="cpu")[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.empty((0, 4)), np.empty(0)
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    # Filter low confidence
    mask = confs > 0.25
    return boxes[mask], confs[mask]


def get_torso_crop(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Extract the jersey/torso region from a bounding box."""
    x1, y1, x2, y2 = bbox.astype(int)
    h = y2 - y1
    w = x2 - x1
    if h < 20 or w < 10:
        return None

    # Torso: upper-middle region of person bbox
    ty1 = y1 + int(h * 0.20)
    ty2 = y1 + int(h * 0.55)
    tx1 = x1 + int(w * 0.15)
    tx2 = x2 - int(w * 0.15)

    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    ty1, ty2 = max(0, ty1), min(fh, ty2)
    tx1, tx2 = max(0, tx1), min(fw, tx2)

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    return crop


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Preprocess a jersey crop for better OCR accuracy."""
    # Upscale to at least 128px tall
    h, w = crop.shape[:2]
    scale = max(1.0, 128.0 / h)
    if scale > 1:
        new_h = int(h * scale * 2)
        new_w = int(w * scale * 2)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened


def read_jersey_number(crop: np.ndarray) -> tuple[Optional[str], float]:
    """Read a jersey number from a preprocessed crop."""
    if crop is None:
        return None, 0.0

    try:
        processed = preprocess_for_ocr(crop)
        reader = _get_ocr()
        results = reader.readtext(processed, allowlist="0123456789", detail=1)

        best_text = None
        best_conf = 0.0

        for _bbox, text, conf in results:
            text = text.strip()
            if text.isdigit() and 1 <= len(text) <= 2 and conf > best_conf:
                best_text = text
                best_conf = conf

        return best_text, best_conf
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------------
# Play state classification
# ---------------------------------------------------------------------------

def classify_frame_state(
    boxes: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> str:
    """Classify the frame as 'play', 'dead_ball', or 'sideline'.

    Heuristics:
    - 'play': >= 8 people, spread across field, reasonable size
    - 'dead_ball': people present but clustered (huddle)
    - 'sideline': most people near bottom edge (camera showing sideline)
    """
    n = len(boxes)
    if n < 5:
        return "sideline"

    # Player center X positions normalized
    cx = [(b[0] + b[2]) / 2 / frame_w for b in boxes]
    cy = [(b[1] + b[3]) / 2 / frame_h for b in boxes]
    heights = [(b[3] - b[1]) / frame_h for b in boxes]

    x_spread = max(cx) - min(cx) if cx else 0
    avg_height = float(np.mean(heights)) if heights else 0

    # If players near bottom of frame only — sideline shot
    bottom_count = sum(1 for y in cy if y > 0.75)
    if bottom_count > n * 0.6:
        return "sideline"

    # If average player height > 40% of frame — zoomed in, not a play
    if avg_height > 0.4:
        return "sideline"

    # If players clustered (low spread) — huddle or dead ball
    if x_spread < 0.25:
        return "dead_ball"

    # If enough players spread across field — active play
    if n >= 8 and x_spread > 0.35:
        return "play"

    # Moderate player count, moderate spread — could be formation
    if n >= 6 and x_spread > 0.3:
        return "play"

    return "dead_ball"


def classify_team_hsv(
    crop: np.ndarray,
    target_color: str,
) -> str:
    """Classify a player crop as 'target' or 'opponent' based on jersey color."""
    if crop is None or crop.size == 0:
        return "unknown"

    # HSV ranges for common jersey colors
    color_ranges = {
        "navy":  ([100, 40, 10],  [130, 255, 120]),
        "blue":  ([100, 80, 50],  [130, 255, 255]),
        "white": ([0, 0, 180],    [180, 40, 255]),
        "red":   ([0, 100, 80],   [10, 255, 255]),
        "black": ([0, 0, 0],      [180, 255, 60]),
        "green": ([40, 50, 50],   [80, 255, 255]),
        "gold":  ([15, 80, 80],   [35, 255, 255]),
        "yellow":([20, 80, 80],   [35, 255, 255]),
        "orange":([10, 100, 100], [20, 255, 255]),
        "purple":([130, 50, 30],  [160, 255, 255]),
        "maroon":([0, 50, 30],    [10, 200, 150]),
    }

    key = target_color.lower().strip()
    if key not in color_ranges:
        return "unknown"

    lower, upper = color_ranges[key]
    lower = np.array(lower)
    upper = np.array(upper)

    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.sum(mask > 0) / mask.size
        return "target" if ratio > 0.06 else "opponent"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_bytetrack_detection(
    video_path: str,
    target_jersey: str,
    target_color: str = "white",
    position: str = "quarterback",
    sample_fps: float = 1.0,
    progress_callback: Any = None,
) -> list[dict]:
    """
    Run the full ByteTrack detection pipeline on a video.

    Returns list of clip dicts compatible with the existing pipeline output format.
    """
    import supervision as sv

    t_start = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.error("ByteTrack: cannot open video %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = max(1, int(fps / sample_fps))

    LOGGER.info(
        "ByteTrack: video=%.0fs fps=%.1f frames=%d interval=%d target=#%s(%s) pos=%s",
        duration, fps, total_frames, frame_interval, target_jersey, target_color, position,
    )

    # Initialize ByteTrack tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=45,  # 1.5s at 30fps
        minimum_matching_threshold=0.85,
        frame_rate=int(fps),
        minimum_consecutive_frames=3,
    )

    # Per-track jersey accumulator: {track_id: {jersey_num_str: count}}
    track_jerseys: dict[int, dict[str, int]] = {}

    # Moments where play is happening with target visible or target team on field
    play_moments: list[dict] = []

    frame_idx = 0
    frames_processed = 0
    last_progress = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        frames_processed += 1
        timestamp = frame_idx / fps
        fh, fw = frame.shape[:2]

        # Progress reporting
        pct = int(frame_idx / total_frames * 100)
        if progress_callback and pct >= last_progress + 5:
            last_progress = pct
            LOGGER.info("ByteTrack: %d%% (%d frames)", pct, frames_processed)

        try:
            # 1. Detect people
            boxes, confs = detect_people(frame)
            if len(boxes) == 0:
                continue

            # 2. Build supervision Detections
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confs,
                class_id=np.zeros(len(boxes), dtype=int),
            )

            # 3. Track with ByteTrack
            tracked = tracker.update_with_detections(detections)
            if tracked.tracker_id is None or len(tracked) == 0:
                continue

            # 4. Classify frame state
            state = classify_frame_state(tracked.xyxy, fh, fw)
            if state != "play":
                continue

            # 5. For each tracked person: team color + jersey OCR
            target_seen = False
            best_jersey_conf = 0.0
            target_team_count = 0

            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                bbox = tracked.xyxy[i]

                # Torso crop for team classification
                crop = get_torso_crop(frame, bbox)
                if crop is None:
                    continue

                team = classify_team_hsv(crop, target_color)
                if team == "target":
                    target_team_count += 1

                # Jersey OCR (only on every 3rd tracked player to save CPU)
                if i % 3 == 0 or target_team_count > 0:
                    jersey_num, jersey_conf = read_jersey_number(crop)
                    if jersey_num:
                        if track_id not in track_jerseys:
                            track_jerseys[track_id] = {}
                        track_jerseys[track_id][jersey_num] = (
                            track_jerseys[track_id].get(jersey_num, 0) + 1
                        )

                        if jersey_num == str(target_jersey) and jersey_conf > best_jersey_conf:
                            target_seen = True
                            best_jersey_conf = jersey_conf

            # 6. Score this moment
            # Only keep moments where we see target team or target jersey
            if target_seen or target_team_count >= 3:
                score = (
                    (80 if target_seen else 0)
                    + (best_jersey_conf * 30)
                    + (min(target_team_count, 5) * 5)
                    + (min(len(tracked), 15) * 2)
                )
                play_moments.append({
                    "timestamp": round(timestamp, 1),
                    "player_count": len(tracked),
                    "target_visible": target_seen,
                    "jersey_conf": round(best_jersey_conf, 3),
                    "target_team_count": target_team_count,
                    "score": round(score),
                    "state": state,
                })

        except Exception as exc:
            if frames_processed < 5:
                LOGGER.error("ByteTrack: frame %d error: %s", frame_idx, exc)
            continue

    cap.release()
    elapsed = time.perf_counter() - t_start

    LOGGER.info(
        "ByteTrack: done — %d frames in %.1fs, %d play moments, %d tracked players",
        frames_processed, elapsed, len(play_moments), len(track_jerseys),
    )

    # Log jersey number votes
    for tid, votes in sorted(track_jerseys.items()):
        best = max(votes, key=votes.get) if votes else "?"
        LOGGER.info("ByteTrack: track #%d → jersey votes: %s (best: #%s)", tid, votes, best)

    # 7. Cluster moments into clips
    clips = _cluster_to_clips(play_moments, duration, target_jersey)

    LOGGER.info("ByteTrack: %d clips assembled in %.1fs total", len(clips), elapsed)

    # Free models
    _free_models()

    return clips


def _cluster_to_clips(
    moments: list[dict],
    video_duration: float,
    target_jersey: str,
) -> list[dict]:
    """Cluster play moments into clip segments."""
    if not moments:
        return []

    moments = sorted(moments, key=lambda m: m["timestamp"])

    clusters: list[list[dict]] = []
    current: list[dict] = [moments[0]]

    for m in moments[1:]:
        # Gap > 20s → new cluster
        if m["timestamp"] - current[-1]["timestamp"] > 20:
            clusters.append(current)
            current = [m]
        else:
            current.append(m)
    clusters.append(current)

    clips = []
    for cluster in clusters:
        clip = _finalize_clip(cluster, video_duration, target_jersey)
        if clip:
            clips.append(clip)

    # Sort by score descending
    clips.sort(key=lambda c: c.get("score", 0), reverse=True)
    return clips[:15]


def _finalize_clip(
    moments: list[dict],
    video_duration: float,
    target_jersey: str,
) -> Optional[dict]:
    """Convert a cluster of play moments into a single clip."""
    if not moments:
        return None

    peak = max(moments, key=lambda m: m["score"])
    peak_time = peak["timestamp"]

    # Clip window: 3s before peak, 8s after
    start = max(0, peak_time - 3)
    end = min(video_duration, peak_time + 8)

    jersey_confirmed = any(m["target_visible"] for m in moments)
    max_score = max(m["score"] for m in moments)
    avg_players = int(np.mean([m["player_count"] for m in moments]))
    max_jersey_conf = max(m["jersey_conf"] for m in moments)

    # Determine play type
    if jersey_confirmed:
        play_type = "game_action"
    elif max(m["target_team_count"] for m in moments) >= 4:
        play_type = "game_action"
    else:
        play_type = "formation"

    # Determine grade
    if max_score >= 60:
        grade = "Strong"
    elif max_score >= 35:
        grade = "Decent"
    else:
        grade = "Cut"

    return {
        "startTime": round(start, 1),
        "endTime": round(end, 1),
        "playType": play_type,
        "score": round(max_score),
        "jerseyDetected": jersey_confirmed,
        "jerseyConfidence": round(max_jersey_conf, 3),
        "motionScore": avg_players * 3,
        "grade": grade,
        "peakTimestamp": peak_time,
        "playerCount": avg_players,
        "momentCount": len(moments),
        # Fields expected by existing pipeline consumers
        "jerseyVisible": jersey_confirmed,
        "recruitingScore": round(max_score),
        "playerSpecificScore": round(max_score * 0.5),
    }
