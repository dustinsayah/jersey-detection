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
    y_spread = max(cy) - min(cy) if cy else 0
    avg_height = float(np.mean(heights)) if heights else 0

    # If most detections near frame edges — coaches/sideline crew, not on-field play
    edge_count = sum(1 for x in cx if x < 0.08 or x > 0.92)
    if edge_count > n * 0.4:
        return "sideline"

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

    # Transition detection: players spread out but no formation lines.
    # In a real pre-snap formation, offense and defense form two distinct
    # y-bands (line of scrimmage). Walking-between-plays is a single blob
    # with even y-distribution. Look for the largest y-gap between adjacent
    # mid-field players — a real LOS produces a meaningful gap.
    # v8.31.4: reverted to strict 0.04/0.30/count>=8 thresholds. The looser
    # v8.31.3 thresholds caught more transitions but also flagged real
    # pre-snap formations as transitions when the camera angle made both
    # team y-bands sit close together (1686s real play came back as 0 clips).
    middle_y = sorted(y for y in cy if 0.25 < y < 0.75)
    if len(middle_y) >= 8:
        gaps = [middle_y[i + 1] - middle_y[i] for i in range(len(middle_y) - 1)]
        max_gap = max(gaps) if gaps else 0.0
        if max_gap < 0.04 and y_spread < 0.30:
            return "transition"

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
    """Classify a player crop as 'target' or 'opponent' based on jersey color.

    v8.33.4: discriminates by PARTNER color presence rather than the user's
    primary color. Many teams' user-supplied color (e.g. "navy") is the
    accent that BOTH teams' jerseys carry as numbers/trim — so checking
    primary-color overlap mislabels everything as target. Partner colors
    (yellow/gold for navy, etc.) are the team's distinctive base hue and
    only appear on the actual target team. Fallback to primary if we have
    no partner palette for the user color.
    """
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

    # Partner-color discriminator: pick the team-distinctive partner colors
    # for the user-supplied accent. Vision-verified on St. Mark's 2024
    # footage: yellow appears in 10-20% of target torsos, 0% of opponent
    # torsos — a much cleaner signal than navy itself.
    from app.services.team_classifier import _expand_target_palette
    palette = _expand_target_palette(key)
    partner_colors = [c for c in palette[1:] if c in color_ranges]

    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Compute partner-color presence (the discriminative signal)
        partner_max = 0.0
        for pc in partner_colors:
            lower_p, upper_p = color_ranges[pc]
            mask_p = cv2.inRange(hsv, np.array(lower_p), np.array(upper_p))
            partner_max = max(partner_max, np.sum(mask_p > 0) / mask_p.size)

        # Compute primary-color presence (legacy signal, kept as backstop)
        lower, upper = color_ranges[key]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        primary_ratio = np.sum(mask > 0) / mask.size

        # Decision rule:
        # 1. partner_max > 0.005 → "target" (any visible yellow/gold means
        #    this is a target-team player; threshold tuned empirically on
        #    St. Mark's footage where target torsos average 0.001-0.16
        #    yellow ratio and opponent torsos average ~0.0).
        # 2. partner_max == 0 AND primary_ratio > 0.06 → "opponent"
        #    (player wears the accent color but no partner — typical of
        #    opponents whose base jerseys are blue/white/navy without
        #    yellow accents).
        # 3. otherwise → "unknown" (target player with back to camera, in
        #    shadow, or too far away to register yellow). Returning
        #    "unknown" instead of defaulting to "opponent" stops the
        #    downstream filter from cutting target plays where most
        #    target players happen to be facing away.
        if partner_colors:
            if partner_max > 0.005:
                return "target"
            if primary_ratio > 0.06:
                return "opponent"
            return "unknown"
        return "target" if primary_ratio > 0.06 else "opponent"
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

    # SigLIP team classifier — falls back to HSV during warmup and on any failure.
    # v8.33.1: replaces fixed HSV ranges with data-driven 2-cluster KMeans on
    # SigLIP embeddings of torso crops. Robust to unusual jersey colors and
    # broadcast lighting.
    from app.services.team_classifier import TeamClassifier
    team_clf = TeamClassifier(target_color=target_color)

    # v8.34.0: QB position detector — formation-clustering approach (no new
    # model). Identifies the line of scrimmage as the densest target-team
    # y-band and the QB as the target player furthest behind it. Pure NumPy.
    # Only applied when position == "quarterback".
    from app.services.qb_detector import QBDetector
    qb_detector = QBDetector() if position.lower() == "quarterback" else None

    # Moments where play is happening with target visible or target team on field
    play_moments: list[dict] = []

    frame_idx = 0
    frames_processed = 0
    last_progress = 0
    # v8.32.0: real frame-diff motion. At 360p categories overlap with camera
    # pans (walking+pan can exceed a tight-zoom play), so this is recorded as
    # a signal only — no hard-drop filter until 720p data validates separation.
    prev_gray_motion: Optional[np.ndarray] = None

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

        # Real frame-diff motion (computed before YOLO so it's always available)
        gray_motion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray_motion is not None and prev_gray_motion.shape == gray_motion.shape:
            real_motion = float(cv2.absdiff(prev_gray_motion, gray_motion).mean())
        else:
            real_motion = 0.0
        prev_gray_motion = gray_motion

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
            opp_team_count = 0  # v8.33.4: explicit opp count (was inferred)
            # v8.34.0: collect target-team bboxes for QB position detection
            target_bbox_list: list[np.ndarray] = []
            target_track_id_list: list[int] = []

            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                bbox = tracked.xyxy[i]

                # Torso crop for team classification
                crop = get_torso_crop(frame, bbox)
                if crop is None:
                    continue

                team = team_clf.predict(crop, track_id=track_id)
                if team == "target":
                    target_team_count += 1
                    target_bbox_list.append(bbox)
                    target_track_id_list.append(track_id)
                elif team == "opponent":
                    opp_team_count += 1
                # else "unknown" — don't count as either team

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
            # Keep moments where the target jersey OCR fired, OR any
            # yellow torso was seen (target team is on the field). The
            # earlier >= 3 threshold was tuned to v8.33.1's permissive
            # HSV which counted nearly everyone as target; v8.33.4's
            # selective HSV typically reports 1-4 yellow torsos for a
            # target play, so the gate is lowered to 1 to match.
            if target_seen or target_team_count >= 1:
                score = (
                    (80 if target_seen else 0)
                    + (best_jersey_conf * 30)
                    + (min(target_team_count, 5) * 5)
                    + (min(len(tracked), 15) * 2)
                )
                # Asymmetric team-majority rule (v8.33.4):
                #   target_team_count >= 1  → "target"  (any visible yellow
                #     torso means the target team is on the field — a
                #     high-confidence signal because the opponent never
                #     wears yellow on St. Mark's footage)
                #   opp_team_count >= 1     → "opponent"
                #   otherwise               → "mixed"
                # Yellow DETECTION is high-confidence; yellow ABSENCE is
                # not (target players facing away show no yellow). So we
                # trust the positive signal asymmetrically.
                if target_team_count >= 1:
                    team_majority = "target"
                elif opp_team_count >= 1:
                    team_majority = "opponent"
                else:
                    team_majority = "mixed"
                # v8.34.0: per-frame QB position analysis (only for QB jobs)
                formation_dict: dict | None = None
                if qb_detector is not None and len(target_bbox_list) >= 4:
                    target_bb = np.array(target_bbox_list, dtype=float)
                    target_ids = np.array(target_track_id_list, dtype=int)
                    formation = qb_detector.analyze_frame(target_bb, target_ids, fh, fw)
                    formation_dict = formation.as_dict()

                play_moments.append({
                    "timestamp": round(timestamp, 1),
                    "player_count": len(tracked),
                    "target_visible": target_seen,
                    "jersey_conf": round(best_jersey_conf, 3),
                    "target_team_count": target_team_count,
                    "team_majority": team_majority,
                    "score": round(score),
                    "state": state,
                    "real_motion": round(real_motion, 2),
                    "formation": formation_dict,
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

    # 7. Cluster moments into clips (passes position so finalize can score)
    clips = _cluster_to_clips(play_moments, duration, target_jersey, position)

    LOGGER.info("ByteTrack: %d clips assembled in %.1fs total", len(clips), elapsed)

    # Free models
    _free_models()

    return clips


def _cluster_to_clips(
    moments: list[dict],
    video_duration: float,
    target_jersey: str,
    position: str = "quarterback",
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
        # Require at least 2 valid play moments per cluster — singletons are
        # almost always frame-state classifier flukes (a single misread frame).
        if len(cluster) < 2:
            continue
        clip = _finalize_clip(cluster, video_duration, target_jersey, position)
        if clip:
            clips.append(clip)

    # Sort by score descending
    clips.sort(key=lambda c: c.get("score", 0), reverse=True)
    return clips[:15]


def _finalize_clip(
    moments: list[dict],
    video_duration: float,
    target_jersey: str,
    position: str = "quarterback",
) -> Optional[dict]:
    """Convert a cluster of play moments into a single clip."""
    if not moments:
        return None

    # v8.31.3: drop suspiciously long clusters. Real football plays produce
    # 5-15 sampled "play" frames at 1fps. Clusters with >25 moments are
    # almost always long static dead-ball shots (timeouts, end-of-quarter,
    # injury delays) where the camera holds on a wide field view with
    # players present but not playing. Vision-verified examples: clip at
    # 2338s with mom=45 was a 30+s static dead ball.
    if len(moments) > 25:
        return None

    peak = max(moments, key=lambda m: m["score"])
    peak_time = peak["timestamp"]

    # v8.31.3: extend window end from peak+8 to peak+12. The score peaks
    # when OCR fires (typically pre-snap when QB jersey faces camera). With
    # the old +8s end, plays where peak hit during huddle/walkup cut off
    # before the actual snap. Vision-verified examples: clips 1/4/11 in
    # the v8.31.2 full-game run all showed pre-snap formation at end frame
    # because the actual play happened past the +8s boundary.
    start = max(0, peak_time - 3)
    end = min(video_duration, peak_time + 12)

    # v8.31: Multi-frame persistence for target visibility.
    # A single-frame OCR misread can flip target_visible=True on one moment
    # and give the whole cluster a "jersey confirmed" stamp. Require the
    # target to be seen in multiple frames OR the cluster to be very tight
    # (≤3 moments where 1-frame visibility is plausible).
    target_visible_count = sum(1 for m in moments if m["target_visible"])
    target_visibility_ratio = target_visible_count / len(moments)
    jersey_confirmed = (
        target_visible_count >= 2
        or (len(moments) <= 3 and target_visible_count >= 1)
    )
    max_score = max(m["score"] for m in moments)
    avg_players = int(np.mean([m["player_count"] for m in moments]))
    max_jersey_conf = max(m["jersey_conf"] for m in moments)

    # v8.32.0: real motion stats per clip. Recorded only — no filter applied at
    # 360p (categories overlap with camera pan). Used for downstream analysis
    # and will become a filter once 720p downloads land.
    _rm_vals = [m.get("real_motion", 0.0) for m in moments if m.get("real_motion") is not None]
    real_motion_avg = float(np.mean(_rm_vals)) if _rm_vals else 0.0
    real_motion_max = float(max(_rm_vals)) if _rm_vals else 0.0

    # v8.33.1: team-majority signal (SigLIP-clustered). If the cluster is
    # mostly opponent-on-offense, the clip is showing the wrong team's play —
    # demote to Cut so the post-filter drops it.
    _team_counts = {"target": 0, "opponent": 0, "mixed": 0, "unknown": 0}
    for m in moments:
        tm = m.get("team_majority", "unknown")
        _team_counts[tm] = _team_counts.get(tm, 0) + 1
    opponent_majority = (
        _team_counts["opponent"] > _team_counts["target"]
        and _team_counts["opponent"] >= len(moments) * 0.5
    )

    # Determine play type
    if jersey_confirmed:
        play_type = "game_action"
    elif max(m["target_team_count"] for m in moments) >= 4:
        play_type = "game_action"
    else:
        play_type = "formation"

    # v8.31: Grade respects target persistence — a +80 single-frame OCR boost
    # shouldn't promote a clip to "Strong" if the target was barely visible.
    # v8.33.1: opponent-majority forces grade = Cut regardless of score, since
    # the clip is showing the other team's play.
    if opponent_majority and not jersey_confirmed:
        grade = "Cut"
    elif max_score >= 60 and target_visibility_ratio >= 0.20:
        grade = "Strong"
    elif max_score >= 35:
        grade = "Decent"
    else:
        grade = "Cut"

    # Build signals dict compatible with existing pipeline consumers.
    # v8.31.1: signals.jersey is the post-filter's "jersey confirmed" gate
    # (jconf >= 0.20). Without persistence gating here, a single-frame OCR
    # misread (jersey_conf=0.21, target_visible in 1/4 frames) would pass
    # the gate and bypass the dead-ball filter — re-introducing the exact
    # 461s false positive we just filtered. Mirror the persistence-gated
    # jersey_confirmed flag into signals.jersey so both filters agree.
    motion_score = avg_players * 5  # 10 players → 50 motion score
    signals = {
        "jersey": round(max_jersey_conf, 3) if jersey_confirmed else 0.0,
        "motion": round(motion_score, 1),
        "audio": False,
        "audio_confidence": 0.0,
        "pose": "standing",
        "crowd": 0.0,
        "v4_outcome": None,
        "jersey_dist": 0.0 if jersey_confirmed else 9999.0,
    }

    # Build the base clip dict first so we can pass it to the position scorer.
    clip_dict = {
        "startTime": round(start, 1),
        "endTime": round(end, 1),
        "playType": play_type,
        "confidence": round(max_jersey_conf, 3),
        "score": round(max_score),
        "jerseyDetected": jersey_confirmed,
        "jerseyConfidence": round(max_jersey_conf, 3),
        "jerseyVisible": jersey_confirmed,
        "jerseyNumberSeen": int(target_jersey) if jersey_confirmed else None,
        "trackingId": None,
        "description": "Game Action" if play_type == "game_action" else "Formation",
        "signals": signals,
        "motionScore": round(motion_score, 1),
        "grade": grade,
        "peakTimestamp": peak_time,
        "playerCount": avg_players,
        "momentCount": len(moments),
        "targetVisibleFrames": target_visible_count,
        "targetVisibilityRatio": round(target_visibility_ratio, 3),
        "recruitingScore": round(max_score),
        "playerSpecificScore": round(max_score * 0.5),
        "realMotionAvg": round(real_motion_avg, 2),
        "realMotionMax": round(real_motion_max, 2),
        "teamMajority": "opponent" if opponent_majority else (
            "target" if _team_counts["target"] >= _team_counts["opponent"] else "mixed"
        ),
        "teamMajorityCounts": {k: int(v) for k, v in _team_counts.items()},
        "caption": f"{'#' + target_jersey + ' — ' if jersey_confirmed else ''}{play_type.replace('_', ' ').title()}",
    }

    # v8.34.0: position-specific scoring rubric.
    # Replaces the generic recruitingScore with a rubric tuned to what college
    # coaches actually want to see for the requested position.  The legacy
    # max_score is preserved as `legacyScore` for debugging.
    try:
        from app.services.position_scorer import score_clip
        formations = [m.get("formation") for m in moments if m.get("formation")]
        scoring = score_clip(position, clip_dict, moments, formations or None)
        scoring_dict = scoring.as_dict()
        clip_dict["legacyScore"] = clip_dict["recruitingScore"]
        clip_dict["recruitingScore"] = round(scoring.final_score)
        clip_dict["positionScore"] = scoring_dict
        clip_dict["grade"] = scoring.grade
        # qb_track_id consensus for downstream use (spotlight, render hint)
        if formations:
            from app.services.qb_detector import vote_qb_track_id
            from app.services.qb_detector import FormationFrame  # noqa: F401
            # vote_qb_track_id wants FormationFrame objects; rebuild lite shells
            shells = []
            for f in formations:
                shell = type("F", (), {})()
                shell.qb_track_id = f.get("qb_track_id")
                shell.qb_confidence = f.get("qb_confidence", 0.0)
                shells.append(shell)
            qb_id = vote_qb_track_id(shells)
            clip_dict["qbTrackId"] = qb_id
    except Exception as exc:
        LOGGER.warning("position_scorer failed (%s) — keeping legacy score", exc)

    return clip_dict
