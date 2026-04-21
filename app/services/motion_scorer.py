# Motion scoring via Farneback dense optical flow (OpenCV)

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Downsample width for speed (~5ms/frame on CPU)
FLOW_WIDTH = 320

# Flow parameters
FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


@dataclass
class MotionScore:
    score: float  # 0-100 normalized
    raw_magnitude: float  # Mean optical flow magnitude
    camera_motion: float  # Estimated camera motion magnitude
    player_motion: float  # Motion after camera subtraction


def compute_motion_score(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    person_boxes: list[tuple[float, float, float, float]] | None = None,
) -> MotionScore:
    """Compute motion score between two consecutive frames.

    Args:
        frame_prev: Previous BGR frame
        frame_curr: Current BGR frame
        person_boxes: Optional list of (x1, y1, x2, y2) person bounding boxes.
            If provided, computes flow only within person regions.

    Returns:
        MotionScore with 0-100 normalized score
    """
    try:
        # Downsample for speed
        h, w = frame_prev.shape[:2]
        scale = FLOW_WIDTH / w if w > FLOW_WIDTH else 1.0
        new_w = int(w * scale)
        new_h = int(h * scale)

        prev_small = cv2.resize(frame_prev, (new_w, new_h))
        curr_small = cv2.resize(frame_curr, (new_w, new_h))

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **FLOW_PARAMS
        )

        # Full-frame magnitude
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        raw_magnitude = float(mag.mean())

        # Camera motion estimation: median flow vector
        median_dx = float(np.median(flow[..., 0]))
        median_dy = float(np.median(flow[..., 1]))
        camera_motion = float(np.sqrt(median_dx ** 2 + median_dy ** 2))

        # Subtract camera motion
        flow_corrected = flow.copy()
        flow_corrected[..., 0] -= median_dx
        flow_corrected[..., 1] -= median_dy

        # If person boxes provided, compute flow only within person regions
        if person_boxes:
            mask = np.zeros((new_h, new_w), dtype=bool)
            for box in person_boxes:
                bx1 = max(0, int(box[0] * scale))
                by1 = max(0, int(box[1] * scale))
                bx2 = min(new_w, int(box[2] * scale))
                by2 = min(new_h, int(box[3] * scale))
                mask[by1:by2, bx1:bx2] = True

            if mask.any():
                player_mag = np.sqrt(
                    flow_corrected[mask, 0] ** 2 + flow_corrected[mask, 1] ** 2
                )
                player_motion = float(player_mag.mean())
            else:
                corrected_mag = np.sqrt(
                    flow_corrected[..., 0] ** 2 + flow_corrected[..., 1] ** 2
                )
                player_motion = float(corrected_mag.mean())
        else:
            corrected_mag = np.sqrt(
                flow_corrected[..., 0] ** 2 + flow_corrected[..., 1] ** 2
            )
            player_motion = float(corrected_mag.mean())

        # Normalize to 0-100
        # Empirical: 0 flow → 0, ~5 pixels/frame → 50, ~10+ → 100
        score = min(100.0, player_motion * 10.0)

        return MotionScore(
            score=round(score, 1),
            raw_magnitude=round(raw_magnitude, 3),
            camera_motion=round(camera_motion, 3),
            player_motion=round(player_motion, 3),
        )

    except Exception as exc:
        LOGGER.warning("Motion scoring failed: %s", exc)
        return MotionScore(score=0.0, raw_magnitude=0.0, camera_motion=0.0, player_motion=0.0)


def compute_motion_scores_batch(
    frames: list[np.ndarray],
    person_boxes_per_frame: list[list[tuple[float, float, float, float]] | None] | None = None,
) -> list[MotionScore]:
    """Compute motion scores for a sequence of frames.

    Returns one MotionScore per consecutive frame pair (len = len(frames) - 1).
    """
    scores: list[MotionScore] = []
    if len(frames) < 2:
        return scores

    boxes_list = person_boxes_per_frame or [None] * len(frames)

    for i in range(len(frames) - 1):
        boxes = boxes_list[i + 1] if i + 1 < len(boxes_list) else None
        score = compute_motion_score(frames[i], frames[i + 1], boxes)
        scores.append(score)

    return scores
