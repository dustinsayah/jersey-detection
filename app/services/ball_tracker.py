# Ball tracker — detects and tracks ball position per sport
# Loads sport-specific ball detection models lazily.
# If model file missing → log warning, return empty, never crash.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

_BALL_MODELS = {
    "basketball": "basketball_ball_detector.pt",
    "football": "football_ball_detector.pt",
    "lacrosse": "lacrosse_ball_detector.pt",
}


@dataclass
class BallFrame:
    ball_visible: bool = False
    ball_position: list[float] = field(default_factory=lambda: [0.0, 0.0])
    ball_trajectory: str = "unknown"
    ball_confidence: float = 0.0


class BallTracker:
    """Track ball position across frames using sport-specific YOLO models."""

    def __init__(self):
        self._models: dict = {}
        self._prev_positions: list[list[float]] = []

    def _load_model(self, sport: str):
        if sport in self._models:
            return self._models[sport]
        filename = _BALL_MODELS.get(sport.lower())
        if not filename:
            logger.info("ball_tracker: no ball model defined for sport=%s", sport)
            self._models[sport] = None
            return None
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            logger.warning("ball_tracker: model not found %s — ball tracking disabled", path)
            self._models[sport] = None
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            logger.info("ball_tracker: loaded %s", filename)
            self._models[sport] = model
            return model
        except Exception as e:
            logger.error("ball_tracker: failed to load %s: %s", filename, e)
            self._models[sport] = None
            return None

    def _estimate_trajectory(self, current_pos: list[float], frame_width: int) -> str:
        """Estimate ball trajectory from recent positions."""
        if len(self._prev_positions) < 2:
            return "unknown"
        prev = self._prev_positions[-1]
        dx = current_pos[0] - prev[0]
        # Positive x = moving right (typically toward basket/goal)
        midpoint = frame_width / 2
        if current_pos[0] > midpoint:
            return "toward_basket" if dx > 0 else "away_from_basket"
        else:
            return "toward_basket" if dx > 0 else "away_from_basket"

    def track_frame(
        self, frame: np.ndarray, sport: str, conf: float = 0.3
    ) -> BallFrame:
        """Detect ball in a single frame."""
        model = self._load_model(sport)
        if model is None:
            return BallFrame()
        try:
            results = model(frame, conf=conf, verbose=False)[0]
            best_conf = 0.0
            best_pos = [0.0, 0.0]
            for box in results.boxes:
                name = model.names[int(box.cls)].lower()
                if "ball" in name or "basketball" in name or "football" in name:
                    c = float(box.conf)
                    if c > best_conf:
                        best_conf = c
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        best_pos = [(x1 + x2) / 2, (y1 + y2) / 2]
            if best_conf > 0:
                h, w = frame.shape[:2]
                trajectory = self._estimate_trajectory(best_pos, w)
                self._prev_positions.append(best_pos)
                if len(self._prev_positions) > 10:
                    self._prev_positions = self._prev_positions[-10:]
                return BallFrame(
                    ball_visible=True,
                    ball_position=best_pos,
                    ball_trajectory=trajectory,
                    ball_confidence=round(best_conf, 3),
                )
        except Exception as e:
            logger.error("ball_tracker: detection error: %s", e)
        return BallFrame()

    def reset(self):
        self._prev_positions.clear()
