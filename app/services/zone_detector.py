# Zone detector — identifies which zone of court/field the action is in
# Primary: trained zone model (if available)
# Fallback: geometry estimation from player position in frame

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

BASKETBALL_ZONES = ["paint", "midrange", "three_point", "backcourt", "transition"]
FOOTBALL_ZONES = ["own_endzone", "own_territory", "midfield", "opponent_territory", "redzone", "opponent_endzone"]
LACROSSE_ZONES = ["defensive", "midfield", "attack", "crease"]


@dataclass
class ZoneResult:
    zone: str = "unknown"
    zone_confidence: float = 0.0
    method: str = "geometry_estimate"


class ZoneDetector:
    """Detect court/field zone from frame content or player position."""

    def __init__(self):
        self._model = None
        self._model_loaded = False

    def _try_load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        path = os.path.join(MODEL_DIR, "zone_detector.pt")
        if os.path.exists(path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(path)
                logger.info("zone_detector: model loaded")
            except Exception as e:
                logger.warning("zone_detector: model load failed: %s", e)

    def detect_zone(
        self,
        frame: np.ndarray,
        sport: str,
        player_bbox: list[float] | None = None,
    ) -> ZoneResult:
        """Detect zone using model or geometry fallback."""
        self._try_load_model()

        # Try model first
        if self._model is not None:
            try:
                results = self._model(frame, conf=0.3, verbose=False)[0]
                if results.boxes and len(results.boxes) > 0:
                    best = max(results.boxes, key=lambda b: float(b.conf))
                    zone_name = self._model.names[int(best.cls)]
                    return ZoneResult(
                        zone=zone_name,
                        zone_confidence=round(float(best.conf), 3),
                        method="model",
                    )
            except Exception as e:
                logger.warning("zone_detector: model inference error: %s", e)

        # Fallback: geometry estimation
        return self._estimate_from_geometry(frame, sport, player_bbox)

    def _estimate_from_geometry(
        self,
        frame: np.ndarray,
        sport: str,
        player_bbox: list[float] | None = None,
    ) -> ZoneResult:
        """Estimate zone from player position using court/field geometry rules."""
        h, w = frame.shape[:2]

        # Get player center position (use bbox center or frame center)
        if player_bbox and len(player_bbox) >= 4:
            cx = (player_bbox[0] + player_bbox[2]) / 2 / w
            cy = (player_bbox[1] + player_bbox[3]) / 2 / h
        else:
            cx, cy = 0.5, 0.5

        sport_lower = sport.lower()

        if sport_lower == "basketball":
            return self._basketball_zone(cx, cy)
        elif sport_lower in ("football", "american_football"):
            return self._football_zone(cx, cy)
        elif sport_lower == "lacrosse":
            return self._lacrosse_zone(cx, cy)
        else:
            return ZoneResult(zone="unknown", zone_confidence=0.3, method="geometry_estimate")

    def _basketball_zone(self, cx: float, cy: float) -> ZoneResult:
        """Estimate basketball court zone from normalized position."""
        if cx < 0.25:
            zone = "backcourt"
        elif cx > 0.75:
            # Near basket side
            if 0.3 < cy < 0.7 and cx > 0.85:
                zone = "paint"
            elif cx > 0.80:
                zone = "midrange"
            else:
                zone = "three_point"
        elif cx > 0.5:
            zone = "three_point"
        else:
            zone = "transition"
        return ZoneResult(zone=zone, zone_confidence=0.5, method="geometry_estimate")

    def _football_zone(self, cx: float, cy: float) -> ZoneResult:
        """Estimate football field zone from normalized position."""
        if cx < 0.1:
            zone = "own_endzone"
        elif cx < 0.35:
            zone = "own_territory"
        elif cx < 0.65:
            zone = "midfield"
        elif cx < 0.85:
            zone = "opponent_territory"
        elif cx < 0.95:
            zone = "redzone"
        else:
            zone = "opponent_endzone"
        return ZoneResult(zone=zone, zone_confidence=0.4, method="geometry_estimate")

    def _lacrosse_zone(self, cx: float, cy: float) -> ZoneResult:
        """Estimate lacrosse field zone from normalized position."""
        if cx < 0.3:
            zone = "defensive"
        elif cx < 0.6:
            zone = "midfield"
        elif cx < 0.85:
            zone = "attack"
        else:
            zone = "crease"
        return ZoneResult(zone=zone, zone_confidence=0.4, method="geometry_estimate")
