# Zoom detector — identifies broadcast camera zoom level
# Uses bounding box height relative to frame height.
# Zero additional model cost — pure arithmetic on existing detections.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Zoom thresholds (player bbox height / frame height)
CLOSE_UP_THRESHOLD = 0.25   # Player fills >25% of frame height
MEDIUM_THRESHOLD = 0.15     # Player fills 15-25%
WIDE_THRESHOLD = 0.08       # Player fills <8% (full formation view)

# Player count thresholds
FORMATION_MIN_PLAYERS = 8   # Full formation has 8+ visible
REPLAY_MAX_PLAYERS = 4      # Replay/close-up has <5 visible


@dataclass
class ZoomEvent:
    """A detected zoom/importance event."""
    timestamp: float
    zoom_level: str  # "close_up", "medium", "wide"
    max_player_height_ratio: float
    player_count: int
    importance_score: float  # 0-1, how important this frame likely is
    is_transition: bool = False  # Wide→close_up transition detected


@dataclass
class ZoomAnalysis:
    """Full zoom analysis across all frames."""
    events: list[ZoomEvent] = field(default_factory=list)
    close_up_timestamps: list[float] = field(default_factory=list)
    transition_timestamps: list[float] = field(default_factory=list)


class ZoomDetector:
    """Detect broadcast camera zoom level from player bounding boxes."""

    def __init__(self):
        self._prev_zoom: str = "wide"
        self._prev_timestamp: float = 0.0

    def analyze_frames(
        self,
        timestamps: list[float],
        player_boxes_per_frame: dict[int, list[list[float]]],
        frame_heights: dict[int, int],
    ) -> ZoomAnalysis:
        """Analyze zoom level across all frames.

        Args:
            timestamps: list of timestamps corresponding to frame indices
            player_boxes_per_frame: frame_index → list of [x1,y1,x2,y2]
            frame_heights: frame_index → frame height in pixels
        """
        analysis = ZoomAnalysis()

        for frame_idx, boxes in sorted(player_boxes_per_frame.items()):
            if frame_idx >= len(timestamps):
                continue

            ts = timestamps[frame_idx]
            frame_h = frame_heights.get(frame_idx, 720)

            event = self._analyze_single_frame(ts, boxes, frame_h)
            if event:
                analysis.events.append(event)
                if event.zoom_level == "close_up":
                    analysis.close_up_timestamps.append(ts)
                if event.is_transition:
                    analysis.transition_timestamps.append(ts)

        logger.info(
            "zoom_detector: %d events, %d close-ups, %d transitions",
            len(analysis.events),
            len(analysis.close_up_timestamps),
            len(analysis.transition_timestamps),
        )
        return analysis

    def _analyze_single_frame(
        self,
        timestamp: float,
        player_boxes: list[list[float]],
        frame_height: int,
    ) -> ZoomEvent | None:
        """Analyze zoom level for a single frame."""
        if not player_boxes:
            return None

        # Compute max player height ratio
        max_ratio = 0.0
        for box in player_boxes:
            box_h = box[3] - box[1]
            ratio = box_h / frame_height if frame_height > 0 else 0
            max_ratio = max(max_ratio, ratio)

        player_count = len(player_boxes)

        # Classify zoom level
        if max_ratio > CLOSE_UP_THRESHOLD:
            zoom_level = "close_up"
        elif max_ratio > MEDIUM_THRESHOLD:
            zoom_level = "medium"
        else:
            zoom_level = "wide"

        # Compute importance score
        importance = self._compute_importance(zoom_level, max_ratio, player_count)

        # Detect transition (wide → close_up within 3s = broadcast highlighting)
        is_transition = False
        if zoom_level == "close_up" and self._prev_zoom == "wide":
            time_gap = timestamp - self._prev_timestamp
            if 0 < time_gap < 3.0:
                is_transition = True

        self._prev_zoom = zoom_level
        self._prev_timestamp = timestamp

        return ZoomEvent(
            timestamp=timestamp,
            zoom_level=zoom_level,
            max_player_height_ratio=round(max_ratio, 3),
            player_count=player_count,
            importance_score=round(importance, 3),
            is_transition=is_transition,
        )

    def _compute_importance(
        self, zoom_level: str, max_ratio: float, player_count: int
    ) -> float:
        """Compute frame importance from zoom signals."""
        score = 0.0

        # Zoom level base score
        if zoom_level == "close_up":
            score += 0.6
        elif zoom_level == "medium":
            score += 0.3

        # Few players visible = zoomed in on action
        if player_count <= REPLAY_MAX_PLAYERS:
            score += 0.2

        # Very large player = extreme close-up (celebration, post-play)
        if max_ratio > 0.4:
            score += 0.15

        # Full formation = setup, less important
        if player_count >= FORMATION_MIN_PLAYERS and zoom_level == "wide":
            score = max(0.05, score - 0.2)

        return min(1.0, score)

    def get_importance_at_time(
        self, analysis: ZoomAnalysis, timestamp: float, window: float = 2.0
    ) -> float:
        """Get the max importance score near a given timestamp."""
        best = 0.0
        for event in analysis.events:
            if abs(event.timestamp - timestamp) <= window:
                best = max(best, event.importance_score)
        return best

    def has_close_up_near(
        self, analysis: ZoomAnalysis, start_time: float, end_time: float
    ) -> bool:
        """Check if there's a close-up within a time range."""
        for ts in analysis.close_up_timestamps:
            if start_time - 1.0 <= ts <= end_time + 1.0:
                return True
        return False

    def has_transition_near(
        self, analysis: ZoomAnalysis, start_time: float, end_time: float
    ) -> bool:
        """Check if there's a wide→close transition within a time range."""
        for ts in analysis.transition_timestamps:
            if start_time - 2.0 <= ts <= end_time + 2.0:
                return True
        return False
