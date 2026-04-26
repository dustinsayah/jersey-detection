# Formation detector — identify player positions from pre-snap frames
# Football-specific: finds QB, WR, RB, TE by formation geometry.
# CPU-only, ~10ms per pre-snap frame. Uses team color assignments.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Position detection parameters
PRESNAP_MOTION_THRESHOLD = 20.0  # Low motion = pre-snap
PRESNAP_MIN_PLAYERS = 7          # Need enough players visible
QB_BEHIND_LINE_YARDS = 0.04      # QB is ~4% of frame width behind O-line center
WR_WIDE_THRESHOLD = 0.15         # WR is >15% of frame width from O-line center
RB_BEHIND_QB_YARDS = 0.03        # RB is behind QB


@dataclass
class FormationPosition:
    """A detected player position in a formation."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    position: str  # "QB", "WR", "RB", "TE", "OL", "unknown"
    confidence: float = 0.0
    team: str = "unknown"  # "my_team" or "opponent"


@dataclass
class FormationResult:
    """Formation detection result for a single frame."""
    timestamp: float
    is_presnap: bool = False
    positions: list[FormationPosition] = field(default_factory=list)
    qb_bbox: Optional[tuple[float, float, float, float]] = None
    oline_center_x: float = 0.5  # Normalized x-position of O-line center
    direction: str = "unknown"  # "left_to_right" or "right_to_left"
    confidence: float = 0.0


@dataclass
class FormationAnalysis:
    """Full formation analysis across all frames."""
    results: list[FormationResult] = field(default_factory=list)
    presnap_timestamps: list[float] = field(default_factory=list)
    qb_timestamps: list[float] = field(default_factory=list)  # Timestamps where QB detected


class FormationDetector:
    """Detect football formations and identify player positions."""

    def __init__(self, target_position: str = ""):
        self.target_position = target_position.upper().strip()
        # Normalize position names
        _pos_map = {
            "QUARTERBACK": "QB", "WIDE RECEIVER": "WR", "RUNNING BACK": "RB",
            "TIGHT END": "TE", "OFFENSIVE LINE": "OL", "CENTER": "OL",
            "GUARD": "OL", "TACKLE": "OL", "FULLBACK": "RB",
            "SLOT RECEIVER": "WR", "FLANKER": "WR",
        }
        self.target_position = _pos_map.get(self.target_position, self.target_position)

    def analyze_frames(
        self,
        frames_with_ts: list[tuple[float, "np.ndarray"]],
        player_boxes_per_frame: dict[int, list[list[float]]],
        motion_scores: dict[float, float],
        team_assignments_per_frame: dict[int, list[dict]],
    ) -> FormationAnalysis:
        """Analyze formations across all frames.

        Args:
            frames_with_ts: list of (timestamp, frame) tuples
            player_boxes_per_frame: frame_index → list of [x1,y1,x2,y2]
            motion_scores: timestamp → motion score (0-100)
            team_assignments_per_frame: frame_index → list of {"bbox": [...], "team": "my_team"|"opponent"}
        """
        analysis = FormationAnalysis()

        for frame_idx, (ts, frame) in enumerate(frames_with_ts):
            boxes = player_boxes_per_frame.get(frame_idx, [])
            motion = motion_scores.get(ts, 50.0)
            team_info = team_assignments_per_frame.get(frame_idx, [])

            # Check if this is a pre-snap frame
            if not self._is_presnap(motion, len(boxes)):
                continue

            h, w = frame.shape[:2] if hasattr(frame, 'shape') else (720, 1280)
            result = self._detect_formation(ts, boxes, team_info, w, h)

            if result.is_presnap:
                analysis.results.append(result)
                analysis.presnap_timestamps.append(ts)
                if result.qb_bbox is not None:
                    analysis.qb_timestamps.append(ts)

        logger.info(
            "formation_detector: %d pre-snap frames, %d with QB detected",
            len(analysis.presnap_timestamps),
            len(analysis.qb_timestamps),
        )
        return analysis

    def _is_presnap(self, motion: float, player_count: int) -> bool:
        """Check if frame looks like a pre-snap formation."""
        return motion < PRESNAP_MOTION_THRESHOLD and player_count >= PRESNAP_MIN_PLAYERS

    def _detect_formation(
        self,
        timestamp: float,
        all_boxes: list[list[float]],
        team_assignments: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> FormationResult:
        """Detect formation positions from a single frame."""
        result = FormationResult(timestamp=timestamp)

        if len(all_boxes) < PRESNAP_MIN_PLAYERS:
            return result

        # Build team-aware player list
        players: list[dict] = []
        for i, box in enumerate(all_boxes):
            cx = (box[0] + box[2]) / 2 / frame_w  # Normalized center x
            cy = (box[1] + box[3]) / 2 / frame_h  # Normalized center y
            team = "unknown"
            if i < len(team_assignments):
                team = team_assignments[i].get("team", "unknown")
            players.append({
                "bbox": box,
                "cx": cx,
                "cy": cy,
                "team": team,
                "idx": i,
            })

        # Separate by team
        my_team = [p for p in players if p["team"] == "my_team"]
        opponent = [p for p in players if p["team"] == "opponent"]

        # If we can't separate teams, try geometric approach (find the cluster)
        if len(my_team) < 5 and len(opponent) < 5:
            # Fall back to finding the tighter cluster (O-line)
            offensive_players = self._find_offensive_cluster(players)
        else:
            # Use the team with more central players (offense has more players near center)
            my_central = sum(1 for p in my_team if 0.25 < p["cx"] < 0.75)
            opp_central = sum(1 for p in opponent if 0.25 < p["cx"] < 0.75)
            offensive_players = my_team if my_central >= opp_central else opponent

        if len(offensive_players) < 5:
            return result

        result.is_presnap = True

        # Find O-line: the tightest cluster of 3-5 players at similar y-coordinates
        oline, direction = self._find_offensive_line(offensive_players, frame_w)
        if not oline:
            return result

        result.direction = direction
        oline_cx = np.mean([p["cx"] for p in oline])
        oline_cy = np.mean([p["cy"] for p in oline])
        result.oline_center_x = float(oline_cx)

        # Non-OL offensive players
        non_oline = [p for p in offensive_players if p not in oline]

        # Classify each non-OL player
        positions: list[FormationPosition] = []

        for p in oline:
            positions.append(FormationPosition(
                bbox=(p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3]),
                position="OL",
                confidence=0.7,
                team="my_team",
            ))

        for p in non_oline:
            pos, conf = self._classify_position(
                p, oline_cx, oline_cy, direction, frame_w
            )
            fp = FormationPosition(
                bbox=(p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3]),
                position=pos,
                confidence=conf,
                team="my_team",
            )
            positions.append(fp)

            if pos == "QB":
                result.qb_bbox = fp.bbox
                result.confidence = max(result.confidence, conf)

        result.positions = positions
        return result

    def _find_offensive_cluster(self, players: list[dict]) -> list[dict]:
        """Find offensive players by looking for the tight cluster (O-line pattern)."""
        if len(players) < 7:
            return players

        # Sort by x position and find the densest cluster
        sorted_by_x = sorted(players, key=lambda p: p["cx"])

        best_cluster: list[dict] = []
        best_density = 0.0

        for i in range(len(sorted_by_x) - 4):
            window = sorted_by_x[i:i + 7]
            x_range = window[-1]["cx"] - window[0]["cx"]
            density = len(window) / (x_range + 0.01)
            if density > best_density:
                best_density = density
                # Include players near this cluster
                center_x = np.mean([p["cx"] for p in window])
                best_cluster = [
                    p for p in players
                    if abs(p["cx"] - center_x) < 0.25
                ]

        return best_cluster if len(best_cluster) >= 5 else players[:11]

    def _find_offensive_line(
        self, offensive: list[dict], frame_w: int
    ) -> tuple[list[dict], str]:
        """Find the offensive line cluster and determine play direction."""
        if len(offensive) < 5:
            return [], "unknown"

        # O-line players have similar y-coordinates and are clustered in x
        # Sort by x position
        sorted_x = sorted(offensive, key=lambda p: p["cx"])

        # Find the group of 3-5 players with smallest x-spread and similar y
        best_group: list[dict] = []
        best_spread = float("inf")

        for size in [5, 4, 3]:
            for i in range(len(sorted_x) - size + 1):
                group = sorted_x[i:i + size]
                x_spread = group[-1]["cx"] - group[0]["cx"]
                y_spread = max(p["cy"] for p in group) - min(p["cy"] for p in group)

                # O-line should be tight in x (< 0.15) and similar y (< 0.08)
                if x_spread < 0.20 and y_spread < 0.10 and x_spread < best_spread:
                    best_spread = x_spread
                    best_group = group

        if not best_group:
            return [], "unknown"

        # Determine direction: where are backfield players relative to O-line?
        oline_cx = np.mean([p["cx"] for p in best_group])
        backfield = [p for p in offensive if p not in best_group]

        if not backfield:
            return best_group, "unknown"

        backfield_cx = np.mean([p["cx"] for p in backfield])
        # If backfield is to the left of O-line, offense goes right
        direction = "left_to_right" if backfield_cx < oline_cx else "right_to_left"

        return best_group, direction

    def _classify_position(
        self,
        player: dict,
        oline_cx: float,
        oline_cy: float,
        direction: str,
        frame_w: int,
    ) -> tuple[str, float]:
        """Classify a non-OL offensive player's position."""
        px = player["cx"]
        py = player["cy"]

        # Distance from O-line center
        x_offset = px - oline_cx
        y_offset = py - oline_cy
        lateral_dist = abs(x_offset)

        # "Behind" depends on direction
        if direction == "left_to_right":
            is_behind = x_offset < -QB_BEHIND_LINE_YARDS
        elif direction == "right_to_left":
            is_behind = x_offset > QB_BEHIND_LINE_YARDS
        else:
            # Default: use y position (below center = behind in broadcast view)
            is_behind = y_offset > 0.02

        # QB: close behind center, not too far laterally
        if is_behind and lateral_dist < 0.06 and abs(y_offset) < 0.08:
            return "QB", 0.75

        # WR: far from center laterally, near line of scrimmage
        if lateral_dist > WR_WIDE_THRESHOLD:
            return "WR", 0.7

        # TE: moderate lateral distance, near line of scrimmage
        if 0.08 < lateral_dist <= WR_WIDE_THRESHOLD and abs(y_offset) < 0.06:
            return "TE", 0.55

        # RB: behind QB position, moderate lateral offset
        if is_behind and lateral_dist < 0.10:
            return "RB", 0.6

        return "unknown", 0.3

    def get_target_position_score(
        self, analysis: FormationAnalysis, start_time: float, end_time: float
    ) -> float:
        """Score how likely a clip shows the target position player.

        Returns 0.0-1.0 score.
        """
        if not self.target_position or not analysis.results:
            return 0.0

        best_score = 0.0
        search_start = start_time - 3.0  # Pre-snap often 3s before play
        search_end = start_time + 2.0    # Pre-snap is early in the clip

        for result in analysis.results:
            if not (search_start <= result.timestamp <= search_end):
                continue

            for pos in result.positions:
                if pos.position == self.target_position:
                    best_score = max(best_score, pos.confidence)

        return best_score

    def get_qb_in_clip(
        self, analysis: FormationAnalysis, start_time: float, end_time: float
    ) -> bool:
        """Check if a QB was detected in formation near this clip's timeframe."""
        search_start = start_time - 4.0
        search_end = start_time + 2.0
        for ts in analysis.qb_timestamps:
            if search_start <= ts <= search_end:
                return True
        return False
