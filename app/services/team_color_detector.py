# Team color detector — classifies players as "my team" or "opponent"
# Uses HSV color analysis of jersey crops + KMeans clustering.
# CPU-only, ~5ms per frame. No new ML models.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Map user-friendly color names to HSV ranges (H: 0-179, S: 0-255, V: 0-255)
# Each color has multiple ranges to handle variation in lighting/broadcast
COLOR_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (179, 255, 255))],
    "blue": [((100, 80, 60), (130, 255, 255))],
    "navy": [((100, 50, 30), (130, 255, 180))],
    "dark blue": [((100, 50, 30), (130, 255, 180))],
    "green": [((35, 60, 60), (85, 255, 255))],
    "yellow": [((20, 80, 100), (35, 255, 255))],
    "orange": [((10, 80, 100), (22, 255, 255))],
    "purple": [((130, 50, 50), (160, 255, 255))],
    "maroon": [((0, 50, 30), (10, 255, 150)), ((170, 50, 30), (179, 255, 150))],
    "white": [((0, 0, 180), (179, 40, 255))],
    "black": [((0, 0, 0), (179, 80, 60))],
    "gray": [((0, 0, 60), (179, 40, 180))],
    "grey": [((0, 0, 60), (179, 40, 180))],
    "gold": [((18, 80, 100), (30, 255, 255))],
    "crimson": [((0, 100, 80), (8, 255, 220)), ((172, 100, 80), (179, 255, 220))],
    "teal": [((80, 60, 60), (100, 255, 255))],
}


@dataclass
class TeamColorProfile:
    """Learned team color profile from sampled frames."""
    my_team_hue_center: float = 0.0
    my_team_hue_range: float = 20.0
    my_team_sat_min: float = 40.0
    my_team_val_min: float = 40.0
    opponent_hue_center: float = 0.0
    opponent_hue_range: float = 20.0
    cluster_centers: list[list[float]] = field(default_factory=list)
    my_team_cluster_idx: int = 0
    calibrated: bool = False


@dataclass
class PlayerTeamAssignment:
    """Per-player team assignment for a single frame."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    team: str  # "my_team", "opponent", "unknown"
    confidence: float = 0.0
    dominant_hue: float = 0.0


class TeamColorDetector:
    """Detect team jersey colors and classify players."""

    def __init__(self, jersey_color: str = "white"):
        self.jersey_color = jersey_color.lower().strip()
        self.profile = TeamColorProfile()
        self._hsv_ranges = COLOR_HSV_RANGES.get(self.jersey_color, [])

    def calibrate(
        self,
        frames: list[tuple[float, np.ndarray]],
        player_boxes_per_frame: dict[int, list[list[float]]],
    ) -> TeamColorProfile:
        """Calibrate team colors from sample frames.

        Args:
            frames: List of (timestamp, BGR_frame) tuples
            player_boxes_per_frame: frame_index → list of [x1,y1,x2,y2] bounding boxes
        """
        all_torso_hues: list[float] = []
        all_torso_colors: list[list[float]] = []

        sample_indices = list(player_boxes_per_frame.keys())[:20]

        for frame_idx in sample_indices:
            if frame_idx >= len(frames):
                continue
            _, frame = frames[frame_idx]
            boxes = player_boxes_per_frame[frame_idx]

            for box in boxes:
                hue, sat, val = self._extract_torso_hsv(frame, box)
                if sat < 20:
                    continue  # Skip very desaturated (probably field/sky)
                all_torso_hues.append(hue)
                all_torso_colors.append([hue, sat, val])

        if len(all_torso_colors) < 4:
            logger.warning("team_color: too few player crops (%d) to calibrate", len(all_torso_colors))
            # Fall back to HSV range matching only
            self.profile.calibrated = False
            return self.profile

        # KMeans with 2 clusters (two teams)
        try:
            from sklearn.cluster import MiniBatchKMeans
            colors_arr = np.array(all_torso_colors, dtype=np.float32)
            kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=3)
            labels = kmeans.fit_predict(colors_arr)
            centers = kmeans.cluster_centers_.tolist()
        except ImportError:
            # sklearn not available — use simple median split
            logger.info("team_color: sklearn not available, using median split")
            hues = np.array(all_torso_hues)
            median_hue = float(np.median(hues))
            labels = np.array([0 if h < median_hue else 1 for h in all_torso_hues])
            c0 = np.mean([c for c, l in zip(all_torso_colors, labels) if l == 0], axis=0).tolist()
            c1 = np.mean([c for c, l in zip(all_torso_colors, labels) if l == 1], axis=0).tolist()
            centers = [c0, c1]

        self.profile.cluster_centers = centers

        # Match user's jersey_color to one of the clusters
        my_team_idx = self._match_color_to_cluster(centers)
        self.profile.my_team_cluster_idx = my_team_idx

        my_center = centers[my_team_idx]
        opp_idx = 1 - my_team_idx
        opp_center = centers[opp_idx]

        self.profile.my_team_hue_center = my_center[0]
        self.profile.my_team_hue_range = 25.0
        self.profile.my_team_sat_min = max(20, my_center[1] - 40)
        self.profile.my_team_val_min = max(20, my_center[2] - 40)
        self.profile.opponent_hue_center = opp_center[0]
        self.profile.opponent_hue_range = 25.0
        self.profile.calibrated = True

        logger.info(
            "team_color: calibrated — my_team hue=%.0f (cluster %d), "
            "opponent hue=%.0f, %d samples",
            my_center[0], my_team_idx, opp_center[0], len(all_torso_colors),
        )
        return self.profile

    def classify_players(
        self,
        frame: np.ndarray,
        player_boxes: list[list[float]],
    ) -> list[PlayerTeamAssignment]:
        """Classify each player in a frame as my_team/opponent/unknown."""
        results: list[PlayerTeamAssignment] = []

        for box in player_boxes:
            hue, sat, val = self._extract_torso_hsv(frame, box)
            bbox_tuple = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

            if sat < 20 or val < 25:
                results.append(PlayerTeamAssignment(
                    bbox=bbox_tuple, team="unknown", confidence=0.1, dominant_hue=hue,
                ))
                continue

            if self.profile.calibrated:
                team, conf = self._classify_by_cluster(hue, sat, val)
            else:
                team, conf = self._classify_by_hsv_range(hue, sat, val)

            results.append(PlayerTeamAssignment(
                bbox=bbox_tuple, team=team, confidence=conf, dominant_hue=hue,
            ))
        return results

    def count_my_team_in_center(
        self,
        assignments: list[PlayerTeamAssignment],
        frame_width: int,
    ) -> tuple[int, int]:
        """Count my_team vs opponent players in the center 60% of frame.

        Returns (my_team_count, opponent_count).
        """
        left_bound = frame_width * 0.2
        right_bound = frame_width * 0.8
        my_count = 0
        opp_count = 0
        for a in assignments:
            cx = (a.bbox[0] + a.bbox[2]) / 2
            if left_bound <= cx <= right_bound:
                if a.team == "my_team":
                    my_count += 1
                elif a.team == "opponent":
                    opp_count += 1
        return my_count, opp_count

    def _extract_torso_hsv(
        self, frame: np.ndarray, box: list[float]
    ) -> tuple[float, float, float]:
        """Extract dominant HSV from the torso region of a player bbox."""
        h, w = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0

        # Torso = middle 40% of height, center 60% of width
        bh = y2 - y1
        bw = x2 - x1
        torso_y1 = y1 + int(bh * 0.20)
        torso_y2 = y1 + int(bh * 0.60)
        torso_x1 = x1 + int(bw * 0.20)
        torso_x2 = x2 - int(bw * 0.20)

        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            torso_y1, torso_y2 = y1, y2
            torso_x1, torso_x2 = x1, x2

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            return 0.0, 0.0, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Compute mean HSV (simple and fast)
        mean_h = float(np.mean(hsv[:, :, 0]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))
        return mean_h, mean_s, mean_v

    def _match_color_to_cluster(self, centers: list[list[float]]) -> int:
        """Match user's jersey_color to the closest KMeans cluster center."""
        if not self._hsv_ranges:
            # Unknown color name — pick cluster with higher saturation
            # (field grass is typically green with high saturation, but so are some jerseys)
            return 0

        best_idx = 0
        best_dist = float("inf")

        for idx, center in enumerate(centers):
            ch, cs, cv = center[0], center[1], center[2]
            # Check if cluster center falls within any HSV range for this color
            min_dist = float("inf")
            for (lo, hi) in self._hsv_ranges:
                # Distance to range center
                range_h = (lo[0] + hi[0]) / 2
                range_s = (lo[1] + hi[1]) / 2
                range_v = (lo[2] + hi[2]) / 2
                # Hue is circular (0-179)
                h_diff = min(abs(ch - range_h), 179 - abs(ch - range_h))
                dist = h_diff * 2 + abs(cs - range_s) * 0.5 + abs(cv - range_v) * 0.3
                min_dist = min(min_dist, dist)

            if min_dist < best_dist:
                best_dist = min_dist
                best_idx = idx

        return best_idx

    def _classify_by_cluster(
        self, hue: float, sat: float, val: float
    ) -> tuple[str, float]:
        """Classify using calibrated KMeans clusters."""
        centers = self.profile.cluster_centers
        if len(centers) < 2:
            return "unknown", 0.0

        my_center = centers[self.profile.my_team_cluster_idx]
        opp_center = centers[1 - self.profile.my_team_cluster_idx]

        # Distance to each cluster (hue is circular)
        my_h_diff = min(abs(hue - my_center[0]), 179 - abs(hue - my_center[0]))
        my_dist = my_h_diff * 2 + abs(sat - my_center[1]) * 0.5 + abs(val - my_center[2]) * 0.3

        opp_h_diff = min(abs(hue - opp_center[0]), 179 - abs(hue - opp_center[0]))
        opp_dist = opp_h_diff * 2 + abs(sat - opp_center[1]) * 0.5 + abs(val - opp_center[2]) * 0.3

        if my_dist < opp_dist:
            conf = max(0.3, min(0.95, 1.0 - my_dist / (my_dist + opp_dist + 1e-6)))
            return "my_team", conf
        else:
            conf = max(0.3, min(0.95, 1.0 - opp_dist / (my_dist + opp_dist + 1e-6)))
            return "opponent", conf

    def _classify_by_hsv_range(
        self, hue: float, sat: float, val: float
    ) -> tuple[str, float]:
        """Classify using predefined HSV ranges (no calibration)."""
        if not self._hsv_ranges:
            return "unknown", 0.0

        for (lo, hi) in self._hsv_ranges:
            # Check hue (handle wrap-around for red)
            h_match = lo[0] <= hue <= hi[0]
            s_match = lo[1] <= sat <= hi[1]
            v_match = lo[2] <= val <= hi[2]
            if h_match and s_match and v_match:
                return "my_team", 0.7
        return "opponent", 0.5
