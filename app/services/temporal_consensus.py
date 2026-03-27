from __future__ import annotations

import logging
from dataclasses import replace

from app.services.detection_runtime import DetectedFrame

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bbox_iou(left: DetectedFrame, right: DetectedFrame) -> float:
    ix1 = max(left.x1, right.x1)
    iy1 = max(left.y1, right.y1)
    ix2 = min(left.x2, right.x2)
    iy2 = min(left.y2, right.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    left_area = max(1.0, (left.x2 - left.x1) * (left.y2 - left.y1))
    right_area = max(1.0, (right.x2 - right.x1) * (right.y2 - right.y1))
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def apply_temporal_consensus(
    detections: list[DetectedFrame],
    *,
    enabled: bool,
    max_gap_seconds: float,
    min_iou: float,
    min_votes: int,
    keep_strong_single_confidence: float,
) -> tuple[list[DetectedFrame], int]:
    if not enabled or not detections:
        return detections, 0

    clusters: list[list[DetectedFrame]] = []
    current: list[DetectedFrame] = [detections[0]]

    for detection in detections[1:]:
        previous = current[-1]
        gap = detection.timestamp - previous.timestamp
        if gap <= max_gap_seconds and _bbox_iou(previous, detection) >= min_iou:
            current.append(detection)
            continue
        clusters.append(current)
        current = [detection]
    clusters.append(current)

    accepted: list[DetectedFrame] = []
    rejected = 0
    for cluster in clusters:
        if len(cluster) < min_votes:
            strongest = max(cluster, key=lambda item: item.confidence)
            if strongest.confidence >= keep_strong_single_confidence:
                accepted.extend(cluster)
            else:
                rejected += len(cluster)
            continue

        bonus = min(0.12, 0.04 * (len(cluster) - 1))
        accepted.extend(
            replace(item, confidence=_clamp01(item.confidence + bonus))
            for item in cluster
        )

    return accepted, rejected


# ── Dict-based temporal consensus for analyze_pipeline.py ──────────────
# The above function uses Ali's DetectedFrame dataclass.
# The class below works with raw dicts from the analyze pipeline
# (merged Ali + Roboflow detections) before they become DetectionPoints.


class TemporalConsensus:
    """Filter detections by requiring temporal agreement across frames.

    A detection is confirmed only when the same jersey number appears at
    least ``min_confirmations`` times within ``time_window`` seconds.
    This replicates Ali's key accuracy advantage: requiring temporal
    consistency eliminates false positives dramatically.
    """

    def __init__(
        self,
        min_confirmations: int = 3,
        time_window: float = 2.0,
        confidence_threshold: float = 0.5,
    ):
        self.min_confirmations = min_confirmations
        self.time_window = time_window
        self.confidence_threshold = confidence_threshold

    def filter_detections(
        self, detections: list[dict], jersey_number: int, *, adaptive: bool = True
    ) -> list[dict]:
        """Keep only detections confirmed by temporal consensus.

        Input:  [{timestamp, confidence, number_detected, layer, bbox, ...}]
        Output: filtered list with consensus_score, consensus_confirmations,
                and consensus_layers added to each surviving detection.

        When ``adaptive=True`` (default), min_confirmations is lowered
        automatically when overall detection count is sparse:
          < 9 detections  → min_confirmations = 1
          < 18 detections → min_confirmations = 2
          18+             → use configured default (3)
        """
        if not detections:
            return []

        # Adaptive threshold for poor-quality video with few detections
        effective_min = self.min_confirmations
        if adaptive:
            total = len(detections)
            if total < 9:
                effective_min = 1
            elif total < 18:
                effective_min = 2
            if effective_min != self.min_confirmations:
                logger.info(
                    "temporal_consensus: adaptive min_confirmations %d → %d (total=%d)",
                    self.min_confirmations, effective_min, total,
                )

        sorted_dets = sorted(detections, key=lambda x: x.get("timestamp", 0))

        confirmed: list[dict] = []
        for det in sorted_dets:
            ts = det.get("timestamp", 0)

            # Find all detections within time_window that match target number
            window_dets = [
                d
                for d in sorted_dets
                if abs(d.get("timestamp", 0) - ts) <= self.time_window
                and d.get("number_detected") == jersey_number
            ]

            confirmations = len(window_dets)

            if confirmations >= effective_min:
                avg_conf = (
                    sum(d.get("confidence", 0) for d in window_dets)
                    / confirmations
                )
                # Boost confidence based on number of confirmations
                consensus_boost = min(0.2, confirmations * 0.03)
                consensus_score = min(1.0, avg_conf + consensus_boost)

                det["consensus_confirmations"] = confirmations
                det["consensus_score"] = round(consensus_score, 4)
                det["consensus_layers"] = list(
                    {d.get("layer", "unknown") for d in window_dets}
                )
                confirmed.append(det)

        # Deduplicate — keep highest consensus_score within 0.5s windows
        deduped: list[dict] = []
        for det in confirmed:
            if not deduped:
                deduped.append(det)
                continue
            prev_ts = deduped[-1].get("timestamp", 0)
            cur_ts = det.get("timestamp", 0)
            if cur_ts - prev_ts > 0.5:
                deduped.append(det)
            elif det.get("consensus_score", 0) > deduped[-1].get(
                "consensus_score", 0
            ):
                deduped[-1] = det

        return deduped

    def cross_layer_boost(self, detections: list[dict]) -> list[dict]:
        """Boost confidence when multiple different layers agree.

        Ali + Roboflow both detecting the same number at the same time
        is a very strong signal. Each additional unique layer adds a
        +0.1 confidence boost.
        """
        for det in detections:
            layers = det.get("consensus_layers", [])
            unique_layers = len(set(layers))
            if unique_layers >= 2:
                det["cross_layer_confirmed"] = True
                det["consensus_score"] = min(
                    1.0,
                    det.get("consensus_score", 0.5) + (unique_layers * 0.1),
                )
            else:
                det["cross_layer_confirmed"] = False
        return detections


# Singleton instance for analyze pipeline
temporal_consensus = TemporalConsensus()
