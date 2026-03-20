from __future__ import annotations

from dataclasses import replace

from app.services.detection_runtime import DetectedFrame


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
