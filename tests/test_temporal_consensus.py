from __future__ import annotations

from app.services.detection_runtime import DetectedFrame
from app.services.temporal_consensus import apply_temporal_consensus


def _detection(timestamp: float, confidence: float, x1: float = 10.0) -> DetectedFrame:
    return DetectedFrame(
        timestamp=timestamp,
        confidence=confidence,
        x1=x1,
        y1=10.0,
        x2=x1 + 30.0,
        y2=70.0,
        frame_w=1920.0,
        frame_h=1080.0,
    )


def test_temporal_consensus_boosts_clustered_detections() -> None:
    detections = [
        _detection(1.0, 0.60),
        _detection(1.4, 0.62),
        _detection(1.8, 0.64),
    ]
    accepted, rejected = apply_temporal_consensus(
        detections,
        enabled=True,
        max_gap_seconds=0.6,
        min_iou=0.1,
        min_votes=2,
        keep_strong_single_confidence=0.85,
    )
    assert rejected == 0
    assert len(accepted) == 3
    assert accepted[0].confidence > detections[0].confidence


def test_temporal_consensus_rejects_weak_singletons() -> None:
    accepted, rejected = apply_temporal_consensus(
        [_detection(1.0, 0.61)],
        enabled=True,
        max_gap_seconds=0.6,
        min_iou=0.1,
        min_votes=2,
        keep_strong_single_confidence=0.85,
    )
    assert accepted == []
    assert rejected == 1


def test_temporal_consensus_keeps_strong_singletons() -> None:
    strong = _detection(1.0, 0.91)
    accepted, rejected = apply_temporal_consensus(
        [strong],
        enabled=True,
        max_gap_seconds=0.6,
        min_iou=0.1,
        min_votes=2,
        keep_strong_single_confidence=0.85,
    )
    assert rejected == 0
    assert accepted == [strong]
