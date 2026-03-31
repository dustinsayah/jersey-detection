"""Regression tests for v4 fixes — dict/object mismatch, logger NameError,
position normalization, camelCase alias contract, pipeline shape.

These tests cover bugs that caused 0-detection regressions in production.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.detect import DetectRequest, normalize_position
from app.schemas.analyze import AnalyzeRequest


# ── Bug #1: dict/object mismatch in _run_jersey_detection ──────────────────


class TestDictDetectionHandling:
    """detect_jersey_in_frames returns list[dict] via DetectedFrame.to_dict().
    _run_jersey_detection must handle dicts, not attribute-access objects.
    """

    SAMPLE_DICTS: list[dict[str, Any]] = [
        {
            "timestamp": 8.4,
            "confidence": 0.92,
            "bbox": {
                "x1": 340, "y1": 180, "x2": 490, "y2": 520,
                "x1_pct": 21.25, "y1_pct": 15.0,
                "x2_pct": 30.63, "y2_pct": 43.33,
            },
        },
        {
            "timestamp": 9.1,
            "confidence": 0.88,
            "bbox": {
                "x1": 338, "y1": 178, "x2": 488, "y2": 518,
                "x1_pct": 21.13, "y1_pct": 14.83,
                "x2_pct": 30.5, "y2_pct": 43.17,
            },
        },
    ]

    def test_run_jersey_detection_handles_dicts(self) -> None:
        """_run_jersey_detection must not raise AttributeError on dict detections."""
        from app.services.analyze_pipeline import _run_jersey_detection

        with patch(
            "app.services.detection_service.DetectionService"
        ) as MockService:
            instance = MockService.return_value
            instance.detect.return_value = self.SAMPLE_DICTS

            results = _run_jersey_detection(
                video_url=None,
                video_path="/fake/path.mp4",
                jersey_number=2,
                jersey_color="white",
                sport="basketball",
                position=None,
            )

        assert len(results) == 2
        assert results[0]["timestamp"] == 8.4
        assert results[0]["confidence"] == 0.92
        assert "x1" in results[0]
        assert results[0]["x1"] == 340

    def test_run_jersey_detection_handles_dict_without_bbox(self) -> None:
        """Detections without bbox should still produce timestamp+confidence."""
        from app.services.analyze_pipeline import _run_jersey_detection

        bare_dicts = [{"timestamp": 5.0, "confidence": 0.7}]

        with patch(
            "app.services.detection_service.DetectionService"
        ) as MockService:
            instance = MockService.return_value
            instance.detect.return_value = bare_dicts

            results = _run_jersey_detection(
                video_url=None,
                video_path="/fake/path.mp4",
                jersey_number=2,
                jersey_color="white",
                sport="basketball",
                position=None,
            )

        assert len(results) == 1
        assert results[0]["timestamp"] == 5.0
        assert results[0]["confidence"] == 0.7
        assert "x1" not in results[0]

    def test_run_jersey_detection_handles_empty_list(self) -> None:
        """Empty detection list should return empty, not crash."""
        from app.services.analyze_pipeline import _run_jersey_detection

        with patch(
            "app.services.detection_service.DetectionService"
        ) as MockService:
            instance = MockService.return_value
            instance.detect.return_value = []

            results = _run_jersey_detection(
                video_url=None,
                video_path="/fake/path.mp4",
                jersey_number=2,
                jersey_color="white",
                sport="basketball",
                position=None,
            )

        assert results == []


# ── Bug #2: logger NameError in _download_direct_video ──────────────────────


class TestLoggerReference:
    """detection_pipeline.py must use LOGGER (uppercase), not logger (lowercase)."""

    def test_no_lowercase_logger_in_detection_pipeline(self) -> None:
        """Ensure no bare 'logger.' calls exist in detection_pipeline.py.
        The module defines LOGGER = logging.getLogger(__name__).
        """
        import inspect
        from app.services import detection_pipeline

        source = inspect.getsource(detection_pipeline)
        # Find lines that use 'logger.' (lowercase) — these would be NameError at runtime
        lines = source.split("\n")
        bad_lines = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments and strings
            if stripped.startswith("#"):
                continue
            if "logger." in stripped and "LOGGER." not in stripped and "# " not in stripped.split("logger.")[0]:
                bad_lines.append((i, stripped))

        assert bad_lines == [], (
            f"Found lowercase 'logger.' references in detection_pipeline.py "
            f"(should be LOGGER): {bad_lines}"
        )


# ── Bug #3: position normalization ──────────────────────────────────────────


class TestPositionNormalization:
    """Position strings should be normalized: misspellings fixed, slash-separated handled."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("guard", "guard"),
            ("Guard", "guard"),
            ("GUARD", "guard"),
            ("gaurd", "guard"),           # common misspelling
            ("guard/forward", "guard"),   # slash-separated → first
            ("  guard  ", "guard"),        # whitespace
            ("quarterback", "quarterback"),
            ("qb", "quarterback"),         # abbreviation
            ("QB", "quarterback"),
            ("quarter back", "quarterback"),
            ("wide reciever", "wide receiver"),  # misspelling
            ("wr", "wide receiver"),
            ("goalie", "goalie"),
            ("goaltender", "goalie"),
            ("midfielder", "midfielder"),
            ("middie", "midfielder"),
            (None, None),
            ("", None),
            ("  ", None),
        ],
    )
    def test_normalize_position(self, raw: str | None, expected: str | None) -> None:
        assert normalize_position(raw) == expected

    def test_detect_request_normalizes_position(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
            "position": "gaurd/forward",
        })
        assert req.position == "guard"

    def test_analyze_request_normalizes_position(self) -> None:
        req = AnalyzeRequest.model_validate({
            "videoUrl": "https://example.com/v.mp4",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
            "position": "Quarter Back",
        })
        assert req.position == "quarterback"

    def test_analyze_request_empty_position(self) -> None:
        req = AnalyzeRequest.model_validate({
            "videoUrl": "https://example.com/v.mp4",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
            "position": "",
        })
        assert req.position is None


# ── camelCase alias contract (AnalyzeRequest) ───────────────────────────────


class TestAnalyzeRequestAliases:
    """AnalyzeRequest must accept camelCase from the website and snake_case internally."""

    def test_camel_case_accepted(self) -> None:
        req = AnalyzeRequest.model_validate({
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
            "timeRangeStart": 10,
            "timeRangeEnd": 60,
        })
        assert req.video_url == "https://youtube.com/watch?v=abc"
        assert req.jersey_number == 2
        assert req.jersey_color == "white"
        assert req.time_range_start == 10
        assert req.time_range_end == 60

    def test_unsupported_sport_defaults(self) -> None:
        req = AnalyzeRequest.model_validate({
            "videoUrl": "https://example.com/v.mp4",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "cricket",
        })
        assert req.sport == "basketball"

    def test_missing_video_source_rejected(self) -> None:
        with pytest.raises(ValidationError, match="videoUrl.*videoPath"):
            AnalyzeRequest.model_validate({
                "jerseyNumber": 2,
                "jerseyColor": "white",
                "sport": "basketball",
            })


# ── DetectedFrame.to_dict() shape contract ──────────────────────────────────


class TestDetectedFrameShape:
    """Verify DetectedFrame.to_dict() returns the expected shape."""

    def test_to_dict_has_required_keys(self) -> None:
        from app.services.detection_runtime import DetectedFrame

        df = DetectedFrame(
            timestamp=10.5,
            confidence=0.95,
            x1=100, y1=200, x2=300, y2=400,
            frame_w=1920, frame_h=1080,
        )
        d = df.to_dict()

        assert "timestamp" in d
        assert "confidence" in d
        assert "bbox" in d
        assert isinstance(d["bbox"], dict)
        assert d["bbox"]["x1"] == 100
        assert d["timestamp"] == 10.5
        assert d["confidence"] == 0.95

    def test_to_dict_bbox_pct_values(self) -> None:
        from app.services.detection_runtime import DetectedFrame

        df = DetectedFrame(
            timestamp=1.0, confidence=0.5,
            x1=192, y1=108, x2=384, y2=216,
            frame_w=1920, frame_h=1080,
        )
        d = df.to_dict()
        assert d["bbox"]["x1_pct"] == 10.0
        assert d["bbox"]["y1_pct"] == 10.0
        assert d["bbox"]["x2_pct"] == 20.0
        assert d["bbox"]["y2_pct"] == 20.0


# ── Temporal consensus with dict detections ─────────────────────────────────


class TestTemporalConsensusDict:
    """TemporalConsensus.filter_detections works with dict detections."""

    def test_filter_passes_matching_detections(self) -> None:
        from app.services.temporal_consensus import TemporalConsensus

        tc = TemporalConsensus(min_confirmations=2, time_window=2.0)
        dets = [
            {"timestamp": 8.0, "confidence": 0.9, "number_detected": 2, "layer": "ali"},
            {"timestamp": 8.5, "confidence": 0.85, "number_detected": 2, "layer": "v3_ocr"},
            {"timestamp": 9.0, "confidence": 0.8, "number_detected": 2, "layer": "ali"},
        ]
        confirmed = tc.filter_detections(dets, jersey_number=2)
        assert len(confirmed) >= 1

    def test_filter_adapts_for_single_isolated_detection(self) -> None:
        """With adaptive min_confirmations, a single detection is rescued
        (min drops from 3 to 1 when total detections is very low)."""
        from app.services.temporal_consensus import TemporalConsensus

        tc = TemporalConsensus(min_confirmations=3, time_window=2.0)
        dets = [
            {"timestamp": 8.0, "confidence": 0.9, "number_detected": 2, "layer": "ali"},
        ]
        confirmed = tc.filter_detections(dets, jersey_number=2)
        # Adaptive behavior: min_confirmations drops to 1 for single detections
        assert len(confirmed) == 1
