"""Shared fixtures for the test suite."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.detection_service import DetectionService


# ---------------------------------------------------------------------------
# Fake detection service — avoids loading YOLO / ffmpeg / yt-dlp in tests
# ---------------------------------------------------------------------------

MOCK_DETECTIONS: list[dict[str, Any]] = [
    {
        "timestamp": 8.4,
        "confidence": 0.92,
        "bbox": {
            "x1": 340,
            "y1": 180,
            "x2": 490,
            "y2": 520,
            "x1_pct": 21.25,
            "y1_pct": 15.0,
            "x2_pct": 30.63,
            "y2_pct": 43.33,
        },
    },
    {
        "timestamp": 9.1,
        "confidence": 0.88,
        "bbox": {
            "x1": 338,
            "y1": 178,
            "x2": 488,
            "y2": 518,
            "x1_pct": 21.13,
            "y1_pct": 14.83,
            "x2_pct": 30.5,
            "y2_pct": 43.17,
        },
    },
    {
        "timestamp": 38.6,
        "confidence": 0.94,
        "bbox": {
            "x1": 352,
            "y1": 190,
            "x2": 500,
            "y2": 530,
            "x1_pct": 22.0,
            "y1_pct": 15.83,
            "x2_pct": 31.25,
            "y2_pct": 44.17,
        },
    },
    {
        "timestamp": 39.3,
        "confidence": 0.96,
        "bbox": {
            "x1": 354,
            "y1": 192,
            "x2": 502,
            "y2": 532,
            "x1_pct": 22.13,
            "y1_pct": 16.0,
            "x2_pct": 31.38,
            "y2_pct": 44.33,
        },
    },
    {
        "timestamp": 104.0,
        "confidence": 0.97,
        "bbox": {
            "x1": 360,
            "y1": 200,
            "x2": 510,
            "y2": 540,
            "x1_pct": 22.5,
            "y1_pct": 16.67,
            "x2_pct": 31.88,
            "y2_pct": 45.0,
        },
    },
]


class FakeDetectionService(DetectionService):
    """Returns canned detections so tests run without a real model."""

    def detect(self, request: Any) -> list[dict[str, float]]:
        return MOCK_DETECTIONS


def _fake_detection_service() -> FakeDetectionService:
    return FakeDetectionService()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """TestClient with the real pipeline mocked out."""
    from app.services.detection_service import get_detection_service

    app.dependency_overrides[get_detection_service] = _fake_detection_service
    app.state.detector_ready = True
    app.state.startup_error = None
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.state.detector_ready = False
    app.state.startup_error = None


@pytest.fixture()
def sample_video_b64() -> str:
    """Tiny valid base64 string (not a real video, but passes schema validation)."""
    return base64.b64encode(b"\x00\x00\x00\x1cftypisom").decode()


# ---------------------------------------------------------------------------
# Client's real test payloads (from conversation)
# ---------------------------------------------------------------------------


@pytest.fixture()
def basketball_payload() -> dict[str, Any]:
    """Basketball test case — camelCase (as Clipt backend would send)."""
    return {
        "videoUrl": "https://www.youtube.com/live/SyXhzqhTuzI?feature=shared",
        "jerseyNumber": 2,
        "jerseyColor": "white",
        "sport": "basketball",
        "position": "guard",
    }


@pytest.fixture()
def football_payload() -> dict[str, Any]:
    """Football test case — camelCase."""
    return {
        "videoUrl": "https://www.youtube.com/live/BMsdbAVOUPM?feature=shared",
        "jerseyNumber": 2,
        "jerseyColor": "blue",
        "sport": "football",
        "position": "quarterback",
    }


@pytest.fixture()
def lacrosse_payload() -> dict[str, Any]:
    """Lacrosse test case — camelCase."""
    return {
        "videoUrl": "https://www.youtube.com/live/8MkNFAbPcwo?feature=shared",
        "jerseyNumber": 26,
        "jerseyColor": "white",
        "sport": "lacrosse",
        "position": "midfielder",
    }


@pytest.fixture()
def snake_case_payload() -> dict[str, Any]:
    """Same basketball case but using snake_case field names."""
    return {
        "video_url": "https://www.youtube.com/live/SyXhzqhTuzI?feature=shared",
        "jersey_number": 2,
        "jersey_color": "white",
        "sport": "basketball",
        "position": "guard",
    }


@pytest.fixture()
def integration_guide_payload() -> dict[str, Any]:
    """Payload matching the integration guide example (no position field)."""
    return {
        "video_url": "https://www.youtube.com/live/SyXhzqhTuzI?feature=shared",
        "jersey_number": 2,
        "jersey_color": "white",
        "sport": "basketball",
    }
