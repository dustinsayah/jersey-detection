"""Regression tests for /detect 503 and YouTube URL normalization.

These tests cover the two production blockers:
1. /detect returns 503 when roboflow_detector.status() eagerly loads all models
2. /analyze fails with "All 5 YouTube download strategies failed" when URL has &t=36s
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# /detect readiness tests
# ---------------------------------------------------------------------------


class TestDetect503:
    """Verify /detect returns 503 with actionable error when detector not ready."""

    def test_detect_503_when_not_ready(self, client: TestClient) -> None:
        client.app.state.detector_ready = False
        client.app.state.startup_error = "jersey_number_yolo11m.pt not found"
        response = client.post(
            "/detect",
            json={
                "videoUrl": "https://example.com/test.mp4",
                "jerseyNumber": 10,
                "jerseyColor": "blue",
                "sport": "basketball",
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "not ready" in data["error"].lower()
        assert "startup_error" in data
        assert data["startup_error"] == "jersey_number_yolo11m.pt not found"
        assert "hint" in data

    def test_detect_503_hint_is_actionable(self, client: TestClient) -> None:
        """The 503 hint must tell the user what to check."""
        client.app.state.detector_ready = False
        client.app.state.startup_error = "timeout"
        response = client.post(
            "/detect",
            json={
                "videoUrl": "https://example.com/test.mp4",
                "jerseyNumber": 10,
                "jerseyColor": "blue",
                "sport": "basketball",
            },
        )
        assert response.status_code == 503
        data = response.json()
        hint = data.get("hint", "")
        assert "model" in hint.lower() or "Railway" in hint or "logs" in hint.lower()

    def test_detect_200_when_ready(self, client: TestClient) -> None:
        """When detector_ready is True, /detect should work (mocked)."""
        response = client.post(
            "/detect",
            json={
                "videoUrl": "https://example.com/test.mp4",
                "jerseyNumber": 10,
                "jerseyColor": "blue",
                "sport": "basketball",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_analyze_503_when_not_ready(self, client: TestClient) -> None:
        """Verify /analyze also returns 503 when detector not ready."""
        client.app.state.detector_ready = False
        client.app.state.startup_error = "OOM during startup"
        response = client.post(
            "/analyze",
            json={
                "videoUrl": "https://www.youtube.com/watch?v=test123test",
                "jerseyNumber": 5,
                "jerseyColor": "white",
                "sport": "basketball",
            },
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "not ready" in data["error"].lower()


# ---------------------------------------------------------------------------
# YouTube URL normalization tests
# ---------------------------------------------------------------------------


class TestYouTubeUrlNormalization:
    """Verify normalize_youtube_url strips tracking/timestamp params."""

    def test_clean_url_unchanged(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 0.0

    def test_strips_t_param_seconds(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY&t=36s")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 36.0

    def test_strips_t_param_no_suffix(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY&t=120")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 120.0

    def test_strips_t_param_minutes_seconds(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY&t=1m30s")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 90.0

    def test_strips_time_continue(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY&time_continue=45")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 45.0

    def test_strips_feature_param(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/watch?v=JPbQ3i3gkCY&feature=shared")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 0.0

    def test_strips_list_and_index(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url(
            "https://www.youtube.com/watch?v=JPbQ3i3gkCY&list=PLtest123&index=5"
        )
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 0.0

    def test_strips_si_param(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url(
            "https://www.youtube.com/watch?v=JPbQ3i3gkCY&si=abc123def"
        )
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 0.0

    def test_strips_combined_params(self) -> None:
        """The exact URL from the production failure."""
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url(
            "https://www.youtube.com/watch?v=JPbQ3i3gkCY&t=36s"
        )
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 36.0

    def test_youtu_be_short_url(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://youtu.be/JPbQ3i3gkCY?t=60")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 60.0

    def test_shorts_url(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://www.youtube.com/shorts/JPbQ3i3gkCY")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 0.0

    def test_mobile_url(self) -> None:
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://m.youtube.com/watch?v=JPbQ3i3gkCY&t=10s")
        assert url == "https://www.youtube.com/watch?v=JPbQ3i3gkCY"
        assert t == 10.0

    def test_non_youtube_url_passthrough(self) -> None:
        """Non-YouTube URLs should pass through unchanged."""
        from app.services.youtube_proxy import normalize_youtube_url

        url, t = normalize_youtube_url("https://res.cloudinary.com/dc33vjyyv/video.mp4")
        assert url == "https://res.cloudinary.com/dc33vjyyv/video.mp4"
        assert t == 0.0


# ---------------------------------------------------------------------------
# RoboflowDetector.status() regression
# ---------------------------------------------------------------------------


class TestRoboflowDetectorStatus:
    """Verify status() does not eagerly load models."""

    def test_status_does_not_call_load(self) -> None:
        """Regression: status() must check file existence, not load models."""
        from app.services.roboflow_detector import RoboflowDetector

        detector = RoboflowDetector()
        assert detector._loaded is False
        _ = detector.status()
        assert detector._loaded is False

    def test_status_reports_file_existence(self) -> None:
        """status() reports 'available' for files that exist but aren't loaded."""
        from app.services.roboflow_detector import RoboflowDetector
        import os

        detector = RoboflowDetector()
        status = detector.status()

        # All models should be either 'available', 'loaded', or 'missing'
        for key, value in status.items():
            if key == "basketball_jersey_ocr":
                continue  # This one has a special status
            assert value in ("available", "loaded", "missing"), \
                f"Model {key} has unexpected status: {value}"


# ---------------------------------------------------------------------------
# /analyze error response structure
# ---------------------------------------------------------------------------


class TestAnalyzeErrorResponse:
    """Verify /analyze error responses include debug info."""

    def test_error_response_has_layer_breakdown(self) -> None:
        from app.services.analyze_pipeline import _error_response

        result = _error_response(
            "YouTube download failed: timeout",
            elapsed=5.0,
            layer_timings={"youtube_download": {"status": "failed", "error": "timeout"}},
        )
        assert result["error"] == "YouTube download failed: timeout"
        assert result["clips"] == []
        assert result["debug"]["layer_breakdown"]["youtube_download"]["status"] == "failed"

    def test_error_response_without_layer_timings(self) -> None:
        from app.services.analyze_pipeline import _error_response

        result = _error_response("generic error", elapsed=1.0)
        assert result["error"] == "generic error"
        assert result["debug"]["layer_breakdown"] == {}
