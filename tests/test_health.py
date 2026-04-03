"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    def test_live_returns_ok(self, client: TestClient) -> None:
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_ok_when_ready(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "detector_ready" in data

    def test_ready_returns_ok(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ready_returns_error_when_detector_not_ready(self, client: TestClient) -> None:
        client.app.state.detector_ready = False
        client.app.state.startup_error = "missing jersey model"
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["detail"] == "missing jersey model"
        assert "hint" in data
        assert "phases" in data

    def test_ready_503_includes_actionable_hint(self, client: TestClient) -> None:
        """Regression: /ready 503 must include an actionable hint for debugging."""
        client.app.state.detector_ready = False
        client.app.state.startup_error = "OOM during model load"
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert "hint" in data
        assert "Railway" in data["hint"] or "logs" in data["hint"]

    def test_health_post_not_allowed(self, client: TestClient) -> None:
        response = client.post("/health")
        assert response.status_code == 405

    def test_health_returns_200_even_when_not_ready(self, client: TestClient) -> None:
        """Liveness probe /health must always return 200."""
        client.app.state.detector_ready = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warming_up"

    def test_health_does_not_load_roboflow_models(self, client: TestClient) -> None:
        """Regression: /health must NOT trigger roboflow_detector.load().

        Before the fix, calling /health triggered roboflow_detector.status()
        which called self.load() and eagerly loaded ALL 50+ YOLO models,
        causing OOM and /detect 503.
        """
        from app.services.roboflow_detector import roboflow_detector

        # Reset loaded state
        was_loaded = roboflow_detector._loaded
        roboflow_detector._loaded = False

        try:
            response = client.get("/health")
            assert response.status_code == 200
            # The health check should NOT have loaded models
            assert roboflow_detector._loaded is False
        finally:
            # Restore
            roboflow_detector._loaded = was_loaded
