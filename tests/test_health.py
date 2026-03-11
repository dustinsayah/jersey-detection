"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealth:
    def test_live_returns_ok(self, client: TestClient) -> None:
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_returns_ok(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_returns_error_when_detector_not_ready(self, client: TestClient) -> None:
        client.app.state.detector_ready = False
        client.app.state.startup_error = "missing jersey model"
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "error",
            "detail": "missing jersey model",
        }

    def test_health_post_not_allowed(self, client: TestClient) -> None:
        response = client.post("/health")
        assert response.status_code == 405
