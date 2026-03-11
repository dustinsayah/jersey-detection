"""Tests for POST /detect endpoint.

Uses FakeDetectionService from conftest to avoid model/ffmpeg dependencies.
Validates request parsing, error handling, and response shape.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient


# ---- Successful detection requests ----------------------------------------


class TestDetectSuccess:
    def test_returns_list_of_detections(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=basketball_payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_each_detection_has_timestamp_confidence_and_bbox(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=basketball_payload)
        data = response.json()
        for item in data:
            assert "timestamp" in item
            assert "confidence" in item
            assert isinstance(item["timestamp"], (int, float))
            assert isinstance(item["confidence"], (int, float))
            assert 0.0 <= item["confidence"] <= 1.0
            assert set(item["bbox"].keys()) == {
                "x1", "y1", "x2", "y2", "x1_pct", "y1_pct", "x2_pct", "y2_pct"
            }

    def test_detection_output_matches_response_schema(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        """Output must match the detection schema including bbox percentages."""
        response = client.post("/detect", json=basketball_payload)
        data = response.json()
        for item in data:
            assert set(item.keys()) == {"timestamp", "confidence", "bbox"}

    def test_snake_case_request_works(
        self, client: TestClient, snake_case_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=snake_case_payload)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_camel_case_request_works(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=basketball_payload)
        assert response.status_code == 200

    def test_without_position_field(
        self, client: TestClient, integration_guide_payload: dict[str, Any]
    ) -> None:
        """Integration guide doesn't include position — must still succeed."""
        response = client.post("/detect", json=integration_guide_payload)
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# ---- Client's three real test cases (from conversation) -------------------


class TestClientTestCases:
    def test_basketball_jersey2_white_guard(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=basketball_payload)
        assert response.status_code == 200

    def test_football_jersey2_blue_quarterback(
        self, client: TestClient, football_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=football_payload)
        assert response.status_code == 200

    def test_lacrosse_jersey26_white_midfielder(
        self, client: TestClient, lacrosse_payload: dict[str, Any]
    ) -> None:
        response = client.post("/detect", json=lacrosse_payload)
        assert response.status_code == 200


# ---- Request validation error cases --------------------------------------


class TestDetectValidation:
    def test_empty_body_rejected(self, client: TestClient) -> None:
        response = client.post("/detect", content=b"", headers={"Content-Type": "application/json"})
        assert response.status_code == 400
        assert "error" in response.json()

    def test_non_json_rejected(self, client: TestClient) -> None:
        response = client.post("/detect", content=b"not json")
        assert response.status_code == 400

    def test_array_body_rejected(self, client: TestClient) -> None:
        """Body must be an object, not an array."""
        response = client.post("/detect", json=[1, 2, 3])
        assert response.status_code == 400
        assert "error" in response.json()

    def test_missing_jersey_number(self, client: TestClient) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert response.status_code == 400
        assert "jersey_number" in response.json()["error"].lower() or "integer" in response.json()["error"].lower()

    def test_missing_jersey_color(self, client: TestClient) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "sport": "basketball",
        })
        assert response.status_code == 400

    def test_missing_sport(self, client: TestClient) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": "white",
        })
        assert response.status_code == 400

    def test_missing_video_source(self, client: TestClient) -> None:
        response = client.post("/detect", json={
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert response.status_code == 400

    def test_invalid_jersey_number_out_of_range(self, client: TestClient) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 100,
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert response.status_code == 400

    def test_error_response_has_error_key(self, client: TestClient) -> None:
        """All 400 responses should have {"error": "..."}."""
        response = client.post("/detect", json={})
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert isinstance(body["error"], str)


# ---- Integration guide curl example reproduction -------------------------


class TestIntegrationGuideExamples:
    """Reproduce the exact payloads from AI_INTEGRATION_GUIDE.md."""

    def test_guide_example_payload(self, client: TestClient) -> None:
        """From the guide's 'Example API Call' section (adapted for our endpoint)."""
        response = client.post("/detect", json={
            "videoUrl": "https://www.youtube.com/watch?v=example123",
            "jerseyNumber": 23,
            "jerseyColor": "#0000FF",
            "sport": "Basketball",
        })
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Each item must match DetectedFrame shape
        for item in data:
            assert set(item.keys()) == {"timestamp", "confidence", "bbox"}

    def test_guide_readme_example_payload(self, client: TestClient) -> None:
        """From README.md example."""
        response = client.post("/detect", json={
            "video_url": "https://www.youtube.com/live/SyXhzqhTuzI?feature=shared",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
            "position": "guard",
        })
        assert response.status_code == 200

    def test_video_bytes_b64_input(self, client: TestClient, sample_video_b64: str) -> None:
        """Alternative input: base64 encoded video bytes."""
        response = client.post("/detect", json={
            "videoBytesB64": sample_video_b64,
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert response.status_code == 200

    def test_video_path_input(self, client: TestClient) -> None:
        """Alternative input: local file path."""
        response = client.post("/detect", json={
            "videoPath": "/tmp/game.mp4",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert response.status_code == 200
