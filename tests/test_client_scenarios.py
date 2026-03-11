"""End-to-end client scenario tests.

Simulates the full Clipt backend flow:
  1. POST /detect with athlete's video + jersey info
  2. Validate the response can be consumed by scoreAndRankClips()
  3. Validate all three sports the client specified
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


class TestCliptBackendFlow:
    """Simulates what the Clipt Node.js backend does when it calls our API."""

    def _assert_valid_clipt_response(self, response_data: list[dict]) -> None:
        """Validate output matches what scoreAndRankClips() expects."""
        assert isinstance(response_data, list)
        for frame in response_data:
            assert set(frame.keys()) == {"timestamp", "confidence", "bbox"}
            # timestamp must be a non-negative float (seconds into video)
            assert isinstance(frame["timestamp"], (int, float))
            assert frame["timestamp"] >= 0.0
            # confidence must be 0.0–1.0
            assert isinstance(frame["confidence"], (int, float))
            assert 0.0 <= frame["confidence"] <= 1.0
            # bbox must include both absolute and normalized coordinates
            assert set(frame["bbox"].keys()) == {
                "x1", "y1", "x2", "y2", "x1_pct", "y1_pct", "x2_pct", "y2_pct"
            }

        # Timestamps should be sorted (pipeline dedupes and sorts)
        timestamps = [f["timestamp"] for f in response_data]
        assert timestamps == sorted(timestamps)

    def test_basketball_full_flow(
        self, client: TestClient, basketball_payload: dict[str, Any]
    ) -> None:
        """Basketball: Jersey #2, white, guard."""
        response = client.post("/detect", json=basketball_payload)
        assert response.status_code == 200
        self._assert_valid_clipt_response(response.json())

    def test_football_full_flow(
        self, client: TestClient, football_payload: dict[str, Any]
    ) -> None:
        """Football: Jersey #2, blue, quarterback."""
        response = client.post("/detect", json=football_payload)
        assert response.status_code == 200
        self._assert_valid_clipt_response(response.json())

    def test_lacrosse_full_flow(
        self, client: TestClient, lacrosse_payload: dict[str, Any]
    ) -> None:
        """Lacrosse: Jersey #26, white, midfielder."""
        response = client.post("/detect", json=lacrosse_payload)
        assert response.status_code == 200
        self._assert_valid_clipt_response(response.json())

    def test_without_position_still_works(
        self, client: TestClient, integration_guide_payload: dict[str, Any]
    ) -> None:
        """Client's integration guide has no position — must still work."""
        response = client.post("/detect", json=integration_guide_payload)
        assert response.status_code == 200
        self._assert_valid_clipt_response(response.json())


class TestColorVariations:
    """Client uses both color names and hex codes."""

    @pytest.mark.parametrize(
        "color",
        [
            "white",                # basketball jersey
            "blue",                 # football jersey
            "#0000FF",              # hex blue (integration guide example)
            "#FF0000",              # hex red (integration guide example)
            "royal blue",           # named color override
        ],
    )
    def test_various_jersey_colors(self, client: TestClient, color: str) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": color,
            "sport": "basketball",
        })
        assert response.status_code == 200


class TestSportSpecific:
    """Each supported sport should be accepted (case-insensitive)."""

    @pytest.mark.parametrize(
        "sport,position",
        [
            ("basketball", "guard"),
            ("Basketball", "Guard"),
            ("football", "quarterback"),
            ("Football", "Quarterback"),
            ("lacrosse", "midfielder"),
            ("Lacrosse", "Midfielder"),
        ],
    )
    def test_sport_and_position_combinations(
        self, client: TestClient, sport: str, position: str
    ) -> None:
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": sport,
            "position": position,
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("bad_sport", ["soccer", "baseball", "hockey", ""])
    def test_unsupported_sports_rejected_at_schema(
        self, client: TestClient, bad_sport: str
    ) -> None:
        """Only basketball, football, lacrosse are supported."""
        response = client.post("/detect", json={
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": bad_sport,
        })
        assert response.status_code == 400
