"""Tests for DetectRequest schema validation.

Covers camelCase/snake_case aliases, required fields, jersey_number range,
color handling, position optionality, and to_pipeline_kwargs conversion.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.detect import DetectRequest, first_validation_error_message


# ---- camelCase / snake_case acceptance ------------------------------------


class TestFieldAliases:
    """Integration guide sends camelCase; our internal code uses snake_case."""

    def test_camel_case_accepted(self) -> None:
        req = DetectRequest.model_validate({
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jerseyNumber": 2,
            "jerseyColor": "white",
            "sport": "basketball",
        })
        assert req.video_url == "https://youtube.com/watch?v=abc"
        assert req.jersey_number == 2
        assert req.jersey_color == "white"

    def test_snake_case_accepted(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://youtube.com/watch?v=abc",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
        })
        assert req.video_url == "https://youtube.com/watch?v=abc"
        assert req.jersey_number == 2

    def test_mixed_case_accepted(self) -> None:
        """Some fields camelCase, some snake_case — both work."""
        req = DetectRequest.model_validate({
            "videoUrl": "https://youtube.com/watch?v=abc",
            "jersey_number": 23,
            "jerseyColor": "#FF0000",
            "sport": "football",
        })
        assert req.video_url == "https://youtube.com/watch?v=abc"
        assert req.jersey_number == 23
        assert req.jersey_color == "#FF0000"


# ---- Video source validation ----------------------------------------------


class TestVideoSource:
    def test_video_url_accepted(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/video.mp4",
            "jersey_number": 5,
            "jersey_color": "red",
            "sport": "basketball",
        })
        assert req.video_url == "https://example.com/video.mp4"
        assert req.video_path is None
        assert req.video_bytes_b64 is None

    def test_video_path_accepted(self) -> None:
        req = DetectRequest.model_validate({
            "video_path": "/data/game.mp4",
            "jersey_number": 5,
            "jersey_color": "red",
            "sport": "basketball",
        })
        assert req.video_path == "/data/game.mp4"

    def test_video_bytes_b64_accepted(self) -> None:
        raw = base64.b64encode(b"\x00\x00\x00\x1cftypisom").decode()
        req = DetectRequest.model_validate({
            "video_bytes_b64": raw,
            "jersey_number": 5,
            "jersey_color": "red",
            "sport": "basketball",
        })
        assert req.video_bytes_b64 == raw

    def test_no_video_source_rejected(self) -> None:
        with pytest.raises(ValidationError, match="video_url.*video_path.*video_bytes_b64"):
            DetectRequest.model_validate({
                "jersey_number": 5,
                "jersey_color": "red",
                "sport": "basketball",
            })

    def test_invalid_base64_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not valid base64"):
            DetectRequest.model_validate({
                "video_bytes_b64": "not-valid-base64!!!",
                "jersey_number": 5,
                "jersey_color": "red",
                "sport": "basketball",
            })


# ---- Jersey number validation ---------------------------------------------


class TestJerseyNumber:
    @pytest.mark.parametrize("number", [0, 1, 2, 23, 26, 50, 99])
    def test_valid_numbers(self, number: int) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": number,
            "jersey_color": "white",
            "sport": "basketball",
        })
        assert req.jersey_number == number

    def test_string_number_coerced(self) -> None:
        """Client might send "2" as string — should still work."""
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": "2",
            "jersey_color": "white",
            "sport": "basketball",
        })
        assert req.jersey_number == 2

    @pytest.mark.parametrize("bad_number", [-1, 100, 999])
    def test_out_of_range_rejected(self, bad_number: int) -> None:
        with pytest.raises(ValidationError, match="between 0 and 99"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": bad_number,
                "jersey_color": "white",
                "sport": "basketball",
            })

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be an integer"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": "abc",
                "jersey_color": "white",
                "sport": "basketball",
            })

    def test_null_jersey_number_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be an integer"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": None,
                "jersey_color": "white",
                "sport": "basketball",
            })


# ---- Jersey color validation ----------------------------------------------


class TestJerseyColor:
    @pytest.mark.parametrize("color", ["white", "blue", "#FF0000", "royal blue", "#0000FF"])
    def test_valid_colors(self, color: str) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": color,
            "sport": "basketball",
        })
        assert req.jersey_color == color.strip()

    def test_empty_color_rejected(self) -> None:
        with pytest.raises(ValidationError, match="jersey_color"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": 2,
                "jersey_color": "",
                "sport": "basketball",
            })

    def test_null_color_rejected(self) -> None:
        with pytest.raises(ValidationError, match="jersey_color"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": 2,
                "jersey_color": None,
                "sport": "basketball",
            })


# ---- Sport validation -----------------------------------------------------


class TestSport:
    @pytest.mark.parametrize("sport", ["basketball", "football", "lacrosse",
                                        "Basketball", "FOOTBALL", "Lacrosse"])
    def test_valid_sports(self, sport: str) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": sport,
        })
        assert req.sport == sport.strip()

    def test_empty_sport_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sport"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": 2,
                "jersey_color": "white",
                "sport": "",
            })

    def test_missing_sport_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sport"):
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": 2,
                "jersey_color": "white",
            })


# ---- Position (optional per integration guide) ----------------------------


class TestPosition:
    def test_position_accepted_when_provided(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
            "position": "guard",
        })
        assert req.position == "guard"

    def test_position_optional(self) -> None:
        """Integration guide signature has no position — must not fail."""
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
        })
        assert req.position is None

    def test_position_null_accepted(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
            "position": None,
        })
        assert req.position is None

    def test_position_empty_string_becomes_none(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://example.com/v.mp4",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
            "position": "  ",
        })
        assert req.position is None


# ---- to_pipeline_kwargs ---------------------------------------------------


class TestToPipelineKwargs:
    def test_full_kwargs(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://youtube.com/watch?v=abc",
            "jersey_number": 26,
            "jersey_color": "white",
            "sport": "Lacrosse",
            "position": "Midfielder",
        })
        kwargs = req.to_pipeline_kwargs()
        assert kwargs["video_url"] == "https://youtube.com/watch?v=abc"
        assert kwargs["video_path"] is None
        assert kwargs["video_bytes"] is None
        assert kwargs["jersey_number"] == 26
        assert kwargs["jersey_color"] == "white"
        assert kwargs["sport"] == "lacrosse"  # lowercased
        assert kwargs["position"] == "midfielder"  # lowercased

    def test_kwargs_without_position(self) -> None:
        req = DetectRequest.model_validate({
            "video_url": "https://youtube.com/watch?v=abc",
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
        })
        kwargs = req.to_pipeline_kwargs()
        assert kwargs["position"] is None

    def test_kwargs_with_video_bytes(self) -> None:
        raw_bytes = b"\x00\x00\x00\x1cftypisom"
        b64 = base64.b64encode(raw_bytes).decode()
        req = DetectRequest.model_validate({
            "video_bytes_b64": b64,
            "jersey_number": 2,
            "jersey_color": "white",
            "sport": "basketball",
        })
        kwargs = req.to_pipeline_kwargs()
        assert kwargs["video_bytes"] == raw_bytes
        assert kwargs["video_url"] is None


# ---- first_validation_error_message helper --------------------------------


class TestFirstValidationErrorMessage:
    def test_extracts_message(self) -> None:
        try:
            DetectRequest.model_validate({
                "video_url": "https://example.com/v.mp4",
                "jersey_number": "abc",
                "jersey_color": "white",
                "sport": "basketball",
            })
        except ValidationError as error:
            message = first_validation_error_message(error)
            assert "jersey_number" in message.lower() or "integer" in message.lower()
