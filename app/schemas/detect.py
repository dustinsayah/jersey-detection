# Request/response models for /detect

from __future__ import annotations

import base64
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

SUPPORTED_SPORTS = {"basketball", "football", "lacrosse"}


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    x1_pct: float
    y1_pct: float
    x2_pct: float
    y2_pct: float


class DetectionFrame(BaseModel):
    timestamp: float
    confidence: float
    bbox: BBox


class DetectRequest(BaseModel):
    model_config = ConfigDict(validate_default=True, populate_by_name=True)

    video_url: str | None = Field(default=None, alias="videoUrl")
    video_path: str | None = Field(default=None, alias="videoPath")
    video_bytes_b64: str | None = Field(default=None, alias="videoBytesB64")
    jersey_number: int | None = Field(default=None, alias="jerseyNumber")
    jersey_color: str | None = Field(default=None, alias="jerseyColor")
    sport: str | None = None
    position: str | None = None

    @field_validator("video_url", "video_path", mode="before")
    @classmethod
    def _validate_optional_path_or_url(
        cls, value: Any, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"'{info.field_name}' must be a string when provided.")
        return value.strip()

    @field_validator("jersey_number", mode="before")
    @classmethod
    def _validate_jersey_number(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("'jersey_number' must be an integer.") from error
        if not (0 <= parsed <= 99):
            raise ValueError("'jersey_number' must be between 0 and 99.")
        return parsed

    @field_validator("jersey_color", mode="before")
    @classmethod
    def _require_non_empty_string(cls, value: Any, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{info.field_name}' must be a non-empty string.")
        return value.strip()

    @field_validator("sport", mode="before")
    @classmethod
    def _validate_sport(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'sport' must be a non-empty string.")
        stripped = value.strip()
        normalized = stripped.lower()
        if normalized not in SUPPORTED_SPORTS:
            raise ValueError("'sport' must be one of: basketball, football, lacrosse.")
        return stripped

    @field_validator("position", mode="before")
    @classmethod
    def _validate_position(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    @field_validator("video_bytes_b64", mode="before")
    @classmethod
    def _validate_video_bytes_b64(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("'video_bytes_b64' must be a non-empty base64 string.")
        try:
            base64.b64decode(value, validate=True)
        except Exception as error:
            raise ValueError("'video_bytes_b64' is not valid base64.") from error
        return value

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "DetectRequest":
        if not (self.video_url or self.video_path or self.video_bytes_b64 is not None):
            raise ValueError(
                "Provide one of 'video_url', 'video_path', or 'video_bytes_b64'."
            )
        return self

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        video_bytes: bytes | None = None
        if self.video_bytes_b64 is not None:
            video_bytes = base64.b64decode(self.video_bytes_b64, validate=True)

        return {
            "video_url": self.video_url,
            "video_path": self.video_path,
            "video_bytes": video_bytes,
            "jersey_number": int(self.jersey_number),
            "jersey_color": str(self.jersey_color),
            "sport": str(self.sport).lower(),
            "position": (self.position or "").lower() or None,
        }


def first_validation_error_message(error: ValidationError) -> str:
    details = error.errors()
    if not details:
        return "Invalid request payload."
    message = str(details[0].get("msg", "Invalid request payload."))
    prefix = "Value error, "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message
