# Request/response models for /analyze

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


from app.schemas.detect import normalize_position

SUPPORTED_SPORTS = {"basketball", "football", "lacrosse"}


# ── Response models ──────────────────────────────────────────────────────────

class SignalBreakdown(BaseModel):
    jersey: float = 0.0
    motion: float = 0.0
    audio: bool = False
    pose: str = ""
    crowd: float = 0.0


class DetectedClip(BaseModel):
    start_time: float = Field(alias="startTime")
    end_time: float = Field(alias="endTime")
    confidence: float = 0.0
    score: int = 0
    play_type: str = Field(default="game_action", alias="playType")
    grade: str = "Decent"
    jersey_visible: bool = Field(default=False, alias="jerseyVisible")
    jersey_number_seen: int | None = Field(default=None, alias="jerseyNumberSeen")
    tracking_id: int | None = Field(default=None, alias="trackingId")
    description: str = ""
    signals: SignalBreakdown = SignalBreakdown()

    model_config = ConfigDict(populate_by_name=True)


class AudioEvent(BaseModel):
    timestamp: float
    event_type: str = Field(alias="eventType")
    confidence: float = 0.0

    model_config = ConfigDict(populate_by_name=True)


class PlayerTrack(BaseModel):
    track_id: int = Field(alias="trackId")
    jersey_number: int | None = Field(default=None, alias="jerseyNumber")
    frames_tracked: int = Field(default=0, alias="framesTracked")

    model_config = ConfigDict(populate_by_name=True)


class DebugInfo(BaseModel):
    ali_detections: int = 0
    roboflow_detections: int = 0
    combined_detections: int = 0
    ali_working: bool = False
    ali_status: str = "not_run"
    youtube_strategy_used: str | None = None
    total_elapsed_ms: int = 0
    layers_that_contributed: list[str] = Field(default_factory=list)
    layer_breakdown: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class AnalyzeResponse(BaseModel):
    clips: list[DetectedClip] = []
    layer_used: str = Field(default="full_pipeline", alias="layerUsed")
    elapsed: float = 0.0
    video_duration: float = Field(default=0.0, alias="videoDuration")
    frames_processed: int = Field(default=0, alias="framesProcessed")
    audio_events: list[AudioEvent] = Field(default_factory=list, alias="audioEvents")
    player_tracks: list[PlayerTrack] = Field(default_factory=list, alias="playerTracks")
    game_stats: dict[str, Any] = Field(default_factory=dict, alias="gameStats")
    per_clip_stats: list[dict] = Field(default_factory=list, alias="perClipStats")
    actions_detected: list[dict] = Field(default_factory=list, alias="actionsDetected")
    debug: DebugInfo = Field(default_factory=DebugInfo)

    model_config = ConfigDict(populate_by_name=True)


# ── Request model ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(validate_default=True, populate_by_name=True)

    video_url: str | None = Field(default=None, alias="videoUrl")
    video_path: str | None = Field(default=None, alias="videoPath")
    jersey_number: int = Field(default=0, alias="jerseyNumber")
    jersey_color: str = Field(default="white", alias="jerseyColor")
    sport: str = "basketball"
    position: str | None = None

    time_range_start: float = Field(default=0, alias="timeRangeStart")
    time_range_end: float = Field(default=0, alias="timeRangeEnd")

    @model_validator(mode="before")
    @classmethod
    def _accept_snake_case_aliases(cls, data: Any) -> Any:
        """Accept common snake_case field names in addition to camelCase aliases.

        Maps: start_time → timeRangeStart, end_time → timeRangeEnd,
              jersey_number → jerseyNumber, jersey_color → jerseyColor,
              video_url → videoUrl
        """
        if isinstance(data, dict):
            _ALIAS_MAP = {
                "start_time": "timeRangeStart",
                "end_time": "timeRangeEnd",
            }
            for snake, camel in _ALIAS_MAP.items():
                if snake in data and camel not in data:
                    data[camel] = data.pop(snake)
        return data

    enable_audio: bool = Field(default=True, alias="enableAudio")
    enable_tracking: bool = Field(default=True, alias="enableTracking")
    enable_pose: bool = Field(default=True, alias="enablePose")

    quality_mode: str = Field(default="auto", alias="qualityMode")

    google_access_token: str | None = Field(default=None, alias="googleAccessToken")

    @field_validator("jersey_number", mode="before")
    @classmethod
    def _validate_jersey_number(cls, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("'jerseyNumber' must be an integer.") from error
        if not (0 <= parsed <= 99):
            raise ValueError("'jerseyNumber' must be between 0 and 99.")
        return parsed

    @field_validator("jersey_color", mode="before")
    @classmethod
    def _validate_jersey_color(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "white"
        return value.strip().lower()

    @field_validator("sport", mode="before")
    @classmethod
    def _validate_sport(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "basketball"
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_SPORTS:
            return "basketball"
        return normalized

    @field_validator("quality_mode", mode="before")
    @classmethod
    def _validate_quality_mode(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "auto"
        normalized = value.strip().lower()
        if normalized not in {"auto", "standard", "aggressive"}:
            return "auto"
        return normalized

    @field_validator("position", mode="before")
    @classmethod
    def _validate_position(cls, value: Any) -> str | None:
        return normalize_position(value)

    @model_validator(mode="after")
    def _validate_video_source(self) -> "AnalyzeRequest":
        if not self.video_url and not self.video_path:
            raise ValueError("Provide 'videoUrl' or 'videoPath'.")
        return self
