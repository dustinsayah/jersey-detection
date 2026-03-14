"""Behavior tests for runtime settings and pipeline helpers."""

from __future__ import annotations

from dataclasses import replace
import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, create_app
from app.services.detection_pipeline import detect_jersey_in_frames
from app.services.detection_pipeline import _download_youtube_video
from app.services.detection_runtime import DetectedFrame, InMemoryFrame, PipelineSettings
from app.services.detection_service import DetectionService
from app.services.detection_service import get_detection_service


class _FakeDetectionService(DetectionService):
    def detect(self, request):
        return []


def _fake_detection_service() -> _FakeDetectionService:
    return _FakeDetectionService()


class TestPipelineSettings:
    def test_defaults_are_full_game_safe(self, monkeypatch) -> None:
        monkeypatch.delenv("YOUTUBE_CLIP_SECONDS", raising=False)
        monkeypatch.delenv("EARLY_EXIT_CONSECUTIVE", raising=False)

        settings = PipelineSettings()

        assert settings.youtube_clip_seconds is None
        assert settings.early_exit_consecutive == 0

    def test_detected_frame_exports_bbox_percentages(self) -> None:
        payload = DetectedFrame(
            timestamp=8.4,
            confidence=0.92,
            x1=320,
            y1=180,
            x2=480,
            y2=540,
            frame_w=1280,
            frame_h=720,
        ).to_dict()

        assert payload["bbox"] == {
            "x1": 320,
            "y1": 180,
            "x2": 480,
            "y2": 540,
            "x1_pct": 25.0,
            "y1_pct": 25.0,
            "x2_pct": 37.5,
            "y2_pct": 75.0,
        }


class TestYoutubeDownloadBehavior:
    def test_download_youtube_video_uses_full_video_when_clip_limit_unset(
        self, tmp_path
    ) -> None:
        commands: list[list[str]] = []
        settings = PipelineSettings()

        def fake_run(command, **kwargs):
            commands.append(command)
            (tmp_path / "youtube_input.mp4").write_bytes(b"video")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with patch("app.services.detection_pipeline.subprocess.run", side_effect=fake_run):
            output = _download_youtube_video(
                video_url="https://www.youtube.com/watch?v=abc123",
                work_dir=tmp_path,
                settings=settings,
            )

        assert output.name == "youtube_input.mp4"
        assert "--download-sections" not in commands[0]

    def test_download_youtube_video_applies_clip_limit_when_configured(
        self, tmp_path
    ) -> None:
        commands: list[list[str]] = []
        settings = replace(PipelineSettings(), youtube_clip_seconds=120)

        def fake_run(command, **kwargs):
            commands.append(command)
            (tmp_path / "youtube_input.mp4").write_bytes(b"video")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with patch("app.services.detection_pipeline.subprocess.run", side_effect=fake_run):
            _download_youtube_video(
                video_url="https://www.youtube.com/watch?v=abc123",
                work_dir=tmp_path,
                settings=settings,
            )

        assert "--download-sections" in commands[0]
        option_index = commands[0].index("--download-sections")
        assert commands[0][option_index + 1] == "*0-120"
        ffmpeg_index = commands[0].index("--ffmpeg-location")
        assert commands[0][ffmpeg_index + 1] == settings.ffmpeg_binary


class TestReadinessGuards:
    def test_detect_returns_503_when_detector_is_not_ready(self) -> None:
        app.dependency_overrides[get_detection_service] = _fake_detection_service
        app.state.detector_ready = False
        app.state.startup_error = "missing jersey model"

        try:
            response = TestClient(app).post(
                "/detect",
                json={
                    "videoUrl": "https://example.com/game.mp4",
                    "jerseyNumber": 2,
                    "jerseyColor": "white",
                    "sport": "basketball",
                },
            )
        finally:
            app.dependency_overrides.clear()
            app.state.detector_ready = False
            app.state.startup_error = None

        assert response.status_code == 503
        assert "missing jersey model" in response.json()["error"]

    def test_detect_not_ready_is_logged(self, caplog) -> None:
        app.dependency_overrides[get_detection_service] = _fake_detection_service
        app.state.detector_ready = False
        app.state.startup_error = "missing jersey model"
        caplog.set_level(logging.WARNING)

        try:
            response = TestClient(app).post(
                "/detect",
                json={
                    "videoUrl": "https://example.com/game.mp4",
                    "jerseyNumber": 2,
                    "jerseyColor": "white",
                    "sport": "basketball",
                },
            )
        finally:
            app.dependency_overrides.clear()
            app.state.detector_ready = False
            app.state.startup_error = None

        assert response.status_code == 503
        assert "detect.rejected_not_ready" in caplog.text
        assert "startup_error=missing jersey model" in caplog.text


class TestCorsConfiguration:
    def test_create_app_enables_local_dev_cors_by_default(self) -> None:
        test_app = create_app()
        client = TestClient(test_app)

        response = client.options(
            "/detect",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_create_app_uses_explicit_cors_origins(self, monkeypatch) -> None:
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://clipt.ai,https://www.clipt.ai")
        test_app = create_app()
        client = TestClient(test_app)

        response = client.options(
            "/detect",
            headers={
                "Origin": "https://clipt.ai",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://clipt.ai"


class _FakeDetector:
    def detect_persons_batch(self, frames):
        return [[] for _ in frames]


class TestPipelineLogging:
    def test_detect_logs_source_and_empty_result_summary(self, tmp_path, caplog) -> None:
        sample_video = tmp_path / "sample.mp4"
        sample_video.write_bytes(b"video")
        frame = InMemoryFrame(
            index=0,
            timestamp=0.0,
            image=np.zeros((32, 32, 3), dtype=np.uint8),
        )
        caplog.set_level(logging.INFO)

        with patch(
            "app.services.detection_pipeline._resolve_video_source",
            return_value=sample_video,
        ), patch(
            "app.services.detection_pipeline._get_video_duration_seconds",
            return_value=12.5,
        ), patch(
            "app.services.detection_pipeline._iter_frames_in_memory",
            return_value=iter([[frame]]),
        ), patch(
            "app.services.detection_pipeline.get_or_create_detector",
            return_value=_FakeDetector(),
        ):
            result = detect_jersey_in_frames(
                video_url="https://example.com/game.mp4",
                video_path=None,
                video_bytes=None,
                jersey_number=2,
                jersey_color="white",
                sport="basketball",
                position="guard",
                settings=PipelineSettings(),
            )

        assert result == []
        assert "detect.source_resolved" in caplog.text
        assert "source=video_url" in caplog.text
        assert "detect.empty_result" in caplog.text
        assert "no_persons=1" in caplog.text
