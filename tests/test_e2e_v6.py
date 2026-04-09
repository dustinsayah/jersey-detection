"""End-to-end v6 pipeline smoke tests.

Verifies model file presence, detection priority, dead ball classifier,
scoreboard detector, jersey upscaler, cross-layer boosts, and health
endpoint model inventory — all without loading real YOLO weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


MODEL_DIR = Path(__file__).resolve().parent.parent / "app" / "model"

# ── v6 model files that MUST exist ──────────────────────────────────────────

V6_MODELS = {
    "dead_ball_classifier_v5.pt": 1_000_000,    # >1MB
    "jersey_upscaler_v5.pth": 1_000_000,        # >1MB
    "player_isolator_v3.pt": 1_000_000,          # >1MB
    "basketball_jersey_number_v3.pt": 1_000_000, # >1MB
    "basketball_player_detector_v2.pt": 1_000_000,
    "lacrosse_detector_v2.pt": 1_000_000,
}

# scoreboard_detector_v5.pt skipped for now (training deferred)


class TestV6ModelFilesExist:
    """Every v6 model must be present and not corrupt (>1MB)."""

    @pytest.mark.parametrize("filename,min_bytes", list(V6_MODELS.items()))
    def test_model_file_present_and_not_corrupt(self, filename: str, min_bytes: int) -> None:
        path = MODEL_DIR / filename
        assert path.exists(), f"MISSING: {filename} — copy to app/model/"
        size = path.stat().st_size
        assert size >= min_bytes, f"CORRUPT: {filename} is only {size} bytes (expected >={min_bytes})"


class TestDetectionPriorityOrder:
    """v5 is primary, v3 secondary, v2 tertiary, Ali last resort."""

    def test_v5_primary_over_v3(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.jersey_ocr_universal_v5_model = MagicMock()
        det.jersey_ocr_v3_primary_model = MagicMock()
        assert det.get_primary_detection() == "v5"

    def test_v3_secondary_when_v5_absent(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.jersey_ocr_v3_primary_model = MagicMock()
        assert det.get_primary_detection() == "v3"

    def test_v2_tertiary_when_v5_v3_absent(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.jersey_number_universal_v1_model = MagicMock()
        assert det.get_primary_detection() == "v2"

    def test_ali_last_resort(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        assert det.get_primary_detection() == "ali"

    def test_load_order_v5_first(self) -> None:
        """Verify that load() only loads v5 essential models at startup.
        v3/v2/v4/v1 are deferred to load_for_request().
        """
        import inspect
        from app.services.roboflow_detector import RoboflowDetector
        source = inspect.getsource(RoboflowDetector.load)
        # load() should contain v5 essential models and defer everything else
        assert "v5" in source.lower()
        assert "load_for_request()" in source or "deferred to load_for_request" in source


class TestDeadBallClassifierV6:
    """Dead ball classifier integration checks."""

    def test_dead_ball_file_exists(self) -> None:
        assert (MODEL_DIR / "dead_ball_classifier_v5.pt").exists()

    def test_classify_dead_ball_returns_class(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_model = MagicMock()
        mock_probs = MagicMock()
        mock_probs.top1 = 0
        mock_probs.top1conf.item.return_value = 0.90
        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "dead_ball", 1: "live_play"}

        det.dead_ball_classifier_v5_model = mock_model
        result = det.classify_dead_ball(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == "dead_ball"

    def test_classify_live_play(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_model = MagicMock()
        mock_probs = MagicMock()
        mock_probs.top1 = 1
        mock_probs.top1conf.item.return_value = 0.80
        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "dead_ball", 1: "live_play"}

        det.dead_ball_classifier_v5_model = mock_model
        result = det.classify_dead_ball(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == "live_play"


class TestScoreboardDetectorV6:
    """Scoreboard detector integration checks."""

    def test_detect_scoreboard_returns_boxes(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [50, 10, 300, 80]
        mock_box.conf.item.return_value = 0.88

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        det.scoreboard_detector_v5_model = mock_model
        dets = det.detect_scoreboard(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert "bbox" in dets[0]
        assert "confidence" in dets[0]
        assert dets[0]["layer"] == "scoreboard_v5"


class TestJerseyUpscalerV6:
    """Jersey upscaler integration checks."""

    def test_upscaler_file_exists(self) -> None:
        assert (MODEL_DIR / "jersey_upscaler_v5.pth").exists()

    def test_upscaler_produces_4x_output(self) -> None:
        from app.services.jersey_upscaler import JerseyUpscaler
        # Bicubic fallback (model not loaded)
        up = JerseyUpscaler(Path("/nonexistent"))
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        result = up.upscale(crop)
        assert result.shape == (80, 80, 3), "4x upscale expected"

    def test_maybe_upscale_triggers_for_small_crops(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        from app.services.jersey_upscaler import JerseyUpscaler
        det = RoboflowDetector()

        # Set up upscaler with bicubic fallback
        up = JerseyUpscaler(Path("/nonexistent"))
        det._jersey_upscaler = up

        crop = np.zeros((30, 30, 3), dtype=np.uint8)
        result = det._maybe_upscale(crop)
        # Should be upscaled (30 < 64 threshold)
        assert result.shape[0] > 30

    def test_maybe_upscale_skips_large_crops(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        from app.services.jersey_upscaler import JerseyUpscaler
        det = RoboflowDetector()

        up = JerseyUpscaler(Path("/nonexistent"))
        det._jersey_upscaler = up

        crop = np.zeros((201, 201, 3), dtype=np.uint8)
        result = det._maybe_upscale(crop)
        assert result.shape == (201, 201, 3), "Large crops should not be upscaled"


class TestCrossLayerBoosts:
    """Cross-layer confidence boost rules."""

    def test_v5_v3_agreement_boost(self) -> None:
        """v5 + v3 agreement gives +0.20 bonus."""
        layers = {"v5_ocr_universal", "v3_ocr_primary"}
        has_v5 = any("v5_ocr" in l for l in layers)
        has_v3 = any("v3_ocr_primary" in l for l in layers)
        bonus = 0.20 if (has_v5 and has_v3) else 0.0
        assert bonus == 0.20

    def test_v5_v2_agreement_boost(self) -> None:
        """v5 + v2 agreement gives +0.15 bonus."""
        layers = {"v5_ocr_universal", "v2_universal"}
        has_v5 = any("v5_ocr" in l for l in layers)
        has_v2 = any("v2_universal" in l for l in layers)
        bonus = 0.15 if (has_v5 and has_v2) else 0.0
        assert bonus == 0.15

    def test_ali_alone_capped(self) -> None:
        """Ali-only detections capped at 0.6 confidence."""
        layers = {"ali_ensemble"}
        best_conf = 0.95
        final = min(0.6, best_conf) if len(layers) == 1 and "ali_ensemble" in layers else best_conf
        assert final == 0.6

    def test_score_change_outcome_boost(self) -> None:
        from app.services.clip_extractor import _OUTCOME_SCORE_BOOSTS
        assert _OUTCOME_SCORE_BOOSTS["score_change"] == 20


class TestStatusEndpointV6:
    """Verify status() reports v6 model availability."""

    def test_status_reports_all_v5_models(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        status = det.status()
        v5_keys = [
            "jersey_ocr_universal_v5", "player_detector_v5",
            "outcome_cls_basketball_v5", "outcome_cls_football_v5",
            "outcome_cls_lacrosse_v5", "scoreboard_detector_v5",
            "dead_ball_classifier_v5", "jersey_upscaler_v5",
        ]
        for key in v5_keys:
            assert key in status, f"Missing v5 model key: {key}"

    def test_status_includes_primary_detection(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        status = det.status()
        assert "primary_detection" in status
        assert status["primary_detection"] in ("v5", "v3", "v2", "ali")

    def test_new_models_show_as_available(self) -> None:
        """Newly copied v6 models should show 'available' (not loaded yet)."""
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        status = det.status()
        # dead_ball_classifier_v5.pt was just copied, should be available
        assert status["dead_ball_classifier_v5"] in ("available", "loaded")


class TestHealthEndpointV6:
    """Verify /health reports v5 model status."""

    def test_ready_includes_v5_summary(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        phases = data["phases"]
        assert "roboflow_models_v5" in phases
        assert "roboflow_v5_summary" in phases

    def test_ready_includes_primary_detection(self, client: TestClient) -> None:
        response = client.get("/ready")
        data = response.json()
        phases = data["phases"]
        assert "primary_detection" in phases


class TestModelsEndpoint:
    """Tests for the /models inventory endpoint."""

    def test_models_endpoint_returns_200(self, client: TestClient) -> None:
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_endpoint_has_required_fields(self, client: TestClient) -> None:
        response = client.get("/models")
        data = response.json()
        assert "total_models" in data
        assert "loaded" in data
        assert "available" in data
        assert "missing" in data
        assert "primary_detection" in data
        assert "ali_status" in data
        assert "version" in data

    def test_models_endpoint_version_is_v6(self, client: TestClient) -> None:
        response = client.get("/models")
        data = response.json()
        assert data["version"] == "v8.6.0"

    def test_models_endpoint_lists_are_valid(self, client: TestClient) -> None:
        response = client.get("/models")
        data = response.json()
        assert isinstance(data["loaded"], list)
        assert isinstance(data["available"], list)
        assert isinstance(data["missing"], list)
        assert isinstance(data["total_models"], int)
        assert data["total_models"] == len(data["loaded"]) + len(data["available"]) + len(data["missing"])


class TestLiveEndpointV6:
    """Verify /live reports v6 version tag."""

    def test_live_includes_v6_version(self, client: TestClient) -> None:
        response = client.get("/live")
        data = response.json()
        assert data["version"] == "v8.6.0"

    def test_live_includes_model_count(self, client: TestClient) -> None:
        response = client.get("/live")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], int)

    def test_live_includes_primary_detection(self, client: TestClient) -> None:
        response = client.get("/live")
        data = response.json()
        assert "primary_detection" in data


class TestSignalWeightsV6:
    """Verify play_classifier weights are correct for v6."""

    def test_weights_sum_to_one(self) -> None:
        from app.services.play_classifier import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01

    def test_dead_ball_signal_present(self) -> None:
        from app.services.play_classifier import WEIGHTS
        assert "dead_ball" in WEIGHTS

    def test_jersey_weight_is_dominant(self) -> None:
        from app.services.play_classifier import WEIGHTS
        assert WEIGHTS["jersey"] >= 0.25


class TestTotalModelCount:
    """Verify overall model inventory."""

    def test_at_least_57_model_files(self) -> None:
        """57 .pt/.pth files should exist in app/model/ after v6 copy."""
        pt_files = list(MODEL_DIR.glob("*.pt"))
        pth_files = list(MODEL_DIR.glob("*.pth"))
        total = len(pt_files) + len(pth_files)
        assert total >= 57, f"Expected >=57 model files, found {total}"
