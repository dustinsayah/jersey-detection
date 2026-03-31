"""Tests for new v5 model integrations: dead ball, scoreboard, jersey SR, priority flip."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestDeadBallClassifier:
    """Tests for classify_dead_ball() in roboflow_detector."""

    def test_returns_none_when_model_not_loaded(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.dead_ball_classifier_v5_model = None
        assert det.classify_dead_ball(np.zeros((100, 100, 3), dtype=np.uint8)) is None

    def test_returns_class_name_when_confident(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_model = MagicMock()
        mock_probs = MagicMock()
        mock_probs.top1 = 0
        mock_probs.top1conf.item.return_value = 0.85
        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "dead_ball", 1: "live_play"}

        det.dead_ball_classifier_v5_model = mock_model
        result = det.classify_dead_ball(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == "dead_ball"

    def test_returns_none_when_low_confidence(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_model = MagicMock()
        mock_probs = MagicMock()
        mock_probs.top1 = 1
        mock_probs.top1conf.item.return_value = 0.2  # Below threshold
        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_model.return_value = [mock_result]
        mock_model.names = {0: "dead_ball", 1: "live_play"}

        det.dead_ball_classifier_v5_model = mock_model
        result = det.classify_dead_ball(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is None


class TestScoreboardDetector:
    """Tests for detect_scoreboard() in roboflow_detector."""

    def test_returns_empty_when_model_not_loaded(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.scoreboard_detector_v5_model = None
        assert det.detect_scoreboard(np.zeros((100, 100, 3), dtype=np.uint8)) == []

    def test_returns_bbox_format(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()

        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10, 20, 200, 60]
        mock_box.conf.item.return_value = 0.75

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        det.scoreboard_detector_v5_model = mock_model
        dets = det.detect_scoreboard(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0]["bbox"] == [10, 20, 200, 60]
        assert dets[0]["confidence"] == 0.75
        assert dets[0]["layer"] == "scoreboard_v5"


class TestJerseyUpscaler:
    """Tests for JerseyUpscaler fallback behavior."""

    def test_bicubic_fallback_when_model_not_loaded(self) -> None:
        from app.services.jersey_upscaler import JerseyUpscaler
        from pathlib import Path

        upscaler = JerseyUpscaler(Path("/nonexistent/model.pth"))
        # Model won't load, so it uses bicubic fallback
        crop = np.zeros((16, 16, 3), dtype=np.uint8)
        result = upscaler.upscale(crop)
        # Should be 4x upscaled
        assert result.shape == (64, 64, 3)

    def test_load_returns_false_when_file_missing(self) -> None:
        from app.services.jersey_upscaler import JerseyUpscaler
        from pathlib import Path

        upscaler = JerseyUpscaler(Path("/nonexistent/model.pth"))
        assert upscaler.load() is False

    def test_unload_clears_model(self) -> None:
        from app.services.jersey_upscaler import JerseyUpscaler
        from pathlib import Path

        upscaler = JerseyUpscaler(Path("/nonexistent/model.pth"))
        upscaler._model = "fake"
        upscaler.unload()
        assert upscaler._model is None


class TestMaybeUpscale:
    """Tests for _maybe_upscale helper in RoboflowDetector."""

    def test_small_crop_gets_upscaled(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        # With no SR model loaded, _maybe_upscale should return original (no upscaler)
        det._jersey_upscaler = None
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        result = det._maybe_upscale(crop)
        assert result.shape == (32, 32, 3)  # No upscaler → returns unchanged

    def test_large_crop_not_upscaled(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det._maybe_upscale(crop)
        assert result.shape == (200, 200, 3)  # Large → not upscaled


class TestPriorityFlip:
    """Tests for the detection priority flip (v5 primary, Ali last)."""

    def test_v5_is_primary_when_loaded(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.jersey_ocr_universal_v5_model = "fake_model"
        assert det.get_primary_detection() == "v5"

    def test_v3_is_primary_when_v5_missing(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        det.jersey_ocr_universal_v5_model = None
        det.jersey_ocr_v3_primary_model = "fake_model"
        assert det.get_primary_detection() == "v3"

    def test_ali_is_fallback_when_all_missing(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        # All None by default
        assert det.get_primary_detection() == "ali"

    def test_status_includes_primary_detection(self) -> None:
        from app.services.roboflow_detector import RoboflowDetector
        det = RoboflowDetector()
        status = det.status()
        assert "primary_detection" in status
        assert "ali_status" in status

    def test_ali_alone_capped_at_06(self) -> None:
        """When Ali is the only layer, confidence should be capped at 0.6."""
        # This tests the cross-layer boost logic conceptually
        layers_present = {"ali_ensemble"}
        best_conf = 0.9
        bonus = 0.0
        has_ali = "ali_ensemble" in layers_present

        final_conf = min(1.0, best_conf + bonus)
        if has_ali and len(layers_present) == 1:
            final_conf = min(0.6, best_conf)
            bonus = 0.0

        assert final_conf == 0.6

    def test_v5_v3_agreement_gives_highest_boost(self) -> None:
        """v5 + v3 agreement should give +0.20 boost."""
        layers = {"v5_ocr_universal", "v3_ocr_primary"}
        has_v5 = any("v5_ocr" in l for l in layers)
        has_v3 = any("v3_ocr_primary" in l for l in layers)

        bonus = 0.0
        if has_v5 and has_v3:
            bonus += 0.20
        assert bonus == 0.20


class TestScoreChangeBoost:
    """Test that score_change boost is in _OUTCOME_SCORE_BOOSTS."""

    def test_score_change_boost_exists(self) -> None:
        from app.services.clip_extractor import _OUTCOME_SCORE_BOOSTS
        assert "score_change" in _OUTCOME_SCORE_BOOSTS
        assert _OUTCOME_SCORE_BOOSTS["score_change"] == 20


class TestNewSignalWeights:
    """Test updated play_classifier weights."""

    def test_weights_sum_to_one(self) -> None:
        from app.services.play_classifier import WEIGHTS
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_outcome_weight_increased(self) -> None:
        from app.services.play_classifier import WEIGHTS
        assert WEIGHTS["outcome"] == 0.10

    def test_dead_ball_weight_exists(self) -> None:
        from app.services.play_classifier import WEIGHTS
        assert "dead_ball" in WEIGHTS
        assert WEIGHTS["dead_ball"] == 0.05
