# Stat pipeline orchestrator — chains ball_tracker → zone_detector → action_detector → game_stats
# Called from analyze_pipeline.py after clip extraction.
# Wrapped in try/except — if stat generation fails, returns empty stats rather than crashing.

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from app.services.audio_types import AudioAnalysisResult

logger = logging.getLogger(__name__)


def run_stat_pipeline(
    *,
    frames: list[tuple[float, Any]],
    sport: str,
    jersey_number: int = 0,
    position: str | None = None,
    pose_results: dict[float, dict] | None = None,
    motion_scores: dict[float, float] | None = None,
    audio_result: Any | None = None,
    clips: list[dict] | None = None,
) -> dict:
    """Run full stat generation pipeline. Never raises — returns empty on failure."""
    try:
        return _run_stat_pipeline_inner(
            frames=frames,
            sport=sport,
            jersey_number=jersey_number,
            position=position,
            pose_results=pose_results or {},
            motion_scores=motion_scores or {},
            audio_result=audio_result,
            clips=clips or [],
        )
    except Exception as exc:
        logger.error("stat_pipeline: failed: %s", exc)
        return {
            "game_stats": _empty_stats(sport, jersey_number),
            "per_clip_stats": [],
            "actions_detected": [],
        }


def _run_stat_pipeline_inner(
    *,
    frames: list[tuple[float, Any]],
    sport: str,
    jersey_number: int,
    position: str | None,
    pose_results: dict[float, dict],
    motion_scores: dict[float, float],
    audio_result: Any | None,
    clips: list[dict],
) -> dict:
    """Inner pipeline — may raise."""
    from app.services.ball_tracker import BallTracker
    from app.services.zone_detector import ZoneDetector
    from app.services import action_detector, game_stats

    # Step 1: Ball tracking
    ball_tracker = BallTracker()
    ball_frames: dict[float, dict] = {}
    for t, frame in frames:
        bf = ball_tracker.track_frame(frame, sport)
        ball_frames[t] = {
            "ball_visible": bf.ball_visible,
            "ball_position": bf.ball_position,
            "ball_trajectory": bf.ball_trajectory,
            "ball_confidence": bf.ball_confidence,
        }

    # Step 2: Zone detection
    zone_det = ZoneDetector()
    zone_results: dict[float, dict] = {}
    for t, frame in frames:
        zr = zone_det.detect_zone(frame, sport)
        zone_results[t] = {
            "zone": zr.zone,
            "zone_confidence": zr.zone_confidence,
            "method": zr.method,
        }

    # Step 3: Action detection
    sport_lower = sport.lower()
    crowd_energy: dict[float, float] = {}
    if audio_result and hasattr(audio_result, "energy_curve") and audio_result.energy_curve:
        for p in audio_result.energy_curve:
            crowd_energy[p.timestamp] = p.energy

    if sport_lower == "basketball":
        actions = action_detector.detect_actions_basketball(
            pose_results=pose_results,
            ball_frames=ball_frames,
            zone_results=zone_results,
            motion_scores=motion_scores,
        )
    elif sport_lower in ("football", "american_football"):
        actions = action_detector.detect_actions_football(
            pose_results=pose_results,
            ball_frames=ball_frames,
            zone_results=zone_results,
            motion_scores=motion_scores,
            crowd_energy=crowd_energy,
        )
    elif sport_lower == "lacrosse":
        actions = action_detector.detect_actions_lacrosse(
            pose_results=pose_results,
            ball_frames=ball_frames,
            zone_results=zone_results,
            motion_scores=motion_scores,
        )
    else:
        actions = []

    actions_as_dicts = [
        {
            "action_type": a.action_type,
            "confidence": round(a.confidence, 3),
            "timestamp": a.timestamp,
            "zone": a.zone,
            "ball_involved": a.ball_involved,
        }
        for a in actions
    ]

    # Step 4: Game stats aggregation
    if sport_lower == "basketball":
        result = game_stats.generate_basketball_stats(actions_as_dicts, clips, jersey_number)
    elif sport_lower in ("football", "american_football"):
        result = game_stats.generate_football_stats(actions_as_dicts, clips, jersey_number, position)
    elif sport_lower == "lacrosse":
        result = game_stats.generate_lacrosse_stats(actions_as_dicts, clips, jersey_number)
    else:
        result = game_stats.StatResult()

    return {
        "game_stats": result.game_stats,
        "per_clip_stats": result.per_clip_stats,
        "actions_detected": result.actions_detected,
    }


def _empty_stats(sport: str, jersey_number: int = 0) -> dict:
    """Return empty stats skeleton for a sport."""
    base = {
        "sport": sport,
        "player_jersey": jersey_number,
        "confidence": "none",
        "confidence_note": "Stat generation failed — video analysis only",
    }
    return base
