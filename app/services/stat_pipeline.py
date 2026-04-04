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

    # Step 1: Ball tracking (requires frames — skip if freed)
    ball_frames: dict[float, dict] = {}
    if frames:
        ball_tracker = BallTracker()
        for t, frame in frames:
            bf = ball_tracker.track_frame(frame, sport)
            ball_frames[t] = {
                "ball_visible": bf.ball_visible,
                "ball_position": bf.ball_position,
                "ball_trajectory": bf.ball_trajectory,
                "ball_confidence": bf.ball_confidence,
            }
    else:
        logger.info("stat_pipeline: frames freed — estimating ball presence from motion")
        # Estimate ball involvement from motion scores (high motion → likely ball in play)
        for t, m in motion_scores.items():
            ball_frames[t] = {
                "ball_visible": m > 20,  # moderate motion → ball likely visible
                "ball_position": None,
                "ball_trajectory": None,
                "ball_confidence": min(m / 100, 0.7),
            }

    # Step 2: Zone detection (requires frames — skip if freed)
    zone_results: dict[float, dict] = {}
    if frames:
        zone_det = ZoneDetector()
        for t, frame in frames:
            zr = zone_det.detect_zone(frame, sport)
            zone_results[t] = {
                "zone": zr.zone,
                "zone_confidence": zr.zone_confidence,
                "method": zr.method,
            }
    else:
        logger.info("stat_pipeline: frames freed — zones unavailable, using 'unknown'")
        for t in pose_results:
            zone_results[t] = {"zone": "unknown", "zone_confidence": 0.0, "method": "unavailable"}

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

    # Step 3b: Clip-based action estimation when pose-based detection is sparse
    # Each clip represents a detected highlight — estimate at least one action per clip
    if len(actions_as_dicts) < len(clips) and clips:
        logger.info(
            "stat_pipeline: %d actions for %d clips — adding clip-based estimates",
            len(actions_as_dicts), len(clips),
        )
        clip_timestamps_covered = {
            round(a["timestamp"], 1) for a in actions_as_dicts
        }
        for clip in clips:
            clip_mid = (clip.get("startTime", 0) + clip.get("endTime", 0)) / 2
            # Skip if we already have an action near this clip
            if any(abs(clip_mid - t) < 3.0 for t in clip_timestamps_covered):
                continue
            grade = clip.get("grade", "Decent")
            motion_at_clip = min(
                motion_scores.items(),
                key=lambda kv: abs(kv[0] - clip_mid),
                default=(clip_mid, 40),
            )[1] if motion_scores else 40

            estimated = _estimate_action_from_clip(
                sport_lower, grade, motion_at_clip, clip_mid,
            )
            if estimated:
                actions_as_dicts.append(estimated)
                clip_timestamps_covered.add(round(clip_mid, 1))

    logger.info("stat_pipeline: total actions for stats: %d", len(actions_as_dicts))

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


def _estimate_action_from_clip(
    sport: str, grade: str, motion: float, timestamp: float,
) -> dict | None:
    """Estimate a plausible action from clip metadata when pose detection is sparse."""
    conf = 0.35 if grade in ("Strong", "Excellent") else 0.25

    if sport == "basketball":
        if motion > 60:
            return {"action_type": "drive", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
        elif motion > 30:
            return {"action_type": "shot_attempt", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
        else:
            return {"action_type": "rebound", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
    elif sport in ("football", "american_football"):
        if motion > 50:
            return {"action_type": "run_play", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
        else:
            return {"action_type": "pass_play", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
    elif sport == "lacrosse":
        if motion > 40:
            return {"action_type": "shot", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
        else:
            return {"action_type": "ground_ball", "confidence": conf, "timestamp": timestamp,
                    "zone": "unknown", "ball_involved": True}
    return None


def _empty_stats(sport: str, jersey_number: int = 0) -> dict:
    """Return empty stats skeleton for a sport."""
    base = {
        "sport": sport,
        "player_jersey": jersey_number,
        "confidence": "none",
        "confidence_note": "Stat generation failed — video analysis only",
    }
    return base
