# Play type classification — rule-based + signal fusion

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.play_rules import SPORT_RULES, PlayTypeRule

LOGGER = logging.getLogger(__name__)

# Signal weights for final score computation
WEIGHTS = {
    "jersey": 0.30,
    "motion": 0.20,
    "audio": 0.20,
    "pose": 0.15,
    "tracking": 0.10,
    "crowd": 0.05,
}


@dataclass
class ClassificationResult:
    play_type: str = "game_action"
    play_label: str = "Game Action"
    score: int = 50
    grade: str = "Decent"
    signals: dict = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = {}


def classify_play(
    sport: str,
    position: str | None = None,
    *,
    jersey_confidence: float = 0.0,
    motion_score: float = 0.0,
    has_audio_boundary: bool = False,
    pose_action: str = "standing",
    tracking_continuity: float = 0.0,
    crowd_energy: float = 0.0,
    play_duration: float = 0.0,
    motion_spike: bool = False,
    arm_extension: float = 0.0,
) -> ClassificationResult:
    """Classify a segment using signal fusion + sport-specific rules.

    Returns:
        ClassificationResult with play_type, score, and grade
    """
    rules = SPORT_RULES.get(sport.lower(), SPORT_RULES.get("basketball", []))

    # Sort by priority (highest first)
    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    # Try each rule in priority order
    matched_rule: PlayTypeRule | None = None
    for rule in sorted_rules:
        if _rule_matches(
            rule,
            position=position,
            motion_score=motion_score,
            has_audio_boundary=has_audio_boundary,
            pose_action=pose_action,
            crowd_energy=crowd_energy,
            play_duration=play_duration,
            motion_spike=motion_spike,
            arm_extension=arm_extension,
        ):
            matched_rule = rule
            break

    play_type = matched_rule.play_type if matched_rule else "game_action"
    play_label = matched_rule.label if matched_rule else "Game Action"

    # Compute weighted score
    score = _compute_score(
        jersey_confidence=jersey_confidence,
        motion_score=motion_score,
        has_audio_boundary=has_audio_boundary,
        pose_action=pose_action,
        tracking_continuity=tracking_continuity,
        crowd_energy=crowd_energy,
    )

    # Bonus for high-priority play types
    if matched_rule and matched_rule.priority >= 8:
        score = min(100, score + 10)
    elif matched_rule and matched_rule.priority >= 6:
        score = min(100, score + 5)

    # Grade assignment
    if score >= 90:
        grade = "Elite"
    elif score >= 70:
        grade = "Strong"
    elif score >= 50:
        grade = "Decent"
    else:
        grade = "Cut"

    return ClassificationResult(
        play_type=play_type,
        play_label=play_label,
        score=score,
        grade=grade,
        signals={
            "jersey": round(jersey_confidence, 2),
            "motion": round(motion_score, 1),
            "audio": has_audio_boundary,
            "pose": pose_action,
            "crowd": round(crowd_energy, 2),
        },
    )


def _compute_score(
    *,
    jersey_confidence: float,
    motion_score: float,
    has_audio_boundary: bool,
    pose_action: str,
    tracking_continuity: float,
    crowd_energy: float,
) -> int:
    """Compute weighted final score from all signals."""
    # Jersey: 0-1 → 0-100
    jersey_component = jersey_confidence * 100

    # Motion: already 0-100
    motion_component = motion_score

    # Audio: binary → 0 or 80
    audio_component = 80.0 if has_audio_boundary else 20.0

    # Pose: action intensity mapping
    pose_intensity_map = {
        "jumping": 90,
        "throwing": 85,
        "running": 70,
        "crouching": 50,
        "standing": 30,
    }
    pose_component = float(pose_intensity_map.get(pose_action, 40))

    # Tracking: 0-1 → 0-100
    tracking_component = tracking_continuity * 100

    # Crowd: 0-1 → 0-100
    crowd_component = crowd_energy * 100

    # Weighted sum
    total = (
        WEIGHTS["jersey"] * jersey_component
        + WEIGHTS["motion"] * motion_component
        + WEIGHTS["audio"] * audio_component
        + WEIGHTS["pose"] * pose_component
        + WEIGHTS["tracking"] * tracking_component
        + WEIGHTS["crowd"] * crowd_component
    )

    return min(100, max(0, round(total)))


def _rule_matches(
    rule: PlayTypeRule,
    *,
    position: str | None,
    motion_score: float,
    has_audio_boundary: bool,
    pose_action: str,
    crowd_energy: float,
    play_duration: float,
    motion_spike: bool,
    arm_extension: float,
) -> bool:
    """Check if a signal set matches a rule's conditions."""
    conds = rule.conditions

    if "motion_min" in conds and motion_score < conds["motion_min"]:
        return False

    if "motion_max" in conds and motion_score > conds["motion_max"]:
        return False

    if "crowd_energy_min" in conds and crowd_energy < conds["crowd_energy_min"]:
        return False

    if "whistle_bounded" in conds and conds["whistle_bounded"] and not has_audio_boundary:
        return False

    if "pose_any" in conds and pose_action not in conds["pose_any"]:
        return False

    if "play_duration_range" in conds:
        lo, hi = conds["play_duration_range"]
        if play_duration > 0 and not (lo <= play_duration <= hi):
            return False

    if "motion_spike" in conds and conds["motion_spike"] and not motion_spike:
        return False

    if "position" in conds and position and position.lower() != conds["position"].lower():
        return False

    if "arm_extension_min" in conds and arm_extension < conds["arm_extension_min"]:
        return False

    return True
