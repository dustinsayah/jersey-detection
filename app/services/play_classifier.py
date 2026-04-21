# Play type classification — rule-based + signal fusion

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.services.play_rules import SPORT_RULES, PlayTypeRule

LOGGER = logging.getLogger(__name__)

# Signal weights for final score computation
# Jersey is primary signal (identifies the player); outcome added for v4 models
WEIGHTS = {
    "jersey": 0.30,
    "motion": 0.20,
    "audio": 0.15,
    "pose": 0.10,
    "outcome": 0.10,
    "tracking": 0.05,
    "crowd": 0.05,
    "dead_ball": 0.05,
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
    audio_confidence: float = 0.0,
    pose_action: str = "standing",
    tracking_continuity: float = 0.0,
    crowd_energy: float = 0.0,
    play_duration: float = 0.0,
    motion_spike: bool = False,
    arm_extension: float = 0.0,
    v4_outcome: str = "",
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

    # v4 outcome override: if v4 model detected a specific play type, prefer it
    if v4_outcome and v4_outcome != "game_action":
        # Look up if v4 outcome matches a known play type
        v4_rule = next(
            (r for r in sorted_rules if r.play_type == v4_outcome),
            None,
        )
        if v4_rule:
            play_type = v4_rule.play_type
            play_label = v4_rule.label
            matched_rule = v4_rule

    # Compute weighted score
    score = _compute_score(
        jersey_confidence=jersey_confidence,
        motion_score=motion_score,
        has_audio_boundary=has_audio_boundary,
        audio_confidence=audio_confidence,
        pose_action=pose_action,
        tracking_continuity=tracking_continuity,
        crowd_energy=crowd_energy,
        v4_outcome=v4_outcome,
        sport=sport,
    )

    # Bonus for high-priority play types
    if matched_rule and matched_rule.priority >= 8:
        score = min(100, score + 10)
    elif matched_rule and matched_rule.priority >= 6:
        score = min(100, score + 5)

    # Jersey confirmation bonus: when jersey is detected with ANY confidence
    # AND there's meaningful motion, the clip shows the player in action.
    # v5 OCR returns ~0.3-0.5 for confirmed numbers — threshold at 0.2 to
    # include all real OCR hits (false positives are filtered by temporal consensus).
    if jersey_confidence >= 0.2 and motion_score > 50:
        score = max(score, 55)
        LOGGER.debug("Jersey confirmation bonus: jersey=%.2f motion=%.1f → floor 55", jersey_confidence, motion_score)

    # Football QB boost: QBs are in every offensive play, so motion/outcome
    # signals are highly reliable even without jersey detection at 360p
    if sport.lower() == "football" and position and position.lower() in ("qb", "quarterback"):
        if v4_outcome in ("pass_play", "qb_scramble"):
            score = min(100, score + 25)
            LOGGER.debug("QB outcome boost +25 for v4_outcome=%s", v4_outcome)
        elif v4_outcome in ("completion", "sack", "touchdown"):
            score = min(100, score + 20)
            LOGGER.debug("QB outcome boost +20 for v4_outcome=%s", v4_outcome)
        if pose_action == "throwing":
            score = min(100, score + 25)
            LOGGER.debug("QB pose boost +25 for throwing")
        elif pose_action == "running":
            score = min(100, score + 15)
            LOGGER.debug("QB pose boost +15 for running")
        elif pose_action == "standing":
            score = min(100, score + 5)
            LOGGER.debug("QB pose boost +5 for standing")
        if crowd_energy > 0.3:
            score = min(100, score + 10)
            LOGGER.debug("QB crowd boost +10 for crowd_energy=%.2f", crowd_energy)

    # Basketball position-aware boosts
    if sport.lower() == "basketball":
        if v4_outcome in ("made_shot", "fast_break"):
            score = min(100, score + 20)
            LOGGER.debug("Basketball outcome boost +20 for v4_outcome=%s", v4_outcome)
        elif v4_outcome in ("drive", "assist"):
            score = min(100, score + 15)
            LOGGER.debug("Basketball outcome boost +15 for v4_outcome=%s", v4_outcome)
        elif v4_outcome in ("rebound",):
            score = min(100, score + 10)
            LOGGER.debug("Basketball outcome boost +10 for v4_outcome=%s", v4_outcome)
        if pose_action == "jumping" and motion_score > 60:
            score = min(100, score + 10)
            LOGGER.debug("Basketball jumping+motion boost +10")

    # Position-appropriate action without OCR: still valuable clip
    # If jersey wasn't detected but high-confidence action matches position role
    if jersey_confidence < 0.1 and motion_score > 40:
        if sport.lower() == "football" and position and position.lower() in ("qb", "quarterback"):
            if v4_outcome in ("pass_play", "qb_scramble", "touchdown", "completion", "sack"):
                score = max(score, 40)  # floor at Decent
                LOGGER.debug("Position-action floor 40 for QB without OCR: v4=%s", v4_outcome)
        elif sport.lower() == "basketball":
            if v4_outcome in ("made_shot", "fast_break", "drive"):
                score = max(score, 40)
                LOGGER.debug("Position-action floor 40 for basketball without OCR: v4=%s", v4_outcome)

    # Grade assignment — calibrated so without v4 outcome models:
    #   Elite: top ~10% (scoring plays or QB highlight)
    #   Strong: next ~10% (clear highlight play)
    #   Decent: middle ~60% (player involved)
    #   Cut: bottom ~20% (not involved / dead ball)
    # Football: moderate thresholds — raised from 45/25 to reduce false positives
    _is_football = sport.lower() == "football"
    _strong_thresh = 50 if _is_football else 55
    _decent_thresh = 35 if _is_football else 35
    if score >= 75:
        grade = "Elite"
    elif score >= _strong_thresh:
        grade = "Strong"
    elif score >= _decent_thresh:
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
            "audio_confidence": round(audio_confidence, 2),
            "pose": pose_action,
            "crowd": round(crowd_energy, 2),
            "v4_outcome": v4_outcome or None,
        },
    )


def _compute_score(
    *,
    jersey_confidence: float,
    motion_score: float,
    has_audio_boundary: bool,
    audio_confidence: float = 0.0,
    pose_action: str,
    tracking_continuity: float,
    crowd_energy: float,
    v4_outcome: str = "",
    sport: str = "basketball",
) -> int:
    """Compute weighted final score from all signals."""
    # Jersey: 0-1 → 0-100
    jersey_component = jersey_confidence * 100

    # Motion: already 0-100
    motion_component = motion_score

    # Audio: use confidence when available, else binary fallback
    if has_audio_boundary and audio_confidence > 0:
        audio_component = 20.0 + (audio_confidence * 60.0)
    elif has_audio_boundary:
        audio_component = 80.0
    else:
        audio_component = 20.0

    # Pose: action intensity mapping
    pose_intensity_map = {
        "jumping": 90,
        "throwing": 85,
        "running": 70,
        "defensive_stance": 60,
        "crouching": 50,
        "standing": 30,
    }
    pose_component = float(pose_intensity_map.get(pose_action, 40))

    # Tracking: 0-1 → 0-100
    tracking_component = tracking_continuity * 100

    # Crowd: 0-1 → 0-100
    crowd_component = crowd_energy * 100

    # Outcome: v4 detector confirmed a play type → 100, else 0
    # This is additive — when v4 models aren't loaded, outcome weight
    # contributes 0 and other signals compensate via their weights
    outcome_component = 100.0 if v4_outcome else 0.0

    # Dead ball: inverted — live play = 100, dead ball = 0
    # When dead ball models aren't loaded, default to 50 (neutral)
    dead_ball_component = 50.0  # neutral default

    # Football OCR compensation: when jersey OCR returns 0 (all football
    # OCR models fail), redistribute jersey weight to motion + audio so
    # football clips can still score in the 50-70 range based on game signals.
    weights = dict(WEIGHTS)
    if sport.lower() == "football" and jersey_confidence < 0.01:
        jersey_weight_to_redistribute = weights["jersey"]  # 0.30
        weights["jersey"] = 0.0
        weights["motion"] += jersey_weight_to_redistribute * 0.50   # +0.15
        weights["audio"] += jersey_weight_to_redistribute * 0.30    # +0.09
        weights["pose"] += jersey_weight_to_redistribute * 0.20     # +0.06

    # Weighted sum
    total = (
        weights["jersey"] * jersey_component
        + weights["motion"] * motion_component
        + weights["audio"] * audio_component
        + weights["pose"] * pose_component
        + weights["tracking"] * tracking_component
        + weights["crowd"] * crowd_component
        + weights["outcome"] * outcome_component
        + weights["dead_ball"] * dead_ball_component
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
