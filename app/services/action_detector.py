# Action detector — combines pose + ball + zone to classify sport actions
# Basketball: shot, drive, pass, rebound, steal, assist, block, dunk
# Football: pass_play, completion, run_play, touchdown, tackle, sack, reception
# Lacrosse: shot, goal, ground_ball, clear, face_off, save, assist

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectedAction:
    action_type: str = "unknown"
    confidence: float = 0.0
    timestamp: float = 0.0
    zone: str = "unknown"
    ball_involved: bool = False


def detect_actions_basketball(
    *,
    pose_results: dict[float, dict],
    ball_frames: dict[float, dict],
    zone_results: dict[float, dict],
    motion_scores: dict[float, float],
) -> list[DetectedAction]:
    """Classify basketball actions from combined signals."""
    actions: list[DetectedAction] = []

    for ts, pose in pose_results.items():
        action = pose.get("action", "standing")
        intensity = pose.get("intensity", 0.0)
        zone_info = zone_results.get(ts, {})
        zone = zone_info.get("zone", "unknown") if isinstance(zone_info, dict) else "unknown"
        ball = ball_frames.get(ts, {})
        ball_visible = ball.get("ball_visible", False) if isinstance(ball, dict) else False
        motion = motion_scores.get(ts, 0)

        detected = None

        # Shot attempt: jumping + high intensity near basket
        if action in ("jumping", "throwing") and intensity > 0.5:
            if zone in ("paint", "midrange"):
                detected = DetectedAction(
                    action_type="shot_attempt", confidence=0.7,
                    timestamp=ts, zone=zone, ball_involved=ball_visible,
                )
            elif zone == "three_point":
                detected = DetectedAction(
                    action_type="three_point_attempt", confidence=0.65,
                    timestamp=ts, zone=zone, ball_involved=ball_visible,
                )
            elif zone == "paint" and intensity > 0.8:
                detected = DetectedAction(
                    action_type="dunk", confidence=0.5,
                    timestamp=ts, zone=zone, ball_involved=ball_visible,
                )

        # Drive: fast lateral movement toward paint
        elif motion > 50 and zone in ("three_point", "midrange", "transition"):
            detected = DetectedAction(
                action_type="drive", confidence=0.55,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Rebound: jumping near basket after potential shot
        elif action == "jumping" and zone == "paint":
            detected = DetectedAction(
                action_type="rebound", confidence=0.5,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Pass: arm extended
        elif action == "throwing" and intensity < 0.5 and ball_visible:
            detected = DetectedAction(
                action_type="pass", confidence=0.45,
                timestamp=ts, zone=zone, ball_involved=True,
            )

        if detected:
            actions.append(detected)

    return actions


def detect_actions_football(
    *,
    pose_results: dict[float, dict],
    ball_frames: dict[float, dict],
    zone_results: dict[float, dict],
    motion_scores: dict[float, float],
    crowd_energy: dict[float, float] | None = None,
) -> list[DetectedAction]:
    """Classify football actions from combined signals."""
    actions: list[DetectedAction] = []

    for ts, pose in pose_results.items():
        action = pose.get("action", "standing")
        intensity = pose.get("intensity", 0.0)
        zone_info = zone_results.get(ts, {})
        zone = zone_info.get("zone", "unknown") if isinstance(zone_info, dict) else "unknown"
        ball = ball_frames.get(ts, {})
        ball_visible = ball.get("ball_visible", False) if isinstance(ball, dict) else False
        motion = motion_scores.get(ts, 0)
        energy = (crowd_energy or {}).get(ts, 0)

        detected = None

        # Pass play: throwing pose
        if action == "throwing" and intensity > 0.5:
            detected = DetectedAction(
                action_type="pass_play", confidence=0.65,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Touchdown: high energy + near endzone
        elif energy > 0.7 and zone in ("redzone", "opponent_endzone"):
            detected = DetectedAction(
                action_type="touchdown", confidence=0.55,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Run play: high motion, no throwing pose
        elif motion > 60 and action != "throwing":
            detected = DetectedAction(
                action_type="run_play", confidence=0.5,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Tackle: player goes to ground
        elif action in ("falling", "crouching") and motion > 30:
            detected = DetectedAction(
                action_type="tackle", confidence=0.45,
                timestamp=ts, zone=zone, ball_involved=False,
            )

        if detected:
            actions.append(detected)

    return actions


def detect_actions_lacrosse(
    *,
    pose_results: dict[float, dict],
    ball_frames: dict[float, dict],
    zone_results: dict[float, dict],
    motion_scores: dict[float, float],
) -> list[DetectedAction]:
    """Classify lacrosse actions from combined signals."""
    actions: list[DetectedAction] = []

    for ts, pose in pose_results.items():
        action = pose.get("action", "standing")
        intensity = pose.get("intensity", 0.0)
        zone_info = zone_results.get(ts, {})
        zone = zone_info.get("zone", "unknown") if isinstance(zone_info, dict) else "unknown"
        ball = ball_frames.get(ts, {})
        ball_visible = ball.get("ball_visible", False) if isinstance(ball, dict) else False
        motion = motion_scores.get(ts, 0)

        detected = None

        # Shot: arm extension near goal
        if action in ("throwing", "jumping") and zone in ("attack", "crease"):
            detected = DetectedAction(
                action_type="shot", confidence=0.6,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Ground ball: low pose
        elif action == "crouching" and ball_visible:
            detected = DetectedAction(
                action_type="ground_ball", confidence=0.5,
                timestamp=ts, zone=zone, ball_involved=True,
            )

        # Clear: defensive player moving upfield
        elif zone == "defensive" and motion > 40:
            detected = DetectedAction(
                action_type="clear", confidence=0.45,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        # Face off: crouching at midfield
        elif action == "crouching" and zone == "midfield":
            detected = DetectedAction(
                action_type="face_off", confidence=0.4,
                timestamp=ts, zone=zone, ball_involved=ball_visible,
            )

        if detected:
            actions.append(detected)

    return actions
