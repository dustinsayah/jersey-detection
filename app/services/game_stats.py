# Game stats aggregator — builds box-score-style stats from detected actions
# Returns game_stats (aggregate) + per_clip_stats (per clip breakdown)

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClipStat:
    clip_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    actions: list[str] = field(default_factory=list)
    zone: str = "unknown"
    grade: str = "Decent"
    stat_contribution: str = ""


@dataclass
class StatResult:
    game_stats: dict[str, Any] = field(default_factory=dict)
    per_clip_stats: list[dict] = field(default_factory=list)
    actions_detected: list[dict] = field(default_factory=list)


def generate_basketball_stats(
    actions: list[dict],
    clips: list[dict],
    jersey_number: int = 0,
) -> StatResult:
    """Aggregate basketball actions into box-score stats."""
    stats = {
        "sport": "basketball",
        "player_jersey": jersey_number,
        "points_estimated": 0,
        "shot_attempts": 0,
        "shots_made": 0,
        "three_point_attempts": 0,
        "three_pointers_made": 0,
        "dunks": 0,
        "assists": 0,
        "rebounds": 0,
        "steals": 0,
        "blocks": 0,
        "turnovers": 0,
        "drives": 0,
        "field_goal_percentage": 0.0,
        "three_point_percentage": 0.0,
        "zone_breakdown": {"paint": 0, "midrange": 0, "three_point": 0, "backcourt": 0},
        "confidence": "medium",
        "confidence_note": "Stats estimated from video analysis — may not be 100% accurate",
    }

    for a in actions:
        at = a.get("action_type", "")
        zone = a.get("zone", "unknown")
        if zone in stats["zone_breakdown"]:
            stats["zone_breakdown"][zone] += 1

        if at == "shot_attempt":
            stats["shot_attempts"] += 1
            # Estimate ~45% make rate
            if a.get("confidence", 0) > 0.6:
                stats["shots_made"] += 1
                stats["points_estimated"] += 2
        elif at == "three_point_attempt":
            stats["three_point_attempts"] += 1
            stats["shot_attempts"] += 1
            if a.get("confidence", 0) > 0.65:
                stats["three_pointers_made"] += 1
                stats["shots_made"] += 1
                stats["points_estimated"] += 3
        elif at == "dunk":
            stats["dunks"] += 1
            stats["shots_made"] += 1
            stats["shot_attempts"] += 1
            stats["points_estimated"] += 2
        elif at == "drive":
            stats["drives"] += 1
        elif at == "rebound":
            stats["rebounds"] += 1
        elif at == "pass":
            pass  # Counted separately if followed by made shot
        elif at == "steal":
            stats["steals"] += 1
        elif at == "block":
            stats["blocks"] += 1
        elif at == "assist":
            stats["assists"] += 1

    if stats["shot_attempts"] > 0:
        stats["field_goal_percentage"] = round(stats["shots_made"] / stats["shot_attempts"] * 100, 1)
    if stats["three_point_attempts"] > 0:
        stats["three_point_percentage"] = round(
            stats["three_pointers_made"] / stats["three_point_attempts"] * 100, 1
        )

    per_clip = _build_per_clip_stats(actions, clips)

    return StatResult(
        game_stats=stats,
        per_clip_stats=[_clip_stat_to_dict(cs) for cs in per_clip],
        actions_detected=[a for a in actions],
    )


def generate_football_stats(
    actions: list[dict],
    clips: list[dict],
    jersey_number: int = 0,
    position: str | None = None,
) -> StatResult:
    """Aggregate football actions into box-score stats."""
    stats = {
        "sport": "football",
        "player_jersey": jersey_number,
        "position_detected": position or "unknown",
        "pass_attempts": 0,
        "completions": 0,
        "completion_percentage": 0.0,
        "touchdowns": 0,
        "interceptions": 0,
        "sacks_taken": 0,
        "run_plays": 0,
        "total_plays": 0,
        "redzone_plays": 0,
        "zone_breakdown": {"own_territory": 0, "midfield": 0, "opponent_territory": 0, "redzone": 0},
        "confidence": "medium",
        "confidence_note": "Stats estimated from video analysis — may not be 100% accurate",
    }

    for a in actions:
        at = a.get("action_type", "")
        zone = a.get("zone", "unknown")
        stats["total_plays"] += 1

        if zone in stats["zone_breakdown"]:
            stats["zone_breakdown"][zone] += 1
        if zone in ("redzone", "opponent_endzone"):
            stats["redzone_plays"] += 1

        if at == "pass_play":
            stats["pass_attempts"] += 1
        elif at == "completion":
            stats["completions"] += 1
        elif at == "touchdown":
            stats["touchdowns"] += 1
        elif at == "run_play":
            stats["run_plays"] += 1
        elif at == "sack":
            stats["sacks_taken"] += 1
        elif at == "interception":
            stats["interceptions"] += 1
        elif at == "tackle":
            pass

    if stats["pass_attempts"] > 0:
        stats["completion_percentage"] = round(
            stats["completions"] / stats["pass_attempts"] * 100, 1
        )

    per_clip = _build_per_clip_stats(actions, clips)

    return StatResult(
        game_stats=stats,
        per_clip_stats=[_clip_stat_to_dict(cs) for cs in per_clip],
        actions_detected=[a for a in actions],
    )


def generate_lacrosse_stats(
    actions: list[dict],
    clips: list[dict],
    jersey_number: int = 0,
) -> StatResult:
    """Aggregate lacrosse actions into box-score stats."""
    stats = {
        "sport": "lacrosse",
        "player_jersey": jersey_number,
        "shot_attempts": 0,
        "goals": 0,
        "shooting_percentage": 0.0,
        "ground_balls": 0,
        "clears": 0,
        "assists": 0,
        "saves_forced": 0,
        "zone_breakdown": {"attack": 0, "midfield": 0, "defensive": 0},
        "confidence": "medium",
        "confidence_note": "Stats estimated from video analysis — may not be 100% accurate",
    }

    for a in actions:
        at = a.get("action_type", "")
        zone = a.get("zone", "unknown")
        if zone in stats["zone_breakdown"]:
            stats["zone_breakdown"][zone] += 1

        if at == "shot":
            stats["shot_attempts"] += 1
        elif at == "goal":
            stats["goals"] += 1
            stats["shot_attempts"] += 1
        elif at == "ground_ball":
            stats["ground_balls"] += 1
        elif at == "clear":
            stats["clears"] += 1
        elif at == "assist":
            stats["assists"] += 1
        elif at == "save":
            stats["saves_forced"] += 1

    if stats["shot_attempts"] > 0:
        stats["shooting_percentage"] = round(
            stats["goals"] / stats["shot_attempts"] * 100, 1
        )

    per_clip = _build_per_clip_stats(actions, clips)

    return StatResult(
        game_stats=stats,
        per_clip_stats=[_clip_stat_to_dict(cs) for cs in per_clip],
        actions_detected=[a for a in actions],
    )


def _build_per_clip_stats(
    actions: list[dict], clips: list[dict]
) -> list[ClipStat]:
    """Map actions to clips based on timestamp overlap."""
    per_clip: list[ClipStat] = []
    for i, clip in enumerate(clips):
        cs = ClipStat(
            clip_index=i,
            start_time=clip.get("startTime", clip.get("start_time", 0)),
            end_time=clip.get("endTime", clip.get("end_time", 0)),
        )
        clip_actions = []
        for a in actions:
            ts = a.get("timestamp", 0)
            if cs.start_time <= ts <= cs.end_time:
                clip_actions.append(a.get("action_type", "unknown"))
                if cs.zone == "unknown":
                    cs.zone = a.get("zone", "unknown")

        cs.actions = list(set(clip_actions))
        cs.grade = clip.get("grade", "Decent")
        cs.stat_contribution = " + ".join(cs.actions) if cs.actions else "game action"
        per_clip.append(cs)
    return per_clip


def _clip_stat_to_dict(cs: ClipStat) -> dict:
    return {
        "clip_index": cs.clip_index,
        "start_time": cs.start_time,
        "end_time": cs.end_time,
        "actions": cs.actions,
        "zone": cs.zone,
        "grade": cs.grade,
        "stat_contribution": cs.stat_contribution,
    }
