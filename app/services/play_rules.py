# Sport-specific play type classification rules

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayTypeRule:
    play_type: str
    label: str  # Human-readable name
    conditions: dict  # Signal conditions
    priority: int = 0  # Higher = checked first


# ── Football rules ───────────────────────────────────────────────────────────

FOOTBALL_RULES: list[PlayTypeRule] = [
    PlayTypeRule(
        play_type="touchdown",
        label="Touchdown",
        conditions={
            "crowd_energy_min": 0.8,
            "motion_min": 70,
            "pose_any": ["running", "jumping"],
        },
        priority=10,
    ),
    PlayTypeRule(
        play_type="pass_play",
        label="Pass Play",
        conditions={
            "pose_any": ["throwing"],
            "whistle_bounded": True,
            "play_duration_range": (3, 15),
        },
        priority=8,
    ),
    PlayTypeRule(
        play_type="run_play",
        label="Run Play",
        conditions={
            "pose_any": ["running"],
            "motion_min": 60,
            "whistle_bounded": True,
        },
        priority=7,
    ),
    PlayTypeRule(
        play_type="sack",
        label="Sack",
        conditions={
            "pose_any": ["crouching", "throwing"],
            "motion_spike": True,
            "play_duration_range": (2, 8),
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="scramble",
        label="QB Scramble",
        conditions={
            "pose_any": ["running"],
            "motion_min": 65,
            "position": "quarterback",
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="big_gain",
        label="Big Gain",
        conditions={
            "motion_min": 70,
            "play_duration_range": (4, 20),
            "crowd_energy_min": 0.5,
        },
        priority=5,
    ),
    PlayTypeRule(
        play_type="kickoff",
        label="Kickoff",
        conditions={
            "motion_min": 60,
            "play_duration_range": (5, 30),
            # Kickoffs have high sustained motion, long duration, wide camera
            # Distinguish from pass/run by NOT requiring throwing/whistle
        },
        priority=3,  # Below pass/run/sack so real plays win
    ),
    PlayTypeRule(
        play_type="punt",
        label="Punt",
        conditions={
            "motion_min": 50,
            "play_duration_range": (3, 20),
        },
        priority=3,
    ),
    PlayTypeRule(
        play_type="formation",
        label="Pre-Snap Formation",
        conditions={
            "motion_max": 30,
            "pose_any": ["standing", "crouching"],
        },
        priority=2,
    ),
    PlayTypeRule(
        play_type="game_action",
        label="Game Action",
        conditions={
            "whistle_bounded": True,
        },
        priority=1,
    ),
]

# ── Basketball rules ─────────────────────────────────────────────────────────

BASKETBALL_RULES: list[PlayTypeRule] = [
    PlayTypeRule(
        play_type="dunk",
        label="Dunk",
        conditions={
            "pose_any": ["jumping"],
            "motion_min": 75,
            "crowd_energy_min": 0.7,
        },
        priority=10,
    ),
    PlayTypeRule(
        play_type="fast_break",
        label="Fast Break",
        conditions={
            "motion_min": 75,
            "pose_any": ["running"],
            "play_duration_range": (3, 10),
        },
        priority=9,
    ),
    PlayTypeRule(
        play_type="shot",
        label="Shot Attempt",
        conditions={
            "pose_any": ["jumping", "throwing"],
            "crowd_energy_min": 0.3,
        },
        priority=7,
    ),
    PlayTypeRule(
        play_type="drive",
        label="Drive to Basket",
        conditions={
            "pose_any": ["running"],
            "motion_min": 60,
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="steal",
        label="Steal",
        conditions={
            "motion_spike": True,
            "play_duration_range": (1, 5),
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="rebound",
        label="Rebound",
        conditions={
            "pose_any": ["jumping"],
            "motion_min": 40,
        },
        priority=5,
    ),
    PlayTypeRule(
        play_type="defense",
        label="Defensive Play",
        conditions={
            "pose_any": ["crouching", "standing"],
            "motion_min": 30,
            "motion_max": 60,
        },
        priority=3,
    ),
    PlayTypeRule(
        play_type="half_court_set",
        label="Half Court Set",
        conditions={
            "motion_max": 40,
        },
        priority=2,
    ),
]

# ── Lacrosse rules ───────────────────────────────────────────────────────────

LACROSSE_RULES: list[PlayTypeRule] = [
    PlayTypeRule(
        play_type="goal",
        label="Goal",
        conditions={
            "crowd_energy_min": 0.7,
            "pose_any": ["throwing"],
            "motion_min": 60,
        },
        priority=10,
    ),
    PlayTypeRule(
        play_type="shot",
        label="Shot on Goal",
        conditions={
            "pose_any": ["throwing"],
            "arm_extension_min": 0.5,
        },
        priority=8,
    ),
    PlayTypeRule(
        play_type="dodge",
        label="Dodge",
        conditions={
            "pose_any": ["running"],
            "motion_min": 65,
            "motion_spike": True,
        },
        priority=7,
    ),
    PlayTypeRule(
        play_type="ground_ball",
        label="Ground Ball",
        conditions={
            "pose_any": ["crouching"],
            "motion_min": 40,
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="faceoff",
        label="Face-Off",
        conditions={
            "pose_any": ["crouching"],
            "whistle_bounded": True,
            "play_duration_range": (1, 5),
        },
        priority=6,
    ),
    PlayTypeRule(
        play_type="clear",
        label="Clear",
        conditions={
            "motion_min": 55,
            "pose_any": ["running"],
            "play_duration_range": (5, 20),
        },
        priority=5,
    ),
    PlayTypeRule(
        play_type="transition",
        label="Transition",
        conditions={
            "motion_min": 50,
            "pose_any": ["running"],
        },
        priority=4,
    ),
    PlayTypeRule(
        play_type="check",
        label="Body Check",
        conditions={
            "motion_spike": True,
            "pose_any": ["standing", "running"],
        },
        priority=4,
    ),
    PlayTypeRule(
        play_type="settled_offense",
        label="Settled Offense",
        conditions={
            "motion_max": 50,
        },
        priority=2,
    ),
]

SPORT_RULES: dict[str, list[PlayTypeRule]] = {
    "football": FOOTBALL_RULES,
    "basketball": BASKETBALL_RULES,
    "lacrosse": LACROSSE_RULES,
}
