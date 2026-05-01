"""Position-specific clip scoring rubrics.

The original ``recruitingScore`` in bytetrack_pipeline mixes signals (jersey
OCR, motion, player count) without any concept of position-specific value. A
QB highlight reel should reward different things than a LB reel:

  QB:  accurate throws, full-field reads, pocket presence, scrambles.
       Penalize: sacks, INTs, plays where target jersey is never visible.
  RB:  long runs, broken tackles, red-zone TDs, contact runs.
  WR:  contested catches, YAC, deep balls, big-play 3rd downs.
  LB:  sacks, TFLs, INTs, pass break-ups, sideline-to-sideline tackles.

This module bakes the **research findings about what college coaches want**
into per-position scoring functions. The signals available right now are:

  - jersey OCR confidence (target jersey visible)
  - per-frame target-team count (offense on field)
  - real-frame motion (action vs static)
  - clip duration (longer = more action visible)
  - QB position confidence from QBDetector (NEW)
  - QB-in-frame ratio (QB tracked through the play, NEW)
  - team_majority (already filters opponent plays)

What we DON'T have yet (and the rubric documents the gap):
  - throw detection
  - completion vs incompletion
  - red-zone vs midfield (need field registration)
  - touchdown detection

The scorer is honest about what it can and can't measure. Where a signal is
missing it weights heavily on what we DO have. When the X3D-S action model
ships (Phase 5), its per-clip score plugs in as an additional dimension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Per-clip scoring breakdown."""
    final_score: float = 0.0          # 0-100
    grade: str = "Cut"                # Strong / Decent / Cut
    components: dict = field(default_factory=dict)  # name -> contribution
    reasons_kept: list[str] = field(default_factory=list)
    reasons_cut: list[str] = field(default_factory=list)
    position: str = ""

    def as_dict(self) -> dict:
        return {
            "finalScore": round(self.final_score, 1),
            "grade": self.grade,
            "components": {k: round(v, 1) for k, v in self.components.items()},
            "reasonsKept": self.reasons_kept,
            "reasonsCut": self.reasons_cut,
            "scoredFor": self.position,
        }


# ---------------------------------------------------------------------------
# QB scoring rubric — derived directly from research synthesis
# ---------------------------------------------------------------------------
#
# Coaches want (in order):
#   1. Accurate downfield throws to varied route concepts
#   2. Full-field reads / progressions
#   3. Throws into tight windows / over the middle
#   4. Pocket presence under pressure
#   5. Designed runs / scrambles for first downs
#   6. Play-action / boot completions
#   7. Touch passes between zones
#   8. Deep-ball arm strength
#   9. Red-zone / 3rd-down conversions
#   10. Pre-snap QB ID
#
# Of these, what we can MEASURE today (Phase 4):
#   - #10 (pre-snap QB ID) → QBDetector + qb_in_frame ratio
#   - partial #4 / #5 (action) → motion intensity + duration
#   - #1-#3, #6-#9 → require throw/completion detection (Phase 5+)
#
# Therefore the v8.34 scorer weights the available signals heavily and
# documents which dimensions are stubbed. As Phase 5 lands we add weights.

QB_WEIGHTS = {
    # Direct evidence the QB was on the field (we identified the position)
    "qb_position_identified": 25.0,    # QBDetector found a QB at all
    "qb_position_confidence": 15.0,    # how confident we are about QB id
    "qb_in_frame_ratio": 20.0,         # fraction of clip frames where QB was tracked
    "jersey_confirmed": 20.0,          # jersey OCR actually read target #
    "jersey_persistence": 10.0,        # multi-frame jersey reads (not single-frame fluke)
    # Action quality signals (proxy for "real play happened")
    "motion_intensity": 10.0,          # higher motion = more action
    "play_duration": 5.0,              # longer = more chance of meaningful action
    "play_state_consistency": 5.0,     # frames consistently classified as "play"
    # Anti-signals (penalize)
    "opponent_majority_ratio": -30.0,  # heavily penalize opponent-on-offense moments
    "static_clip": -15.0,              # cluster with no motion at all = dead ball/huddle
}

QB_GRADE_THRESHOLDS = {
    "Strong": 65.0,
    "Decent": 35.0,
}


def score_qb_clip(
    clip_meta: dict,
    moments: list[dict],
    formation_per_moment: list[dict] | None = None,
) -> ScoringResult:
    """Score a clip on QB highlight quality (0-100).

    Args:
      clip_meta: the clip dict produced by _finalize_clip (has playerCount,
                 momentCount, jerseyConfidence, motionScore, realMotionAvg, etc)
      moments:   the list of per-frame moments that made up this clip
      formation_per_moment: list of FormationFrame.as_dict() for each moment,
                            same length as moments, OR None if QBDetector wasn't
                            run yet. When None, qb_position_* components are 0.

    Returns: ScoringResult with breakdown.
    """
    result = ScoringResult(position="quarterback")
    components: dict[str, float] = {}

    # ------- QB position evidence (NEW signals from QBDetector) -------
    if formation_per_moment:
        qb_ids = [f.get("qb_track_id") for f in formation_per_moment if f.get("qb_track_id") is not None]
        qb_confs = [f.get("qb_confidence", 0.0) for f in formation_per_moment if f.get("qb_track_id") is not None]
        if qb_ids:
            # qb_position_identified: max if any frame found a QB
            components["qb_position_identified"] = QB_WEIGHTS["qb_position_identified"]
            # qb_position_confidence: scaled by mean confidence of QB picks
            mean_conf = sum(qb_confs) / max(1, len(qb_confs))
            components["qb_position_confidence"] = QB_WEIGHTS["qb_position_confidence"] * mean_conf
            result.reasons_kept.append(
                f"QB identified in {len(qb_ids)}/{len(formation_per_moment)} frames (mean conf {mean_conf:.2f})"
            )

            # qb_in_frame_ratio: fraction of frames where the most-voted QB id was present
            # (Already computed implicitly: if qb_track_id is present, QB is in frame)
            ratio = len(qb_ids) / max(1, len(formation_per_moment))
            components["qb_in_frame_ratio"] = QB_WEIGHTS["qb_in_frame_ratio"] * ratio
            if ratio < 0.4:
                result.reasons_cut.append(f"QB in only {ratio:.0%} of frames (camera lost him)")
        else:
            components["qb_position_identified"] = 0.0
            components["qb_position_confidence"] = 0.0
            components["qb_in_frame_ratio"] = 0.0
            result.reasons_cut.append("no QB found in any frame")
    else:
        # QBDetector data not provided — can't score these
        components["qb_position_identified"] = 0.0
        components["qb_position_confidence"] = 0.0
        components["qb_in_frame_ratio"] = 0.0

    # ------- Jersey OCR signals -------
    jconf = float(clip_meta.get("jerseyConfidence", 0))
    jersey_visible_frames = int(clip_meta.get("targetVisibleFrames", 0))
    moment_count = max(1, len(moments))

    if jconf >= 0.5:
        components["jersey_confirmed"] = QB_WEIGHTS["jersey_confirmed"]
        result.reasons_kept.append(f"jersey #{clip_meta.get('jerseyNumberSeen','?')} read confidently ({jconf:.2f})")
    elif jconf >= 0.2:
        components["jersey_confirmed"] = QB_WEIGHTS["jersey_confirmed"] * 0.5
        result.reasons_kept.append(f"jersey partially visible (jconf {jconf:.2f})")
    else:
        components["jersey_confirmed"] = 0.0

    persistence_ratio = jersey_visible_frames / moment_count
    if persistence_ratio >= 0.3:
        components["jersey_persistence"] = QB_WEIGHTS["jersey_persistence"]
        result.reasons_kept.append(f"jersey visible in {persistence_ratio:.0%} of frames")
    elif persistence_ratio >= 0.1:
        components["jersey_persistence"] = QB_WEIGHTS["jersey_persistence"] * 0.5
    else:
        components["jersey_persistence"] = 0.0

    # ------- Action / motion signals -------
    rm_avg = float(clip_meta.get("realMotionAvg", 0))
    # rm_avg ranges roughly 0-15 on this footage; >5 is meaningful action
    if rm_avg >= 8:
        components["motion_intensity"] = QB_WEIGHTS["motion_intensity"]
        result.reasons_kept.append(f"high motion ({rm_avg:.1f})")
    elif rm_avg >= 5:
        components["motion_intensity"] = QB_WEIGHTS["motion_intensity"] * 0.7
    elif rm_avg >= 3:
        components["motion_intensity"] = QB_WEIGHTS["motion_intensity"] * 0.4
    else:
        components["motion_intensity"] = 0.0
        if rm_avg < 1:
            components["static_clip"] = QB_WEIGHTS["static_clip"]
            result.reasons_cut.append(f"static clip (motion {rm_avg:.1f})")

    duration = float(clip_meta.get("endTime", 0)) - float(clip_meta.get("startTime", 0))
    if 8 <= duration <= 20:
        components["play_duration"] = QB_WEIGHTS["play_duration"]
    elif duration > 20:
        components["play_duration"] = QB_WEIGHTS["play_duration"] * 0.6
    else:
        components["play_duration"] = QB_WEIGHTS["play_duration"] * (duration / 8.0)

    # play_state_consistency: how many of the moments were "play" (not transition)
    play_state_count = sum(1 for m in moments if m.get("state") == "play")
    consistency = play_state_count / moment_count
    components["play_state_consistency"] = QB_WEIGHTS["play_state_consistency"] * consistency

    # ------- Anti-signals -------
    opp_count = sum(1 for m in moments if m.get("team_majority") == "opponent")
    opp_ratio = opp_count / moment_count
    if opp_ratio >= 0.3:
        components["opponent_majority_ratio"] = QB_WEIGHTS["opponent_majority_ratio"] * opp_ratio
        result.reasons_cut.append(f"opponent on offense in {opp_ratio:.0%} of frames")
    else:
        components["opponent_majority_ratio"] = 0.0

    # ------- Sum -------
    final = sum(components.values())
    final = max(0.0, min(100.0, final))

    if final >= QB_GRADE_THRESHOLDS["Strong"]:
        grade = "Strong"
    elif final >= QB_GRADE_THRESHOLDS["Decent"]:
        grade = "Decent"
    else:
        grade = "Cut"

    result.final_score = final
    result.grade = grade
    result.components = components
    return result


# ---------------------------------------------------------------------------
# Future per-position rubrics — stubs ready for Phase 5+
# ---------------------------------------------------------------------------

def score_clip(
    position: str,
    clip_meta: dict,
    moments: list[dict],
    formation_per_moment: list[dict] | None = None,
) -> ScoringResult:
    """Dispatch to position-specific scorer."""
    pos = (position or "").lower().strip()
    if pos == "quarterback":
        return score_qb_clip(clip_meta, moments, formation_per_moment)
    # TODO Phase 5+: RB, WR, LB rubrics
    # For now non-QB positions get the QB rubric without QB-position bonus
    LOGGER.info("position_scorer: %s rubric not yet implemented, using QB without position bonus", pos)
    return score_qb_clip(clip_meta, moments, formation_per_moment=None)
