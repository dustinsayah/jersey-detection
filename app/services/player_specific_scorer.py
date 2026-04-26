# Player-specific scorer — combines team color, zoom, and formation signals
# into a unified player_specific_score (0-60 points) added to clip scoring.

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Score weights
TEAM_COLOR_OFFENSE_BONUS = 20   # My team has more players in center (on offense)
ZOOM_CLOSE_UP_BONUS = 15       # Broadcast zoomed in during this clip
ZOOM_TRANSITION_BONUS = 10     # Wide→close_up transition near clip
FORMATION_POSITION_BONUS = 25  # Correct position detected in pre-snap
FORMATION_QB_PRESENT_BONUS = 10  # QB detected in formation (for QB reels)

# Penalty weights
WRONG_TEAM_PENALTY = -15       # Opponent team dominates the clip
NO_FORMATION_DETECTED = 0      # No penalty — formation detection isn't always possible


@dataclass
class PlayerSpecificScore:
    """Breakdown of player-specific scoring for a clip."""
    total: float = 0.0
    team_color_score: float = 0.0
    zoom_score: float = 0.0
    formation_score: float = 0.0
    details: str = ""


def calculate_player_specific_score(
    clip_start: float,
    clip_end: float,
    position: str,
    # Team color data
    my_team_frame_counts: list[tuple[float, int, int]] | None = None,  # [(ts, my_count, opp_count)]
    # Zoom data
    zoom_analysis: object | None = None,  # ZoomAnalysis
    # Formation data
    formation_analysis: object | None = None,  # FormationAnalysis
) -> PlayerSpecificScore:
    """Calculate player-specific score for a clip.

    Args:
        clip_start: clip start timestamp
        clip_end: clip end timestamp
        position: target position (QB, WR, RB, etc.)
        my_team_frame_counts: per-frame team counts near this clip
        zoom_analysis: ZoomAnalysis from zoom_detector
        formation_analysis: FormationAnalysis from formation_detector

    Returns:
        PlayerSpecificScore with total 0-60 range
    """
    result = PlayerSpecificScore()
    details_parts: list[str] = []
    position_upper = (position or "").upper().strip()

    # ── Team Color Score ──────────────────────────────────────────────
    if my_team_frame_counts:
        relevant_frames = [
            (ts, my_c, opp_c) for ts, my_c, opp_c in my_team_frame_counts
            if clip_start - 2.0 <= ts <= clip_end + 2.0
        ]
        if relevant_frames:
            total_my = sum(mc for _, mc, _ in relevant_frames)
            total_opp = sum(oc for _, _, oc in relevant_frames)

            if total_my > total_opp:
                # My team has more players in the frame center = likely on offense
                result.team_color_score = TEAM_COLOR_OFFENSE_BONUS
                details_parts.append(f"team_color:+{TEAM_COLOR_OFFENSE_BONUS}(my={total_my}>opp={total_opp})")
            elif total_opp > total_my * 1.5:
                # Opponent dominates = likely opponent's play
                result.team_color_score = WRONG_TEAM_PENALTY
                details_parts.append(f"team_color:{WRONG_TEAM_PENALTY}(opp dominant)")

    # ── Zoom Score ────────��─────────────────────────────���─────────────
    if zoom_analysis is not None:
        try:
            has_closeup = zoom_analysis.has_close_up_near(zoom_analysis, clip_start, clip_end)  # type: ignore
        except (AttributeError, TypeError):
            # ZoomAnalysis might not have the method — use events directly
            has_closeup = False
            for ts in getattr(zoom_analysis, 'close_up_timestamps', []):
                if clip_start - 1.0 <= ts <= clip_end + 1.0:
                    has_closeup = True
                    break

        if has_closeup:
            result.zoom_score += ZOOM_CLOSE_UP_BONUS
            details_parts.append(f"zoom:+{ZOOM_CLOSE_UP_BONUS}(close_up)")

        has_transition = False
        for ts in getattr(zoom_analysis, 'transition_timestamps', []):
            if clip_start - 2.0 <= ts <= clip_end + 2.0:
                has_transition = True
                break

        if has_transition:
            result.zoom_score += ZOOM_TRANSITION_BONUS
            details_parts.append(f"zoom:+{ZOOM_TRANSITION_BONUS}(transition)")

    # ── Formation Score ───────��───────────────────────────────────────
    if formation_analysis is not None and position_upper:
        # Check if target position was detected near this clip
        target_score = 0.0
        try:
            target_score = formation_analysis.get_target_position_score(  # type: ignore
                formation_analysis, clip_start, clip_end
            )
        except (AttributeError, TypeError):
            # Try direct access
            for r in getattr(formation_analysis, 'results', []):
                if clip_start - 3.0 <= r.timestamp <= clip_start + 2.0:
                    for pos in getattr(r, 'positions', []):
                        if pos.position == position_upper:
                            target_score = max(target_score, pos.confidence)

        if target_score > 0.4:
            bonus = int(FORMATION_POSITION_BONUS * target_score)
            result.formation_score += bonus
            details_parts.append(f"formation:+{bonus}({position_upper} conf={target_score:.2f})")

        # For QB reels: extra bonus if QB detected in formation
        if position_upper == "QB":
            qb_found = False
            for ts in getattr(formation_analysis, 'qb_timestamps', []):
                if clip_start - 4.0 <= ts <= clip_start + 2.0:
                    qb_found = True
                    break
            if qb_found:
                result.formation_score += FORMATION_QB_PRESENT_BONUS
                details_parts.append(f"formation:+{FORMATION_QB_PRESENT_BONUS}(QB in formation)")

    # ── Total ─────��───────────────────────────────────────────────────
    result.total = max(0.0, min(60.0,
        result.team_color_score + result.zoom_score + result.formation_score
    ))
    result.details = ", ".join(details_parts) if details_parts else "no_signals"

    return result
