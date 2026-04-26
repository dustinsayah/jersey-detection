# Clip boundary detection — merge all signals into ranked clips
#
# v7 scoring tiers (calibrated for pre-v4-model reality):
#   Elite (>=75): confirmed outcome or QB highlight play
#   Strong (55-74): highlight play without confirmed outcome
#   Decent (35-54): player involved but unclear outcome
#   Cut (<35): player not involved or dead ball

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.services.audio_types import AudioAnalysisResult, PlayBoundary
from app.services.play_classifier import ClassificationResult, classify_play

LOGGER = logging.getLogger(__name__)

# v4 outcome actions that auto-boost clip grade to Elite
_ELITE_OUTCOMES = {"made_shot", "touchdown", "goal"}
# v4 outcome actions that boost clip score by fixed amount
_OUTCOME_SCORE_BOOSTS = {
    "made_shot": 25,        # basketball_made_shot_v4 + basketball_hoop_detector_v4
    "touchdown": 30,        # football_touchdown_detector_v4
    "goal": 30,             # lacrosse_goal_detector_v4
    "pass_play": 25,        # football — confirmed offensive play, QB involved
    "completion": 20,       # football_completion_detector_v4
    "sack": 25,             # football_sack_detector_v4
    "rebound": 10,          # basketball_rebound_v4
    "ground_ball": 10,      # lacrosse_ground_ball_v4
    "drive": 10,            # basketball_dribble_drive_v4
    "qb_scramble": 25,      # football_qb_scramble_v4 (boosted for QB plays)
    "crowd_energy": 10,     # crowd_energy_detector_v4 (all sports)
    "score_change": 20,     # scoreboard_detector_v5 (score change detected)
    "reception_yac": 20,    # football_reception_yac_v4
}

# Clip boundary expansion — sport-specific defaults (seconds)
_SPORT_EXPAND = {
    "basketball": (1.5, 3.0),  # Fast game, tighter clips
    "football": (2.0, 6.0),    # 2s pre-snap + 6s post-play follow-through
    "lacrosse": (1.5, 3.5),    # Fast transitions
}
_DEFAULT_EXPAND = (2.0, 4.0)

# Minimum clip duration — football needs longer clips to capture full plays
MIN_CLIP_DURATION = 3.0
FOOTBALL_MIN_CLIP_DURATION = 8.0
MAX_CLIP_DURATION = 30.0
# Shorter clips for full-game analysis (>1800s videos)
FULL_GAME_MAX_CLIP_DURATION = 20.0  # Allow longer plays to capture full action

# Maximum distance between jersey detections to cluster them (seconds)
# Basketball: 3s gap — fast-paced, plays change quickly
# Football: 2.5s gap — each play is a distinct burst (snap → whistle)
# Default: 4s for other sports
DETECTION_CLUSTER_GAP = 3.0
FOOTBALL_CLUSTER_GAP = 2.5


@dataclass
class DetectionPoint:
    """A single frame-level detection from any signal source."""
    timestamp: float
    confidence: float = 0.0
    jersey_visible: bool = False
    jersey_number: int | None = None
    motion_score: float = 0.0
    pose_action: str = "standing"
    crowd_energy: float = 0.0
    tracking_id: int | None = None
    v4_outcome: str = ""  # v4 outcome detector result (made_shot, touchdown, goal, etc.)


@dataclass
class ExtractedClip:
    start_time: float
    end_time: float
    confidence: float = 0.0
    score: int = 0
    play_type: str = "game_action"
    play_label: str = "Game Action"
    grade: str = "Decent"
    jersey_visible: bool = False
    jersey_number_seen: int | None = None
    tracking_id: int | None = None
    description: str = ""
    signals: dict = field(default_factory=dict)
    detection_count: int = 0


def extract_clips(
    detections: list[DetectionPoint],
    audio_result: AudioAnalysisResult | None = None,
    sport: str = "basketball",
    position: str | None = None,
    video_duration: float = 0.0,
) -> list[ExtractedClip]:
    """Merge all detection signals into ranked, scored clips.

    Algorithm:
    1. Use audio play boundaries as primary segmentation (if available)
    2. Find jersey detection clusters
    3. Score each segment
    4. Expand boundaries
    5. Merge overlapping clips
    6. Grade and rank
    """
    if not detections:
        return []

    clips: list[ExtractedClip] = []

    # Get audio play boundaries
    audio_boundaries: list[PlayBoundary] = []
    if audio_result and audio_result.play_boundaries:
        audio_boundaries = audio_result.play_boundaries

    # Build energy lookup for quick access
    energy_lookup: dict[int, float] = {}
    if audio_result and audio_result.energy_curve:
        for pt in audio_result.energy_curve:
            second = int(pt.timestamp)
            energy_lookup[second] = max(energy_lookup.get(second, 0), pt.energy)

    # ── Step 0: Jitter filter — remove isolated 1-frame false positives ──
    # Full-game mode: wider neighbor window because frames are 8+ seconds
    # apart at 1fps.  Also lower confidence bypass (v5 OCR averages 0.4-0.45).
    _is_full_game = video_duration > 1800
    _jitter_window = 10.0 if _is_full_game else 2.0
    _jitter_conf_bypass = 0.3 if _is_full_game else 0.5
    sorted_dets = sorted(detections, key=lambda d: d.timestamp)
    if len(sorted_dets) >= 3:
        filtered_dets: list[DetectionPoint] = []
        for det in sorted_dets:
            neighbors = sum(
                1 for d in sorted_dets
                if d is not det and abs(d.timestamp - det.timestamp) <= _jitter_window
            )
            if neighbors >= 1 or det.confidence > _jitter_conf_bypass:
                filtered_dets.append(det)
        if filtered_dets:
            jitter_removed = len(sorted_dets) - len(filtered_dets)
            if jitter_removed > 0:
                LOGGER.info("Jitter filter: removed %d isolated detections (window=%.0fs, conf_bypass=%.2f)",
                            jitter_removed, _jitter_window, _jitter_conf_bypass)
            sorted_dets = filtered_dets

    # ── Step 1: Find detection clusters ──────────────────────────────────
    clusters: list[list[DetectionPoint]] = []
    current_cluster: list[DetectionPoint] = []

    # Full-game mode: wider cluster gap because frames are 1fps (8-10s apart)
    if _is_full_game:
        cluster_gap = 8.0  # Wider gap for sparse frames
    elif sport.lower() == "football":
        cluster_gap = FOOTBALL_CLUSTER_GAP
    else:
        cluster_gap = DETECTION_CLUSTER_GAP
    for det in sorted_dets:
        if not current_cluster:
            current_cluster = [det]
        elif det.timestamp - current_cluster[-1].timestamp <= cluster_gap:
            current_cluster.append(det)
        else:
            clusters.append(current_cluster)
            current_cluster = [det]

    if current_cluster:
        clusters.append(current_cluster)

    # ── Step 2: Create clips from clusters ───────────────────────────────
    for cluster in clusters:
        first_t = cluster[0].timestamp
        last_t = cluster[-1].timestamp

        # Check if cluster falls within an audio boundary
        audio_bounded = False
        audio_confidence = 0.0
        for boundary in audio_boundaries:
            if boundary.start_time - 3 <= first_t and last_t <= boundary.end_time + 3:
                audio_bounded = True
                audio_confidence = boundary.confidence
                # Use audio boundary for clip edges if tighter
                first_t = min(first_t, boundary.start_time)
                last_t = max(last_t, boundary.end_time)
                break

        # Aggregate signals from all detections in cluster
        jersey_confs = [d.confidence for d in cluster if d.jersey_visible]
        avg_jersey_conf = sum(jersey_confs) / len(jersey_confs) if jersey_confs else 0.0

        motion_scores = [d.motion_score for d in cluster if d.motion_score > 0]
        avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0

        # Motion spike detection
        motion_spike = False
        if len(motion_scores) >= 2:
            max_motion = max(motion_scores)
            min_motion = min(motion_scores)
            if max_motion > 60 and max_motion - min_motion > 30:
                motion_spike = True

        pose_actions = [d.pose_action for d in cluster if d.pose_action != "standing"]
        dominant_pose = max(set(pose_actions), key=pose_actions.count) if pose_actions else "standing"

        # Crowd energy at cluster time
        cluster_seconds = range(int(first_t), int(last_t) + 1)
        crowd_energies = [energy_lookup.get(s, 0.0) for s in cluster_seconds]
        avg_crowd = sum(crowd_energies) / len(crowd_energies) if crowd_energies else 0.0

        # Jersey info
        jersey_numbers = [d.jersey_number for d in cluster if d.jersey_number is not None]
        jersey_number_seen = max(set(jersey_numbers), key=jersey_numbers.count) if jersey_numbers else None

        tracking_ids = [d.tracking_id for d in cluster if d.tracking_id is not None]
        tracking_id = max(set(tracking_ids), key=tracking_ids.count) if tracking_ids else None

        # Tracking continuity: use actual player tracking persistence
        if tracking_ids and tracking_id is not None:
            continuous_frames = sum(1 for tid in tracking_ids if tid == tracking_id)
            tracking_continuity = min(1.0, continuous_frames / 20.0)
        else:
            tracking_continuity = min(1.0, len(cluster) / 10.0)

        # v4 outcome: aggregate from cluster
        v4_outcomes = [d.v4_outcome for d in cluster if d.v4_outcome]
        dominant_v4 = max(set(v4_outcomes), key=v4_outcomes.count) if v4_outcomes else ""

        # Sport-aware boundary expansion
        expand_before, expand_after = _SPORT_EXPAND.get(sport.lower(), _DEFAULT_EXPAND)
        start_time = max(0, first_t - expand_before)
        end_time = min(video_duration, last_t + expand_after) if video_duration > 0 else last_t + expand_after

        # Enforce duration limits — shorter clips for full games
        _max_dur = FULL_GAME_MAX_CLIP_DURATION if _is_full_game else MAX_CLIP_DURATION
        _min_dur = FOOTBALL_MIN_CLIP_DURATION if sport.lower() == "football" else MIN_CLIP_DURATION
        duration = end_time - start_time
        if duration < _min_dur:
            # Expand symmetrically to meet minimum
            needed = _min_dur - duration
            start_time = max(0, start_time - needed / 2)
            end_time = start_time + _min_dur
        elif duration > _max_dur:
            end_time = start_time + _max_dur

        play_duration = end_time - start_time

        # Classify play type
        classification = classify_play(
            sport=sport,
            position=position,
            jersey_confidence=avg_jersey_conf,
            motion_score=avg_motion,
            has_audio_boundary=audio_bounded,
            audio_confidence=audio_confidence,
            pose_action=dominant_pose,
            tracking_continuity=tracking_continuity,
            crowd_energy=avg_crowd,
            play_duration=play_duration,
            motion_spike=motion_spike,
            v4_outcome=dominant_v4,
        )

        # ── v4: Apply outcome detection boosts ─────────────────────────
        v4_outcomes = [d.v4_outcome for d in cluster if d.v4_outcome]
        outcome_score = classification.score
        outcome_grade = classification.grade

        if v4_outcomes:
            # Count unique outcomes in this cluster
            unique_outcomes = set(v4_outcomes)
            for outcome in unique_outcomes:
                boost = _OUTCOME_SCORE_BOOSTS.get(outcome, 0)
                outcome_score = min(100, outcome_score + boost)
            # Elite override: confirmed scoring plays
            if unique_outcomes & _ELITE_OUTCOMES:
                outcome_grade = "Elite"
                outcome_score = max(outcome_score, 90)

        # Re-grade based on v4-boosted score
        # Football: lower thresholds (45/25) because jersey OCR rarely works
        # at 360p, so clips rely on motion/audio/pose signals which score lower
        _is_football = sport.lower() == "football"
        _strong_threshold = 45 if _is_football else 55
        _decent_threshold = 25 if _is_football else 35
        if outcome_grade != "Elite":
            if outcome_score >= 75:
                outcome_grade = "Elite"
            elif outcome_score >= _strong_threshold:
                outcome_grade = "Strong"
            elif outcome_score >= _decent_threshold:
                outcome_grade = "Decent"
            else:
                outcome_grade = "Cut"

        clips.append(ExtractedClip(
            start_time=round(start_time, 1),
            end_time=round(end_time, 1),
            confidence=round(avg_jersey_conf, 3),
            score=outcome_score,
            play_type=classification.play_type,
            play_label=classification.play_label,
            grade=outcome_grade,
            jersey_visible=bool(jersey_confs),
            jersey_number_seen=jersey_number_seen,
            tracking_id=tracking_id,
            description=classification.play_label,
            signals=classification.signals or {},
            detection_count=len(cluster),
        ))

    # ── Step 2.5: Touchdown sanity check ─────────────────────────────────
    # Max 2 touchdowns per 60 seconds of footage — more than that is a
    # false positive from the rule classifier
    _SCORING_PLAY_TYPES = {"touchdown", "made_shot", "goal"}
    scoring_clips = [c for c in clips if c.play_type in _SCORING_PLAY_TYPES]
    if len(scoring_clips) > 2:
        # Check time span
        time_span = max(c.end_time for c in scoring_clips) - min(c.start_time for c in scoring_clips)
        max_scoring_per_60 = max(2, int(time_span / 60) * 2) if time_span > 0 else 2
        if len(scoring_clips) > max_scoring_per_60:
            LOGGER.warning(
                "Scoring sanity check: %d '%s' clips in %.0fs — capping at %d, "
                "demoting rest to game_action",
                len(scoring_clips), scoring_clips[0].play_type, time_span, max_scoring_per_60,
            )
            # Keep the top N by score, demote the rest
            scoring_clips.sort(key=lambda c: c.score, reverse=True)
            for clip in scoring_clips[max_scoring_per_60:]:
                clip.play_type = "game_action"
                clip.play_label = "Game Action"
                clip.description = "Game Action"

    # ── Step 3: Merge overlapping AND adjacent clips ─────────────────────
    # Merge clips that overlap OR are within 2s of each other with similar
    # motion scores (±15). This prevents consecutive 5s windows with
    # identical motion scores from appearing as separate clips.
    PROXIMITY_GAP = 2.0
    MOTION_SIMILARITY = 15.0
    clips.sort(key=lambda c: c.start_time)
    merged: list[ExtractedClip] = []

    for clip in clips:
        if merged:
            prev = merged[-1]
            gap = clip.start_time - prev.end_time
            overlaps = gap < 0
            # Adjacent: within PROXIMITY_GAP and similar motion
            prev_motion = prev.signals.get("motion", 0) if prev.signals else 0
            clip_motion = clip.signals.get("motion", 0) if clip.signals else 0
            adjacent_similar = (
                0 <= gap <= PROXIMITY_GAP
                and abs(prev_motion - clip_motion) <= MOTION_SIMILARITY
            )

            if overlaps or adjacent_similar:
                # Merge — keep the higher-scoring one's metadata
                if clip.score > prev.score:
                    merged[-1] = ExtractedClip(
                        start_time=min(prev.start_time, clip.start_time),
                        end_time=max(prev.end_time, clip.end_time),
                        confidence=max(prev.confidence, clip.confidence),
                        score=max(prev.score, clip.score),
                        play_type=clip.play_type,
                        play_label=clip.play_label,
                        grade=clip.grade,
                        jersey_visible=prev.jersey_visible or clip.jersey_visible,
                        jersey_number_seen=clip.jersey_number_seen or prev.jersey_number_seen,
                        tracking_id=clip.tracking_id or prev.tracking_id,
                        description=clip.description,
                        signals=clip.signals,
                        detection_count=prev.detection_count + clip.detection_count,
                    )
                else:
                    merged[-1].end_time = max(prev.end_time, clip.end_time)
                    merged[-1].detection_count += clip.detection_count
                    merged[-1].jersey_visible = prev.jersey_visible or clip.jersey_visible
                continue

        merged.append(clip)

    # ── Step 3.5: Quality gate — require at least one meaningful signal ──
    # Clips with no jersey, no v4 outcome, low motion, and no audio event
    # are likely noise from dead ball frames or camera pans.
    # Special teams (kickoff/punt) are always filtered for QB/skill positions.
    _SPECIAL_TEAMS = {"kickoff", "punt", "kickoff_return", "punt_return",
                       "field_goal", "extra_point", "special_teams"}
    # QB/skill positions never play special teams — hard filter ALL of them
    _is_qb = position and position.lower() in ("qb", "quarterback")
    _is_skill = position and position.lower() in (
        "qb", "quarterback", "wr", "wide receiver", "rb", "running back",
        "te", "tight end",
    )
    gated: list[ExtractedClip] = []
    for clip in merged:
        # Hard filter: QBs don't play special teams — remove ALL kickoffs/punts
        if clip.play_type in _SPECIAL_TEAMS:
            if _is_skill or not clip.jersey_visible:
                LOGGER.info(
                    "Quality gate: dropped special teams clip %.1f-%.1f (%s, position=%s)",
                    clip.start_time, clip.end_time, clip.play_type, position,
                )
                continue

        has_jersey = clip.jersey_visible
        has_outcome = bool(clip.signals.get("v4_outcome"))
        clip_motion = clip.signals.get("motion", 0) or 0
        has_audio = bool(clip.signals.get("audio"))
        # Full-game football at 360p: motion scores are 0-10 (wide-angle JV),
        # cadence supplement is the primary clip source — lower thresholds
        _is_football = sport.lower() == "football"
        if _is_full_game and _is_football:
            has_high_motion = clip_motion > 3
            motion_solo_threshold = 8
        else:
            has_high_motion = clip_motion > 50
            motion_solo_threshold = 65
        # Require jersey OR (high motion + audio) OR outcome for quality
        if has_jersey or has_outcome or (has_high_motion and has_audio) or (clip_motion > motion_solo_threshold):
            gated.append(clip)
        else:
            LOGGER.debug(
                "Quality gate: dropped clip %.1f-%.1f (score=%d, no meaningful signal)",
                clip.start_time, clip.end_time, clip.score,
            )
    if gated:
        merged = gated
    else:
        LOGGER.info("Quality gate: all clips filtered — keeping all (fallback)")

    # ── Step 3.6: Late-game dead ball filter (full-game football) ──────
    # Camera pans during late-game timeouts/huddles (>95 min) generate
    # moderate motion with no confirming signal.  Only applied to clips
    # deep in the game where dead ball frequency is highest.
    # Safety valve: never remove more than 20 % of clips.
    if _is_full_game and sport.lower() == "football":
        _pre_dead = len(merged)
        _dead_filtered: list[ExtractedClip] = []
        for clip in merged:
            _cm = clip.signals.get("motion", 0) or 0
            _hj = clip.jersey_visible
            _ho = bool(clip.signals.get("v4_outcome"))
            _ha = bool(clip.signals.get("audio"))
            _late_game = clip.start_time > 5700  # >95 minutes
            # Late-game dead ball: no confirming signal + moderate motion
            if _late_game and not _hj and not _ho and not _ha and _cm < 65:
                LOGGER.info(
                    "Dead ball filter: dropped late-game clip %.1f-%.1f "
                    "(motion=%.0f, no jersey/outcome/audio, t>95min)",
                    clip.start_time, clip.end_time, _cm,
                )
                continue
            _dead_filtered.append(clip)
        # Safety valve: keep at least 80 % of clips
        if _dead_filtered and len(_dead_filtered) >= _pre_dead * 0.8:
            merged = _dead_filtered
            if _pre_dead != len(merged):
                LOGGER.info("Dead ball filter: %d → %d clips", _pre_dead, len(merged))
        elif _pre_dead != len(_dead_filtered):
            LOGGER.info("Dead ball filter: would remove >20%% — skipping")

    # ── Step 4: Sort by score descending ─────────────────────────────────
    merged.sort(key=lambda c: c.score, reverse=True)

    # Filter out "Cut" grade clips
    # Full-game football: very low threshold — cadence clips score low but are real plays
    _is_football_fg = _is_full_game and sport.lower() == "football"
    if _is_football_fg:
        # At 360p, cadence clips score very low — take everything, cap at 40
        result = list(merged)
    elif _is_full_game:
        result = [c for c in merged if c.score >= 20]
    else:
        result = [c for c in merged if c.grade != "Cut"]

    # ── Rescue logic: if ALL clips were "Cut", rescue the best ones ──────
    # Football: rescue aggressively — jersey OCR rarely works on football footage
    rescue_threshold = 5 if sport.lower() == "football" else 25
    if not result and merged:
        rescued = [c for c in merged if c.score >= rescue_threshold]
        if rescued:
            for clip in rescued:
                clip.grade = "Decent"
            result = rescued
            LOGGER.info(
                "Rescue: all %d clips were Cut — rescued %d with score >= %d (sport=%s)",
                len(merged), len(result), rescue_threshold, sport,
            )

    # ── Cap: full-game mode — keep top 40 clips to avoid overwhelming the UI ──
    _MAX_FULL_GAME_CLIPS = 40
    if _is_full_game and len(result) > _MAX_FULL_GAME_CLIPS:
        LOGGER.info("Full game clip cap: %d → %d clips", len(result), _MAX_FULL_GAME_CLIPS)
        result = result[:_MAX_FULL_GAME_CLIPS]

    LOGGER.info(
        "Extracted %d clips from %d detections (%d clusters, %d after merge/filter)",
        len(result), len(detections), len(clusters), len(result),
    )

    return result
