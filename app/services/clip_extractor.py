# Clip boundary detection — merge all signals into ranked clips
#
# v4 scoring tiers (with outcome detection):
#   Elite (>=90): confirmed outcome — made shot, TD, goal
#   Strong (70-89): highlight play without confirmed outcome
#   Decent (50-69): player involved but unclear outcome
#   Cut (<50): player not involved or dead ball

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
    "completion": 15,       # football_completion_detector_v4
    "sack": 15,             # football_sack_detector_v4
    "rebound": 10,          # basketball_rebound_v4
    "ground_ball": 10,      # lacrosse_ground_ball_v4
    "drive": 10,            # basketball_dribble_drive_v4
    "qb_scramble": 15,      # football_qb_scramble_v4
    "crowd_energy": 10,     # crowd_energy_detector_v4 (all sports)
}

# Clip boundary expansion (seconds)
EXPAND_BEFORE = 2.0
EXPAND_AFTER = 4.0

# Minimum clip duration
MIN_CLIP_DURATION = 3.0
MAX_CLIP_DURATION = 30.0

# Maximum distance between jersey detections to cluster them (seconds)
DETECTION_CLUSTER_GAP = 5.0


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

    # ── Step 1: Find detection clusters ──────────────────────────────────
    sorted_dets = sorted(detections, key=lambda d: d.timestamp)
    clusters: list[list[DetectionPoint]] = []
    current_cluster: list[DetectionPoint] = []

    for det in sorted_dets:
        if not current_cluster:
            current_cluster = [det]
        elif det.timestamp - current_cluster[-1].timestamp <= DETECTION_CLUSTER_GAP:
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

        # Expand boundaries
        start_time = max(0, first_t - EXPAND_BEFORE)
        end_time = min(video_duration, last_t + EXPAND_AFTER) if video_duration > 0 else last_t + EXPAND_AFTER

        # Enforce duration limits
        duration = end_time - start_time
        if duration < MIN_CLIP_DURATION:
            # Expand symmetrically
            needed = MIN_CLIP_DURATION - duration
            start_time = max(0, start_time - needed / 2)
            end_time = start_time + MIN_CLIP_DURATION
        elif duration > MAX_CLIP_DURATION:
            end_time = start_time + MAX_CLIP_DURATION

        play_duration = end_time - start_time

        # Classify play type
        classification = classify_play(
            sport=sport,
            position=position,
            jersey_confidence=avg_jersey_conf,
            motion_score=avg_motion,
            has_audio_boundary=audio_bounded,
            pose_action=dominant_pose,
            tracking_continuity=min(1.0, len(cluster) / 10.0),
            crowd_energy=avg_crowd,
            play_duration=play_duration,
            motion_spike=motion_spike,
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
        if outcome_grade != "Elite":
            if outcome_score >= 90:
                outcome_grade = "Elite"
            elif outcome_score >= 70:
                outcome_grade = "Strong"
            elif outcome_score >= 50:
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

    # ── Step 3: Merge overlapping clips ──────────────────────────────────
    clips.sort(key=lambda c: c.start_time)
    merged: list[ExtractedClip] = []

    for clip in clips:
        if merged and clip.start_time < merged[-1].end_time:
            # Overlap — keep the higher-scoring one
            if clip.score > merged[-1].score:
                # Expand the better clip to cover both
                merged[-1] = ExtractedClip(
                    start_time=min(merged[-1].start_time, clip.start_time),
                    end_time=max(merged[-1].end_time, clip.end_time),
                    confidence=max(merged[-1].confidence, clip.confidence),
                    score=max(merged[-1].score, clip.score),
                    play_type=clip.play_type,
                    play_label=clip.play_label,
                    grade=clip.grade,
                    jersey_visible=merged[-1].jersey_visible or clip.jersey_visible,
                    jersey_number_seen=clip.jersey_number_seen or merged[-1].jersey_number_seen,
                    tracking_id=clip.tracking_id or merged[-1].tracking_id,
                    description=clip.description,
                    signals=clip.signals,
                    detection_count=merged[-1].detection_count + clip.detection_count,
                )
            else:
                # Expand existing clip
                merged[-1].end_time = max(merged[-1].end_time, clip.end_time)
                merged[-1].detection_count += clip.detection_count
        else:
            merged.append(clip)

    # ── Step 4: Sort by score descending ─────────────────────────────────
    merged.sort(key=lambda c: c.score, reverse=True)

    # Filter out "Cut" grade clips
    result = [c for c in merged if c.grade != "Cut"]

    # ── Rescue logic: if ALL clips were "Cut", rescue the best ones ──────
    # Score 20+ covers motion-only fallback clips (no jersey confidence).
    # Returning something is better than returning 0 clips.
    if not result and merged:
        rescued = [c for c in merged if c.score >= 20]
        if rescued:
            for clip in rescued:
                clip.grade = "Decent"
            result = rescued
            LOGGER.info(
                "Rescue: all %d clips were Cut — rescued %d with score >= 20",
                len(merged), len(result),
            )

    LOGGER.info(
        "Extracted %d clips from %d detections (%d clusters, %d after merge/filter)",
        len(result), len(detections), len(clusters), len(result),
    )

    return result
