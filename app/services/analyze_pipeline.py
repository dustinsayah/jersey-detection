# Analyze pipeline orchestrator — chains all detection layers
#
# DETECTION CALL CHAIN (v5 PRIMARY, Ali LAST RESORT):
# POST /analyze (app/routes/analyze.py)
#   → run_analyze_pipeline() [this file]
#     Step 1: Acquire video (YouTube download or direct URL download)
#     Step 2: DEFERRED — Ali runs LAST, only if needed
#     Step 3: Frame extraction
#     Step 3.5: Dead ball filtering (skip dead frames before OCR)
#     Step 4: Motion scoring (optical flow)
#     Step 5: Audio analysis (whistle + crowd energy)
#     Step 6: Player tracking (BoT-SORT)
#     Step 7: Pose estimation (YOLO11n-pose)
#     Step 7.5a: v5 OCR + v5 player detector (PRIMARY)
#     Step 7.5b: v3 OCR pipeline (secondary confirmation)
#     Step 7.5c: v2 universal OCR (tertiary — 0.995 mAP50)
#     Step 7.5d: v2 sport-specific + v1 legacy (fallback)
#     Step 7.5e: Ali ensemble (LAST RESORT — only if <3 detections from above)
#     Step 7.6: v4 outcome detection + v5 outcome classification
#     Step 7.8: Scoreboard detection (score change → clip boost)
#     Step 8: Cross-layer validation + merge + temporal consensus
#     Step 9: Stat pipeline

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.audio_types import AudioAnalysisResult
from app.services.clip_extractor import DetectionPoint, ExtractedClip, extract_clips
from app.services.motion_scorer import compute_motion_score
from app.services.play_classifier import classify_play
from functools import partial

from starlette.concurrency import run_in_threadpool

from app.services.youtube_proxy import (
    DownloadResult,
    download_youtube_sync,
    extract_audio,
    get_video_duration,
    get_video_resolution,
    is_youtube_url,
)

LOGGER = logging.getLogger(__name__)

# Football-specific overrides
FOOTBALL_CONF_THRESHOLD = 0.08  # Lower for better recall on 720p crops
FOOTBALL_MIN_CLIP = 3.0
FOOTBALL_MAX_CLIP = 12.0

# Frame sampling: how many FPS to extract for Roboflow layers.
# Default 2 = 1 frame per 0.5s. Railway OOM kills at ~2.3GB;
# each 1280×720 frame ≈ 2.7MB, so 200 frames ≈ 540MB.
# Configurable via ANALYZE_FPS env var.
ANALYZE_FPS = int(os.getenv("ANALYZE_FPS", "2"))

# Hard cap on total frames — 0 means use adaptive cap from _get_adaptive_fps()
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))

# ── Blocker 3: Request semaphore — only 1 concurrent analyze request ──
import asyncio as _asyncio
_REQUEST_SEMAPHORE = _asyncio.Semaphore(1)
# Memory threshold (MB) — if RSS exceeds this before a request, force full cleanup
_MEMORY_THRESHOLD_MB = 5000


def _get_adaptive_fps(video_duration: float, sport: str = "basketball") -> tuple[int, int]:
    """Return (fps, max_frames) based on video duration.

    Strategy — more frames for longer videos to avoid missing plays:
      - Short clips (<120s): 2 fps, 150 frames → full coverage
      - Medium clips (120-600s): 1 fps, 200 frames → every second for ~3 min
      - Long videos (600-1800s): 1 fps, 300 frames → one per second, 5 min
      - Full games (1800-3600s): 1 fps, 600 frames → every ~3-6s over 30-60 min
      - Long games (3600-7200s): 1 fps, 750 frames → every ~5-10s over 1-2 hrs
      - Extra long (>7200s): 1 fps, 900 frames → every ~8s+ over 2+ hrs
    """
    if video_duration <= 120:
        return 2, 150
    elif video_duration <= 600:
        return 1, 200
    elif video_duration <= 1800:
        return 1, 300
    elif video_duration <= 3600:
        return 1, 600
    elif video_duration <= 7200:
        return 1, 750
    else:
        return 1, 900


def _generate_clip_caption(
    clip_dict: dict[str, Any],
    sport: str,
    position: str | None,
    jersey_number: int,
) -> str:
    """Generate a coach-friendly caption for a clip.

    Format: "QB #2 — Pass Play — High Motion — Crowd Reaction"
    """
    parts: list[str] = []

    # Player identifier
    pos_label = (position or "").upper()
    if pos_label and clip_dict.get("jerseyNumberSeen"):
        parts.append(f"{pos_label} #{clip_dict['jerseyNumberSeen']}")
    elif clip_dict.get("jerseyNumberSeen"):
        parts.append(f"#{clip_dict['jerseyNumberSeen']}")
    elif pos_label:
        parts.append(pos_label)

    # Play type — prefer v4 outcome over generic playType
    v4 = clip_dict.get("v4Outcome") or (clip_dict.get("signals") or {}).get("v4_outcome", "")
    play_label = clip_dict.get("description") or clip_dict.get("playType", "Game Action")
    if v4 and v4 != "game_action":
        parts.append(v4.replace("_", " ").title())
    else:
        parts.append(play_label.replace("_", " ").title())

    # Rule-based play type supplement (adds detail v4 models may miss)
    signals = clip_dict.get("signals") or {}
    pose = signals.get("pose", "standing")
    motion = signals.get("motion", 0) or 0
    crowd = signals.get("crowd", 0) or 0
    audio = signals.get("audio")

    if pose == "throwing" and motion > 50 and "throw" not in (v4 or "").lower() and "pass" not in (v4 or "").lower():
        parts.append("Pass Attempt")
    elif pose == "running" and motion > 70 and "scramble" not in (v4 or "").lower():
        parts.append("Scramble")

    # Motion level
    if motion > 70:
        parts.append("High Motion")
    elif motion > 40:
        parts.append("Active")

    # Crowd energy
    if crowd > 0.5:
        parts.append("Crowd Reaction")
    elif crowd > 0.3:
        parts.append("Crowd Energy")

    # Audio context
    if audio and "whistle" in str(audio).lower():
        parts.append("Whistle")

    return " — ".join(parts) if parts else "Game Action"


def _compute_recruiting_score(
    clip_dict: dict[str, Any],
    sport: str,
    position: str | None,
) -> int:
    """Compute a 0-100 recruiting score for how impressive a clip looks to a college coach.

    Uses the raw clip score (0-100 from clip_extractor) as the PRIMARY base,
    then applies modifiers. This ensures clips with different underlying quality
    get different recruiting scores even when they share the same grade.

    Positive modifiers (add):
      +20  jerseyVisible AND jerseyNumberSeen matches target
      +15  touchdown / made_shot / goal detected
      +10  crowd energy > 0.7
      +10  pose = throwing or jumping (athletic action)
      +5   audio whistle (end of play confirmed)
      +5   completion / sack / qb_scramble / reception_yac

    Negative modifiers (subtract):
      -10  jerseyVisible is false (can't confirm player identity)
      -15  playType = formation (pre-snap, not action)
      -20  deadBallRatio > 0.5 (mostly dead ball footage)
      -5   playType = dead_ball
    """
    # Use raw clip score as primary base (already 0-100 from clip_extractor)
    # Scale it to 0-60 range to leave room for modifiers to push up to 100
    raw_score = clip_dict.get("score", 50)
    score = int(raw_score * 0.6)  # 0-60 base from clip quality

    # ── Positive modifiers ──

    # Jersey visibility + number match
    if clip_dict.get("jerseyVisible") and clip_dict.get("jerseyNumberSeen"):
        score += 20
    elif clip_dict.get("jerseyVisible"):
        score += 10

    # v4 outcome boosts
    v4 = clip_dict.get("v4Outcome") or (clip_dict.get("signals") or {}).get("v4_outcome", "")
    if v4:
        if v4 in ("touchdown", "made_shot", "goal"):
            score += 15
        elif v4 in ("completion", "sack", "qb_scramble", "reception_yac"):
            score += 5

    # Crowd energy (from signals)
    crowd = (clip_dict.get("signals") or {}).get("crowd", 0) or 0
    if crowd > 0.7:
        score += 10
    elif crowd > 0.5:
        score += 5

    # Pose action — athletic actions boost
    pose = (clip_dict.get("signals") or {}).get("pose", "standing")
    if pose in ("throwing", "jumping"):
        score += 10

    # Audio whistle — confirms end of play
    audio = (clip_dict.get("signals") or {}).get("audio")
    if audio and "whistle" in str(audio).lower():
        score += 5

    # ── Negative modifiers ──

    # No jersey visible — can't confirm player identity
    if not clip_dict.get("jerseyVisible"):
        score -= 10

    # Formation / pre-snap — not actual game action
    play_type = clip_dict.get("playType", "game_action")
    if play_type == "formation":
        score -= 15
    elif play_type == "dead_ball":
        score -= 5

    # Dead ball ratio — mostly dead ball footage in clip
    dead_ball_ratio = clip_dict.get("deadBallRatio", 0) or 0
    if dead_ball_ratio > 0.5:
        score -= 20
    elif dead_ball_ratio > 0.3:
        score -= 10

    return max(0, min(100, score))


# ── Highlight reel ordering priority ──────────────────────────────────
# Maps play_type and v4_outcome to a coach-priority order.
# Lower number = shown first in the reel. Touchdowns first, etc.
_HIGHLIGHT_PRIORITY: dict[str, int] = {
    # Scoring plays — coaches want to see these first
    "touchdown": 1,
    "made_shot": 1,
    "goal": 1,
    # Big throws / completions
    "pass_play": 2,
    "completion": 2,
    "reception_yac": 2,
    # QB scrambles / athletic plays
    "qb_scramble": 3,
    "sack": 3,
    "drive": 3,
    "fast_break": 3,
    # Standard plays
    "game_action": 4,
    "rebound": 4,
    "ground_ball": 4,
    # Low-priority
    "crowd_energy": 5,
    "dead_ball": 6,
}


def _get_highlight_sort_key(clip_dict: dict[str, Any]) -> tuple[int, int]:
    """Return (priority_tier, -recruitingScore) for coach-friendly sort order.

    Primary sort: play type priority (TDs first, then big throws, etc.)
    Secondary sort: recruiting score descending within each tier.
    """
    v4 = clip_dict.get("v4Outcome") or (clip_dict.get("signals") or {}).get("v4_outcome", "")
    play_type = clip_dict.get("playType", "game_action")
    # Use v4 outcome if available, else fall back to playType
    key = v4 if v4 else play_type
    tier = _HIGHLIGHT_PRIORITY.get(key, 4)
    return (tier, -clip_dict.get("recruitingScore", 0))


def _estimate_game_quarter(
    clip_start_time: float,
    video_duration: float,
    sport: str,
) -> str:
    """Estimate which quarter/half a clip falls in based on video timestamp.

    Football: ~3 hours total → 4 quarters of ~45 min each
    Basketball: ~2 hours total → 4 quarters of ~30 min each
    Lacrosse: ~2 hours total → 4 quarters of ~30 min each
    """
    if video_duration <= 0:
        return "unknown"

    fraction = clip_start_time / video_duration
    sport_lower = sport.lower()

    if sport_lower == "football":
        # Football: 4 quarters, halftime at ~50% of video
        if fraction < 0.25:
            return "1st Quarter"
        elif fraction < 0.50:
            return "2nd Quarter"
        elif fraction < 0.55:
            return "Halftime"
        elif fraction < 0.78:
            return "3rd Quarter"
        else:
            return "4th Quarter"
    elif sport_lower == "basketball":
        if fraction < 0.25:
            return "1st Quarter"
        elif fraction < 0.50:
            return "2nd Quarter"
        elif fraction < 0.75:
            return "3rd Quarter"
        else:
            return "4th Quarter"
    elif sport_lower == "lacrosse":
        if fraction < 0.25:
            return "1st Quarter"
        elif fraction < 0.50:
            return "2nd Quarter"
        elif fraction < 0.75:
            return "3rd Quarter"
        else:
            return "4th Quarter"
    return "unknown"


def _generate_sequence_note(
    clip_dict: dict[str, Any],
    clip_index: int,
    total_clips: int,
    prev_clip: dict[str, Any] | None,
    sport: str,
) -> str:
    """Generate play context / drive sequence note for a clip.

    Provides coaches with context about where this clip sits in the game flow.
    """
    parts: list[str] = []

    quarter = clip_dict.get("estimatedQuarter", "")
    if quarter and quarter != "unknown":
        parts.append(quarter)

    play_type = clip_dict.get("playType", "game_action")
    v4 = clip_dict.get("v4Outcome", "")

    # Sequence position
    if clip_index == 0:
        parts.append("Opening highlight")
    elif clip_index == total_clips - 1:
        parts.append("Final highlight")

    # Time gap from previous clip — indicates drive continuity
    if prev_clip:
        gap = clip_dict.get("startTime", 0) - prev_clip.get("endTime", 0)
        if gap < 15:
            parts.append("Same drive" if sport.lower() == "football" else "Same possession")
        elif gap < 60:
            parts.append("Next series")

    # Play description
    action = v4 or play_type
    action_label = action.replace("_", " ").title()
    if action_label and action_label not in ("Game Action",):
        parts.append(action_label)

    # Jersey context
    if clip_dict.get("jerseyVisible") and clip_dict.get("jerseyNumberSeen"):
        parts.append(f"#{clip_dict['jerseyNumberSeen']} confirmed")

    return " — ".join(parts) if parts else "Game Action"


def _build_player_summary(
    clips_out: list[dict[str, Any]],
    jersey_number: int,
    sport: str,
    position: str | None,
    video_duration: float,
    elapsed: float,
) -> dict[str, Any]:
    """Build aggregate player stats summary from all detected clips.

    Returns a dict with stats coaches care about: total clips, jersey detection
    rate, top plays, total highlight time, average recruiting score, etc.
    """
    total = len(clips_out)
    jersey_clips = [c for c in clips_out if c.get("jerseyVisible")]
    scoring_plays = [c for c in clips_out if c.get("v4Outcome") in ("touchdown", "made_shot", "goal")]
    recruiting_scores = [c.get("recruitingScore", 0) for c in clips_out]

    # Play type breakdown
    play_types: dict[str, int] = {}
    for c in clips_out:
        pt = c.get("v4Outcome") or c.get("playType", "game_action")
        play_types[pt] = play_types.get(pt, 0) + 1

    # Total highlight seconds
    total_highlight_secs = sum(
        c.get("endTime", 0) - c.get("startTime", 0) for c in clips_out
    )

    # Grade breakdown
    grades: dict[str, int] = {}
    for c in clips_out:
        g = c.get("grade", "Decent")
        grades[g] = grades.get(g, 0) + 1

    return {
        "jerseyNumber": jersey_number,
        "sport": sport,
        "position": position,
        "totalClips": total,
        "jerseyDetectionRate": round(len(jersey_clips) / total * 100, 1) if total else 0,
        "scoringPlays": len(scoring_plays),
        "avgRecruitingScore": round(sum(recruiting_scores) / total, 1) if total else 0,
        "topRecruitingScore": max(recruiting_scores) if recruiting_scores else 0,
        "totalHighlightSeconds": round(total_highlight_secs, 1),
        "videoDurationSeconds": round(video_duration, 1),
        "processingTimeSeconds": round(elapsed, 1),
        "gradeBreakdown": grades,
        "playTypeBreakdown": play_types,
    }


def _detect_play_type_rules(
    motion_score: float,
    pose: str,
    crowd_energy: float,
    jersey_visible: bool,
    sport: str,
) -> str:
    """Rule-based play type detection — supplements v4 model output.

    Assigns meaningful play types based on signal combinations
    even when v4 models return no detections.
    """
    sl = sport.lower()
    if sl == "football":
        if pose == "throwing" and motion_score > 40:
            return "pass_play"
        if pose == "running" and motion_score > 60:
            return "qb_scramble"
        if crowd_energy > 0.75 and motion_score > 50:
            return "big_play"
        if motion_score > 70:
            return "game_action"  # High motion = active play
        return "game_action"
    elif sl == "basketball":
        if pose == "jumping" and motion_score > 40:
            return "shot_attempt"
        if pose == "running" and motion_score > 60:
            return "fast_break"
        if crowd_energy > 0.7:
            return "big_play"
        return "game_action"
    elif sl == "lacrosse":
        if pose == "throwing" and motion_score > 40:
            return "shot_attempt"
        if crowd_energy > 0.7:
            return "big_play"
        return "game_action"
    return "game_action"


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# Models that should stay loaded between requests (minimal footprint)
_ALWAYS_LOADED_MODELS = {
    "dead_ball_classifier_v5_model",
    "player_detector_v5_model",
}


def _force_cleanup_memory():
    """Force cleanup of ALL loaded models and caches to reclaim memory.

    Called when RSS exceeds threshold between requests.
    Unloads everything except _ALWAYS_LOADED_MODELS.
    """
    import gc
    rss_before = _get_rss_mb()
    LOGGER.info("Pipeline: force memory cleanup starting (RSS=%.0fMB)", rss_before)
    try:
        from app.services.roboflow_detector import roboflow_detector
        for attr in dir(roboflow_detector):
            if attr.endswith("_model") and attr not in _ALWAYS_LOADED_MODELS:
                if getattr(roboflow_detector, attr, None) is not None:
                    setattr(roboflow_detector, attr, None)
        roboflow_detector._loaded = False
        if roboflow_detector._jersey_upscaler is not None:
            roboflow_detector._jersey_upscaler = None
    except Exception:
        pass
    try:
        from app.services.detection_detector import clear_detector_cache
        clear_detector_cache()
    except Exception:
        pass
    gc.collect()
    # Free PyTorch GPU cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    rss_after = _get_rss_mb()
    LOGGER.info("Pipeline: force memory cleanup done (RSS: %.0fMB → %.0fMB, freed %.0fMB)",
                rss_before, rss_after, rss_before - rss_after)


async def run_analyze_pipeline(
    *,
    video_url: str | None = None,
    video_path: str | None = None,
    jersey_number: int = 0,
    jersey_color: str = "white",
    sport: str = "basketball",
    position: str | None = None,
    time_range_start: float = 0,
    time_range_end: float = 0,
    enable_audio: bool = True,
    enable_tracking: bool = True,
    enable_pose: bool = True,
    quality_mode: str = "auto",
    cancel_event: "threading.Event | None" = None,
) -> dict[str, Any]:
    """Run the full analysis pipeline with single-request concurrency.

    Steps:
    1. Acquire video (YouTube download or direct path)
    2. Run existing YOLO + OCR jersey detection
    3. Run motion scoring (optical flow)
    4. Run audio analysis (whistle + crowd)
    5. Run player tracking (BoT-SORT)
    6. Run pose estimation
    7. Classify play types
    8. Extract and rank clips
    """
    # ── Blocker 3: Semaphore — only 1 analyze request at a time ──
    async with _REQUEST_SEMAPHORE:
        return await _run_analyze_pipeline_impl(
            video_url=video_url,
            video_path=video_path,
            jersey_number=jersey_number,
            jersey_color=jersey_color,
            sport=sport,
            position=position,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            enable_audio=enable_audio,
            enable_tracking=enable_tracking,
            enable_pose=enable_pose,
            quality_mode=quality_mode,
            cancel_event=cancel_event,
        )


async def _run_analyze_pipeline_impl(
    *,
    video_url: str | None = None,
    video_path: str | None = None,
    jersey_number: int = 0,
    jersey_color: str = "white",
    sport: str = "basketball",
    position: str | None = None,
    time_range_start: float = 0,
    time_range_end: float = 0,
    enable_audio: bool = True,
    enable_tracking: bool = True,
    enable_pose: bool = True,
    quality_mode: str = "auto",
    cancel_event: "threading.Event | None" = None,
) -> dict[str, Any]:
    """Internal pipeline implementation (called under semaphore)."""
    start_time = time.perf_counter()
    phases_used: list[str] = []
    local_video_path: Path | None = None
    tmp_dir: Path | None = None
    frames_processed = 0
    youtube_strategy_used: str | None = None

    # ── Pre-request cleanup: free ALL models from previous requests ──
    rss_start = _get_rss_mb()
    LOGGER.info("Pipeline: REQUEST START — RSS=%.0fMB (threshold=%dMB)", rss_start, _MEMORY_THRESHOLD_MB)
    _force_cleanup_memory()
    rss_after_cleanup = _get_rss_mb()
    LOGGER.info("Pipeline: pre-request cleanup done (RSS: %.0fMB → %.0fMB)", rss_start, rss_after_cleanup)

    # Per-layer timing and debug info
    layer_timings: dict[str, dict] = {}

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    try:
        # ── Step 1: Acquire video ────────────────────────────────────────
        if video_url and is_youtube_url(video_url):
            # For long videos (>10 min), skip upfront download — chunked pipeline
            # will download each 10-min chunk separately. This enables strategy caching:
            # chunk 1 finds the working strategy, chunks 2+ use it directly (saves 60-200s each).
            _requested_duration = (time_range_end - time_range_start) if time_range_end > time_range_start else 0
            if _requested_duration > 600:
                LOGGER.info(
                    "Pipeline: SKIPPING upfront download for %.0fs game — will download per-chunk",
                    _requested_duration,
                )
                extract_start = time_range_start
                extract_end = time_range_end
                # local_video_path stays None — chunked pipeline will download
            else:
                LOGGER.info("Pipeline: downloading YouTube video")
                t0 = time.perf_counter()
                try:
                    from app.services.detection_runtime import PipelineSettings
                    settings = PipelineSettings()
                    dl_result: DownloadResult = await run_in_threadpool(
                        partial(
                            download_youtube_sync,
                            video_url,
                            start_time=time_range_start,
                            end_time=time_range_end,
                            yt_dlp_binary=settings.yt_dlp_binary,
                            ffmpeg_binary=settings.ffmpeg_binary,
                        )
                    )
                    local_video_path = dl_result.path
                    # If video was pre-trimmed by --download-sections, adjust frame extraction range
                    if dl_result.was_sectioned:
                        LOGGER.info("Pipeline: video was pre-sectioned (%.0f-%.0f) → extracting from 0", time_range_start, time_range_end)
                        extract_start = 0.0
                        extract_end = (time_range_end - time_range_start) if time_range_end > 0 else 0.0
                    else:
                        extract_start = time_range_start
                        extract_end = time_range_end
                    phases_used.append("youtube_download")
                    youtube_strategy_used = getattr(dl_result, "strategy_used", "download_success")
                    layer_timings["youtube_download"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "success"}
                    LOGGER.info("Pipeline: YouTube video downloaded to %s (sectioned=%s)", local_video_path, dl_result.was_sectioned)
                    # Log resolution for diagnostic purposes — also include in API response
                    vid_w, vid_h = get_video_resolution(local_video_path)
                    LOGGER.info("Pipeline: Video resolution = %dx%d", vid_w, vid_h)
                    layer_timings["youtube_download"]["video_resolution"] = f"{vid_w}x{vid_h}"
                    layer_timings["youtube_download"]["was_sectioned"] = dl_result.was_sectioned
                    # Log file size
                    file_size_mb = round(local_video_path.stat().st_size / 1024 / 1024, 1)
                    layer_timings["youtube_download"]["file_size_mb"] = file_size_mb
                    LOGGER.info("Pipeline: Downloaded file size = %sMB", file_size_mb)
                except Exception as exc:
                    layer_timings["youtube_download"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "failed", "error": str(exc)}
                    youtube_strategy_used = "all_failed"
                    LOGGER.error("Pipeline: YouTube download failed: %s", exc)
                    return _error_response(
                        f"YouTube download failed: {exc}",
                        time.perf_counter() - start_time,
                        layer_timings=layer_timings,
                    )
        elif video_url:
            # Direct URL — download it first
            extract_start = time_range_start
            extract_end = time_range_end
            try:
                import httpx
                tmp_dir = Path(tempfile.mkdtemp(prefix="clipt_dl_"))
                local_video_path = tmp_dir / "video.mp4"
                async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
                    resp = await client.get(video_url)
                    if resp.status_code == 200:
                        local_video_path.write_bytes(resp.content)
                        LOGGER.info("Pipeline: direct URL downloaded, %d bytes", len(resp.content))
                    else:
                        local_video_path = None
            except Exception as exc:
                LOGGER.warning("Pipeline: direct download failed, will pass URL to detector: %s", exc)
                local_video_path = None
        elif video_path:
            extract_start = time_range_start
            extract_end = time_range_end
            local_video_path = Path(video_path)
        else:
            extract_start = time_range_start
            extract_end = time_range_end

        # Get video duration
        video_duration = 0.0
        if local_video_path and local_video_path.exists():
            from app.services.detection_runtime import PipelineSettings
            settings = PipelineSettings()
            video_duration = get_video_duration(local_video_path, settings.ffprobe_binary)
            LOGGER.info("Pipeline: video duration = %.1fs", video_duration)

        # ── Determine effective duration for chunked vs standard path ──
        _effective_duration = video_duration
        if time_range_end > time_range_start:
            _effective_duration = time_range_end - time_range_start

        # ── CHUNKED PIPELINE for long videos (>10 min) ─────────────────
        # Process in 10-min chunks with per-chunk YouTube downloads.
        # Enables strategy caching: chunk 1 finds working strategy, chunks 2+
        # reuse it directly. Each chunk: download → extract → OCR → free.
        _CHUNK_THRESHOLD = 600  # 10 minutes — matches per-chunk download threshold
        _CHUNK_SIZE = 1800  # Overridden to 600 in _run_chunked_full_game if per-chunk
        if _effective_duration > _CHUNK_THRESHOLD:
            LOGGER.info(
                "Pipeline: CHUNKED MODE — %.0fs video, processing in %ds chunks",
                _effective_duration, _CHUNK_SIZE,
            )
            return await _run_chunked_full_game(
                local_video_path=local_video_path if (local_video_path and local_video_path.exists()) else None,
                video_url=video_url if (video_url and is_youtube_url(video_url)) else None,
                video_duration=video_duration if video_duration > 0 else _effective_duration,
                jersey_number=jersey_number,
                jersey_color=jersey_color,
                sport=sport,
                position=position,
                extract_start=time_range_start,
                extract_end=time_range_end if time_range_end > 0 else (video_duration if video_duration > 0 else _effective_duration),
                enable_audio=enable_audio,
                quality_mode=quality_mode,
                youtube_strategy_used=youtube_strategy_used,
                layer_timings=layer_timings,
                start_time=start_time,
                phases_used=phases_used,
                cancel_event=cancel_event,
            )

        # ── Step 2a: Load context-aware models for this request ──────────
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.reset_request_tracking()
            roboflow_detector._request_jersey_color = jersey_color
            request_models = roboflow_detector.load_for_request(sport, jersey_color)
            LOGGER.info("Pipeline: loaded %d models for sport=%s color=%s", len(request_models), sport, jersey_color)
        except Exception as exc:
            LOGGER.warning("Pipeline: load_for_request failed (non-fatal): %s", exc)

        # ── Step 2: Ali's ensemble is DEFERRED to Step 7.5e (last resort) ──
        jersey_detections: list[dict] = []
        ali_status = "deferred"  # Will run ONLY if other layers find <3 detections

        # ── Step 3: Extract frames for additional analysis ───────────────
        frames: list[tuple[float, np.ndarray]] = []
        _actual_fps_used = ANALYZE_FPS
        if local_video_path and local_video_path.exists():
            t0 = time.perf_counter()
            try:
                _adaptive_fps, _adaptive_max = _get_adaptive_fps(_effective_duration, sport)
                # Use env override if set, otherwise adaptive
                _use_fps = ANALYZE_FPS if os.getenv("ANALYZE_FPS") else _adaptive_fps
                _use_max = int(os.getenv("MAX_FRAMES", "0")) or _adaptive_max
                LOGGER.info(
                    "Pipeline: adaptive FPS = %d (duration=%.0fs), max_frames = %d",
                    _use_fps, _effective_duration, _use_max,
                )
                _actual_fps_used = _use_fps
                # Standard path — full games (>1800s) use chunked pipeline above
                frames = _extract_frames(
                        local_video_path, fps=_use_fps, sport=sport,
                        start_sec=extract_start, end_sec=extract_end,
                        max_frames=_use_max,
                    )
                frames_processed = len(frames)
                layer_timings["frame_extraction"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "success", "frames": frames_processed}
                LOGGER.info("Pipeline: extracted %d frames", len(frames))
            except Exception as exc:
                layer_timings["frame_extraction"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
                LOGGER.warning("Pipeline: frame extraction failed: %s", exc)

        # ── Step 3.5: Dead ball filtering (BEFORE OCR layers) ──────────────
        dead_ball_count = 0
        dead_ball_ratio = 0.0
        dead_ball_by_ts: dict[float, str] = {}  # timestamp → "dead_ball" or "live_play"
        live_frames: list[tuple[float, np.ndarray]] = frames  # default: all frames
        if frames:
            t0 = time.perf_counter()
            try:
                from app.services.roboflow_detector import roboflow_detector
                roboflow_detector.load()

                _live: list[tuple[float, np.ndarray]] = []
                # Sport-specific dead ball confidence thresholds
                # Classifier was trained primarily on basketball — football field
                # looks like "dead ball" to it, so require very high confidence.
                _sport_lower = sport.lower()
                if _sport_lower == "football":
                    _db_conf = 0.85  # Very high bar — classifier unreliable on football
                elif _sport_lower == "lacrosse":
                    _db_conf = 0.70  # Moderate — outdoor fields similar to football
                else:
                    _db_conf = 0.40  # Basketball default — classifier trained on this
                # Football: skip dead ball entirely (classifier unreliable — always
                # triggers safety bypass, wasting 18+ seconds).
                # Full games: sample every 4th frame (saves ~75% time).
                if _sport_lower == "football":
                    _db_step = 0  # Skip entirely
                elif video_duration > 1800:
                    _db_step = 4
                else:
                    _db_step = 2  # Every other frame for short/medium videos
                if _db_step == 0:
                    # Skip dead ball entirely (e.g. football — classifier unreliable)
                    _live = list(frames)
                    dead_ball_ratio = 0.0
                else:
                    for idx, (ts, frame) in enumerate(frames):
                        if idx % _db_step != 0:
                            # Skip dead ball check — assume live for unsampled frames
                            _live.append((ts, frame))
                            continue
                        db_result = roboflow_detector.classify_dead_ball(frame, conf=_db_conf)
                        if db_result:
                            dead_ball_by_ts[ts] = db_result
                        if db_result == "dead_ball":
                            dead_ball_count += 1
                        else:
                            _live.append((ts, frame))

                if _db_step > 0:
                    _db_sampled = len(frames) // _db_step + (1 if len(frames) % _db_step else 0)
                    dead_ball_ratio = dead_ball_count / _db_sampled if _db_sampled else 0.0
                live_ratio = len(_live) / len(frames) if frames else 1.0

                # Safety: if dead ball ratio > 50% OR less than 30% of frames survive,
                # the classifier is likely miscalibrated — use ALL frames instead.
                if dead_ball_ratio > 0.5:
                    LOGGER.warning(
                        "Pipeline: dead ball ratio %.0f%% exceeds 50%% cap — classifier likely miscalibrated. "
                        "Using ALL %d frames instead of %d filtered.",
                        dead_ball_ratio * 100, len(frames), len(_live),
                    )
                    live_frames = frames  # override: keep all frames
                elif live_ratio < 0.3:
                    LOGGER.warning(
                        "Pipeline: only %.0f%% frames survived dead ball filter (minimum 30%%). "
                        "Using ALL %d frames instead of %d filtered.",
                        live_ratio * 100, len(frames), len(_live),
                    )
                    live_frames = frames  # override: keep all frames
                elif _live:
                    live_frames = _live
                # If ALL frames are dead ball, keep all frames (something is better than nothing)
                layer_timings["dead_ball_filter"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success",
                    "dead_frames": dead_ball_count,
                    "total_frames": len(frames),
                    "dead_ratio": round(dead_ball_ratio, 2),
                    "live_frames_for_ocr": len(live_frames),
                    "filter_bypassed": dead_ball_ratio > 0.5 or live_ratio < 0.3,
                }
                LOGGER.info(
                    "Pipeline: dead ball filter — %d/%d frames skipped (%.0f%%), %d live frames for OCR%s",
                    dead_ball_count, len(frames), dead_ball_ratio * 100, len(live_frames),
                    " [BYPASSED — too aggressive]" if (dead_ball_ratio > 0.5 or live_ratio < 0.3) else "",
                )
            except Exception as exc:
                layer_timings["dead_ball_filter"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
                LOGGER.warning("Pipeline: dead ball filter failed (non-fatal): %s", exc)

        # ── Resolve quality_mode ──────────────────────────────────────────
        resolved_quality = quality_mode
        if quality_mode == "auto" and frames:
            # Check source video width (before preprocessing) via VideoCapture
            _probe_w = 0
            if local_video_path and local_video_path.exists():
                _cap = cv2.VideoCapture(str(local_video_path))
                if _cap.isOpened():
                    _probe_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    _cap.release()
            resolved_quality = "aggressive" if _probe_w < 960 else "standard"
            LOGGER.info(
                "Pipeline: quality_mode auto → %s (source width=%d)",
                resolved_quality, _probe_w,
            )

        # ── Step 4: Motion scoring ───────────────────────────────────────
        motion_scores: dict[float, float] = {}
        t0 = time.perf_counter()
        try:
            if len(frames) >= 2:
                for i in range(len(frames) - 1):
                    t, prev_frame = frames[i]
                    t_next, curr_frame = frames[i + 1]
                    score = compute_motion_score(prev_frame, curr_frame)
                    motion_scores[t_next] = score.score
                phases_used.append("motion_scoring")
                layer_timings["motion_scoring"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "success", "scores": len(motion_scores)}
                LOGGER.info("Pipeline: computed %d motion scores", len(motion_scores))
        except Exception as exc:
            layer_timings["motion_scoring"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
            LOGGER.warning("Pipeline: motion scoring failed: %s", exc)

        # ── Step 5: Audio analysis ───────────────────────────────────────
        audio_result = AudioAnalysisResult(has_audio=False)
        if enable_audio and local_video_path and local_video_path.exists():
            t0 = time.perf_counter()
            try:
                from app.services.audio_analyzer import analyze_audio
                from app.services.detection_runtime import PipelineSettings
                settings = PipelineSettings()
                audio_path = extract_audio(local_video_path, settings.ffmpeg_binary)
                if audio_path:
                    audio_result = analyze_audio(audio_path)
                    if audio_result.has_audio:
                        phases_used.append("audio_analysis")
                    layer_timings["audio_analysis"] = {
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                        "status": "success" if audio_result.has_audio else "no_audio",
                        "events": len(audio_result.events),
                        "boundaries": len(audio_result.play_boundaries),
                    }
                    LOGGER.info("Pipeline: audio analysis complete, %d events, %d boundaries",
                                len(audio_result.events), len(audio_result.play_boundaries))
            except Exception as exc:
                layer_timings["audio_analysis"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
                LOGGER.warning("Pipeline: audio analysis failed: %s", exc)

        # ── Step 6: Player tracking ──────────────────────────────────────
        tracking_result = None
        if enable_tracking and frames:
            t0 = time.perf_counter()
            try:
                from app.services.player_tracker import PlayerTracker
                tracker = PlayerTracker(target_jersey=jersey_number)

                # Convert jersey detections to tracker format
                jersey_det_by_frame: dict[int, list[dict]] = {}
                for det in jersey_detections:
                    frame_idx = int(det.get("timestamp", 0) * 2)  # Approximate frame index
                    if frame_idx not in jersey_det_by_frame:
                        jersey_det_by_frame[frame_idx] = []
                    jersey_det_by_frame[frame_idx].append({
                        "jersey_number": jersey_number,
                        "confidence": det.get("confidence", 0),
                        "x1": det.get("x1", 0), "y1": det.get("y1", 0),
                        "x2": det.get("x2", 0), "y2": det.get("y2", 0),
                    })

                for i, (t, frame) in enumerate(frames):
                    frame_dets = jersey_det_by_frame.get(i, None)
                    tracker.track_frame(frame, i, frame_dets)

                tracking_result = tracker.get_result()
                if tracking_result.tracks:
                    phases_used.append("player_tracking")
                layer_timings["player_tracking"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if tracking_result.tracks else "no_tracks",
                    "tracks": len(tracking_result.tracks) if tracking_result else 0,
                }
                LOGGER.info("Pipeline: tracking found %d tracks, target=%s",
                            len(tracking_result.tracks), tracking_result.target_track_id)
            except Exception as exc:
                layer_timings["player_tracking"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
                LOGGER.warning("Pipeline: player tracking failed: %s", exc)

        # ── Step 7: Pose estimation ──────────────────────────────────────
        pose_results: dict[float, dict] = {}
        if enable_pose and frames:
            t0 = time.perf_counter()
            try:
                from app.services.pose_analyzer import PoseAnalyzer
                analyzer = PoseAnalyzer()

                # Sample every 3rd frame for speed
                for i in range(0, len(frames), 3):
                    t, frame = frames[i]
                    result = analyzer.analyze_frame(frame)
                    pose_results[t] = {
                        "action": result.action,
                        "intensity": result.intensity,
                        "is_facing": result.is_facing_camera,
                    }
                if pose_results:
                    phases_used.append("pose_estimation")
                layer_timings["pose_estimation"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if pose_results else "no_poses",
                    "frames_analyzed": len(pose_results),
                }
                LOGGER.info("Pipeline: pose estimated for %d frames", len(pose_results))
            except Exception as exc:
                layer_timings["pose_estimation"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
                LOGGER.warning("Pipeline: pose estimation failed: %s", exc)

        # OCR confidence threshold — lower for dark jerseys, football, aggressive mode
        from app.services.roboflow_detector import is_dark_color, is_navy
        _is_dark_jersey = is_dark_color(jersey_color)
        _is_navy_jersey = is_navy(jersey_color)
        if _is_navy_jersey:
            ocr_conf = 0.08  # Navy jerseys: dark + reflective = very hard to read
        elif _is_dark_jersey:
            ocr_conf = 0.12
        elif sport.lower() == "football":
            ocr_conf = FOOTBALL_CONF_THRESHOLD  # 0.08 — smaller numbers on helmets/distance
        elif resolved_quality == "aggressive":
            ocr_conf = 0.15
        else:
            ocr_conf = 0.18  # Lowered from 0.2 for better recall

        # Use live_frames (dead ball filtered) for OCR steps, all frames for motion
        ocr_frames = live_frames if live_frames else frames

        # ── Frame preprocessing for dark/navy jerseys ──────────────────
        if _is_dark_jersey and ocr_frames:
            LOGGER.info("Pipeline: applying dark jersey preprocessing in-place (navy=%s)", _is_navy_jersey)
            # Build gamma LUT once (shared across all frames)
            gamma = 1.5 if _is_navy_jersey else 1.3
            inv_gamma = 1.0 / gamma
            _gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            clip_limit = 5.0 if _is_navy_jersey else 4.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            # Process in-place to avoid doubling memory
            for idx in range(len(ocr_frames)):
                t, frame = ocr_frames[idx]
                # CLAHE for dark jerseys (skip upscale at 720p — already sufficient)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                # Gamma boost
                frame = cv2.LUT(frame, _gamma_table)
                # Unsharp mask
                gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
                frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
                ocr_frames[idx] = (t, frame)

        # ── Free Ali's models before loading Roboflow ──────────────────
        # Ali's warmup detector (jersey_number_yolo11m.pt + yolo26n-seg.pt +
        # public_reader_ensemble) uses ~600MB. Railway kills at ~2.3GB.
        # Freeing Ali here gives Roboflow enough room for v5 inference.
        try:
            from app.services.detection_detector import clear_detector_cache
            import gc as _gc
            clear_detector_cache()
            _gc.collect()
            LOGGER.info("Pipeline: freed Ali detector cache before Roboflow loading")
        except Exception:
            pass

        # ── Step 7.4: v7 football-specialist OCR (runs BEFORE v5 for football) ──
        _is_full_game = video_duration > 1800
        v7_football_detections: list[dict] = []
        v7_navy_detections = 0
        v7_player_crops = 0
        if sport.lower() == "football" and ocr_frames:
            # Log v7 model availability + memory for diagnostics
            _v7_rss = _get_rss_mb()
            LOGGER.info(
                "Pipeline: v7 football model status — ocr=%s, crop=%s, navy=%s, RSS=%.0fMB",
                roboflow_detector.football_jersey_ocr_v7_model is not None,
                roboflow_detector.football_player_crop_v7_model is not None,
                roboflow_detector.navy_jersey_specialist_v7_model is not None,
                _v7_rss,
            )
            t0 = time.perf_counter()
            try:
                _v7_time_limit = 60 if _is_full_game else 30  # seconds
                # v7 takes ~5.5s/frame on CPU.  Limit sample to ~6-10 UNIFORMLY
                # spaced frames so detections cover the whole video, not just
                # the first few seconds.
                _v7_budget_frames = max(6, _v7_time_limit // 5)  # ~6 for 30s, ~12 for 60s
                _v7_step = max(1, len(ocr_frames) // _v7_budget_frames)
                _v7_sample = ocr_frames[::_v7_step][:_v7_budget_frames * 2]  # 2x budget for headroom
                _v7_count = 0
                _v7_navy_count = 0
                _v7_crops = 0

                for ts, frame in _v7_sample:
                    if time.perf_counter() - t0 > _v7_time_limit:
                        LOGGER.info("Pipeline: v7 football OCR hit time limit (%ds)", _v7_time_limit)
                        break
                    dets = roboflow_detector.detect_football_jersey_v7(
                        frame, jersey_number, conf=ocr_conf,
                    )
                    if dets:
                        for d in dets:
                            d["timestamp"] = ts
                            # Use bbox as player_bbox for OCR-to-track mapping
                            if "bbox" in d and "player_bbox" not in d:
                                d["player_bbox"] = d["bbox"]
                            v7_football_detections.append(d)
                            _v7_count += 1
                            if "v7_navy" in d.get("layer", ""):
                                _v7_navy_count += 1
                    _v7_crops += 1  # count frames processed

                v7_navy_detections = _v7_navy_count
                v7_player_crops = _v7_crops
                LOGGER.info(
                    "Pipeline: v7 football OCR — %d detections (%d navy) from %d frames in %.1fs",
                    _v7_count, _v7_navy_count, _v7_crops, time.perf_counter() - t0,
                )
                layer_timings["v7_football_ocr"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if _v7_count > 0 else "no_detections",
                    "detections": _v7_count,
                    "navy_detections": _v7_navy_count,
                    "frames_processed": _v7_crops,
                }
            except Exception as exc:
                LOGGER.warning("Pipeline: v7 football OCR failed (non-fatal): %s", exc)
                layer_timings["v7_football_ocr"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "error",
                    "error": str(exc)[:200],
                }

        # ── Step 7.5a: v5 player detection → v5 OCR on crops (PRIMARY) ──
        # Guardrails: time limit, crop limit scales with frame count, early exit
        # after consecutive FRAMES with zero detections.
        _is_full_game = video_duration > 1800
        _V5_TIME_LIMIT = 120 if _is_full_game else 60  # seconds — reduced from 180/90 for speed
        # Scale crop limit: 200 for short videos, up to 1200 for full games.
        # At ~120ms/crop, 1200 crops ≈ 144s — within the 180s time limit for full games.
        _V5_MAX_CROPS = min(1200 if _is_full_game else 500, max(200, frames_processed))
        # Full games + football: more patient early exit
        if sport.lower() == "football":
            _V5_EARLY_EXIT_FRAMES = 200
        elif _is_full_game:
            _V5_EARLY_EXIT_FRAMES = 100  # basketball full game: 100 frame patience
        else:
            _V5_EARLY_EXIT_FRAMES = 50
        v5_ocr_detections: list[dict] = []
        v5_players_found = 0
        v5_no_player_frames = 0
        v5_crop_dims: list[str] = []
        v5_total_crops = 0
        v5_consecutive_frame_misses = 0
        v5_exit_reason = ""
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            if ocr_frames:
                # v5 takes ~2s/frame. Ensure UNIFORM coverage of the video:
                # With a 60s budget → ~30 frames max. Step through the full
                # frame list so detections aren't biased to the first N seconds.
                _v5_budget = max(20, _V5_TIME_LIMIT // 2)  # ~30 for 60s, ~60 for 120s
                _v5_step = max(1, len(ocr_frames) // _v5_budget)
                sampled_frames = ocr_frames[::_v5_step]
                LOGGER.info("Pipeline: v5 OCR layer running on %d/%d frames (sampled)", len(sampled_frames), len(ocr_frames))
                _is_football = sport.lower() == "football"
                _v5_oversized_skipped = 0
                _v5_redetected = 0
                for t, frame in sampled_frames:
                    # Time limit check
                    if time.perf_counter() - t0 > _V5_TIME_LIMIT:
                        v5_exit_reason = f"time_limit ({_V5_TIME_LIMIT}s)"
                        LOGGER.info("Pipeline: v5 OCR hit time limit (%ds), moving on", _V5_TIME_LIMIT)
                        break
                    # For football: validate crop sizes and re-detect within oversized regions
                    # Football: higher conf to reduce false positives (tiny garbage bboxes)
                    _player_conf = 0.35 if _is_football else 0.20
                    players = roboflow_detector.detect_players_v5(
                        frame, conf=_player_conf,
                        validate_crop_size=_is_football,
                    )
                    v5_players_found += len(players) if players else 0

                    _frame_had_detection = False
                    _frame_hit_crop_limit = False

                    if players:
                        for player in players[:3]:  # Max 3 players per frame (better coverage)
                            if v5_total_crops >= _V5_MAX_CROPS:
                                v5_exit_reason = f"crop_limit ({_V5_MAX_CROPS})"
                                _frame_hit_crop_limit = True
                                break
                            x1, y1, x2, y2 = [int(c) for c in player["bbox"]]
                            h, w = frame.shape[:2]
                            # Football: tighter padding to avoid oversized crops
                            _pad_ratio = 0.10 if _is_football else 0.25
                            pad_x = int((x2 - x1) * _pad_ratio)
                            pad_y = int((y2 - y1) * _pad_ratio)
                            cx1 = max(0, x1 - pad_x)
                            cy1 = max(0, y1 - pad_y)
                            cx2 = min(w, x2 + pad_x)
                            cy2 = min(h, y2 + pad_y)
                            crop = frame[cy1:cy2, cx1:cx2]
                            if crop.size == 0:
                                continue
                            ch, cw = crop.shape[:2]
                            # Skip truly unusable crops (< 8px wide)
                            if cw < 8 or ch < 12:
                                v5_crop_dims.append(f"{cw}x{ch}(skip)")
                                continue

                            # ── Navy/dark jersey CLAHE enhancement (before upscale) ──
                            # For dark jerseys, enhance contrast first to preserve
                            # digit edges through the upscale process
                            _is_dark = jersey_color.lower() in (
                                "navy", "dark", "black", "dark blue", "dark green",
                                "maroon", "dark red", "purple",
                            )
                            if _is_dark and min(cw, ch) >= 8:
                                _lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                                _tile = (4, 4) if max(cw, ch) < 50 else (8, 8)
                                _clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=_tile)
                                _lab[:, :, 0] = _clahe.apply(_lab[:, :, 0])
                                crop = cv2.cvtColor(_lab, cv2.COLOR_LAB2BGR)

                            # ── Adaptive upscale: more aggressive for smaller crops ──
                            if cw < 50:
                                _scale = 8  # Tiny crops: 9px → 72px
                            elif cw < 100:
                                _scale = 4  # Small crops: 80px → 320px
                            elif _is_football and max(cw, ch) < 500:
                                _scale = 2  # Football medium crops
                            else:
                                _scale = 1

                            if _scale > 1:
                                crop = cv2.resize(crop, (cw * _scale, ch * _scale), interpolation=cv2.INTER_CUBIC)
                                # Sharpen after upscale
                                gaussian = cv2.GaussianBlur(crop, (0, 0), 1.5)
                                crop = cv2.addWeighted(crop, 1.5, gaussian, -0.5, 0)
                                v5_crop_dims.append(f"{cw}x{ch}→{cw*_scale}x{ch*_scale}")
                            else:
                                v5_crop_dims.append(f"{cw}x{ch}")
                            v5_total_crops += 1
                            dets = roboflow_detector.detect_jersey_v5(
                                crop, jersey_number=jersey_number, conf=ocr_conf,
                                skip_preprocess=True,
                            )
                            if dets:
                                v5_ocr_detections.extend({
                                    **d, "timestamp": t,
                                    "player_bbox": [cx1, cy1, cx2, cy2],
                                } for d in dets)
                                _frame_had_detection = True
                    else:
                        v5_no_player_frames += 1

                    # ── Football grid scan fallback ──
                    # When player crops fail for football (distant camera),
                    # try OCR on 3x3 grid sections of the full frame
                    if _is_football and not _frame_had_detection and v5_consecutive_frame_misses > 10:
                        fh, fw = frame.shape[:2]
                        for row in range(3):
                            for col in range(3):
                                if v5_total_crops >= _V5_MAX_CROPS:
                                    break
                                gx1 = int(col * fw / 3 * 0.8)
                                gy1 = int(row * fh / 3 * 0.8)
                                gx2 = int(min(fw, gx1 + fw / 3 * 1.2))
                                gy2 = int(min(fh, gy1 + fh / 3 * 1.2))
                                region = frame[gy1:gy2, gx1:gx2]
                                if region.size == 0:
                                    continue
                                rh, rw = region.shape[:2]
                                # Upscale region 2x for OCR
                                region_up = cv2.resize(region, (rw * 2, rh * 2), interpolation=cv2.INTER_CUBIC)
                                v5_total_crops += 1
                                v5_crop_dims.append(f"grid_{rw}x{rh}→{rw*2}x{rh*2}")
                                grid_dets = roboflow_detector.detect_jersey_v5(
                                    region_up, jersey_number=jersey_number, conf=ocr_conf,
                                    skip_preprocess=True,
                                )
                                if grid_dets:
                                    v5_ocr_detections.extend({**d, "timestamp": t} for d in grid_dets)
                                    _frame_had_detection = True

                    if _frame_hit_crop_limit:
                        break

                    # Frame-based early exit: count consecutive FRAMES with zero
                    # detections.  A single frame with 5 crop misses = 1 miss,
                    # not 5.  Resets whenever ANY crop in a frame produces a hit.
                    if _frame_had_detection:
                        v5_consecutive_frame_misses = 0
                    else:
                        v5_consecutive_frame_misses += 1
                        if v5_consecutive_frame_misses >= _V5_EARLY_EXIT_FRAMES:
                            v5_exit_reason = f"early_exit ({_V5_EARLY_EXIT_FRAMES} consecutive frame misses)"
                            LOGGER.info("Pipeline: v5 OCR early exit after %d consecutive frame misses", _V5_EARLY_EXIT_FRAMES)
                            break
                if v5_ocr_detections:
                    phases_used.append("v5_ocr_detection")
            layer_timings["v5_ocr_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success" if v5_ocr_detections else "no_model_or_no_detections",
                "detections": len(v5_ocr_detections),
                "players_found": v5_players_found,
                "no_player_frames": v5_no_player_frames,
                "crops_processed": v5_total_crops,
                "consecutive_frame_misses": v5_consecutive_frame_misses,
                "exit_reason": v5_exit_reason or "completed",
                "crop_dimensions_sample": v5_crop_dims[:10],
            }
            LOGGER.info("Pipeline: v5 OCR found %d detections, %d crops processed, %d frame misses, exit: %s",
                        len(v5_ocr_detections), v5_total_crops, v5_consecutive_frame_misses, v5_exit_reason or "completed")
            # Pipeline-level tracking safety net: ensure v5 models are tracked
            if v5_players_found > 0:
                roboflow_detector._track_model_call("player_detector_v5_pipeline", v5_players_found)
            if v5_ocr_detections:
                roboflow_detector._track_model_call("jersey_ocr_universal_v5_pipeline", len(v5_ocr_detections))
        except Exception as exc:
            layer_timings["v5_ocr_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: v5 OCR layer failed (non-fatal): %s", exc)

        # ── Step 7.5a-2: Football full-frame OCR fallback ──────────────────
        # Football crops struggle at distance — try digit detector on full frames
        football_ff_detections: list[dict] = []
        _is_football_sport = sport.lower() == "football"
        if _is_football_sport and len(v5_ocr_detections) == 0:
            t0 = time.perf_counter()
            try:
                from app.services.roboflow_detector import roboflow_detector
                roboflow_detector.load()
                _ff_frames = ocr_frames[::3] if len(ocr_frames) > 20 else ocr_frames
                LOGGER.info("Pipeline: football full-frame OCR on %d frames", len(_ff_frames))
                for t, frame in _ff_frames[:30]:  # Max 30 frames
                    if time.perf_counter() - t0 > 30:
                        break
                    # Try full-frame with v5 OCR model
                    ff_dets = roboflow_detector.detect_jersey_v5(
                        frame, jersey_number=jersey_number, conf=0.10,
                        skip_preprocess=False,
                    )
                    if ff_dets:
                        football_ff_detections.extend({**d, "timestamp": t, "layer": "v5_football_fullframe"} for d in ff_dets)
                    # Also try football digit detector
                    fd_dets = roboflow_detector.detect_football_digits(frame, jersey_number, conf=0.15)
                    if fd_dets:
                        football_ff_detections.extend({**d, "timestamp": t} for d in fd_dets)
                if football_ff_detections:
                    v5_ocr_detections.extend(football_ff_detections)
                    phases_used.append("football_fullframe_ocr")
                    LOGGER.info("Pipeline: football full-frame OCR found %d detections", len(football_ff_detections))
                layer_timings["football_fullframe_ocr"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if football_ff_detections else "no_detections",
                    "detections": len(football_ff_detections),
                }
            except Exception as exc:
                LOGGER.warning("Pipeline: football full-frame OCR failed (non-fatal): %s", exc)
                layer_timings["football_fullframe_ocr"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}

        # ── Step 7.5b: v3 OCR pipeline (secondary — skip if v5 found enough) ──
        _V3_TIME_LIMIT = 45  # seconds
        v3_ocr_detections: list[dict] = []
        t0 = time.perf_counter()
        if len(v5_ocr_detections) >= 3:
            LOGGER.info("Pipeline: skipping v3 OCR — v5 already found %d detections", len(v5_ocr_detections))
            layer_timings["v3_ocr_detection"] = {"elapsed_ms": 0, "status": "skipped_v5_sufficient", "detections": 0}
        else:
            try:
                from app.services.roboflow_detector import roboflow_detector
                roboflow_detector.load()

                # Football: sample more aggressively (every 2 instead of 3) since v5 player detector struggles
                _v3_step = 2 if sport.lower() == "football" else 3
                sampled = ocr_frames[::_v3_step] if len(ocr_frames) > 20 else ocr_frames
                LOGGER.info("Pipeline: v3 OCR layer running on %d/%d frames for sport=%s (step=%d)", len(sampled), len(ocr_frames), sport, _v3_step)
                for t, frame in sampled:
                    if time.perf_counter() - t0 > _V3_TIME_LIMIT:
                        LOGGER.info("Pipeline: v3 OCR hit time limit (%ds)", _V3_TIME_LIMIT)
                        break
                    dets = roboflow_detector.detect_with_player_crops(
                        frame, jersey_number=jersey_number, sport=sport, conf=ocr_conf,
                    )
                    v3_only = [d for d in dets if d.get("layer", "").startswith("v3_")]
                    v3_ocr_detections.extend({**d, "timestamp": t} for d in v3_only)
                if v3_ocr_detections:
                    phases_used.append("v3_ocr_detection")
                layer_timings["v3_ocr_detection"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if v3_ocr_detections else "no_detections",
                    "detections": len(v3_ocr_detections),
                }
                LOGGER.info("Pipeline: v3 OCR (secondary) found %d detections", len(v3_ocr_detections))
            except Exception as exc:
                layer_timings["v3_ocr_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
                LOGGER.warning("Pipeline: v3 OCR layer failed (non-fatal): %s", exc)

        # ── Step 7.5c: v2 universal OCR (tertiary — skip if prior layers found enough) ──
        _PRIOR_DETECTIONS = len(v5_ocr_detections) + len(v3_ocr_detections)
        universal_v2_detections: list[dict] = []
        t0 = time.perf_counter()
        if _PRIOR_DETECTIONS >= 3:
            LOGGER.info("Pipeline: skipping v2/v1 layers — prior layers found %d detections", _PRIOR_DETECTIONS)
            layer_timings["universal_v2_ocr"] = {"elapsed_ms": 0, "status": "skipped_sufficient", "detections": 0}
            v2_sport_detections: list[dict] = []
            v3_primary_detections: list[dict] = []
            layer_timings["v2_sport_detection"] = {"elapsed_ms": 0, "status": "skipped_sufficient"}
        else:
            _V2_TIME_LIMIT = 30
            try:
                from app.services.roboflow_detector import roboflow_detector
                roboflow_detector.load()

                if roboflow_detector.jersey_number_universal_v1_model is not None:
                    sampled = ocr_frames[::4] if len(ocr_frames) > 15 else ocr_frames
                    LOGGER.info("Pipeline: Universal v2 layer running on %d frames (conf=%.2f)", len(sampled), ocr_conf)
                    for t, frame in sampled:
                        if time.perf_counter() - t0 > _V2_TIME_LIMIT:
                            break
                        dets = roboflow_detector._run_universal_ocr(frame, jersey_number, conf=ocr_conf)
                        universal_v2_detections.extend({**d, "timestamp": t} for d in dets)
                    if universal_v2_detections:
                        phases_used.append("universal_v2_ocr")
                layer_timings["universal_v2_ocr"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if universal_v2_detections else "no_detections",
                    "detections": len(universal_v2_detections),
                }
                LOGGER.info("Pipeline: Universal v2 found %d detections", len(universal_v2_detections))
            except Exception as exc:
                layer_timings["universal_v2_ocr"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
                LOGGER.warning("Pipeline: Universal v2 layer failed (non-fatal): %s", exc)

            # ── Step 7.5d: v2 sport-specific + v1 legacy (fallback — 30s limit) ──
            v2_sport_detections = []
            v3_primary_detections = []
            t0 = time.perf_counter()
            try:
                from app.services.roboflow_detector import roboflow_detector
                roboflow_detector.load()

                sampled = ocr_frames[::4] if len(ocr_frames) > 15 else ocr_frames
                LOGGER.info("Pipeline: v2 sport-specific + v1 fallback running on %d frames", len(sampled))
                for t, frame in sampled:
                    if time.perf_counter() - t0 > _V2_TIME_LIMIT:
                        break
                    dets = roboflow_detector._run_sport_specific_v2(frame, jersey_number, sport, conf=ocr_conf)
                    v2_sport_detections.extend({**d, "timestamp": t} for d in dets)
                    dets = roboflow_detector.detect_football_digits(frame, jersey_number, conf=ocr_conf)
                    v2_sport_detections.extend({**d, "timestamp": t} for d in dets)
                    dets = roboflow_detector.detect_football_tracker(frame, jersey_number, conf=ocr_conf)
                    v2_sport_detections.extend({**d, "timestamp": t} for d in dets)

                if v2_sport_detections:
                    phases_used.append("v2_sport_detection")
                layer_timings["v2_sport_detection"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if v2_sport_detections else "no_detections",
                    "v2_sport_detections": len(v2_sport_detections),
                }
                LOGGER.info("Pipeline: v2/v1 found %d detections", len(v2_sport_detections))
            except Exception as exc:
                layer_timings["v2_sport_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
                LOGGER.warning("Pipeline: v2/v1 layer failed (non-fatal): %s", exc)

        # ── Free ALL models + OCR frames before Ali ──
        import gc as _gc2
        try:
            from app.services.roboflow_detector import roboflow_detector
            # Unload ALL models (including v5 essentials) to make room for Ali
            for _attr_name in dir(roboflow_detector):
                if _attr_name.endswith("_model") and getattr(roboflow_detector, _attr_name, None) is not None:
                    setattr(roboflow_detector, _attr_name, None)
            roboflow_detector._loaded = False
            _gc2.collect()
            LOGGER.info("Pipeline: freed ALL Roboflow models before Ali")
        except Exception:
            pass
        # Save timestamps BEFORE clearing frame data (needed for motion supplement later)
        _frame_timestamps = [t for t, _ in frames] if frames else []
        _frame_count = len(_frame_timestamps)
        # Free ALL frame data (no longer needed — Ali processes its own frames)
        ocr_frames = []
        frames.clear()
        if live_frames is not frames:
            live_frames.clear()
        _gc2.collect()
        LOGGER.info("Pipeline: freed frames + models, RSS before Ali")

        # ── Step 7.5e: Ali ensemble (LAST RESORT — only if <3 detections) ──
        # Count total OCR detections from all trained layers
        total_trained_ocr = len(v5_ocr_detections) + len(v3_ocr_detections) + len(universal_v2_detections) + len(v2_sport_detections) + len(v3_primary_detections)
        t0 = time.perf_counter()

        # Skip Ali for football — Ali is trained on soccer/basketball jerseys and
        # consistently returns 0 detections on football footage.  Saves ~2GB RAM.
        _skip_ali = sport.lower() == "football"
        if _skip_ali:
            LOGGER.info("Pipeline: skipping Ali for football (not trained on football jerseys)")
            ali_status = "skipped_football"
            layer_timings["ali_jersey_detection"] = {
                "elapsed_ms": 0, "status": ali_status, "detections": 0,
                "reason": "Ali not trained on football jerseys",
            }
        elif total_trained_ocr < 3:
            LOGGER.info("Pipeline: Ali ensemble (LAST RESORT) — only %d detections from trained layers, running Ali", total_trained_ocr)
            try:
                if local_video_path and local_video_path.exists():
                    ali_video_url = None
                    ali_video_path = str(local_video_path)
                else:
                    ali_video_url = video_url
                    ali_video_path = video_path

                jersey_detections = _run_jersey_detection(
                    video_url=ali_video_url,
                    video_path=ali_video_path,
                    jersey_number=jersey_number,
                    jersey_color=jersey_color,
                    sport=sport,
                    position=position,
                )
                if jersey_detections:
                    phases_used.append("jersey_detection")
                    ali_status = "working_fallback"
                else:
                    ali_status = "no_detections"
                layer_timings["ali_jersey_detection"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": ali_status,
                    "detections": len(jersey_detections),
                    "reason": f"last_resort (trained_layers_found={total_trained_ocr})",
                }
                LOGGER.info("Pipeline: Ali (last resort) found %d detections", len(jersey_detections))
            except Exception as exc:
                ali_status = "error"
                layer_timings["ali_jersey_detection"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "error",
                    "error": str(exc)[:200],
                    "detections": 0,
                }
                LOGGER.error("Pipeline: Ali jersey detection failed: %s", exc)
        else:
            ali_status = "skipped_not_needed"
            layer_timings["ali_jersey_detection"] = {
                "elapsed_ms": 0,
                "status": ali_status,
                "detections": 0,
                "reason": f"trained_layers_found={total_trained_ocr} (>=3, Ali not needed)",
            }
            LOGGER.info("Pipeline: Ali SKIPPED — %d detections from trained layers sufficient", total_trained_ocr)

        # ── Step 7.6: v4 outcome detection + v5 outcome classification ──
        v4_outcomes_by_ts: dict[float, str] = {}
        v4_outcome_detections: list[dict] = []
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            if frames:
                LOGGER.info("Pipeline: v4 outcome + v5 classifier layer checking availability")
                # Sample every 3rd frame for outcome detection (speed)
                for i in range(0, len(frames), 3):
                    t, frame = frames[i]

                    # v4 outcome models (sport-specific, returns [] if not loaded)
                    v4_dets = roboflow_detector.detect_outcome_v4(frame, sport=sport)
                    if v4_dets:
                        # Filter low-confidence scoring outcomes (TD/goal must be > 0.6)
                        _HIGH_STAKES = {"touchdown", "made_shot", "goal"}
                        v4_dets = [d for d in v4_dets
                                   if d.get("confidence", 0) >= 0.6
                                   or d.get("outcome", "") not in _HIGH_STAKES]
                        for d in v4_dets:
                            LOGGER.info(
                                "v4 outcome: %s confidence=%.3f at t=%.1f",
                                d.get("outcome"), d.get("confidence", 0), t,
                            )
                    if v4_dets:
                        best = max(v4_dets, key=lambda d: d.get("confidence", 0))
                        v4_outcomes_by_ts[t] = best["outcome"]
                        v4_outcome_detections.extend(
                            {**d, "timestamp": t} for d in v4_dets
                        )

                    # v5 outcome classifier fallback (returns None if not loaded)
                    if t not in v4_outcomes_by_ts:
                        v5_outcome = roboflow_detector.classify_outcome_v5(
                            frame, sport=sport,
                        )
                        if v5_outcome:
                            v4_outcomes_by_ts[t] = v5_outcome

                if v4_outcome_detections:
                    phases_used.append("v4_outcome_detection")

            layer_timings["v4_outcome_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success" if v4_outcomes_by_ts else "no_model_or_no_outcomes",
                "outcomes_found": len(v4_outcomes_by_ts),
                "detections": len(v4_outcome_detections),
            }
            LOGGER.info(
                "Pipeline: v4/v5 outcome found %d outcomes across %d timestamps",
                len(v4_outcome_detections), len(v4_outcomes_by_ts),
            )
        except Exception as exc:
            layer_timings["v4_outcome_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: v4 outcome detection failed (non-fatal): %s", exc)

        # ── Step 7.8: Scoreboard detection (score change → clip boost) ──
        scoreboard_detections: list[dict] = []
        score_change_timestamps: list[float] = []
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            if frames and roboflow_detector.scoreboard_detector_v5_model is not None:
                LOGGER.info("Pipeline: scoreboard detection running (every 5th frame)")
                prev_crop_hash = None
                for i in range(0, len(frames), 5):
                    t, frame = frames[i]
                    sb_dets = roboflow_detector.detect_scoreboard(frame)
                    if sb_dets:
                        scoreboard_detections.extend({**d, "timestamp": t} for d in sb_dets)
                        # Track score changes via pixel diff of scoreboard region
                        best = max(sb_dets, key=lambda d: d["confidence"])
                        bx1, by1, bx2, by2 = [int(c) for c in best["bbox"]]
                        crop = frame[max(0,by1):min(frame.shape[0],by2), max(0,bx1):min(frame.shape[1],bx2)]
                        if crop.size > 0:
                            crop_hash = hash(crop.tobytes()[:1000])
                            if prev_crop_hash is not None and crop_hash != prev_crop_hash:
                                score_change_timestamps.append(t)
                            prev_crop_hash = crop_hash

                if scoreboard_detections:
                    phases_used.append("scoreboard_detection")
                layer_timings["scoreboard_detection"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if scoreboard_detections else "no_detections",
                    "detections": len(scoreboard_detections),
                    "score_changes": len(score_change_timestamps),
                }
                LOGGER.info("Pipeline: scoreboard found %d detections, %d score changes",
                            len(scoreboard_detections), len(score_change_timestamps))
        except Exception as exc:
            layer_timings["scoreboard_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: scoreboard detection failed (non-fatal): %s", exc)

        # ── Free remaining frame pixel data ──
        # _frame_timestamps already saved before Ali (line ~668)
        # Clear any remaining references
        if frames:
            frames.clear()
        if live_frames:
            live_frames.clear()
        try:
            ocr_frames.clear()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        try:
            import psutil
            _rss = psutil.Process().memory_info().rss // (1024 * 1024)
            LOGGER.info("Pipeline: freed %d frames, RSS after gc=%dMB", _frame_count, _rss)
        except Exception:
            LOGGER.info("Pipeline: freed %d frames", _frame_count)

        # ── Step 8: Cross-layer validation + merge ────────────────────
        detection_points: list[DetectionPoint] = []
        cross_layer_agreements: list[dict] = []

        # Collect ALL detections by timestamp (within 0.5s buckets)
        all_layer_dets_raw: list[dict] = []
        for det in jersey_detections:
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": jersey_number,
                "layer": "ali_ensemble",
            })
        for det in universal_v2_detections:
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v2_universal_v1"),
            })
        for det in v3_ocr_detections:
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v3_ocr"),
            })
        for det in v2_sport_detections:
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v2_sport"),
            })
        for det in v3_primary_detections:
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v3_ocr_primary_standalone"),
            })
        for det in v5_ocr_detections:
            _entry: dict[str, Any] = {
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v5_ocr_universal"),
            }
            if det.get("player_bbox"):
                _entry["player_bbox"] = det["player_bbox"]
            all_layer_dets_raw.append(_entry)
        for det in v7_football_detections:
            _entry_v7: dict[str, Any] = {
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v7_football_ocr"),
            }
            if det.get("player_bbox"):
                _entry_v7["player_bbox"] = det["player_bbox"]
            all_layer_dets_raw.append(_entry_v7)

        # ── Diagnostic: log ALL numbers detected before filtering ─────
        all_numbers_seen = set(
            d.get("number_detected", "unknown") for d in all_layer_dets_raw
        )
        LOGGER.info(
            "Numbers detected across all layers (before filter): %s  "
            "(target jersey_number=%d)",
            all_numbers_seen, jersey_number,
        )
        if all_numbers_seen and jersey_number not in all_numbers_seen:
            LOGGER.warning(
                "Pipeline: target jersey #%d NOT FOUND in any layer. "
                "Detected numbers: %s",
                jersey_number, all_numbers_seen,
            )

        # ── Filter: keep only detections for the requested jersey_number ──
        all_layer_dets = [
            d for d in all_layer_dets_raw
            if d.get("number_detected") == jersey_number
        ]
        wrong_number_count = len(all_layer_dets_raw) - len(all_layer_dets)
        if wrong_number_count > 0:
            LOGGER.info(
                "Pipeline: filtered out %d detections for wrong jersey numbers "
                "(kept %d for #%d)",
                wrong_number_count, len(all_layer_dets), jersey_number,
            )

        # ── OCR-to-Track mapping (BLOCKER 4) ──────────────────────────
        # Map OCR detections to player tracks via IoU, then apply
        # temporal voting so each track gets a confirmed jersey number.
        ocr_track_assignments: dict[int, list[tuple[int, float]]] = {}  # track_id → [(jersey_number, conf), ...]
        if tracking_result and tracking_result.tracks:
            _ocr_dets_with_bbox = [
                d for d in all_layer_dets if d.get("player_bbox")
            ]
            if _ocr_dets_with_bbox:
                for det in _ocr_dets_with_bbox:
                    det_ts = det.get("timestamp", 0)
                    det_bbox = det["player_bbox"]  # [x1, y1, x2, y2] in frame space
                    best_iou = 0.0
                    best_track_id = None
                    for track in tracking_result.tracks:
                        if not track.positions:
                            continue
                        # Find track's position closest to this detection's timestamp
                        # Approximate: frame_index ≈ timestamp * 2 (2fps extraction)
                        approx_frame = int(det_ts * 2)
                        # Use last known position if frame doesn't match exactly
                        pos_idx = min(approx_frame, len(track.positions) - 1)
                        pos_idx = max(0, pos_idx)
                        if pos_idx < len(track.positions):
                            tx1, ty1, tx2, ty2 = track.positions[pos_idx]
                        else:
                            tx1, ty1, tx2, ty2 = track.positions[-1]
                        # Compute IoU
                        ix1 = max(det_bbox[0], tx1)
                        iy1 = max(det_bbox[1], ty1)
                        ix2 = min(det_bbox[2], tx2)
                        iy2 = min(det_bbox[3], ty2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                            trk_area = (tx2 - tx1) * (ty2 - ty1)
                            union = det_area + trk_area - inter
                            iou = inter / union if union > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_track_id = track.track_id
                    if best_track_id is not None and best_iou > 0.1:
                        if best_track_id not in ocr_track_assignments:
                            ocr_track_assignments[best_track_id] = []
                        ocr_track_assignments[best_track_id].append((
                            det.get("number_detected", 0),
                            det.get("confidence", 0),
                        ))

                # Temporal voting: assign majority jersey number to each track
                for track in tracking_result.tracks:
                    votes = ocr_track_assignments.get(track.track_id, [])
                    if not votes:
                        continue
                    # Count votes per jersey number, weighted by confidence
                    from collections import Counter
                    vote_counts: dict[int, float] = {}
                    for num, conf in votes:
                        vote_counts[num] = vote_counts.get(num, 0) + conf
                    # Winner = highest weighted vote count
                    winner = max(vote_counts, key=lambda k: vote_counts[k])
                    track.jersey_number = winner
                    track.jersey_confidence = vote_counts[winner] / len(votes)
                    if winner == jersey_number:
                        track.is_target = True
                        tracking_result.target_track_id = track.track_id

                _tracks_with_jersey = sum(
                    1 for t in tracking_result.tracks if t.jersey_number is not None
                )
                LOGGER.info(
                    "Pipeline: OCR→Track mapping: %d OCR dets with bbox, "
                    "%d tracks assigned jersey numbers, target_track=%s",
                    len(_ocr_dets_with_bbox), _tracks_with_jersey,
                    tracking_result.target_track_id,
                )
            layer_timings["ocr_track_mapping"] = {
                "tracks_with_ocr": len(ocr_track_assignments),
                "total_ocr_votes": sum(len(v) for v in ocr_track_assignments.values()),
                "target_track_id": tracking_result.target_track_id,
            }

        # Group by timestamp bucket (0.5s window)
        from collections import defaultdict
        ts_buckets: dict[float, list[dict]] = defaultdict(list)
        for det in all_layer_dets:
            ts = det["timestamp"]
            # Find existing bucket within 0.5s
            bucket_key = None
            for existing_ts in ts_buckets:
                if abs(existing_ts - ts) < 0.5:
                    bucket_key = existing_ts
                    break
            if bucket_key is None:
                bucket_key = ts
            ts_buckets[bucket_key].append(det)

        # Cross-validate each bucket
        for bucket_ts, bucket_dets in sorted(ts_buckets.items()):
            layers_present = set(d["layer"] for d in bucket_dets)
            best_conf = max(d["confidence"] for d in bucket_dets)

            # Cross-layer confidence boosts (v5 PRIMARY, Ali low trust)
            bonus = 0.0
            high_confidence = False
            has_v5 = any("v5_ocr" in l for l in layers_present)
            has_v3 = any("v3_ocr_primary" in l for l in layers_present)
            has_v2 = any("v2_universal" in l for l in layers_present)
            has_ali = "ali_ensemble" in layers_present

            # v5 agreement is highest trust
            if has_v5 and has_v3:
                bonus += 0.20
            elif has_v5 and has_v2:
                bonus += 0.15
            elif has_v3 and has_v2:
                bonus += 0.10

            # Ali agreement is LOW trust
            if has_ali:
                if len(layers_present) >= 3:
                    bonus += 0.05  # Ali confirms others — small bonus
                # Ali alone — cap confidence, don't trust it
                # (handled below when computing final_conf)

            # Multi-layer bonus (3+ layers agree = very high confidence)
            if len(layers_present) >= 3:
                bonus += 0.10
                high_confidence = True

            final_conf = min(1.0, best_conf + bonus)

            # Ali-alone hard cap
            if has_ali and len(layers_present) == 1:
                final_conf = min(0.6, best_conf)
                bonus = 0.0

            if len(layers_present) >= 2:
                cross_layer_agreements.append({
                    "timestamp": round(bucket_ts, 1),
                    "number_detected": jersey_number,
                    "layers_agreed": sorted(layers_present),
                    "final_confidence": round(final_conf, 3),
                    "high_confidence": high_confidence,
                })

            # Find nearest v4 outcome for this timestamp
            bucket_v4_outcome = ""
            if v4_outcomes_by_ts:
                for ts, outcome in v4_outcomes_by_ts.items():
                    if abs(ts - bucket_ts) < 2.0:
                        bucket_v4_outcome = outcome
                        break

            detection_points.append(DetectionPoint(
                timestamp=bucket_ts,
                confidence=final_conf,
                jersey_visible=True,
                jersey_number=jersey_number,
                motion_score=motion_scores.get(bucket_ts, _nearest_value(motion_scores, bucket_ts)),
                pose_action=pose_results.get(bucket_ts, _nearest_pose(pose_results, bucket_ts)).get("action", "standing") if pose_results else "standing",
                crowd_energy=_get_crowd_energy(audio_result, bucket_ts),
                tracking_id=tracking_result.target_track_id if tracking_result else None,
                v4_outcome=bucket_v4_outcome,
            ))

        # ── Temporal consensus filtering ─────────────────────────────
        # Apply temporal consensus to filter detection_points down to
        # only temporally-confirmed detections.
        tc_stats = {"raw_detections": 0, "confirmed_detections": 0,
                    "filtered_out": 0, "cross_layer_confirmed": 0}
        try:
            from app.services.temporal_consensus import TemporalConsensus

            # Relaxed temporal consensus — 2 confirmations in 3s window
            # Aggressive mode (low quality or dark jerseys) → even more relaxed
            # Also relax when few detections (<10) to avoid filtering them all out
            # Full games (>1800s): always relaxed — frames are sparse (1fps),
            # requiring 2 confirmations in 3s is too aggressive
            _few_detections = len(all_layer_dets) < 10
            _is_full_game_tc = video_duration > 1800
            if resolved_quality == "aggressive" or _is_dark_jersey or _few_detections or _is_full_game_tc:
                # Match OCR confidence threshold: navy=0.08, dark=0.12.
                # Previous 0.15 threshold killed valid navy detections.
                _tc_conf = ocr_conf if _is_dark_jersey else (0.12 if _is_full_game_tc else 0.15)
                tc_instance = TemporalConsensus(
                    min_confirmations=1,
                    time_window=5.0 if _is_full_game_tc else 4.0,
                    confidence_threshold=_tc_conf,
                )
            else:
                tc_instance = TemporalConsensus(
                    min_confirmations=2,
                    time_window=3.0,
                    confidence_threshold=0.3,
                )

            if all_layer_dets:
                # adaptive=False: pipeline already handles its own adaptive
                # logic (dark jersey, few detections, full game) above.
                confirmed_dets = tc_instance.filter_detections(
                    all_layer_dets, jersey_number, adaptive=False
                )
                confirmed_dets = tc_instance.cross_layer_boost(
                    confirmed_dets
                )
                tc_stats = {
                    "raw_detections": len(all_layer_dets),
                    "confirmed_detections": len(confirmed_dets),
                    "filtered_out": len(all_layer_dets) - len(confirmed_dets),
                    "cross_layer_confirmed": sum(
                        1 for d in confirmed_dets
                        if d.get("cross_layer_confirmed")
                    ),
                }
                LOGGER.info(
                    "Pipeline: temporal consensus — %d raw → %d confirmed "
                    "(%d filtered, %d cross-layer)",
                    tc_stats["raw_detections"],
                    tc_stats["confirmed_detections"],
                    tc_stats["filtered_out"],
                    tc_stats["cross_layer_confirmed"],
                )

                # Boost (don't filter) detection_points near confirmed timestamps.
                # Previously this REMOVED detection_points not near jersey sightings,
                # which threw away valid motion/pose/audio detections and caused
                # only 4 clips from 288 detections.  Now we keep ALL detection_points
                # but boost confidence of those near confirmed jersey timestamps.
                if confirmed_dets:
                    confirmed_timestamps = set()
                    for cd in confirmed_dets:
                        confirmed_timestamps.add(cd.get("timestamp", 0))
                    boosted_count = 0
                    for dp in detection_points:
                        near_confirmed = any(
                            abs(dp.timestamp - cts) < 2.0
                            for cts in confirmed_timestamps
                        )
                        if near_confirmed and not dp.jersey_visible:
                            dp.confidence = min(1.0, dp.confidence + 0.1)
                            boosted_count += 1
                    LOGGER.info(
                        "Pipeline: temporal consensus boosted %d of %d "
                        "detection_points (kept all, %d confirmed timestamps)",
                        boosted_count, len(detection_points),
                        len(confirmed_timestamps),
                    )
                else:
                    # Consensus found 0 confirmed — keep detection_points anyway
                    # The jitter filter in clip_extractor handles false positives
                    LOGGER.warning(
                        "Pipeline: temporal consensus confirmed 0 of %d raw — "
                        "keeping %d detection_points (jitter filter will handle FPs)",
                        len(all_layer_dets), len(detection_points),
                    )

        except Exception as exc:
            LOGGER.warning("Pipeline: temporal consensus failed (non-fatal): %s", exc)

        # Log detection source breakdown
        ali_count = len(jersey_detections)
        univ_count = len(universal_v2_detections)
        v3_count = len(v3_ocr_detections)
        v2_sport_count = len(v2_sport_detections)
        v3_primary_count = len(v3_primary_detections)
        v5_ocr_count = len(v5_ocr_detections)
        v4_outcome_count = len(v4_outcome_detections)
        LOGGER.info(
            "Layer results — Ali: %d, Universal_v1: %d, V3_ocr: %d, "
            "V2_sport: %d, V3_primary: %d, V5_ocr: %d, V4_outcome: %d, After_filter: %d",
            ali_count, univ_count, v3_count, v2_sport_count,
            v3_primary_count, v5_ocr_count, v4_outcome_count, len(detection_points),
        )
        if ali_count == 0 and (univ_count + v3_count + v2_sport_count + v3_primary_count + v5_ocr_count) > 0:
            LOGGER.info("Pipeline: Ali found 0 — other layers saved detection!")
        total_raw = ali_count + univ_count + v3_count + v2_sport_count + v3_primary_count + v5_ocr_count
        if total_raw > 0 and len(detection_points) == 0:
            LOGGER.warning(
                "Pipeline: %d raw detections across all layers → 0 detection_points "
                "after jersey number filter + temporal consensus.",
                total_raw,
            )

        # ── Motion supplement: if we have detections but they cluster into few
        # clips, add high-motion frames as supplementary points.  The player IS
        # in the video (confirmed by OCR), so motion peaks are likely their plays.
        #
        # SKIP for football — football uses cadence-based supplement below,
        # which is more reliable at 640x360 resolution.  Motion supplement adds
        # many low-quality points (conf 0.05-0.25) that inflate detection_points
        # count, prevent cadence from firing, then get jitter-filtered anyway.
        #
        # Uses PERCENTILE-BASED threshold instead of fixed values, so it adapts
        # to low-resolution video (640x360 from WARP has much lower optical flow
        # scores than 720p — fixed threshold of 15 misses everything).
        _supplement_limit = 60 if video_duration > 1800 else 20
        _skip_motion_supp = sport.lower() == "football"  # football uses cadence
        if not _skip_motion_supp and 1 <= len(detection_points) <= _supplement_limit and _frame_timestamps:
            _existing_ts = {dp.timestamp for dp in detection_points}
            _is_football_supp = sport.lower() == "football"

            # Percentile-based threshold: top 30% of motion scores for football,
            # top 20% for other sports. Falls back to fixed minimum if percentile
            # is too low (video is entirely static).
            _all_motions = sorted(
                [motion_scores.get(t, 0) for t in _frame_timestamps],
                reverse=True,
            )
            if _all_motions:
                _pct_idx = max(1, int(len(_all_motions) * (0.30 if _is_football_supp else 0.20)))
                _pct_thresh = _all_motions[min(_pct_idx, len(_all_motions) - 1)]
                _motion_thresh = max(5.0 if _is_football_supp else 15.0, _pct_thresh)
            else:
                _motion_thresh = 15 if _is_football_supp else 30

            LOGGER.info("Pipeline: motion supplement threshold=%.1f (percentile-based, "
                        "top scores: %s)", _motion_thresh,
                        [round(m, 1) for m in _all_motions[:5]])

            _supplement_count = 0
            for t in _frame_timestamps:
                if t in _existing_ts:
                    continue
                _skip_gap = 1.5 if (video_duration > 1800 or _is_football_supp) else 3.0
                if any(abs(t - ets) < _skip_gap for ets in _existing_ts):
                    continue
                motion = motion_scores.get(t, 0)
                in_boundary = _in_audio_boundary(audio_result, t)
                if motion >= _motion_thresh or (in_boundary and motion > _motion_thresh * 0.6):
                    pose = pose_results.get(t, _nearest_pose(pose_results, t)) if pose_results else {}
                    conf = motion / 100.0 * 0.5
                    if in_boundary:
                        conf = min(0.8, conf + 0.1)
                    detection_points.append(DetectionPoint(
                        timestamp=t,
                        confidence=conf,
                        jersey_visible=False,
                        jersey_number=None,
                        motion_score=motion,
                        pose_action=pose.get("action", "standing"),
                        crowd_energy=_get_crowd_energy(audio_result, t),
                    ))
                    _existing_ts.add(t)
                    _supplement_count += 1
            if _supplement_count:
                LOGGER.info("Pipeline: motion supplement added %d high-motion points "
                            "(total detection_points now %d)", _supplement_count, len(detection_points))

        # ── Supplement detection points when OCR found too few for good clips.
        # For football: use cadence-based supplement (1 play every ~30s).
        # For other sports: motion-based play segmentation + fallback.
        _dp_before_supplement = len(detection_points)
        is_football = sport.lower() == "football"

        if is_football and _dp_before_supplement < 30 and video_duration > 30 and _frame_timestamps:
            # Football cadence supplement — PRIMARY strategy for football.
            # Motion scores at 640x360 are too low (0-10 range) for threshold-
            # based detection. Instead, leverage football's predictable rhythm:
            # 1 play every 25-40 seconds (snap → whistle → huddle → snap).
            # Use 22s cadence to target 20+ clips for 600s video (28 cadence
            # points + OCR = ~35, minus merging = 20-25 clips).
            _existing_play_ts = {dp.timestamp for dp in detection_points}
            _cadence = 22.0
            _cadence_added = 0
            _t_cursor = 5.0
            while _t_cursor < video_duration - 5.0:
                best_t = None
                best_motion = -1
                for ft in _frame_timestamps:
                    if abs(ft - _t_cursor) < _cadence / 2:
                        m = motion_scores.get(ft, 0)
                        if m > best_motion and ft not in _existing_play_ts:
                            best_t = ft
                            best_motion = m
                if best_t is not None:
                    pose = pose_results.get(best_t, _nearest_pose(pose_results, best_t)) if pose_results else {}
                    in_boundary = _in_audio_boundary(audio_result, best_t)
                    conf = max(0.55, best_motion / 100.0 * 0.8)
                    if in_boundary:
                        conf = min(0.9, conf + 0.15)
                    detection_points.append(DetectionPoint(
                        timestamp=best_t,
                        confidence=conf,
                        jersey_visible=False,
                        motion_score=best_motion,
                        pose_action=pose.get("action", "standing"),
                        crowd_energy=_get_crowd_energy(audio_result, best_t),
                    ))
                    _existing_play_ts.add(best_t)
                    _cadence_added += 1
                _t_cursor += _cadence
            if _cadence_added:
                LOGGER.info("Pipeline: football cadence supplement added %d points "
                            "every %.0fs (total now %d)",
                            _cadence_added, _cadence, len(detection_points))

        elif not is_football and _dp_before_supplement < 15 and _frame_timestamps:
            # Non-football: play segmentation + motion/audio fallback
            _existing_play_ts = {dp.timestamp for dp in detection_points}

            if motion_scores:
                from app.services.clip_extractor import segment_plays_from_motion
                play_segments = segment_plays_from_motion(motion_scores, sport)
                if play_segments:
                    _play_added = 0
                    for seg_start, seg_end in play_segments:
                        seg_mid = (seg_start + seg_end) / 2
                        if any(abs(seg_mid - ets) < 2.0 for ets in _existing_play_ts):
                            continue
                        seg_motions = [
                            motion_scores.get(t, 0) for t in _frame_timestamps
                            if seg_start <= t <= seg_end
                        ]
                        peak_motion = max(seg_motions) if seg_motions else 0
                        pose = pose_results.get(seg_mid, _nearest_pose(pose_results, seg_mid)) if pose_results else {}
                        in_boundary = _in_audio_boundary(audio_result, seg_mid)
                        conf = min(0.9, peak_motion / 100.0 * 0.8)
                        if in_boundary:
                            conf = min(1.0, conf + 0.15)
                        detection_points.append(DetectionPoint(
                            timestamp=seg_mid,
                            confidence=conf,
                            jersey_visible=False,
                            motion_score=peak_motion,
                            pose_action=pose.get("action", "standing"),
                            crowd_energy=_get_crowd_energy(audio_result, seg_mid),
                        ))
                        _existing_play_ts.add(seg_mid)
                        _play_added += 1
                    if _play_added:
                        LOGGER.info("Pipeline: play segmentation added %d points "
                                    "(total now %d)", _play_added, len(detection_points))

            # Motion/audio fallback (if still too few points)
            if len(detection_points) < 10:
                motion_threshold = 30
                LOGGER.info("Pipeline: low detections (%d), using motion/audio fallback "
                            "(threshold=%d, sport=%s)",
                            len(detection_points), motion_threshold, sport)
                for t in _frame_timestamps:
                    if t in _existing_play_ts:
                        continue
                    if any(abs(t - ets) < 1.5 for ets in _existing_play_ts):
                        continue
                    motion = motion_scores.get(t, 0)
                    in_boundary = _in_audio_boundary(audio_result, t)
                    if motion > motion_threshold or in_boundary:
                        pose = pose_results.get(t, _nearest_pose(pose_results, t)) if pose_results else {}
                        conf = motion / 100.0 * 0.7
                        if in_boundary:
                            conf = min(1.0, conf + 0.15)
                        fallback_v4 = ""
                        if v4_outcomes_by_ts:
                            for ts_key, outcome in v4_outcomes_by_ts.items():
                                if abs(ts_key - t) < 2.0:
                                    fallback_v4 = outcome
                                    break
                        detection_points.append(DetectionPoint(
                            timestamp=t,
                            confidence=conf,
                            jersey_visible=False,
                            motion_score=motion,
                            pose_action=pose.get("action", "standing"),
                            crowd_energy=_get_crowd_energy(audio_result, t),
                            v4_outcome=fallback_v4,
                        ))
                        _existing_play_ts.add(t)

        _dp_after_supplement = len(detection_points)
        _supplement_added = _dp_after_supplement - _dp_before_supplement
        if _supplement_added > 0:
            LOGGER.info("Pipeline: motion/play supplement added %d points "
                        "(%d → %d detection_points)",
                        _supplement_added, _dp_before_supplement, _dp_after_supplement)
        LOGGER.info("Pipeline: total detection_points before clip extraction: %d",
                    _dp_after_supplement)

        # Extract and rank clips
        clips = extract_clips(
            detections=detection_points,
            audio_result=audio_result if audio_result.has_audio else None,
            sport=sport,
            position=position,
            video_duration=video_duration,
        )

        # ── Unload request-specific models to free memory ──
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.unload_request_models()
            LOGGER.info("Pipeline: unloaded request models after inference")
        except Exception:
            pass

        # Clip splitting REMOVED in v8.4 — clip_extractor now handles hard caps
        # and aggressive merging. Splitting 50 clips into 117 sub-clips was
        # destroying the UX. Clips are already capped at MAX_CLIPS_PER_GAME=30.
        # Just enforce max duration per clip.
        _MAX_CLIP_DUR = {"football": FOOTBALL_MAX_CLIP, "basketball": 12.0, "lacrosse": 15.0}
        _max_dur = _MAX_CLIP_DUR.get(sport.lower(), 15.0)
        for clip in clips:
            if clip.end_time - clip.start_time > _max_dur:
                clip.end_time = clip.start_time + _max_dur
        LOGGER.info("Pipeline: %d clips (max_dur=%.0fs, no splitting)", len(clips), _max_dur)

        # ── Temporal jersey attribution (standard path) ──
        _jersey_ts_std = sorted(
            d["timestamp"] for d in all_layer_dets
            if d.get("number_detected") == jersey_number
        ) if all_layer_dets else []
        if _jersey_ts_std and clips:
            _attr_count_std = 0
            for clip in clips:
                if clip.jersey_visible:
                    continue
                for jts in _jersey_ts_std:
                    if clip.start_time - 15 <= jts <= clip.end_time + 15:
                        clip.jersey_visible = True
                        clip.jersey_number_seen = jersey_number
                        _attr_count_std += 1
                        break
            if _attr_count_std:
                LOGGER.info("Pipeline: temporal jersey attribution: %d clips gained jersey=%d",
                            _attr_count_std, jersey_number)

        # ── Step 9: Stat generation pipeline ───────────────────────────
        stat_result: dict = {"game_stats": {}, "per_clip_stats": [], "actions_detected": []}
        t0 = time.perf_counter()
        try:
            from app.services.stat_pipeline import run_stat_pipeline
            clips_as_dicts = [
                {"startTime": c.start_time, "endTime": c.end_time, "grade": c.grade}
                for c in clips
            ]
            stat_result = run_stat_pipeline(
                frames=[],  # frames freed after OCR to save memory
                sport=sport,
                jersey_number=jersey_number,
                position=position,
                pose_results=pose_results if pose_results else None,
                motion_scores=motion_scores if motion_scores else None,
                audio_result=audio_result if audio_result.has_audio else None,
                clips=clips_as_dicts,
            )
            if stat_result.get("actions_detected"):
                phases_used.append("stat_generation")
            layer_timings["stat_generation"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success",
                "actions_found": len(stat_result.get("actions_detected", [])),
            }
            LOGGER.info("Pipeline: stat generation found %d actions", len(stat_result.get("actions_detected", [])))
        except Exception as exc:
            layer_timings["stat_generation"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
            LOGGER.warning("Pipeline: stat generation failed (non-fatal): %s", exc)

        elapsed = time.perf_counter() - start_time

        # Build response
        layer_used = "+".join(phases_used) if phases_used else "none"

        # Audio events for response
        audio_events_out = []
        if audio_result.has_audio:
            for evt in audio_result.events[:50]:  # Cap at 50
                audio_events_out.append({
                    "timestamp": evt.timestamp,
                    "eventType": evt.event_type,
                    "confidence": round(evt.confidence, 2),
                })

        # Player tracks for response
        player_tracks_out = []
        if tracking_result:
            for track in tracking_result.tracks[:20]:  # Cap at 20
                player_tracks_out.append({
                    "trackId": track.track_id,
                    "jerseyNumber": track.jersey_number,
                    "framesTracked": track.frames_tracked,
                })

        # Convert clips to response format
        clips_out = []
        for clip in clips:
            clip_dict: dict[str, Any] = {
                "startTime": clip.start_time,
                "endTime": clip.end_time,
                "confidence": clip.confidence,
                "score": clip.score,
                "playType": clip.play_type,
                "grade": clip.grade,
                "jerseyVisible": clip.jersey_visible,
                "jerseyNumberSeen": clip.jersey_number_seen,
                "trackingId": clip.tracking_id,
                "description": clip.description,
                "signals": clip.signals,
            }
            # Include v4_outcome from signals if present
            v4_out = (clip.signals or {}).get("v4_outcome")
            if v4_out:
                clip_dict["v4Outcome"] = v4_out
            # Rule-based play type supplement (when v4 models didn't detect)
            if not v4_out and clip_dict["playType"] == "game_action":
                rule_type = _detect_play_type_rules(
                    motion_score=(clip.signals or {}).get("motion", 0) or 0,
                    pose=(clip.signals or {}).get("pose", "standing"),
                    crowd_energy=(clip.signals or {}).get("crowd", 0) or 0,
                    jersey_visible=clip.jersey_visible,
                    sport=sport,
                )
                if rule_type != "game_action":
                    clip_dict["playType"] = rule_type
                    clip_dict["description"] = rule_type.replace("_", " ").title()
            # Include enriched fields for frontend
            clip_dict["deadBallRatio"] = round(dead_ball_ratio, 2)
            clip_dict["scoreboardDetected"] = len(scoreboard_detections) > 0
            # Which detection layers found the jersey in this clip's time range
            clip_layers = set()
            for det in all_layer_dets:
                if clip.start_time - 1 <= det.get("timestamp", 0) <= clip.end_time + 1:
                    clip_layers.add(det.get("layer", "unknown"))
            clip_dict["detectionLayers"] = sorted(clip_layers)
            # Caption + recruiting score + game clock for coach-friendly display
            clip_dict["caption"] = _generate_clip_caption(clip_dict, sport, position, jersey_number)
            clip_dict["recruitingScore"] = _compute_recruiting_score(clip_dict, sport, position)
            clip_dict["estimatedQuarter"] = _estimate_game_quarter(clip.start_time, video_duration, sport)
            clips_out.append(clip_dict)

        # Sort clips in highlight reel order (TDs first, then big plays, etc.)
        clips_out.sort(key=_get_highlight_sort_key)

        # Add sequenceNote to each clip
        for i, clip_dict in enumerate(clips_out):
            prev = clips_out[i - 1] if i > 0 else None
            clip_dict["sequenceNote"] = _generate_sequence_note(
                clip_dict, i, len(clips_out), prev, sport,
            )

        # ── Build debug field ──────────────────────────────────────────
        ali_working = ali_status == "working"
        layers_that_contributed = [
            layer for layer in phases_used
            if layer not in ("youtube_download", "frame_extraction")
        ]
        # Determine primary detection layer
        try:
            from app.services.roboflow_detector import roboflow_detector
            primary_layer = roboflow_detector.get_primary_detection()
        except Exception:
            primary_layer = "unknown"

        # Get per-request model tracking
        request_summary = {}
        try:
            from app.services.roboflow_detector import roboflow_detector
            request_summary = roboflow_detector.get_request_summary()
            LOGGER.info("Pipeline: request_summary = %s", request_summary)
        except Exception as exc:
            LOGGER.warning("Pipeline: get_request_summary failed: %s", exc)

        # Fallback: build models_called from layer_timings when detector tracking is empty
        if not request_summary.get("models_called"):
            _mc: list[str] = []
            _dpm: dict[str, int] = {}
            _layer_model_map = {
                "dead_ball_filter": ("dead_ball_classifier_v5", "dead_frames"),
                "v5_ocr_detection": ("jersey_ocr_universal_v5", "detections"),
                "v7_football_ocr": ("football_jersey_ocr_v7", "detections"),
                "v3_ocr_detection": ("v3_ocr", "detections"),
                "universal_v2_ocr": ("universal_v2_ocr", "detections"),
                "v2_sport_detection": ("v2_sport_detector", "v2_sport_detections"),
                "ali_jersey_detection": ("ali_ensemble", "detections"),
                "v4_outcome_detection": ("v4_outcome_classifier", "detections"),
                "football_fullframe_ocr": ("football_fullframe_ocr", "detections"),
            }
            # Add player_detector_v5 if v5 OCR ran
            if layer_timings.get("v5_ocr_detection", {}).get("players_found", 0) > 0:
                _mc.append("player_detector_v5")
                _dpm["player_detector_v5"] = layer_timings["v5_ocr_detection"]["players_found"]
            for layer_key, (model_name, count_key) in _layer_model_map.items():
                lt = layer_timings.get(layer_key, {})
                if lt.get("status") in ("success", "no_detections", "no_model_or_no_detections", "no_model_or_no_outcomes"):
                    _mc.append(model_name)
                    _dpm[model_name] = lt.get(count_key, 0)
            request_summary = {"models_called": _mc, "detections_per_model": _dpm}
            LOGGER.info("Pipeline: built models_called from layer_timings: %s", _mc)

        # Memory info
        memory_rss_mb = 0
        try:
            import psutil
            memory_rss_mb = round(psutil.Process().memory_info().rss / 1024 / 1024)
        except Exception:
            pass

        # ── Detection summary log ──────────────────────────────────────
        target_found = jersey_number in all_numbers_seen
        LOGGER.info("=== DETECTION SUMMARY ===")
        LOGGER.info("Sport: %s, Jersey: #%d, Color: %s", sport, jersey_number, jersey_color)
        LOGGER.info("Frames processed: %d", len(frames))
        LOGGER.info("Models called this request: %s", request_summary.get("models_called", []))
        LOGGER.info("Detections per model:")
        for model_name, count in request_summary.get("detections_per_model", {}).items():
            LOGGER.info("  %s: %d detections", model_name, count)
        LOGGER.info("Numbers detected: %s", sorted(all_numbers_seen))
        LOGGER.info("Target number #%d found: %s", jersey_number, target_found)
        LOGGER.info("Clips before filter: %d", len(clips))
        LOGGER.info("Clips after filter: %d", len(clips_out))
        LOGGER.info("Final clips: %d", len(clips_out))
        LOGGER.info("Memory RSS: %dMB", memory_rss_mb)
        LOGGER.info("=========================")

        debug = {
            "primary_layer": primary_layer,
            "ali_detections": len(jersey_detections),
            "universal_v2_detections": len(universal_v2_detections),
            "v3_ocr_detections": len(v3_ocr_detections),
            "v2_sport_detections": len(v2_sport_detections),
            "v3_primary_detections": len(v3_primary_detections),
            "v5_ocr_detections": len(v5_ocr_detections),
            "v7_football_detections": len(v7_football_detections),
            "v7_navy_detections": v7_navy_detections,
            "v7_player_crops": v7_player_crops,
            "v4_outcome_detections": len(v4_outcome_detections),
            "v4_outcomes_found": len(v4_outcomes_by_ts),
            "combined_detections": len(detection_points),
            "numbers_detected": sorted(str(n) for n in all_numbers_seen),
            "target_jersey_number": jersey_number,
            "target_number_found": target_found,
            "frames_with_target": sum(1 for d in all_layer_dets if d.get("number_detected") == jersey_number),
            "wrong_number_filtered": wrong_number_count,
            "analyze_fps": _actual_fps_used,
            "frames_extracted": frames_processed,
            "dead_ball_frames_skipped": dead_ball_count,
            "dead_ball_ratio": round(dead_ball_ratio, 2),
            "scoreboard_detections": len(scoreboard_detections),
            "score_changes_detected": len(score_change_timestamps),
            "quality_mode": quality_mode,
            "resolved_quality": resolved_quality,
            "ocr_confidence": ocr_conf,
            "ali_working": ali_status in ("working_fallback",),
            "ali_status": ali_status,
            "youtube_strategy_used": youtube_strategy_used,
            "total_elapsed_ms": round(elapsed * 1000),
            "layers_that_contributed": layers_that_contributed,
            "layer_breakdown": layer_timings,
            "temporal_consensus": tc_stats,
            "detection_points_total": _dp_after_supplement,
            "detection_points_from_ocr": _dp_before_supplement,
            "detection_points_from_supplement": _supplement_added,
            "cross_layer_agreements": cross_layer_agreements,
            "models_called": request_summary.get("models_called", []),
            "detections_per_model": request_summary.get("detections_per_model", {}),
            "ocr_track_mapping": layer_timings.get("ocr_track_mapping", {}),
            "clips_before_filter": len(clips),
            "clips_after_filter": len(clips_out),
            "memory_rss_mb": memory_rss_mb,
        }

        # Player summary for coaches
        player_summary = _build_player_summary(
            clips_out, jersey_number, sport, position, video_duration, elapsed,
        )

        return {
            "playerSummary": player_summary,
            "clips": clips_out,
            "layerUsed": layer_used,
            "elapsed": round(elapsed, 1),
            "videoDuration": round(video_duration, 1),
            "framesProcessed": frames_processed,
            "audioEvents": audio_events_out,
            "playerTracks": player_tracks_out,
            "gameStats": stat_result.get("game_stats", {}),
            "perClipStats": stat_result.get("per_clip_stats", []),
            "actionsDetected": stat_result.get("actions_detected", []),
            "debug": debug,
        }

    finally:
        # ── Post-request cleanup (Blocker 3) ──
        # Unload ALL models and force garbage collection.
        elapsed_total = time.perf_counter() - start_time
        _force_cleanup_memory()
        rss_end = _get_rss_mb()
        LOGGER.info(
            "Pipeline: REQUEST END — Memory delta: +%.0fMB (start=%.0fMB end=%.0fMB) elapsed=%.1fs",
            rss_end - rss_start, rss_start, rss_end, elapsed_total,
        )
        # Cleanup temp files
        if tmp_dir and tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        if local_video_path and local_video_path.parent.name.startswith("clipt_"):
            try:
                shutil.rmtree(local_video_path.parent, ignore_errors=True)
            except Exception:
                pass


async def _run_chunked_full_game(
    *,
    local_video_path: Path | None,
    video_url: str | None = None,
    video_duration: float,
    jersey_number: int,
    jersey_color: str,
    sport: str,
    position: str | None,
    extract_start: float,
    extract_end: float,
    enable_audio: bool,
    quality_mode: str,
    youtube_strategy_used: str | None,
    layer_timings: dict,
    start_time: float,
    phases_used: list[str],
    cancel_event: "threading.Event | None" = None,
) -> dict[str, Any]:
    """Process full game video in 30-minute chunks to avoid OOM.

    If video_url is provided and local_video_path is None, downloads each
    30-min chunk separately (avoids 2hr+ downloads that timeout/403).
    Each chunk: download → extract ~400 frames → run OCR → collect detections → free.
    After all chunks: merge detections → temporal consensus → clip extraction.
    Memory stays under ~2.5GB per chunk (vs 3.5GB+ for full game at once).
    """
    import gc
    from collections import defaultdict
    from app.services.roboflow_detector import roboflow_detector, is_dark_color, is_navy

    # Per-chunk download uses 10-min chunks (fits in WARP 300s timeout at 720p)
    # Pre-downloaded video uses 30-min chunks (no download timeout concern)
    _per_chunk_download = (video_url is not None and local_video_path is None)
    CHUNK_SIZE = 600 if _per_chunk_download else 1800
    CHUNK_MAX_FRAMES = 250  # Balanced: enough for play detection, fast enough for timeout
    _is_football = sport.lower() == "football"
    _is_dark = is_dark_color(jersey_color)
    _is_navy_jersey = is_navy(jersey_color)

    # OCR confidence threshold
    if _is_navy_jersey:
        ocr_conf = 0.08  # Navy jerseys need very low threshold (dark + reflective)
    elif _is_dark:
        ocr_conf = 0.12
    elif _is_football:
        ocr_conf = FOOTBALL_CONF_THRESHOLD
    elif quality_mode == "aggressive":
        ocr_conf = 0.15
    else:
        ocr_conf = 0.18

    # ── Audio analysis (one-time, on full video) ──
    audio_result = AudioAnalysisResult(has_audio=False)
    if enable_audio and local_video_path and local_video_path.exists():
        t0 = time.perf_counter()
        try:
            from app.services.audio_analyzer import analyze_audio
            from app.services.detection_runtime import PipelineSettings
            settings = PipelineSettings()
            audio_path = extract_audio(local_video_path, settings.ffmpeg_binary)
            if audio_path:
                audio_result = analyze_audio(audio_path)
                if audio_result.has_audio:
                    phases_used.append("audio_analysis")
                layer_timings["audio_analysis"] = {
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "status": "success" if audio_result.has_audio else "no_audio",
                    "events": len(audio_result.events),
                    "boundaries": len(audio_result.play_boundaries),
                }
        except Exception as exc:
            layer_timings["audio_analysis"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "error", "error": str(exc)[:200],
            }
            LOGGER.warning("Chunked: audio analysis failed: %s", exc)

    # ── Load models once for all chunks (skip v4 — not used in chunked mode) ──
    try:
        roboflow_detector.reset_request_tracking()
        roboflow_detector._request_jersey_color = jersey_color
        request_models = roboflow_detector.load_for_request(sport, jersey_color)
        LOGGER.info("Chunked: loaded %d models", len(request_models))
        # Immediately unload v4 models — chunked mode uses rule-based play detection
        for _attr in dir(roboflow_detector):
            if "_v4" in _attr and _attr.endswith("_model"):
                if getattr(roboflow_detector, _attr, None) is not None:
                    setattr(roboflow_detector, _attr, None)
        gc.collect()
    except Exception as exc:
        LOGGER.warning("Chunked: load_for_request failed: %s", exc)

    # ── Process chunks ──
    all_ocr_dets: list[dict] = []
    all_v7_dets: list[dict] = []
    all_v4_dets: list[dict] = []
    all_motion: dict[float, float] = {}
    all_frame_timestamps: list[float] = []
    total_frames = 0
    total_dead = 0
    chunk_count = 0
    _chunked_ocr_start = time.perf_counter()

    _chunk_video_path = local_video_path  # Will be overwritten per chunk if downloading

    chunk_start = extract_start
    while chunk_start < extract_end:
        # Check for cancellation (client disconnected)
        if cancel_event is not None and cancel_event.is_set():
            LOGGER.warning("Pipeline: CANCELLED by client disconnect at chunk %d", chunk_count + 1)
            break
        chunk_end = min(chunk_start + CHUNK_SIZE, extract_end)
        chunk_count += 1
        LOGGER.info(
            "Chunked: processing chunk %d (%.0f-%.0fs of %.0fs)%s",
            chunk_count, chunk_start, chunk_end, extract_end,
            " [per-chunk download]" if _per_chunk_download else "",
        )

        # ── Per-chunk download from YouTube ──
        if _per_chunk_download:
            t_dl = time.perf_counter()
            try:
                from app.services.detection_runtime import PipelineSettings
                settings = PipelineSettings()
                from functools import partial
                from starlette.concurrency import run_in_threadpool
                dl_result = await run_in_threadpool(
                    partial(
                        download_youtube_sync,
                        video_url,
                        start_time=chunk_start,
                        end_time=chunk_end,
                        yt_dlp_binary=settings.yt_dlp_binary,
                        ffmpeg_binary=settings.ffmpeg_binary,
                    )
                )
                _chunk_video_path = dl_result.path
                dl_elapsed = time.perf_counter() - t_dl
                vid_w, vid_h = get_video_resolution(_chunk_video_path)
                file_mb = round(_chunk_video_path.stat().st_size / 1024 / 1024, 1)
                LOGGER.info(
                    "Chunked: chunk %d downloaded %dx%d %sMB in %.1fs via %s",
                    chunk_count, vid_w, vid_h, file_mb, dl_elapsed,
                    getattr(dl_result, "strategy_used", "?"),
                )
                if chunk_count == 1:
                    youtube_strategy_used = getattr(dl_result, "strategy_used", "chunked_download")
                    layer_timings["youtube_download"] = {
                        "elapsed_ms": round(dl_elapsed * 1000),
                        "status": "success",
                        "video_resolution": f"{vid_w}x{vid_h}",
                        "was_sectioned": dl_result.was_sectioned,
                        "file_size_mb": file_mb,
                        "mode": "per_chunk",
                    }
                    phases_used.append("youtube_download")
            except Exception as exc:
                LOGGER.error("Chunked: chunk %d download failed: %s", chunk_count, exc)
                chunk_start = chunk_end
                continue

        # Extract frames for this chunk only
        t0 = time.perf_counter()
        # For per-chunk downloads, the video starts at 0 (it was pre-sectioned)
        _frame_start = 0.0 if _per_chunk_download else chunk_start
        _frame_end = (chunk_end - chunk_start) if _per_chunk_download else chunk_end
        chunk_frames = _extract_frames(
            _chunk_video_path,
            fps=1,  # 1fps for full games (capped by max_frames)
            sport=sport,
            start_sec=_frame_start,
            end_sec=_frame_end,
            max_frames=CHUNK_MAX_FRAMES,  # 400 frames per 30-min chunk
        )
        # For per-chunk downloads, offset timestamps to real game time
        if _per_chunk_download and chunk_frames:
            chunk_frames = [(ts + chunk_start, frame) for ts, frame in chunk_frames]

        total_frames += len(chunk_frames)
        all_frame_timestamps.extend(ts for ts, _ in chunk_frames)
        LOGGER.info("Chunked: chunk %d extracted %d frames in %.1fs",
                     chunk_count, len(chunk_frames), time.perf_counter() - t0)

        if not chunk_frames:
            chunk_start = chunk_end
            continue

        # Check cancellation between download and OCR
        if cancel_event is not None and cancel_event.is_set():
            LOGGER.warning("Pipeline: CANCELLED before OCR on chunk %d", chunk_count)
            break

        # Reload models if needed (unloaded by previous chunk cleanup)
        try:
            roboflow_detector.load_for_request(sport, jersey_color)
        except Exception:
            pass

        # Run OCR + v4 outcome detection on this chunk
        ocr_dets, v7_dets, motion, v4_dets = _run_chunk_ocr(
            chunk_frames,
            jersey_number=jersey_number,
            jersey_color=jersey_color,
            sport=sport,
            ocr_conf=ocr_conf,
            time_limit=90,  # 1.5 min per chunk — keep total under 600s
        )
        all_ocr_dets.extend(ocr_dets)
        all_v7_dets.extend(v7_dets)
        all_motion.update(motion)
        all_v4_dets.extend(v4_dets)

        # Free chunk frames immediately
        chunk_frames.clear()

        # Clean up per-chunk download file to save disk space
        if _per_chunk_download and _chunk_video_path and _chunk_video_path.exists():
            try:
                import shutil
                shutil.rmtree(_chunk_video_path.parent, ignore_errors=True)
            except Exception:
                pass

        # Unload non-essential models between chunks to prevent memory growth
        try:
            for _attr in dir(roboflow_detector):
                if _attr.endswith("_model") and _attr not in _ALWAYS_LOADED_MODELS:
                    if getattr(roboflow_detector, _attr, None) is not None:
                        setattr(roboflow_detector, _attr, None)
        except Exception:
            pass
        gc.collect()
        try:
            import torch as _torch_chunk
            if _torch_chunk.cuda.is_available():
                _torch_chunk.cuda.empty_cache()
        except Exception:
            pass

        # Log memory after chunk + gate for next chunk
        rss_chunk = _get_rss_mb()
        LOGGER.info("Chunked: chunk %d done — %d v5 dets, %d v7 dets, RSS=%.0fMB",
                     chunk_count, len(ocr_dets), len(v7_dets), rss_chunk)
        if rss_chunk > 5000:
            LOGGER.warning("Chunked: Memory high (%.0fMB > 5000MB) — forcing full cleanup before next chunk",
                           rss_chunk)
            _force_cleanup_memory()

        chunk_start = chunk_end

    layer_timings["chunked_processing"] = {
        "elapsed_ms": round((time.perf_counter() - _chunked_ocr_start) * 1000),
        "chunks": chunk_count,
        "total_frames": total_frames,
        "total_v5_detections": len(all_ocr_dets),
        "total_v7_detections": len(all_v7_dets),
        "total_v4_detections": len(all_v4_dets),
        "total_motion_scores": len(all_motion),
    }
    LOGGER.info(
        "Chunked: ALL %d chunks done — %d v5, %d v7, %d v4 outcomes, %d frames",
        chunk_count, len(all_ocr_dets), len(all_v7_dets), len(all_v4_dets), total_frames,
    )
    phases_used.append("chunked_ocr")

    # ── Merge all detections into detection points ──
    all_layer_dets_raw: list[dict] = []
    for det in all_ocr_dets:
        entry: dict[str, Any] = {
            "timestamp": det.get("timestamp", 0),
            "confidence": det.get("confidence", 0),
            "number_detected": det.get("number_detected", jersey_number),
            "layer": det.get("layer", "v5_ocr_universal"),
        }
        if det.get("player_bbox"):
            entry["player_bbox"] = det["player_bbox"]
        all_layer_dets_raw.append(entry)
    for det in all_v7_dets:
        entry_v7: dict[str, Any] = {
            "timestamp": det.get("timestamp", 0),
            "confidence": det.get("confidence", 0),
            "number_detected": det.get("number_detected", jersey_number),
            "layer": det.get("layer", "v7_football_ocr"),
        }
        if det.get("player_bbox"):
            entry_v7["player_bbox"] = det["player_bbox"]
        all_layer_dets_raw.append(entry_v7)

    # Filter for target jersey number
    all_numbers_seen = set(d.get("number_detected", "unknown") for d in all_layer_dets_raw)
    LOGGER.info("Chunked: numbers detected: %s (target=%d)", all_numbers_seen, jersey_number)
    all_layer_dets = [d for d in all_layer_dets_raw if d.get("number_detected") == jersey_number]
    wrong_number_count = len(all_layer_dets_raw) - len(all_layer_dets)

    # ── Build v4 outcome lookup ──
    v4_outcomes_by_ts: dict[float, str] = {}
    for d in all_v4_dets:
        ts = d.get("timestamp", 0)
        outcome = d.get("outcome", "")
        if outcome:
            # Keep highest-confidence outcome per timestamp
            existing = v4_outcomes_by_ts.get(ts)
            if not existing:
                v4_outcomes_by_ts[ts] = outcome
    LOGGER.info("Chunked: %d v4 outcome detections, %d unique timestamps",
                len(all_v4_dets), len(v4_outcomes_by_ts))

    # ── Build detection points ──
    detection_points: list[DetectionPoint] = []
    ts_buckets: dict[float, list[dict]] = defaultdict(list)
    for det in all_layer_dets:
        ts = det["timestamp"]
        bucket_key = None
        for existing_ts in ts_buckets:
            if abs(existing_ts - ts) < 0.5:
                bucket_key = existing_ts
                break
        if bucket_key is None:
            bucket_key = ts
        ts_buckets[bucket_key].append(det)

    pose_results: dict[float, dict] = {}  # No pose in chunked mode
    for bucket_ts, bucket_dets in sorted(ts_buckets.items()):
        best_conf = max(d["confidence"] for d in bucket_dets)
        layers_present = set(d["layer"] for d in bucket_dets)
        bonus = 0.0
        has_v5 = any("v5_ocr" in l for l in layers_present)
        has_v7 = any("v7" in l for l in layers_present)
        if has_v5 and has_v7:
            bonus += 0.20
        elif len(layers_present) >= 2:
            bonus += 0.15
        final_conf = min(1.0, best_conf + bonus)

        # Find v4 outcome near this timestamp (±2s)
        v4_out = ""
        for v4_ts, v4_outcome in v4_outcomes_by_ts.items():
            if abs(v4_ts - bucket_ts) < 2.0:
                v4_out = v4_outcome
                break

        detection_points.append(DetectionPoint(
            timestamp=bucket_ts,
            confidence=final_conf,
            jersey_visible=True,
            jersey_number=jersey_number,
            motion_score=all_motion.get(bucket_ts, _nearest_value(all_motion, bucket_ts)),
            crowd_energy=_get_crowd_energy(audio_result, bucket_ts),
            v4_outcome=v4_out,
        ))

    # ── Temporal consensus (relaxed for full games) ──
    tc_stats = {"raw_detections": len(all_layer_dets), "confirmed_detections": len(detection_points),
                "filtered_out": 0, "cross_layer_confirmed": 0}
    try:
        from app.services.temporal_consensus import TemporalConsensus
        tc_instance = TemporalConsensus(
            min_confirmations=1,
            time_window=5.0,
            confidence_threshold=0.12,
        )
        if all_layer_dets:
            confirmed_dets = tc_instance.filter_detections(all_layer_dets, jersey_number)
            confirmed_dets = tc_instance.cross_layer_boost(confirmed_dets)
            tc_stats["confirmed_detections"] = len(confirmed_dets)
            tc_stats["filtered_out"] = len(all_layer_dets) - len(confirmed_dets)
    except Exception as exc:
        LOGGER.warning("Chunked: temporal consensus failed: %s", exc)

    # ── Motion supplement for full games ──
    # Lowered thresholds: with 600 frames/chunk we get enough motion data
    # to reliably detect plays. Previous threshold (30) missed most action.
    _motion_supp_threshold = 15 if _is_football else 10
    _motion_supp_audio = 10 if _is_football else 8
    if 1 <= len(detection_points) <= 120 and all_frame_timestamps:
        _existing_ts = {dp.timestamp for dp in detection_points}
        _supplement_count = 0
        for t in all_frame_timestamps:
            if t in _existing_ts:
                continue
            if any(abs(t - ets) < 1.5 for ets in _existing_ts):
                continue
            motion = all_motion.get(t, 0)
            in_boundary = _in_audio_boundary(audio_result, t)
            if motion > _motion_supp_threshold or (in_boundary and motion > _motion_supp_audio):
                conf = motion / 100.0 * 0.5
                if in_boundary:
                    conf = min(0.8, conf + 0.1)
                # Check for v4 outcome near this timestamp
                _v4_out_supp = ""
                for _v4ts, _v4oc in v4_outcomes_by_ts.items():
                    if abs(_v4ts - t) < 2.0:
                        _v4_out_supp = _v4oc
                        break
                detection_points.append(DetectionPoint(
                    timestamp=t,
                    confidence=conf,
                    jersey_visible=False,  # Motion-inferred, NOT confirmed by OCR
                    jersey_number=None,    # Don't stamp requested number on unconfirmed frames
                    motion_score=motion,
                    crowd_energy=_get_crowd_energy(audio_result, t),
                    v4_outcome=_v4_out_supp,
                ))
                _supplement_count += 1
        if _supplement_count:
            LOGGER.info("Chunked: motion supplement added %d points (total=%d)",
                        _supplement_count, len(detection_points))

    # ── Motion/audio fallback if no OCR detections ──
    if not detection_points and all_frame_timestamps:
        motion_threshold = 8 if _is_football else 10
        LOGGER.info("Chunked: no OCR, using motion/audio fallback (threshold=%d)", motion_threshold)
        for t in all_frame_timestamps:
            motion = all_motion.get(t, 0)
            in_boundary = _in_audio_boundary(audio_result, t)
            if motion > motion_threshold or in_boundary:
                conf = motion / 100.0 * 0.7
                if in_boundary:
                    conf = min(1.0, conf + 0.15)
                detection_points.append(DetectionPoint(
                    timestamp=t,
                    confidence=conf,
                    jersey_visible=False,
                    motion_score=motion,
                    crowd_energy=_get_crowd_energy(audio_result, t),
                ))

    # ── Extract clips ──
    _t_clip_extract = time.perf_counter()
    # Use effective analyzed duration, NOT full video duration.
    # Full video might be 7200s but we only extracted 0-1200s.
    # Passing full duration forces _is_full_game=True → cluster_gap=5.0 (too wide).
    _effective_clip_dur = extract_end - extract_start
    clips = extract_clips(
        detections=detection_points,
        audio_result=audio_result if audio_result.has_audio else None,
        sport=sport,
        position=position,
        video_duration=_effective_clip_dur,
    )
    layer_timings["clip_extraction"] = {
        "elapsed_ms": round((time.perf_counter() - _t_clip_extract) * 1000),
        "detection_points": len(detection_points),
        "clips_extracted": len(clips),
    }

    # Clip splitting REMOVED in v8.4 — clip_extractor handles hard caps.
    # Just enforce max duration per clip.
    _MAX_CLIP_C = {"football": FOOTBALL_MAX_CLIP, "basketball": 12.0, "lacrosse": 15.0}
    _max_c = _MAX_CLIP_C.get(sport.lower(), 15.0)
    for clip in clips:
        if clip.end_time - clip.start_time > _max_c:
            clip.end_time = clip.start_time + _max_c
    LOGGER.info("Chunked: %d clips (max_dur=%.0fs, no splitting)", len(clips), _max_c)

    # ── Temporal jersey attribution ──
    # If jersey was confirmed in nearby clips (within 15s), attribute it to
    # unconfirmed clips too.  This dramatically improves jerseyDetectionRate
    # for footage where the jersey is intermittently visible.
    _jersey_ts = sorted(
        d["timestamp"] for d in all_layer_dets
        if d.get("number_detected") == jersey_number
    ) if all_layer_dets else []
    if _jersey_ts and clips:
        _attr_count = 0
        for clip in clips:
            if clip.jersey_visible:
                continue
            # Check if any OCR detection of this jersey is within 15s of this clip
            for jts in _jersey_ts:
                if clip.start_time - 15 <= jts <= clip.end_time + 15:
                    clip.jersey_visible = True
                    clip.jersey_number_seen = jersey_number
                    _attr_count += 1
                    break
        if _attr_count:
            LOGGER.info("Chunked: temporal jersey attribution: %d clips gained jersey=%d",
                        _attr_count, jersey_number)

    # ── Unload models ──
    try:
        roboflow_detector.unload_request_models()
    except Exception:
        pass

    elapsed = time.perf_counter() - start_time

    # Get request summary
    request_summary = {}
    try:
        request_summary = roboflow_detector.get_request_summary()
    except Exception:
        pass
    if not request_summary.get("models_called"):
        _mc = ["player_detector_v5", "jersey_ocr_universal_v5", "dead_ball_classifier_v5"]
        _dpm: dict[str, int] = {
            "player_detector_v5": total_frames,
            "jersey_ocr_universal_v5": len(all_ocr_dets),
            "dead_ball_classifier_v5": total_frames // 4,
        }
        if all_v7_dets:
            _mc.append("football_jersey_ocr_v7")
            _dpm["football_jersey_ocr_v7"] = len(all_v7_dets)
        request_summary = {"models_called": _mc, "detections_per_model": _dpm}

    # Audio events
    audio_events_out = []
    if audio_result.has_audio:
        for evt in audio_result.events[:50]:
            audio_events_out.append({
                "timestamp": evt.timestamp,
                "eventType": evt.event_type,
                "confidence": round(evt.confidence, 2),
            })

    # Memory
    memory_rss_mb = 0
    try:
        import psutil
        memory_rss_mb = round(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        pass

    # Build clips output
    clips_out = []
    for clip in clips:
        clip_dict: dict[str, Any] = {
            "startTime": clip.start_time,
            "endTime": clip.end_time,
            "confidence": clip.confidence,
            "score": clip.score,
            "playType": clip.play_type,
            "grade": clip.grade,
            "jerseyVisible": clip.jersey_visible,
            "jerseyNumberSeen": clip.jersey_number_seen,
            "trackingId": clip.tracking_id,
            "description": clip.description,
            "signals": clip.signals,
        }
        # Include v4_outcome from signals if present
        v4_out = (clip.signals or {}).get("v4_outcome")
        if v4_out:
            clip_dict["v4Outcome"] = v4_out
        # Rule-based play type supplement (when v4 models didn't detect)
        if not v4_out and clip_dict["playType"] == "game_action":
            rule_type = _detect_play_type_rules(
                motion_score=(clip.signals or {}).get("motion", 0) or 0,
                pose=(clip.signals or {}).get("pose", "standing"),
                crowd_energy=(clip.signals or {}).get("crowd", 0) or 0,
                jersey_visible=clip.jersey_visible,
                sport=sport,
            )
            if rule_type != "game_action":
                clip_dict["playType"] = rule_type
                clip_dict["description"] = rule_type.replace("_", " ").title()
        # Caption + recruiting score + game clock for coach-friendly display
        clip_dict["caption"] = _generate_clip_caption(clip_dict, sport, position, jersey_number)
        clip_dict["recruitingScore"] = _compute_recruiting_score(clip_dict, sport, position)
        clip_dict["estimatedQuarter"] = _estimate_game_quarter(clip.start_time, video_duration, sport)
        clips_out.append(clip_dict)

    # Sort clips in highlight reel order (TDs first, then big plays, etc.)
    clips_out.sort(key=_get_highlight_sort_key)

    # Add sequenceNote to each clip
    for i, clip_dict in enumerate(clips_out):
        prev = clips_out[i - 1] if i > 0 else None
        clip_dict["sequenceNote"] = _generate_sequence_note(
            clip_dict, i, len(clips_out), prev, sport,
        )

    # Get primary detection layer
    try:
        primary_layer = roboflow_detector.get_primary_detection()
    except Exception:
        primary_layer = "v5"

    target_found = jersey_number in all_numbers_seen

    LOGGER.info("=== CHUNKED DETECTION SUMMARY ===")
    LOGGER.info("Sport: %s, Jersey: #%d, Chunks: %d, Frames: %d",
                sport, jersey_number, chunk_count, total_frames)
    LOGGER.info("V5 OCR: %d, V7 OCR: %d, V4 outcomes: %d, Detection points: %d, Clips: %d",
                len(all_ocr_dets), len(all_v7_dets), len(all_v4_dets), len(detection_points), len(clips_out))
    LOGGER.info("Memory RSS: %dMB, Elapsed: %.1fs", memory_rss_mb, elapsed)
    LOGGER.info("=================================")

    debug = {
        "primary_layer": primary_layer,
        "ali_detections": 0,
        "universal_v2_detections": 0,
        "v3_ocr_detections": 0,
        "v2_sport_detections": 0,
        "v3_primary_detections": 0,
        "v5_ocr_detections": len(all_ocr_dets),
        "v7_football_detections": len(all_v7_dets),
        "v7_navy_detections": sum(1 for d in all_v7_dets if "v7_navy" in d.get("layer", "")),
        "v7_player_crops": 0,
        "v4_outcome_detections": len(all_v4_dets),
        "v4_outcomes_found": len(v4_outcomes_by_ts),
        "combined_detections": len(detection_points),
        "numbers_detected": sorted(str(n) for n in all_numbers_seen),
        "target_jersey_number": jersey_number,
        "target_number_found": target_found,
        "frames_with_target": len(all_layer_dets),
        "wrong_number_filtered": wrong_number_count,
        "analyze_fps": 1,
        "frames_extracted": total_frames,
        "dead_ball_frames_skipped": 0,
        "dead_ball_ratio": 0.0,
        "scoreboard_detections": 0,
        "score_changes_detected": 0,
        "quality_mode": quality_mode,
        "resolved_quality": "standard",
        "ocr_confidence": ocr_conf,
        "ali_working": False,
        "ali_status": "skipped_chunked",
        "youtube_strategy_used": youtube_strategy_used,
        "total_elapsed_ms": round(elapsed * 1000),
        "layers_that_contributed": list(set(phases_used)),
        "layer_breakdown": layer_timings,
        "temporal_consensus": tc_stats,
        "cross_layer_agreements": [],
        "models_called": request_summary.get("models_called", []),
        "detections_per_model": request_summary.get("detections_per_model", {}),
        "ocr_track_mapping": {},
        "clips_before_filter": len(clips),
        "clips_after_filter": len(clips_out),
        "memory_rss_mb": memory_rss_mb,
        "chunked_processing": True,
        "chunks_processed": chunk_count,
    }

    # Player summary for coaches
    player_summary = _build_player_summary(
        clips_out, jersey_number, sport, position, video_duration, elapsed,
    )

    return {
        "playerSummary": player_summary,
        "clips": clips_out,
        "layerUsed": "+".join(phases_used),
        "elapsed": round(elapsed, 1),
        "videoDuration": round(video_duration, 1),
        "framesProcessed": total_frames,
        "audioEvents": audio_events_out,
        "playerTracks": [],
        "gameStats": {},
        "perClipStats": [],
        "actionsDetected": [],
        "debug": debug,
    }


def _run_jersey_detection(
    *,
    video_url: str | None,
    video_path: str | None,
    jersey_number: int,
    jersey_color: str,
    sport: str,
    position: str | None,
) -> list[dict]:
    """Run the existing YOLO + OCR ensemble detection pipeline."""
    try:
        from app.services.detection_service import DetectionService
        from app.schemas.detect import DetectRequest

        # Build request
        req_data: dict = {
            "jerseyNumber": jersey_number,
            "jerseyColor": jersey_color,
            "sport": sport,
        }
        if video_url:
            req_data["videoUrl"] = video_url
        elif video_path:
            req_data["videoPath"] = video_path
        if position:
            req_data["position"] = position

        LOGGER.info("Ali request: %s", {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in req_data.items()})

        request = DetectRequest(**req_data)
        service = DetectionService()
        detections = service.detect(request)

        LOGGER.info("Ali returned %d raw detections", len(detections))

        # detect_jersey_in_frames returns list[dict] via DetectedFrame.to_dict()
        # Each dict: {"timestamp": float, "confidence": float, "bbox": {"x1": ..., ...}}
        results = []
        for det in detections:
            if isinstance(det, dict):
                d = {
                    "timestamp": det.get("timestamp", 0),
                    "confidence": det.get("confidence", 0),
                }
                bbox = det.get("bbox")
                if bbox and isinstance(bbox, dict):
                    d["x1"] = bbox.get("x1", 0)
                    d["y1"] = bbox.get("y1", 0)
                    d["x2"] = bbox.get("x2", 0)
                    d["y2"] = bbox.get("y2", 0)
            else:
                # Fallback: if an object is returned (e.g. DetectionFrame)
                d = {
                    "timestamp": getattr(det, "timestamp", 0),
                    "confidence": getattr(det, "confidence", 0),
                }
                if hasattr(det, "bbox") and det.bbox:
                    d["x1"] = getattr(det.bbox, "x1", 0)
                    d["y1"] = getattr(det.bbox, "y1", 0)
                    d["x2"] = getattr(det.bbox, "x2", 0)
                    d["y2"] = getattr(det.bbox, "y2", 0)
            results.append(d)

        return results

    except Exception as exc:
        LOGGER.error(
            "Jersey detection failed: %s (type=%s). Check if detect_jersey_in_frames "
            "return shape changed (expected list[dict]).",
            exc, type(exc).__name__,
        )
        return []


def _run_chunk_ocr(
    chunk_frames: list[tuple[float, np.ndarray]],
    jersey_number: int,
    jersey_color: str,
    sport: str,
    ocr_conf: float,
    time_limit: float = 90,
) -> tuple[list[dict], list[dict], dict[float, float], list[dict]]:
    """Run OCR + v4 outcome detection on a chunk of frames.

    Returns (ocr_detections, v7_detections, motion_scores, v4_outcome_detections).
    Used by chunked pipeline to process one 30-min chunk at a time.
    """
    import gc
    from app.services.roboflow_detector import roboflow_detector, is_dark_color, is_navy

    chunk_ocr_dets: list[dict] = []
    chunk_v7_dets: list[dict] = []
    chunk_motion: dict[float, float] = {}
    _is_football = sport.lower() == "football"

    t0 = time.perf_counter()

    # ── Dead ball filter ──
    _sport_lower = sport.lower()
    if _sport_lower == "football":
        _db_conf = 0.85
    elif _sport_lower == "lacrosse":
        _db_conf = 0.70
    else:
        _db_conf = 0.40

    live_frames: list[tuple[float, np.ndarray]] = []
    dead_count = 0
    # Sample every 4th frame for dead ball (saves time)
    for idx, (ts, frame) in enumerate(chunk_frames):
        if idx % 4 != 0:
            live_frames.append((ts, frame))
            continue
        try:
            db_result = roboflow_detector.classify_dead_ball(frame, conf=_db_conf)
            if db_result == "dead_ball":
                dead_count += 1
            else:
                live_frames.append((ts, frame))
        except Exception:
            live_frames.append((ts, frame))

    # Safety: if >50% dead, keep all
    sampled_count = len(chunk_frames) // 4 + (1 if len(chunk_frames) % 4 else 0)
    if sampled_count > 0 and dead_count / sampled_count > 0.5:
        live_frames = chunk_frames

    if not live_frames:
        live_frames = chunk_frames

    LOGGER.info("Chunk OCR: %d frames, %d live after dead ball filter", len(chunk_frames), len(live_frames))

    # ── Motion scoring ──
    for i in range(len(chunk_frames) - 1):
        t_val, prev_frame = chunk_frames[i]
        t_next, curr_frame = chunk_frames[i + 1]
        try:
            score = compute_motion_score(prev_frame, curr_frame)
            chunk_motion[t_next] = score.score
        except Exception:
            pass

    # ── Dark jersey preprocessing ──
    _is_dark = is_dark_color(jersey_color)
    _is_navy_jersey = is_navy(jersey_color)
    if _is_dark and live_frames:
        gamma = 1.5 if _is_navy_jersey else 1.3
        inv_gamma = 1.0 / gamma
        _gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        clip_limit = 5.0 if _is_navy_jersey else 4.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        for idx in range(len(live_frames)):
            t_val, frame = live_frames[idx]
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            frame = cv2.LUT(frame, _gamma_table)
            gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
            frame = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
            live_frames[idx] = (t_val, frame)

    # ── v7 football OCR ──
    if _is_football:
        _v7_t0 = time.perf_counter()
        _v7_sample = live_frames[::max(1, len(live_frames) // 50)][:50]
        for ts, frame in _v7_sample:
            if time.perf_counter() - _v7_t0 > 30:
                break
            try:
                dets = roboflow_detector.detect_football_jersey_v7(frame, jersey_number, conf=ocr_conf)
                if dets:
                    for d in dets:
                        d["timestamp"] = ts
                        if "bbox" in d and "player_bbox" not in d:
                            d["player_bbox"] = d["bbox"]
                        chunk_v7_dets.append(d)
            except Exception:
                pass

    # ── v5 player detection → OCR ──
    _player_conf = 0.35 if _is_football else 0.20
    # Uniform coverage: step through frames so time-limited OCR covers
    # the full chunk, not just the first N seconds.
    _v5_chunk_budget = max(20, time_limit // 3)
    _v5_chunk_step = max(1, len(live_frames) // _v5_chunk_budget)
    sampled = live_frames[::_v5_chunk_step]
    _v5_crops = 0
    _V5_MAX_CROPS = 200  # Per chunk (balanced: speed vs detection rate)
    for ts, frame in sampled:
        if time.perf_counter() - t0 > time_limit:
            break
        if _v5_crops >= _V5_MAX_CROPS:
            break
        try:
            players = roboflow_detector.detect_players_v5(
                frame, conf=_player_conf, validate_crop_size=_is_football,
            )
        except Exception:
            continue
        if not players:
            continue
        for player in players[:3]:
            if _v5_crops >= _V5_MAX_CROPS:
                break
            x1, y1, x2, y2 = [int(c) for c in player["bbox"]]
            h, w = frame.shape[:2]
            _pad_ratio = 0.10 if _is_football else 0.25
            pad_x = int((x2 - x1) * _pad_ratio)
            pad_y = int((y2 - y1) * _pad_ratio)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if cw < 8 or ch < 12:
                continue
            # Adaptive upscale
            if cw < 50:
                _scale = 8
            elif cw < 100:
                _scale = 4
            elif _is_football and max(cw, ch) < 500:
                _scale = 2
            else:
                _scale = 1
            if _scale > 1:
                crop = cv2.resize(crop, (cw * _scale, ch * _scale), interpolation=cv2.INTER_CUBIC)
                gaussian = cv2.GaussianBlur(crop, (0, 0), 1.5)
                crop = cv2.addWeighted(crop, 1.5, gaussian, -0.5, 0)
            _v5_crops += 1
            try:
                dets = roboflow_detector.detect_jersey_v5(
                    crop, jersey_number=jersey_number, conf=ocr_conf, skip_preprocess=True,
                )
                if dets:
                    chunk_ocr_dets.extend({
                        **d, "timestamp": ts,
                        "player_bbox": [cx1, cy1, cx2, cy2],
                    } for d in dets)
            except Exception:
                pass

    # ── v4 outcome detection — DISABLED in chunked mode ──
    # v4 models add ~200MB RSS each and 20-30s per chunk.
    # Rule-based play detection (_detect_play_type_rules) provides
    # play types from motion/pose/crowd signals without v4 models.
    chunk_v4_dets: list[dict] = []

    LOGGER.info(
        "Chunk OCR: %d v5, %d v7, %d v4 outcomes, %d crops, %d motion in %.1fs",
        len(chunk_ocr_dets), len(chunk_v7_dets), len(chunk_v4_dets), _v5_crops,
        len(chunk_motion), time.perf_counter() - t0,
    )

    # Free chunk models and check memory
    gc.collect()
    rss_after_chunk = _get_rss_mb()
    LOGGER.info("Chunk OCR cleanup: RSS=%.0fMB", rss_after_chunk)

    return chunk_ocr_dets, chunk_v7_dets, chunk_motion, chunk_v4_dets


def _smart_sample_frames(
    video_path: Path,
    target_frames: int = 750,
    start_sec: float = 0,
    end_sec: float = 0,
) -> list[tuple[float, np.ndarray]]:
    """Two-pass motion-aware frame sampling for full games.

    Pass 1: Quick scan at ~0.2fps to compute motion scores across video.
    Pass 2: Sample densely (2fps) in high-motion windows, sparsely elsewhere.
    Returns up to target_frames frames sorted by timestamp.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames_count / video_fps if video_fps > 0 else 0
    _start = start_sec
    _end = end_sec if end_sec > 0 else video_dur

    # Detect pre-trimmed video
    if _start > 0 and video_dur > 0 and _start > video_dur * 0.9:
        _end = min(_end - _start, video_dur)
        _start = 0

    LOGGER.info("Smart sampling: video=%.0fs, range=%.0f-%.0f, target=%d frames",
                video_dur, _start, _end, target_frames)

    # Pass 1: Quick motion scan (~0.2fps = 1 frame every 5s)
    scan_interval = max(1, int(video_fps * 5))  # every 5 seconds
    prev_frame = None
    motion_windows: list[tuple[float, float]] = []  # (timestamp, motion_score)

    if _start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(_start * video_fps))

    frame_idx = int(_start * video_fps) if _start > 0 else 0
    scan_count = 0
    while scan_count < 2000:  # cap scan at 2000 samples
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / video_fps
        if ts > _end:
            break
        if frame_idx % scan_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180))
            if prev_frame is not None:
                diff = cv2.absdiff(small, prev_frame)
                score = float(diff.mean())
                motion_windows.append((ts, score))
            prev_frame = small
            scan_count += 1
        frame_idx += 1

    cap.release()

    if not motion_windows:
        LOGGER.warning("Smart sampling: no motion data, falling back to uniform")
        return _extract_frames(video_path, fps=1, start_sec=start_sec,
                               end_sec=end_sec, max_frames=target_frames)

    # Sort by motion score to find high-action windows
    sorted_windows = sorted(motion_windows, key=lambda x: x[1], reverse=True)
    avg_motion = sum(m for _, m in motion_windows) / len(motion_windows)
    high_motion_thresh = max(avg_motion * 1.5, 5.0)

    # Select top windows with motion above threshold
    high_motion_times = sorted(
        [ts for ts, score in sorted_windows if score >= high_motion_thresh]
    )

    # Deduplicate: merge windows within 10s of each other
    dense_ranges: list[tuple[float, float]] = []
    for ts in high_motion_times:
        if dense_ranges and ts - dense_ranges[-1][1] < 10:
            dense_ranges[-1] = (dense_ranges[-1][0], ts + 5)  # extend range
        else:
            dense_ranges.append((max(_start, ts - 5), ts + 5))

    LOGGER.info("Smart sampling: %d motion windows, %d high-motion ranges (thresh=%.1f, avg=%.1f)",
                len(motion_windows), len(dense_ranges), high_motion_thresh, avg_motion)

    # Pass 2: Extract frames — dense (2fps) in high-motion, sparse (0.2fps) elsewhere
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    dense_frame_interval = max(1, int(video_fps / 2))  # 2fps in action
    sparse_frame_interval = max(1, int(video_fps * 5))  # 0.2fps elsewhere
    frames: list[tuple[float, np.ndarray]] = []

    if _start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(_start * video_fps))

    frame_idx = int(_start * video_fps) if _start > 0 else 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / video_fps
        if ts > _end:
            break

        # Determine if this timestamp is in a high-motion range
        in_dense = any(r[0] <= ts <= r[1] for r in dense_ranges)
        interval = dense_frame_interval if in_dense else sparse_frame_interval

        if frame_idx % interval == 0:
            h, w = frame.shape[:2]
            if w < 1280:
                scale = min(2.0, 1280 / w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            frames.append((ts, frame))
            if len(frames) >= target_frames:
                break

        frame_idx += 1

    cap.release()
    LOGGER.info("Smart sampling: extracted %d frames (%d dense ranges, %d total motion windows)",
                len(frames), len(dense_ranges), len(motion_windows))
    return frames


def _extract_frames(
    video_path: Path,
    fps: int = 2,
    sport: str = "basketball",
    start_sec: float = 0,
    end_sec: float = 0,
    max_frames: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """Extract frames from video at given FPS.

    Args:
        start_sec/end_sec: Only extract frames in [start, end]. 0/0 = full video.
        max_frames: Hard cap on total frames (0 = use global MAX_FRAMES).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_est = total_frames / video_fps if video_fps > 0 else 0

    # Robust pre-trim detection: if start_sec is beyond 90% of the video
    # duration, the video was likely pre-trimmed by --download-sections or
    # the render server.  Reset to scan from the beginning.
    if start_sec > 0 and video_duration_est > 0 and start_sec > video_duration_est * 0.9:
        # Video was pre-trimmed (e.g. start=120, end=180, but file is only 60s long)
        # Compute expected segment duration before resetting start_sec
        segment_duration = (end_sec - start_sec) if end_sec > start_sec else video_duration_est
        LOGGER.warning(
            "_extract_frames: start_sec=%.1f > video_duration=%.1fs — "
            "video likely pre-trimmed, resetting to start=0 end=%.1f",
            start_sec, video_duration_est, min(segment_duration, video_duration_est),
        )
        start_sec = 0
        end_sec = min(segment_duration, video_duration_est)

    frame_interval = max(1, int(video_fps / fps))
    cap_limit = max_frames if max_frames > 0 else MAX_FRAMES

    # ── Uniform sampling: if total frames at target FPS exceeds cap,
    #    increase interval so frames are spread across the FULL video
    #    instead of being concentrated at the start. ──
    _range_start = max(start_sec, 0)
    _range_end = end_sec if end_sec > 0 else video_duration_est
    _range_frames_at_fps = int((_range_end - _range_start) * video_fps / frame_interval) if frame_interval > 0 else 0
    if cap_limit > 0 and _range_frames_at_fps > cap_limit:
        _uniform_skip = max(1, _range_frames_at_fps // cap_limit)
        frame_interval *= _uniform_skip
        LOGGER.info(
            "Frame extraction: uniform sampling skip=%d "
            "(range_frames=%d, cap=%d, new_interval=%d)",
            _uniform_skip, _range_frames_at_fps, cap_limit, frame_interval,
        )

    frames: list[tuple[float, np.ndarray]] = []
    frame_idx = 0

    # If start_sec > 0, seek ahead to save time
    if start_sec > 0:
        start_frame = int(start_sec * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / video_fps

        # Stop if past end of time range
        if end_sec > 0 and timestamp > end_sec:
            break

        if frame_idx % frame_interval == 0 and timestamp >= start_sec:
            # Upscale low-res video to ~1280px width for better detection
            h, w = frame.shape[:2]
            if w < 1280:
                scale = min(2.0, 1280 / w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            frames.append((timestamp, frame))

            if len(frames) >= cap_limit:
                LOGGER.info("Frame extraction: hit MAX_FRAMES cap (%d)", cap_limit)
                break

        frame_idx += 1

    cap.release()
    LOGGER.info(
        "Extracted %d frames (fps=%d, range=%.1f-%.1f, cap=%d, video_dur=%.1f)",
        len(frames), fps, start_sec, end_sec, cap_limit, video_duration_est,
    )
    return frames


def _nearest_value(lookup: dict[float, float], target: float) -> float:
    """Find the nearest value in a timestamp→value dict."""
    if not lookup:
        return 0.0
    nearest_key = min(lookup.keys(), key=lambda k: abs(k - target))
    if abs(nearest_key - target) < 2.0:
        return lookup[nearest_key]
    return 0.0


def _nearest_pose(lookup: dict[float, dict], target: float) -> dict:
    """Find the nearest pose result."""
    if not lookup:
        return {"action": "standing", "intensity": 0.0, "is_facing": False}
    nearest_key = min(lookup.keys(), key=lambda k: abs(k - target))
    if abs(nearest_key - target) < 3.0:
        return lookup[nearest_key]
    return {"action": "standing", "intensity": 0.0, "is_facing": False}


def _get_crowd_energy(audio_result: AudioAnalysisResult, timestamp: float) -> float:
    """Get crowd energy at a given timestamp."""
    if not audio_result.has_audio or not audio_result.energy_curve:
        return 0.0

    nearest = min(audio_result.energy_curve, key=lambda p: abs(p.timestamp - timestamp))
    if abs(nearest.timestamp - timestamp) < 3.0:
        return nearest.energy
    return 0.0


def _in_audio_boundary(audio_result: AudioAnalysisResult, timestamp: float) -> bool:
    """Check if timestamp falls within any audio play boundary."""
    if not audio_result.has_audio:
        return False
    for boundary in audio_result.play_boundaries:
        if boundary.start_time <= timestamp <= boundary.end_time:
            return True
    return False


def _error_response(
    message: str,
    elapsed: float,
    layer_timings: dict | None = None,
) -> dict:
    return {
        "clips": [],
        "layerUsed": "none",
        "elapsed": round(elapsed, 1),
        "videoDuration": 0,
        "framesProcessed": 0,
        "audioEvents": [],
        "playerTracks": [],
        "gameStats": {},
        "perClipStats": [],
        "actionsDetected": [],
        "debug": {
            "primary_layer": "none",
            "ali_detections": 0,
            "universal_v2_detections": 0,
            "v3_ocr_detections": 0,
            "v2_sport_detections": 0,
            "v3_primary_detections": 0,
            "v5_ocr_detections": 0,
            "v4_outcome_detections": 0,
            "v4_outcomes_found": 0,
            "combined_detections": 0,
            "dead_ball_frames_skipped": 0,
            "dead_ball_ratio": 0.0,
            "scoreboard_detections": 0,
            "score_changes_detected": 0,
            "ali_working": False,
            "ali_status": "not_run",
            "youtube_strategy_used": None,
            "total_elapsed_ms": round(elapsed * 1000),
            "layers_that_contributed": [],
            "layer_breakdown": layer_timings or {},
            "temporal_consensus": {
                "raw_detections": 0,
                "confirmed_detections": 0,
                "filtered_out": 0,
                "cross_layer_confirmed": 0,
            },
            "cross_layer_agreements": [],
        },
        "error": message,
    }
