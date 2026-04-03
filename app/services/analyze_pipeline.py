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
FOOTBALL_CONF_THRESHOLD = 0.15  # Lower than default 0.35
FOOTBALL_MIN_CLIP = 3.0
FOOTBALL_MAX_CLIP = 12.0

# Frame sampling: how many FPS to extract for Roboflow layers.
# Default 2 = 1 frame per 0.5s. Railway OOM kills at ~2.3GB;
# each 1280×720 frame ≈ 2.7MB, so 200 frames ≈ 540MB.
# Configurable via ANALYZE_FPS env var.
ANALYZE_FPS = int(os.getenv("ANALYZE_FPS", "2"))

# Hard cap on total frames to prevent OOM on long videos
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "150"))

# ── Blocker 3: Request semaphore — only 1 concurrent analyze request ──
import asyncio as _asyncio
_REQUEST_SEMAPHORE = _asyncio.Semaphore(1)
# Memory threshold (MB) — if RSS exceeds this before a request, force full cleanup
_MEMORY_THRESHOLD_MB = 5000


def _get_adaptive_fps(video_duration: float, sport: str = "basketball") -> tuple[int, int]:
    """Return (fps, max_frames) based on video duration.

    Strategy:
      - Short clips (<120s): 2 fps, 150 frames → full coverage
      - Medium clips (120-600s): 1 fps, 300 frames → every second for 5 min
      - Long videos (600-1800s): 0.5 fps, 450 frames → every 2s for 15 min
      - Full games (>1800s): 0.33 fps, 600 frames → every 3s for 30 min
    """
    if video_duration <= 120:
        return 2, 150
    elif video_duration <= 600:
        return 1, 300
    elif video_duration <= 1800:
        # ~30 min video: sample every 2s
        return 1, 450  # will use vid_stride in frame extraction
    else:
        # Full game (>30 min): aggressive skip
        return 1, 600


def _force_cleanup_memory():
    """Force cleanup of ALL loaded models and caches to reclaim memory.

    Called when RSS exceeds threshold between requests.
    """
    import gc
    LOGGER.info("Pipeline: force memory cleanup starting")
    try:
        from app.services.roboflow_detector import roboflow_detector
        for attr in dir(roboflow_detector):
            if attr.endswith("_model") and getattr(roboflow_detector, attr, None) is not None:
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
    # Delete any lingering YOLO Results references
    gc.collect()
    LOGGER.info("Pipeline: force memory cleanup done")


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
) -> dict[str, Any]:
    """Internal pipeline implementation (called under semaphore)."""
    start_time = time.perf_counter()
    phases_used: list[str] = []
    local_video_path: Path | None = None
    tmp_dir: Path | None = None
    frames_processed = 0
    youtube_strategy_used: str | None = None

    # ── Pre-request cleanup: free ALL models from previous requests ──
    # This prevents OOM when back-to-back requests accumulate models in memory.
    import gc as _gc_pre
    # Check RSS before cleanup
    _pre_rss = 0.0
    try:
        import psutil as _ps
        _pre_rss = _ps.Process().memory_info().rss / 1024 / 1024
        LOGGER.info("Pipeline: pre-request RSS = %.0fMB (threshold=%dMB)", _pre_rss, _MEMORY_THRESHOLD_MB)
    except Exception:
        pass
    # Always clean up between requests
    try:
        from app.services.roboflow_detector import roboflow_detector
        for _attr in dir(roboflow_detector):
            if _attr.endswith("_model") and getattr(roboflow_detector, _attr, None) is not None:
                setattr(roboflow_detector, _attr, None)
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
    _gc_pre.collect()
    # Check RSS after cleanup
    try:
        _post_rss = _ps.Process().memory_info().rss / 1024 / 1024
        LOGGER.info("Pipeline: pre-request cleanup done (RSS: %.0fMB → %.0fMB, freed %.0fMB)",
                     _pre_rss, _post_rss, _pre_rss - _post_rss)
    except Exception:
        LOGGER.info("Pipeline: pre-request cleanup done")

    # Per-layer timing and debug info
    layer_timings: dict[str, dict] = {}

    try:
        # ── Step 1: Acquire video ────────────────────────────────────────
        if video_url and is_youtube_url(video_url):
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
                youtube_strategy_used = "download_success"
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
                # Adaptive FPS: lower sampling rate for longer videos
                _effective_duration = video_duration
                if time_range_end > time_range_start:
                    _effective_duration = time_range_end - time_range_start
                _adaptive_fps, _adaptive_max = _get_adaptive_fps(_effective_duration, sport)
                # Use env override if set, otherwise adaptive
                _use_fps = ANALYZE_FPS if os.getenv("ANALYZE_FPS") else _adaptive_fps
                _use_max = int(os.getenv("MAX_FRAMES", "0")) or _adaptive_max
                LOGGER.info(
                    "Pipeline: adaptive FPS = %d (duration=%.0fs), max_frames = %d",
                    _use_fps, _effective_duration, _use_max,
                )
                _actual_fps_used = _use_fps
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
                for ts, frame in frames:
                    db_result = roboflow_detector.classify_dead_ball(frame)
                    if db_result:
                        dead_ball_by_ts[ts] = db_result
                    if db_result == "dead_ball":
                        dead_ball_count += 1
                    else:
                        _live.append((ts, frame))

                dead_ball_ratio = dead_ball_count / len(frames) if frames else 0.0
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
            ocr_conf = 0.12
        elif _is_dark_jersey:
            ocr_conf = 0.15
        elif sport.lower() == "football":
            ocr_conf = FOOTBALL_CONF_THRESHOLD  # 0.15 — smaller numbers on helmets/distance
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

        # ── Step 7.5a: v5 player detection → v5 OCR on crops (PRIMARY) ──
        # Guardrails: max 90s, max 200 crops, early exit after 50 zero-match crops
        _V5_TIME_LIMIT = 90  # seconds
        _V5_MAX_CROPS = 200
        _V5_EARLY_EXIT_AFTER = 50  # consecutive zero-match crops before giving up
        v5_ocr_detections: list[dict] = []
        v5_players_found = 0
        v5_no_player_frames = 0
        v5_crop_dims: list[str] = []
        v5_total_crops = 0
        v5_consecutive_misses = 0
        v5_exit_reason = ""
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            if ocr_frames:
                # Sample every 2nd frame for memory safety
                sampled_frames = ocr_frames[::2] if len(ocr_frames) > 30 else ocr_frames
                LOGGER.info("Pipeline: v5 OCR layer running on %d/%d frames (sampled)", len(sampled_frames), len(ocr_frames))
                _v5_break = False
                _is_football = sport.lower() == "football"
                _v5_oversized_skipped = 0
                _v5_redetected = 0
                for t, frame in sampled_frames:
                    if _v5_break:
                        break
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
                    if players:
                        for player in players[:5]:  # Max 5 players per frame
                            if v5_total_crops >= _V5_MAX_CROPS:
                                v5_exit_reason = f"crop_limit ({_V5_MAX_CROPS})"
                                _v5_break = True
                                break
                            if v5_consecutive_misses >= _V5_EARLY_EXIT_AFTER:
                                v5_exit_reason = f"early_exit ({_V5_EARLY_EXIT_AFTER} consecutive misses)"
                                _v5_break = True
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
                            # Skip tiny crops that can't contain readable jersey numbers
                            if cw < 40 or ch < 60:
                                LOGGER.debug("Skipping tiny crop %dx%d at frame t=%.1f", cw, ch, t)
                                v5_crop_dims.append(f"{cw}x{ch}(skip)")
                                continue
                            v5_crop_dims.append(f"{cw}x{ch}")
                            v5_total_crops += 1
                            dets = roboflow_detector.detect_jersey_v5(
                                crop, jersey_number=jersey_number, conf=ocr_conf,
                                skip_preprocess=True,
                            )
                            if dets:
                                v5_ocr_detections.extend({**d, "timestamp": t} for d in dets)
                                v5_consecutive_misses = 0
                            else:
                                v5_consecutive_misses += 1
                    else:
                        v5_no_player_frames += 1
                if v5_ocr_detections:
                    phases_used.append("v5_ocr_detection")
            layer_timings["v5_ocr_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success" if v5_ocr_detections else "no_model_or_no_detections",
                "detections": len(v5_ocr_detections),
                "players_found": v5_players_found,
                "no_player_frames": v5_no_player_frames,
                "crops_processed": v5_total_crops,
                "exit_reason": v5_exit_reason or "completed",
                "crop_dimensions_sample": v5_crop_dims[:10],
            }
            LOGGER.info("Pipeline: v5 OCR found %d detections, %d crops processed, exit: %s",
                        len(v5_ocr_detections), v5_total_crops, v5_exit_reason or "completed")
        except Exception as exc:
            layer_timings["v5_ocr_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: v5 OCR layer failed (non-fatal): %s", exc)

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
            all_layer_dets_raw.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v5_ocr_universal"),
            })

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
            _few_detections = len(all_layer_dets) < 10
            if resolved_quality == "aggressive" or _is_dark_jersey or _few_detections:
                tc_instance = TemporalConsensus(
                    min_confirmations=1,
                    time_window=4.0,
                    confidence_threshold=0.15,
                )
            else:
                tc_instance = TemporalConsensus(
                    min_confirmations=2,
                    time_window=3.0,
                    confidence_threshold=0.3,
                )

            if all_layer_dets:
                confirmed_dets = tc_instance.filter_detections(
                    all_layer_dets, jersey_number
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

                # Use confirmed timestamps to filter detection_points
                if confirmed_dets:
                    confirmed_timestamps = set()
                    for cd in confirmed_dets:
                        confirmed_timestamps.add(cd.get("timestamp", 0))
                    pre_filter_count = len(detection_points)
                    # Widen timestamp matching to 1.5s (was 0.5s)
                    filtered_dp = [
                        dp for dp in detection_points
                        if any(
                            abs(dp.timestamp - cts) < 1.5
                            for cts in confirmed_timestamps
                        )
                    ]
                    # Only apply filter if it keeps at least some detections
                    if filtered_dp:
                        detection_points = filtered_dp
                        LOGGER.info(
                            "Pipeline: temporal consensus reduced detection_points "
                            "%d → %d",
                            pre_filter_count, len(detection_points),
                        )
                    else:
                        LOGGER.warning(
                            "Pipeline: temporal consensus timestamp match found 0 "
                            "of %d — keeping original detection_points",
                            pre_filter_count,
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
        # Threshold raised from 5→20 because improved v5 OCR now finds 10-15
        # detections that all chain into a single cluster at 5s gap.
        if 1 <= len(detection_points) <= 20 and _frame_timestamps:
            _existing_ts = {dp.timestamp for dp in detection_points}
            _motion_thresh = 30  # moderate-to-strong motion (basketball avg ~37)
            _supplement_count = 0
            for t in _frame_timestamps:
                if t in _existing_ts:
                    continue
                # Skip if within 3s of an existing detection (would merge anyway)
                if any(abs(t - ets) < 3.0 for ets in _existing_ts):
                    continue
                motion = motion_scores.get(t, 0)
                in_boundary = _in_audio_boundary(audio_result, t)
                if motion > _motion_thresh or (in_boundary and motion > 25):
                    pose = pose_results.get(t, _nearest_pose(pose_results, t)) if pose_results else {}
                    conf = motion / 100.0 * 0.5  # Lower confidence than direct OCR
                    if in_boundary:
                        conf = min(0.8, conf + 0.1)
                    detection_points.append(DetectionPoint(
                        timestamp=t,
                        confidence=conf,
                        jersey_visible=True,  # Inferred from OCR match elsewhere
                        jersey_number=jersey_number,
                        motion_score=motion,
                        pose_action=pose.get("action", "standing"),
                        crowd_energy=_get_crowd_energy(audio_result, t),
                    ))
                    _supplement_count += 1
            if _supplement_count:
                LOGGER.info("Pipeline: motion supplement added %d high-motion points "
                            "(total detection_points now %d)", _supplement_count, len(detection_points))

        # If jersey detection found nothing, generate detection points from motion/audio
        if not detection_points and _frame_timestamps:
            # Football: very low threshold (10) — each motion burst is likely a play
            # Other sports: moderate threshold (30)
            is_football = sport.lower() == "football"
            motion_threshold = 10 if is_football else 30
            LOGGER.info("Pipeline: no jersey detections, using motion/audio fallback (threshold=%d, sport=%s)", motion_threshold, sport)
            for t in _frame_timestamps:
                motion = motion_scores.get(t, 0)
                in_boundary = _in_audio_boundary(audio_result, t)
                if motion > motion_threshold or in_boundary:
                    pose = pose_results.get(t, _nearest_pose(pose_results, t)) if pose_results else {}
                    # Higher cap (0.7) + audio boundary bonus (0.15) to push
                    # above the 50-point "Cut" threshold in clip_extractor
                    conf = motion / 100.0 * 0.7
                    if in_boundary:
                        conf = min(1.0, conf + 0.15)
                    # Check for v4 outcome even in fallback mode
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

        # Apply football-specific clip splitting — break long clips into 5s sub-clips
        # This creates highlight-reel-friendly short clips from continuous plays
        if sport.lower() == "football" and clips:
            _split_clips: list = []
            _FOOTBALL_CLIP_LEN = 5.0  # Each sub-clip is ~5 seconds
            for clip in clips:
                duration = clip.end_time - clip.start_time
                if duration > _FOOTBALL_CLIP_LEN * 1.5:  # Only split if significantly longer
                    # Split into N sub-clips of ~5s each
                    n_parts = max(2, int(duration / _FOOTBALL_CLIP_LEN))
                    part_len = duration / n_parts
                    for i in range(n_parts):
                        sub_start = round(clip.start_time + i * part_len, 1)
                        sub_end = round(clip.start_time + (i + 1) * part_len, 1)
                        from copy import copy
                        sub_clip = copy(clip)
                        sub_clip.start_time = sub_start
                        sub_clip.end_time = sub_end
                        # Adjust score slightly for variety
                        sub_clip.score = max(5, clip.score - i * 2)
                        _split_clips.append(sub_clip)
                else:
                    if duration > FOOTBALL_MAX_CLIP:
                        clip.end_time = clip.start_time + FOOTBALL_MAX_CLIP
                    _split_clips.append(clip)
            clips = _split_clips
            LOGGER.info("Pipeline: football clip split → %d clips", len(clips))

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
            # Include enriched fields for frontend
            clip_dict["deadBallRatio"] = round(dead_ball_ratio, 2)
            clip_dict["scoreboardDetected"] = len(scoreboard_detections) > 0
            # Which detection layers found the jersey in this clip's time range
            clip_layers = set()
            for det in all_layer_dets:
                if clip.start_time - 1 <= det.get("timestamp", 0) <= clip.end_time + 1:
                    clip_layers.add(det.get("layer", "unknown"))
            clip_dict["detectionLayers"] = sorted(clip_layers)
            clips_out.append(clip_dict)

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
        except Exception:
            pass

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
            "cross_layer_agreements": cross_layer_agreements,
            "models_called": request_summary.get("models_called", []),
            "detections_per_model": request_summary.get("detections_per_model", {}),
            "clips_before_filter": len(clips),
            "clips_after_filter": len(clips_out),
            "memory_rss_mb": memory_rss_mb,
        }

        return {
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
        # Unload ALL models (not just request-specific) to free memory
        # for the next request. Models are re-loaded per-request anyway.
        import gc as _gc_post
        try:
            from app.services.roboflow_detector import roboflow_detector
            for _attr in dir(roboflow_detector):
                if _attr.endswith("_model") and getattr(roboflow_detector, _attr, None) is not None:
                    setattr(roboflow_detector, _attr, None)
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
        _gc_post.collect()
        try:
            import psutil as _ps_post
            _rss_after = _ps_post.Process().memory_info().rss / 1024 / 1024
            LOGGER.info("Pipeline: post-request cleanup done (RSS=%.0fMB)", _rss_after)
        except Exception:
            LOGGER.info("Pipeline: post-request cleanup done")
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
