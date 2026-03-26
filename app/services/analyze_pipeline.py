# Analyze pipeline orchestrator — chains all detection layers
#
# DETECTION CALL CHAIN:
# POST /analyze (app/routes/analyze.py)
#   → run_analyze_pipeline() [this file]
#     Step 1: Acquire video (YouTube download or direct URL download)
#     Step 2: Ali's ensemble — highest priority
#     Step 3: Frame extraction
#     Step 4: Motion scoring (optical flow)
#     Step 5: Audio analysis (whistle + crowd energy)
#     Step 6: Player tracking (BoT-SORT)
#     Step 7: Pose estimation (YOLO11n-pose)
#     Step 7.5a: Universal v2 OCR (jersey_number_universal_v1, mAP50 0.995)
#     Step 7.5b: v3 OCR pipeline (12 models — Ali replacement layer)
#     Step 7.5c: v2 sport-specific models (basketball/football/lacrosse)
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
from app.services.youtube_proxy import (
    download_youtube,
    extract_audio,
    get_video_duration,
    is_youtube_url,
)

LOGGER = logging.getLogger(__name__)

# Football-specific overrides
FOOTBALL_CONF_THRESHOLD = 0.15  # Lower than default 0.35
FOOTBALL_MIN_CLIP = 3.0
FOOTBALL_MAX_CLIP = 12.0


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
) -> dict[str, Any]:
    """Run the full analysis pipeline.

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
    start_time = time.perf_counter()
    phases_used: list[str] = []
    local_video_path: Path | None = None
    tmp_dir: Path | None = None
    frames_processed = 0
    youtube_strategy_used: str | None = None

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
                local_video_path = await download_youtube(
                    video_url,
                    start_time=time_range_start,
                    end_time=time_range_end,
                    yt_dlp_binary=settings.yt_dlp_binary,
                    ffmpeg_binary=settings.ffmpeg_binary,
                )
                phases_used.append("youtube_download")
                youtube_strategy_used = "download_success"
                layer_timings["youtube_download"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "success"}
                LOGGER.info("Pipeline: YouTube video downloaded to %s", local_video_path)
            except Exception as exc:
                layer_timings["youtube_download"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "failed", "error": str(exc)}
                youtube_strategy_used = "all_failed"
                LOGGER.error("Pipeline: YouTube download failed: %s", exc)
                return _error_response(f"YouTube download failed: {exc}", time.perf_counter() - start_time)
        elif video_url:
            # Direct URL — download it first
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
                        # Fall through to existing detection pipeline which handles URLs
                        local_video_path = None
            except Exception as exc:
                LOGGER.warning("Pipeline: direct download failed, will pass URL to detector: %s", exc)
                local_video_path = None
        elif video_path:
            local_video_path = Path(video_path)

        # Get video duration
        video_duration = 0.0
        if local_video_path and local_video_path.exists():
            from app.services.detection_runtime import PipelineSettings
            settings = PipelineSettings()
            video_duration = get_video_duration(local_video_path, settings.ffprobe_binary)
            LOGGER.info("Pipeline: video duration = %.1fs", video_duration)

        # ── Step 2: Run existing jersey detection (Ali's ensemble) ─────
        jersey_detections: list[dict] = []
        t0 = time.perf_counter()
        ali_status = "not_run"
        try:
            # If we already downloaded the video (YouTube or direct URL),
            # pass the local file to Ali — avoids Ali re-downloading.
            # If no local file, pass the original URL for Ali to handle.
            if local_video_path and local_video_path.exists():
                LOGGER.info("Pipeline: passing local file to Ali: %s (%d bytes)",
                            local_video_path, local_video_path.stat().st_size)
                ali_video_url = None
                ali_video_path = str(local_video_path)
            else:
                LOGGER.info("Pipeline: passing URL to Ali: %s", (video_url or video_path or "")[:80])
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
                ali_status = "working"
            else:
                ali_status = "no_detections"
            layer_timings["ali_jersey_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": ali_status,
                "detections": len(jersey_detections),
                "input_type": "local_file" if ali_video_path else "url",
                "input": (ali_video_path or ali_video_url or "")[:100],
            }
            LOGGER.info("Pipeline: jersey detection found %d frames", len(jersey_detections))
        except Exception as exc:
            ali_status = "error"
            layer_timings["ali_jersey_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "error",
                "error": str(exc)[:200],
                "detections": 0,
            }
            LOGGER.error("Pipeline: jersey detection failed: %s", exc)

        # ── Step 3: Extract frames for additional analysis ───────────────
        frames: list[tuple[float, np.ndarray]] = []
        if local_video_path and local_video_path.exists():
            t0 = time.perf_counter()
            try:
                frames = _extract_frames(local_video_path, fps=2, sport=sport)
                frames_processed = len(frames)
                layer_timings["frame_extraction"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "success", "frames": frames_processed}
                LOGGER.info("Pipeline: extracted %d frames", len(frames))
            except Exception as exc:
                layer_timings["frame_extraction"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)}
                LOGGER.warning("Pipeline: frame extraction failed: %s", exc)

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

        # ── Step 7.5a: Universal v2 OCR (best single model) ──────────
        universal_v2_detections: list[dict] = []
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            if roboflow_detector.jersey_number_universal_v1_model is not None:
                LOGGER.info("Pipeline: Universal v2 layer running")
                for t, frame in frames:
                    dets = roboflow_detector._run_universal_ocr(frame, jersey_number, conf=0.2)
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

        # ── Step 7.5b: v3 OCR pipeline (Ali replacement layer) ───────
        v3_ocr_detections: list[dict] = []
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            LOGGER.info("Pipeline: v3 OCR layer running for sport=%s", sport)
            for t, frame in frames:
                # Full v3 pipeline: player_isolator → color → number_region → OCR
                dets = roboflow_detector.detect_with_player_crops(
                    frame, jersey_number=jersey_number, sport=sport, conf=0.2,
                )
                # Only keep v3 layer detections (exclude v1/v2 that detect_with_player_crops also runs)
                v3_only = [d for d in dets if d.get("layer", "").startswith("v3_")]
                v3_ocr_detections.extend({**d, "timestamp": t} for d in v3_only)
            if v3_ocr_detections:
                phases_used.append("v3_ocr_detection")
            layer_timings["v3_ocr_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success" if v3_ocr_detections else "no_detections",
                "detections": len(v3_ocr_detections),
            }
            LOGGER.info("Pipeline: v3 OCR found %d detections", len(v3_ocr_detections))
        except Exception as exc:
            layer_timings["v3_ocr_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: v3 OCR layer failed (non-fatal): %s", exc)

        # ── Step 7.5c: v2 sport-specific models ──────────────────────
        v2_sport_detections: list[dict] = []
        t0 = time.perf_counter()
        try:
            from app.services.roboflow_detector import roboflow_detector
            roboflow_detector.load()

            LOGGER.info("Pipeline: v2 sport-specific layer running for sport=%s", sport)
            for t, frame in frames:
                dets = roboflow_detector._run_sport_specific_v2(frame, jersey_number, sport, conf=0.2)
                v2_sport_detections.extend({**d, "timestamp": t} for d in dets)

            # Also run v1 fallback detectors
            for t, frame in frames:
                dets = roboflow_detector.detect_football_digits(frame, jersey_number, conf=0.2)
                v2_sport_detections.extend({**d, "timestamp": t} for d in dets)
                dets = roboflow_detector.detect_football_tracker(frame, jersey_number, conf=0.2)
                v2_sport_detections.extend({**d, "timestamp": t} for d in dets)

            if v2_sport_detections:
                phases_used.append("v2_sport_detection")
            layer_timings["v2_sport_detection"] = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "status": "success" if v2_sport_detections else "no_detections",
                "detections": len(v2_sport_detections),
            }
            LOGGER.info("Pipeline: v2 sport-specific found %d detections", len(v2_sport_detections))
        except Exception as exc:
            layer_timings["v2_sport_detection"] = {"elapsed_ms": round((time.perf_counter() - t0) * 1000), "status": "error", "error": str(exc)[:200]}
            LOGGER.warning("Pipeline: v2 sport-specific layer failed (non-fatal): %s", exc)

        # ── Step 8: Cross-layer validation + merge ────────────────────
        detection_points: list[DetectionPoint] = []
        cross_layer_agreements: list[dict] = []

        # Collect ALL detections by timestamp (within 0.5s buckets)
        all_layer_dets: list[dict] = []
        for det in jersey_detections:
            all_layer_dets.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": jersey_number,
                "layer": "ali_ensemble",
            })
        for det in universal_v2_detections:
            all_layer_dets.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v2_universal_v1"),
            })
        for det in v3_ocr_detections:
            all_layer_dets.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v3_ocr"),
            })
        for det in v2_sport_detections:
            all_layer_dets.append({
                "timestamp": det.get("timestamp", 0),
                "confidence": det.get("confidence", 0),
                "number_detected": det.get("number_detected", jersey_number),
                "layer": det.get("layer", "v2_sport"),
            })

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
            number_detected = jersey_number

            # Cross-layer confidence boosts
            bonus = 0.0
            high_confidence = False
            needs_confirmation = True

            if "ali_ensemble" in layers_present and len(layers_present) >= 2:
                # Ali + any other layer agrees
                bonus += 0.2
                needs_confirmation = False

            if any("v3_ocr_primary" in l for l in layers_present) and \
               any("v2_universal" in l for l in layers_present):
                # v3 primary + v2 universal agree
                bonus += 0.15
                needs_confirmation = False

            if len(layers_present) >= 3:
                high_confidence = True
                needs_confirmation = False

            final_conf = min(1.0, best_conf + bonus)

            if len(layers_present) >= 2:
                cross_layer_agreements.append({
                    "timestamp": round(bucket_ts, 1),
                    "number_detected": number_detected,
                    "layers_agreed": sorted(layers_present),
                    "final_confidence": round(final_conf, 3),
                    "high_confidence": high_confidence,
                })

            detection_points.append(DetectionPoint(
                timestamp=bucket_ts,
                confidence=final_conf,
                jersey_visible=True,
                jersey_number=jersey_number,
                motion_score=motion_scores.get(bucket_ts, _nearest_value(motion_scores, bucket_ts)),
                pose_action=pose_results.get(bucket_ts, _nearest_pose(pose_results, bucket_ts)).get("action", "standing") if pose_results else "standing",
                crowd_energy=_get_crowd_energy(audio_result, bucket_ts),
                tracking_id=tracking_result.target_track_id if tracking_result else None,
            ))

        # ── Temporal consensus filtering ─────────────────────────────
        # Filter through temporal consensus to eliminate false positives.
        tc_stats = {"raw_detections": 0, "confirmed_detections": 0,
                    "filtered_out": 0, "cross_layer_confirmed": 0}
        try:
            from app.services.temporal_consensus import temporal_consensus

            if all_layer_dets:
                confirmed_dets = temporal_consensus.filter_detections(
                    all_layer_dets, jersey_number
                )
                confirmed_dets = temporal_consensus.cross_layer_boost(
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
        except Exception as exc:
            LOGGER.warning("Pipeline: temporal consensus failed (non-fatal): %s", exc)

        # Log detection source breakdown
        ali_count = len(jersey_detections)
        univ_count = len(universal_v2_detections)
        v3_count = len(v3_ocr_detections)
        v2_sport_count = len(v2_sport_detections)
        LOGGER.info(
            "Layer results — Ali: %d, Universal_v1: %d, V3_primary: %d, V2_sport: %d, After_consensus: %d",
            ali_count, univ_count, v3_count, v2_sport_count, len(detection_points),
        )
        if ali_count == 0 and (univ_count + v3_count + v2_sport_count) > 0:
            LOGGER.info("Pipeline: Ali found 0 — other layers saved detection!")
        total_raw = ali_count + univ_count + v3_count + v2_sport_count
        if total_raw > 0 and len(detection_points) == 0:
            LOGGER.warning(
                "Pipeline: %d raw detections across all layers → 0 detection_points. "
                "Temporal consensus may be too aggressive (min_confirmations=%d, time_window=%.1fs).",
                total_raw,
                tc_stats.get("raw_detections", 0),
                2.0,
            )

        # If jersey detection found nothing but we have frames, generate detection points from motion/audio
        if not detection_points and frames:
            LOGGER.info("Pipeline: no jersey detections, using motion/audio fallback")
            for t, frame in frames:
                motion = motion_scores.get(t, 0)
                if motion > 30 or _in_audio_boundary(audio_result, t):
                    pose = pose_results.get(t, _nearest_pose(pose_results, t)) if pose_results else {}
                    detection_points.append(DetectionPoint(
                        timestamp=t,
                        confidence=motion / 100.0 * 0.5,  # Lower confidence for non-jersey
                        jersey_visible=False,
                        motion_score=motion,
                        pose_action=pose.get("action", "standing"),
                        crowd_energy=_get_crowd_energy(audio_result, t),
                    ))

        # Extract and rank clips
        clips = extract_clips(
            detections=detection_points,
            audio_result=audio_result if audio_result.has_audio else None,
            sport=sport,
            position=position,
            video_duration=video_duration,
        )

        # Apply football-specific clip duration limits
        if sport.lower() == "football":
            for clip in clips:
                duration = clip.end_time - clip.start_time
                if duration > FOOTBALL_MAX_CLIP:
                    clip.end_time = clip.start_time + FOOTBALL_MAX_CLIP

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
                frames=frames,
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
            clips_out.append({
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
            })

        # ── Build debug field ──────────────────────────────────────────
        ali_working = ali_status == "working"
        layers_that_contributed = [
            layer for layer in phases_used
            if layer not in ("youtube_download", "frame_extraction")
        ]
        debug = {
            "ali_detections": len(jersey_detections),
            "universal_v2_detections": len(universal_v2_detections),
            "v3_ocr_detections": len(v3_ocr_detections),
            "v2_sport_detections": len(v2_sport_detections),
            "combined_detections": len(detection_points),
            "ali_working": ali_working,
            "ali_status": ali_status,
            "youtube_strategy_used": youtube_strategy_used,
            "total_elapsed_ms": round(elapsed * 1000),
            "layers_that_contributed": layers_that_contributed,
            "layer_breakdown": layer_timings,
            "temporal_consensus": tc_stats,
            "cross_layer_agreements": cross_layer_agreements,
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
) -> list[tuple[float, np.ndarray]]:
    """Extract frames from video at given FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(video_fps / fps))
    frames: list[tuple[float, np.ndarray]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps

            # Football-specific: upscale frames for better OCR
            if sport.lower() == "football":
                h, w = frame.shape[:2]
                if w < 1920:
                    scale = min(2.0, 1920 / w)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    # Apply CLAHE contrast enhancement
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            frames.append((timestamp, frame))

        frame_idx += 1

    cap.release()
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


def _error_response(message: str, elapsed: float) -> dict:
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
            "ali_detections": 0,
            "universal_v2_detections": 0,
            "v3_ocr_detections": 0,
            "v2_sport_detections": 0,
            "combined_detections": 0,
            "ali_working": False,
            "ali_status": "not_run",
            "youtube_strategy_used": None,
            "total_elapsed_ms": round(elapsed * 1000),
            "layers_that_contributed": [],
            "layer_breakdown": {},
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
