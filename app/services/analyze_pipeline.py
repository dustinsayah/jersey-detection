# Analyze pipeline orchestrator — chains all detection layers

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

    try:
        # ── Step 1: Acquire video ────────────────────────────────────────
        if video_url and is_youtube_url(video_url):
            LOGGER.info("Pipeline: downloading YouTube video")
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
                LOGGER.info("Pipeline: YouTube video downloaded to %s", local_video_path)
            except Exception as exc:
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

        # ── Step 2: Run existing jersey detection ────────────────────────
        jersey_detections: list[dict] = []
        try:
            jersey_detections = _run_jersey_detection(
                video_url=video_url if local_video_path is None else None,
                video_path=str(local_video_path) if local_video_path else None,
                jersey_number=jersey_number,
                jersey_color=jersey_color,
                sport=sport,
                position=position,
            )
            if jersey_detections:
                phases_used.append("jersey_detection")
            LOGGER.info("Pipeline: jersey detection found %d frames", len(jersey_detections))
        except Exception as exc:
            LOGGER.error("Pipeline: jersey detection failed: %s", exc)

        # ── Step 3: Extract frames for additional analysis ───────────────
        frames: list[tuple[float, np.ndarray]] = []
        if local_video_path and local_video_path.exists():
            try:
                frames = _extract_frames(local_video_path, fps=2, sport=sport)
                frames_processed = len(frames)
                LOGGER.info("Pipeline: extracted %d frames", len(frames))
            except Exception as exc:
                LOGGER.warning("Pipeline: frame extraction failed: %s", exc)

        # ── Step 4: Motion scoring ───────────────────────────────────────
        motion_scores: dict[float, float] = {}
        try:
            if len(frames) >= 2:
                for i in range(len(frames) - 1):
                    t, prev_frame = frames[i]
                    t_next, curr_frame = frames[i + 1]
                    score = compute_motion_score(prev_frame, curr_frame)
                    motion_scores[t_next] = score.score
                phases_used.append("motion_scoring")
                LOGGER.info("Pipeline: computed %d motion scores", len(motion_scores))
        except Exception as exc:
            LOGGER.warning("Pipeline: motion scoring failed: %s", exc)

        # ── Step 5: Audio analysis ───────────────────────────────────────
        audio_result = AudioAnalysisResult(has_audio=False)
        if enable_audio and local_video_path and local_video_path.exists():
            try:
                from app.services.audio_analyzer import analyze_audio
                from app.services.detection_runtime import PipelineSettings
                settings = PipelineSettings()
                audio_path = extract_audio(local_video_path, settings.ffmpeg_binary)
                if audio_path:
                    audio_result = analyze_audio(audio_path)
                    if audio_result.has_audio:
                        phases_used.append("audio_analysis")
                    LOGGER.info("Pipeline: audio analysis complete, %d events, %d boundaries",
                                len(audio_result.events), len(audio_result.play_boundaries))
            except Exception as exc:
                LOGGER.warning("Pipeline: audio analysis failed: %s", exc)

        # ── Step 6: Player tracking ──────────────────────────────────────
        tracking_result = None
        if enable_tracking and frames:
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
                LOGGER.info("Pipeline: tracking found %d tracks, target=%s",
                            len(tracking_result.tracks), tracking_result.target_track_id)
            except Exception as exc:
                LOGGER.warning("Pipeline: player tracking failed: %s", exc)

        # ── Step 7: Pose estimation ──────────────────────────────────────
        pose_results: dict[float, dict] = {}
        if enable_pose and frames:
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
                LOGGER.info("Pipeline: pose estimated for %d frames", len(pose_results))
            except Exception as exc:
                LOGGER.warning("Pipeline: pose estimation failed: %s", exc)

        # ── Step 7.5: Roboflow parallel detection layer ─────────────────
        # Only use Roboflow for football/american_football — basketball model
        # has low accuracy (mAP50: 0.10), so basketball falls through to
        # Ali's ensemble + Claude Vision instead.
        roboflow_detections: list[dict] = []
        if sport.lower() in ("football", "american_football"):
            try:
                from app.services.roboflow_detector import roboflow_detector

                for t, frame in frames:
                    dets = roboflow_detector.detect_with_player_crops(
                        frame,
                        jersey_number=jersey_number,
                        sport=sport,
                        conf=0.2,
                    )
                    roboflow_detections.extend(
                        {**d, "timestamp": t} for d in dets
                    )

                if roboflow_detections:
                    phases_used.append("roboflow_detection")
                LOGGER.info(
                    "Pipeline: Roboflow layer found %d detections", len(roboflow_detections)
                )
            except Exception as exc:
                LOGGER.warning("Pipeline: Roboflow layer failed (non-fatal): %s", exc)
        else:
            LOGGER.info(
                "Pipeline: Roboflow layer skipped for sport=%s — using Ali's ensemble + Claude Vision",
                sport,
            )

        # ── Step 8: Build detection points and extract clips ─────────────
        detection_points: list[DetectionPoint] = []

        # From Ali's jersey detections
        for det in jersey_detections:
            ts = det.get("timestamp", 0)
            detection_points.append(DetectionPoint(
                timestamp=ts,
                confidence=det.get("confidence", 0),
                jersey_visible=True,
                jersey_number=jersey_number,
                motion_score=motion_scores.get(ts, _nearest_value(motion_scores, ts)),
                pose_action=pose_results.get(ts, _nearest_pose(pose_results, ts)).get("action", "standing") if pose_results else "standing",
                crowd_energy=_get_crowd_energy(audio_result, ts),
                tracking_id=tracking_result.target_track_id if tracking_result else None,
            ))

        # From Roboflow detections (merge, deduplicate by timestamp proximity)
        for det in roboflow_detections:
            ts = det.get("timestamp", 0)
            # Skip if Ali already found something within 0.5s
            already_covered = any(
                abs(dp.timestamp - ts) < 0.5 for dp in detection_points
            )
            if already_covered:
                # Boost confidence of existing point if Roboflow agrees
                for dp in detection_points:
                    if abs(dp.timestamp - ts) < 0.5:
                        rf_conf = det.get("confidence", 0)
                        # Agreement bonus: boost by 20% of Roboflow confidence
                        dp.confidence = min(1.0, dp.confidence + rf_conf * 0.2)
                        break
            else:
                detection_points.append(DetectionPoint(
                    timestamp=ts,
                    confidence=det.get("confidence", 0) * 0.85,  # Slight discount for RF-only
                    jersey_visible=True,
                    jersey_number=jersey_number,
                    motion_score=motion_scores.get(ts, _nearest_value(motion_scores, ts)),
                    pose_action=pose_results.get(ts, _nearest_pose(pose_results, ts)).get("action", "standing") if pose_results else "standing",
                    crowd_energy=_get_crowd_energy(audio_result, ts),
                    tracking_id=tracking_result.target_track_id if tracking_result else None,
                ))

        # Log detection source breakdown
        ali_count = len(jersey_detections)
        rf_count = len(roboflow_detections)
        LOGGER.info(
            "Pipeline: detection merge — Ali=%d, Roboflow=%d, Combined=%d",
            ali_count, rf_count, len(detection_points),
        )
        if ali_count == 0 and rf_count > 0:
            LOGGER.info("Pipeline: Ali found 0 — Roboflow saved detection!")

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

        return {
            "clips": clips_out,
            "layerUsed": layer_used,
            "elapsed": round(elapsed, 1),
            "videoDuration": round(video_duration, 1),
            "framesProcessed": frames_processed,
            "audioEvents": audio_events_out,
            "playerTracks": player_tracks_out,
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

        request = DetectRequest(**req_data)
        service = DetectionService()
        detections = service.detect(request)

        # Convert DetectionFrame objects to dicts
        results = []
        for det in detections:
            d = {
                "timestamp": det.timestamp,
                "confidence": det.confidence,
            }
            if hasattr(det, "bbox") and det.bbox:
                d["x1"] = det.bbox.x1
                d["y1"] = det.bbox.y1
                d["x2"] = det.bbox.x2
                d["y2"] = det.bbox.y2
            results.append(d)

        return results

    except Exception as exc:
        LOGGER.error("Jersey detection failed: %s", exc)
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
        "error": message,
    }
