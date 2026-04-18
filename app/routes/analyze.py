# POST /analyze — consolidated detection endpoint
#
# Uses StreamingResponse with keepalive to prevent Railway proxy timeout.
# Pipeline runs in a SEPARATE THREAD with its own event loop, so the main
# event loop stays free to yield keepalive lines every 15s.
# Last line of response is the actual JSON result.
#
# Client disconnect detection: when the client drops the connection,
# the cancel_event is set to signal the pipeline to abort. This prevents
# zombie pipeline threads from holding the lock and blocking new requests.
#
# POST /analyze-async + GET /analyze-jobs/{job_id} — async polling pattern
# for clients that hit serverless timeouts (Vercel Hobby = 60s). Uses the
# same _PIPELINE_LOCK so still only 1 pipeline runs at a time.

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# YouTube video ID pattern for preflight check
_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)


async def _youtube_preflight(video_url: str, access_token: str) -> dict | None:
    """Check video availability via YouTube Data API v3 using OAuth token.

    Returns dict with {id, title, duration} on success, None on failure.
    Non-fatal — any error is logged and returns None.
    """
    match = _YT_ID_RE.search(video_url)
    if not match:
        return None
    video_id = match.group(1)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "id": video_id,
                    "part": "snippet,contentDetails,status",
                    "fields": "items(id,snippet/title,contentDetails/duration,status/privacyStatus)",
                },
                headers={"Authorization": f"Bearer {access_token}"},
            )
        if resp.status_code != 200:
            LOGGER.warning("youtube_preflight: API returned %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        items = data.get("items", [])
        if not items:
            LOGGER.warning("youtube_preflight: video %s not found or private", video_id)
            return None

        item = items[0]
        return {
            "id": item.get("id"),
            "title": item.get("snippet", {}).get("title"),
            "duration": item.get("contentDetails", {}).get("duration"),
            "privacy": item.get("status", {}).get("privacyStatus"),
        }
    except Exception as exc:
        LOGGER.warning("youtube_preflight: failed — %s: %s", type(exc).__name__, str(exc)[:200])
        return None


# Keepalive interval in seconds — must be < Railway's ~300s proxy timeout
_KEEPALIVE_INTERVAL = 15

# Ensure only 1 pipeline runs at a time (thread-safe, unlike asyncio.Semaphore)
_PIPELINE_LOCK = threading.Lock()

# In-memory async job store for /analyze-async polling
_jobs: dict[str, dict] = {}
_JOB_TTL_SECONDS = 1800  # reap finished jobs after 30 min


@router.post("/analyze")
async def analyze(
    request: Request,
    analyze_request: AnalyzeRequest,
) -> Any:
    started_at = time.perf_counter()
    LOGGER.info(
        "analyze.request_started video_url=%s jersey_number=%s jersey_color=%s sport=%s position=%s "
        "audio=%s tracking=%s pose=%s time_range=%s-%s",
        (analyze_request.video_url or "")[:80],
        analyze_request.jersey_number,
        analyze_request.jersey_color,
        analyze_request.sport,
        analyze_request.position,
        analyze_request.enable_audio,
        analyze_request.enable_tracking,
        analyze_request.enable_pose,
        analyze_request.time_range_start,
        analyze_request.time_range_end,
    )

    # Check if detector is ready
    if not getattr(request.app.state, "detector_ready", False):
        detail = getattr(request.app.state, "startup_error", "Detector warm-up not complete.")
        LOGGER.warning("analyze.rejected_not_ready startup_error=%s", detail)
        return JSONResponse(
            status_code=503,
            content={"error": f"Detection service is not ready: {detail}"},
        )

    # Pre-flight: if Google access token provided, verify video with YouTube Data API v3
    google_token = analyze_request.google_access_token
    if google_token and analyze_request.video_url:
        preflight_info = await _youtube_preflight(analyze_request.video_url, google_token)
        if preflight_info:
            LOGGER.info("analyze.preflight video_id=%s title=%s duration=%s",
                        preflight_info.get("id", "?"),
                        (preflight_info.get("title") or "")[:60],
                        preflight_info.get("duration", "?"))

    # Capture request params for the pipeline thread
    pipeline_params = dict(
        video_url=analyze_request.video_url,
        video_path=analyze_request.video_path,
        jersey_number=analyze_request.jersey_number,
        jersey_color=analyze_request.jersey_color,
        sport=analyze_request.sport,
        position=analyze_request.position,
        time_range_start=analyze_request.time_range_start,
        time_range_end=analyze_request.time_range_end,
        enable_audio=analyze_request.enable_audio,
        enable_tracking=analyze_request.enable_tracking,
        enable_pose=analyze_request.enable_pose,
        quality_mode=analyze_request.quality_mode,
    )

    result_holder: dict[str, Any] = {}
    error_holder: list[Exception] = []
    pipeline_done = threading.Event()
    cancel_event = threading.Event()

    def _run_pipeline_in_thread() -> None:
        """Run the async pipeline in a dedicated thread with its own event loop."""
        # Try to acquire lock with timeout — if another pipeline is running,
        # wait up to 30s before giving up (prevents infinite queue)
        acquired = _PIPELINE_LOCK.acquire(timeout=30)
        if not acquired:
            error_holder.append(RuntimeError("Server busy — another analysis is in progress"))
            pipeline_done.set()
            return
        if cancel_event.is_set():
            _PIPELINE_LOCK.release()
            pipeline_done.set()
            return
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from app.services.analyze_pipeline import run_analyze_pipeline
                result = loop.run_until_complete(
                    run_analyze_pipeline(**pipeline_params, cancel_event=cancel_event)
                )
                result_holder["data"] = result
            except Exception as exc:
                LOGGER.exception("analyze.pipeline_thread_failed")
                error_holder.append(exc)
            finally:
                loop.close()
        finally:
            _PIPELINE_LOCK.release()
            pipeline_done.set()

    async def _stream_with_keepalive() -> AsyncIterator[bytes]:
        """Yield keepalive lines while pipeline runs in separate thread."""
        # Start pipeline in separate thread (its own event loop)
        pipeline_thread = threading.Thread(target=_run_pipeline_in_thread, daemon=True)
        pipeline_thread.start()

        # Yield keepalive lines using run_in_executor to avoid blocking main event loop
        main_loop = asyncio.get_event_loop()
        while not pipeline_done.is_set():
            finished = await main_loop.run_in_executor(
                None, pipeline_done.wait, _KEEPALIVE_INTERVAL
            )
            if not finished:
                # Check if client disconnected
                if await request.is_disconnected():
                    LOGGER.warning("analyze.client_disconnected elapsed=%.1f — cancelling pipeline",
                                   time.perf_counter() - started_at)
                    cancel_event.set()
                    # Wait briefly for pipeline to notice cancellation
                    await main_loop.run_in_executor(None, pipeline_done.wait, 5)
                    return
                elapsed = round(time.perf_counter() - started_at, 1)
                keepalive = json.dumps({"keepalive": True, "elapsed": elapsed})
                LOGGER.debug("analyze.keepalive elapsed=%.1f", elapsed)
                yield (keepalive + "\n").encode()

        # Pipeline is done — yield final result
        pipeline_thread.join(timeout=10)

        if error_holder:
            LOGGER.exception("analyze.request_failed", exc_info=error_holder[0])
            error_msg = str(error_holder[0])[:200]
            error_json = json.dumps({"error": f"Analysis error: {error_msg}"})
            yield (error_json + "\n").encode()
        elif "data" in result_holder:
            result = result_holder["data"]
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
            clip_count = len(result.get("clips", []))
            LOGGER.info(
                "analyze.request_completed clips=%d layer_used=%s elapsed_ms=%.2f",
                clip_count,
                result.get("layerUsed", "none"),
                elapsed_ms,
            )
            yield (json.dumps(result) + "\n").encode()
        else:
            yield (json.dumps({"error": "Pipeline completed with no result"}) + "\n").encode()

    return StreamingResponse(
        _stream_with_keepalive(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


# ── Async polling endpoints ────────────────────────────────────────────────

def _reap_stale_jobs() -> None:
    """Drop finished jobs older than _JOB_TTL_SECONDS to bound memory."""
    now = time.time()
    stale = [
        jid for jid, job in _jobs.items()
        if (now - job.get("created", now)) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        _jobs.pop(jid, None)


def _run_analyze_job_in_thread(job_id: str, pipeline_params: dict) -> None:
    """Background worker for /analyze-async. Mirrors the same lock+thread pattern as /analyze."""
    job = _jobs.get(job_id)
    if not job:
        return

    cancel_event = job["cancel_event"]
    started_at = time.perf_counter()

    job["status"] = "queued"
    job["message"] = "Waiting for pipeline lock"

    # Wait up to 5 minutes for the lock — long enough that a previous job finishes
    acquired = _PIPELINE_LOCK.acquire(timeout=300)
    if not acquired:
        job["status"] = "failed"
        job["error"] = "Server busy — another analysis is in progress"
        job["progress"] = 100
        return
    if cancel_event.is_set():
        _PIPELINE_LOCK.release()
        job["status"] = "failed"
        job["error"] = "Cancelled before start"
        job["progress"] = 100
        return

    try:
        job["status"] = "processing"
        job["progress"] = 10
        job["message"] = "Downloading video and loading models"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from app.services.analyze_pipeline import run_analyze_pipeline
            result = loop.run_until_complete(
                run_analyze_pipeline(**pipeline_params, cancel_event=cancel_event)
            )
            elapsed_s = round(time.perf_counter() - started_at, 2)
            clip_count = len(result.get("clips", []))
            job["status"] = "complete"
            job["progress"] = 100
            job["result"] = result
            job["message"] = f"Found {clip_count} highlights in {elapsed_s}s"
            LOGGER.info("analyze-async.complete job_id=%s clips=%d elapsed=%.2fs",
                        job_id, clip_count, elapsed_s)
        except Exception as exc:
            LOGGER.exception("analyze-async.failed job_id=%s", job_id)
            job["status"] = "failed"
            job["progress"] = 100
            job["error"] = str(exc)[:300]
            job["message"] = f"Analysis failed: {str(exc)[:200]}"
        finally:
            loop.close()
    finally:
        _PIPELINE_LOCK.release()


@router.post("/analyze-async")
async def analyze_async(
    request: Request,
    analyze_request: AnalyzeRequest,
) -> JSONResponse:
    """Start an analyze job in the background. Returns job_id immediately for polling."""
    if not getattr(request.app.state, "detector_ready", False):
        detail = getattr(request.app.state, "startup_error", "Detector warm-up not complete.")
        return JSONResponse(
            status_code=503,
            content={"error": f"Detection service is not ready: {detail}"},
        )

    _reap_stale_jobs()

    # Pre-flight: verify video with YouTube Data API v3 if token provided
    google_token = analyze_request.google_access_token
    if google_token and analyze_request.video_url:
        preflight_info = await _youtube_preflight(analyze_request.video_url, google_token)
        if preflight_info:
            LOGGER.info("analyze-async.preflight video_id=%s title=%s duration=%s",
                        preflight_info.get("id", "?"),
                        (preflight_info.get("title") or "")[:60],
                        preflight_info.get("duration", "?"))

    pipeline_params = dict(
        video_url=analyze_request.video_url,
        video_path=analyze_request.video_path,
        jersey_number=analyze_request.jersey_number,
        jersey_color=analyze_request.jersey_color,
        sport=analyze_request.sport,
        position=analyze_request.position,
        time_range_start=analyze_request.time_range_start,
        time_range_end=analyze_request.time_range_end,
        enable_audio=analyze_request.enable_audio,
        enable_tracking=analyze_request.enable_tracking,
        enable_pose=analyze_request.enable_pose,
        quality_mode=analyze_request.quality_mode,
    )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "result": None,
        "error": None,
        "created": time.time(),
        "cancel_event": threading.Event(),
    }

    # Start a real background thread (not BackgroundTasks) so the response returns immediately
    thread = threading.Thread(
        target=_run_analyze_job_in_thread,
        args=(job_id, pipeline_params),
        daemon=True,
        name=f"analyze-job-{job_id[:8]}",
    )
    thread.start()

    LOGGER.info("analyze-async.queued job_id=%s sport=%s jersey=%s",
                job_id, analyze_request.sport, analyze_request.jersey_number)
    return JSONResponse(content={"job_id": job_id, "status": "queued"})


@router.get("/analyze-jobs/{job_id}")
async def get_analyze_job(job_id: str) -> JSONResponse:
    """Poll an analyze job. Returns status, progress, and result (when complete)."""
    if job_id not in _jobs:
        return JSONResponse(status_code=404, content={"status": "not_found", "job_id": job_id})
    job = _jobs[job_id]
    return JSONResponse(content={
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job.get("message", ""),
        "result": job.get("result"),
        "error": job.get("error"),
    })
