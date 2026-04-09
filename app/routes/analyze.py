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

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# Keepalive interval in seconds — must be < Railway's ~300s proxy timeout
_KEEPALIVE_INTERVAL = 15

# Ensure only 1 pipeline runs at a time (thread-safe, unlike asyncio.Semaphore)
_PIPELINE_LOCK = threading.Lock()


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
