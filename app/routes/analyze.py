# POST /analyze — consolidated detection endpoint
#
# Uses StreamingResponse with keepalive to prevent Railway proxy timeout.
# Sends {"keepalive": true, "elapsed": N}\n every 15s while processing,
# then sends the full result JSON as the final line.

from __future__ import annotations

import asyncio
import json
import logging
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

    async def _stream_with_keepalive() -> AsyncIterator[bytes]:
        """Run pipeline in background, yield keepalive lines until done."""
        result_holder: dict[str, Any] = {}
        error_holder: list[Exception] = []
        done_event = asyncio.Event()

        async def _run_pipeline() -> None:
            try:
                from app.services.analyze_pipeline import run_analyze_pipeline

                result = await run_analyze_pipeline(
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
                result_holder["data"] = result
            except Exception as exc:
                error_holder.append(exc)
            finally:
                done_event.set()

        # Start pipeline as concurrent task
        pipeline_task = asyncio.create_task(_run_pipeline())

        # Send keepalive lines until pipeline completes
        while not done_event.is_set():
            try:
                await asyncio.wait_for(done_event.wait(), timeout=_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                elapsed = round(time.perf_counter() - started_at, 1)
                keepalive = json.dumps({"keepalive": True, "elapsed": elapsed})
                LOGGER.debug("analyze.keepalive elapsed=%.1f", elapsed)
                yield (keepalive + "\n").encode()

        # Pipeline is done — yield final result
        if error_holder:
            LOGGER.exception("analyze.request_failed", exc_info=error_holder[0])
            error_json = json.dumps({"error": "Internal analysis error. See server logs for details."})
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
            "X-Accel-Buffering": "no",  # Disable nginx/Railway buffering
            "Cache-Control": "no-cache",
        },
    )
