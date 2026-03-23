# POST /analyze — consolidated detection endpoint

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

LOGGER = logging.getLogger(__name__)
router = APIRouter()


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
        )

    except Exception:
        LOGGER.exception("analyze.request_failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal analysis error. See server logs for details."},
        )

    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    clip_count = len(result.get("clips", []))
    LOGGER.info(
        "analyze.request_completed clips=%d layer_used=%s elapsed_ms=%.2f",
        clip_count,
        result.get("layerUsed", "none"),
        elapsed_ms,
    )

    return result
