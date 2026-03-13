# POST /detect

from __future__ import annotations

import gc
import logging
import os
from typing import Any

import psutil
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.detect import DetectRequest, DetectionFrame
from app.services.detection_service import DetectionService, get_detection_service

LOGGER = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=list[DetectionFrame])
async def detect(
    request: Request,
    detect_request: DetectRequest,
    detection_service: DetectionService = Depends(get_detection_service),
) -> Any:
    if not getattr(request.app.state, "detector_ready", False):
        detail = getattr(
            request.app.state,
            "startup_error",
            "Detector warm-up has not completed successfully.",
        )
        return JSONResponse(
            status_code=503,
            content={"error": f"Detection service is not ready: {detail}"},
        )

    # Memory guard — prevent OOM crash; return 503 instead of letting Railway kill the process
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        LOGGER.info("CLIPT: memory at request start: %.1f MB", mem_mb)
        if mem_mb > 700:
            gc.collect()
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            if mem_mb > 700:
                LOGGER.warning("CLIPT: memory pressure %.1f MB — returning 503", mem_mb)
                return JSONResponse(
                    status_code=503,
                    content={"error": "Server is under memory pressure. Please try again in 30 seconds."},
                )
    except Exception:
        pass  # psutil unavailable — skip guard

    try:
        detections = await run_in_threadpool(detection_service.detect, detect_request)
    except Exception:  # pragma: no cover
        LOGGER.exception("Detection request failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal detection error. See server logs for details."},
        )
    return detections
