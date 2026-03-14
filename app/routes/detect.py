# POST /detect

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.schemas.detect import DetectRequest, DetectionFrame
from app.services.detection_service import DetectionService, get_detection_service

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def _source_kind(detect_request: DetectRequest) -> str:
    if detect_request.video_bytes_b64 is not None:
        return "video_bytes"
    if detect_request.video_path:
        return "video_path"
    return "video_url"


@router.post("/detect", response_model=list[DetectionFrame])
async def detect(
    request: Request,
    detect_request: DetectRequest,
    detection_service: DetectionService = Depends(get_detection_service),
) -> Any:
    started_at = time.perf_counter()
    source_kind = _source_kind(detect_request)
    LOGGER.info(
        "detect.request_started source=%s jersey_number=%s jersey_color=%s sport=%s position=%s",
        source_kind,
        detect_request.jersey_number,
        detect_request.jersey_color,
        detect_request.sport,
        detect_request.position,
    )
    if not getattr(request.app.state, "detector_ready", False):
        detail = getattr(
            request.app.state,
            "startup_error",
            "Detector warm-up has not completed successfully.",
        )
        LOGGER.warning(
            "detect.rejected_not_ready source=%s startup_error=%s",
            source_kind,
            detail,
        )
        return JSONResponse(
            status_code=503,
            content={"error": f"Detection service is not ready: {detail}"},
        )
    try:
        detections = await run_in_threadpool(detection_service.detect, detect_request)
    except Exception:  # pragma: no cover
        LOGGER.exception("Detection request failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal detection error. See server logs for details."},
        )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    LOGGER.info(
        "detect.request_completed source=%s result_count=%s elapsed_ms=%.2f",
        source_kind,
        len(detections),
        elapsed_ms,
    )
    return detections
