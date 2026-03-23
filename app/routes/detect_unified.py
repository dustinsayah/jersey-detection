# POST /detect-unified
# Multi-layer detection: Ali's ensemble → motion fallback → YOLO fallback

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.schemas.detect import DetectRequest, DetectionFrame
from app.services.detection_service import DetectionService, get_detection_service

LOGGER = logging.getLogger(__name__)
router = APIRouter()

# In-memory async job store
_jobs: dict[str, dict] = {}


class UnifiedDetectRequest(BaseModel):
    videoUrl: str
    jerseyNumber: int | None = None
    jerseyColor: str | None = "white"
    sport: str | None = "football"
    position: str | None = None
    timeRangeStart: float | None = None
    timeRangeEnd: float | None = None


@router.get("/health-unified")
async def health_unified(request: Request) -> JSONResponse:
    base_ready = getattr(request.app.state, "detector_ready", False)
    return JSONResponse(content={
        "status": "ok" if base_ready else "degraded",
        "version": "unified-v2",
        "aliEnsembleAvailable": base_ready,
        "layers": _get_available_layers(base_ready),
        "modelFiles": _check_model_files(),
    })


@router.get("/layers")
async def list_layers(request: Request) -> JSONResponse:
    base_ready = getattr(request.app.state, "detector_ready", False)
    return JSONResponse(content={
        "ali_ensemble": {
            "available": base_ready,
            "description": "Jersey number tracking via ML ensemble",
        },
        "motion": {
            "available": True,
            "description": "Optical flow motion detection (no ML models)",
        },
        "yolo_player": {
            "available": _check_yolo(),
            "description": "Player + ball proximity detection",
        },
    })


@router.post("/detect-unified")
async def detect_unified(
    request: Request,
    body: UnifiedDetectRequest,
) -> Any:
    """Multi-layer detection with fallback.
    1. Ali's ensemble (highest accuracy)
    2. Motion detection (optical flow)
    3. YOLO player+ball (color matching)
    Returns results with layer_used field.
    """
    start = time.time()
    all_detections: list[dict] = []
    layers_used: list[str] = []

    # ── LAYER 1: Ali's ensemble reader ──────────────────────────────────
    ali_ready = getattr(request.app.state, "detector_ready", False)
    if ali_ready and body.jerseyNumber is not None:
        try:
            LOGGER.info("detect-unified: running Layer 1 (Ali ensemble)")
            ali_req = DetectRequest(
                videoUrl=body.videoUrl,
                jerseyNumber=body.jerseyNumber,
                jerseyColor=body.jerseyColor or "white",
                sport=body.sport or "football",
                position=body.position,
            )
            svc = get_detection_service()
            detections = await run_in_threadpool(svc.detect, ali_req)
            if detections:
                for d in detections:
                    entry = d.model_dump() if hasattr(d, "model_dump") else d.dict()
                    entry["layer"] = "ali_ensemble"
                    all_detections.append(entry)
                layers_used.append("ali_ensemble")
                LOGGER.info("Layer 1 (Ali): %d detections", len(detections))
        except Exception as e:
            LOGGER.warning("Layer 1 (Ali) failed: %s", e)

    # ── LAYER 2: Motion detection ───────────────────────────────────────
    if len(all_detections) == 0:
        try:
            LOGGER.info("detect-unified: Layer 1 returned 0 — running Layer 2 (motion)")
            from layers.motion_detector import detect_motion_moments

            motion = await run_in_threadpool(
                detect_motion_moments,
                body.videoUrl,
                body.timeRangeStart,
                body.timeRangeEnd,
            )
            if motion:
                all_detections.extend(motion)
                layers_used.append("motion")
                LOGGER.info("Layer 2 (Motion): %d moments", len(motion))
        except Exception as e:
            LOGGER.warning("Layer 2 (Motion) failed: %s", e)

    # ── LAYER 3: YOLO player+ball ───────────────────────────────────────
    if len(all_detections) < 3:
        try:
            LOGGER.info("detect-unified: running Layer 3 (YOLO player+ball)")
            from layers.yolo_player import detect_with_yolo

            yolo = await run_in_threadpool(
                detect_with_yolo,
                body.videoUrl,
                body.jerseyColor or "white",
                body.timeRangeStart,
                body.timeRangeEnd,
            )
            if yolo:
                all_detections.extend(yolo)
                layers_used.append("yolo_player")
                LOGGER.info("Layer 3 (YOLO): %d detections", len(yolo))
        except Exception as e:
            LOGGER.warning("Layer 3 (YOLO) failed: %s", e)

    # ── Fuse & return ───────────────────────────────────────────────────
    final = _fuse_detections(all_detections)
    elapsed = round(time.time() - start, 2)
    LOGGER.info(
        "detect-unified complete: %d results via %s in %ss",
        len(final), layers_used, elapsed,
    )

    return {
        "detections": final,
        "layers_used": layers_used,
        "elapsed": elapsed,
        "count": len(final),
    }


@router.post("/detect-async")
async def detect_async(body: UnifiedDetectRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id, "status": "pending", "progress": 0,
        "message": "Starting...", "detections": [], "created": time.time(),
    }
    background_tasks.add_task(_run_async_job, job_id, body)
    return JSONResponse(content={"jobId": job_id, "status": "pending"})


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    if job_id not in _jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return JSONResponse(content=_jobs[job_id])


# ── Helpers ─────────────────────────────────────────────────────────────

def _fuse_detections(detections: list[dict]) -> list[dict]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda x: x.get("timestamp", 0))
    merged: list[dict] = []
    for det in sorted_dets:
        ts = det.get("timestamp", 0)
        nearby = next((m for m in merged if abs(m["timestamp"] - ts) <= 10), None)
        if nearby:
            nearby["confidence"] = min(0.99, nearby.get("confidence", 0) + 0.1)
        else:
            merged.append(dict(det))
    return [d for d in merged if d.get("confidence", 0) >= 0.3]


async def _run_async_job(job_id: str, body: UnifiedDetectRequest) -> None:
    job = _jobs[job_id]
    try:
        job["status"] = "running"
        job["progress"] = 10

        # Can't easily call the route handler directly, so replicate the logic
        from layers.motion_detector import detect_motion_moments
        motion = await run_in_threadpool(
            detect_motion_moments, body.videoUrl,
            body.timeRangeStart, body.timeRangeEnd,
        )
        job["status"] = "complete"
        job["progress"] = 100
        job["detections"] = motion or []
        job["message"] = f"Found {len(job['detections'])} highlights"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


def _get_available_layers(ali_ready: bool = False) -> list[str]:
    layers = []
    if ali_ready:
        layers.append("ali_ensemble")
    layers.append("motion")
    if _check_yolo():
        layers.append("yolo_player")
    return layers


def _check_model_files() -> dict[str, bool]:
    return {
        "yolo_seg": os.path.exists("app/model/yolo26n-seg.pt"),
        "yolo_number": os.path.exists("app/model/jersey_number_yolo11m.pt"),
        "vitb": os.path.exists("app/model/public/uncertainty_jnr_vitb.pth"),
        "legibility": os.path.exists("app/model/public/koshkina_legibility_soccer.pth"),
        "parseq": os.path.exists("app/model/public/koshkina_parseq_soccer.ckpt"),
    }


def _check_yolo() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False
