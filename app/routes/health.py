# GET /health

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/live")
def live() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.get("/ready")
@router.get("/health")
def health(request: Request) -> JSONResponse:
    if getattr(request.app.state, "detector_ready", False):
        return JSONResponse(status_code=200, content={"status": "ok"})
    detail = getattr(
        request.app.state,
        "startup_error",
        "Detector warm-up has not completed successfully.",
    )
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": detail},
    )
