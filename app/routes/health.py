# GET /health

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _check_phases() -> dict:
    """Report which analysis phases are available."""
    phases = {}

    # Audio: check YAMNet model + tflite-runtime
    yamnet_path = os.getenv("YAMNET_MODEL_PATH", "/app/app/model/yamnet.tflite")
    phases["audio_yamnet"] = Path(yamnet_path).exists()
    try:
        import tflite_runtime  # noqa: F401
        phases["audio_tflite"] = True
    except ImportError:
        phases["audio_tflite"] = False

    # Whistle DSP (always available if scipy is installed)
    try:
        import scipy.signal  # noqa: F401
        phases["audio_whistle_dsp"] = True
    except ImportError:
        phases["audio_whistle_dsp"] = False

    # Pose: check YOLO pose model
    pose_path = os.getenv("POSE_MODEL_PATH", "app/model/yolo11n-pose.pt")
    phases["pose_model"] = Path(pose_path).exists()

    # Motion: always available (OpenCV)
    phases["motion_optical_flow"] = True

    # Tracking: always available (Ultralytics)
    phases["player_tracking"] = True

    # YouTube proxy: check render server URL
    phases["youtube_proxy"] = bool(os.getenv("RENDER_SERVER_URL", ""))

    # Roboflow trained models
    try:
        from app.services.roboflow_detector import roboflow_detector

        phases["roboflow_models"] = roboflow_detector.status()
    except Exception:
        phases["roboflow_models"] = {
            "football_digit_detector": "import_error",
            "football_player_detector": "import_error",
            "basketball_jersey_ocr": "import_error",
            "football_jersey_tracker": "import_error",
        }

    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        phases["memory_rss_mb"] = round(mem.rss / 1024 / 1024, 1)
        phases["memory_vms_mb"] = round(mem.vms / 1024 / 1024, 1)
    except Exception:
        pass

    return phases


@router.get("/live")
def live() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.get("/ready")
@router.get("/health")
def health(request: Request) -> JSONResponse:
    phases = _check_phases()

    if getattr(request.app.state, "detector_ready", False):
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "endpoints": ["/detect", "/analyze"],
                "phases": phases,
            },
        )

    detail = getattr(
        request.app.state,
        "startup_error",
        "Detector warm-up has not completed successfully.",
    )
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": detail, "phases": phases},
    )
