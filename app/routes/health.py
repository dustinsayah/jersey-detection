# GET /health

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _model_file_status(path: str) -> str:
    """Check if a model file exists."""
    return "loaded" if Path(path).exists() else "missing"


def _check_phases() -> dict:
    """Report which analysis phases are available."""
    phases = {}
    model_dir = Path("/app/app/model") if Path("/app/app/model").exists() else Path("app/model")
    public_dir = model_dir / "public"

    # ── Ali's models ──
    phases["ali_models"] = {
        "uncertainty_jnr_vitb": _model_file_status(str(public_dir / "uncertainty_jnr_vitb.pth")),
        "koshkina_legibility": _model_file_status(str(public_dir / "koshkina_legibility_soccer.pth")),
        "koshkina_parseq": _model_file_status(str(public_dir / "koshkina_parseq_soccer.ckpt")),
        "jersey_number_yolo11m": _model_file_status(str(model_dir / "jersey_number_yolo11m.pt")),
        "yolo_person_seg": _model_file_status(str(model_dir / "yolo26n-seg.pt")),
    }

    # Audio: check YAMNet model + tflite-runtime
    yamnet_path = os.getenv("YAMNET_MODEL_PATH", str(model_dir / "yamnet.tflite"))
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
    pose_path = os.getenv("POSE_MODEL_PATH", str(model_dir / "yolo11n-pose.pt"))
    phases["pose_model"] = Path(pose_path).exists()

    # Motion: always available (OpenCV)
    phases["motion_optical_flow"] = True

    # Tracking: always available (Ultralytics)
    phases["player_tracking"] = True

    # YouTube proxy: check render server URL
    render_url = os.getenv("RENDER_SERVER_URL", "https://clipt-render-server-production.up.railway.app")
    phases["youtube_proxy"] = bool(render_url)
    phases["render_server_url"] = render_url

    # ── Roboflow trained models ──
    try:
        from app.services.roboflow_detector import roboflow_detector
        rf_status = roboflow_detector.status()
        # Split into v1, v2, v3 sections
        phases["roboflow_models_v1"] = {
            k: v for k, v in rf_status.items()
            if k in ("football_digit_detector", "football_player_detector",
                     "basketball_jersey_ocr", "football_jersey_tracker")
        }
        phases["roboflow_models_v2"] = {
            k: v for k, v in rf_status.items()
            if k not in phases["roboflow_models_v1"]
            and k not in ("basketball_ball_detector", "football_ball_detector",
                          "lacrosse_ball_detector", "basketball_action_detector",
                          "basketball_court_zones", "football_field_zones",
                          "basketball_player_detector_v2")
        }
        phases["roboflow_models_v3"] = {
            k: v for k, v in rf_status.items()
            if k in ("basketball_ball_detector", "football_ball_detector",
                     "lacrosse_ball_detector", "basketball_action_detector",
                     "basketball_court_zones", "football_field_zones",
                     "basketball_player_detector_v2")
        }
    except Exception:
        phases["roboflow_models_v1"] = {"error": "import_failed"}

    # ── Stat pipeline status ──
    phases["stat_pipeline"] = {
        "ball_tracker": "ready - awaiting ball detector models",
        "zone_detector": "ready - using geometry fallback",
        "action_detector": "ready - awaiting action models",
        "game_stats": "ready",
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


@router.get("/test-youtube")
async def test_youtube(url: str) -> JSONResponse:
    """Test YouTube download chain and report which strategy worked."""
    from app.services.youtube_proxy import test_youtube_download

    yt_dlp_bin = "yt-dlp"
    ffmpeg_bin = "ffmpeg"
    try:
        from app.services.detection_runtime import PipelineSettings
        settings = PipelineSettings()
        yt_dlp_bin = settings.yt_dlp_binary
        ffmpeg_bin = settings.ffmpeg_binary
    except Exception:
        pass

    result = await test_youtube_download(url, yt_dlp_binary=yt_dlp_bin, ffmpeg_binary=ffmpeg_bin)
    status_code = 200 if result.get("success") else 502
    return JSONResponse(status_code=status_code, content=result)


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
