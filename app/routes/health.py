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
    _v5_keys = {
        "jersey_ocr_universal_v5", "player_detector_v5",
        "outcome_cls_basketball_v5", "outcome_cls_football_v5", "outcome_cls_lacrosse_v5",
        "scoreboard_detector_v5", "dead_ball_classifier_v5", "jersey_upscaler_v5",
    }
    _v1_keys = {
        "football_digit_detector", "football_player_detector",
        "basketball_jersey_ocr", "football_jersey_tracker",
    }
    _v2_baz_keys = {
        "basketball_ball_detector", "football_ball_detector",
        "lacrosse_ball_detector", "basketball_action_detector",
        "basketball_court_zones",
    }
    _v3_ocr_keys = {
        "jersey_ocr_v3_primary", "jersey_ocr_v3_secondary",
        "basketball_ocr_v3", "football_ocr_v3", "lacrosse_ocr_v3",
        "player_isolator_v3", "jersey_color_classifier_v3", "number_region_detector_v3",
        "motion_blur_specialist_v3", "wide_angle_specialist_v3",
        "dark_jersey_specialist_v3", "partial_visibility_specialist_v3",
    }
    _v4_outcome_keys = {
        "basketball_hoop_detector_v4", "basketball_made_shot_v4",
        "basketball_scoring_zone_v4", "basketball_dribble_drive_v4",
        "basketball_rebound_v4", "football_completion_detector_v4",
        "football_touchdown_detector_v4", "football_sack_detector_v4",
        "football_reception_yac_v4", "football_qb_scramble_v4",
        "lacrosse_goal_detector_v4", "lacrosse_shot_quality_v4",
        "lacrosse_ground_ball_v4", "crowd_energy_detector_v4",
        "night_game_specialist_v4", "indoor_court_specialist_v4",
        "crowd_obstruction_specialist_v4", "helmet_glare_specialist_v4",
        "low_resolution_specialist_v4", "multi_player_cluster_v4",
    }
    _v7_keys = {
        "football_jersey_ocr_v7", "navy_jersey_specialist_v7", "football_player_crop_v7",
    }
    _all_versioned_keys = _v5_keys | _v1_keys | _v2_baz_keys | _v3_ocr_keys | _v4_outcome_keys | _v7_keys
    try:
        from app.services.roboflow_detector import roboflow_detector
        rf_status = roboflow_detector.status()
        # Primary detection layer
        phases["primary_detection"] = rf_status.get("primary_detection", "unknown")
        phases["ali_status"] = rf_status.get("ali_status", "unknown")
        # v5 models (PRIMARY)
        phases["roboflow_models_v5"] = {
            k: v for k, v in rf_status.items() if k in _v5_keys
        }
        v5_loaded = sum(1 for k in _v5_keys if rf_status.get(k) == "loaded")
        phases["roboflow_v5_summary"] = f"{v5_loaded}/{len(_v5_keys)} loaded"
        phases["roboflow_models_v1"] = {
            k: v for k, v in rf_status.items() if k in _v1_keys
        }
        phases["roboflow_models_v2"] = {
            k: v for k, v in rf_status.items()
            if k not in _all_versioned_keys
        }
        phases["roboflow_models_v2_ball_action_zone"] = {
            k: v for k, v in rf_status.items() if k in _v2_baz_keys
        }
        phases["roboflow_models_v3_ocr"] = {
            k: v for k, v in rf_status.items() if k in _v3_ocr_keys
        }
        v3_loaded = sum(1 for k in _v3_ocr_keys if rf_status.get(k) == "loaded")
        phases["roboflow_v3_ocr_summary"] = f"{v3_loaded}/12 loaded"
        phases["roboflow_models_v4"] = {
            k: v for k, v in rf_status.items() if k in _v4_outcome_keys
        }
        v4_loaded = sum(1 for k in _v4_outcome_keys if rf_status.get(k) == "loaded")
        phases["roboflow_v4_outcome_summary"] = f"{v4_loaded}/20 loaded"
        phases["roboflow_models_v7"] = {
            k: v for k, v in rf_status.items() if k in _v7_keys
        }
        v7_loaded = sum(1 for k in _v7_keys if rf_status.get(k) == "loaded")
        phases["roboflow_v7_football_summary"] = f"{v7_loaded}/3 loaded"
    except Exception:
        phases["roboflow_models_v1"] = {"error": "import_failed"}

    # ── Temporal consensus status ──
    try:
        from app.services.temporal_consensus import TemporalConsensus
        tc = TemporalConsensus()
        phases["temporal_consensus"] = {
            "status": "active",
            "min_confirmations": tc.min_confirmations,
            "time_window": tc.time_window,
            "confidence_threshold": tc.confidence_threshold,
        }
    except Exception:
        phases["temporal_consensus"] = {"status": "unavailable"}

    # ── Stat pipeline status ──
    phases["stat_pipeline"] = {
        "ball_tracker": "ready - awaiting ball detector models",
        "zone_detector": "ready - using geometry fallback",
        "action_detector": "ready - awaiting action models",
        "game_stats": "ready",
    }

    # yt-dlp version
    try:
        import subprocess
        _ytdlp_ver = subprocess.check_output(["yt-dlp", "--version"], timeout=5, text=True).strip()
        phases["yt_dlp_version"] = _ytdlp_ver
    except Exception:
        phases["yt_dlp_version"] = "unknown"

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
    from functools import partial

    from starlette.concurrency import run_in_threadpool

    from app.services.youtube_proxy import test_youtube_download_sync

    yt_dlp_bin = "yt-dlp"
    ffmpeg_bin = "ffmpeg"
    try:
        from app.services.detection_runtime import PipelineSettings
        settings = PipelineSettings()
        yt_dlp_bin = settings.yt_dlp_binary
        ffmpeg_bin = settings.ffmpeg_binary
    except Exception:
        pass

    result = await run_in_threadpool(
        partial(test_youtube_download_sync, url, yt_dlp_binary=yt_dlp_bin, ffmpeg_binary=ffmpeg_bin)
    )
    status_code = 200 if result.get("success") else 502
    return JSONResponse(status_code=status_code, content=result)


@router.get("/live")
def live() -> JSONResponse:
    model_dir = Path("/app/app/model") if Path("/app/app/model").exists() else Path("app/model")
    pt_count = len(list(model_dir.glob("*.pt"))) + len(list(model_dir.glob("*.pth")))

    try:
        from app.services.roboflow_detector import roboflow_detector
        primary = roboflow_detector.get_primary_detection()
    except Exception:
        primary = "unknown"

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "version": "v7.7.4",
        "models": pt_count,
        "primary_detection": primary,
    })


@router.get("/models")
def models_inventory() -> JSONResponse:
    """Return a JSON inventory of all loaded/available/missing models."""
    model_dir = Path("/app/app/model") if Path("/app/app/model").exists() else Path("app/model")

    try:
        from app.services.roboflow_detector import roboflow_detector
        status = roboflow_detector.status()
        primary = status.pop("primary_detection", "unknown")
        ali_status = status.pop("ali_status", "unknown")

        loaded = [k for k, v in status.items() if v == "loaded"]
        available = [k for k, v in status.items() if v == "available"]
        missing = [k for k, v in status.items() if v == "missing"]
        # Include special-status items that aren't loaded/available/missing
        other = {k: v for k, v in status.items()
                 if v not in ("loaded", "available", "missing")}
        for k in other:
            available.append(k)  # Count fallbacks as available
    except Exception:
        loaded, available, missing = [], [], []
        primary = "unknown"
        ali_status = "unknown"

    total = len(loaded) + len(available) + len(missing)

    return JSONResponse(status_code=200, content={
        "total_models": total,
        "loaded": sorted(loaded),
        "available": sorted(available),
        "missing": sorted(missing),
        "primary_detection": primary,
        "ali_status": ali_status,
        "version": "v7.7.4",
    })


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Liveness probe — always returns 200 so Railway keeps the container."""
    ready = getattr(request.app.state, "detector_ready", False)

    # Check Decodo residential proxy status
    decodo_user = os.getenv("DECODO_USERNAME", "").strip()
    decodo_pass = os.getenv("DECODO_PASSWORD", "").strip()
    decodo_configured = bool(decodo_user and decodo_pass)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok" if ready else "warming_up",
            "version": "v7.7.4",
            "detector_ready": ready,
            "decodo_proxy_configured": decodo_configured,
        },
    )


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness probe — returns 503 if models not loaded yet."""
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
        content={
            "status": "error",
            "detail": detail,
            "hint": "Check Railway logs for 'Model warm-up failed'. "
                    "Common causes: missing model .pt files, OOM, or startup timeout.",
            "phases": phases,
        },
    )


@router.get("/test-v7")
def test_v7() -> JSONResponse:
    """Diagnostic endpoint: check v7 football model availability."""
    try:
        from app.services.roboflow_detector import roboflow_detector
        model_dir = Path("/app/app/model") if Path("/app/app/model").exists() else Path("app/model")
        v7_files = {
            "football_jersey_ocr_v7.pt": str(model_dir / "football_jersey_ocr_v7.pt"),
            "navy_jersey_specialist_v7.pt": str(model_dir / "navy_jersey_specialist_v7.pt"),
            "football_player_crop_v7.pt": str(model_dir / "football_player_crop_v7.pt"),
        }
        return JSONResponse(status_code=200, content={
            "v7_models_loaded": {
                "football_jersey_ocr_v7": roboflow_detector.football_jersey_ocr_v7_model is not None,
                "navy_jersey_specialist_v7": roboflow_detector.navy_jersey_specialist_v7_model is not None,
                "football_player_crop_v7": roboflow_detector.football_player_crop_v7_model is not None,
            },
            "v7_files_on_disk": {
                name: Path(path).exists() for name, path in v7_files.items()
            },
            "note": "v7 models must be trained in Colab and committed to app/model/",
        })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
