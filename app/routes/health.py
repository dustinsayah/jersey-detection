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

    import os as _os
    _use_new = _os.getenv("USE_NEW_DETECTION", "true").lower() == "true"

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "version": "v8.33.2",
        "models": pt_count,
        "primary_detection": "bytetrack_v1" if _use_new else primary,
        "bytetrack_pipeline": _use_new,
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
        "version": "v8.33.2",
    })


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Liveness probe — always returns 200 so Railway keeps the container."""
    ready = getattr(request.app.state, "detector_ready", False)

    # Check Decodo residential proxy status
    decodo_user = os.getenv("DECODO_USERNAME", "").strip()
    decodo_pass = os.getenv("DECODO_PASSWORD", "").strip()
    decodo_configured = bool(decodo_user and decodo_pass)

    # Check Cloudflare WARP proxy pool status
    try:
        from app.services.youtube_proxy import get_warp_pool_status
        _warp_status = get_warp_pool_status()
        warp_running = _warp_status["pool_size"] > 0
        warp_pool_size = _warp_status["pool_size"]
    except Exception:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 40000), timeout=2):
                warp_running = True
        except (ConnectionRefusedError, OSError, TimeoutError):
            warp_running = False
        warp_pool_size = 1 if warp_running else 0

    # Check YouTube cookie file health
    cookie_health: dict = {"exists": False, "has_auth_tokens": False}
    for cookie_path in ["/app/youtube_cookies.txt", "/app/app/youtube_cookies.txt",
                        "/data/youtube_cookies.txt", os.getenv("YOUTUBE_COOKIES_FILE", "")]:
        if cookie_path and Path(cookie_path).is_file():
            cookie_health["exists"] = True
            cookie_health["path"] = cookie_path
            try:
                import time as _time
                stat = Path(cookie_path).stat()
                age_days = (_time.time() - stat.st_mtime) / 86400
                cookie_health["age_days"] = round(age_days, 1)
                cookie_health["size_bytes"] = stat.st_size
                # Check for real auth tokens (not just anonymous session)
                content = Path(cookie_path).read_text(errors="replace")
                auth_tokens = ["LOGIN_INFO", "SID", "SSID", "HSID", "APISID", "SAPISID"]
                found_tokens = [t for t in auth_tokens if t in content]
                cookie_health["has_auth_tokens"] = len(found_tokens) >= 3
                cookie_health["auth_tokens_found"] = found_tokens
                has_youtube = ".youtube.com" in content
                cookie_health["has_youtube_cookies"] = has_youtube
                # Estimate expiry: cookies last ~3-5 days
                if cookie_health["has_auth_tokens"]:
                    est_expiry_days = max(0, 3.0 - age_days)
                    cookie_health["estimated_days_remaining"] = round(est_expiry_days, 1)
                    cookie_health["likely_expired"] = age_days > 3.0
                    if est_expiry_days < 1.0:
                        cookie_health["warning"] = "Cookies expiring soon — refresh within 24h"
                        cookie_health["action_needed"] = (
                            "POST new cookies via: curl -X POST "
                            "https://jersey-detection-production-d8d8.up.railway.app/upload-cookies "
                            "--data-binary @youtube_cookies.txt"
                        )
                    elif est_expiry_days < 2.0:
                        cookie_health["warning"] = "Cookies expiring in ~1 day — plan refresh"
                else:
                    cookie_health["likely_expired"] = True
                    cookie_health["action_needed"] = "Export authenticated YouTube cookies from Firefox"
            except Exception:
                pass
            break

    # Check PO Token server status
    po_token_status = "not_installed"
    try:
        import socket as _sock
        with _sock.create_connection(("127.0.0.1", 4416), timeout=2):
            po_token_status = "running"
    except (ConnectionRefusedError, OSError, TimeoutError):
        if Path("/app/bgutil-pot/server/build/main.js").exists():
            po_token_status = "installed_not_running"

    # Memory usage
    memory_rss_mb = 0
    try:
        import psutil
        memory_rss_mb = round(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        pass

    # Check home proxy status
    home_proxy_url = os.getenv("HOME_PROXY_URL", "").strip().rstrip("/")
    home_proxy_status = "not_configured"
    if home_proxy_url:
        try:
            import httpx as _httpx
            _hp_resp = _httpx.get(f"{home_proxy_url}/health", timeout=3)
            if _hp_resp.status_code == 200:
                home_proxy_status = "online"
            else:
                home_proxy_status = f"error_http_{_hp_resp.status_code}"
        except Exception:
            home_proxy_status = "offline"

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok" if ready else "warming_up",
            "version": "v8.33.2",
            "detector_ready": ready,
            "home_proxy": {"url": home_proxy_url or None, "status": home_proxy_status},
            "decodo_proxy_configured": decodo_configured,
            "free_proxies": [name for name, _ in [
                ("proxying_io", os.getenv("PROXYING_IO_URL", "")),
                ("proxiware", os.getenv("PROXIWARE_URL", "")),
                ("owlproxy", os.getenv("OWLPROXY_URL", "")),
                ("scrapeops", os.getenv("SCRAPEOPS_URL", "")),
                ("free_proxy", os.getenv("FREE_PROXY_URL", "")),
            ] if _],
            "warp_proxy_running": warp_running,
            "warp_pool_size": warp_pool_size,
            "po_token_server": po_token_status,
            "cookie_health": cookie_health,
            "memory_rss_mb": memory_rss_mb,
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


@router.get("/test-proxy")
def test_proxy() -> JSONResponse:
    """Diagnostic: test Decodo proxy connection independently."""
    import httpx
    import time
    from urllib.parse import quote

    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()
    results: dict = {
        "username_set": bool(user),
        "password_set": bool(passwd),
        "username_preview": user[:4] + "..." if user else "",
        "password_len": len(passwd),
    }

    if not (user and passwd):
        return JSONResponse(status_code=200, content={**results, "status": "not_configured"})

    encoded_user = quote(user, safe="")
    encoded_passwd = quote(passwd, safe="")
    results["username_encoded"] = encoded_user[:4] + "..."
    results["encoding_changed"] = (user != encoded_user or passwd != encoded_passwd)

    # Test 1: URL-encoded proxy
    proxy_url = f"http://{encoded_user}:{encoded_passwd}@gate.decodo.com:10001"
    t = time.perf_counter()
    try:
        resp = httpx.get(
            "https://httpbin.org/ip",
            proxy=proxy_url,
            timeout=httpx.Timeout(20),
            follow_redirects=True,
        )
        results["test_encoded"] = {
            "status": resp.status_code,
            "body": resp.text[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }
    except Exception as exc:
        results["test_encoded"] = {
            "error": str(exc)[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }

    # Test 2: Raw (non-encoded) proxy
    raw_proxy_url = f"http://{user}:{passwd}@gate.decodo.com:10001"
    t = time.perf_counter()
    try:
        resp = httpx.get(
            "https://httpbin.org/ip",
            proxy=raw_proxy_url,
            timeout=httpx.Timeout(20),
            follow_redirects=True,
        )
        results["test_raw"] = {
            "status": resp.status_code,
            "body": resp.text[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }
    except Exception as exc:
        results["test_raw"] = {
            "error": str(exc)[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }

    # Test 3: YouTube via encoded proxy
    t = time.perf_counter()
    try:
        resp = httpx.head(
            "https://www.youtube.com/",
            proxy=proxy_url,
            timeout=httpx.Timeout(20),
            follow_redirects=True,
        )
        results["test_youtube"] = {
            "status": resp.status_code,
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }
    except Exception as exc:
        results["test_youtube"] = {
            "error": str(exc)[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }

    return JSONResponse(status_code=200, content=results)


@router.get("/test-warp-pool")
def test_warp_pool() -> JSONResponse:
    """Diagnostic: test all WARP proxy instances in the pool."""
    import time
    import httpx

    from app.services.youtube_proxy import get_warp_pool_status

    pool_status = get_warp_pool_status()
    results: dict = {"pool": pool_status, "connectivity": []}

    for inst in pool_status["instances"]:
        if inst.get("status") == "not_running":
            results["connectivity"].append({
                "label": inst["label"],
                "status": "not_running",
            })
            continue
        # Test SOCKS5 connectivity
        t = time.perf_counter()
        try:
            resp = httpx.get(
                "https://httpbin.org/ip",
                proxy=inst["socks"],
                timeout=httpx.Timeout(15),
            )
            results["connectivity"].append({
                "label": inst["label"],
                "status": "ok",
                "ip": resp.text.strip()[:100],
                "elapsed_ms": round((time.perf_counter() - t) * 1000),
            })
        except Exception as exc:
            results["connectivity"].append({
                "label": inst["label"],
                "status": "error",
                "error": str(exc)[:200],
                "elapsed_ms": round((time.perf_counter() - t) * 1000),
            })

    ok_count = sum(1 for c in results["connectivity"] if c["status"] == "ok")
    results["summary"] = f"{ok_count}/{len(results['connectivity'])} instances connected"
    return JSONResponse(status_code=200, content=results)


@router.get("/test-warp")
def test_warp() -> JSONResponse:
    """Diagnostic: test Cloudflare WARP SOCKS5 proxy."""
    import socket
    import time
    import httpx

    results: dict = {}

    # Test 1: Is wireproxy listening?
    try:
        with socket.create_connection(("127.0.0.1", 40000), timeout=2):
            results["wireproxy_listening"] = True
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        results["wireproxy_listening"] = False
        results["wireproxy_error"] = str(exc)[:200]
        return JSONResponse(status_code=200, content={**results, "status": "warp_not_running"})

    # Test 2: Can we reach httpbin.org through WARP?
    t = time.perf_counter()
    try:
        resp = httpx.get(
            "https://httpbin.org/ip",
            proxy="socks5://127.0.0.1:40000",
            timeout=httpx.Timeout(15),
        )
        results["httpbin_test"] = {
            "status": resp.status_code,
            "body": resp.text[:200],
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }
    except Exception as exc:
        results["httpbin_test"] = {"error": str(exc)[:200]}

    # Test 3: Can we reach YouTube through WARP?
    t = time.perf_counter()
    try:
        resp = httpx.head(
            "https://www.youtube.com/",
            proxy="socks5://127.0.0.1:40000",
            timeout=httpx.Timeout(15),
            follow_redirects=True,
        )
        results["youtube_test"] = {
            "status": resp.status_code,
            "elapsed_ms": round((time.perf_counter() - t) * 1000),
        }
    except Exception as exc:
        results["youtube_test"] = {"error": str(exc)[:200]}

    results["warp_wg_config_set"] = bool(os.getenv("WARP_WG_CONFIG", "").strip())
    return JSONResponse(status_code=200, content={**results, "status": "warp_running"})


@router.get("/test-dash")
async def test_dash(url: str = "https://www.youtube.com/watch?v=7cEcpiciADQ") -> JSONResponse:
    """Diagnostic: list available YouTube formats per client through WARP.

    Tests android_vr, web, ios, android clients through WARP proxy and reports
    what resolutions are available for DASH vs muxed. No download — just format listing.
    """
    from starlette.concurrency import run_in_threadpool
    import time

    def _list_formats() -> dict:
        import yt_dlp
        # Test key combinations: client + proxy + cookies
        tests = [
            ("android_vr", "warp_http", "http://127.0.0.1:40001", False),
            ("web", "warp_http", "http://127.0.0.1:40001", False),
            ("ios", "warp_http", "http://127.0.0.1:40001", False),
            ("android", "warp_http", "http://127.0.0.1:40001", False),
            ("android_vr", "warp_cookies", "http://127.0.0.1:40001", True),
            ("web", "warp_cookies", "http://127.0.0.1:40001", True),
            ("android_vr", "direct", "", False),
            ("web", "direct_cookies", "", True),
        ]
        results = {}

        for client, proxy_name, proxy_url, use_cookies in tests:
            key = f"{client}_{proxy_name}"
            t = time.perf_counter()
            try:
                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "no_check_certificate": True,
                    "extractor_args": {"youtube": {"player_client": [client]}},
                    "socket_timeout": 15,
                    # Don't filter formats — list ALL available
                    "format": "all",
                }
                if proxy_url:
                    ydl_opts["proxy"] = proxy_url
                if use_cookies:
                    for cp in ["/app/youtube_cookies.txt", "/app/app/youtube_cookies.txt"]:
                        if os.path.isfile(cp):
                            ydl_opts["cookiefile"] = cp
                            break

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    formats = info.get("formats", [])
                    video_fmts = []
                    for f in formats:
                        h = f.get("height")
                        if h and h > 50:  # Skip storyboard thumbnails
                            video_fmts.append({
                                "id": f.get("format_id"),
                                "ext": f.get("ext"),
                                "height": h,
                                "width": f.get("width"),
                                "vcodec": (f.get("vcodec") or "")[:20],
                                "acodec": (f.get("acodec") or "")[:20],
                                "tbr": f.get("tbr"),
                                "filesize": f.get("filesize"),
                            })
                    video_fmts.sort(key=lambda x: x["height"], reverse=True)
                    max_h = video_fmts[0]["height"] if video_fmts else 0
                    has_720 = any(f["height"] >= 720 for f in video_fmts)
                    has_h264_720 = any(
                        f["height"] >= 720 and "avc1" in (f.get("vcodec") or "")
                        for f in video_fmts
                    )
                    results[key] = {
                        "status": "ok",
                        "max_height": max_h,
                        "has_720p": has_720,
                        "has_h264_720p": has_h264_720,
                        "format_count": len(video_fmts),
                        "formats": video_fmts[:15],
                        "elapsed_ms": round((time.perf_counter() - t) * 1000),
                    }
            except Exception as exc:
                results[key] = {
                    "status": "error",
                    "error": str(exc)[:300],
                    "elapsed_ms": round((time.perf_counter() - t) * 1000),
                }

        return results

    results = await run_in_threadpool(_list_formats)
    return JSONResponse(status_code=200, content=results)


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


def _get_cookie_path() -> Path:
    """Return the cookie file path (prefer Railway /app path)."""
    for p in ["/app/app/youtube_cookies.txt", "/app/youtube_cookies.txt",
              "/data/youtube_cookies.txt", "app/youtube_cookies.txt"]:
        if Path(p).is_file():
            return Path(p)
    return Path("/app/app/youtube_cookies.txt")


def _validate_cookie_content(content: str) -> dict:
    """Check cookie content for required auth tokens."""
    auth_tokens = ["LOGIN_INFO", "SID", "SSID", "HSID", "APISID", "SAPISID"]
    found = [t for t in auth_tokens if t in content]
    has_youtube = ".youtube.com" in content
    return {
        "has_auth_tokens": len(found) >= 3,
        "auth_tokens_found": found,
        "has_youtube_cookies": has_youtube,
        "line_count": content.count("\n"),
        "size_bytes": len(content.encode("utf-8")),
    }


@router.get("/refresh-cookies")
def refresh_cookies() -> JSONResponse:
    """Check cookie freshness and return refresh instructions if needed."""
    import time as _time
    cookie_path = _get_cookie_path()

    if not cookie_path.is_file():
        return JSONResponse(status_code=200, content={
            "status": "no_cookies",
            "action_needed": "Upload cookies via POST /upload-cookies",
            "instructions": (
                "1. Open Firefox/Chrome, log into YouTube with a Premium account\n"
                "2. Install 'Get cookies.txt LOCALLY' extension\n"
                "3. Export cookies for youtube.com\n"
                "4. Upload: curl -X POST https://jersey-detection-production-d8d8.up.railway.app/upload-cookies "
                "--data-binary @cookies.txt"
            ),
        })

    stat = cookie_path.stat()
    age_days = (_time.time() - stat.st_mtime) / 86400
    content = cookie_path.read_text(errors="replace")
    validation = _validate_cookie_content(content)
    days_remaining = max(0, 3.0 - age_days)

    if not validation["has_auth_tokens"]:
        status = "invalid"
        message = "Cookie file exists but has no auth tokens — re-export while logged into YouTube"
    elif days_remaining < 0.5:
        status = "expired"
        message = "Cookies are expired or expiring very soon — upload new cookies now"
    elif days_remaining < 1.5:
        status = "expiring_soon"
        message = f"Cookies expire in ~{days_remaining:.1f} days — refresh within 24h"
    else:
        status = "ok"
        message = f"Cookies are fresh — {days_remaining:.1f} days remaining"

    return JSONResponse(status_code=200, content={
        "status": status,
        "age_days": round(age_days, 2),
        "days_remaining": round(days_remaining, 1),
        "message": message,
        **validation,
        "instructions": (
            "To refresh:\n"
            "1. Open Chrome/Firefox, verify you're logged into YouTube\n"
            "2. Use 'Get cookies.txt LOCALLY' extension to export\n"
            "3. Upload: curl -X POST https://jersey-detection-production-d8d8.up.railway.app/upload-cookies "
            "--data-binary @cookies.txt"
        ),
    })


@router.post("/upload-cookies")
async def upload_cookies(request: Request) -> JSONResponse:
    """Upload new YouTube cookies without redeploying.

    Usage (raw body):
        curl -X POST https://jersey-detection-production-d8d8.up.railway.app/upload-cookies \\
             --data-binary @youtube_cookies.txt
    """
    body = await request.body()
    if not body:
        return JSONResponse(status_code=400, content={
            "status": "rejected",
            "reason": "Empty request body",
            "hint": "Send cookie file as raw body: curl -X POST URL --data-binary @cookies.txt",
        })
    content = body.decode("utf-8", errors="replace")

    # Validate the uploaded cookie file
    validation = _validate_cookie_content(content)

    if not validation["has_youtube_cookies"]:
        return JSONResponse(status_code=400, content={
            "status": "rejected",
            "reason": "File does not contain youtube.com cookies",
            "hint": "Make sure you exported cookies while on youtube.com",
        })

    if not validation["has_auth_tokens"]:
        return JSONResponse(status_code=400, content={
            "status": "rejected",
            "reason": "File has no auth tokens (LOGIN_INFO, SID, etc.)",
            "auth_tokens_found": validation["auth_tokens_found"],
            "hint": "Make sure you're logged into YouTube before exporting cookies",
        })

    # Save the cookies
    cookie_path = _get_cookie_path()
    try:
        # Backup existing cookies
        if cookie_path.is_file():
            backup = cookie_path.with_suffix(".txt.bak")
            import shutil
            shutil.copy2(cookie_path, backup)

        cookie_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "reason": f"Failed to write cookie file: {exc}",
        })

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "message": "Cookies uploaded successfully",
        "path": str(cookie_path),
        **validation,
        "estimated_days_remaining": 3.0,
        "next_refresh": "Upload new cookies in ~3 days",
    })


@router.get("/cookie-status")
def cookie_status() -> JSONResponse:
    """Dashboard-friendly cookie status for monitoring."""
    import time as _time

    accounts: dict = {}
    cookie_paths = [
        ("primary", "/app/app/youtube_cookies.txt"),
        ("primary_alt", "/app/youtube_cookies.txt"),
        ("data", "/data/youtube_cookies.txt"),
    ]

    for label, path in cookie_paths:
        if not Path(path).is_file():
            continue
        try:
            stat = Path(path).stat()
            age_hours = (_time.time() - stat.st_mtime) / 3600
            days_remaining = max(0, 3.0 - age_hours / 24)
            content = Path(path).read_text(errors="replace")
            validation = _validate_cookie_content(content)

            if days_remaining < 0.5:
                status = "critical"
            elif days_remaining < 1.5:
                status = "warning"
            else:
                status = "healthy"

            if not validation["has_auth_tokens"]:
                status = "critical"

            accounts[label] = {
                "age_hours": round(age_hours, 1),
                "days_remaining": round(days_remaining, 1),
                "status": status,
                "has_auth_tokens": validation["has_auth_tokens"],
                "path": path,
            }
        except Exception:
            accounts[label] = {"status": "error", "path": path}

    if not accounts:
        return JSONResponse(status_code=200, content={
            "status": "no_cookies",
            "accounts": {},
            "action": "Upload cookies via POST /upload-cookies",
        })

    overall = "healthy"
    for acc in accounts.values():
        if acc.get("status") == "critical":
            overall = "critical"
            break
        if acc.get("status") == "warning":
            overall = "warning"

    return JSONResponse(status_code=200, content={
        "status": overall,
        "accounts": accounts,
    })


@router.get("/test-basketball")
async def test_basketball(request: Request) -> Any:
    """Test endpoint — runs full detection on a known-good public basketball game.

    Uses a public NCAA basketball highlight that doesn't require cookies.
    Returns full detection result for testing without needing fresh cookies.
    """
    from starlette.responses import StreamingResponse
    import json
    import time
    import threading

    # Known-good public basketball highlight (NCAA March Madness official channel)
    TEST_URL = "https://www.youtube.com/watch?v=puzER6mJ5c4"

    if not getattr(request.app.state, "detector_ready", False):
        return JSONResponse(status_code=503, content={
            "error": "Detector not ready",
            "version": "v8.33.2",
        })

    started_at = time.perf_counter()
    result_holder: dict = {}
    error_holder: list = []
    done = threading.Event()
    cancel = threading.Event()

    def _run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from app.services.analyze_pipeline import run_analyze_pipeline
            result = loop.run_until_complete(run_analyze_pipeline(
                video_url=TEST_URL,
                jersey_number=0,
                jersey_color="white",
                sport="basketball",
                time_range_start=0,
                time_range_end=60,
                enable_audio=True,
                enable_tracking=False,
                enable_pose=False,
                quality_mode="auto",
                cancel_event=cancel,
            ))
            result_holder["data"] = result
        except Exception as exc:
            error_holder.append(exc)
        finally:
            loop.close()
            done.set()

    async def _stream():
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        main_loop = asyncio.get_event_loop()
        while not done.is_set():
            finished = await main_loop.run_in_executor(None, done.wait, 15)
            if not finished:
                elapsed = round(time.perf_counter() - started_at, 1)
                yield (json.dumps({"keepalive": True, "elapsed": elapsed}) + "\n").encode()
        t.join(timeout=10)
        if error_holder:
            yield (json.dumps({"error": str(error_holder[0])[:200]}) + "\n").encode()
        elif "data" in result_holder:
            yield (json.dumps(result_holder["data"]) + "\n").encode()
        else:
            yield (json.dumps({"error": "No result"}) + "\n").encode()

    import asyncio
    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Test endpoints for Decodo residential proxy ─────────────────────────────

@router.get("/test-decodo")
async def test_decodo() -> JSONResponse:
    """Test Decodo proxy connectivity — returns IP info."""
    import subprocess
    from urllib.parse import quote

    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()

    if not user or not passwd:
        return JSONResponse(content={"success": False, "error": "DECODO_USERNAME or DECODO_PASSWORD not set"})

    encoded_user = quote(user, safe="")
    encoded_passwd = quote(passwd, safe="")
    proxy_url = f"http://{encoded_user}:{encoded_passwd}@gate.decodo.com:10001"

    try:
        result = subprocess.run(
            ["curl", "-s", "--proxy", proxy_url, "--max-time", "15", "https://ipinfo.io/json"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return JSONResponse(content={"success": False, "error": result.stderr[:200]})

        import json as _json
        data = _json.loads(result.stdout)
        return JSONResponse(content={
            "success": True,
            "ip": data.get("ip"),
            "org": data.get("org"),
            "country": data.get("country"),
            "region": data.get("region"),
            "city": data.get("city"),
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)[:200]})


@router.get("/test-decodo-youtube")
async def test_decodo_youtube() -> JSONResponse:
    """Test yt-dlp through Decodo proxy — tries to get video title."""
    import subprocess
    from urllib.parse import quote

    user = os.getenv("DECODO_USERNAME", "").strip()
    passwd = os.getenv("DECODO_PASSWORD", "").strip()

    if not user or not passwd:
        return JSONResponse(content={"success": False, "error": "DECODO_USERNAME or DECODO_PASSWORD not set"})

    encoded_user = quote(user, safe="")
    encoded_passwd = quote(passwd, safe="")
    proxy_url = f"http://{encoded_user}:{encoded_passwd}@gate.decodo.com:10001"

    try:
        # Try web client first (works for school sports), then android_vr fallback
        clients_to_try = ["web", "web_creator", "android_vr"]
        for client in clients_to_try:
            result = subprocess.run(
                [
                    "python3", "-m", "yt_dlp",
                    "--proxy", proxy_url,
                    "--skip-download",
                    "--print", "title",
                    "--no-warnings",
                    "--geo-bypass",
                    "--force-ipv4",
                    "--socket-timeout", "30",
                    "--extractor-args", f"youtube:player_client={client}",
                    "https://www.youtube.com/watch?v=x0bZnfVHDx4",
                ],
                capture_output=True, text=True, timeout=45,
            )
            title = result.stdout.strip()
            if result.returncode == 0 and title:
                return JSONResponse(content={
                    "success": True,
                    "title": title[:100],
                    "client_used": client,
                })
        # All clients failed
        return JSONResponse(content={
            "success": False,
            "title": None,
            "stderr": result.stderr[:300] if result else "No result",
            "clients_tried": clients_to_try,
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)[:200]})


# ── Test endpoint for bgutil poToken server ────────────────────────────────

@router.get("/test-po-token")
async def test_po_token() -> JSONResponse:
    """Test bgutil poToken server — generates a token for a known video."""
    import httpx as _httpx

    bgutil_url = "http://127.0.0.1:4416"
    result: dict = {
        "server_url": bgutil_url,
        "ping": None,
        "token_generated": False,
        "token_preview": None,
    }

    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            # Ping test
            try:
                ping_resp = await client.get(f"{bgutil_url}/ping")
                result["ping"] = ping_resp.json() if ping_resp.status_code == 200 else f"HTTP {ping_resp.status_code}"
            except Exception as e:
                result["ping"] = f"unreachable: {str(e)[:100]}"
                return JSONResponse(content=result)

            # Generate a token for a test video
            try:
                pot_resp = await client.post(
                    f"{bgutil_url}/get_pot",
                    json={"content_binding": "dQw4w9WgXcQ", "bypass_cache": True},
                )
                if pot_resp.status_code == 200:
                    pot_data = pot_resp.json()
                    token = pot_data.get("poToken", "")
                    result["token_generated"] = bool(token)
                    result["token_preview"] = token[:30] + "..." if len(token) > 30 else token
                    result["token_length"] = len(token)
                else:
                    result["token_error"] = f"HTTP {pot_resp.status_code}: {pot_resp.text[:200]}"
            except Exception as e:
                result["token_error"] = str(e)[:200]

            # Check cache
            try:
                cache_resp = await client.get(f"{bgutil_url}/minter_cache")
                if cache_resp.status_code == 200:
                    result["cache_keys"] = cache_resp.json()
            except Exception:
                pass

    except Exception as e:
        result["error"] = str(e)[:200]

    return JSONResponse(content=result)


# ── Test endpoint for Cobalt API ────────────────────────────────────────────

@router.get("/test-cobalt")
async def test_cobalt() -> JSONResponse:
    """Test Cobalt API connectivity."""
    import httpx as _httpx

    cobalt_url = os.getenv("COBALT_API_URL", "").strip().rstrip("/")
    if not cobalt_url:
        return JSONResponse(content={"configured": False, "error": "COBALT_API_URL not set"})

    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            # Health check
            health_resp = await client.get(f"{cobalt_url}/")
            health_ok = health_resp.status_code == 200

            # Try a YouTube download request
            dl_resp = await client.post(
                f"{cobalt_url}/",
                json={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "videoQuality": "360"},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            dl_data = dl_resp.json() if dl_resp.status_code == 200 else {"status": "error", "http": dl_resp.status_code}

            return JSONResponse(content={
                "configured": True,
                "cobalt_url": cobalt_url,
                "health_ok": health_ok,
                "download_test": {
                    "status": dl_data.get("status"),
                    "has_url": bool(dl_data.get("url")),
                    "error": dl_data.get("error"),
                },
            })
    except Exception as e:
        return JSONResponse(content={"configured": True, "cobalt_url": cobalt_url, "error": str(e)[:200]})


# ── Bulk proxy tester for YouTube ──────────────────────────────────────────

@router.post("/test-proxies")
async def test_proxies(request: Request) -> JSONResponse:
    """Test a list of proxies for YouTube video access.

    POST body: {"proxies": ["ip:port", ...], "max_test": 20}
    Tests each proxy:
      1. ipinfo.io — check if residential
      2. YouTube innertube /player — check if video playback allowed
    Returns working proxies sorted by quality.
    """
    import httpx as _httpx
    import asyncio

    body = await request.json()
    proxy_list = body.get("proxies", [])
    max_test = min(body.get("max_test", 20), 50)  # Cap at 50

    proxy_list = proxy_list[:max_test]
    results = []

    async def _test_one(proxy_str: str) -> dict:
        proxy_url = f"http://{proxy_str}" if not proxy_str.startswith("http") else proxy_str
        result = {"proxy": proxy_str, "ip_info": None, "residential": False, "youtube_ok": False}

        try:
            async with _httpx.AsyncClient(
                proxy=proxy_url, timeout=_httpx.Timeout(12), follow_redirects=True
            ) as client:
                # Step 1: Check IP info
                try:
                    ip_resp = await client.get("https://ipinfo.io/json")
                    if ip_resp.status_code == 200:
                        ip_data = ip_resp.json()
                        result["ip_info"] = {
                            "ip": ip_data.get("ip"),
                            "org": ip_data.get("org", ""),
                            "city": ip_data.get("city", ""),
                            "country": ip_data.get("country", ""),
                        }
                        org = ip_data.get("org", "").upper()
                        dc_keywords = ["AMAZON", "GOOGLE", "MICROSOFT", "CLOUDFLARE",
                                       "DIGITALOCEAN", "LINODE", "VULTR", "OVH",
                                       "HETZNER", "CONTABO", "ORACLE"]
                        result["residential"] = not any(k in org for k in dc_keywords)
                except Exception:
                    pass

                # Step 2: Test YouTube InnerTube access
                try:
                    yt_resp = await client.post(
                        "https://www.youtube.com/youtubei/v1/player",
                        json={
                            "context": {
                                "client": {
                                    "clientName": "ANDROID",
                                    "clientVersion": "19.29.37",
                                    "hl": "en", "gl": "US",
                                }
                            },
                            "videoId": "jNQXAC9IVRw",
                            "contentCheckOk": True,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "com.google.android.youtube/19.29.37",
                        },
                    )
                    if yt_resp.status_code == 200:
                        yt_data = yt_resp.json()
                        playability = yt_data.get("playabilityStatus", {})
                        status = playability.get("status", "")
                        streaming = yt_data.get("streamingData", {})
                        has_formats = bool(streaming.get("formats") or streaming.get("adaptiveFormats"))
                        result["youtube_ok"] = status == "OK" and has_formats
                        result["youtube_status"] = status
                        result["youtube_formats"] = len(streaming.get("formats", [])) + len(streaming.get("adaptiveFormats", []))
                    else:
                        result["youtube_status"] = f"HTTP {yt_resp.status_code}"
                except Exception as e:
                    result["youtube_error"] = str(e)[:100]

        except Exception as e:
            result["error"] = str(e)[:100]

        return result

    # Run tests concurrently in batches of 10
    for i in range(0, len(proxy_list), 10):
        batch = proxy_list[i:i + 10]
        batch_results = await asyncio.gather(
            *[_test_one(p) for p in batch],
            return_exceptions=True,
        )
        for r in batch_results:
            if isinstance(r, dict):
                results.append(r)
            else:
                results.append({"error": str(r)[:100]})

    # Sort: residential + youtube first
    working = [r for r in results if r.get("youtube_ok")]
    residential_working = [r for r in working if r.get("residential")]

    return JSONResponse(content={
        "tested": len(results),
        "working_youtube": len(working),
        "residential_working": len(residential_working),
        "results": results,
        "best_proxies": [r["proxy"] for r in residential_working] or [r["proxy"] for r in working],
    })
