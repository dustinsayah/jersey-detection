# App factory + startup

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes.detect import router as detect_router
from app.routes.health import router as health_router

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_binary_available(command_or_path: str) -> bool:
    if not command_or_path:
        return False
    return bool(shutil.which(command_or_path) or os.path.exists(command_or_path))


def _verify_runtime_dependencies(settings) -> None:
    dependencies = {
        "ffmpeg": settings.ffmpeg_binary,
        "yt-dlp": settings.yt_dlp_binary,
    }
    missing = [
        f"{name} ({value})"
        for name, value in dependencies.items()
        if not _is_binary_available(value)
    ]
    if missing:
        raise RuntimeError(f"Missing required runtime dependencies: {', '.join(missing)}")

    try:
        subprocess.run(
            [settings.ffmpeg_binary, "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as error:
        raise RuntimeError(f"Runtime dependency verification failed: {error}") from error


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Load models at startup so the first request doesn't wait."""
    _configure_logging()
    application.state.detector_ready = False
    application.state.startup_error = None
    try:
        from app.services.detection_detector import get_or_create_detector
        from app.services.detection_runtime import PipelineSettings

        settings = PipelineSettings()
        _verify_runtime_dependencies(settings)
        get_or_create_detector(settings)
        application.state.detector_ready = True
        LOGGER.info("Detection stack warmed up at startup")
    except Exception as error:
        application.state.startup_error = str(error)
        LOGGER.exception("Model warm-up failed")
    yield


def create_app() -> FastAPI:
    _configure_logging()
    application = FastAPI(title="Layer 1 Jersey Detection API", lifespan=_lifespan)
    application.state.detector_ready = False
    application.state.startup_error = None

    cors_origins = _parse_csv_env(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=_parse_csv_env("CORS_ALLOW_METHODS", "GET,POST,OPTIONS"),
            allow_headers=_parse_csv_env("CORS_ALLOW_HEADERS", "*"),
        )

    application.include_router(health_router)
    application.include_router(detect_router)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        messages = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            messages.append(f"{loc}: {err.get('msg', 'invalid')}")
        message = "; ".join(messages) if messages else "Validation error"
        body = getattr(exc, "body", None)
        LOGGER.warning(
            "detect.validation_failed path=%s body_type=%s error=%s",
            request.url.path,
            type(body).__name__ if body is not None else "none",
            message,
        )
        return JSONResponse(
            status_code=400,
            content={"error": message},
        )

    return application


app = create_app()
