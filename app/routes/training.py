# Training API routes — data collection status + fine-tuning trigger

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.training_pipeline import (
    download_roboflow_dataset,
    get_training_data_stats,
    get_training_status,
    start_finetuning,
    ROBOFLOW_DATASETS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["training"])


# ── Request models ──────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    base_model: str = Field(default="yolo11n.pt", alias="baseModel")
    epochs: int = 50
    batch_size: int = Field(default=16, alias="batchSize")
    img_size: int = Field(default=640, alias="imgSize")
    device: str = "cpu"
    download_datasets: bool = Field(default=True, alias="downloadDatasets")
    roboflow_api_key: str | None = Field(default=None, alias="roboflowApiKey")


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("/training/status")
async def training_status() -> dict[str, Any]:
    """Get current training pipeline status."""
    status = get_training_status()
    stats = get_training_data_stats()
    return {
        "status": "ok",
        "training": {
            "is_running": status.is_running,
            "phase": status.phase,
            "progress": status.progress,
            "current_epoch": status.current_epoch,
            "total_epochs": status.total_epochs,
            "best_map50": status.best_map50,
            "error": status.error,
            "elapsed_s": round(status.elapsed_s, 1),
        },
        "data": stats,
        "available_datasets": [
            {"name": d["name"], "description": d["description"]}
            for d in ROBOFLOW_DATASETS
        ],
    }


@router.post("/training/start")
async def start_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Start model fine-tuning in the background.

    1. Optionally downloads Roboflow datasets
    2. Builds training YAML from collected crops + datasets
    3. Runs YOLO fine-tuning
    4. Saves best model to training_data/runs/

    Poll /training/status to check progress.
    """
    status = get_training_status()
    if status.is_running:
        return {
            "status": "already_running",
            "phase": status.phase,
            "elapsed_s": round(status.elapsed_s, 1),
        }

    # Download datasets first (in foreground — quick check)
    if request.download_datasets:
        for ds in ROBOFLOW_DATASETS:
            download_roboflow_dataset(
                ds["name"],
                api_key=request.roboflow_api_key,
            )

    # Start training in background thread
    def _run_training():
        start_finetuning(
            base_model=request.base_model,
            epochs=request.epochs,
            batch_size=request.batch_size,
            img_size=request.img_size,
            device=request.device,
        )

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {
        "status": "started",
        "config": {
            "base_model": request.base_model,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "img_size": request.img_size,
            "device": request.device,
        },
        "message": "Training started in background. Poll /training/status for progress.",
    }


@router.get("/training/data")
async def training_data() -> dict[str, Any]:
    """Get detailed training data statistics."""
    return get_training_data_stats()
