# Training pipeline — collect training data from confirmed detections
# and fine-tune YOLO models for improved jersey OCR.
#
# Two modes:
# 1. DATA COLLECTION: Save confirmed detection crops + labels from /analyze runs
# 2. FINE-TUNING: Train YOLO model on collected + Roboflow datasets

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/app/training_data"))
CROPS_DIR = DATA_DIR / "crops"
LABELS_DIR = DATA_DIR / "labels"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"

# Roboflow public datasets for football jersey detection
ROBOFLOW_DATASETS = [
    {
        "name": "jerseynumbers",
        "url": "https://universe.roboflow.com/ds/jerseynumbers",
        "description": "Jersey number detection — digit-level bounding boxes",
        "format": "yolov8",
    },
    {
        "name": "footballplayertracking",
        "url": "https://universe.roboflow.com/ds/footballplayertracking",
        "description": "Football player tracking — 13K images, player bboxes",
        "format": "yolov8",
    },
    {
        "name": "nfl-players",
        "url": "https://universe.roboflow.com/ds/nfl-players",
        "description": "NFL player detection with jersey numbers",
        "format": "yolov8",
    },
]


@dataclass
class TrainingCrop:
    """A single training crop from a confirmed detection."""
    image_path: str
    label_path: str
    jersey_number: int
    confidence: float
    source_video: str
    timestamp: float
    sport: str = "football"


@dataclass
class TrainingStatus:
    """Current training pipeline status."""
    is_running: bool = False
    phase: str = "idle"  # idle, collecting, downloading, training, evaluating
    progress: float = 0.0
    total_crops: int = 0
    datasets_downloaded: int = 0
    current_epoch: int = 0
    total_epochs: int = 50
    best_map50: float = 0.0
    error: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0


# Global training state
_training_status = TrainingStatus()


def get_training_status() -> TrainingStatus:
    """Get current training pipeline status."""
    if _training_status.is_running:
        _training_status.elapsed_s = time.time() - _training_status.started_at
    return _training_status


def collect_training_crop(
    frame: np.ndarray,
    bbox: list[float],
    jersey_number: int,
    confidence: float,
    source_video: str,
    timestamp: float,
    sport: str = "football",
) -> TrainingCrop | None:
    """Save a confirmed detection as a training crop + YOLO label.

    Called from analyze_pipeline when a jersey number is confirmed
    with high confidence. Saves the player crop and bounding box label
    in YOLO format for later training.
    """
    if confidence < 0.4:
        return None  # Only save high-confidence detections

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return None

    # Generate unique filename
    ts_str = f"{timestamp:.1f}".replace(".", "_")
    fname = f"j{jersey_number}_{sport}_{ts_str}_{int(time.time() * 1000) % 100000}"

    # Save full frame
    img_path = str(CROPS_DIR / f"{fname}.jpg")
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save YOLO label (class_id cx cy w h) normalized
    # Class = jersey_number (0-99)
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    label_path = str(LABELS_DIR / f"{fname}.txt")
    with open(label_path, "w") as f:
        f.write(f"{jersey_number} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Also save crop for inspection
    crop = frame[y1:y2, x1:x2]
    crop_path = str(CROPS_DIR / f"{fname}_crop.jpg")
    cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    logger.info(
        "training_data: saved crop jersey #%d conf=%.2f at %.1fs (%s)",
        jersey_number, confidence, timestamp, fname,
    )

    return TrainingCrop(
        image_path=img_path,
        label_path=label_path,
        jersey_number=jersey_number,
        confidence=confidence,
        source_video=source_video,
        timestamp=timestamp,
        sport=sport,
    )


def get_training_data_stats() -> dict[str, Any]:
    """Get statistics about collected training data."""
    stats: dict[str, Any] = {
        "total_crops": 0,
        "total_labels": 0,
        "jersey_numbers": {},
        "sports": {},
        "disk_usage_mb": 0.0,
    }

    if CROPS_DIR.exists():
        crops = list(CROPS_DIR.glob("*.jpg"))
        # Don't count _crop.jpg files (they're inspection copies)
        full_frames = [c for c in crops if "_crop" not in c.stem]
        stats["total_crops"] = len(full_frames)

    if LABELS_DIR.exists():
        labels = list(LABELS_DIR.glob("*.txt"))
        stats["total_labels"] = len(labels)

        # Parse labels to get jersey number distribution
        for label_file in labels:
            try:
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            jnum = int(parts[0])
                            stats["jersey_numbers"][str(jnum)] = (
                                stats["jersey_numbers"].get(str(jnum), 0) + 1
                            )
            except (ValueError, IndexError):
                continue

    # Disk usage
    if DATA_DIR.exists():
        total_size = sum(
            f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()
        )
        stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 1)

    return stats


def download_roboflow_dataset(
    dataset_name: str,
    api_key: str | None = None,
) -> bool:
    """Download a Roboflow Universe dataset for training.

    Uses the Roboflow Python SDK if available, otherwise falls back
    to direct API download.
    """
    global _training_status
    _training_status.phase = "downloading"

    target_dir = DATASETS_DIR / dataset_name
    if target_dir.exists() and any(target_dir.rglob("*.jpg")):
        logger.info("Dataset %s already downloaded at %s", dataset_name, target_dir)
        return True

    target_dir.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        logger.warning(
            "No ROBOFLOW_API_KEY set — cannot download dataset %s. "
            "Set ROBOFLOW_API_KEY env var or pass api_key parameter.",
            dataset_name,
        )
        return False

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)

        # Map dataset names to Roboflow project paths
        _PROJECT_MAP = {
            "jerseynumbers": ("jersey-number-detection", 1),
            "footballplayertracking": ("football-player-tracking", 1),
            "nfl-players": ("nfl-player-detection", 1),
        }

        if dataset_name not in _PROJECT_MAP:
            logger.warning("Unknown dataset: %s", dataset_name)
            return False

        project_name, version = _PROJECT_MAP[dataset_name]
        project = rf.workspace().project(project_name)
        dataset = project.version(version).download(
            "yolov8", location=str(target_dir)
        )
        logger.info("Downloaded dataset %s to %s", dataset_name, target_dir)
        _training_status.datasets_downloaded += 1
        return True

    except ImportError:
        logger.warning("roboflow SDK not installed — pip install roboflow")
        return False
    except Exception as exc:
        logger.error("Failed to download dataset %s: %s", dataset_name, exc)
        return False


def build_training_yaml(
    output_path: str | None = None,
    include_roboflow: bool = True,
) -> str:
    """Build a YOLO training data.yaml from collected crops + datasets.

    Returns path to the generated YAML file.
    """
    yaml_path = output_path or str(DATA_DIR / "training_data.yaml")

    # Collect all image directories
    train_dirs: list[str] = []
    val_dirs: list[str] = []

    # 1. Our collected crops (80% train, 20% val)
    if CROPS_DIR.exists():
        full_frames = [f for f in CROPS_DIR.glob("*.jpg") if "_crop" not in f.stem]
        if full_frames:
            # Split into train/val
            train_split = DATA_DIR / "splits" / "train"
            val_split = DATA_DIR / "splits" / "val"
            train_split.mkdir(parents=True, exist_ok=True)
            val_split.mkdir(parents=True, exist_ok=True)

            n_val = max(1, len(full_frames) // 5)
            for i, img in enumerate(sorted(full_frames)):
                dest_dir = val_split if i < n_val else train_split
                # Symlink image + label
                dest_img = dest_dir / img.name
                dest_lbl = dest_dir / img.with_suffix(".txt").name
                src_lbl = LABELS_DIR / img.with_suffix(".txt").name

                if not dest_img.exists():
                    try:
                        os.symlink(img, dest_img)
                    except OSError:
                        shutil.copy2(img, dest_img)
                if src_lbl.exists() and not dest_lbl.exists():
                    try:
                        os.symlink(src_lbl, dest_lbl)
                    except OSError:
                        shutil.copy2(src_lbl, dest_lbl)

            train_dirs.append(str(train_split))
            val_dirs.append(str(val_split))

    # 2. Roboflow datasets
    if include_roboflow and DATASETS_DIR.exists():
        for ds_dir in DATASETS_DIR.iterdir():
            if ds_dir.is_dir():
                train_path = ds_dir / "train"
                val_path = ds_dir / "valid"
                if train_path.exists():
                    train_dirs.append(str(train_path / "images"))
                if val_path.exists():
                    val_dirs.append(str(val_path / "images"))

    # Build YAML
    # Classes: 0-99 for jersey numbers
    classes = {i: str(i) for i in range(100)}

    yaml_content = f"""# Auto-generated training config
# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}

path: {DATA_DIR}
train: {train_dirs}
val: {val_dirs}

nc: 100
names: {json.dumps(classes)}
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info("Built training YAML at %s (train=%d dirs, val=%d dirs)",
                yaml_path, len(train_dirs), len(val_dirs))
    return yaml_path


def start_finetuning(
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cpu",
) -> dict[str, Any]:
    """Start YOLO fine-tuning on collected training data.

    Runs training in the current process (meant to be called from
    a background task/thread).

    Returns training results dict.
    """
    global _training_status
    _training_status.is_running = True
    _training_status.phase = "training"
    _training_status.started_at = time.time()
    _training_status.total_epochs = epochs
    _training_status.error = ""

    try:
        # Build training data YAML
        yaml_path = build_training_yaml()

        # Verify we have enough data
        stats = get_training_data_stats()
        if stats["total_crops"] < 10:
            _training_status.error = (
                f"Not enough training data: {stats['total_crops']} crops "
                "(need at least 10). Run more /analyze jobs first."
            )
            _training_status.is_running = False
            _training_status.phase = "error"
            return {"error": _training_status.error}

        from ultralytics import YOLO

        # Load base model
        model = YOLO(base_model)

        # Run training
        run_name = f"jersey_finetune_{int(time.time())}"
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            patience=10,  # Early stopping
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
        )

        # Get best model path
        best_model = RUNS_DIR / run_name / "weights" / "best.pt"

        _training_status.phase = "completed"
        _training_status.is_running = False

        result = {
            "status": "completed",
            "best_model": str(best_model) if best_model.exists() else None,
            "run_name": run_name,
            "epochs": epochs,
            "training_data": stats,
            "elapsed_s": time.time() - _training_status.started_at,
        }

        logger.info("Fine-tuning completed: %s", json.dumps(result, indent=2))
        return result

    except Exception as exc:
        _training_status.error = str(exc)
        _training_status.is_running = False
        _training_status.phase = "error"
        logger.exception("Fine-tuning failed: %s", exc)
        return {"error": str(exc)}
