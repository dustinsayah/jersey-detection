# Thin wrapper around the pipeline

from __future__ import annotations

from functools import lru_cache

from app.schemas.detect import DetectRequest
from app.services.detection_pipeline import detect_jersey_in_frames


class DetectionService:
    def detect(self, request: DetectRequest) -> list[dict[str, float]]:
        return detect_jersey_in_frames(**request.to_pipeline_kwargs())


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    return DetectionService()
