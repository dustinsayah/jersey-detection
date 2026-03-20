from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.detection_pipeline import detect_jersey_in_frames
from app.services.detection_runtime import InMemoryFrame, NumberCandidate, PersonBox, PipelineSettings


class _CropReadFakeDetector:
    def __init__(
        self,
        *,
        person_batches: list[list[list[PersonBox]]],
        read_results: list[dict[int, NumberCandidate]],
    ) -> None:
        self._person_batches = list(person_batches)
        self._read_results = list(read_results)

    def detect_persons_batch(self, frames):
        return self._person_batches.pop(0)

    def read_numbers_in_person_crops(self, *, frame_bgr, persons):
        return self._read_results.pop(0)


def _run_pipeline(
    tmp_path: Path,
    *,
    detector: _CropReadFakeDetector,
    color_filter_result,
    jersey_number: int,
):
    sample_video = tmp_path / "sample.mp4"
    sample_video.write_bytes(b"video")
    frame = InMemoryFrame(
        index=0,
        timestamp=0.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )

    with patch(
        "app.services.detection_pipeline._resolve_video_source",
        return_value=sample_video,
    ), patch(
        "app.services.detection_pipeline._get_video_duration_seconds",
        return_value=12.5,
    ), patch(
        "app.services.detection_pipeline._iter_frames_in_memory",
        side_effect=lambda *args, **kwargs: iter([[frame]]),
    ), patch(
        "app.services.detection_pipeline.get_or_create_detector",
        return_value=detector,
    ), patch(
        "app.services.detection_pipeline._color_filter_persons_for_frame",
        return_value=color_filter_result,
    ):
        return detect_jersey_in_frames(
            video_url="https://example.com/game.mp4",
            video_path=None,
            video_bytes=None,
            jersey_number=jersey_number,
            jersey_color="blue",
            sport="basketball",
            position="guard",
            settings=replace(PipelineSettings(), pipeline_workers=1),
        )


def test_detection_first_filters_target_number_after_crop_read(tmp_path) -> None:
    person = PersonBox(x1=5, y1=6, x2=25, y2=40, confidence=0.88)
    detector = _CropReadFakeDetector(
        person_batches=[[[person]]],
        read_results=[{0: NumberCandidate(10, 0.77, tuple(), 8.0, 10.0, 16.0, 18.0)}],
    )

    result = _run_pipeline(
        tmp_path,
        detector=detector,
        color_filter_result=([person], {0: 0.41}, [0]),
        jersey_number=10,
    )

    assert len(result) == 1
    assert result[0]["bbox"]["x1"] == 5
    assert result[0]["bbox"]["y1"] == 6
    assert result[0]["bbox"]["x2"] == 25
    assert result[0]["bbox"]["y2"] == 40


def test_detection_first_rejects_non_target_crop_read(tmp_path) -> None:
    person = PersonBox(x1=5, y1=6, x2=25, y2=40, confidence=0.88)
    detector = _CropReadFakeDetector(
        person_batches=[[[person]]],
        read_results=[{0: NumberCandidate(11, 0.77, tuple(), 8.0, 10.0, 16.0, 18.0)}],
    )

    result = _run_pipeline(
        tmp_path,
        detector=detector,
        color_filter_result=([person], {0: 0.41}, [0]),
        jersey_number=10,
    )

    assert result == []
