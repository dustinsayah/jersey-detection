from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np

from app.services.detection_pipeline import detect_jersey_in_frames
from app.services.detection_runtime import (
    InMemoryFrame,
    NumberCandidate,
    PersonBox,
    PipelineSettings,
)
from app.services.detection_visualizer import DebugFrameData, _match_label


class _StrictFakeDetector:
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


def _queue_side_effect(values: list[tuple[list[PersonBox], dict[int, float], list[int]]]) -> Callable:
    queued = list(values)

    def _inner(*args, **kwargs):
        return queued.pop(0)

    return _inner


def _run_pipeline(
    tmp_path: Path,
    *,
    frames: list[InMemoryFrame],
    detector: _StrictFakeDetector,
    color_filter_values: list[tuple[list[PersonBox], dict[int, float], list[int]]],
    jersey_number: int = 10,
    sport: str = "basketball",
    position: str | None = "guard",
    settings: PipelineSettings | None = None,
):
    sample_video = tmp_path / "sample.mp4"
    sample_video.write_bytes(b"video")
    pipeline_settings = settings or replace(PipelineSettings(), pipeline_workers=1)

    with patch(
        "app.services.detection_pipeline._resolve_video_source",
        return_value=sample_video,
    ), patch(
        "app.services.detection_pipeline._get_video_duration_seconds",
        return_value=12.5,
    ), patch(
        "app.services.detection_pipeline._iter_frames_in_memory",
        side_effect=lambda *args, **kwargs: iter([frames]),
    ), patch(
        "app.services.detection_pipeline.get_or_create_detector",
        return_value=detector,
    ), patch(
        "app.services.detection_pipeline._color_filter_persons_for_frame",
        side_effect=_queue_side_effect(color_filter_values),
    ):
        return detect_jersey_in_frames(
            video_url="https://example.com/game.mp4",
            video_path=None,
            video_bytes=None,
            jersey_number=jersey_number,
            jersey_color="blue",
            sport=sport,
            position=position,
            settings=pipeline_settings,
        )


def test_strict_pipeline_returns_all_target_matching_people_with_person_bboxes(tmp_path) -> None:
    frame = InMemoryFrame(
        index=0,
        timestamp=0.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    persons = [
        PersonBox(x1=1, y1=2, x2=10, y2=20, confidence=0.95),
        PersonBox(x1=20, y1=30, x2=40, y2=60, confidence=0.91),
    ]
    detector = _StrictFakeDetector(
        person_batches=[[persons]],
        read_results=[
            {
                0: NumberCandidate(10, 0.91, tuple(), 2.0, 3.0, 5.0, 6.0),
                1: NumberCandidate(10, 0.82, tuple(), 22.0, 33.0, 25.0, 36.0),
            }
        ],
    )

    result = _run_pipeline(
        tmp_path,
        frames=[frame],
        detector=detector,
        color_filter_values=[(persons, {0: 0.42, 1: 0.37}, [0, 1])],
    )

    assert len(result) == 2
    assert [item["timestamp"] for item in result] == [0.0, 0.0]
    assert result[0]["confidence"] == 0.91
    assert result[1]["confidence"] == 0.82
    assert result[0]["bbox"]["x1"] == 1
    assert result[0]["bbox"]["y1"] == 2
    assert result[0]["bbox"]["x2"] == 10
    assert result[0]["bbox"]["y2"] == 20
    assert result[1]["bbox"]["x1"] == 20
    assert result[1]["bbox"]["y1"] == 30
    assert result[1]["bbox"]["x2"] == 40
    assert result[1]["bbox"]["y2"] == 60


def test_strict_pipeline_ignores_sport_and_position_when_number_matches(tmp_path) -> None:
    frame = InMemoryFrame(
        index=0,
        timestamp=0.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    person = PersonBox(x1=5, y1=6, x2=25, y2=40, confidence=0.88)
    settings = replace(PipelineSettings(), pipeline_workers=1, position_prior_weight=1.0)

    result_a = _run_pipeline(
        tmp_path,
        frames=[frame],
        detector=_StrictFakeDetector(
            person_batches=[[[person]]],
            read_results=[{0: NumberCandidate(10, 0.77, tuple(), 8.0, 10.0, 16.0, 18.0)}],
        ),
        color_filter_values=[([person], {0: 0.41}, [0])],
        sport="basketball",
        position="guard",
        settings=settings,
    )
    result_b = _run_pipeline(
        tmp_path,
        frames=[frame],
        detector=_StrictFakeDetector(
            person_batches=[[[person]]],
            read_results=[{0: NumberCandidate(10, 0.77, tuple(), 8.0, 10.0, 16.0, 18.0)}],
        ),
        color_filter_values=[([person], {0: 0.41}, [0])],
        sport="football",
        position="quarterback",
        settings=settings,
    )

    assert result_a == result_b


def test_strict_pipeline_does_not_synthesize_match_without_number_read(tmp_path) -> None:
    frame = InMemoryFrame(
        index=0,
        timestamp=0.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    person = PersonBox(x1=5, y1=6, x2=25, y2=40, confidence=0.88)

    result = _run_pipeline(
        tmp_path,
        frames=[frame],
        detector=_StrictFakeDetector(
            person_batches=[[[person]]],
            read_results=[{}],
        ),
        color_filter_values=[([person], {0: 0.41}, [0])],
    )

    assert result == []


def test_strict_pipeline_does_not_skip_duplicate_frames_or_early_exit(tmp_path) -> None:
    frame_a = InMemoryFrame(
        index=0,
        timestamp=0.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    frame_b = InMemoryFrame(
        index=1,
        timestamp=1.0,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
    )
    person_a = PersonBox(x1=2, y1=3, x2=20, y2=30, confidence=0.92)
    person_b = PersonBox(x1=4, y1=5, x2=22, y2=32, confidence=0.94)

    result = _run_pipeline(
        tmp_path,
        frames=[frame_a, frame_b],
        detector=_StrictFakeDetector(
            person_batches=[[[person_a], [person_b]]],
            read_results=[
                {0: NumberCandidate(10, 0.79, tuple(), 6.0, 7.0, 12.0, 13.0)},
                {0: NumberCandidate(10, 0.81, tuple(), 7.0, 8.0, 13.0, 14.0)},
            ],
        ),
        color_filter_values=[
            ([person_a], {0: 0.45}, [0]),
            ([person_b], {0: 0.48}, [0]),
        ],
        settings=replace(
            PipelineSettings(),
            pipeline_workers=1,
            early_exit_consecutive=1,
            skip_similarity_threshold=0.0,
        ),
    )

    assert len(result) == 2
    assert [item["timestamp"] for item in result] == [0.0, 1.0]


def test_match_label_includes_detected_number_confidence_and_color_ratio() -> None:
    person = PersonBox(x1=1, y1=2, x2=10, y2=20, confidence=0.95)
    data = DebugFrameData(
        timestamp=0.0,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        persons=[person],
        color_persons=[person],
        color_ratios={0: 0.31},
        matching_person_indices={0},
        matched_person_numbers={0: (22, 0.67)},
    )

    assert _match_label(0, data.persons[0], data) == "MATCH #22 conf=0.67 cr=31%"
