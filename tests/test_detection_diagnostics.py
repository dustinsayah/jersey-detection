from __future__ import annotations

from app.services.detection_diagnostics import (
    FailureDiagnostics,
    classify_color_first_failure,
    classify_detection_first_failure,
)


def test_detection_first_failure_prefers_crop_quality_when_people_are_too_small() -> None:
    failure = classify_detection_first_failure(
        person_count=3,
        readable_person_count=0,
        color_match_count=0,
        candidate_count=0,
        passed_score=False,
    )
    assert failure == "too_small_for_read"


def test_color_first_failure_reports_number_reading_after_rois_exist() -> None:
    failure = classify_color_first_failure(
        roi_count=2,
        candidate_count=0,
        passed_score=False,
    )
    assert failure == "no_number_candidates"


def test_failure_diagnostics_summary_groups_counts() -> None:
    diagnostics = FailureDiagnostics(
        no_persons=2,
        too_small_for_read=3,
        no_color_match=1,
        no_number_candidates=4,
        score_filtered=2,
        temporal_rejected=1,
    )
    summary = diagnostics.as_summary()
    assert summary["detection"] == 2
    assert summary["crop_quality"] == 4
    assert summary["number_reading"] == 7
    assert summary["primary_failure_mode"] == "number_reading"
