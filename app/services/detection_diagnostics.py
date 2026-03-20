from __future__ import annotations

from dataclasses import dataclass


def _max_key(values: dict[str, int]) -> str:
    return max(values, key=values.get) if values else "success"


@dataclass
class FailureDiagnostics:
    no_persons: int = 0
    too_small_for_read: int = 0
    no_color_match: int = 0
    no_number_candidates: int = 0
    score_filtered: int = 0
    temporal_rejected: int = 0

    def as_summary(self) -> dict[str, int | str]:
        grouped = {
            "detection": self.no_persons,
            "crop_quality": self.too_small_for_read + self.no_color_match,
            "number_reading": (
                self.no_number_candidates + self.score_filtered + self.temporal_rejected
            ),
        }
        return {
            **grouped,
            "primary_failure_mode": _max_key(grouped),
        }


def classify_detection_first_failure(
    *,
    person_count: int,
    readable_person_count: int,
    color_match_count: int,
    candidate_count: int,
    passed_score: bool,
) -> str:
    if person_count <= 0:
        return "no_persons"
    if readable_person_count <= 0:
        return "too_small_for_read"
    if color_match_count <= 0:
        return "no_color_match"
    if candidate_count <= 0:
        return "no_number_candidates"
    if not passed_score:
        return "score_filtered"
    return "success"


def classify_color_first_failure(
    *,
    roi_count: int,
    candidate_count: int,
    passed_score: bool,
) -> str:
    if roi_count <= 0:
        return "no_color_match"
    if candidate_count <= 0:
        return "no_number_candidates"
    if not passed_score:
        return "score_filtered"
    return "success"
