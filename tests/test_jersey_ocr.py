from __future__ import annotations

import cv2
import numpy as np

from app.services.jersey_ocr import match_target_number_in_crop


def _make_crop(text: str) -> np.ndarray:
    canvas = np.zeros((120, 180, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.4,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    return canvas


def test_match_target_number_in_crop_reads_two_digits() -> None:
    crop = _make_crop("23")
    result = match_target_number_in_crop(crop, target_number=23, min_confidence=0.30)
    assert result is not None
    assert result.number == 23
    assert result.confidence >= 0.30


def test_match_target_number_in_crop_reads_single_digit() -> None:
    crop = _make_crop("7")
    result = match_target_number_in_crop(crop, target_number=7, min_confidence=0.28)
    assert result is not None
    assert result.number == 7
    assert result.confidence >= 0.28


def test_match_target_number_in_crop_rejects_wrong_number() -> None:
    crop = _make_crop("12")
    result = match_target_number_in_crop(crop, target_number=21, min_confidence=0.30)
    assert result is None
