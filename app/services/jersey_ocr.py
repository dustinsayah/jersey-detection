from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass(frozen=True)
class OcrTargetMatch:
    number: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class _DigitBlob:
    x: int
    y: int
    w: int
    h: int
    image: np.ndarray


_TEMPLATE_SIZE = (40, 64)


@lru_cache(maxsize=1)
def _digit_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}
    for digit in "0123456789":
        rendered: list[np.ndarray] = []
        for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX):
            for scale in (1.6, 1.8, 2.0, 2.2):
                for thickness in (2, 3, 4):
                    canvas = np.zeros((_TEMPLATE_SIZE[1], _TEMPLATE_SIZE[0]), dtype=np.uint8)
                    text_size, baseline = cv2.getTextSize(digit, font, scale, thickness)
                    tx = max(0, (_TEMPLATE_SIZE[0] - text_size[0]) // 2)
                    ty = max(
                        text_size[1],
                        (_TEMPLATE_SIZE[1] + text_size[1]) // 2 - baseline,
                    )
                    cv2.putText(
                        canvas,
                        digit,
                        (tx, ty),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    rendered.append(canvas)
        templates[digit] = rendered
    return templates


def _prepare_binary_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        9,
    )
    variants = [otsu, cv2.bitwise_not(otsu), adaptive, cv2.bitwise_not(adaptive)]
    kernel = np.ones((3, 3), dtype=np.uint8)
    return [cv2.morphologyEx(item, cv2.MORPH_CLOSE, kernel) for item in variants]


def _extract_blobs(binary: np.ndarray) -> list[_DigitBlob]:
    height, width = binary.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs: list[_DigitBlob] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < max(20, int(width * height * 0.003)):
            continue
        if h < int(height * 0.25) or h > int(height * 0.98):
            continue
        if w < 4 or w > int(width * 0.80):
            continue
        region = binary[y : y + h, x : x + w]
        fill_ratio = float(cv2.countNonZero(region)) / float(max(1, w * h))
        if not (0.10 <= fill_ratio <= 0.90):
            continue
        blobs.append(_DigitBlob(x=x, y=y, w=w, h=h, image=region))
    blobs.sort(key=lambda item: (item.x, -item.h))
    return blobs[:4]


def _normalize_blob(blob_image: np.ndarray) -> np.ndarray:
    canvas = np.zeros((_TEMPLATE_SIZE[1], _TEMPLATE_SIZE[0]), dtype=np.uint8)
    h, w = blob_image.shape[:2]
    if h <= 0 or w <= 0:
        return canvas

    scale = min((_TEMPLATE_SIZE[0] - 8) / w, (_TEMPLATE_SIZE[1] - 8) / h)
    resized = cv2.resize(
        blob_image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    rh, rw = resized.shape[:2]
    y = (_TEMPLATE_SIZE[1] - rh) // 2
    x = (_TEMPLATE_SIZE[0] - rw) // 2
    canvas[y : y + rh, x : x + rw] = resized
    return canvas


def _largest_contour(binary: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _classify_digit(blob: _DigitBlob) -> tuple[str, float]:
    normalized = _normalize_blob(blob.image)
    blob_contour = _largest_contour(normalized)
    best_digit = ""
    best_score = -1.0
    for digit, templates in _digit_templates().items():
        for template in templates:
            template_score = float(
                cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0][0]
            )
            shape_score = 0.0
            template_contour = _largest_contour(template)
            if blob_contour is not None and template_contour is not None:
                distance = cv2.matchShapes(
                    blob_contour,
                    template_contour,
                    cv2.CONTOURS_MATCH_I1,
                    0.0,
                )
                shape_score = 1.0 / (1.0 + (distance * 10.0))
            score = (0.35 * max(0.0, template_score)) + (0.65 * shape_score)
            if score > best_score:
                best_digit = digit
                best_score = score
    return best_digit, best_score


def match_target_number_in_crop(
    crop_bgr: np.ndarray,
    *,
    target_number: int,
    min_confidence: float,
) -> OcrTargetMatch | None:
    if crop_bgr.size == 0:
        return None

    target = str(target_number)
    target_len = len(target)
    best: OcrTargetMatch | None = None

    for binary in _prepare_binary_variants(crop_bgr):
        blobs = _extract_blobs(binary)
        if not blobs:
            continue

        candidates: list[tuple[list[_DigitBlob], str, float]] = []
        if target_len == 1:
            for blob in blobs:
                digit, score = _classify_digit(blob)
                candidates.append(([blob], digit, score))
        else:
            for left, right in zip(blobs, blobs[1:]):
                gap = right.x - (left.x + left.w)
                if gap > max(left.w, right.w):
                    continue
                left_digit, left_score = _classify_digit(left)
                right_digit, right_score = _classify_digit(right)
                candidates.append(
                    ([left, right], f"{left_digit}{right_digit}", (left_score + right_score) / 2.0)
                )

        for parts, read_value, score in candidates:
            if read_value != target or score < min_confidence:
                continue
            x1 = min(item.x for item in parts) // 2
            y1 = min(item.y for item in parts) // 2
            x2 = max(item.x + item.w for item in parts) // 2
            y2 = max(item.y + item.h for item in parts) // 2
            match = OcrTargetMatch(
                number=target_number,
                confidence=float(score),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            if best is None or match.confidence > best.confidence:
                best = match

    return best
