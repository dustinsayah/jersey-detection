# Debug video renderer — bbox overlays on pipeline frames

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.services.detection_runtime import (
    NumberCandidate,
    PersonBox,
    ROI,
)

LOGGER = logging.getLogger(__name__)

# cap stored debug frames at 960px wide to save RAM
_DEBUG_MAX_WIDTH = 960


@dataclass
class DebugFrameData:

    timestamp: float
    image: np.ndarray  # BGR, stored at ≤ _DEBUG_MAX_WIDTH

    # Detection-first artefacts
    persons: list[PersonBox] = field(default_factory=list)
    color_persons: list[PersonBox] = field(default_factory=list)

    # Color-first artefacts
    rois: list[ROI] = field(default_factory=list)

    # Common
    candidates: list[NumberCandidate] = field(default_factory=list)
    best_confidence: float | None = None

    # Color ratio per person (keyed by index into persons list)
    color_ratios: dict[int, float] = field(default_factory=dict)
    matching_person_indices: set[int] = field(default_factory=set)
    matched_person_numbers: dict[int, tuple[int, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        if w <= _DEBUG_MAX_WIDTH:
            return
        scale = _DEBUG_MAX_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.image = cv2.resize(self.image, (new_w, new_h))

        # scale all box coordinates to match the resized image
        if self.persons:
            self.persons = [
                PersonBox(
                    x1=p.x1 * scale, y1=p.y1 * scale,
                    x2=p.x2 * scale, y2=p.y2 * scale,
                    confidence=p.confidence,
                    mask=(
                        cv2.resize(
                            p.mask,
                            (max(1, round(p.mask.shape[1] * scale)),
                             max(1, round(p.mask.shape[0] * scale))),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if p.mask is not None else None
                    ),
                )
                for p in self.persons
            ]
        if self.color_persons:
            self.color_persons = [
                PersonBox(
                    x1=p.x1 * scale, y1=p.y1 * scale,
                    x2=p.x2 * scale, y2=p.y2 * scale,
                    confidence=p.confidence,
                    mask=(
                        cv2.resize(
                            p.mask,
                            (max(1, round(p.mask.shape[1] * scale)),
                             max(1, round(p.mask.shape[0] * scale))),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if p.mask is not None else None
                    ),
                )
                for p in self.color_persons
            ]
        if self.candidates:
            self.candidates = [
                NumberCandidate(
                    number=c.number, confidence=c.confidence,
                    digits=c.digits,
                    x1=c.x1 * scale, y1=c.y1 * scale,
                    x2=c.x2 * scale, y2=c.y2 * scale,
                )
                for c in self.candidates
            ]
        if self.rois:
            self.rois = [
                ROI(
                    x1=int(r.x1 * scale), y1=int(r.y1 * scale),
                    x2=int(r.x2 * scale), y2=int(r.y2 * scale),
                    area=r.area * scale * scale,
                )
                for r in self.rois
            ]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COLOR_PERSON = (0, 255, 0)
_COLOR_REJECTED = (0, 128, 255)
_COLOR_COLOR_MATCH = (0, 255, 255)
_COLOR_CANDIDATE = (0, 0, 255)
_COLOR_DETECTED = (0, 200, 0)
_COLOR_SKIPPED = (128, 128, 128)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    label: str = "",
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), _FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def _match_label(idx: int, person: PersonBox, data: DebugFrameData) -> str:
    ratio = data.color_ratios.get(idx, -1.0)
    ratio_str = f" cr={ratio:.0%}" if ratio >= 0 else ""
    is_match = idx in data.matching_person_indices
    if not is_match:
        key = (int(person.x1), int(person.y1), int(person.x2), int(person.y2))
        is_match = any(
            key == (int(match.x1), int(match.y1), int(match.x2), int(match.y2))
            for match in data.color_persons
        )

    if is_match and idx in data.matched_person_numbers:
        number, confidence = data.matched_person_numbers[idx]
        return f"MATCH #{number} conf={confidence:.2f}{ratio_str}"
    if is_match:
        return f"MATCH {person.confidence:.2f}{ratio_str}"
    return f"rejected {person.confidence:.2f}{ratio_str}"


def _annotate_frame(data: DebugFrameData) -> np.ndarray:
    frame = data.image.copy()
    h, w = frame.shape[:2]

    # fast lookup for colour-matched boxes
    color_match_set = set()
    for p in data.color_persons:
        color_match_set.add((int(p.x1), int(p.y1), int(p.x2), int(p.y2)))

    # seg mask tint
    for p in data.persons:
        if p.mask is not None:
            key = (int(p.x1), int(p.y1), int(p.x2), int(p.y2))
            is_match = key in color_match_set
            color = _COLOR_COLOR_MATCH if is_match else _COLOR_REJECTED
            # mask is bbox-local
            bx1, by1 = max(0, int(p.x1)), max(0, int(p.y1))
            bx2 = min(w, bx1 + p.mask.shape[1])
            by2 = min(h, by1 + p.mask.shape[0])
            mw, mh = bx2 - bx1, by2 - by1
            if mw <= 0 or mh <= 0:
                continue
            overlay = frame.copy()
            region = overlay[by1:by2, bx1:bx2]
            mask_crop = p.mask[:mh, :mw]
            mask_bool = mask_crop > 127
            region[mask_bool] = color
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # person boxes
    for idx, p in enumerate(data.persons):
        key = (int(p.x1), int(p.y1), int(p.x2), int(p.y2))
        if idx in data.matching_person_indices or key in color_match_set:
            _draw_rect(
                frame,
                int(p.x1), int(p.y1), int(p.x2), int(p.y2),
                _COLOR_COLOR_MATCH,
                thickness=2,
                label=_match_label(idx, p, data),
            )
        else:
            _draw_rect(
                frame,
                int(p.x1), int(p.y1), int(p.x2), int(p.y2),
                _COLOR_REJECTED,
                thickness=1,
                label=_match_label(idx, p, data),
            )

    # colour-first ROIs
    for roi in data.rois:
        _draw_rect(
            frame,
            roi.x1, roi.y1, roi.x2, roi.y2,
            _COLOR_COLOR_MATCH,
            thickness=2,
            label=f"ROI {roi.area:.0f}",
        )

    # final matches
    for c in data.candidates:
        _draw_rect(
            frame,
            int(c.x1), int(c.y1), int(c.x2), int(c.y2),
            _COLOR_CANDIDATE,
            thickness=3,
            label=f"#{c.number} conf={c.confidence:.2f}",
        )

    # top banner
    ts_text = f"t={data.timestamp:.2f}s"
    if data.best_confidence is not None:
        banner_color = _COLOR_DETECTED
        banner_text = f"{ts_text}  DETECTED conf={data.best_confidence:.2f}"
    else:
        banner_color = _COLOR_SKIPPED
        banner_text = f"{ts_text}  no detection"

    cv2.rectangle(frame, (0, 0), (w, 30), banner_color, -1)
    cv2.putText(frame, banner_text, (8, 22), _FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # legend
    legend_y = h - 60
    cv2.putText(frame, "Yellow=Color-match  Orange=Rejected  Red=Candidate  cr=color-ratio", (8, legend_y), _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return frame



def write_debug_video(
    frames: list[DebugFrameData],
    output_path: str | Path,
    fps: int = 2,
) -> Path:
    if not frames:
        LOGGER.warning("No debug frames to write — skipping debug video.")
        return Path(output_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # size from first frame
    sample = _annotate_frame(frames[0])
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))

    if not writer.isOpened():
        LOGGER.error("Cannot open VideoWriter for %s — falling back to AVI", output)
        output = output.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {output}")

    try:
        for i, data in enumerate(frames):
            annotated = _annotate_frame(data)
            writer.write(annotated)
            if (i + 1) % 50 == 0:
                LOGGER.info("Debug video: wrote %s / %s frames", i + 1, len(frames))
    finally:
        writer.release()

    LOGGER.info("Debug video saved: %s (%s frames, %sx%s @ %sfps)", output, len(frames), w, h, fps)
    return output
