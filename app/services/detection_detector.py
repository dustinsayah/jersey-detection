# YOLO model wrappers for jersey number + person detection

from __future__ import annotations

import logging
import re
from pathlib import Path
from statistics import mean

import numpy as np
import cv2
from ultralytics import YOLO

from app.services.detection_runtime import (
    DigitDetection,
    NumberCandidate,
    PersonBox,
    PipelineSettings,
    ROI,
)

LOGGER = logging.getLogger(__name__)

# keep one detector per model combo so we don't reload every request
_detector_cache: dict[tuple[str, str], "YoloDigitDetector"] = {}


def get_or_create_detector(settings: PipelineSettings) -> "YoloDigitDetector":
    cache_key = (settings.yolo_model_source, settings.person_model_source, settings.yolo_device)
    if cache_key not in _detector_cache:
        LOGGER.info("Creating new YoloDigitDetector for key=%s", cache_key)
        _detector_cache[cache_key] = YoloDigitDetector(settings)
    return _detector_cache[cache_key]


def clear_detector_cache() -> None:
    _detector_cache.clear()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class YoloDigitDetector:

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.device = self._resolve_device()
        self.model = self._load_model()
        self._label_to_number: dict[int, int] = self._build_label_number_map()
        self.whole_number_detection_enabled = self._supports_whole_number_detection()
        self.digit_detection_enabled = (
            not self.whole_number_detection_enabled and self._supports_digit_detection()
        )

        # separate COCO person model when the jersey model doesn't have person classes
        self.person_model: YOLO | None = None
        if self.whole_number_detection_enabled:
            self.person_model = self._load_person_model()

        if self.whole_number_detection_enabled:
            LOGGER.info(
                "Dual-model mode: person model=%s, jersey number model=%s (%d classes).",
                settings.person_model_source,
                settings.yolo_model_source,
                len(self._label_to_number),
            )
        elif self.digit_detection_enabled:
            LOGGER.info("Individual-digit detection enabled.")
        elif self.settings.enable_person_fallback:
            LOGGER.warning(
                "Model has no digit/number labels. Person-overlap fallback mode is enabled."
            )

    def _resolve_device(self) -> str:
        preference = self.settings.yolo_device.strip().lower()
        if preference and preference != "auto":
            return preference
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _load_model(self) -> YOLO:
        model_ref = self.settings.yolo_model_source
        local_path = Path(model_ref).expanduser()
        if local_path.exists():
            model_ref = str(local_path.resolve())
        LOGGER.info("Loading YOLO model: %s", model_ref)
        model = YOLO(model_ref)
        model.to(self.device)
        return model

    def _load_person_model(self) -> YOLO:
        model_ref = self.settings.person_model_source
        local_path = Path(model_ref).expanduser()
        if local_path.exists():
            model_ref = str(local_path.resolve())
        LOGGER.info("Loading person-detection model: %s", model_ref)
        model = YOLO(model_ref)
        model.to(self.device)
        return model

    def _class_label(self, class_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _class_to_digit(self, class_id: int) -> int | None:
        label = self._class_label(class_id).strip().lower()
        if label.isdigit():
            value = int(label)
            return value if 0 <= value <= 9 else None
        match = re.search(r"(\d)", label)
        if match:
            return int(match.group(1))
        return None

    def _class_to_jersey_number(self, class_id: int) -> int | None:
        return self._label_to_number.get(class_id)

    def _build_label_number_map(self) -> dict[int, int]:
        names = self.model.names
        mapping: dict[int, int] = {}
        items: list[tuple[int, str]] = []
        if isinstance(names, dict):
            items = [(int(k), str(v)) for k, v in names.items()]
        elif isinstance(names, list):
            items = list(enumerate(str(v) for v in names))
        for class_id, label in items:
            normalized = label.strip()
            if normalized.isdigit():
                mapping[class_id] = int(normalized)
        return mapping

    def _supports_whole_number_detection(self) -> bool:
        multi_digit_count = sum(
            1 for num in self._label_to_number.values() if num >= 10
        )
        return multi_digit_count >= 5  # at least 5 multi-digit classes

    def _supports_digit_detection(self) -> bool:
        names = self.model.names
        labels: list[str] = []
        if isinstance(names, dict):
            labels = [str(value) for value in names.values()]
        elif isinstance(names, list):
            labels = [str(value) for value in names]
        for label in labels:
            normalized = label.strip().lower()
            if normalized.isdigit() and 0 <= int(normalized) <= 9:
                return True
            if re.fullmatch(r"(digit[_\s-]?)?[0-9]", normalized):
                return True
            if re.search(r"\b[0-9]\b", normalized):
                return True
        return False

    def _predict_roi_digits(self, frame_bgr: np.ndarray, roi: ROI) -> list[DigitDetection]:
        crop = frame_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]
        if crop.size == 0:
            return []
        try:
            predictions = self.model.predict(
                source=crop,
                conf=self.settings.detector_conf_threshold,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=320,
            )
        except Exception as error:
            LOGGER.warning("ROI inference failed: %s", error)
            return []
        if not predictions:
            return []

        result = predictions[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        digits: list[DigitDetection] = []

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
            digit = self._class_to_digit(int(class_id))
            if digit is None:
                continue
            digits.append(
                DigitDetection(
                    digit=digit,
                    confidence=float(conf),
                    x1=float(roi.x1 + x1),
                    y1=float(roi.y1 + y1),
                    x2=float(roi.x1 + x2),
                    y2=float(roi.y1 + y2),
                )
            )
        return digits

    def _predict_roi_digits_batched(
        self, frame_bgr: np.ndarray, rois: list[ROI]
    ) -> dict[int, list[DigitDetection]]:
        if not rois:
            return {}

        crops: list[np.ndarray] = []
        crop_roi_indices: list[int] = []
        for idx, roi in enumerate(rois):
            crop = frame_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_roi_indices.append(idx)

        if not crops:
            return {}

        if len(crops) == 1:
            digits = self._predict_roi_digits(frame_bgr, rois[crop_roi_indices[0]])
            return {crop_roi_indices[0]: digits} if digits else {}

        target_size = 640
        resized_crops: list[np.ndarray] = []
        scale_factors: list[tuple[float, float]] = []  # (scale_x, scale_y) per crop
        for crop in crops:
            h, w = crop.shape[:2]
            scale_x = w / target_size
            scale_y = h / target_size
            scale_factors.append((scale_x, scale_y))
            resized = cv2.resize(crop, (target_size, target_size))
            resized_crops.append(resized)

        try:
            predictions = self.model.predict(
                source=resized_crops,
                conf=self.settings.detector_conf_threshold,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=320,
            )
        except Exception as error:
            LOGGER.warning("Batched ROI inference failed: %s", error)
            return {}

        if not predictions:
            return {}

        result_map: dict[int, list[DigitDetection]] = {}
        for pred_idx, result in enumerate(predictions):
            if pred_idx >= len(crop_roi_indices):
                break
            roi_idx = crop_roi_indices[pred_idx]
            roi = rois[roi_idx]
            sx, sy = scale_factors[pred_idx]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            digits: list[DigitDetection] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
                digit = self._class_to_digit(int(class_id))
                if digit is None:
                    continue
                digits.append(
                    DigitDetection(
                        digit=digit,
                        confidence=float(conf),
                        x1=float(roi.x1 + x1 * sx),
                        y1=float(roi.y1 + y1 * sy),
                        x2=float(roi.x1 + x2 * sx),
                        y2=float(roi.y1 + y2 * sy),
                    )
                )
            if digits:
                result_map[roi_idx] = digits

        return result_map

    def _predict_people(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[float, float, float, float, float]]:
        try:
            predictions = self.model.predict(
                source=frame_bgr,
                conf=self.settings.person_fallback_min_conf,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as error:
            LOGGER.warning("Person fallback inference failed: %s", error)
            return []
        if not predictions:
            return []

        result = predictions[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        people: list[tuple[float, float, float, float, float]] = []

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
            if class_id != 0:  # COCO person class
                continue
            people.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
        return people

    @staticmethod
    def _intersection_ratio(
        person_box: tuple[float, float, float, float, float],
        roi: ROI,
    ) -> float:
        px1, py1, px2, py2, _ = person_box
        ix1 = max(px1, float(roi.x1))
        iy1 = max(py1, float(roi.y1))
        ix2 = min(px2, float(roi.x2))
        iy2 = min(py2, float(roi.y2))
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        roi_area = max(1.0, float((roi.x2 - roi.x1) * (roi.y2 - roi.y1)))
        return float(intersection / roi_area)

    def _person_overlap_candidates(
        self,
        *,
        frame_bgr: np.ndarray,
        rois: list[ROI],
        target_number: int,
    ) -> list[NumberCandidate]:
        if not self.settings.enable_person_fallback:
            return []
        people = self._predict_people(frame_bgr)
        candidates: list[NumberCandidate] = []

        for roi in rois:
            best_overlap = 0.0
            best_person_conf = 0.0
            for person_box in people:
                overlap = self._intersection_ratio(person_box, roi)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person_conf = person_box[4]

            if best_overlap < self.settings.person_fallback_min_overlap:
                continue

            score = _clamp01((0.45 * best_person_conf) + (0.40 * best_overlap) + 0.20)
            if score < 0.45:
                area_score = _clamp01(float(roi.area) / self.settings.person_fallback_area_scale)
                score = _clamp01(score + (0.20 * area_score))
            candidates.append(
                NumberCandidate(
                    number=target_number,
                    confidence=score,
                    digits=tuple(),
                    x1=float(roi.x1),
                    y1=float(roi.y1),
                    x2=float(roi.x2),
                    y2=float(roi.y2),
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates

    def _build_target_candidates(
        self,
        *,
        digits: list[DigitDetection],
        target_number: int,
    ) -> list[NumberCandidate]:
        if not digits:
            return []
        target_str = str(target_number)
        target_len = len(target_str)
        ordered = sorted(digits, key=lambda item: item.center_x)
        candidates: list[NumberCandidate] = []

        if target_len == 1:
            for digit in ordered:
                if str(digit.digit) != target_str:
                    continue
                score = _clamp01((0.8 * digit.confidence) + 0.2)
                candidates.append(
                    NumberCandidate(
                        number=target_number,
                        confidence=score,
                        digits=(digit,),
                        x1=digit.x1,
                        y1=digit.y1,
                        x2=digit.x2,
                        y2=digit.y2,
                    )
                )
            return candidates

        for left, right in zip(ordered, ordered[1:]):
            gap = right.x1 - left.x2
            allowed_gap = max(left.width, right.width) * self.settings.digit_max_gap_ratio
            vertical_distance = abs(left.center_y - right.center_y)
            allowed_vertical = max(left.height, right.height) * 0.8
            if gap > allowed_gap or vertical_distance > allowed_vertical:
                continue

            merged_value = f"{left.digit}{right.digit}"
            if merged_value != target_str:
                continue

            avg_conf = mean((left.confidence, right.confidence))
            min_conf = min(left.confidence, right.confidence)
            score = _clamp01((0.55 * avg_conf) + (0.35 * min_conf) + 0.10)
            candidates.append(
                NumberCandidate(
                    number=target_number,
                    confidence=score,
                    digits=(left, right),
                    x1=min(left.x1, right.x1),
                    y1=min(left.y1, right.y1),
                    x2=max(left.x2, right.x2),
                    y2=max(left.y2, right.y2),
                )
            )
        return candidates

    def _predict_roi_numbers(
        self,
        frame_bgr: np.ndarray,
        rois: list[ROI],
        target_number: int,
    ) -> list[NumberCandidate]:
        if not rois:
            return []

        crops: list[np.ndarray] = []
        crop_roi_indices: list[int] = []
        for idx, roi in enumerate(rois):
            crop = frame_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_roi_indices.append(idx)

        if not crops:
            return []

        # resize for batched inference
        target_size = 640
        resized_crops: list[np.ndarray] = []
        scale_factors: list[tuple[float, float]] = []
        for crop in crops:
            h, w = crop.shape[:2]
            sx, sy = w / target_size, h / target_size
            scale_factors.append((sx, sy))
            resized_crops.append(cv2.resize(crop, (target_size, target_size)))

        try:
            predictions = self.model.predict(
                source=resized_crops if len(resized_crops) > 1 else resized_crops[0],
                conf=self.settings.detector_conf_threshold,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=320,
            )
        except Exception as error:
            LOGGER.warning("Whole-number ROI inference failed: %s", error)
            return []

        if not predictions:
            return []

        candidates: list[NumberCandidate] = []
        pred_list = predictions if isinstance(predictions, list) else [predictions]
        for pred_idx, result in enumerate(pred_list):
            if pred_idx >= len(crop_roi_indices):
                break
            roi_idx = crop_roi_indices[pred_idx]
            roi = rois[roi_idx]
            sx, sy = scale_factors[pred_idx]
            crop_w = float(roi.x2 - roi.x1)
            crop_h = float(roi.y2 - roi.y1)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
                number = self._class_to_jersey_number(int(class_id))
                if number is None or number != target_number:
                    continue

                # size filter: reject boxes that are obviously wrong
                det_w = float(x2 - x1) * sx
                det_h = float(y2 - y1) * sy
                w_ratio = det_w / crop_w if crop_w > 0 else 0
                h_ratio = det_h / crop_h if crop_h > 0 else 0
                det_center_y = float(y1 + y2) / 2.0 * sy
                relative_y = det_center_y / crop_h if crop_h > 0 else 0.5

                LOGGER.debug(
                    "JerseyNum #%d conf=%.3f crop=%dx%d det_w_ratio=%.2f det_h_ratio=%.2f rel_y=%.2f",
                    target_number, float(conf), int(crop_w), int(crop_h),
                    w_ratio, h_ratio, relative_y,
                )

                # too wide or too narrow
                if w_ratio > 0.85 or w_ratio < 0.03:
                    LOGGER.debug("  FILTERED: w_ratio=%.2f out of [0.03, 0.85]", w_ratio)
                    continue
                if h_ratio > 0.60 or h_ratio < 0.02:
                    LOGGER.debug("  FILTERED: h_ratio=%.2f out of [0.02, 0.60]", h_ratio)
                    continue

                # must be in the torso area (not head/feet)
                if relative_y < 0.10 or relative_y > 0.75:
                    LOGGER.debug("  FILTERED: rel_y=%.2f out of [0.10, 0.75]", relative_y)
                    continue

                score = _clamp01(float(conf))
                candidates.append(
                    NumberCandidate(
                        number=target_number,
                        confidence=score,
                        digits=tuple(),
                        x1=float(roi.x1 + x1 * sx),
                        y1=float(roi.y1 + y1 * sy),
                        x2=float(roi.x1 + x2 * sx),
                        y2=float(roi.y1 + y2 * sy),
                    )
                )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def find_target_matches(
        self,
        *,
        frame_bgr: np.ndarray,
        rois: list[ROI],
        target_number: int,
    ) -> list[NumberCandidate]:
        # Whole-number detection path (SultanRafi22-style models)
        if self.whole_number_detection_enabled:
            return self._predict_roi_numbers(frame_bgr, rois, target_number)

        if not self.digit_detection_enabled:
            return self._person_overlap_candidates(
                frame_bgr=frame_bgr,
                rois=rois,
                target_number=target_number,
            )

        # Use batched inference for multiple ROIs
        all_candidates: list[NumberCandidate] = []
        roi_digits_map = self._predict_roi_digits_batched(frame_bgr, rois)
        for roi_idx, digits in roi_digits_map.items():
            if not digits:
                continue
            all_candidates.extend(
                self._build_target_candidates(
                    digits=digits,
                    target_number=target_number,
                )
            )
        all_candidates.sort(key=lambda item: item.confidence, reverse=True)
        return all_candidates

    # ------------------------------------------------------------------
    # detection-first helpers
    # ------------------------------------------------------------------

    def detect_numbers_full_frame(
        self, frame_bgr: np.ndarray, target_number: int
    ) -> list[NumberCandidate]:
        try:
            predictions = self.model.predict(
                source=frame_bgr,
                conf=self.settings.detector_conf_threshold,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as error:
            LOGGER.warning("Full-frame number detection failed: %s", error)
            return []
        if not predictions:
            return []

        result = predictions[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        candidates: list[NumberCandidate] = []

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
            number = self._class_to_jersey_number(int(class_id))
            if number is None or number != target_number:
                continue
            candidates.append(
                NumberCandidate(
                    number=target_number,
                    confidence=_clamp01(float(conf)),
                    digits=tuple(),
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect_persons_full_frame(
        self, frame_bgr: np.ndarray
    ) -> list[PersonBox]:
        results = self.detect_persons_batch([frame_bgr])
        return results[0] if results else []

    def detect_persons_batch(
        self, frames: list[np.ndarray]
    ) -> list[list[PersonBox]]:
        """Run person seg on multiple frames at once.

        Masks are bbox-local and clipped to torso zone (15%-70%).
        """
        if not frames:
            return []

        person_mdl = self.person_model if self.person_model is not None else self.model
        source = frames if len(frames) > 1 else frames[0]
        try:
            predictions = person_mdl.predict(
                source=source,
                conf=self.settings.person_fallback_min_conf,
                iou=self.settings.detector_iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as error:
            LOGGER.warning("Batch person detection failed: %s", error)
            return [[] for _ in frames]
        if not predictions:
            return [[] for _ in frames]

        # normalise to list
        pred_list = predictions if isinstance(predictions, list) else [predictions]

        all_persons: list[list[PersonBox]] = []
        for frame_idx, result in enumerate(pred_list):
            if frame_idx >= len(frames):
                break
            frame_bgr = frames[frame_idx]
            frame_h, frame_w = frame_bgr.shape[:2]

            if result.boxes is None or len(result.boxes) == 0:
                all_persons.append([])
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            has_masks = result.masks is not None and result.masks.data is not None

            persons: list[PersonBox] = []
            for det_idx, ((x1, y1, x2, y2), conf, class_id) in enumerate(
                zip(xyxy, confs, classes)
            ):
                if class_id != 0:
                    continue

                seg_mask: np.ndarray | None = None
                if has_masks:
                    raw = result.masks.data[det_idx].cpu().numpy()
                    if raw.shape[0] != frame_h or raw.shape[1] != frame_w:
                        raw = cv2.resize(raw, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    seg_mask = (raw > 0.5).astype(np.uint8) * 255
                    # crop mask to bbox, saves ~10x memory
                    bx1 = max(0, int(x1))
                    by1 = max(0, int(y1))
                    bx2 = min(frame_w, int(x2))
                    by2 = min(frame_h, int(y2))
                    if bx2 > bx1 and by2 > by1:
                        seg_mask = seg_mask[by1:by2, bx1:bx2].copy()
                        # zero out head and legs so we only keep torso pixels
                        box_h = by2 - by1
                        torso_top = int(box_h * 0.15)
                        torso_bot = int(box_h * 0.70)
                        seg_mask[:torso_top, :] = 0
                        seg_mask[torso_bot:, :] = 0
                    else:
                        seg_mask = None

                persons.append(
                    PersonBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=float(conf),
                        mask=seg_mask,
                    )
                )
            all_persons.append(persons)

        return all_persons

    def find_digits_in_person_crops(
        self,
        *,
        frame_bgr: np.ndarray,
        persons: list[PersonBox],
        target_number: int,
    ) -> list[NumberCandidate]:
        if not persons:
            return []

        # crop to torso zone (15%-70%) so the number model sees jersey, not head/legs
        frame_h, frame_w = frame_bgr.shape[:2]
        rois: list[ROI] = []
        for p in persons:
            x1 = max(0, int(p.x1))
            y1 = max(0, int(p.y1))
            x2 = min(frame_w, int(p.x2))
            y2 = min(frame_h, int(p.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            box_h = y2 - y1
            torso_y1 = y1 + int(box_h * 0.15)
            torso_y2 = y1 + int(box_h * 0.70)
            area = float((x2 - x1) * (torso_y2 - torso_y1))
            rois.append(ROI(x1=x1, y1=torso_y1, x2=x2, y2=torso_y2, area=area))

        if not rois:
            return []

        # Whole-number detection path
        if self.whole_number_detection_enabled:
            return self._predict_roi_numbers(frame_bgr, rois, target_number)

        # batched digit detection
        all_candidates: list[NumberCandidate] = []
        roi_digits_map = self._predict_roi_digits_batched(frame_bgr, rois)
        for roi_idx, digits in roi_digits_map.items():
            if not digits:
                continue
            all_candidates.extend(
                self._build_target_candidates(
                    digits=digits,
                    target_number=target_number,
                )
            )
        all_candidates.sort(key=lambda item: item.confidence, reverse=True)
        return all_candidates
