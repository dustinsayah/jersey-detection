# Roboflow-trained model detector — loads custom YOLOv8n models
# trained from Roboflow Universe datasets for jersey/player detection.
# Models are loaded locally (no API calls during inference).
# ALL models run for ALL sports — digit detection and player detection
# work regardless of sport label.

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def _load_model(filename: str):
    """Load a YOLO .pt model from the model directory."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Model not found: %s — layer disabled", path)
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(path)
        logger.info("Loaded Roboflow model: %s", filename)
        return model
    except Exception as e:
        logger.error("Failed to load %s: %s", filename, e)
        return None


class RoboflowDetector:
    """Loads and runs Roboflow-trained YOLO models for jersey detection.

    Current models (v1 — trained Mar 2026):
      - football_digit_detector.pt — digit-level jersey number OCR
      - football_player_detector.pt — player bounding boxes
      - football_jersey_tracker.pt — jersey tracking across frames
      - basketball_jersey_ocr.pt — SKIPPED (mAP50: 0.10, pending retrain)

    Pending models (v2 — next Colab session):
      - basketball_jersey_number_v2.pt
      - basketball_jersey_number_v3.pt
      - basketball_player_detector.pt
      - football_positions_detector.pt
      - football_presnap_detector.pt
      - jersey_number_universal_v1.pt
      - jersey_number_universal_v2.pt
      - lacrosse_detector_v1.pt
      - lacrosse_detector_v2.pt
    """

    def __init__(self):
        self._loaded = False
        # v1 models
        self.football_digit_model = None
        self.football_player_model = None
        self.basketball_jersey_model = None
        self.football_tracker_model = None
        # v2 models (pending training)
        self.basketball_jersey_number_v2_model = None
        self.basketball_jersey_number_v3_model = None
        self.basketball_player_detector_model = None
        self.football_positions_model = None
        self.football_presnap_model = None
        self.jersey_number_universal_v1_model = None
        self.jersey_number_universal_v2_model = None
        self.lacrosse_v1_model = None
        self.lacrosse_v2_model = None

    def load(self):
        """Lazy-load all models on first use."""
        if self._loaded:
            return

        # ── v1 models (currently deployed) ──
        self.football_digit_model = _load_model("football_digit_detector.pt")
        self.football_player_model = _load_model("football_player_detector.pt")
        # basketball_jersey_ocr.pt — DISABLED: mAP50 0.10, useless
        logger.warning(
            "basketball_jersey_ocr skipped — low accuracy (mAP50: 0.10), pending retrain next Colab session"
        )
        self.basketball_jersey_model = None
        self.football_tracker_model = _load_model("football_jersey_tracker.pt")

        # ── v2 models (load if present, warn if not yet trained) ──
        _v2_models = {
            "basketball_jersey_number_v2.pt": "basketball_jersey_number_v2_model",
            "basketball_jersey_number_v3.pt": "basketball_jersey_number_v3_model",
            "basketball_player_detector.pt": "basketball_player_detector_model",
            "football_positions_detector.pt": "football_positions_model",
            "football_presnap_detector.pt": "football_presnap_model",
            "jersey_number_universal_v1.pt": "jersey_number_universal_v1_model",
            "jersey_number_universal_v2.pt": "jersey_number_universal_v2_model",
            "lacrosse_detector_v1.pt": "lacrosse_v1_model",
            "lacrosse_detector_v2.pt": "lacrosse_v2_model",
        }
        for filename, attr in _v2_models.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                setattr(self, attr, _load_model(filename))
            else:
                logger.info("v2 model not yet trained: %s — pending next Colab session", filename)

        self._loaded = True
        v1_loaded = sum(
            1
            for m in [
                self.football_digit_model,
                self.football_player_model,
                self.football_tracker_model,
            ]
            if m is not None
        )
        v2_loaded = sum(
            1
            for attr in _v2_models.values()
            if getattr(self, attr) is not None
        )
        logger.info(
            "RoboflowDetector: %d/3 v1 models loaded (basketball skipped), %d/9 v2 models loaded",
            v1_loaded, v2_loaded,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """2x upscale + CLAHE contrast enhancement for better digit reading."""
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _parse_number(self, class_name: str) -> int:
        """Extract integer from a class name like '23' or 'digit_5'."""
        try:
            return int("".join(filter(str.isdigit, str(class_name))))
        except (ValueError, TypeError):
            return -1

    def _check_adjacent_digits(
        self, target: int, all_boxes, model
    ) -> bool:
        """Check if two adjacent single-digit detections form the target number."""
        if target < 10:
            return False
        target_str = str(target)
        single_digits: list[tuple[int, list[float]]] = []
        for box in all_boxes:
            name = model.names[int(box.cls)]
            d = self._parse_number(name)
            if 0 <= d <= 9:
                single_digits.append((d, box.xyxy[0].tolist()))

        for i, (d1, b1) in enumerate(single_digits):
            for d2, b2 in single_digits[i + 1 :]:
                # Check horizontal adjacency (within 100px)
                if abs(b1[0] - b2[0]) < 100:
                    if str(d1) + str(d2) == target_str or str(d2) + str(d1) == target_str:
                        return True
        return False

    def detect_football_digits(
        self, frame: np.ndarray, jersey_number: int, conf: float = 0.25
    ) -> list[dict]:
        """Run digit detector on a frame. Works for ANY sport — digits are digits."""
        self.load()
        if self.football_digit_model is None:
            return []
        try:
            enhanced = self._preprocess(frame)
            results = self.football_digit_model(enhanced, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                name = self.football_digit_model.names[int(box.cls)]
                num = self._parse_number(name)
                match = num == jersey_number or self._check_adjacent_digits(
                    jersey_number, results.boxes, self.football_digit_model
                )
                if match:
                    dets.append(
                        {
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": "roboflow_digit",
                        }
                    )
            return dets
        except Exception as e:
            logger.error("digit detect error: %s", e)
            return []

    def detect_football_players(
        self, frame: np.ndarray, conf: float = 0.3
    ) -> list[dict]:
        """Detect all players in frame (bounding boxes). Works for ANY sport."""
        self.load()
        if self.football_player_model is None:
            return []
        try:
            results = self.football_player_model(frame, conf=conf, verbose=False)[0]
            players = []
            for box in results.boxes:
                name = self.football_player_model.names[int(box.cls)]
                if "player" in name.lower() or "person" in name.lower():
                    players.append(
                        {
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": float(box.conf),
                            "class": name,
                            "layer": "roboflow_player",
                        }
                    )
            return players
        except Exception as e:
            logger.error("player detect error: %s", e)
            return []

    def detect_basketball_jerseys(
        self, frame: np.ndarray, jersey_number: int, conf: float = 0.3
    ) -> list[dict]:
        """Run basketball jersey OCR model on a frame."""
        self.load()
        if self.basketball_jersey_model is None:
            return []
        try:
            enhanced = self._preprocess(frame)
            results = self.basketball_jersey_model(enhanced, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                name = self.basketball_jersey_model.names[int(box.cls)]
                num = self._parse_number(name)
                if num == jersey_number:
                    dets.append(
                        {
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": "roboflow_basketball_jersey",
                        }
                    )
            return dets
        except Exception as e:
            logger.error("basketball jersey detect error: %s", e)
            return []

    def detect_football_tracker(
        self, frame: np.ndarray, jersey_number: int, conf: float = 0.25
    ) -> list[dict]:
        """Run jersey tracker model on a frame. Works for ANY sport."""
        self.load()
        if self.football_tracker_model is None:
            return []
        try:
            enhanced = self._preprocess(frame)
            results = self.football_tracker_model(enhanced, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                name = self.football_tracker_model.names[int(box.cls)]
                num = self._parse_number(name)
                if num == jersey_number or self._check_adjacent_digits(
                    jersey_number, results.boxes, self.football_tracker_model
                ):
                    dets.append(
                        {
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": "roboflow_tracker",
                        }
                    )
            return dets
        except Exception as e:
            logger.error("tracker detect error: %s", e)
            return []

    def detect_with_player_crops(
        self,
        frame: np.ndarray,
        jersey_number: int,
        sport: str,
        conf: float = 0.25,
    ) -> list[dict]:
        """Two-pass detection: find players, crop each, run digit detector on crop.

        Runs ALL available models for ALL sports — no sport gating.
        The digit detector reads numbers regardless of sport label.
        The player detector finds players regardless of sport label.
        """
        self.load()
        all_detections: list[dict] = []

        # Pass 1: get player bounding boxes (works for any sport)
        players = self.detect_football_players(frame, conf=0.25)

        if not players:
            # No player boxes — run all digit detectors on full frame
            dets = self.detect_football_digits(frame, jersey_number, conf)
            dets.extend(self.detect_football_tracker(frame, jersey_number, conf))
            # Also try basketball model if available (currently disabled)
            dets.extend(self.detect_basketball_jerseys(frame, jersey_number, conf))
            return dets

        # Pass 2: crop each player and run ALL digit detectors on the crop
        for player in players:
            x1, y1, x2, y2 = [int(c) for c in player["bbox"]]
            # Add padding around player crop
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Run ALL digit detectors on every crop — no sport gating
            dets = self.detect_football_digits(crop, jersey_number, conf=0.2)
            dets.extend(self.detect_football_tracker(crop, jersey_number, conf=0.2))
            dets.extend(self.detect_basketball_jerseys(crop, jersey_number, conf=0.2))

            # Adjust bbox coordinates back to full frame space
            for det in dets:
                det["bbox"] = [
                    det["bbox"][0] + x1,
                    det["bbox"][1] + y1,
                    det["bbox"][2] + x1,
                    det["bbox"][3] + y1,
                ]
                det["player_bbox"] = player["bbox"]
            all_detections.extend(dets)

        return all_detections

    def status(self) -> dict:
        """Report which models are loaded vs missing vs pending."""
        self.load()

        def _v2_status(attr: str) -> str:
            model = getattr(self, attr, None)
            if model is not None:
                return "loaded"
            return "pending - not yet trained"

        return {
            # v1 models
            "football_digit_detector": "loaded" if self.football_digit_model else "missing",
            "football_player_detector": "loaded" if self.football_player_model else "missing",
            "basketball_jersey_ocr": "skipped - pending retrain (mAP50: 0.10)",
            "football_jersey_tracker": "loaded" if self.football_tracker_model else "missing",
            # v2 models
            "basketball_jersey_number_v2": _v2_status("basketball_jersey_number_v2_model"),
            "basketball_jersey_number_v3": _v2_status("basketball_jersey_number_v3_model"),
            "basketball_player_detector": _v2_status("basketball_player_detector_model"),
            "football_positions_detector": _v2_status("football_positions_model"),
            "football_presnap_detector": _v2_status("football_presnap_model"),
            "jersey_number_universal_v1": _v2_status("jersey_number_universal_v1_model"),
            "jersey_number_universal_v2": _v2_status("jersey_number_universal_v2_model"),
            "lacrosse_detector_v1": _v2_status("lacrosse_v1_model"),
            "lacrosse_detector_v2": _v2_status("lacrosse_v2_model"),
        }


# Singleton instance
roboflow_detector = RoboflowDetector()
