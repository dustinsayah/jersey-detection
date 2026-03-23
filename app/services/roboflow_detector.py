# Roboflow-trained model detector — loads 4 custom YOLOv8n models
# trained from Roboflow Universe datasets for jersey/player detection.
# Models are loaded locally (no API calls during inference).

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
    """Loads and runs 4 Roboflow-trained YOLO models for jersey detection."""

    def __init__(self):
        self._loaded = False
        self.football_digit_model = None
        self.football_player_model = None
        self.basketball_jersey_model = None
        self.football_tracker_model = None

    def load(self):
        """Lazy-load all models on first use."""
        if self._loaded:
            return
        self.football_digit_model = _load_model("football_digit_detector.pt")
        self.football_player_model = _load_model("football_player_detector.pt")
        # basketball_jersey_ocr skipped — low accuracy (mAP50: 0.10), pending retrain next Colab session
        logger.warning(
            "basketball_jersey_ocr skipped — low accuracy (mAP50: 0.10), pending retrain next Colab session"
        )
        self.basketball_jersey_model = None
        self.football_tracker_model = _load_model("football_jersey_tracker.pt")
        self._loaded = True
        loaded = sum(
            1
            for m in [
                self.football_digit_model,
                self.football_player_model,
                self.football_tracker_model,
            ]
            if m is not None
        )
        logger.info("RoboflowDetector: %d/3 models loaded (basketball skipped)", loaded)

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
        """Run football digit detector on a frame."""
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
                            "layer": "roboflow_football_digit",
                        }
                    )
            return dets
        except Exception as e:
            logger.error("football digit detect error: %s", e)
            return []

    def detect_football_players(
        self, frame: np.ndarray, conf: float = 0.3
    ) -> list[dict]:
        """Detect all players in frame (bounding boxes)."""
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
                            "layer": "roboflow_football_player",
                        }
                    )
            return players
        except Exception as e:
            logger.error("football player detect error: %s", e)
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
        """Run football jersey tracker model on a frame."""
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
                            "layer": "roboflow_football_tracker",
                        }
                    )
            return dets
        except Exception as e:
            logger.error("football tracker detect error: %s", e)
            return []

    def detect_with_player_crops(
        self,
        frame: np.ndarray,
        jersey_number: int,
        sport: str,
        conf: float = 0.25,
    ) -> list[dict]:
        """Two-pass detection: find players, crop each, run digit detector on crop.

        Much more accurate than running digit detection on full frame.
        """
        self.load()
        all_detections: list[dict] = []

        # Pass 1: get player bounding boxes
        players = self.detect_football_players(frame, conf=0.25)

        if not players:
            # No player boxes — run digit detection on full frame
            if sport.lower() in ("football", "american_football"):
                dets = self.detect_football_digits(frame, jersey_number, conf)
                # Also try tracker model
                dets.extend(self.detect_football_tracker(frame, jersey_number, conf))
                return dets
            else:
                return self.detect_basketball_jerseys(frame, jersey_number, conf)

        # Pass 2: crop each player and run digit detection on the crop
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

            if sport.lower() in ("football", "american_football"):
                dets = self.detect_football_digits(crop, jersey_number, conf=0.2)
                dets.extend(self.detect_football_tracker(crop, jersey_number, conf=0.2))
            else:
                dets = self.detect_basketball_jerseys(crop, jersey_number, conf=0.2)

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
        """Report which models are loaded vs missing."""
        self.load()
        return {
            "football_digit_detector": "loaded" if self.football_digit_model else "missing",
            "football_player_detector": "loaded" if self.football_player_model else "missing",
            "basketball_jersey_ocr": "skipped - pending retrain",
            "football_jersey_tracker": "loaded" if self.football_tracker_model else "missing",
        }


# Singleton instance
roboflow_detector = RoboflowDetector()
