# Roboflow-trained model detector — loads custom YOLO models
# trained from Roboflow Universe datasets for jersey/player detection.
# Models are loaded locally (no API calls during inference).
# ALL models run for ALL sports — digit detection and player detection
# work regardless of sport label.
#
# v1/v2 models: YOLOv8n (fast, lower accuracy)
# v3 OCR models: YOLOv8m (slower, much better for small text OCR)
# v3 trains via notebooks/train_models_v3.ipynb
# v4 outcome models: YOLOv8m (play outcome detection + niche specialists)
# v4 trains via notebooks/train_models_v4.ipynb

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

    v2 models (Colab train_models_v2.ipynb):
      - basketball_jersey_number_v2.pt
      - basketball_jersey_number_v3.pt
      - basketball_player_detector_v2.pt
      - football_positions_detector.pt
      - football_presnap_detector.pt
      - jersey_number_universal_v1.pt
      - jersey_number_universal_v2.pt
      - lacrosse_detector_v1.pt
      - lacrosse_detector_v2.pt

    v2 ball/action/zone (Colab train_models_v2.ipynb Chunk 4):
      - basketball_ball_detector.pt
      - football_ball_detector.pt
      - lacrosse_ball_detector.pt
      - basketball_action_detector.pt
      - basketball_court_zones.pt

    v3 OCR models (Colab train_models_v3.ipynb — YOLOv8m, Ali replacement):
      Chunk 1 — Multi-sport jersey OCR:
        - jersey_ocr_v3_primary.pt
        - jersey_ocr_v3_secondary.pt
      Chunk 2 — Sport-specific OCR:
        - basketball_ocr_v3.pt
        - football_ocr_v3.pt
        - lacrosse_ocr_v3.pt
      Chunk 3 — Player isolation + color:
        - player_isolator_v3.pt
        - jersey_color_classifier_v3.pt
        - number_region_detector_v3.pt
      Chunk 4 — Augmentation specialists:
        - motion_blur_specialist_v3.pt
        - wide_angle_specialist_v3.pt
        - dark_jersey_specialist_v3.pt
        - partial_visibility_specialist_v3.pt

    v4 outcome models (Colab train_models_v4.ipynb — YOLOv8m, play outcome detection):
      Section A — Basketball Outcome:
        - basketball_hoop_detector_v4.pt
        - basketball_made_shot_v4.pt
        - basketball_scoring_zone_v4.pt
        - basketball_dribble_drive_v4.pt
        - basketball_rebound_v4.pt
      Section B — Football Outcome:
        - football_completion_detector_v4.pt
        - football_touchdown_detector_v4.pt
        - football_sack_detector_v4.pt
        - football_reception_yac_v4.pt
        - football_qb_scramble_v4.pt
      Section C — Lacrosse Outcome:
        - lacrosse_goal_detector_v4.pt
        - lacrosse_shot_quality_v4.pt
        - lacrosse_ground_ball_v4.pt
      Section D — Cross-Sport:
        - crowd_energy_detector_v4.pt
      Section E — Niche Specialists:
        - night_game_specialist_v4.pt
        - indoor_court_specialist_v4.pt
        - crowd_obstruction_specialist_v4.pt
        - helmet_glare_specialist_v4.pt
        - low_resolution_specialist_v4.pt
        - multi_player_cluster_v4.pt
    """

    def __init__(self):
        self._loaded = False
        # v1 models
        self.football_digit_model = None
        self.football_player_model = None
        self.basketball_jersey_model = None
        self.football_tracker_model = None
        # v2 models
        self.basketball_jersey_number_v2_model = None
        self.basketball_jersey_number_v3_model = None
        self.basketball_player_detector_v2_model = None
        self.football_positions_model = None
        self.football_presnap_model = None
        self.jersey_number_universal_v1_model = None
        self.jersey_number_universal_v2_model = None
        self.lacrosse_v1_model = None
        self.lacrosse_v2_model = None
        # v2 ball/action/zone models
        self.basketball_ball_detector_model = None
        self.football_ball_detector_model = None
        self.lacrosse_ball_detector_model = None
        self.basketball_action_detector_model = None
        self.basketball_court_zones_model = None
        # v3 OCR models (YOLOv8m — Ali replacement)
        self.jersey_ocr_v3_primary_model = None
        self.jersey_ocr_v3_secondary_model = None
        self.basketball_ocr_v3_model = None
        self.football_ocr_v3_model = None
        self.lacrosse_ocr_v3_model = None
        self.player_isolator_v3_model = None
        self.jersey_color_classifier_v3_model = None
        self.number_region_detector_v3_model = None
        self.motion_blur_specialist_v3_model = None
        self.wide_angle_specialist_v3_model = None
        self.dark_jersey_specialist_v3_model = None
        self.partial_visibility_specialist_v3_model = None
        # v4 outcome models (YOLOv8m — play outcome detection)
        self.basketball_hoop_detector_v4_model = None
        self.basketball_made_shot_v4_model = None
        self.basketball_scoring_zone_v4_model = None
        self.basketball_dribble_drive_v4_model = None
        self.basketball_rebound_v4_model = None
        self.football_completion_detector_v4_model = None
        self.football_touchdown_detector_v4_model = None
        self.football_sack_detector_v4_model = None
        self.football_reception_yac_v4_model = None
        self.football_qb_scramble_v4_model = None
        self.lacrosse_goal_detector_v4_model = None
        self.lacrosse_shot_quality_v4_model = None
        self.lacrosse_ground_ball_v4_model = None
        self.crowd_energy_detector_v4_model = None
        self.night_game_specialist_v4_model = None
        self.indoor_court_specialist_v4_model = None
        self.crowd_obstruction_specialist_v4_model = None
        self.helmet_glare_specialist_v4_model = None
        self.low_resolution_specialist_v4_model = None
        self.multi_player_cluster_v4_model = None

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
            "basketball_player_detector_v2.pt": "basketball_player_detector_v2_model",
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

        # ── v2 ball/action/zone models (load if present) ──
        _v2_baz_models = {
            "basketball_ball_detector.pt": "basketball_ball_detector_model",
            "football_ball_detector.pt": "football_ball_detector_model",
            "lacrosse_ball_detector.pt": "lacrosse_ball_detector_model",
            "basketball_action_detector.pt": "basketball_action_detector_model",
            "basketball_court_zones.pt": "basketball_court_zones_model",
        }
        for filename, attr in _v2_baz_models.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                setattr(self, attr, _load_model(filename))
            else:
                logger.info("v2 ball/action/zone model not yet trained: %s", filename)

        # ── v3 OCR models (YOLOv8m — Ali replacement, load if present) ──
        _v3_ocr_models = {
            "jersey_ocr_v3_primary.pt": "jersey_ocr_v3_primary_model",
            "jersey_ocr_v3_secondary.pt": "jersey_ocr_v3_secondary_model",
            "basketball_ocr_v3.pt": "basketball_ocr_v3_model",
            "football_ocr_v3.pt": "football_ocr_v3_model",
            "lacrosse_ocr_v3.pt": "lacrosse_ocr_v3_model",
            "player_isolator_v3.pt": "player_isolator_v3_model",
            "jersey_color_classifier_v3.pt": "jersey_color_classifier_v3_model",
            "number_region_detector_v3.pt": "number_region_detector_v3_model",
            "motion_blur_specialist_v3.pt": "motion_blur_specialist_v3_model",
            "wide_angle_specialist_v3.pt": "wide_angle_specialist_v3_model",
            "dark_jersey_specialist_v3.pt": "dark_jersey_specialist_v3_model",
            "partial_visibility_specialist_v3.pt": "partial_visibility_specialist_v3_model",
        }
        for filename, attr in _v3_ocr_models.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                setattr(self, attr, _load_model(filename))
            else:
                logger.info("v3 OCR model pending: %s — train via train_models_v3.ipynb", filename)

        # ── v4 outcome models (YOLOv8m — play outcome detection, load if present) ──
        _v4_outcome_models = {
            "basketball_hoop_detector_v4.pt": "basketball_hoop_detector_v4_model",
            "basketball_made_shot_v4.pt": "basketball_made_shot_v4_model",
            "basketball_scoring_zone_v4.pt": "basketball_scoring_zone_v4_model",
            "basketball_dribble_drive_v4.pt": "basketball_dribble_drive_v4_model",
            "basketball_rebound_v4.pt": "basketball_rebound_v4_model",
            "football_completion_detector_v4.pt": "football_completion_detector_v4_model",
            "football_touchdown_detector_v4.pt": "football_touchdown_detector_v4_model",
            "football_sack_detector_v4.pt": "football_sack_detector_v4_model",
            "football_reception_yac_v4.pt": "football_reception_yac_v4_model",
            "football_qb_scramble_v4.pt": "football_qb_scramble_v4_model",
            "lacrosse_goal_detector_v4.pt": "lacrosse_goal_detector_v4_model",
            "lacrosse_shot_quality_v4.pt": "lacrosse_shot_quality_v4_model",
            "lacrosse_ground_ball_v4.pt": "lacrosse_ground_ball_v4_model",
            "crowd_energy_detector_v4.pt": "crowd_energy_detector_v4_model",
            "night_game_specialist_v4.pt": "night_game_specialist_v4_model",
            "indoor_court_specialist_v4.pt": "indoor_court_specialist_v4_model",
            "crowd_obstruction_specialist_v4.pt": "crowd_obstruction_specialist_v4_model",
            "helmet_glare_specialist_v4.pt": "helmet_glare_specialist_v4_model",
            "low_resolution_specialist_v4.pt": "low_resolution_specialist_v4_model",
            "multi_player_cluster_v4.pt": "multi_player_cluster_v4_model",
        }
        for filename, attr in _v4_outcome_models.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                setattr(self, attr, _load_model(filename))
            else:
                logger.info("v4 outcome model pending: %s — train via train_models_v4.ipynb", filename)

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
        v2_baz_loaded = sum(
            1
            for attr in _v2_baz_models.values()
            if getattr(self, attr) is not None
        )
        v3_ocr_loaded = sum(
            1
            for attr in _v3_ocr_models.values()
            if getattr(self, attr) is not None
        )
        v4_outcome_loaded = sum(
            1
            for attr in _v4_outcome_models.values()
            if getattr(self, attr) is not None
        )
        logger.info(
            "RoboflowDetector: %d/3 v1, %d/9 v2, %d/5 v2-baz, %d/12 v3-ocr, %d/20 v4-outcome loaded",
            v1_loaded, v2_loaded, v2_baz_loaded, v3_ocr_loaded, v4_outcome_loaded,
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

    def _run_v3_ocr_on_crop(
        self,
        crop: np.ndarray,
        jersey_number: int,
        sport: str,
        conf: float = 0.2,
    ) -> list[dict]:
        """Run v3 OCR models on a player crop. Returns matched detections."""
        dets: list[dict] = []

        # v3 OCR pipeline priority:
        # 1. number_region_detector finds exact number region
        # 2. jersey_ocr_v3_primary reads the number
        # 3. Sport-specific OCR model also runs
        # 4. Specialist models (dark jersey, motion blur, etc.)

        ocr_crop = crop
        # Step 1: number_region_detector narrows to number area
        if self.number_region_detector_v3_model is not None:
            try:
                nr_results = self.number_region_detector_v3_model(
                    crop, conf=0.3, verbose=False
                )[0]
                if nr_results.boxes and len(nr_results.boxes) > 0:
                    best = max(nr_results.boxes, key=lambda b: float(b.conf))
                    nx1, ny1, nx2, ny2 = [int(c) for c in best.xyxy[0].tolist()]
                    nr_crop = crop[
                        max(0, ny1) : min(crop.shape[0], ny2),
                        max(0, nx1) : min(crop.shape[1], nx2),
                    ]
                    if nr_crop.size > 0:
                        ocr_crop = nr_crop
            except Exception as e:
                logger.debug("number_region_detector_v3 error: %s", e)

        enhanced = self._preprocess(ocr_crop)

        # Step 2: primary v3 OCR
        for model, layer_name in [
            (self.jersey_ocr_v3_primary_model, "v3_ocr_primary"),
            (self.jersey_ocr_v3_secondary_model, "v3_ocr_secondary"),
        ]:
            if model is None:
                continue
            try:
                results = model(enhanced, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    name = model.names[int(box.cls)]
                    num = self._parse_number(name)
                    if num == jersey_number or self._check_adjacent_digits(
                        jersey_number, results.boxes, model
                    ):
                        dets.append({
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": layer_name,
                        })
            except Exception as e:
                logger.debug("%s error: %s", layer_name, e)

        # Step 3: sport-specific OCR
        sport_model = None
        sport_layer = "v3_sport_ocr"
        sl = sport.lower()
        if sl == "basketball" and self.basketball_ocr_v3_model:
            sport_model = self.basketball_ocr_v3_model
            sport_layer = "v3_basketball_ocr"
        elif sl in ("football", "american_football") and self.football_ocr_v3_model:
            sport_model = self.football_ocr_v3_model
            sport_layer = "v3_football_ocr"
        elif sl == "lacrosse" and self.lacrosse_ocr_v3_model:
            sport_model = self.lacrosse_ocr_v3_model
            sport_layer = "v3_lacrosse_ocr"

        if sport_model is not None:
            try:
                results = sport_model(enhanced, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    name = sport_model.names[int(box.cls)]
                    num = self._parse_number(name)
                    if num == jersey_number or self._check_adjacent_digits(
                        jersey_number, results.boxes, sport_model
                    ):
                        dets.append({
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": sport_layer,
                        })
            except Exception as e:
                logger.debug("%s error: %s", sport_layer, e)

        # Step 4: specialist models
        specialists = [
            (self.dark_jersey_specialist_v3_model, "v3_dark_jersey"),
            (self.motion_blur_specialist_v3_model, "v3_motion_blur"),
            (self.wide_angle_specialist_v3_model, "v3_wide_angle"),
            (self.partial_visibility_specialist_v3_model, "v3_partial"),
        ]
        for spec_model, spec_layer in specialists:
            if spec_model is None:
                continue
            try:
                results = spec_model(enhanced, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    name = spec_model.names[int(box.cls)]
                    num = self._parse_number(name)
                    if num == jersey_number or self._check_adjacent_digits(
                        jersey_number, results.boxes, spec_model
                    ):
                        dets.append({
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": spec_layer,
                        })
            except Exception as e:
                logger.debug("%s error: %s", spec_layer, e)

        return dets

    def _run_universal_ocr(
        self, crop: np.ndarray, jersey_number: int, conf: float = 0.2
    ) -> list[dict]:
        """Run jersey_number_universal models on a crop (v2, highest accuracy)."""
        dets: list[dict] = []
        for model, layer_name in [
            (self.jersey_number_universal_v1_model, "v2_universal_v1"),
            (self.jersey_number_universal_v2_model, "v2_universal_v2"),
        ]:
            if model is None:
                continue
            try:
                enhanced = self._preprocess(crop)
                results = model(enhanced, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    name = model.names[int(box.cls)]
                    num = self._parse_number(name)
                    if num == jersey_number or self._check_adjacent_digits(
                        jersey_number, results.boxes, model
                    ):
                        dets.append({
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": layer_name,
                        })
            except Exception as e:
                logger.debug("%s error: %s", layer_name, e)
        return dets

    def _run_sport_specific_v2(
        self, crop: np.ndarray, jersey_number: int, sport: str, conf: float = 0.2
    ) -> list[dict]:
        """Run sport-specific v2 models on a crop."""
        dets: list[dict] = []
        sl = sport.lower()

        models_to_run: list[tuple] = []
        if sl == "basketball":
            models_to_run = [
                (self.basketball_jersey_number_v2_model, "v2_basketball_jersey"),
            ]
        elif sl in ("football", "american_football"):
            models_to_run = [
                (self.football_positions_model, "v2_football_positions"),
                (self.football_presnap_model, "v2_football_presnap"),
            ]
        elif sl == "lacrosse":
            models_to_run = [
                (self.lacrosse_v1_model, "v2_lacrosse"),
            ]

        enhanced = self._preprocess(crop)
        for model, layer_name in models_to_run:
            if model is None:
                continue
            try:
                results = model(enhanced, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    name = model.names[int(box.cls)]
                    num = self._parse_number(name)
                    if num == jersey_number or self._check_adjacent_digits(
                        jersey_number, results.boxes, model
                    ):
                        dets.append({
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "number_detected": num,
                            "layer": layer_name,
                        })
            except Exception as e:
                logger.debug("%s error: %s", layer_name, e)
        return dets

    def detect_with_player_crops(
        self,
        frame: np.ndarray,
        jersey_number: int,
        sport: str,
        conf: float = 0.25,
    ) -> list[dict]:
        """Multi-pass detection: find players, crop each, run OCR on crop.

        Detection priority order:
        1. player_isolator_v3 → football_player_detector (universal player finder)
        2. jersey_number_universal_v1 (0.995 mAP50 — best universal reader)
        3. Sport-specific v2 models (basketball/football/lacrosse)
        4. v3 OCR pipeline (when models available)
        5. v1 fallback detectors (digit, tracker)

        Runs ALL available models for ALL sports — no sport gating.
        """
        self.load()
        all_detections: list[dict] = []

        # Pass 1: get player bounding boxes
        # Try v3 player_isolator first, fall back to v1 player detector
        players = []
        if self.player_isolator_v3_model is not None:
            try:
                results = self.player_isolator_v3_model(frame, conf=0.25, verbose=False)[0]
                for box in results.boxes:
                    name = self.player_isolator_v3_model.names[int(box.cls)]
                    if "player" in name.lower() or "person" in name.lower():
                        players.append({
                            "bbox": box.xyxy[0].tolist(),
                            "confidence": float(box.conf),
                            "class": name,
                            "layer": "v3_player_isolator",
                        })
            except Exception as e:
                logger.debug("player_isolator_v3 error: %s", e)

        if not players:
            players = self.detect_football_players(frame, conf=0.25)

        if not players:
            # No player boxes — run all detectors on full frame
            dets = self._run_universal_ocr(frame, jersey_number, conf)
            dets.extend(self._run_sport_specific_v2(frame, jersey_number, sport, conf))
            dets.extend(self._run_v3_ocr_on_crop(frame, jersey_number, sport, conf))
            dets.extend(self.detect_football_digits(frame, jersey_number, conf))
            dets.extend(self.detect_football_tracker(frame, jersey_number, conf))
            return dets

        # Pass 2: crop each player and run full OCR chain
        for player in players:
            x1, y1, x2, y2 = [int(c) for c in player["bbox"]]
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Priority 1: universal v2 (best accuracy — 0.995 mAP50)
            dets = self._run_universal_ocr(crop, jersey_number, conf=0.2)
            # Priority 2: sport-specific v2 models
            dets.extend(self._run_sport_specific_v2(crop, jersey_number, sport, conf=0.2))
            # Priority 3: v3 OCR pipeline
            dets.extend(self._run_v3_ocr_on_crop(crop, jersey_number, sport, conf=0.2))
            # Priority 4: v1 fallback detectors
            dets.extend(self.detect_football_digits(crop, jersey_number, conf=0.2))
            dets.extend(self.detect_football_tracker(crop, jersey_number, conf=0.2))

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

        def _model_status(attr: str) -> str:
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
            "basketball_jersey_number_v2": _model_status("basketball_jersey_number_v2_model"),
            "basketball_jersey_number_v3": _model_status("basketball_jersey_number_v3_model"),
            "basketball_player_detector_v2": _model_status("basketball_player_detector_v2_model"),
            "football_positions_detector": _model_status("football_positions_model"),
            "football_presnap_detector": _model_status("football_presnap_model"),
            "jersey_number_universal_v1": _model_status("jersey_number_universal_v1_model"),
            "jersey_number_universal_v2": _model_status("jersey_number_universal_v2_model"),
            "lacrosse_detector_v1": _model_status("lacrosse_v1_model"),
            "lacrosse_detector_v2": _model_status("lacrosse_v2_model"),
            # v2 ball/action/zone models
            "basketball_ball_detector": _model_status("basketball_ball_detector_model"),
            "football_ball_detector": _model_status("football_ball_detector_model"),
            "lacrosse_ball_detector": _model_status("lacrosse_ball_detector_model"),
            "basketball_action_detector": _model_status("basketball_action_detector_model"),
            "basketball_court_zones": _model_status("basketball_court_zones_model"),
            # v3 OCR models (YOLOv8m — Ali replacement, train via train_models_v3.ipynb)
            "jersey_ocr_v3_primary": _model_status("jersey_ocr_v3_primary_model"),
            "jersey_ocr_v3_secondary": _model_status("jersey_ocr_v3_secondary_model"),
            "basketball_ocr_v3": _model_status("basketball_ocr_v3_model"),
            "football_ocr_v3": _model_status("football_ocr_v3_model"),
            "lacrosse_ocr_v3": _model_status("lacrosse_ocr_v3_model"),
            "player_isolator_v3": _model_status("player_isolator_v3_model"),
            "jersey_color_classifier_v3": _model_status("jersey_color_classifier_v3_model"),
            "number_region_detector_v3": _model_status("number_region_detector_v3_model"),
            "motion_blur_specialist_v3": _model_status("motion_blur_specialist_v3_model"),
            "wide_angle_specialist_v3": _model_status("wide_angle_specialist_v3_model"),
            "dark_jersey_specialist_v3": _model_status("dark_jersey_specialist_v3_model"),
            "partial_visibility_specialist_v3": _model_status("partial_visibility_specialist_v3_model"),
            # v4 outcome models (YOLOv8m — play outcome detection, train via train_models_v4.ipynb)
            "basketball_hoop_detector_v4": _model_status("basketball_hoop_detector_v4_model"),
            "basketball_made_shot_v4": _model_status("basketball_made_shot_v4_model"),
            "basketball_scoring_zone_v4": _model_status("basketball_scoring_zone_v4_model"),
            "basketball_dribble_drive_v4": _model_status("basketball_dribble_drive_v4_model"),
            "basketball_rebound_v4": _model_status("basketball_rebound_v4_model"),
            "football_completion_detector_v4": _model_status("football_completion_detector_v4_model"),
            "football_touchdown_detector_v4": _model_status("football_touchdown_detector_v4_model"),
            "football_sack_detector_v4": _model_status("football_sack_detector_v4_model"),
            "football_reception_yac_v4": _model_status("football_reception_yac_v4_model"),
            "football_qb_scramble_v4": _model_status("football_qb_scramble_v4_model"),
            "lacrosse_goal_detector_v4": _model_status("lacrosse_goal_detector_v4_model"),
            "lacrosse_shot_quality_v4": _model_status("lacrosse_shot_quality_v4_model"),
            "lacrosse_ground_ball_v4": _model_status("lacrosse_ground_ball_v4_model"),
            "crowd_energy_detector_v4": _model_status("crowd_energy_detector_v4_model"),
            "night_game_specialist_v4": _model_status("night_game_specialist_v4_model"),
            "indoor_court_specialist_v4": _model_status("indoor_court_specialist_v4_model"),
            "crowd_obstruction_specialist_v4": _model_status("crowd_obstruction_specialist_v4_model"),
            "helmet_glare_specialist_v4": _model_status("helmet_glare_specialist_v4_model"),
            "low_resolution_specialist_v4": _model_status("low_resolution_specialist_v4_model"),
            "multi_player_cluster_v4": _model_status("multi_player_cluster_v4_model"),
        }


# Singleton instance
roboflow_detector = RoboflowDetector()
