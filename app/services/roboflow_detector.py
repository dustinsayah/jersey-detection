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
# v5 models: YOLO11m + VideoMAE (auto-labeled, best accuracy)
# v5 trains via notebooks/train_models_v5_autolabel.ipynb
#
# Detection priority order (v5 PRIMARY, Ali LAST RESORT):
#   YouTube URL → download → extract frames every 3 frames
#   → dead_ball_classifier_v5 (skip dead frames before OCR)
#   → jersey_ocr_universal_v5 (PRIMARY OCR — best accuracy)
#   → player_detector_v5 (PRIMARY player isolation)
#   → v3 OCR pipeline (secondary confirmation)
#   → v2 universal OCR (tertiary — 0.995 mAP50)
#   → v1 football digit detector (legacy fallback)
#   → Ali ensemble (LAST RESORT — only if other layers found <3 detections)
#   → outcome_classifier_v5 + v4 outcome models
#   → scoreboard_detector_v5 (score change → clip boost)
#   → videomae temporal confirmation (if available)
#   → temporal consensus → clip_extractor → build reel

from __future__ import annotations

import gc
import logging
import os
import time

import cv2
import numpy as np

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# Memory limits (MB) — configurable via env vars
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "4000"))
MEMORY_WARNING_MB = int(os.getenv("MEMORY_WARNING_MB", "2200"))


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

    v5 models (Colab train_models_v5_autolabel.ipynb — YOLO11m + VideoMAE):
      YOLO Detection:
        - jersey_ocr_universal_v5.pt — replaces/supplements v1-v3 OCR layers
        - player_detector_v5.pt — supplements existing player detection
      YOLO Classification (frame-level outcome):
        - outcome_classifier_basketball_v5.pt — fast frame-level outcome
        - outcome_classifier_football_v5.pt
        - outcome_classifier_lacrosse_v5.pt
      VideoMAE (temporal outcome confirmation, 16-frame window):
        - videomae_basketball_v5.zip → HuggingFace model directory
        - videomae_football_v5.zip
        - videomae_lacrosse_v5.zip
    """

    def __init__(self):
        self._loaded = False
        self._last_used: dict[str, float] = {}
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
        # v5 models (YOLO11m + VideoMAE — auto-labeled, best accuracy)
        # PRIORITY GROUP 1: loaded FIRST — primary detection layer
        self.jersey_ocr_universal_v5_model = None   # jersey_ocr_universal_v5.pt — PRIMARY OCR
        self.player_detector_v5_model = None         # player_detector_v5.pt — PRIMARY player isolation
        self.outcome_cls_basketball_v5_model = None  # outcome_classifier_basketball_v5.pt
        self.outcome_cls_football_v5_model = None    # outcome_classifier_football_v5.pt
        self.outcome_cls_lacrosse_v5_model = None    # outcome_classifier_lacrosse_v5.pt
        self.scoreboard_detector_v5_model = None     # scoreboard_detector_v5.pt — NEW
        self.dead_ball_classifier_v5_model = None    # dead_ball_classifier_v5.pt — NEW
        self.videomae_basketball_v5 = None           # videomae_basketball_v5/ (HuggingFace dir)
        self.videomae_football_v5 = None             # videomae_football_v5/ (HuggingFace dir)
        self.videomae_lacrosse_v5 = None             # videomae_lacrosse_v5/ (HuggingFace dir)
        self._jersey_upscaler = None                 # jersey_upscaler_v5.pth — SR model (NEW)
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

    def _get_rss_mb(self) -> float:
        """Get current process RSS in MB. Returns 0 if psutil unavailable."""
        if not _PSUTIL_AVAILABLE:
            return 0.0
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _evict_lru_models(self, target_mb: float) -> None:
        """Evict least-recently-used models until RSS drops below target_mb."""
        if not _PSUTIL_AVAILABLE:
            return
        rss = self._get_rss_mb()
        if rss <= target_mb:
            return

        # Build list of (attr_name, last_used_time) for loaded models
        evictable = []
        for attr in list(self._last_used.keys()):
            if getattr(self, attr, None) is not None:
                evictable.append((attr, self._last_used[attr]))

        # Sort by last used time (oldest first)
        evictable.sort(key=lambda x: x[1])

        for attr, _ in evictable:
            if self._get_rss_mb() <= target_mb:
                break
            logger.warning("Memory pressure: evicting model %s (RSS=%.0fMB, target=%.0fMB)",
                           attr, self._get_rss_mb(), target_mb)
            setattr(self, attr, None)
            gc.collect()

    def _memory_ok_for_loading(self, group_name: str) -> bool:
        """Check if RSS is below MEMORY_WARNING_MB. Log and return False if exceeded."""
        rss = self._get_rss_mb()
        if rss <= 0:
            return True  # psutil unavailable, allow loading
        if rss >= MEMORY_WARNING_MB:
            logger.warning(
                "Memory warning: RSS=%.0fMB >= %dMB after loading %s — skipping remaining model groups",
                rss, MEMORY_WARNING_MB, group_name,
            )
            return False
        return True

    def load(self):
        """Lazy-load all models on first use.

        Loading order: v5 FIRST (primary), v3, v2, v4, v1, Ali LAST.
        Memory check after each group — Ali gets evicted first if pressure.
        """
        if self._loaded:
            return

        # ── PRIORITY GROUP 1a: v5 LIGHTWEIGHT models (classifier + upscaler only) ──
        # NOTE: YOLO11m models are ~330MB each in RAM. On Railway (8GB limit),
        # loading 3+ v5 models + video processing causes OOM. Only load the
        # lightweight dead_ball classifier here. OCR via Ali ensemble is primary.
        _v5_lightweight = {
            "dead_ball_classifier_v5.pt": "dead_ball_classifier_v5_model",
        }
        for filename, attr in _v5_lightweight.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                setattr(self, attr, _load_model(filename))
            else:
                logger.info("v5 model pending: %s", filename)

        # ── PRIORITY GROUP 1b: v5 HEAVY models (loaded if memory allows) ──
        _v5_heavy = {
            "jersey_ocr_universal_v5.pt": "jersey_ocr_universal_v5_model",
            "player_detector_v5.pt": "player_detector_v5_model",
            "outcome_classifier_basketball_v5.pt": "outcome_cls_basketball_v5_model",
            "outcome_classifier_football_v5.pt": "outcome_cls_football_v5_model",
            "outcome_classifier_lacrosse_v5.pt": "outcome_cls_lacrosse_v5_model",
            "scoreboard_detector_v5.pt": "scoreboard_detector_v5_model",
        }
        _v5_yolo_models = {**_v5_lightweight, **_v5_heavy}
        if self._memory_ok_for_loading("v5-lightweight"):
            for filename, attr in _v5_heavy.items():
                path = os.path.join(MODEL_DIR, filename)
                if os.path.exists(path):
                    setattr(self, attr, _load_model(filename))
                else:
                    logger.info("v5 model pending: %s", filename)
        else:
            logger.info("v5 heavy models skipped — memory pressure (each ~330MB)")

        # Load jersey super-resolution model (PyTorch .pth, not YOLO)
        try:
            from app.services.jersey_upscaler import JerseyUpscaler
            from pathlib import Path
            self._jersey_upscaler = JerseyUpscaler(Path(os.path.join(MODEL_DIR, "jersey_upscaler_v5.pth")))
            if self._jersey_upscaler.load():
                logger.info("Jersey SR upscaler loaded")
            else:
                logger.info("Jersey SR upscaler not available — bicubic fallback")
        except Exception as exc:
            logger.info("Jersey SR import failed: %s — bicubic fallback", exc)
            self._jersey_upscaler = None

        if not self._memory_ok_for_loading("v5"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, {}, {}, {}, {}, {}, {})
            return

        # ── v5 VideoMAE models (HuggingFace dirs, loaded on demand) ──
        _v5_videomae_dirs = {
            "videomae_basketball_v5": "videomae_basketball_v5",
            "videomae_football_v5": "videomae_football_v5",
            "videomae_lacrosse_v5": "videomae_lacrosse_v5",
        }
        for dirname, attr in _v5_videomae_dirs.items():
            dirpath = os.path.join(MODEL_DIR, dirname)
            if os.path.isdir(dirpath):
                setattr(self, attr, dirpath)
                logger.info("v5 VideoMAE dir found: %s", dirname)
            else:
                logger.info("v5 VideoMAE pending: %s/", dirname)

        if not self._memory_ok_for_loading("v5-videomae"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, {}, {}, {}, {}, {}, _v5_videomae_dirs)
            return

        # ── PRIORITY GROUP 2: v3 OCR models (secondary confirmation) ──
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
                logger.info("v3 OCR model pending: %s", filename)

        if not self._memory_ok_for_loading("v3"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, _v3_ocr_models, {}, {}, {}, {}, _v5_videomae_dirs)
            return

        # ── PRIORITY GROUP 3: v2 models (tertiary — 0.995 mAP50 universal) ──
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
                logger.info("v2 model pending: %s", filename)

        if not self._memory_ok_for_loading("v2"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, _v3_ocr_models, _v2_models, {}, {}, {}, _v5_videomae_dirs)
            return

        # ── v2 ball/action/zone models ──
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
                logger.info("v2 ball/action/zone model pending: %s", filename)

        if not self._memory_ok_for_loading("v2-baz"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, _v3_ocr_models, _v2_models, _v2_baz_models, {}, {}, _v5_videomae_dirs)
            return

        # ── PRIORITY GROUP 4: v4 outcome models ──
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
                logger.info("v4 outcome model pending: %s", filename)

        if not self._memory_ok_for_loading("v4"):
            self._loaded = True
            self._log_load_summary(_v5_yolo_models, _v3_ocr_models, _v2_models, _v2_baz_models, _v4_outcome_models, {}, _v5_videomae_dirs)
            return

        # ── PRIORITY GROUP 5: v1 models (legacy football, load last before Ali) ──
        _v1_models_map = {}
        self.football_digit_model = _load_model("football_digit_detector.pt")
        self.football_player_model = _load_model("football_player_detector.pt")
        self.football_tracker_model = _load_model("football_jersey_tracker.pt")
        # basketball_jersey_ocr.pt — DISABLED: mAP50 0.10
        self.basketball_jersey_model = None

        self._loaded = True
        self._log_load_summary(
            _v5_yolo_models, _v3_ocr_models, _v2_models,
            _v2_baz_models, _v4_outcome_models, _v1_models_map, _v5_videomae_dirs,
        )

    def _log_load_summary(self, v5_map, v3_map, v2_map, v2baz_map, v4_map, v1_map, vmae_map):
        """Log model load counts."""
        v5_loaded = sum(1 for attr in v5_map.values() if getattr(self, attr, None) is not None)
        v3_loaded = sum(1 for attr in v3_map.values() if getattr(self, attr, None) is not None)
        v2_loaded = sum(1 for attr in v2_map.values() if getattr(self, attr, None) is not None)
        v2baz_loaded = sum(1 for attr in v2baz_map.values() if getattr(self, attr, None) is not None)
        v4_loaded = sum(1 for attr in v4_map.values() if getattr(self, attr, None) is not None)
        v1_loaded = sum(1 for m in [self.football_digit_model, self.football_player_model, self.football_tracker_model] if m is not None)
        vmae_loaded = sum(1 for attr in vmae_map.values() if getattr(self, attr, None) is not None)
        # Determine primary detection layer
        if self.jersey_ocr_universal_v5_model is not None:
            primary = "v5"
        elif self.jersey_ocr_v3_primary_model is not None:
            primary = "v3"
        elif self.jersey_number_universal_v1_model is not None:
            primary = "v2"
        else:
            primary = "ali"
        logger.info(
            "RoboflowDetector loaded (primary=%s): %d/7 v5, %d/12 v3-ocr, "
            "%d/9 v2, %d/5 v2-baz, %d/20 v4-outcome, %d/3 v1, %d/3 videomae, "
            "SR=%s, deadball=%s, scoreboard=%s",
            primary, v5_loaded, v3_loaded, v2_loaded, v2baz_loaded,
            v4_loaded, v1_loaded, vmae_loaded,
            "yes" if self._jersey_upscaler and hasattr(self._jersey_upscaler, '_model') and self._jersey_upscaler._model is not None else "fallback",
            "yes" if self.dead_ball_classifier_v5_model is not None else "no",
            "yes" if self.scoreboard_detector_v5_model is not None else "no",
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """2x upscale + CLAHE contrast + unsharp mask for better digit reading."""
        h, w = frame.shape[:2]
        upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Unsharp mask — sharpen digit edges degraded by compression
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        return sharpened

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
        self._last_used["football_digit_model"] = time.time()
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
        self._last_used["football_player_model"] = time.time()
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
        self._last_used["basketball_jersey_model"] = time.time()
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
        self._last_used["football_tracker_model"] = time.time()
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

        ocr_crop = self._maybe_upscale(crop)
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
        upscaled_crop = self._maybe_upscale(crop)
        for model, layer_name in [
            (self.jersey_number_universal_v1_model, "v2_universal_v1"),
            (self.jersey_number_universal_v2_model, "v2_universal_v2"),
        ]:
            if model is None:
                continue
            try:
                enhanced = self._preprocess(upscaled_crop)
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
        # Check memory pressure before heavy multi-model detection
        rss = self._get_rss_mb()
        if rss >= MEMORY_LIMIT_MB:
            self._evict_lru_models(MEMORY_WARNING_MB)
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

    def detect_with_v3_primary(
        self,
        frame: np.ndarray,
        jersey_number: int,
        sport: str,
        conf: float = 0.2,
    ) -> list[dict]:
        """Run jersey_ocr_v3_primary on player crop regions.

        Fallback chain for when the full v3 pipeline (player_isolator,
        color_classifier, number_region_detector) is not yet trained.
        Uses football_player_detector (v2) to find player bboxes first,
        then runs v3 primary OCR on each crop.
        """
        self.load()
        if self.jersey_ocr_v3_primary_model is None:
            return []

        results: list[dict] = []

        # Step 1: find player bboxes using v2 player detector
        players = self.detect_football_players(frame, conf=0.25)
        if not players:
            # No player boxes — run on full frame as single "crop"
            players = [{"bbox": [0, 0, frame.shape[1], frame.shape[0]]}]

        # Step 2: crop each player and run v3 primary OCR
        for player in players:
            x1, y1, x2, y2 = [int(c) for c in player["bbox"]]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Upscale small crops 2x for better OCR
            h, w = crop.shape[:2]
            if w < 200 or h < 200:
                upscaled = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            else:
                upscaled = crop

            enhanced = self._preprocess(upscaled)

            try:
                ocr_results = self.jersey_ocr_v3_primary_model(
                    enhanced, conf=conf, verbose=False
                )[0]
                for box in ocr_results.boxes:
                    class_name = self.jersey_ocr_v3_primary_model.names[int(box.cls)]
                    detected_num = self._parse_number(class_name)
                    if detected_num == jersey_number:
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        results.append({
                            "confidence": float(box.conf),
                            "bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                            "number_detected": detected_num,
                            "layer": "v3_ocr_primary_standalone",
                        })
            except Exception as e:
                logger.debug("v3 primary standalone error: %s", e)
                continue

        return results

    # ── Jersey super-resolution helper ───────────────────────────────────

    def _maybe_upscale(self, crop: np.ndarray) -> np.ndarray:
        """Upscale small jersey crops before OCR. Returns crop unchanged if large enough."""
        h, w = crop.shape[:2]
        if max(h, w) <= 64 and self._jersey_upscaler is not None:
            return self._jersey_upscaler.upscale(crop)
        return crop

    # ── Dead ball classifier ───────────────────────────────────────────────

    def classify_dead_ball(self, frame: np.ndarray, conf: float = 0.4) -> str | None:
        """Returns 'live_play', 'dead_ball', or None if model not loaded.

        Uses YOLO classify model (not detect) — probs-based inference.
        """
        if self.dead_ball_classifier_v5_model is None:
            return None
        self._last_used["dead_ball_classifier_v5_model"] = time.time()
        try:
            results = self.dead_ball_classifier_v5_model(frame, verbose=False)
            probs = results[0].probs
            top_class = probs.top1
            top_conf = probs.top1conf.item()
            class_name = self.dead_ball_classifier_v5_model.names[top_class]
            if top_conf >= conf:
                return class_name  # "live_play" or "dead_ball"
            return None
        except Exception as exc:
            logger.warning("Dead ball classify failed: %s", exc)
            return None

    # ── Scoreboard detector ────────────────────────────────────────────────

    def detect_scoreboard(self, frame: np.ndarray, conf: float = 0.3) -> list[dict]:
        """Detect scoreboard regions. Returns list of {bbox, confidence, layer}."""
        if self.scoreboard_detector_v5_model is None:
            return []
        self._last_used["scoreboard_detector_v5_model"] = time.time()
        try:
            results = self.scoreboard_detector_v5_model(frame, conf=conf, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    dets.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": box.conf.item(),
                        "layer": "scoreboard_v5",
                    })
            return dets
        except Exception as exc:
            logger.warning("Scoreboard detection failed: %s", exc)
            return []

    # ── v4 outcome detection ──────────────────────────────────────────────

    # Maps model attribute → outcome string for _OUTCOME_SCORE_BOOSTS
    _V4_SPORT_MODELS: dict[str, list[tuple[str, str]]] = {
        "basketball": [
            ("basketball_made_shot_v4_model", "made_shot"),
            ("basketball_hoop_detector_v4_model", "made_shot"),
            ("basketball_scoring_zone_v4_model", "made_shot"),
            ("basketball_dribble_drive_v4_model", "drive"),
            ("basketball_rebound_v4_model", "rebound"),
        ],
        "football": [
            ("football_completion_detector_v4_model", "completion"),
            ("football_touchdown_detector_v4_model", "touchdown"),
            ("football_sack_detector_v4_model", "sack"),
            ("football_reception_yac_v4_model", "completion"),
            ("football_qb_scramble_v4_model", "qb_scramble"),
        ],
        "lacrosse": [
            ("lacrosse_goal_detector_v4_model", "goal"),
            ("lacrosse_shot_quality_v4_model", "goal"),
            ("lacrosse_ground_ball_v4_model", "ground_ball"),
        ],
    }
    # Cross-sport models run for all sports
    _V4_CROSS_MODELS: list[tuple[str, str]] = [
        ("crowd_energy_detector_v4_model", "crowd_energy"),
    ]

    def detect_outcome_v4(
        self,
        frame: np.ndarray,
        sport: str,
        conf: float = 0.3,
    ) -> list[dict]:
        """Run sport-specific v4 outcome models on a frame.

        Returns list of {outcome, confidence, bbox, layer} dicts.
        Returns [] if no v4 models are loaded (graceful no-op).
        """
        self.load()
        dets: list[dict] = []

        sport_models = self._V4_SPORT_MODELS.get(sport.lower(), [])
        all_models = sport_models + self._V4_CROSS_MODELS

        for attr, outcome_name in all_models:
            model = getattr(self, attr, None)
            if model is None:
                continue
            self._last_used[attr] = time.time()
            try:
                results = model(frame, conf=conf, verbose=False)[0]
                for box in results.boxes:
                    dets.append({
                        "outcome": outcome_name,
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "class_name": model.names[int(box.cls)],
                        "layer": f"v4_{outcome_name}",
                    })
            except Exception as e:
                logger.debug("v4 %s error: %s", attr, e)

        return dets

    # ── v5 OCR + classification ────────────────────────────────────────────

    def detect_jersey_v5(
        self,
        crop: np.ndarray,
        jersey_number: int,
        conf: float = 0.2,
    ) -> list[dict]:
        """Run jersey_ocr_universal_v5 on a crop. Returns [] if model not loaded."""
        self.load()
        if self.jersey_ocr_universal_v5_model is None:
            return []
        self._last_used["jersey_ocr_universal_v5_model"] = time.time()

        dets: list[dict] = []
        try:
            upscaled_crop = self._maybe_upscale(crop)
            enhanced = self._preprocess(upscaled_crop)
            results = self.jersey_ocr_universal_v5_model(enhanced, conf=conf, verbose=False)[0]
            for box in results.boxes:
                name = self.jersey_ocr_universal_v5_model.names[int(box.cls)]
                num = self._parse_number(name)
                if num == jersey_number or self._check_adjacent_digits(
                    jersey_number, results.boxes, self.jersey_ocr_universal_v5_model
                ):
                    dets.append({
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "number_detected": num,
                        "layer": "v5_ocr_universal",
                    })
        except Exception as e:
            logger.debug("v5 OCR error: %s", e)
        return dets

    def detect_players_v5(
        self,
        frame: np.ndarray,
        conf: float = 0.3,
    ) -> list[dict]:
        """Run player_detector_v5 on a frame. Returns [] if model not loaded."""
        self.load()
        if self.player_detector_v5_model is None:
            return []
        self._last_used["player_detector_v5_model"] = time.time()

        players: list[dict] = []
        try:
            results = self.player_detector_v5_model(frame, conf=conf, verbose=False)[0]
            for box in results.boxes:
                name = self.player_detector_v5_model.names[int(box.cls)]
                if "player" in name.lower() or "person" in name.lower():
                    players.append({
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf),
                        "class": name,
                        "layer": "v5_player_detector",
                    })
        except Exception as e:
            logger.debug("v5 player detector error: %s", e)
        return players

    def classify_outcome_v5(
        self,
        frame: np.ndarray,
        sport: str,
        conf: float = 0.3,
    ) -> str | None:
        """Run sport-specific v5 outcome classifier on a frame.

        Returns outcome string (e.g. "made_shot") or None if no confident prediction.
        Returns None if model not loaded (graceful no-op for pre-training).
        """
        self.load()
        model_map = {
            "basketball": self.outcome_cls_basketball_v5_model,
            "football": self.outcome_cls_football_v5_model,
            "lacrosse": self.outcome_cls_lacrosse_v5_model,
        }
        model = model_map.get(sport.lower())
        if model is None:
            return None

        attr = f"outcome_cls_{sport.lower()}_v5_model"
        self._last_used[attr] = time.time()

        try:
            results = model(frame, conf=conf, verbose=False)[0]
            if results.boxes and len(results.boxes) > 0:
                best = max(results.boxes, key=lambda b: float(b.conf))
                class_name = model.names[int(best.cls)]
                return class_name.lower().replace(" ", "_")
        except Exception as e:
            logger.debug("v5 classifier (%s) error: %s", sport, e)
        return None

    def get_primary_detection(self) -> str:
        """Return which detection layer is primary based on loaded models."""
        if self.jersey_ocr_universal_v5_model is not None:
            return "v5"
        if self.jersey_ocr_v3_primary_model is not None:
            return "v3"
        if self.jersey_number_universal_v1_model is not None:
            return "v2"
        return "ali"

    def status(self) -> dict:
        """Report which models are available (file exists) or loaded (in memory).

        IMPORTANT: Does NOT call self.load(). Models are reported based on
        file existence and in-memory state. This keeps the /health endpoint
        lightweight and prevents OOM from eagerly loading 50+ models.
        """

        def _model_status(attr: str, filename: str | None = None) -> str:
            model = getattr(self, attr, None)
            if model is not None:
                return "loaded"
            if filename:
                path = os.path.join(MODEL_DIR, filename)
                if os.path.exists(path):
                    return "available"
            return "missing"

        return {
            # Detection priority metadata
            "primary_detection": self.get_primary_detection(),
            "ali_status": "demoted_fallback" if self._loaded else "not_loaded",
            # v5 models (PRIMARY)
            "jersey_ocr_universal_v5": _model_status("jersey_ocr_universal_v5_model", "jersey_ocr_universal_v5.pt"),
            "player_detector_v5": _model_status("player_detector_v5_model", "player_detector_v5.pt"),
            "outcome_cls_basketball_v5": _model_status("outcome_cls_basketball_v5_model", "outcome_classifier_basketball_v5.pt"),
            "outcome_cls_football_v5": _model_status("outcome_cls_football_v5_model", "outcome_classifier_football_v5.pt"),
            "outcome_cls_lacrosse_v5": _model_status("outcome_cls_lacrosse_v5_model", "outcome_classifier_lacrosse_v5.pt"),
            "scoreboard_detector_v5": _model_status("scoreboard_detector_v5_model", "scoreboard_detector_v5.pt"),
            "dead_ball_classifier_v5": _model_status("dead_ball_classifier_v5_model", "dead_ball_classifier_v5.pt"),
            "jersey_upscaler_v5": "loaded" if (self._jersey_upscaler and hasattr(self._jersey_upscaler, '_model') and self._jersey_upscaler._model is not None) else ("fallback" if os.path.exists(os.path.join(MODEL_DIR, "jersey_upscaler_v5.pth")) else "missing"),
            # v1 models (legacy)
            "football_digit_detector": _model_status("football_digit_model", "football_digit_detector.pt"),
            "football_player_detector": _model_status("football_player_model", "football_player_detector.pt"),
            "basketball_jersey_ocr": "skipped - pending retrain (mAP50: 0.10)",
            "football_jersey_tracker": _model_status("football_tracker_model", "football_jersey_tracker.pt"),
            # v2 models
            "basketball_jersey_number_v2": _model_status("basketball_jersey_number_v2_model", "basketball_jersey_number_v2.pt"),
            "basketball_jersey_number_v3": _model_status("basketball_jersey_number_v3_model", "basketball_jersey_number_v3.pt"),
            "basketball_player_detector_v2": _model_status("basketball_player_detector_v2_model", "basketball_player_detector_v2.pt"),
            "football_positions_detector": _model_status("football_positions_model", "football_positions_detector.pt"),
            "football_presnap_detector": _model_status("football_presnap_model", "football_presnap_detector.pt"),
            "jersey_number_universal_v1": _model_status("jersey_number_universal_v1_model", "jersey_number_universal_v1.pt"),
            "jersey_number_universal_v2": _model_status("jersey_number_universal_v2_model", "jersey_number_universal_v2.pt"),
            "lacrosse_detector_v1": _model_status("lacrosse_v1_model", "lacrosse_detector_v1.pt"),
            "lacrosse_detector_v2": _model_status("lacrosse_v2_model", "lacrosse_detector_v2.pt"),
            # v2 ball/action/zone models
            "basketball_ball_detector": _model_status("basketball_ball_detector_model", "basketball_ball_detector.pt"),
            "football_ball_detector": _model_status("football_ball_detector_model", "football_ball_detector.pt"),
            "lacrosse_ball_detector": _model_status("lacrosse_ball_detector_model", "lacrosse_ball_detector.pt"),
            "basketball_action_detector": _model_status("basketball_action_detector_model", "basketball_action_detector.pt"),
            "basketball_court_zones": _model_status("basketball_court_zones_model", "basketball_court_zones.pt"),
            # v3 OCR models
            "jersey_ocr_v3_primary": _model_status("jersey_ocr_v3_primary_model", "jersey_ocr_v3_primary.pt"),
            "jersey_ocr_v3_secondary": _model_status("jersey_ocr_v3_secondary_model", "jersey_ocr_v3_secondary.pt"),
            "basketball_ocr_v3": _model_status("basketball_ocr_v3_model", "basketball_ocr_v3.pt"),
            "football_ocr_v3": _model_status("football_ocr_v3_model", "football_ocr_v3.pt"),
            "lacrosse_ocr_v3": _model_status("lacrosse_ocr_v3_model", "lacrosse_ocr_v3.pt"),
            "player_isolator_v3": _model_status("player_isolator_v3_model", "player_isolator_v3.pt"),
            "jersey_color_classifier_v3": _model_status("jersey_color_classifier_v3_model", "jersey_color_classifier_v3.pt"),
            "number_region_detector_v3": _model_status("number_region_detector_v3_model", "number_region_detector_v3.pt"),
            "motion_blur_specialist_v3": _model_status("motion_blur_specialist_v3_model", "motion_blur_specialist_v3.pt"),
            "wide_angle_specialist_v3": _model_status("wide_angle_specialist_v3_model", "wide_angle_specialist_v3.pt"),
            "dark_jersey_specialist_v3": _model_status("dark_jersey_specialist_v3_model", "dark_jersey_specialist_v3.pt"),
            "partial_visibility_specialist_v3": _model_status("partial_visibility_specialist_v3_model", "partial_visibility_specialist_v3.pt"),
            # v4 outcome models
            "basketball_hoop_detector_v4": _model_status("basketball_hoop_detector_v4_model", "basketball_hoop_detector_v4.pt"),
            "basketball_made_shot_v4": _model_status("basketball_made_shot_v4_model", "basketball_made_shot_v4.pt"),
            "basketball_scoring_zone_v4": _model_status("basketball_scoring_zone_v4_model", "basketball_scoring_zone_v4.pt"),
            "basketball_dribble_drive_v4": _model_status("basketball_dribble_drive_v4_model", "basketball_dribble_drive_v4.pt"),
            "basketball_rebound_v4": _model_status("basketball_rebound_v4_model", "basketball_rebound_v4.pt"),
            "football_completion_detector_v4": _model_status("football_completion_detector_v4_model", "football_completion_detector_v4.pt"),
            "football_touchdown_detector_v4": _model_status("football_touchdown_detector_v4_model", "football_touchdown_detector_v4.pt"),
            "football_sack_detector_v4": _model_status("football_sack_detector_v4_model", "football_sack_detector_v4.pt"),
            "football_reception_yac_v4": _model_status("football_reception_yac_v4_model", "football_reception_yac_v4.pt"),
            "football_qb_scramble_v4": _model_status("football_qb_scramble_v4_model", "football_qb_scramble_v4.pt"),
            "lacrosse_goal_detector_v4": _model_status("lacrosse_goal_detector_v4_model", "lacrosse_goal_detector_v4.pt"),
            "lacrosse_shot_quality_v4": _model_status("lacrosse_shot_quality_v4_model", "lacrosse_shot_quality_v4.pt"),
            "lacrosse_ground_ball_v4": _model_status("lacrosse_ground_ball_v4_model", "lacrosse_ground_ball_v4.pt"),
            "crowd_energy_detector_v4": _model_status("crowd_energy_detector_v4_model", "crowd_energy_detector_v4.pt"),
            "night_game_specialist_v4": _model_status("night_game_specialist_v4_model", "night_game_specialist_v4.pt"),
            "indoor_court_specialist_v4": _model_status("indoor_court_specialist_v4_model", "indoor_court_specialist_v4.pt"),
            "crowd_obstruction_specialist_v4": _model_status("crowd_obstruction_specialist_v4_model", "crowd_obstruction_specialist_v4.pt"),
            "helmet_glare_specialist_v4": _model_status("helmet_glare_specialist_v4_model", "helmet_glare_specialist_v4.pt"),
            "low_resolution_specialist_v4": _model_status("low_resolution_specialist_v4_model", "low_resolution_specialist_v4.pt"),
            "multi_player_cluster_v4": _model_status("multi_player_cluster_v4_model", "multi_player_cluster_v4.pt"),
        }


# Singleton instance
roboflow_detector = RoboflowDetector()
