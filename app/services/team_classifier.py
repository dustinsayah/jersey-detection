"""SigLIP-based team classifier.

Replaces the fixed HSV color-range matcher in classify_team_hsv with a
data-driven clustering approach: embed every torso crop with SigLIP,
KMeans into 2 clusters during a warmup window, then label which cluster
is the target team using HSV color overlap as the tiebreaker.

Key reliability properties:
- Lazy load — model only initialized on first use, never at import time.
- Single-shot fit — once fitted on the warmup pool, no re-clustering.
- Hard fallback — every public method has a try/except path that
  delegates to classify_team_hsv if SigLIP/torch/transformers fail to load.
- Env gate — set CLIPT_USE_SIGLIP=0 to force HSV-only operation.

Inspired by roboflow/sports/common/team.py but stripped of UMAP (binary
clustering doesn't need a dim-reduction step, and dropping umap-learn
saves a large dependency install on Railway).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

SIGLIP_MODEL = "google/siglip-base-patch16-224"
WARMUP_TARGET = 30  # min crops collected before fitting
WARMUP_MAX = 80  # max collected before forcing fit
HSV_TIEBREAKER_SAMPLES = 8  # how many crops per cluster to score for target match


def _is_enabled() -> bool:
    """Honor CLIPT_USE_SIGLIP env flag. Default ON; set 0/false to disable."""
    val = os.getenv("CLIPT_USE_SIGLIP", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


class TeamClassifier:
    """Two-cluster team classifier with HSV fallback.

    Lifecycle:
        clf = TeamClassifier(target_color="navy")
        # During warmup window (first ~30 frames), every crop falls back to HSV:
        for crop in crops:
            label = clf.predict(crop)  # "target" or "opponent"
        # After clf has seen enough crops, internal fit() runs once.
        # Subsequent predict() calls use SigLIP embeddings + KMeans.

    The classifier is per-pipeline-run (one fit per video); cluster centers
    are NOT persisted across runs.
    """

    def __init__(self, target_color: str = "white", device: str | None = None):
        self.target_color = (target_color or "white").lower().strip()
        self.device = device or ("cuda" if _torch_cuda_available() else "cpu")
        self._enabled = _is_enabled()
        self._fitted = False
        self._broken = False  # flips True if SigLIP load fails
        self._warmup_crops: list[np.ndarray] = []
        self._model: Any = None
        self._processor: Any = None
        self._kmeans: Any = None
        self._target_cluster_id: int | None = None
        # Stats for debug
        self.n_predictions = 0
        self.n_target = 0
        self.n_opponent = 0

    # ─── Public API ─────────────────────────────────────────────────────────

    def predict(self, crop: np.ndarray) -> str:
        """Return 'target' or 'opponent' (or 'unknown' on bad crop)."""
        self.n_predictions += 1
        if crop is None or crop.size == 0:
            return "unknown"

        # Disabled or already broken → HSV path
        if not self._enabled or self._broken:
            return self._hsv_fallback(crop)

        # Pre-fit: collect for warmup, return HSV result for the live label
        if not self._fitted:
            if len(self._warmup_crops) < WARMUP_MAX:
                self._warmup_crops.append(crop.copy())
            if len(self._warmup_crops) >= WARMUP_TARGET:
                self._try_fit()
            return self._hsv_fallback(crop)

        # Post-fit: SigLIP embed → KMeans predict
        try:
            emb = self._embed([crop])
            cluster = int(self._kmeans.predict(emb)[0])
            label = "target" if cluster == self._target_cluster_id else "opponent"
            if label == "target":
                self.n_target += 1
            else:
                self.n_opponent += 1
            return label
        except Exception as exc:
            LOGGER.warning("TeamClassifier: predict failed, falling back to HSV: %s", exc)
            self._broken = True
            return self._hsv_fallback(crop)

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "fitted": self._fitted,
            "broken": self._broken,
            "warmup_crops": len(self._warmup_crops),
            "target_cluster_id": self._target_cluster_id,
            "n_predictions": self.n_predictions,
            "n_target": self.n_target,
            "n_opponent": self.n_opponent,
        }

    # ─── Internal ───────────────────────────────────────────────────────────

    def _hsv_fallback(self, crop: np.ndarray) -> str:
        try:
            from app.services.bytetrack_pipeline import classify_team_hsv
            return classify_team_hsv(crop, self.target_color)
        except Exception:
            return "unknown"

    def _load_model(self) -> bool:
        """Load SigLIP — return True on success, False (and mark broken) on failure."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoProcessor, SiglipVisionModel
            self._processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
            self._model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL).to(self.device)
            self._model.eval()
            LOGGER.info("TeamClassifier: SigLIP loaded on %s", self.device)
            return True
        except Exception as exc:
            LOGGER.warning("TeamClassifier: SigLIP load failed (%s) — using HSV fallback", exc)
            self._broken = True
            return False

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Batch crops through SigLIP and return mean-pooled embeddings."""
        import torch
        from PIL import Image

        # BGR → RGB → PIL
        pil_imgs = [
            Image.fromarray(c[..., ::-1]) if c.ndim == 3 else Image.fromarray(c)
            for c in crops
        ]
        inputs = self._processor(images=pil_imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        # Mean pool last_hidden_state across token dim → [batch, hidden]
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        return emb

    def _try_fit(self) -> None:
        """One-shot KMeans fit on warmup crops. Sets _target_cluster_id via HSV scoring."""
        if self._fitted or self._broken:
            return
        if not self._load_model():
            return
        try:
            from sklearn.cluster import KMeans
            embs = self._embed(self._warmup_crops)
            self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(embs)
            labels = self._kmeans.labels_

            # Tiebreaker: which cluster has higher HSV target_color overlap?
            cluster_score = {0: 0.0, 1: 0.0}
            cluster_n = {0: 0, 1: 0}
            for crop, lbl in zip(self._warmup_crops, labels):
                cluster_n[lbl] += 1
                if cluster_n[lbl] <= HSV_TIEBREAKER_SAMPLES:
                    cluster_score[lbl] += _hsv_target_ratio(crop, self.target_color)
            self._target_cluster_id = (
                0 if cluster_score[0] > cluster_score[1] else 1
            )
            self._fitted = True
            self._warmup_crops = []  # release memory
            LOGGER.info(
                "TeamClassifier: fitted (target=cluster %d, scores: %s)",
                self._target_cluster_id, cluster_score,
            )
        except Exception as exc:
            LOGGER.warning("TeamClassifier: fit failed (%s) — using HSV fallback", exc)
            self._broken = True


def _hsv_target_ratio(crop: np.ndarray, target_color: str) -> float:
    """Return fraction of pixels that match target_color HSV range. 0 if unknown color."""
    if crop is None or crop.size == 0:
        return 0.0
    color_ranges = {
        "navy":  ([100, 40, 10],  [130, 255, 120]),
        "blue":  ([100, 80, 50],  [130, 255, 255]),
        "white": ([0, 0, 180],    [180, 40, 255]),
        "red":   ([0, 100, 80],   [10, 255, 255]),
        "black": ([0, 0, 0],      [180, 255, 60]),
        "green": ([40, 50, 50],   [80, 255, 255]),
        "gold":  ([15, 80, 80],   [35, 255, 255]),
        "yellow":([20, 80, 80],   [35, 255, 255]),
        "orange":([10, 100, 100], [20, 255, 255]),
        "purple":([130, 50, 30],  [160, 255, 255]),
        "maroon":([0, 50, 30],    [10, 200, 150]),
    }
    key = target_color.lower().strip()
    if key not in color_ranges:
        return 0.0
    import cv2
    lower, upper = color_ranges[key]
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return float(np.sum(mask > 0) / mask.size)
    except Exception:
        return 0.0


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
