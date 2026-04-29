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
# v8.33.3: kept at 30 (was bumped to 60 in v8.33.2). For short windows
# (e.g. 200s phase1 segments) a larger pool delays the fit past most of
# the window, leaving warmup HSV labels in place for nearly every play.
WARMUP_TARGET = 30
WARMUP_MAX = 80
HSV_TIEBREAKER_SAMPLES = 16  # how many crops per cluster to score for target match

# Hue buckets used by _detect_dominant_color: (h_lo, h_hi, color_name)
# Hues are in OpenCV's 0-179 range. Buckets ordered by visual specificity.
HUE_BUCKETS = [
    (0, 10, "red"),
    (10, 20, "orange"),
    (20, 35, "yellow"),    # gold maps here too
    (35, 80, "green"),
    (80, 100, "blue"),     # cyan-ish
    (100, 130, "navy"),    # darker blue / royal
    (130, 160, "purple"),
    (160, 179, "red"),     # red wraps
]


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
        # User-supplied "navy" is often the accent color of a navy/yellow team
        # (e.g. St. Mark's wears yellow jerseys with navy numbers). _try_fit
        # auto-detects the dominant saturated hue from warmup crops; if that
        # disagrees with target_color we update target_color before scoring.
        self.target_color = (target_color or "white").lower().strip()
        self._user_supplied_color = self.target_color
        self._auto_detected_color: str | None = None
        self.device = device or ("cuda" if _torch_cuda_available() else "cpu")
        self._enabled = _is_enabled()
        self._fitted = False
        self._broken = False  # flips True if SigLIP load fails
        self._warmup_crops: list[np.ndarray] = []
        self._model: Any = None
        self._processor: Any = None
        self._kmeans: Any = None
        self._target_cluster_id: int | None = None
        # Per-track cluster cache: once we've decided track_id N is cluster 0,
        # subsequent calls with the same track_id reuse that decision without
        # another SigLIP forward pass. This keeps Railway's CPU-only inference
        # at one embedding per unique track instead of one per (frame, track).
        # v8.33.2: cleared on _try_fit so warmup HSV-fallback decisions don't
        # outlive the KMeans fit.
        self._track_cache: dict[int, str] = {}
        # Stats for debug
        self.n_predictions = 0
        self.n_target = 0
        self.n_opponent = 0
        self.n_cache_hits = 0

    # ─── Public API ─────────────────────────────────────────────────────────

    def predict(self, crop: np.ndarray, track_id: int | None = None) -> str:
        """Return 'target' or 'opponent' (or 'unknown' on bad crop).

        If track_id is provided, the cluster decision is cached so each
        unique track gets at most one SigLIP forward pass over the lifetime
        of this classifier — critical for CPU-only deployment.
        """
        self.n_predictions += 1
        if crop is None or crop.size == 0:
            return "unknown"

        # Track cache (post-fit only): same track keeps the same team
        if track_id is not None and track_id in self._track_cache:
            self.n_cache_hits += 1
            return self._track_cache[track_id]

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

        # Post-fit: SigLIP embed → KMeans predict, cache by track
        try:
            emb = self._embed([crop])
            cluster = int(self._kmeans.predict(emb)[0])
            label = "target" if cluster == self._target_cluster_id else "opponent"
            if track_id is not None:
                self._track_cache[track_id] = label
            if label == "target":
                self.n_target += 1
            else:
                self.n_opponent += 1
            return label
        except Exception as exc:
            LOGGER.warning("TeamClassifier: predict failed, falling back to HSV: %s", exc)
            self._broken = True
            return self._hsv_fallback(crop)

    def add_to_warmup(self, crop: np.ndarray) -> None:
        """Test/debug helper: push a crop into the warmup pool without prediction."""
        if crop is None or crop.size == 0:
            return
        if len(self._warmup_crops) < WARMUP_MAX:
            self._warmup_crops.append(crop.copy())

    def stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "fitted": self._fitted,
            "broken": self._broken,
            "warmup_crops": len(self._warmup_crops),
            "target_cluster_id": self._target_cluster_id,
            "user_supplied_color": self._user_supplied_color,
            "auto_detected_color": self._auto_detected_color,
            "active_target_color": self.target_color,
            "n_predictions": self.n_predictions,
            "n_target": self.n_target,
            "n_opponent": self.n_opponent,
            "n_cache_hits": self.n_cache_hits,
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
        """One-shot KMeans fit on warmup crops. Sets _target_cluster_id via HSV scoring.

        v8.33.2 fixes:
          - Auto-detect dominant saturated hue per cluster; if user-supplied
            target_color doesn't match either cluster strongly, replace it
            with the more vivid cluster's detected color.
          - Score both clusters across the user color AND its likely teammate
            colors (navy ↔ yellow/gold, white ↔ blue, etc.) so a navy/yellow
            team scored against "navy" still picks the yellow cluster as target.
          - Clear track cache so warmup HSV-fallback decisions don't stick.
        """
        if self._fitted or self._broken:
            return
        if not self._load_model():
            return
        try:
            from sklearn.cluster import KMeans
            embs = self._embed(self._warmup_crops)
            self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(embs)
            labels = self._kmeans.labels_

            # Detect each cluster's dominant saturated hue from warmup crops.
            cluster_crops: dict[int, list] = {0: [], 1: []}
            for crop, lbl in zip(self._warmup_crops, labels):
                if len(cluster_crops[lbl]) < HSV_TIEBREAKER_SAMPLES:
                    cluster_crops[lbl].append(crop)
            cluster_color = {
                0: _detect_dominant_color(cluster_crops[0]) or "unknown",
                1: _detect_dominant_color(cluster_crops[1]) or "unknown",
            }

            # Empirical insight (vision-verified on St. Mark's 2024 footage):
            # the user's accent color (e.g. "navy") shows up on BOTH teams
            # because every team has navy/dark numbers/trim somewhere. The
            # target team's distinctive partner color (yellow/gold) only
            # appears on the target team. So score each cluster on the
            # PARTNER colors (palette[1:]) — the user color is excluded
            # from scoring because it's not discriminative.
            target_palette = _expand_target_palette(self._user_supplied_color)
            partner_colors = target_palette[1:] if len(target_palette) > 1 else target_palette
            cluster_partner_score = {0: 0.0, 1: 0.0}
            cluster_primary_score = {0: 0.0, 1: 0.0}
            cluster_n = {0: 0, 1: 0}
            for crop, lbl in zip(self._warmup_crops, labels):
                cluster_n[lbl] += 1
                if cluster_n[lbl] <= HSV_TIEBREAKER_SAMPLES:
                    cluster_partner_score[lbl] += max(
                        _hsv_target_ratio(crop, c) for c in partner_colors
                    )
                    cluster_primary_score[lbl] += _hsv_target_ratio(
                        crop, target_palette[0]
                    )

            # Pick the cluster with HIGHER partner-color presence. Partner
            # colors are the discriminative ones; whichever cluster has
            # more of them is the target team.
            #
            # CRITICAL guardrail (v8.33.3): if the warmup pool didn't see
            # both teams (e.g. a 200s window where only one team is on
            # offense), neither cluster will have meaningful partner signal.
            # Picking arbitrarily then mislabels REAL target plays as
            # opponent and the downstream filter drops them. When that
            # happens, mark broken=True so all subsequent predicts use the
            # HSV fallback (matches v8.33.1 behavior — never false-cuts a
            # real target play just because we couldn't separate teams).
            partner_max = max(cluster_partner_score[0], cluster_partner_score[1])
            partner_diff = abs(cluster_partner_score[0] - cluster_partner_score[1])
            picked: int
            if partner_max < 0.10 or partner_diff < 0.05:
                LOGGER.warning(
                    "TeamClassifier: weak partner-color signal "
                    "(max=%.3f diff=%.3f) — disabling SigLIP, falling back "
                    "to HSV. cluster_colors=%s",
                    partner_max, partner_diff, cluster_color,
                )
                self._broken = True
                self._fitted = False
                return
            picked = 0 if cluster_partner_score[0] > cluster_partner_score[1] else 1
            self._target_cluster_id = picked
            cluster_score = {  # kept for the log line below
                0: f"partner={cluster_partner_score[0]:.3f}/primary={cluster_primary_score[0]:.3f}",
                1: f"partner={cluster_partner_score[1]:.3f}/primary={cluster_primary_score[1]:.3f}",
            }

            self._auto_detected_color = cluster_color[picked]
            # Sync target_color so downstream HSV fallbacks use the right color.
            if self._auto_detected_color and self._auto_detected_color != "unknown":
                self.target_color = self._auto_detected_color

            self._fitted = True
            self._warmup_crops = []  # release memory

            # CRITICAL: clear track cache so warmup HSV-fallback decisions
            # don't lock in the wrong team for the rest of the video.
            n_cleared = len(self._track_cache)
            self._track_cache.clear()

            LOGGER.info(
                "TeamClassifier: fitted target=cluster_%d cluster_colors=%s "
                "palette_scores=%s user_color=%s auto_color=%s cleared_cache=%d",
                self._target_cluster_id, cluster_color, cluster_score,
                self._user_supplied_color, self._auto_detected_color, n_cleared,
            )
        except Exception as exc:
            LOGGER.warning("TeamClassifier: fit failed (%s) — using HSV fallback", exc)
            self._broken = True


def _expand_target_palette(color: str) -> list[str]:
    """Return user color plus its likely *teammate* (paired) colors.

    Many high-school teams pair a dark accent with a vivid base — 'navy'
    jerseys are often actually yellow with navy numbers, 'maroon' often
    pairs with gold, etc. The returned palette deliberately EXCLUDES
    common opponent colors (e.g. plain white) so the cluster-scoring
    favors the target team's distinctive coloration.
    """
    base = (color or "").lower().strip()
    pairings = {
        "navy":   ["navy", "yellow", "gold"],
        "blue":   ["blue", "navy", "yellow"],
        "white":  ["white", "red", "blue"],   # white-base teams usually have a colored accent
        "red":    ["red", "white", "gold"],
        "maroon": ["maroon", "gold", "yellow"],
        "black":  ["black", "gold", "red"],
        "gold":   ["gold", "yellow", "navy"],
        "yellow": ["yellow", "gold", "navy"],
        "green":  ["green", "gold", "yellow"],
        "purple": ["purple", "gold", "yellow"],
        "orange": ["orange", "navy", "black"],
    }
    return pairings.get(base, [base] if base else ["white"])


def _detect_dominant_color(crops: list[np.ndarray]) -> str | None:
    """Return the dominant saturated hue across crops as a color name.

    Masks out low-saturation/low-value pixels (skin, gray, white, black)
    before histogramming hues, so the result reflects actual jersey color
    rather than skin tone or shadows.
    """
    if not crops:
        return None
    import cv2
    hue_counts: dict[int, int] = {}
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            sat_mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)
            hues = hsv[:, :, 0][sat_mask]
            for h in hues.tolist():
                hue_counts[h] = hue_counts.get(h, 0) + 1
        except Exception:
            continue
    if not hue_counts:
        return None
    # Find the color bucket with the most matching hues.
    best_color = None
    best_count = 0
    for h_lo, h_hi, name in HUE_BUCKETS:
        cnt = sum(c for h, c in hue_counts.items() if h_lo <= h < h_hi)
        if cnt > best_count:
            best_count = cnt
            best_color = name
    return best_color


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
