"""ONNX-based action recognition scorer (X3D-S).

Loads the model trained by ``colab/clipt_action_model.ipynb`` and exposes a
single function ``score_clip_video(video_path, start_s, end_s) -> float`` that
returns a 0-1 quality score for the clip.

Activation:
  - Set env var ``QB_ACTION_ONNX_URL`` to the URL hosting ``qb_x3d_s_int8.onnx``
    (Cloudinary, S3, raw GitHub, anywhere reachable from Railway).
  - The model is fetched on first use and cached at ``/tmp/qb_x3d_s_int8.onnx``.
  - If the env var is missing or the fetch fails, ``score_clip_video`` returns
    ``None`` and the caller (position_scorer) keeps its rubric-only score.

This keeps the action model entirely optional. Phase 4 ships without it; once
the Colab fine-tune produces a working ONNX, set the env var and inference
joins the pipeline with no other code change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

T_FRAMES = 16
SIZE = 182
CACHE_PATH = "/tmp/qb_x3d_s_int8.onnx"

# Class index mapping must match cell 4 of the Colab notebook.
LABEL_FROM_IDX = {0: "CUT", 1: "GOOD", 2: "GREAT"}

_session: Any = None
_session_broken = False


def _get_session() -> Any | None:
    """Lazy-load the ONNX model. Returns None if not configured / fetch fails."""
    global _session, _session_broken
    if _session_broken:
        return None
    if _session is not None:
        return _session

    url = os.getenv("QB_ACTION_ONNX_URL", "").strip()
    if not url:
        _session_broken = True
        return None

    try:
        if not os.path.exists(CACHE_PATH):
            LOGGER.info("action_scorer: fetching %s", url)
            import httpx
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            Path(CACHE_PATH).write_bytes(r.content)
            LOGGER.info("action_scorer: cached %d bytes to %s", len(r.content), CACHE_PATH)

        import onnxruntime as ort
        _session = ort.InferenceSession(CACHE_PATH, providers=["CPUExecutionProvider"])
        LOGGER.info("action_scorer: ONNX session ready")
        return _session
    except Exception as exc:
        LOGGER.warning("action_scorer: load failed (%s) — disabling action scoring", exc)
        _session_broken = True
        return None


def _extract_clip_frames(video_path: str, start_s: float, end_s: float) -> np.ndarray | None:
    """Extract T_FRAMES x SIZE x SIZE x 3 RGB frames in (1, 3, T, H, W) layout."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    times = np.linspace(start_s, end_s, T_FRAMES)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
            continue
        h, w = frame.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = frame[y0:y0 + side, x0:x0 + side]
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    arr = arr.transpose(3, 0, 1, 2)  # (3, T, H, W)
    return arr[np.newaxis, ...].astype(np.float32)  # (1, 3, T, H, W)


def score_clip_video(video_path: str, start_s: float, end_s: float) -> dict | None:
    """Run X3D-S on the clip and return a dict with class probs + a 0-1 score.

    Score formula: P(GREAT) * 1.0 + P(GOOD) * 0.5  (CUT contributes 0).
    Returns None if model isn't loaded or extraction fails.
    """
    sess = _get_session()
    if sess is None:
        return None
    try:
        x = _extract_clip_frames(video_path, start_s, end_s)
        if x is None:
            return None
        out = sess.run(None, {"video": x})[0]  # (1, 3) logits
        logits = out[0]
        # softmax
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()
        score = float(probs[2] * 1.0 + probs[1] * 0.5)
        return {
            "score": score,
            "probs": {LABEL_FROM_IDX[i]: float(probs[i]) for i in range(3)},
            "predicted": LABEL_FROM_IDX[int(np.argmax(probs))],
        }
    except Exception as exc:
        LOGGER.warning("action_scorer.score_clip_video failed: %s", exc)
        return None
