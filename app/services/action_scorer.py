"""ONNX-based action recognition scorer (X3D-XS).

Loads the model trained by ``colab/clipt_action_model.ipynb`` and exposes
``score_clip_video(video_path, start_s, end_s)`` which returns a dict with a
0-1 quality score, class probabilities, and the predicted class.

Activation:
  - Set env var ``QB_ACTION_ONNX_URL`` to the URL hosting
    ``qb_x3d_xs_int8.onnx`` (Cloudinary, S3, raw GitHub, anywhere reachable
    from Railway).
  - The model is fetched on first use and cached at ``/tmp/qb_x3d_xs_int8.onnx``.
  - If the env var is missing or the fetch fails, ``score_clip_video`` returns
    ``None`` and the caller (position_scorer) keeps its rubric-only score.

v8.34.7 changes:
  * Switched the documented backbone from X3D-S → X3D-XS. X3D-S inference
    measured at ~8s/clip on Railway CPU (over the 5s budget). X3D-XS is
    2-3x faster and accurate enough for the binary CUT-vs-keep task we're
    actually training.
  * Hard 4-second per-call timeout. If inference exceeds it, the call
    returns None and the pipeline falls back to the rubric scorer. Implemented
    via threading.Timer (signal.alarm doesn't work on non-main threads, and
    Railway runs detection in a thread).
  * ``get_model_status()`` for health checks: reports whether the URL is set,
    the cache file exists, the session is loaded, and the last-inference ms.

Backwards compatible: cache path now resolves to whichever of the two
filenames is present (so an existing X3D-S install keeps working until you
flip QB_ACTION_ONNX_URL to the XS model).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

T_FRAMES = 16
SIZE = 182  # X3D-XS input is 4x182x182 by default but we use the 16-frame variant
PER_CALL_TIMEOUT_S = 4.0  # hard ceiling — Railway budget is 5s

# Class index mapping. Trained model is binary (CUT vs KEEP) when fewer than
# 30 GREAT+GOOD labels exist; falls back to 3-class otherwise. Inference code
# handles both — softmax(logits) yields N values where N matches the trained
# head, and the score formula adapts.
LABEL_FROM_IDX_BINARY = {0: "CUT", 1: "KEEP"}
LABEL_FROM_IDX_3CLASS = {0: "CUT", 1: "GOOD", 2: "GREAT"}

CACHE_DIR = "/tmp"
# Probe order — first hit wins. Lets the same code work whether you uploaded
# x3d_xs (current target) or x3d_s (prior version).
CANDIDATE_FILENAMES = ["qb_x3d_xs_int8.onnx", "qb_x3d_s_int8.onnx"]

_session: Any = None
_session_broken = False
_session_input_name: str | None = None
_session_num_classes: int = 2  # set after loading
_last_inference_ms: float | None = None
_total_inferences: int = 0
_total_timeouts: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal: model load + frame extraction + threaded inference
# ─────────────────────────────────────────────────────────────────────────────


def _cached_path() -> str:
    """Return the cached ONNX path, preferring the XS filename if present."""
    for fname in CANDIDATE_FILENAMES:
        p = os.path.join(CACHE_DIR, fname)
        if os.path.exists(p):
            return p
    return os.path.join(CACHE_DIR, CANDIDATE_FILENAMES[0])


def _get_session() -> Any | None:
    """Lazy-load the ONNX model. Returns None if not configured / fetch fails."""
    global _session, _session_broken, _session_input_name, _session_num_classes
    if _session_broken:
        return None
    if _session is not None:
        return _session

    url = os.getenv("QB_ACTION_ONNX_URL", "").strip()
    if not url:
        _session_broken = True
        LOGGER.info("action_scorer: QB_ACTION_ONNX_URL not set — scorer disabled")
        return None

    cache = _cached_path()
    try:
        if not os.path.exists(cache):
            LOGGER.info("action_scorer: fetching %s -> %s", url, cache)
            import httpx
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            Path(cache).write_bytes(r.content)
            LOGGER.info("action_scorer: cached %d bytes", len(r.content))

        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            cache, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        # Detect input name + output shape so we adapt to whichever head was trained
        _session_input_name = _session.get_inputs()[0].name
        out_shape = _session.get_outputs()[0].shape
        _session_num_classes = out_shape[-1] if out_shape and out_shape[-1] else 2
        LOGGER.info(
            "action_scorer: ONNX session ready (input=%s, num_classes=%d, file=%s)",
            _session_input_name, _session_num_classes, os.path.basename(cache),
        )
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


class _InferenceTimeout(Exception):
    pass


def _run_with_timeout(sess: Any, x: np.ndarray, timeout_s: float) -> np.ndarray | None:
    """Run sess.run on a worker thread; raise _InferenceTimeout if it exceeds budget.

    Returns the output array, or None if cancelled/timeout. ONNX Runtime ignores
    the kill (Python can't cancel a C extension call), but we stop waiting and
    fall back. The orphaned worker finishes and its result is discarded.
    """
    result_box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result_box["out"] = sess.run(None, {_session_input_name or "video": x})[0]
        except Exception as exc:
            result_box["err"] = exc

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        # Timeout — leave the thread to finish on its own; don't block.
        raise _InferenceTimeout(f"exceeded {timeout_s}s budget")
    if "err" in result_box:
        raise result_box["err"]
    return result_box.get("out")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_model_status() -> dict:
    """Health-check helper for routes/health.py and CLI debugging.

    Reports configuration + last-inference stats. Safe to call without first
    triggering a model load (it does NOT eagerly fetch).
    """
    cache = _cached_path()
    return {
        "url_configured": bool(os.getenv("QB_ACTION_ONNX_URL", "").strip()),
        "cache_path": cache,
        "cache_exists": os.path.exists(cache),
        "cache_bytes": os.path.getsize(cache) if os.path.exists(cache) else 0,
        "session_loaded": _session is not None,
        "session_broken": _session_broken,
        "input_name": _session_input_name,
        "num_classes": _session_num_classes,
        "per_call_timeout_s": PER_CALL_TIMEOUT_S,
        "last_inference_ms": _last_inference_ms,
        "total_inferences": _total_inferences,
        "total_timeouts": _total_timeouts,
    }


def score_clip_video(video_path: str, start_s: float, end_s: float) -> dict | None:
    """Run the action model on a clip and return {score, probs, predicted}.

    Score formula:
      - 2-class head (binary CUT/KEEP): score = P(KEEP)
      - 3-class head (CUT/GOOD/GREAT): score = P(GREAT) + 0.5 * P(GOOD)

    Returns None on any of: model not loaded, timeout, frame extraction
    failure, or runtime exception. Caller falls back to rubric-only score.
    """
    global _last_inference_ms, _total_inferences, _total_timeouts

    sess = _get_session()
    if sess is None:
        return None
    try:
        x = _extract_clip_frames(video_path, start_s, end_s)
        if x is None:
            return None

        t0 = time.perf_counter()
        try:
            out = _run_with_timeout(sess, x, PER_CALL_TIMEOUT_S)
        except _InferenceTimeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _total_timeouts += 1
            _last_inference_ms = elapsed_ms
            LOGGER.warning(
                "action_scorer: timeout after %.0fms (budget %.0fms) — falling back",
                elapsed_ms, PER_CALL_TIMEOUT_S * 1000,
            )
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _last_inference_ms = elapsed_ms
        _total_inferences += 1

        if out is None:
            return None

        logits = out[0] if out.ndim > 1 else out
        # softmax
        ex = np.exp(logits - logits.max())
        probs = ex / ex.sum()

        if _session_num_classes == 2:
            score = float(probs[1])  # P(KEEP)
            label_map = LABEL_FROM_IDX_BINARY
        else:
            # 3-class head
            score = float(probs[2] * 1.0 + probs[1] * 0.5)
            label_map = LABEL_FROM_IDX_3CLASS

        return {
            "score": round(score, 3),
            "probs": {label_map[i]: float(round(probs[i], 3)) for i in range(min(len(probs), len(label_map)))},
            "predicted": label_map[int(np.argmax(probs))],
            "inference_ms": round(elapsed_ms, 1),
        }
    except Exception as exc:
        LOGGER.warning("action_scorer.score_clip_video failed: %s", exc)
        return None
