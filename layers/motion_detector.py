"""
Motion detection using optical flow.
No ML models needed — pure OpenCV.
Detects high-motion moments = active plays.
"""
import cv2
import numpy as np
import subprocess
import tempfile
import os
from typing import List, Dict, Optional


def detect_motion_moments(
    video_url: str,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
) -> List[Dict]:
    tmp = None
    try:
        tmp = download_segment(video_url, time_start, time_end)
        if not tmp:
            return []

        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sample = int(fps * 2)  # every 2 seconds

        prev = None
        scores = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    ts = idx / fps + (time_start or 0)
                    scores.append({"timestamp": round(ts, 1), "score": float(np.mean(mag))})
                prev = gray
            idx += 1

        cap.release()
        if not scores:
            return []

        vals = [s["score"] for s in scores]
        threshold = np.mean(vals) + 0.8 * np.std(vals)

        results = []
        for s in scores:
            if s["score"] >= threshold:
                conf = min(0.85, s["score"] / (np.mean(vals) + np.std(vals)))
                results.append({
                    "timestamp": s["timestamp"],
                    "confidence": round(conf, 3),
                    "bbox": None,
                    "layer": "motion",
                })

        deduped = []
        for r in sorted(results, key=lambda x: x["timestamp"]):
            if not deduped or r["timestamp"] - deduped[-1]["timestamp"] >= 15:
                deduped.append(r)

        return deduped

    except Exception as e:
        print(f"Motion error: {e}")
        return []
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def download_segment(url, start=None, end=None):
    tmp = tempfile.mktemp(suffix='.mp4')
    cmd = ['ffmpeg', '-y']
    if start:
        cmd += ['-ss', str(start)]
    cmd += ['-i', url]
    if end and start:
        cmd += ['-t', str(end - start)]
    cmd += ['-vf', 'scale=640:360', '-r', '15', '-c:v', 'libx264', '-crf', '30', '-an', tmp]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode == 0 and os.path.exists(tmp):
        return tmp
    if os.path.exists(tmp):
        os.unlink(tmp)
    return None
