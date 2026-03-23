"""
YOLOv8 player + ball detection.
Finds moments where player matching jersey color is near the ball.
Free — downloads yolov8n.pt automatically on first run.
"""
import cv2
import numpy as np
import os
from typing import List, Dict, Optional

COLOR_RANGES = {
    "white": ([0, 0, 200], [180, 30, 255]),
    "black": ([0, 0, 0], [180, 255, 50]),
    "navy": ([100, 50, 20], [130, 255, 150]),
    "blue": ([90, 50, 50], [130, 255, 255]),
    "red": ([0, 100, 100], [10, 255, 255]),
    "green": ([40, 50, 50], [80, 255, 255]),
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "gold": ([15, 100, 100], [30, 255, 255]),
    "gray": ([0, 0, 80], [180, 30, 200]),
    "purple": ([130, 50, 50], [160, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255]),
}


def detect_with_yolo(
    video_url: str,
    jersey_color: str = "white",
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
) -> List[Dict]:
    try:
        from ultralytics import YOLO
        from layers.motion_detector import download_segment

        model = YOLO('yolov8n.pt')
        color_range = COLOR_RANGES.get(jersey_color.lower(), COLOR_RANGES["white"])

        tmp = download_segment(video_url, time_start, time_end)
        if not tmp:
            return []

        cap = cv2.VideoCapture(tmp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sample = int(fps * 3)

        results = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample == 0:
                ts = idx / fps + (time_start or 0)
                det = model(frame, verbose=False, classes=[0, 32])
                persons, balls = [], []

                for r in det:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if cls == 0 and conf > 0.4:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and color_match(crop, color_range):
                                persons.append([x1, y1, x2, y2])
                        elif cls == 32 and conf > 0.3:
                            balls.append([x1, y1, x2, y2])

                if persons and balls:
                    for p in persons:
                        for b in balls:
                            if dist(p, b) < 200:
                                h, w = frame.shape[:2]
                                results.append({
                                    "timestamp": round(ts, 1),
                                    "confidence": 0.78,
                                    "bbox": {
                                        "x1_pct": round(p[0] / w * 100, 1),
                                        "y1_pct": round(p[1] / h * 100, 1),
                                        "x2_pct": round(p[2] / w * 100, 1),
                                        "y2_pct": round(p[3] / h * 100, 1),
                                    },
                                    "layer": "yolo_player",
                                })
                                break
            idx += 1

        cap.release()
        os.unlink(tmp)

        deduped = []
        for r in sorted(results, key=lambda x: x["timestamp"]):
            if not deduped or r["timestamp"] - deduped[-1]["timestamp"] >= 10:
                deduped.append(r)

        return deduped

    except ImportError:
        print("ultralytics not installed")
        return []
    except Exception as e:
        print(f"YOLO error: {e}")
        return []


def color_match(crop, color_range):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))
    return float(np.sum(mask > 0)) / mask.size > 0.15


def dist(b1, b2):
    return (((b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2) ** 2 +
            ((b1[1] + b1[3]) / 2 - (b2[1] + b2[3]) / 2) ** 2) ** 0.5
