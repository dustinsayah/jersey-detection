# Pose estimation via YOLO Pose — body orientation + action classification

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "app/model/yolo11n-pose.pt")

# COCO keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class PoseResult:
    is_facing_camera: bool = False
    action: str = "standing"  # standing, running, jumping, throwing, crouching
    intensity: float = 0.0  # 0-1 athletic intensity
    shoulder_width: float = 0.0
    arm_extension: float = 0.0
    keypoints: list[tuple[float, float, float]] | None = None  # x, y, conf per keypoint


class PoseAnalyzer:
    """YOLO Pose-based body orientation and action classification."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                # Try bundled model first, then auto-download
                model_path = POSE_MODEL_PATH
                if not Path(model_path).exists():
                    model_path = "yolo11n-pose.pt"
                self._model = YOLO(model_path)
                LOGGER.info("PoseAnalyzer: loaded model from %s", model_path)
            except Exception as exc:
                LOGGER.warning("PoseAnalyzer: failed to load model: %s", exc)
        return self._model

    def analyze_frame(
        self,
        frame: np.ndarray,
        person_box: tuple[float, float, float, float] | None = None,
    ) -> PoseResult:
        """Analyze pose of person in frame.

        Args:
            frame: BGR image
            person_box: Optional (x1, y1, x2, y2) to crop to specific person

        Returns:
            PoseResult with orientation and action classification
        """
        model = self._get_model()
        if model is None:
            return PoseResult()

        try:
            # Crop to person if box provided
            img = frame
            offset_x, offset_y = 0.0, 0.0
            if person_box:
                x1, y1, x2, y2 = [int(v) for v in person_box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    img = frame[y1:y2, x1:x2]
                    offset_x, offset_y = float(x1), float(y1)

            results = model(img, conf=0.3, verbose=False)
            if not results or len(results) == 0:
                return PoseResult()

            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                return PoseResult()

            # Get the most confident person's keypoints
            kpts = result.keypoints.data.cpu().numpy()
            if len(kpts) == 0:
                return PoseResult()

            # Pick the person with highest total confidence
            best_idx = 0
            if len(kpts) > 1:
                confs = [kp[:, 2].sum() for kp in kpts]
                best_idx = int(np.argmax(confs))

            keypoints = kpts[best_idx]  # Shape: (17, 3) — x, y, conf

            return self._classify_pose(keypoints)

        except Exception as exc:
            LOGGER.warning("Pose analysis failed: %s", exc)
            return PoseResult()

    def _classify_pose(self, kpts: np.ndarray) -> PoseResult:
        """Classify body orientation and action from COCO keypoints."""
        kp_list = [(float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2])) for i in range(min(17, len(kpts)))]

        def _kp(idx: int) -> tuple[float, float, float]:
            return kp_list[idx] if idx < len(kp_list) else (0.0, 0.0, 0.0)

        # ── Body orientation ─────────────────────────────────────────────
        ls = _kp(LEFT_SHOULDER)
        rs = _kp(RIGHT_SHOULDER)

        shoulder_width = 0.0
        is_facing = False

        if ls[2] > 0.3 and rs[2] > 0.3:
            shoulder_width = abs(ls[0] - rs[0])
            shoulder_height_diff = abs(ls[1] - rs[1])

            # If shoulders are wide relative to height difference → facing camera
            # Narrow shoulders (side view) means player turned away
            if shoulder_width > 0:
                ratio = shoulder_height_diff / shoulder_width if shoulder_width > 0 else 999
                is_facing = shoulder_width > 30 and ratio < 0.5

        # ── Action classification ────────────────────────────────────────
        action = "standing"
        intensity = 0.0

        lh = _kp(LEFT_HIP)
        rh = _kp(RIGHT_HIP)
        lk = _kp(LEFT_KNEE)
        rk = _kp(RIGHT_KNEE)
        la = _kp(LEFT_ANKLE)
        ra = _kp(RIGHT_ANKLE)
        lw = _kp(LEFT_WRIST)
        rw = _kp(RIGHT_WRIST)
        le = _kp(LEFT_ELBOW)
        re = _kp(RIGHT_ELBOW)
        nose = _kp(NOSE)

        # Hip-to-shoulder line
        hip_y = (lh[1] + rh[1]) / 2 if (lh[2] > 0.2 and rh[2] > 0.2) else 0
        shoulder_y = (ls[1] + rs[1]) / 2 if (ls[2] > 0.2 and rs[2] > 0.2) else 0
        torso_height = abs(hip_y - shoulder_y) if hip_y and shoulder_y else 0

        # Knee bend angle
        knee_y = (lk[1] + rk[1]) / 2 if (lk[2] > 0.2 and rk[2] > 0.2) else 0
        ankle_y = (la[1] + ra[1]) / 2 if (la[2] > 0.2 and ra[2] > 0.2) else 0

        # ── Jumping: hips above expected position, ankles far from ground
        if hip_y > 0 and knee_y > 0 and ankle_y > 0:
            # In image coords, lower Y = higher in frame
            leg_bend = abs(knee_y - ankle_y)
            if hip_y < knee_y and leg_bend < torso_height * 0.5:
                action = "jumping"
                intensity = 0.9

        # ── Crouching: hips close to knees
        if action == "standing" and hip_y > 0 and knee_y > 0 and torso_height > 0:
            hip_knee_ratio = abs(hip_y - knee_y) / torso_height
            if hip_knee_ratio < 0.6:
                action = "crouching"
                intensity = 0.4

        # ── Running: wide stride (ankle spread) + arm swing
        if action == "standing" and la[2] > 0.2 and ra[2] > 0.2:
            ankle_spread = abs(la[0] - ra[0])
            if ankle_spread > shoulder_width * 1.2 and shoulder_width > 0:
                action = "running"
                intensity = 0.7

        # ── Throwing: high arm extension, one arm above shoulder
        if action in ("standing", "running"):
            high_arm = False
            arm_ext = 0.0

            for wrist, elbow, shoulder in [(lw, le, ls), (rw, re, rs)]:
                if wrist[2] > 0.2 and shoulder[2] > 0.2:
                    if wrist[1] < shoulder[1]:  # Wrist above shoulder (image coords)
                        high_arm = True
                        ext = abs(wrist[1] - shoulder[1])
                        arm_ext = max(arm_ext, ext)

            if high_arm and arm_ext > torso_height * 0.3:
                action = "throwing"
                intensity = 0.85

        # ── Athletic intensity from overall movement ─────────────────────
        if action == "standing":
            # Check arm positions relative to body
            arm_activity = 0.0
            for w in [lw, rw]:
                if w[2] > 0.2 and hip_y > 0:
                    if w[1] < hip_y:  # Arms above hips
                        arm_activity += 0.2
            intensity = max(0.2, arm_activity)

        return PoseResult(
            is_facing_camera=is_facing,
            action=action,
            intensity=min(1.0, intensity),
            shoulder_width=shoulder_width,
            arm_extension=0.0,
            keypoints=kp_list,
        )

    def analyze_batch(
        self,
        frames: list[np.ndarray],
        person_boxes: list[tuple[float, float, float, float] | None] | None = None,
    ) -> list[PoseResult]:
        """Analyze poses for a batch of frames."""
        results = []
        boxes = person_boxes or [None] * len(frames)
        for i, frame in enumerate(frames):
            box = boxes[i] if i < len(boxes) else None
            results.append(self.analyze_frame(frame, box))
        return results
