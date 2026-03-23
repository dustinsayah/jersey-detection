# Player tracking via BoT-SORT (built into Ultralytics)

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)

# Re-verify jersey number every N frames to reduce OCR load
REVERIFY_INTERVAL = 5


@dataclass
class TrackedPlayer:
    track_id: int
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    frames_tracked: int = 0
    last_seen_frame: int = 0
    positions: list[tuple[float, float, float, float]] = field(default_factory=list)  # x1, y1, x2, y2
    is_target: bool = False


@dataclass
class TrackingResult:
    tracks: list[TrackedPlayer] = field(default_factory=list)
    target_track_id: int | None = None
    target_frames: list[int] = field(default_factory=list)  # Frame indices where target was tracked


class PlayerTracker:
    """BoT-SORT player tracker using Ultralytics built-in tracking."""

    def __init__(self, target_jersey: int | None = None):
        self.target_jersey = target_jersey
        self.tracks: dict[int, TrackedPlayer] = {}
        self.target_track_id: int | None = None
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                # Use the person detection model already loaded in the pipeline
                self._model = YOLO("yolo26n-seg.pt")
                LOGGER.info("PlayerTracker: loaded tracking model")
            except Exception as exc:
                LOGGER.warning("PlayerTracker: failed to load model: %s", exc)
        return self._model

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        jersey_detections: list[dict] | None = None,
    ) -> list[TrackedPlayer]:
        """Process a single frame with BoT-SORT tracking.

        Args:
            frame: BGR image (numpy array)
            frame_index: Sequential frame number
            jersey_detections: Optional list of jersey detections from OCR
                Each dict has: jersey_number, confidence, x1, y1, x2, y2

        Returns:
            List of tracked players in this frame
        """
        model = self._get_model()
        if model is None:
            return []

        try:
            # Run YOLO tracking with BoT-SORT
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],  # person class only
                conf=0.3,
                verbose=False,
            )

            if not results or len(results) == 0:
                return []

            result = results[0]
            if result.boxes is None or result.boxes.id is None:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            frame_tracks: list[TrackedPlayer] = []

            for i, track_id in enumerate(track_ids):
                tid = int(track_id)
                box = boxes[i]

                if tid not in self.tracks:
                    self.tracks[tid] = TrackedPlayer(track_id=tid)

                player = self.tracks[tid]
                player.frames_tracked += 1
                player.last_seen_frame = frame_index
                player.positions.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))

                # Keep position history manageable
                if len(player.positions) > 100:
                    player.positions = player.positions[-50:]

                # Associate jersey detections with tracks
                if jersey_detections and (
                    player.jersey_number is None
                    or player.frames_tracked % REVERIFY_INTERVAL == 0
                ):
                    best_match = self._match_jersey_to_box(
                        jersey_detections, box
                    )
                    if best_match is not None:
                        player.jersey_number = best_match["jersey_number"]
                        player.jersey_confidence = best_match["confidence"]

                        # Check if this is our target player
                        if (
                            self.target_jersey is not None
                            and player.jersey_number == self.target_jersey
                            and player.jersey_confidence > 0.5
                        ):
                            player.is_target = True
                            self.target_track_id = tid

                frame_tracks.append(player)

            return frame_tracks

        except Exception as exc:
            LOGGER.warning("Tracking frame %d failed: %s", frame_index, exc)
            return []

    def _match_jersey_to_box(
        self, jersey_detections: list[dict], box: np.ndarray
    ) -> dict | None:
        """Find the jersey detection that best overlaps with a person box."""
        best: dict | None = None
        best_iou = 0.0

        bx1, by1, bx2, by2 = box

        for det in jersey_detections:
            dx1, dy1, dx2, dy2 = det["x1"], det["y1"], det["x2"], det["y2"]

            # Compute IoU
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx2, dx2)
            iy2 = min(by2, dy2)

            if ix1 >= ix2 or iy1 >= iy2:
                continue

            intersection = (ix2 - ix1) * (iy2 - iy1)
            box_area = (bx2 - bx1) * (by2 - by1)
            det_area = (dx2 - dx1) * (dy2 - dy1)
            union = box_area + det_area - intersection

            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best = det

        return best if best_iou > 0.1 else None

    def needs_ocr(self, track_id: int, frame_index: int) -> bool:
        """Check if a tracked player needs OCR re-verification.

        Returns True if:
        - Player has no jersey number yet
        - It's time for periodic re-verification
        """
        if track_id not in self.tracks:
            return True

        player = self.tracks[track_id]
        if player.jersey_number is None:
            return True

        return player.frames_tracked % REVERIFY_INTERVAL == 0

    def get_result(self) -> TrackingResult:
        """Get final tracking results."""
        target_frames = []
        if self.target_track_id is not None and self.target_track_id in self.tracks:
            target = self.tracks[self.target_track_id]
            # Estimate frame indices from positions
            target_frames = list(range(
                max(0, target.last_seen_frame - target.frames_tracked + 1),
                target.last_seen_frame + 1,
            ))

        return TrackingResult(
            tracks=list(self.tracks.values()),
            target_track_id=self.target_track_id,
            target_frames=target_frames,
        )
