"""Formation situation classifier.

Classifies each frame as offense / defense / special_teams / unknown
using player bounding box geometry features alone — no jersey reading,
no team color, no pose estimation.

Trained on Dustin's labeled clips (comments provided ground truth).
Runs in <1ms per frame and requires no GPU.
"""
import os
import logging
import pickle

import numpy as np

LOGGER = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'formation_classifier.pkl'
)

FEATURE_COLS = [
    'n_players', 'y_spread', 'x_spread', 'top_cluster_size',
    'los_confidence', 'backfield_players', 'wide_splits',
    'player_density_center', 'top_cluster_y',
]


def _extract_features(boxes, frame_h, frame_w):
    if len(boxes) < 4:
        return None
    centers = np.array([
        [(b[0] + b[2]) / (2 * frame_w), (b[1] + b[3]) / (2 * frame_h)]
        for b in boxes
    ])
    n = len(centers)
    xs, ys = centers[:, 0], centers[:, 1]
    cluster_sizes = [int(np.sum(np.abs(ys - yc) < 0.08)) for yc in ys]
    top_cluster_size = max(cluster_sizes)
    if top_cluster_size >= 4:
        top_cluster_y = float(np.median([
            ys[i] for i in range(n) if cluster_sizes[i] == top_cluster_size
        ]))
        backfield_players = int(np.sum(np.abs(ys - top_cluster_y) > 0.08))
    else:
        top_cluster_y = float(np.median(ys))
        backfield_players = 0
    center_x = float(np.median(xs))
    return {
        'n_players': n,
        'y_spread': round(float(np.std(ys)), 4),
        'x_spread': round(float(np.std(xs)), 4),
        'top_cluster_size': top_cluster_size,
        'los_confidence': float(top_cluster_size >= 4),
        'backfield_players': backfield_players,
        'wide_splits': int(np.sum(np.abs(xs - center_x) > 0.35)),
        'player_density_center': round(float(np.sum((xs > 0.33) & (xs < 0.67)) / n), 4),
        'top_cluster_y': round(top_cluster_y, 4),
    }


def _heuristic_predict(feat):
    """Rule-based fallback when sklearn model is not available."""
    if feat['n_players'] < 4:
        return 'unknown', 0.5
    if feat['wide_splits'] >= 4 and feat['top_cluster_size'] <= 3:
        return 'special_teams', 0.7
    if feat['top_cluster_size'] <= 3 and feat['y_spread'] > 0.15:
        return 'defense', 0.65
    if feat['top_cluster_size'] >= 4 and feat['backfield_players'] >= 1:
        return 'offense', 0.75
    return 'unknown', 0.4


class FormationClassifier:
    def __init__(self):
        self._clf = None
        self._le = None
        self._load_model()

    def _load_model(self):
        path = os.path.abspath(MODEL_PATH)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self._clf = data['classifier']
                self._le = data['label_encoder']
                LOGGER.info('FormationClassifier: loaded sklearn model from %s', path)
            except Exception as e:
                LOGGER.warning('FormationClassifier: failed to load model (%s), using heuristic', e)
        else:
            LOGGER.info('FormationClassifier: no model at %s, using heuristic', path)

    def predict(self, boxes_xyxy, frame_h, frame_w):
        """Returns (situation, confidence) where situation is one of:
        offense / defense / special_teams / unknown.
        """
        feat = _extract_features(boxes_xyxy, frame_h, frame_w)
        if feat is None:
            return 'unknown', 0.0
        if self._clf is not None:
            try:
                x = np.array([[feat[c] for c in FEATURE_COLS]])
                pred = self._clf.predict(x)[0]
                proba = self._clf.predict_proba(x)[0]
                return self._le.inverse_transform([pred])[0], float(proba.max())
            except Exception as e:
                LOGGER.warning('FormationClassifier.predict failed (%s), using heuristic', e)
        return _heuristic_predict(feat)
