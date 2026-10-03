"""Step 5: best-frame selection per track.

Each active track buffers candidate crops as they arrive and keeps only the
highest-scoring one seen so far -- scored on box area (bigger = closer =
more resolvable detail), frontality (how symmetric the 5 landmarks are,
i.e. how square-on to the camera the face is), and sharpness (variance of
Laplacian -- motion blur / defocus tank this fast). We never re-score with
the full embedder; this is cheap enough to run on every detection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.attendance.alignment import align_face
from app.attendance.detector import DetectedFace

# Heuristic normalization constants, not physical units -- tuned so each
# term contributes roughly 0-1 before weighting. A camera with very different
# typical face sizes/lighting may want different caps; not worth making
# these env-configurable for an MVP scorer.
_AREA_NORM_PX = 220.0  # face width (px) considered "large enough" to cap the area term at 1.0
_SHARPNESS_NORM = 400.0  # variance-of-Laplacian considered "sharp enough" to cap that term at 1.0

_W_AREA = 0.4
_W_FRONTAL = 0.35
_W_SHARP = 0.25


def sharpness_score(gray_crop: np.ndarray) -> float:
    """Higher = sharper. Variance of the Laplacian is a standard cheap blur
    metric: a crisp image has strong edges (high second-derivative
    variance); a blurry/motion-smeared one is closer to flat everywhere."""
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def frontality_score(kps: np.ndarray) -> float:
    """1.0 = perfectly frontal (eyes/nose symmetric), lower = more profile
    /turned. Uses the classic eye-nose horizontal-symmetry ratio: how
    centered the nose is between the two eyes horizontally."""
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-3:
        return 0.0
    nose_to_left = nose[0] - left_eye[0]
    nose_to_right = right_eye[0] - nose[0]
    total = nose_to_left + nose_to_right
    if total <= 0:
        return 0.0
    # 0.5/0.5 split (nose exactly centered) -> 1.0; fully off to one side -> 0.0
    balance = min(nose_to_left, nose_to_right) / max(nose_to_left, nose_to_right) if max(nose_to_left, nose_to_right) > 0 else 0.0
    return float(np.clip(balance, 0.0, 1.0))


def score_face(image_bgr: np.ndarray, face: DetectedFace) -> float:
    area_term = min(face.width_px / _AREA_NORM_PX, 1.0)
    frontal_term = frontality_score(face.kps)
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        sharp_term = 0.0
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp_term = min(sharpness_score(gray) / _SHARPNESS_NORM, 1.0)
    return _W_AREA * area_term + _W_FRONTAL * frontal_term + _W_SHARP * sharp_term


@dataclass
class TrackBuffer:
    track_id: int
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    best_score: float = -1.0
    best_aligned_crop: np.ndarray | None = None
    best_face_width_px: int = 0
    best_det_score: float = 0.0
    frames_seen: int = 0

    def consider(self, image_bgr: np.ndarray, face: DetectedFace) -> None:
        self.last_seen = time.monotonic()
        self.frames_seen += 1
        score = score_face(image_bgr, face)
        if score > self.best_score:
            self.best_score = score
            self.best_aligned_crop = align_face(image_bgr, face.kps)
            self.best_face_width_px = face.width_px
            self.best_det_score = face.det_score

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.first_seen

    @property
    def is_ready(self) -> bool:
        return self.best_aligned_crop is not None


class TrackBufferStore:
    """Owns one TrackBuffer per live track ID."""

    def __init__(self) -> None:
        self._buffers: dict[int, TrackBuffer] = {}

    def update(self, track_id: int, image_bgr: np.ndarray, face: DetectedFace) -> None:
        buf = self._buffers.get(track_id)
        if buf is None:
            buf = TrackBuffer(track_id=track_id)
            self._buffers[track_id] = buf
        buf.consider(image_bgr, face)

    def pop(self, track_id: int) -> TrackBuffer | None:
        return self._buffers.pop(track_id, None)

    def get(self, track_id: int) -> TrackBuffer | None:
        return self._buffers.get(track_id)

    def active_ids(self) -> set[int]:
        return set(self._buffers.keys())
