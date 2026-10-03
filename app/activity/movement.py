"""Normalized per-person body-movement score.

Why not raw pixel distance: the same walk measured 3m and 12m from a
1920x1080 camera produces wildly different pixel displacements, so a single
pixel threshold is either deaf to distant movement or permanently triggered
by near movement. Every displacement here is divided by the person's own
bounding-box diagonal, making the score "fraction of a body-length moved per
second" -- comparable between a small distant figure and a large close one.

Why per-joint rather than bbox-only: someone seated and typing barely moves
their bounding box at all, while their wrists move constantly. Tracking
groups of keypoints separately (and weighting wrists/elbows up) is what
separates "at the desk, active" from "asleep at the desk", which a box-centre
delta cannot see.

Real CCTV always contains camera noise, pose jitter and compression
artifacts, so this is explicitly tolerance-based: scores are exponentially
smoothed and compared against configurable thresholds, never against zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO keypoint groups. Each entry is (name, indices-to-average).
_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("nose", (0,)),
    ("shoulder", (5, 6)),
    ("elbow", (7, 8)),
    ("wrist", (9, 10)),
    ("hip", (11, 12)),
    ("knee", (13, 14)),
)

# Order of ActivityConfig.movement_weights: centre first, then _GROUPS.
_WEIGHT_ORDER = ("center",) + tuple(name for name, _ in _GROUPS)


def _group_point(
    keypoints: np.ndarray | None,
    conf: np.ndarray | None,
    indices: tuple[int, ...],
    min_conf: float,
) -> tuple[float, float] | None:
    """Mean xy of the named keypoint group, or None if not confidently seen."""
    if keypoints is None or conf is None:
        return None
    pts = []
    for i in indices:
        if i >= len(conf) or conf[i] < min_conf:
            continue
        pts.append((float(keypoints[i][0]), float(keypoints[i][1])))
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


@dataclass
class _TrackMotion:
    last_ts: float
    points: dict[str, tuple[float, float]]
    score: float = 0.0
    # Raw (unsmoothed) most recent score, kept for debugging/overlay.
    raw_score: float = 0.0
    history: list[float] = field(default_factory=list)


class MovementAnalyzer:
    """Per-track movement scoring. Stateful across frames, keyed by track_id.

    Lives in the parent process (not the detector subprocess) so a detector
    restart cannot wipe motion history mid-event.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        weights: tuple[float, ...] = (1.0, 0.8, 0.8, 1.0, 1.2, 0.6, 0.6),
        keypoint_min_conf: float = 0.25,
        history_len: int = 12,
    ) -> None:
        self._alpha = max(0.0, min(1.0, ema_alpha))
        self._weights = {
            name: (weights[i] if i < len(weights) else 1.0)
            for i, name in enumerate(_WEIGHT_ORDER)
        }
        self._min_conf = keypoint_min_conf
        self._history_len = history_len
        self._tracks: dict[int, _TrackMotion] = {}

    def update(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        keypoints: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
        timestamp: float,
    ) -> float:
        """Feed one observation; returns the smoothed movement score.

        Score units: fraction of the person's body diagonal travelled per
        second, weighted across the bbox centre and each visible keypoint
        group. 0.0 on a track's first frame (no displacement to measure yet).
        """
        x1, y1, x2, y2 = bbox
        diag = float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))

        points: dict[str, tuple[float, float]] = {
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        }
        for name, indices in _GROUPS:
            pt = _group_point(keypoints, keypoints_conf, indices, self._min_conf)
            if pt is not None:
                points[name] = pt

        prev = self._tracks.get(track_id)
        if prev is None:
            self._tracks[track_id] = _TrackMotion(last_ts=timestamp, points=points)
            return 0.0

        dt = timestamp - prev.last_ts
        if dt <= 1e-6:
            return prev.score

        weighted_sum = 0.0
        weight_total = 0.0
        for name, pt in points.items():
            old = prev.points.get(name)
            if old is None:
                continue
            dist = float(np.hypot(pt[0] - old[0], pt[1] - old[1]))
            w = self._weights.get(name, 1.0)
            weighted_sum += w * (dist / diag) / dt
            weight_total += w

        raw = (weighted_sum / weight_total) if weight_total > 0 else prev.raw_score
        smoothed = self._alpha * raw + (1.0 - self._alpha) * prev.score

        prev.last_ts = timestamp
        prev.points = points
        prev.raw_score = raw
        prev.score = smoothed
        prev.history.append(smoothed)
        if len(prev.history) > self._history_len:
            prev.history.pop(0)
        return smoothed

    def score(self, track_id: int) -> float:
        entry = self._tracks.get(track_id)
        return entry.score if entry is not None else 0.0

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def keep_only(self, track_ids) -> None:
        """Drop motion history for tracks no longer alive."""
        alive = set(track_ids)
        for tid in [t for t in self._tracks if t not in alive]:
            del self._tracks[tid]
