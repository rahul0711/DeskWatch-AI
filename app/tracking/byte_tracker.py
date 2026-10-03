"""ByteTrack-based tracker for face detections.

SimpleIouTracker (single-frame IoU, no motion model) was always a Phase 1
placeholder for this -- see its module docstring. ActivityDetector already
gets motion-model tracking "for free" via Ultralytics' own YOLO.track()
(BoT-SORT), but SCRFD/InsightFace face detections never go through a YOLO
model's .track() call, so this drives Ultralytics' bundled BYTETracker
directly off raw DetectedFace boxes instead.

Why this matters for faces specifically: a person turning their head or
briefly occluding their own face (hand, cup, walking past a monitor) is
exactly the case SimpleIouTracker loses -- one skipped detection with low
box overlap on return and it's a new track_id, which resets any
recognition/voting state tied to that id. ByteTrack's motion prediction
(Kalman filter) plus its two-stage high/low-confidence association keeps
the same id through a few missed or low-confidence frames instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.detection.face_detector import DetectedFace
from app.tracking.simple_tracker import TrackedFace

_TRACKER_CFG = Path(__file__).parent.parent.parent / "models" / "face_bytetrack.yaml"


class _DetResults:
    """Minimal stand-in for an Ultralytics Results object.

    BYTETracker only ever reads .conf/.cls/.xywh off what it's given, and
    indexes/lens it like a numpy array (see parse_bboxes and
    _split_detections in ultralytics.trackers.byte_tracker) -- it never
    needs the rest of a real Results object, which only exists when
    detection came from a YOLO model's own .predict()/.track() call.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetResults":
        return _DetResults(self.xywh[mask], self.conf[mask], self.cls[mask])


def _to_xywh(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    # max(1, ...): BYTETracker drops any box with w or h <= 0 outright
    # (tlwh_to_xyah would otherwise divide by a zero height), so a
    # degenerate detector box would silently vanish instead of tracking.
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1))


class ByteTrackFaceTracker:
    """Drop-in replacement for SimpleIouTracker: update(faces) -> list[TrackedFace].

    Keeps one persistent TrackedFace per track_id across calls (mutated in
    place, not recreated) so state attached to it elsewhere -- recognized
    name, name_confidence -- survives across frames the same way
    SimpleIouTracker's in-place mutation did. BYTETracker itself has no
    concept of that state; it only hands back (bbox, track_id, score).
    """

    def __init__(self, tracker_cfg: str | Path | None = None) -> None:
        from ultralytics.cfg import get_cfg
        from ultralytics.trackers.byte_tracker import BYTETracker

        # tracker_cfg lets non-face callers bring their own tuning -- the
        # activity detector tracks person boxes through this same class.
        # Note track_buffer is counted in FRAMES here (this Ultralytics
        # version uses it as-is, with no frame-rate scaling), so a config
        # for a detector sampling at 4fps needs a much smaller number than
        # one written for 30fps video.
        args = get_cfg(str(tracker_cfg or _TRACKER_CFG))
        self._tracker = BYTETracker(args)
        self._tracks: dict[int, TrackedFace] = {}

    def update(self, faces: list[DetectedFace]) -> list[TrackedFace]:
        if faces:
            xywh = np.array([_to_xywh(f.bbox) for f in faces], dtype=np.float32)
            conf = np.array([f.confidence for f in faces], dtype=np.float32)
            cls = np.zeros(len(faces), dtype=np.float32)
        else:
            xywh = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls = np.empty((0,), dtype=np.float32)

        rows = self._tracker.update(_DetResults(xywh, conf, cls))

        matched: list[TrackedFace] = []
        for row in rows:
            track_id = int(row[4])
            idx = int(row[7])
            face = faces[idx]  # original (unsmoothed) box -- best for crop/landmark accuracy

            track = self._tracks.get(track_id)
            if track is None:
                track = TrackedFace(track_id=track_id, face=face)
                self._tracks[track_id] = track
            else:
                track.face = face
            track.frames_since_seen = 0
            matched.append(track)

        # BYTETracker fully forgets a track_id once it's past track_buffer
        # frames lost -- anything still tracked or merely lost stays in one
        # of these two lists, so drop our own bookkeeping in lockstep
        # rather than growing this dict forever.
        known_ids = {t.track_id for t in self._tracker.tracked_stracks} | {
            t.track_id for t in self._tracker.lost_stracks
        }
        self._tracks = {tid: t for tid, t in self._tracks.items() if tid in known_ids}

        return matched
