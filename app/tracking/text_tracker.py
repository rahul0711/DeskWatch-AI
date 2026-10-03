"""Multi-frame temporal voting for OCR text boxes (app/detection/ocr_detector.py).

A single OCR pass on a single frame is noisy -- motion blur, a bad angle, or
a moment of partial occlusion can turn "EXIT" into "EX1T" or drop a
character entirely. Scanning a wall sign or a badge across many frames and
voting on the readings that came back is what actually gets a clean,
trustworthy string instead of a flickering single-frame guess.

This is deliberately the same shape as SimpleIouTracker (see
app/tracking/simple_tracker.py) -- IoU-matched boxes across frames, missed
tracks kept alive briefly -- with two OCR-specific additions on top:
  1. Each track keeps a rolling window of (text, confidence) readings instead
     of just the latest box, and reports a voted/majority string.
  2. Similar-but-not-identical readings ("EXIT", "EX1T", "EXIT ") are
     clustered by string similarity before voting, so one OCR misread
     doesn't split what's obviously the same physical text into two
     low-confidence tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Two readings are treated as "the same text" for voting purposes above this
# similarity ratio (difflib's SequenceMatcher.ratio(), 0..1). Chosen loosely
# enough to absorb single-character OCR noise ("EXIT" vs "EX1T" ~ 0.75-0.9
# depending on length) without merging genuinely different short strings.
_SIMILARITY_THRESHOLD = 0.72

# How many recent readings a track remembers for voting. Long enough to
# outvote a couple of bad frames, short enough that a sign whose text
# genuinely changes (a different card held up to the camera) isn't stuck
# reporting the old string for too long.
_HISTORY_LEN = 8

# A track's voted text is only reported as "confirmed" once this many
# readings have landed in its winning cluster -- avoids flashing a
# single-frame misread into the UI as if it were solid.
_MIN_VOTES_TO_CONFIRM = 2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


@dataclass
class TextReading:
    """One OCR result for one frame, before temporal voting."""
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float
    quad: list | None = None          # rotated 4-point polygon, if known
    moving: bool = False              # came from a tracked-carrier crop, not the static full-frame pass


@dataclass
class TextTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    quad: list | None
    history: list[tuple[str, float]] = field(default_factory=list)
    frames_since_seen: int = 0
    ever_moving: bool = False

    def voted(self) -> tuple[str, float, int]:
        """Cluster recent readings by similarity, return the (text, avg
        confidence, vote count) of the strongest cluster. Empty text still
        counts as "no readable text yet" and never wins over a real string."""
        clusters: list[list[tuple[str, float]]] = []
        for text, conf in self.history:
            if not text:
                continue
            placed = False
            for cluster in clusters:
                if SequenceMatcher(None, text, cluster[0][0]).ratio() >= _SIMILARITY_THRESHOLD:
                    cluster.append((text, conf))
                    placed = True
                    break
            if not placed:
                clusters.append([(text, conf)])
        if not clusters:
            return "", 0.0, 0

        best = max(clusters, key=lambda c: sum(conf for _, conf in c))
        # Canonical string: the single highest-confidence reading in the
        # winning cluster (not e.g. the most common one) -- OCR confidence
        # is a decent proxy for "which exact spelling/casing was cleanest."
        text = max(best, key=lambda tc: tc[1])[0]
        avg_conf = sum(conf for _, conf in best) / len(best)
        return text, avg_conf, len(best)


class TextTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed_frames: int = 20) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed_frames
        self._next_id = 1
        self._tracks: list[TextTrack] = []

    def update(self, readings: list[TextReading]) -> list[TextTrack]:
        unmatched_tracks = list(self._tracks)
        matched: list[TextTrack] = []

        for reading in readings:
            best_track, best_iou = None, 0.0
            for track in unmatched_tracks:
                score = _iou(reading.bbox, track.bbox)
                if score > best_iou:
                    best_track, best_iou = track, score

            if best_track is not None and best_iou >= self._iou_threshold:
                track = best_track
                unmatched_tracks.remove(track)
            else:
                track = TextTrack(track_id=self._next_id, bbox=reading.bbox, quad=reading.quad)
                self._next_id += 1

            track.bbox = reading.bbox
            track.quad = reading.quad
            track.frames_since_seen = 0
            track.ever_moving = track.ever_moving or reading.moving
            track.history.append((reading.text, reading.confidence))
            if len(track.history) > _HISTORY_LEN:
                track.history.pop(0)
            matched.append(track)

        for track in unmatched_tracks:
            track.frames_since_seen += 1

        self._tracks = matched + [
            t for t in unmatched_tracks if t.frames_since_seen <= self._max_missed
        ]
        return matched

    @property
    def min_votes_to_confirm(self) -> int:
        return _MIN_VOTES_TO_CONFIRM
