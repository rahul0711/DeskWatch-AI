"""Associate face-recognition identities with person tracks.

Tracking and identity are separate concerns, deliberately:

    face track  (SCRFD + ByteTrack)   answers WHERE a face is, frame to frame
    /Recognize                        answers WHO that face is
    person track (YOLO + BoT-SORT)    answers WHERE a body is, frame to frame

Activity state belongs to the PERSON track (a body stays trackable when the
head turns away, which is exactly when face recognition fails), so an
identity learned about a face track has to be transferred onto the person
track that contains it. That is all this module does: pure geometry, no
clock, no model.

The containment test runs face-in-body rather than IoU because a face box is
a small fraction of a person box -- their IoU is low even for a perfect
match, so IoU would reject every correct pairing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# A face must be at least this much inside a person box to be considered
# that person's face.
_MIN_FACE_CONTAINMENT = 0.6
# ...and must sit in the upper portion of the body box. Without this, a face
# detected on a poster or a second person crouching behind a desk can be
# credited to whoever's box happens to overlap it.
_MAX_FACE_DEPTH = 0.55


@dataclass(frozen=True)
class Identity:
    person_id: Optional[str]
    person_name: str
    confidence: float


def _containment(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0
    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    return inter / inner_area


def match_face_to_person(
    face_bbox: tuple[int, int, int, int],
    person_tracks: list,
) -> Optional[int]:
    """Return the track_id of the person whose body contains this face.

    Picks the best-containing candidate, breaking ties toward the smaller
    body box -- when two boxes both contain a face (someone standing behind
    someone seated), the tighter one is the better owner.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    face_cy = (fy1 + fy2) / 2

    best_id: Optional[int] = None
    best_key: tuple[float, float] | None = None

    for p in person_tracks:
        px1, py1, px2, py2 = p.face.bbox
        ph = py2 - py1
        if ph <= 0:
            continue
        containment = _containment(face_bbox, p.face.bbox)
        if containment < _MIN_FACE_CONTAINMENT:
            continue
        # How far down the body the face sits, 0 = top of head.
        depth = (face_cy - py1) / ph
        if depth > _MAX_FACE_DEPTH:
            continue
        area = max(1, (px2 - px1) * ph)
        key = (-containment, area)
        if best_key is None or key < best_key:
            best_key, best_id = key, p.track_id

    return best_id


def bind(
    face_tracks: list,
    person_tracks: list,
    identities_by_face_track: dict[int, Identity],
) -> dict[int, Identity]:
    """Map person track_id -> Identity, for whichever faces are identified.

    `identities_by_face_track` is what recognition has learned so far, keyed
    by FACE track id (see app/web/server.py's /Recognize handling). Faces
    with no identity yet are skipped rather than binding "Unknown" over a
    previously good identity -- letting the state manager's grace period
    decide when an identity has really gone stale.
    """
    out: dict[int, Identity] = {}
    for face in face_tracks:
        identity = identities_by_face_track.get(face.track_id)
        if identity is None or not identity.person_name:
            continue
        person_track_id = match_face_to_person(face.face.bbox, person_tracks)
        if person_track_id is None:
            continue
        existing = out.get(person_track_id)
        if existing is None or identity.confidence > existing.confidence:
            out[person_track_id] = identity
    return out
