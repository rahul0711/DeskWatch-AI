"""Persistent USER identity for short-lived person tracks.

    tracker (ByteTrack)  -> short-term visual continuity  (track_id)
    InsightFace          -> persistent identity           (user_id)

A track id is NOT a user id. Tracks die on occlusion/re-detection and come
back with a new id; the person behind them does not change. This cache is
what reconnects them:

    Track 15 -> User 1   (identified by /Recognize or the local recognizer)
    ...occlusion, ByteTrack drops 15...
    Track 42 -> User 1   (re-identified by matching its face embedding
                          against User 1's cached embeddings)

Identity sources, strongest first:
  1. explicit -- /Recognize result for the face track, or the local
     InsightFace recognizer's gallery name (data/known_faces/). These also
     teach the cache: the face's embedding is stored under that user.
  2. re-id    -- the face's InsightFace embedding matched (cosine) against
     the embeddings cached for users already seen on this camera.

Two rules keep one identity from spreading over a room:
  * one user, one track: a user is bound to at most one person track
    present in a pass, and re-id never picks a user another present track
    already holds. The same face cannot be in two places at once.
  * the gallery only learns from explicit evidence. Adding re-id matches
    to it would let one wrong match teach the gallery a stranger's face,
    which then matches that stranger every time after.

The embeddings themselves are computed in the face worker process on a
schedule (new track / after occlusion / box jump / periodic revalidation,
see app/processing/process_detector.py), never every frame -- this module
only consumes whatever embedding a face track currently carries.

Pure in-memory and clock-injected, so it is unit-testable without models.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.activity.identity_binder import Identity, match_face_to_person

logger = logging.getLogger(__name__)


@dataclass
class IdentityBinding:
    person_id: str
    person_name: str
    confidence: float
    source: str           # "explicit" | "reid"
    verified_at: float    # monotonic time of the last confirming evidence


class _Gallery:
    """Recent L2-normalised embeddings per user (a small ring each, so the
    gallery follows lighting/angle drift through the day)."""

    def __init__(self, per_user: int) -> None:
        self._per_user = per_user
        self._embs: dict[str, deque] = {}
        self.names: dict[str, str] = {}

    def add(self, person_id: str, name: str, emb: np.ndarray) -> None:
        self._embs.setdefault(person_id, deque(maxlen=self._per_user)).append(emb)
        self.names[person_id] = name

    def best_two(self, emb: np.ndarray) -> list[tuple[str, float]]:
        scored = []
        for pid, embs in self._embs.items():
            if embs:
                scored.append((pid, float(max(float(np.dot(emb, e)) for e in embs))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:2]

    def similarity(self, person_id: str, emb: np.ndarray) -> float:
        embs = self._embs.get(person_id)
        if not embs:
            return -1.0
        return float(max(float(np.dot(emb, e)) for e in embs))

    def __len__(self) -> int:
        return len(self._embs)


def _normalise(emb) -> Optional[np.ndarray]:
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    if arr.size == 0 or n <= 0:
        return None
    return arr / n


class IdentityCache:
    def __init__(
        self,
        match_threshold: float = 0.45,
        match_margin: float = 0.05,
        switch_votes: int = 2,
        binding_ttl_seconds: float = 30.0,
        embeddings_per_user: int = 12,
    ) -> None:
        # Cosine similarity a re-id match must reach, and how far it must
        # beat the runner-up user (avoids flipping between look-alikes).
        self._threshold = match_threshold
        self._margin = match_margin
        # Consecutive re-id passes needed to MOVE an already-bound track to
        # a different user (an explicit /Recognize result switches at once).
        self._switch_votes = switch_votes
        self._ttl = binding_ttl_seconds
        self._gallery = _Gallery(embeddings_per_user)
        self._bindings: dict[int, IdentityBinding] = {}
        self._last_seen: dict[int, float] = {}
        self._switch_pending: dict[int, tuple[str, int]] = {}

    # -- public ---------------------------------------------------------

    @property
    def bindings(self) -> dict[int, IdentityBinding]:
        return self._bindings

    def binding(self, track_id: int) -> Optional[IdentityBinding]:
        return self._bindings.get(track_id)

    def needs_recognition(self, track_id: int, now: float, revalidate_seconds: float) -> bool:
        """True when this person track has no identity yet, or its identity
        has not been re-confirmed for revalidate_seconds."""
        b = self._bindings.get(track_id)
        return b is None or (now - b.verified_at) > revalidate_seconds

    def update(
        self,
        face_tracks: list,
        person_tracks: list,
        explicit_by_face_track: dict[int, Identity],
        now: float,
    ) -> dict[int, IdentityBinding]:
        """Process one pass. Returns {person_track_id: binding} for the
        person tracks present in this pass that have an identity."""
        present = {p.track_id for p in person_tracks}
        for tid in present:
            self._last_seen[tid] = now
        before = {tid: b.person_id for tid, b in self._bindings.items()}

        for face in face_tracks or ():
            person_tid = match_face_to_person(face.face.bbox, person_tracks)
            if person_tid is None:
                continue
            emb = _normalise(getattr(face, "embedding", None))
            explicit = explicit_by_face_track.get(face.track_id)
            if explicit is None:
                local_name = getattr(face, "name", "Unknown")
                if local_name and local_name != "Unknown":
                    explicit = Identity(
                        # Registered users carry their employee id; a
                        # known_faces/ match only has a name.
                        person_id=getattr(face, "person_id", None) or local_name, person_name=local_name,
                        confidence=float(getattr(face, "name_confidence", 0.0) or 0.0),
                    )
            if explicit is not None and explicit.person_name:
                self._bind_explicit(person_tid, explicit, emb, now)
            elif emb is not None:
                taken = {b.person_id for tid, b in self._bindings.items() if tid in present and tid != person_tid}
                self._bind_reid(person_tid, emb, now, taken)

        self._expire(now)
        self._enforce_one_track_per_user(present, before)
        return {tid: b for tid, b in self._bindings.items() if tid in present}

    # -- internal -------------------------------------------------------

    def _bind_explicit(self, person_tid: int, ident: Identity, emb: Optional[np.ndarray], now: float) -> None:
        pid = str(ident.person_id or ident.person_name)
        prev = self._bindings.get(person_tid)
        if prev is not None and prev.person_id != pid:
            logger.info("Track %s re-identified %s -> %s (explicit)", person_tid, prev.person_name, ident.person_name)
        elif prev is None:
            logger.info("Track %s -> user %s (%s)", person_tid, ident.person_name, pid)
        self._bindings[person_tid] = IdentityBinding(
            person_id=pid, person_name=ident.person_name,
            confidence=float(ident.confidence or 1.0), source="explicit", verified_at=now,
        )
        self._switch_pending.pop(person_tid, None)
        if emb is not None:
            self._gallery.add(pid, ident.person_name, emb)

    def _bind_reid(self, person_tid: int, emb: np.ndarray, now: float, taken: set[str]) -> None:
        prev = self._bindings.get(person_tid)
        if prev is not None:
            # Already identified: re-confirm. The embedding is NOT added to
            # the gallery -- only explicit evidence teaches it (see top).
            sim = self._gallery.similarity(prev.person_id, emb)
            if sim >= self._threshold:
                prev.verified_at = now
                prev.confidence = max(prev.confidence * 0.9, sim) if prev.source == "reid" else prev.confidence
                self._switch_pending.pop(person_tid, None)
                return

        match = self._match(emb, taken)
        if match is None:
            return
        pid, sim = match
        name = self._gallery.names.get(pid, pid)
        if prev is None:
            logger.info("Track %s -> user %s (re-id, cos=%.2f)", person_tid, name, sim)
            self._bindings[person_tid] = IdentityBinding(pid, name, sim, "reid", now)
            return
        if prev.person_id == pid:
            prev.verified_at = now
            return
        # Bound to someone else: only switch after consistent evidence --
        # a tracker id swap between two people is real, one odd embedding
        # (motion blur, profile view) is not.
        cand, votes = self._switch_pending.get(person_tid, (pid, 0))
        votes = votes + 1 if cand == pid else 1
        if votes >= self._switch_votes:
            logger.info("Track %s re-identified %s -> %s (re-id, cos=%.2f)", person_tid, prev.person_name, name, sim)
            self._bindings[person_tid] = IdentityBinding(pid, name, sim, "reid", now)
            self._switch_pending.pop(person_tid, None)
        else:
            self._switch_pending[person_tid] = (pid, votes)

    def _match(self, emb: np.ndarray, taken: set[str]) -> Optional[tuple[str, float]]:
        best = self._gallery.best_two(emb)
        if not best:
            return None
        pid, sim = best[0]
        if sim < self._threshold:
            return None
        if len(best) > 1 and sim - best[1][1] < self._margin:
            return None
        if pid in taken:
            # Best match is someone already on screen in another track: this
            # face is not them, and settling for the runner-up would be a guess.
            return None
        return pid, sim

    def _enforce_one_track_per_user(self, present: set[int], before: dict[int, str]) -> None:
        """If several present tracks claim one user, keep the best claim and
        unbind the rest: explicit beats re-id, then the track that already
        held the user, then the freshest and most confident evidence."""
        claims: dict[str, list[int]] = {}
        for tid, b in self._bindings.items():
            if tid in present:
                claims.setdefault(b.person_id, []).append(tid)
        for pid, tids in claims.items():
            if len(tids) < 2:
                continue

            def strength(tid: int) -> tuple:
                b = self._bindings[tid]
                return (b.source == "explicit", before.get(tid) == pid, b.verified_at, b.confidence, -tid)

            keep = max(tids, key=strength)
            for tid in tids:
                if tid != keep:
                    logger.info(
                        "Track %s unbound from %s: track %s holds that user",
                        tid, self._bindings[tid].person_name, keep,
                    )
                    self._bindings.pop(tid, None)
                    self._switch_pending.pop(tid, None)

    def _expire(self, now: float) -> None:
        for tid in [t for t, ts in self._last_seen.items() if now - ts > self._ttl]:
            self._last_seen.pop(tid, None)
            self._bindings.pop(tid, None)
            self._switch_pending.pop(tid, None)
