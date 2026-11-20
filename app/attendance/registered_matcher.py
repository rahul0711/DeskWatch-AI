"""Recognize REGISTERED users (Register page) on any camera.

Uses exactly the enrollment path's model and alignment -- SCRFD 5-point
landmarks -> app/attendance/alignment.align_face -> AdaFace ONNX -> FAISS
cosine search over face_embeddings -- so live embeddings land in the same
space as the registered ones (see alignment.py for why that matters).

Runs inside each camera's face worker process (process_detector.py), only
for the faces its scheduler selects -- never every frame. The FAISS index
and the user list are re-read from the database when enrollment changes
(checked every `refresh_seconds`), so a newly registered person is
recognized without restarting anything.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select

from app.attendance.alignment import align_face
from app.attendance.config import AttendanceConfig, load_attendance_config
from app.attendance.db import FaceEmbedding, User, get_session, init_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredMatch:
    user_id: int
    employee_id: str
    name: str
    score: float


class RegisteredMatcher:
    def __init__(self, cfg: AttendanceConfig | None = None, refresh_seconds: float = 15.0) -> None:
        from app.attendance.embedder import AdaFaceEmbedder
        from app.attendance.faiss_index import FaceIndex

        self._cfg = cfg or load_attendance_config()
        init_db(self._cfg)
        self._embedder = AdaFaceEmbedder(self._cfg)
        self._index = FaceIndex(self._cfg)
        self._threshold = self._cfg.match_threshold
        self._refresh = refresh_seconds
        self._checked_at = time.monotonic()
        self._signature = self._db_signature()
        self._users = self._load_users()
        logger.info(
            "RegisteredMatcher ready: %d embeddings, %d users, threshold=%.2f",
            self._index.size, len(self._users), self._threshold,
        )

    def match(self, frame: np.ndarray, landmarks_list: list[np.ndarray]) -> list[RegisteredMatch | None]:
        """One batched AdaFace pass for several faces in the same frame."""
        self._maybe_refresh()
        if self._index.size == 0 or not landmarks_list:
            return [None] * len(landmarks_list)
        crops = [align_face(frame, np.asarray(k, dtype=np.float32)) for k in landmarks_list]
        embs = self._embedder.embed_aligned_batch(crops)
        out: list[RegisteredMatch | None] = []
        for emb in embs:
            best = self._index.best_match(emb)
            if best is None or best[1] < self._threshold:
                out.append(None)
                continue
            user_id, score = best
            info = self._users.get(user_id)
            if info is None:
                out.append(None)
                continue
            out.append(RegisteredMatch(user_id, info[0], info[1], score))
        return out

    # -- enrollment changes -------------------------------------------------

    def _maybe_refresh(self) -> None:
        now = time.monotonic()
        if now - self._checked_at < self._refresh:
            return
        self._checked_at = now
        try:
            sig = self._db_signature()
            if sig != self._signature:
                self._signature = sig
                self._index.rebuild()
                self._users = self._load_users()
                logger.info("RegisteredMatcher reloaded after enrollment change (%d users)", len(self._users))
            else:
                # Names can change without embeddings changing (rename).
                self._users = self._load_users()
        except Exception as exc:
            logger.warning("RegisteredMatcher refresh failed: %s", exc)

    @staticmethod
    def _db_signature() -> tuple:
        with get_session() as session:
            return tuple(session.execute(
                select(func.count(FaceEmbedding.id), func.max(FaceEmbedding.id))
            ).one())

    @staticmethod
    def _load_users() -> dict[int, tuple[str, str]]:
        with get_session() as session:
            return {
                uid: (emp, name)
                for uid, emp, name in session.execute(select(User.id, User.employee_id, User.name)).all()
            }
