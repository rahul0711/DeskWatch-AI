"""FAISS IndexFlatIP over L2-normalized 512-d AdaFace embeddings.

Inner product on unit-normalized vectors == cosine similarity, which is why
every embedding that reaches here must already be L2-normalized (enforced in
AdaFaceEmbedder.embed_aligned_batch).

The index is rebuilt from the database on every enrollment change (add
user/add face/delete user) rather than incrementally patched -- registration
is low-frequency and a full rebuild from face_embeddings is simple and can't
drift from the DB. Do not recognize against a stale in-memory index after an
enrollment change; call rebuild() (or reload from disk) first.
"""
from __future__ import annotations

import logging
import threading

import faiss
import numpy as np
from sqlalchemy import select

from app.attendance.config import AttendanceConfig
from app.attendance.db import FaceEmbedding, get_session

logger = logging.getLogger(__name__)


class FaceIndex:
    def __init__(self, cfg: AttendanceConfig | None = None) -> None:
        self._cfg = cfg or AttendanceConfig()
        self._lock = threading.Lock()
        self._index = faiss.IndexFlatIP(self._cfg.embedding_dim)
        # face_embedding_ids[i] / user_ids[i] correspond to row i of the index
        self._face_embedding_ids: list[int] = []
        self._user_ids: list[int] = []
        self.rebuild()

    def rebuild(self) -> None:
        """Reload every stored embedding from SQLite and rebuild the index
        from scratch. Call after any enrollment change."""
        session = get_session()
        try:
            rows = session.execute(select(FaceEmbedding)).scalars().all()
        finally:
            session.close()

        with self._lock:
            self._index = faiss.IndexFlatIP(self._cfg.embedding_dim)
            self._face_embedding_ids = [r.id for r in rows]
            self._user_ids = [r.user_id for r in rows]
            if rows:
                vecs = np.stack([r.embedding_array() for r in rows]).astype(np.float32)
                self._index.add(vecs)
        logger.info("FAISS index rebuilt: %d embeddings across %d unique users", len(rows), len(set(self._user_ids)))

    def search(self, embedding: np.ndarray, k: int = 1) -> list[tuple[int, float]]:
        """embedding: (512,) L2-normalized. Returns up to k (user_id,
        cosine_similarity) pairs, best first. Empty list if index is empty."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            k = min(k, self._index.ntotal)
            scores, idxs = self._index.search(embedding.reshape(1, -1).astype(np.float32), k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                results.append((self._user_ids[idx], float(score)))
            return results

    def best_match(self, embedding: np.ndarray) -> tuple[int, float] | None:
        results = self.search(embedding, k=1)
        return results[0] if results else None

    @property
    def size(self) -> int:
        return self._index.ntotal


_shared_index: FaceIndex | None = None
_shared_lock = threading.Lock()


def get_shared_index(cfg: AttendanceConfig | None = None) -> FaceIndex:
    global _shared_index
    with _shared_lock:
        if _shared_index is None:
            _shared_index = FaceIndex(cfg)
        return _shared_index
