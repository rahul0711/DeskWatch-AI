"""Process-wide singletons for the SCRFD detector, AdaFace embedder, and
FAISS index -- loading SCRFD/AdaFace is expensive (GPU session init), so the
web server loads them once at startup and reuses them for every registration
request and every live-recognition frame.
"""
from __future__ import annotations

import logging
import threading

from app.attendance.config import AttendanceConfig, load_attendance_config
from app.attendance.detector import ScrfdDetector
from app.attendance.embedder import AdaFaceEmbedder
from app.attendance.faiss_index import FaceIndex

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_detector: ScrfdDetector | None = None
_embedder: AdaFaceEmbedder | None = None
_index: FaceIndex | None = None
_cfg: AttendanceConfig | None = None


def init_pipeline(cfg: AttendanceConfig | None = None) -> None:
    """Eagerly load everything. Call once at FastAPI startup so the first
    HTTP request isn't the one paying GPU session init cost."""
    global _detector, _embedder, _index, _cfg
    with _lock:
        if _detector is not None:
            return
        _cfg = cfg or load_attendance_config()
        logger.info("Loading SCRFD detector...")
        _detector = ScrfdDetector(_cfg)
        logger.info("Loading AdaFace embedder...")
        _embedder = AdaFaceEmbedder(_cfg)
        logger.info("Loading FAISS index...")
        _index = FaceIndex(_cfg)
        logger.info("Attendance pipeline ready (FAISS index: %d embeddings)", _index.size)


def get_config() -> AttendanceConfig:
    if _cfg is None:
        init_pipeline()
    assert _cfg is not None
    return _cfg


def get_detector() -> ScrfdDetector:
    if _detector is None:
        init_pipeline()
    assert _detector is not None
    return _detector


def get_embedder() -> AdaFaceEmbedder:
    if _embedder is None:
        init_pipeline()
    assert _embedder is not None
    return _embedder


def get_index() -> FaceIndex:
    if _index is None:
        init_pipeline()
    assert _index is not None
    return _index
