"""SCRFD face detection (insightface buffalo_l pack, detection module only --
we don't load buffalo_l's bundled ArcFace recognizer since AdaFace replaces
it)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.attendance.config import AttendanceConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    kps: np.ndarray  # (5, 2) left eye, right eye, nose, left mouth, right mouth
    det_score: float

    @property
    def width_px(self) -> int:
        return int(self.bbox[2] - self.bbox[0])


class ScrfdDetector:
    """Thin wrapper around insightface.app.FaceAnalysis(allowed_modules=['detection']).

    Raises RuntimeError at construction if CUDAExecutionProvider doesn't
    actually load -- silent CPU fallback here would tank throughput without
    any obvious symptom besides "the app feels slow".
    """

    def __init__(self, cfg: AttendanceConfig | None = None) -> None:
        cfg = cfg or AttendanceConfig()
        from insightface.app import FaceAnalysis

        self._fa = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=list(cfg.onnx_provider),
        )
        self._fa.prepare(ctx_id=0, det_size=(cfg.scrfd_det_size, cfg.scrfd_det_size))

        active = self._fa.models["detection"].session.get_providers()
        logger.info("SCRFD active providers: %s", active)
        if "CUDAExecutionProvider" not in active and "CPUExecutionProvider" not in active:
            raise RuntimeError(
                f"SCRFD did not load on any supported provider: {active}."
            )
        if "CUDAExecutionProvider" not in active:
            logger.warning("SCRFD running on CPU fallback (%s).", active[0])

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self._fa.get(image_bgr)
        return [
            DetectedFace(bbox=f.bbox.astype(np.float32), kps=f.kps.astype(np.float32), det_score=float(f.det_score))
            for f in faces
        ]
