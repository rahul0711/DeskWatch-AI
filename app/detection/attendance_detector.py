"""InsightFace FaceAnalysis detector for Face Attendance.

Uses InsightFace buffalo_l pack, detection module only (SCRFD) -- identity
matching happens server-side via the external /Recognize API, which is
handed a JPEG crop and does its own embedding/matching, so a local ArcFace
embedding was computed here and never read by anything (see the
allowed_modules note in __init__).
"""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np

from app.detection.face_detector import DetectedFace

logger = logging.getLogger(__name__)


class InsightFaceAttendanceDetector:
    """InsightFace FaceAnalysis with buffalo_l, detection (SCRFD) only.

    Used for the Face Attendance pipeline -- its output feeds
    CameraWebSession._maybe_recognize (app/web/server.py), which crops and
    POSTs to the external /Recognize API for the actual identity match.
    """

    # See FaceDetector.TRACKER_BACKEND (app/detection/face_detector.py) --
    # same reasoning applies here.
    TRACKER_BACKEND = "bytetrack"

    def __init__(
        self,
        conf_threshold: float = 0.5,
        det_size: int = 640,
        onnx_provider: str = "CPUExecutionProvider",
    ) -> None:
        from insightface.app import FaceAnalysis

        self._conf = conf_threshold
        self._det_size = (det_size, det_size)

        providers = [onnx_provider] if onnx_provider else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        logger.info(
            "Loading InsightFace FaceAnalysis (buffalo_l: SCRFD detection only, "
            "det_size=%d, providers=%s)...",
            det_size,
            providers,
        )
        # "recognition" dropped from allowed_modules: confirmed nothing in
        # this codebase ever reads DetectedFace.embedding (grepped app/ --
        # zero hits) -- identity matching happens server-side via the
        # external /Recognize API (app/web/recognize_client.py), which gets
        # a JPEG crop, not a local embedding. Loading it anyway cost 262MiB
        # VRAM per camera for a value nothing consumed (measured: 419MiB
        # with recognition enabled vs 157MiB detection-only).
        self._app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=providers,
        )
        self._app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=conf_threshold)
        logger.info("InsightFace FaceAnalysis ready (conf=%.2f, det_size=%d)", conf_threshold, det_size)

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        try:
            faces = self._app.get(frame)
        except Exception as exc:
            logger.error("InsightFace FaceAnalysis inference error: %s", exc)
            return []

        detections: list[DetectedFace] = []
        if not faces:
            return detections

        h, w = frame.shape[:2]
        for f in faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < self._conf:
                continue
            bbox = getattr(f, "bbox", None)
            if bbox is None or len(bbox) < 4:
                continue
            x1f, y1f, x2f, y2f = bbox[:4]
            x1, y1 = max(0, int(x1f)), max(0, int(y1f))
            x2, y2 = min(w, int(x2f)), min(h, int(y2f))

            kps = getattr(f, "kps", None)
            landmarks = (
                kps.astype(np.float32)
                if kps is not None
                else np.empty((0, 2), dtype=np.float32)
            )

            embedding = getattr(f, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(f, "embedding", None)

            detections.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    landmarks=landmarks,
                    embedding=embedding,
                )
            )
        return detections

    def close(self) -> None:
        pass
