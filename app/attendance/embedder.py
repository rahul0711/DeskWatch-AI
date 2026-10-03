"""AdaFace ONNX embedding, batched. One session.run() call per batch -- see
step 6 of the build plan: accumulate crops across all active tracks before
embedding, don't embed one at a time.
"""
from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from app.attendance.alignment import align_face, normalize_for_model
from app.attendance.config import AttendanceConfig
from app.attendance.detector import DetectedFace

logger = logging.getLogger(__name__)


class AdaFaceEmbedder:
    def __init__(self, cfg: AttendanceConfig | None = None) -> None:
        cfg = cfg or AttendanceConfig()
        if not cfg.adaface_model_path.exists():
            raise FileNotFoundError(
                f"AdaFace ONNX model not found at {cfg.adaface_model_path}. "
                "Run scripts/export_adaface_onnx.py first."
            )
        self._session = ort.InferenceSession(str(cfg.adaface_model_path), providers=list(cfg.onnx_provider))
        active = self._session.get_providers()
        logger.info("AdaFace embedder active providers: %s", active)
        if "CUDAExecutionProvider" not in active and "CPUExecutionProvider" not in active:
            raise RuntimeError(
                f"AdaFace ONNX session did not load on any supported provider: {active}."
            )
        if "CUDAExecutionProvider" not in active:
            logger.warning("AdaFace ONNX session running on CPU fallback (%s).", active[0])
        self._input_name = self._session.get_inputs()[0].name

    def embed_aligned_batch(self, aligned_bgr_crops: list[np.ndarray]) -> np.ndarray:
        """aligned_bgr_crops: list of already-112x112-aligned BGR uint8
        images (i.e. already passed through align_face). Returns (N, 512)
        L2-normalized float32 embeddings."""
        if not aligned_bgr_crops:
            return np.zeros((0, 512), dtype=np.float32)
        batch = np.stack([normalize_for_model(c) for c in aligned_bgr_crops]).astype(np.float32)
        (out,) = self._session.run(None, {self._input_name: batch})
        # defensive re-normalize: the exported model already L2-normalizes
        # internally, but don't silently trust float drift for a value that
        # feeds directly into FAISS IndexFlatIP (inner product == cosine
        # similarity only if inputs are unit-norm).
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)

    def embed_from_detections(
        self, image_bgr: np.ndarray, faces: list[DetectedFace]
    ) -> np.ndarray:
        """Convenience: align + embed every detected face in one image."""
        crops = [align_face(image_bgr, f.kps) for f in faces]
        return self.embed_aligned_batch(crops)
