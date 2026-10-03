"""Person/body detection -- a different signal than face detection.

A face detector needs visible facial features (eyes/nose/mouth) to find
anything; from a ceiling-mounted CCTV angle, someone looking down at a
laptop often has NO visible face at all, no matter how low the confidence
threshold goes. This module detects whole-body silhouettes instead (stock
YOLO11, trained on COCO, filtered to the "person" class), which works
regardless of head orientation -- the tradeoff for the head-count feature
specifically, not for anything identity-related.

Using the medium (m) variant, not nano -- tested live on this camera's
footage: nano found 5 bodies (incl. a likely false positive at 0.31 conf on
furniture), medium found the correct 4 with much higher confidence across
the board (0.61-0.84). ~40ms/frame vs nano's ~19ms, still fine since this
runs in its own isolated process.

Returns DetectedFace (from face_detector.py) even though these are body
boxes, not face boxes -- same bbox+confidence+landmarks shape, which lets
this flow through SimpleIouTracker and ProcessDetectionWorker completely
unchanged. Landmarks are always empty (person detection has none).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.detection.face_detector import DetectedFace

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_PERSON_MODEL = _MODEL_DIR / "yolo11m-person.pt"
_COCO_PERSON_CLASS = 0


class PersonDetector:
    """Stock YOLO11m (COCO) filtered to the "person" class, running on
    Apple M2 GPU via PyTorch MPS same as the face detector."""

    def __init__(self, conf_threshold: float = 0.4) -> None:
        import torch
        from ultralytics import YOLO

        if not _PERSON_MODEL.exists():
            raise FileNotFoundError(
                f"Person detection model not found at {_PERSON_MODEL}. "
                "It's the stock Ultralytics yolo11m.pt (COCO) -- "
                "YOLO('yolo11m.pt') auto-downloads it, then move/rename it "
                f"to {_PERSON_MODEL}."
            )

        self._conf = conf_threshold
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info("Loading person detector (YOLO11m/COCO) on device=%s ...", self._device)
        self._model = YOLO(str(_PERSON_MODEL))
        self._model.predict(
            np.zeros((320, 320, 3), dtype=np.uint8),
            device=self._device,
            conf=self._conf,
            classes=[_COCO_PERSON_CLASS],
            verbose=False,
        )
        logger.info("Person detector ready (device=%s, conf=%.2f)", self._device, conf_threshold)

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        results = self._model.predict(
            frame,
            device=self._device,
            conf=self._conf,
            classes=[_COCO_PERSON_CLASS],
            verbose=False,
            imgsz=640,
        )
        detections: list[DetectedFace] = []
        h, w = frame.shape[:2]

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # Same fix as the face detector: pull the whole tensor off the
            # device once instead of indexing box-by-box, which forces a
            # separate MPS sync per box.
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (x1f, y1f, x2f, y2f), score in zip(xyxy, confs):
                if score < self._conf:
                    continue
                x1, y1 = max(0, int(x1f)), max(0, int(y1f))
                x2, y2 = min(w, int(x2f)), min(h, int(y2f))
                detections.append(
                    DetectedFace(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(score),
                        landmarks=np.empty((0, 2), dtype=np.float32),
                    )
                )
        return detections

    def close(self) -> None:
        pass
