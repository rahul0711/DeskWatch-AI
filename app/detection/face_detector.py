"""Face detection with pluggable backends.

Backends available, selected via DETECTOR_BACKEND in .env:

  yolo  (default)
      YOLOv11n-face via Ultralytics + PyTorch MPS. Fastest (~15ms/frame
      isolated), but has the weakest recall of the three on angled/side/
      occluded faces -- measured live on this camera's footage: plateaus
      at 1-3 out of 4 real faces in a top-down office scene, no clean
      threshold reaches all 4.
      Model: models/yolo11n-face.pt (5 MB, already downloaded).

  insightface / scrfd
      SCRFD via InsightFace + ONNX Runtime, detection-only (recognition/
      landmarks/age-gender are loaded by the underlying FaceAnalysis but
      never called -- this pipeline only needs boxes). ~94ms/frame on CPU,
      noticeably slower than YOLO, but since detection already runs in its
      own isolated OS process that's invisible to the render loop -- it
      just means detections update somewhat less often, not that video
      stutters. Measured live on the same footage: cleanly finds all 4
      real faces at a stable, non-aggressive threshold (0.3, unchanged
      down to 0.15) -- the only one of the three backends that actually
      solved the side-face undercounting problem in testing.
      Also needed for Phase 3 ArcFace recognition (same library).
      Auto-downloads the buffalo_l model pack (~330MB) on first run.

  mtcnn
      facenet-pytorch MTCNN. Present but not benchmarked against this
      camera's footage -- untested, use at your own judgment.

All backends expose the same FaceDetector.detect() → list[DetectedFace]
interface. Nothing above this module changes when switching backends.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_YOLO_MODEL = _MODEL_DIR / "yolo11n-face.pt"


# ---------------------------------------------------------------------------
# Shared data model (pipeline / tracker / display use this)
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 — pixel coords
    confidence: float
    landmarks: np.ndarray            # 5×2 keypoints (or empty if unavailable)
    embedding: np.ndarray | None = None  # ArcFace 512D embedding if computed


# ---------------------------------------------------------------------------
# YOLO backend (YOLOv11n-face + Ultralytics + PyTorch MPS)
# ---------------------------------------------------------------------------

class _YoloDetector:
    """YOLOv11n face detector running on Apple M2 GPU via PyTorch MPS.

    Uses the community-trained yolo11n-face.pt model (WIDERFace dataset).
    With MPS the M2's GPU runs inference — expect 40-60 FPS on 1080p input.
    Falls back to CPU automatically if MPS is unavailable.
    """

    def __init__(self, conf_threshold: float = 0.5) -> None:
        import torch
        from ultralytics import YOLO

        if not _YOLO_MODEL.exists():
            raise FileNotFoundError(
                f"YOLOv11 face model not found at {_YOLO_MODEL}. "
                "Download from: https://github.com/akanametov/yolo-face/releases"
            )

        self._conf = conf_threshold

        # Choose best available device: MPS (M2 GPU) > CUDA (NVIDIA GPU) > CPU
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info("Loading YOLOv11n-face on device=%s ...", self._device)
        self._model = YOLO(str(_YOLO_MODEL))
        # Warm up on a blank frame so first real frame isn't slow
        import numpy as _np
        self._model.predict(
            _np.zeros((320, 320, 3), dtype=_np.uint8),
            device=self._device,
            conf=self._conf,
            verbose=False,
        )
        logger.info(
            "YOLOv11n-face ready (device=%s, conf=%.2f)",
            self._device,
            conf_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        results = self._model.predict(
            frame,
            device=self._device,
            conf=self._conf,
            verbose=False,
            imgsz=640,
        )
        detections: list[DetectedFace] = []
        h, w = frame.shape[:2]

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # Pull the whole tensor off the device once. Indexing individual
            # boxes (box.conf[0], box.xyxy[0]) instead forces a separate
            # MPS device sync per box -- with N faces that's ~2N Metal
            # command-buffer flushes per frame, which is where most of the
            # frame-to-frame latency variance (worse with more people in
            # frame) was coming from.
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for (x1f, y1f, x2f, y2f), score in zip(xyxy, confs):
                if score < self._conf:
                    continue
                x1, y1 = max(0, int(x1f)), max(0, int(y1f))
                x2, y2 = min(w, int(x2f)), min(h, int(y2f))
                # YOLO face model does not output landmarks in this variant
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


# ---------------------------------------------------------------------------
# SCRFD backend (InsightFace + ONNX Runtime)
# ---------------------------------------------------------------------------

class _ScrfdDetector:
    """SCRFD face detector via InsightFace, running detection only.

    FaceAnalysis.get() would also run recognition/landmarks/age-gender (4
    extra ONNX models) on every call -- ~5x slower for nothing this
    pipeline uses. Loading via FaceAnalysis (which auto-downloads the
    buffalo_l pack if missing) but then keeping only .det_model and calling
    its .detect() directly gets the same recall at a fraction of the cost:
    487ms/frame -> 94ms/frame in testing, same face count either way.
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        det_size: int = 640,
        onnx_provider: str = "CPUExecutionProvider",
    ) -> None:
        from insightface.app import FaceAnalysis

        self._conf = conf_threshold
        size = (det_size, det_size)

        providers = [onnx_provider] if onnx_provider else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        logger.info(
            "Loading SCRFD (InsightFace buffalo_l, det_size=%d, providers=%s) ...",
            det_size, providers,
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=size)

        self._detector = app.det_model
        # app.prepare() sets its own default det_thresh -- override with ours.
        self._detector.prepare(ctx_id=0, input_size=size, det_thresh=conf_threshold)

        # Read back input_size from the constructed SCRFD object itself
        # (insightface stores it there after .prepare()) rather than just
        # echoing the config value passed in -- proves det_size actually
        # took effect on the live detector, not just that we asked for it.
        logger.info(
            "SCRFD ready (conf=%.2f, requested det_size=%d, detector.input_size=%s)",
            conf_threshold, det_size, self._detector.input_size,
        )

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        bboxes, kpss = self._detector.detect(frame)
        detections: list[DetectedFace] = []
        if bboxes is None or len(bboxes) == 0:
            return detections

        h, w = frame.shape[:2]
        for i, box in enumerate(bboxes):
            x1f, y1f, x2f, y2f, score = box
            if score < self._conf:
                continue
            x1, y1 = max(0, int(x1f)), max(0, int(y1f))
            x2, y2 = min(w, int(x2f)), min(h, int(y2f))
            landmarks = (
                kpss[i].astype(np.float32)
                if kpss is not None
                else np.empty((0, 2), dtype=np.float32)
            )
            detections.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    landmarks=landmarks,
                )
            )
        return detections

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# MTCNN backend (facenet-pytorch MTCNN)
# ---------------------------------------------------------------------------

class _MtcnnDetector:
    """MTCNN face detector from facenet-pytorch.

    Runs face detection using PyTorch, utilizing Apple Silicon MPS (or CUDA) if available.
    """

    def __init__(self, conf_threshold: float = 0.5) -> None:
        import torch
        from facenet_pytorch import MTCNN

        self._conf = conf_threshold

        # Choose best available device: MPS (Apple GPU) > CUDA (NVIDIA GPU) > CPU
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info("Loading facenet-pytorch MTCNN on device=%s ...", self._device)
        self._model = MTCNN(keep_all=True, device=self._device)
        logger.info(
            "facenet-pytorch MTCNN ready (device=%s, conf=%.2f)",
            self._device,
            conf_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        # MTCNN expects RGB images; OpenCV streams BGR by default
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, probs = self._model.detect(frame_rgb)

        detections: list[DetectedFace] = []
        if boxes is None or probs is None:
            return detections

        h, w = frame.shape[:2]
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self._conf:
                continue
            x1f, y1f, x2f, y2f = box
            x1, y1 = max(0, int(x1f)), max(0, int(y1f))
            x2, y2 = min(w, int(x2f)), min(h, int(y2f))

            detections.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks=np.empty((0, 2), dtype=np.float32),
                )
            )
        return detections

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public FaceDetector — uses MTCNN backend
# ---------------------------------------------------------------------------

class FaceDetector:
    """Public detector. Pipeline calls detector.detect(frame) only.

    Backend selected via `backend` -- "yolo" (default), "insightface" /
    "scrfd", or "mtcnn". See the module docstring for tradeoffs.
    """

    # Read by process_detector.py: track faces with ByteTrack (motion model
    # + two-stage association) instead of the default SimpleIouTracker, so
    # a brief occlusion/turn doesn't reset the track_id (and with it, any
    # recognition state already attached to that id).
    TRACKER_BACKEND = "bytetrack"

    def __init__(
        self,
        backend: str = "yolo",
        onnx_provider: str = "CPUExecutionProvider",
        det_size: int = 640,
        conf_threshold: float = 0.5,
    ) -> None:
        backend = (backend or "yolo").lower().strip()
        if backend in ("insightface", "scrfd"):
            self._backend = _ScrfdDetector(
                conf_threshold=conf_threshold, det_size=det_size, onnx_provider=onnx_provider
            )
        elif backend == "mtcnn":
            self._backend = _MtcnnDetector(conf_threshold=conf_threshold)
        else:
            if backend != "yolo":
                logger.warning("Unknown DETECTOR_BACKEND=%r, falling back to yolo", backend)
                backend = "yolo"
            self._backend = _YoloDetector(conf_threshold=conf_threshold)
        logger.info("FaceDetector initialized with %s backend", backend)

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        return self._backend.detect(frame)

    def close(self) -> None:
        self._backend.close()


# ---------------------------------------------------------------------------
# Utility — used by pipeline for face crop saving
# ---------------------------------------------------------------------------

def crop_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = 0.45,
    min_size: int = 200,
) -> np.ndarray:
    """Crop a face region with configurable margin and optional minimum output size.

    Provides sufficient head and facial context so face recognition backends
    (like ScriptIndia /Recognize) reliably locate facial keypoints without 400 errors.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    mx, my = int(bw * margin), int(bh * margin)
    fh, fw = frame.shape[:2]
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(fw, x2 + mx)
    cy2 = min(fh, y2 + my)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return crop

    # Ensure minimum dimension for clarity when sending to external recognizer
    ch, cw = crop.shape[:2]
    if min_size > 0 and (cw < min_size or ch < min_size):
        scale = max(min_size / max(1, cw), min_size / max(1, ch))
        new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop


def align_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: np.ndarray,
    margin: float = 0.45,
    min_size: int = 200,
) -> np.ndarray:
    """Like crop_face, but first de-rotates the region so the eye-line is
    horizontal, removing head tilt/roll before cropping. Uses the 5-point
    landmarks (InsightFace order: [left_eye, right_eye, nose, ...]) that
    only the "insightface"/"scrfd" DETECTOR_BACKEND produces -- falls back
    to a plain crop_face otherwise, or when the tilt is negligible.

    This corrects roll; frontalness (yaw) is a separate check done on the
    landmarks before this is ever called -- see _frontalness_score in
    app/web/server.py.
    """
    if landmarks is None or len(landmarks) < 2:
        return crop_face(frame, bbox, margin=margin, min_size=min_size)

    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Not worth a resample for a near-level face.
    if abs(angle) < 3.0:
        return crop_face(frame, bbox, margin=margin, min_size=min_size)

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    # Rotate a padded region around the face, not the whole frame -- pad
    # generously enough that the rotated, margined face can't clip the
    # region edge, but stay local for speed.
    fh, fw = frame.shape[:2]
    pad = int(max(bw, bh) * (1.0 + margin) * 1.5)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx1, ry1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    rx2, ry2 = min(fw, int(cx + pad)), min(fh, int(cy + pad))
    region = frame[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return crop_face(frame, bbox, margin=margin, min_size=min_size)

    region_cx, region_cy = cx - rx1, cy - ry1
    rot_mat = cv2.getRotationMatrix2D((region_cx, region_cy), angle, 1.0)
    rotated = cv2.warpAffine(
        region, rot_mat, (region.shape[1], region.shape[0]), flags=cv2.INTER_LINEAR
    )

    # Carry the bbox corners through the same rotation to find the
    # now-upright face box within the rotated region.
    corners = np.array(
        [
            [x1 - rx1, y1 - ry1], [x2 - rx1, y1 - ry1],
            [x2 - rx1, y2 - ry1], [x1 - rx1, y2 - ry1],
        ],
        dtype=np.float32,
    )
    ones = np.ones((4, 1), dtype=np.float32)
    rotated_corners = (rot_mat @ np.hstack([corners, ones]).T).T
    nx1, ny1 = float(rotated_corners[:, 0].min()), float(rotated_corners[:, 1].min())
    nx2, ny2 = float(rotated_corners[:, 0].max()), float(rotated_corners[:, 1].max())

    return crop_face(
        rotated,
        (max(0, int(nx1)), max(0, int(ny1)), int(nx2), int(ny2)),
        margin=margin,
        min_size=min_size,
    )
