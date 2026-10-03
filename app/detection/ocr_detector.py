"""Scene-text OCR: find and read text anywhere in the frame -- walls, signs,
boards (static) as well as papers/cards/phones/clothes carried by walking
people (moving) -- and vote on a temporally-consistent reading per region.

Two detection passes feed one shared text tracker/voter per frame:

  1. STATIC / full-frame pass -- PaddleOCR's PP-OCR text DETECTION model
     (a DBNet-family detector, same architecture family as DBNet++) runs
     over the whole 1920x1080 frame at native resolution (see
     `limit_side_len`/`limit_type` below), so text anywhere in frame gets
     a candidate box regardless of what -- if anything -- is carrying it.

  2. MOVING / carrier pass -- a stock YOLO11m (COCO) tracks person/book/
     cell-phone/laptop boxes frame-to-frame via BoT-SORT (Ultralytics'
     built-in `.track()`), reusing the exact same checkpoint and tracker
     config as ActivityDetector (models/yolo11m-person.pt +
     models/activity_botsort.yaml) so nothing new has to be downloaded.
     Each tracked box is padded, upscaled, and re-run through the SAME text
     detector at effectively higher resolution than the full-frame pass
     alone would give it -- a card or phone screen in someone's hand is
     often too small to resolve at 1920x1080 but reads fine once that
     region alone fills the detector's input.

Every text region found by either pass -- regardless of source -- then
goes through the same per-crop pipeline: perspective/rotation correction
(cv2.warpPerspective off the detector's own quadrilateral, not just an
axis-aligned bbox crop), upscaling if the corrected crop is short, light
OpenCV contrast enhancement, and finally PP-OCR text RECOGNITION for the
actual string + confidence.

Per-frame readings are deduplicated (the carrier pass can rediscover
something the static pass already found) and handed to TextTracker
(app/tracking/text_tracker.py), which IoU-matches regions across frames and
votes on the most consistent reading -- this is what gives clean output for
moving text instead of a different misread every frame.

Why PaddleOCR instead of MMOCR: MMOCR's detection backbones (including
DBNet++) depend on `mmcv`, which has no working build for macOS arm64 (no
CUDA, no prebuilt wheel for this platform) -- see README.md. PaddleOCR's
PP-OCR detector is also a DBNet-family model, ships official CPU/arm64
wheels, and is actively maintained, so this gets the same detector
architecture without an unbuildable dependency.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from app.detection.face_detector import DetectedFace
from app.tracking.simple_tracker import TrackedFace
from app.tracking.text_tracker import TextReading, TextTracker

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_CARRIER_MODEL = _MODEL_DIR / "yolo11m-person.pt"
_CARRIER_TRACKER_CFG = _MODEL_DIR / "activity_botsort.yaml"

# COCO class ids of things that plausibly carry text and can move: the
# person themself (a badge/t-shirt/held item usually reads as part of their
# box), a book/paper-like object, a phone screen, a laptop screen.
_CLASS_PERSON = 0
_CLASS_LAPTOP = 63
_CLASS_PHONE = 67
_CLASS_BOOK = 73
_CARRIER_CLASSES = [_CLASS_PERSON, _CLASS_LAPTOP, _CLASS_PHONE, _CLASS_BOOK]

_SAME_FRAME_DEDUP_IOU = 0.5


def _quad_to_bbox(quad: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    xs, ys = quad[:, 0], quad[:, 1]
    x1, y1 = max(0, int(xs.min())), max(0, int(ys.min()))
    x2, y2 = min(w, int(xs.max())), min(h, int(ys.max()))
    return x1, y1, x2, y2


def _warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Perspective-correct a (possibly rotated/skewed) text quadrilateral
    into a straight horizontal crop, instead of just bbox-cropping and
    living with the tilt. Falls back to a 90-degree rotation when the
    unwarped box comes out taller than it is wide (sideways text)."""
    pts = quad.astype(np.float32)
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width < 2 or height < 2:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    if height > width * 1.3:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _enhance_contrast(crop: np.ndarray) -> np.ndarray:
    """Mild CLAHE contrast boost on the L channel -- helps low-contrast text
    (dim CCTV lighting, glare, faded signage) without touching color, which
    the recognizer also uses as a signal."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def _upscale_if_small(crop: np.ndarray, target_height: int, max_scale: float) -> np.ndarray | None:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    if h >= target_height:
        return crop
    scale = min(max_scale, target_height / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class OCRDetector:
    """PP-OCR (DBNet-family detector + CRNN/SVTR-family recognizer) scene
    text, plus a YOLO11m+BoT-SORT carrier pass for moving text. See module
    docstring for the full pipeline."""

    PROVIDES_TRACK_IDS = True

    def __init__(
        self,
        det_limit_side_len: int = 1920,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        # Mobile models by default -- ~5-7x faster than PaddleOCR's own
        # "medium" default on CPU-only hardware, for a small accuracy cost.
        # See app/config.py's OcrConfig for the measurements this is based on.
        det_model_name: str | None = "PP-OCRv5_mobile_det",
        rec_model_name: str | None = "PP-OCRv5_mobile_rec",
        rec_min_confidence: float = 0.35,
        min_crop_height_px: int = 10,
        target_rec_height: int = 48,
        max_upscale: float = 6.0,
        max_regions_per_frame: int = 40,
        carrier_conf_threshold: float = 0.35,
        max_carriers_per_frame: int = 6,
        carrier_crop_max_side: int = 800,
        carrier_upscale: float = 2.0,
        enable_carrier_pass: bool = True,
        enable_mkldnn: bool | None = None,
    ) -> None:
        from paddleocr import TextDetection, TextRecognition

        # PaddlePaddle's oneDNN/MKL-DNN CPU path is broken for these models on
        # Windows: every predict() raises
        #   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
        #   not support [pir::ArrayAttribute<pir::DoubleAttribute>]
        # from onednn_instruction.cc. Because _detect_regions/_recognize_quad
        # catch-and-log, that failure is silent at the pipeline level -- OCR
        # mode just reports zero text on every frame forever. Turning oneDNN
        # off routes around the broken kernel and costs ~15% throughput here,
        # so default it off on Windows only and leave macOS/Linux (where the
        # README's timings were measured) untouched.
        if enable_mkldnn is None:
            enable_mkldnn = sys.platform != "win32"

        self._rec_min_confidence = rec_min_confidence
        self._min_crop_height_px = min_crop_height_px
        self._target_rec_height = target_rec_height
        self._max_upscale = max_upscale
        self._max_regions_per_frame = max_regions_per_frame
        self._carrier_conf = carrier_conf_threshold
        self._max_carriers = max_carriers_per_frame
        self._carrier_crop_max_side = carrier_crop_max_side
        self._carrier_upscale = carrier_upscale
        self._enable_carrier_pass = enable_carrier_pass

        logger.info("Loading PaddleOCR text detector (limit_side_len=%d, model=%s, mkldnn=%s) ...",
                    det_limit_side_len, det_model_name or "default", enable_mkldnn)
        det_kwargs = dict(
            limit_side_len=det_limit_side_len,
            limit_type="max",
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
            enable_mkldnn=enable_mkldnn,
        )
        if det_model_name:
            det_kwargs["model_name"] = det_model_name
        self._det = TextDetection(**det_kwargs)

        logger.info("Loading PaddleOCR text recognizer (model=%s) ...", rec_model_name or "default")
        rec_kwargs = dict(enable_mkldnn=enable_mkldnn)
        if rec_model_name:
            rec_kwargs["model_name"] = rec_model_name
        self._rec = TextRecognition(**rec_kwargs)

        self._carrier_model = None
        if self._enable_carrier_pass:
            if not _CARRIER_MODEL.exists():
                logger.warning(
                    "OCR carrier model not found at %s -- moving-text tracking via "
                    "YOLO+BoT-SORT disabled, static full-frame OCR still runs.",
                    _CARRIER_MODEL,
                )
            else:
                from ultralytics import YOLO
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                logger.info("Loading OCR carrier tracker (YOLO11m/COCO, device=%s) ...", device)
                self._carrier_model = YOLO(str(_CARRIER_MODEL))
                self._carrier_device = device
                blank = np.zeros((320, 320, 3), dtype=np.uint8)
                self._carrier_model.track(
                    blank, persist=True, tracker=str(_CARRIER_TRACKER_CFG),
                    classes=_CARRIER_CLASSES, conf=self._carrier_conf,
                    device=self._carrier_device, verbose=False,
                )

        self._selftest()

        self._text_tracker = TextTracker()
        logger.info("OCR detector ready.")

    def _selftest(self) -> None:
        """Run one throwaway predict per model at construction time.

        Both `_detect_regions` and `_recognize_quad` catch-and-log so that a
        single bad crop can't take down a live stream -- which also means a
        wholesale backend failure (see the oneDNN note above) degrades to
        "found no text" on every frame with nothing but a log line to say
        why. Probing once here turns that into a single loud, actionable
        error at startup instead."""
        probe = np.full((64, 256, 3), 255, dtype=np.uint8)
        cv2.putText(probe, "TEXT", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        try:
            list(self._det.predict(probe, batch_size=1))
            list(self._rec.predict(input=probe, batch_size=1))
        except Exception:
            logger.error(
                "OCR self-test FAILED -- the PaddleOCR backend cannot run on this "
                "machine, so OCR mode would silently report zero text on every "
                "frame. If this is the oneDNN/PIR error on Windows, set "
                "OCR_ENABLE_MKLDNN=false in .env.",
                exc_info=True,
            )
            raise

    # -- per-crop pipeline ----------------------------------------------

    def _recognize_quad(self, image: np.ndarray, quad: np.ndarray) -> tuple[str, float] | None:
        crop = _warp_quad(image, quad)
        if crop is None or crop.shape[0] < self._min_crop_height_px:
            return None
        crop = _upscale_if_small(crop, self._target_rec_height, self._max_upscale)
        if crop is None:
            return None
        crop = _enhance_contrast(crop)
        try:
            results = list(self._rec.predict(input=crop, batch_size=1))
        except Exception:
            logger.exception("OCR recognition failed on a crop")
            return None
        if not results:
            return None
        res = results[0]
        text = (res.get("rec_text", "") or "").strip()
        score = float(res.get("rec_score", 0.0) or 0.0)
        if not text or score < self._rec_min_confidence:
            return None
        return text, score

    def _detect_regions(self, image: np.ndarray) -> list[np.ndarray]:
        """Run the DB-family text detector, return a list of 4x2 quads in
        `image`'s own pixel coordinates."""
        try:
            results = list(self._det.predict(image, batch_size=1))
        except Exception:
            logger.exception("OCR text detection failed")
            return []
        if not results:
            return []
        polys = results[0].get("dt_polys")
        if polys is None:
            return []
        return [np.asarray(p, dtype=np.float32) for p in polys]

    # -- passes -----------------------------------------------------------

    def _static_pass(self, frame: np.ndarray) -> list[TextReading]:
        h, w = frame.shape[:2]
        readings: list[TextReading] = []
        for quad in self._detect_regions(frame):
            rec = self._recognize_quad(frame, quad)
            if rec is None:
                continue
            text, score = rec
            bbox = _quad_to_bbox(quad, w, h)
            readings.append(TextReading(bbox=bbox, text=text, confidence=score, quad=quad.tolist(), moving=False))
        return readings

    def _carrier_pass(self, frame: np.ndarray) -> list[TextReading]:
        if self._carrier_model is None:
            return []
        h, w = frame.shape[:2]
        results = self._carrier_model.track(
            frame, persist=True, tracker=str(_CARRIER_TRACKER_CFG),
            classes=_CARRIER_CLASSES, conf=self._carrier_conf,
            device=self._carrier_device, verbose=False, imgsz=960,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        order = np.argsort(-confs)[: self._max_carriers]

        readings: list[TextReading] = []
        for i in order:
            x1f, y1f, x2f, y2f = xyxy[i]
            x1, y1 = max(0, int(x1f)), max(0, int(y1f))
            x2, y2 = min(w, int(x2f)), min(h, int(y2f))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            carrier_crop = frame[y1:y2, x1:x2]

            # Bound the crop's size before upscaling so a person filling
            # most of the frame doesn't turn into a multi-megapixel detect
            # call -- cap the longer side, then apply the upscale on top.
            ch, cw = carrier_crop.shape[:2]
            shrink = min(1.0, self._carrier_crop_max_side / max(ch, cw))
            scale = shrink * self._carrier_upscale
            if scale != 1.0:
                carrier_crop = cv2.resize(
                    carrier_crop, (max(1, round(cw * scale)), max(1, round(ch * scale))),
                    interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
                )
            if carrier_crop.size == 0:
                continue

            for quad in self._detect_regions(carrier_crop):
                rec = self._recognize_quad(carrier_crop, quad)
                if rec is None:
                    continue
                text, score = rec
                # Map the crop-local quad back to full-frame coordinates.
                full_quad = quad / scale
                full_quad[:, 0] += x1
                full_quad[:, 1] += y1
                bbox = _quad_to_bbox(full_quad, w, h)
                readings.append(
                    TextReading(bbox=bbox, text=text, confidence=score, quad=full_quad.tolist(), moving=True)
                )
        return readings

    @staticmethod
    def _dedup_same_frame(readings: list[TextReading]) -> list[TextReading]:
        """The carrier pass can rediscover a region the static pass already
        found (e.g. a wall sign behind someone's shoulder). Greedy NMS by
        confidence keeps one reading per overlapping region per frame --
        cross-frame identity is TextTracker's job, not this."""
        readings = sorted(readings, key=lambda r: r.confidence, reverse=True)
        kept: list[TextReading] = []
        for r in readings:
            overlaps = False
            for k in kept:
                ax1, ay1, ax2, ay2 = r.bbox
                bx1, by1, bx2, by2 = k.bbox
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
                area_b = max(1, (bx2 - bx1) * (by2 - by1))
                if inter / float(area_a + area_b - inter) >= _SAME_FRAME_DEDUP_IOU:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(r)
        return kept

    # -- public API ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[TrackedFace]:
        readings = self._static_pass(frame)
        if self._enable_carrier_pass:
            try:
                readings += self._carrier_pass(frame)
            except Exception:
                logger.exception("OCR carrier pass failed; continuing with static-only readings")

        readings = self._dedup_same_frame(readings)
        if len(readings) > self._max_regions_per_frame:
            readings = sorted(readings, key=lambda r: r.confidence, reverse=True)[: self._max_regions_per_frame]

        tracks = self._text_tracker.update(readings)

        out: list[TrackedFace] = []
        for track in tracks:
            text, conf, votes = track.voted()
            out.append(
                TrackedFace(
                    track_id=track.track_id,
                    face=DetectedFace(bbox=track.bbox, confidence=conf, landmarks=np.empty((0, 2), dtype=np.float32)),
                    text=text,
                    text_confidence=conf,
                    text_votes=votes,
                    text_quad=track.quad,
                    text_moving=track.ever_moving,
                )
            )
        return out

    def close(self) -> None:
        pass
