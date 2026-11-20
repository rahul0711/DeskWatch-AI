"""RF-DETR phone detection on per-person crops (ONNX, no PyTorch needed).

    person box -> upper body + hands, cropped at NATIVE resolution -> RF-DETR

Measured on this site's own hand-checked frames (data/training/site_eval,
scripts in ~/rfdetr-test): RF-DETR Medium on these crops found 7 of 9 real
phones at confidence >= 0.25 with none of the 6 look-alikes (router, earbud
cases, bottle cap, mouse pad), against 1-2 of 9 for the stock YOLOv8x on the
full frame and 3 of 9 for YOLOv8x on the same crops.

RF-DETR (Roboflow) is Apache-2.0. The model is the stock COCO checkpoint
exported to ONNX (models/rfdetr-medium.onnx), run with onnxruntime -- already
used here for the face models -- so the live system needs no new packages.

Pre/post-processing mirror rfdetr 1.11 RFDETR.predict / PostProcess:
RGB, /255, ImageNet mean/std, resize to the model's square input; sigmoid
class scores; boxes are normalised (cx, cy, w, h) scaled to the crop.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_CELL_PHONE = 77          # RF-DETR's 91-way COCO index for "cell phone"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def upper_body_crop(person_bbox, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """The part of a person where a phone in use is: upper ~80% of the body,
    widened 15% each side for outstretched arms."""
    x1, y1, x2, y2 = person_bbox
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - 0.15 * bw)), max(0, int(y1 - 0.05 * bh)),
            min(img_w, int(x2 + 0.15 * bw)), min(img_h, int(y1 + 0.80 * bh)))


class RFDETRPhoneDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.25, provider: str = "CUDAExecutionProvider") -> None:
        import onnxruntime as ort

        providers = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = int(inp.shape[2])
        self._conf = conf
        logger.info("RF-DETR phone detector ready (%s, %dpx, conf %.2f, %s)",
                    Path(model_path).name, self._size, conf, self._session.get_providers()[0])

    def _detect_one(self, crop_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        h, w = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self._size, self._size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        dets, logits = self._session.run(None, {self._input: np.ascontiguousarray(x)})
        scores = 1.0 / (1.0 + np.exp(-logits[0, :, COCO_CELL_PHONE]))
        out = []
        for q in np.nonzero(scores >= self._conf)[0]:
            cx, cy, bw, bh = dets[0, q]
            out.append((max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h),
                        min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h), float(scores[q])))
        return out

    def detect_in_people(self, frame: np.ndarray, person_boxes: list) -> list[tuple[tuple[int, int, int, int], float]]:
        """Phones found in each person's native-resolution upper-body crop,
        as (bbox in frame coordinates, confidence). Overlapping finds from
        two neighbours' crops are merged (best kept)."""
        fh, fw = frame.shape[:2]
        found: list[tuple[tuple[int, int, int, int], float]] = []
        for box in person_boxes:
            x1, y1, x2, y2 = upper_body_crop(box, fw, fh)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            for bx1, by1, bx2, by2, sc in self._detect_one(frame[y1:y2, x1:x2]):
                found.append(((int(bx1 + x1), int(by1 + y1), int(bx2 + x1), int(by2 + y1)), sc))
        found.sort(key=lambda f: -f[1])
        kept: list[tuple[tuple[int, int, int, int], float]] = []
        for b, sc in found:
            if all(_iou(b, k) < 0.5 for k, _ in kept):
                kept.append((b, sc))
        return kept


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
