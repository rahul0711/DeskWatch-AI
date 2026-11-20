"""Collect pre-labelled training data from the office cameras, for

    scripts/train_phone_model.py    a phone detector that sees phones the
                                    way these ceiling cameras do
    scripts/train_posture_model.py  a sitting/standing classifier for these
                                    cameras' angles

Every --interval seconds, from each camera, a frame with at least one person
in it is saved, with the system's CURRENT guesses already filled in -- so the
manual work is correcting, not drawing from scratch:

    data/training/phones/images/<camera>_<time>.jpg
    data/training/phones/labels/<camera>_<time>.txt     YOLO format, class 0 = phone
    data/training/phones/classes.txt
    data/training/posture/to_review/sitting/<camera>_<time>_<n>.jpg
    data/training/posture/to_review/standing/<camera>_<time>_<n>.jpg

Then:
  phones   open data/training/phones in Label Studio or CVAT (both import
           images/ + labels/ + classes.txt as a YOLO dataset), fix and add
           phone boxes -- every phone in a hand, on a desk, anywhere -- and
           export back to the same folder in YOLO format. Frames with no
           phone are useful too: keep them (empty .txt) so the model learns
           what is NOT a phone.
  posture  look through to_review/sitting and to_review/standing, move each
           wrong crop to the other folder (delete unclear ones), then move
           both folders under data/training/posture/reviewed/.

Aim for 300-500 phone frames (with phones in at least half) and 500+ crops
per posture, across different people, times of day and cameras.

Usage:
    python scripts/collect_training_data.py                       # all cameras, every 20s, 2 hours
    python scripts/collect_training_data.py --camera camera-01 --interval 10 --minutes 60
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.activity.config import load_activity_config  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detection.activity_detector import (  # noqa: E402
    _CLASS_CHAIR,
    _CLASS_PERSON,
    _CLASS_PHONE,
    _classify_posture,
    _iou,
    posture_crop,
)

OUT = ROOT / "data" / "training"


class LatestFrame:
    """Keeps reading an RTSP stream in the background so a grab is always
    the newest frame, not one buffered seconds ago."""

    def __init__(self, url: str) -> None:
        self._cap = cv2.VideoCapture(url)
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = frame

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop = True
        self._cap.release()


def slug(name: str, index: int) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s or f"camera-{index}"


def yolo_line(box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", action="append", help="camera id (e.g. camera-01); repeat for several; default all")
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between saves per camera")
    ap.add_argument("--minutes", type=float, default=120.0, help="stop after this long")
    ap.add_argument("--prelabel-conf", type=float, default=0.10, help="phone guesses at or above this are pre-drawn")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_activity_config()
    device = cfg.resolved_device()
    obj = YOLO(str(cfg.obj_model))
    pose = YOLO(str(cfg.pose_model))

    phones_dir = OUT / "phones"
    (phones_dir / "images").mkdir(parents=True, exist_ok=True)
    (phones_dir / "labels").mkdir(parents=True, exist_ok=True)
    (phones_dir / "classes.txt").write_text("phone\n")
    for p in ("sitting", "standing"):
        (OUT / "posture" / "to_review" / p).mkdir(parents=True, exist_ok=True)

    cameras = []
    for i, cam in enumerate(load_config().cameras, start=1):
        cid = slug(cam.name, i)
        if args.camera and cid not in args.camera:
            continue
        cameras.append((cid, LatestFrame(cam.rtsp_url)))
        print(f"opened {cid}")
    if not cameras:
        raise SystemExit("No matching cameras (check --camera against the ids above / .env)")

    saved = crops_saved = 0
    deadline = time.monotonic() + args.minutes * 60
    try:
        while time.monotonic() < deadline:
            cycle = time.monotonic()
            for cid, stream in cameras:
                frame = stream.get()
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                res = obj.predict(frame, imgsz=cfg.img_size, conf=min(args.prelabel_conf, cfg.person_conf),
                                  classes=[_CLASS_PERSON, _CLASS_PHONE, _CLASS_CHAIR], device=device, verbose=False)[0]
                persons, phones, chairs = [], [], []
                for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                       res.boxes.cls.cpu().numpy().astype(int)):
                    box = (int(x1), int(y1), int(x2), int(y2))
                    if cls == _CLASS_PERSON and conf >= cfg.person_conf:
                        persons.append(box)
                    elif cls == _CLASS_PHONE and conf >= args.prelabel_conf:
                        phones.append(box)
                    elif cls == _CLASS_CHAIR and conf >= 0.30:
                        chairs.append(box)
                if not persons:
                    continue

                stem = f"{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cv2.imwrite(str(phones_dir / "images" / f"{stem}.jpg"), frame)
                (phones_dir / "labels" / f"{stem}.txt").write_text("\n".join(yolo_line(b, w, h) for b in phones))
                saved += 1

                # Posture crops, sorted by today's rule-based guess.
                pres = pose.predict(frame, imgsz=cfg.pose_img_size, conf=cfg.pose_conf, device=device, verbose=False)[0]
                poses = []
                if pres.keypoints is not None and pres.boxes is not None and len(pres.boxes):
                    kc = pres.keypoints.conf
                    poses = list(zip(pres.boxes.xyxy.cpu().numpy(), pres.keypoints.xy.cpu().numpy(),
                                     kc.cpu().numpy() if kc is not None else np.ones(pres.keypoints.xy.shape[:2])))
                for n, box in enumerate(persons):
                    crop = posture_crop(frame, box)
                    if crop is None:
                        continue
                    best = max(poses, key=lambda p: _iou(box, tuple(p[0])), default=None)
                    if best is not None and _iou(box, tuple(best[0])) >= 0.3:
                        guess = _classify_posture(best[1], best[2], box, cfg.sitting_knee_torso_ratio,
                                                  cfg.standing_bbox_aspect, cfg.keypoint_min_conf, chairs)
                    else:
                        guess = _classify_posture(np.zeros((0, 2), np.float32), np.zeros(0, np.float32), box,
                                                  cfg.sitting_knee_torso_ratio, cfg.standing_bbox_aspect,
                                                  cfg.keypoint_min_conf, chairs)
                    if guess not in ("sitting", "standing"):
                        continue
                    cv2.imwrite(str(OUT / "posture" / "to_review" / guess / f"{stem}_{n}.jpg"), crop)
                    crops_saved += 1
                print(f"{stem}: {len(persons)} people, {len(phones)} phone guesses "
                      f"(total {saved} frames, {crops_saved} posture crops)")
            time.sleep(max(0.0, args.interval - (time.monotonic() - cycle)))
    except KeyboardInterrupt:
        pass
    finally:
        for _, stream in cameras:
            stream.close()
    print(f"\nDone: {saved} frames in {phones_dir}, {crops_saved} posture crops in {OUT / 'posture' / 'to_review'}")


if __name__ == "__main__":
    main()
