"""Score a phone detector on this site's own cameras.

Uses data/training/site_eval (hand-checked frames from Camera 01 and Inside,
never trained on): how many of the real phones it finds, at what confidence,
and whether it fires on the look-alikes (mouse pad, earbud cases, bottle cap,
wallet).

    python scripts/eval_phone_model.py                                  # stock yolov8x (COCO "cell phone")
    python scripts/eval_phone_model.py --model models/phone_detector.pt --imgsz 1280
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "training" / "site_eval"


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "yolov8x.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--thresholds", default="0.5,0.35,0.25,0.15")
    ap.add_argument("--device", default=None, help="cpu to score while the GPU is busy training")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    classes = [67] if len(model.names) > 1 else None        # stock COCO -> "cell phone" only
    truth = json.loads((EVAL / "truth.json").read_text())
    found, decoys = [], []
    for name, t in truth.items():
        res = model.predict(str(EVAL / "images" / f"{name}.jpg"), imgsz=args.imgsz, conf=0.02,
                            classes=classes, device=args.device, verbose=False)[0]
        dets = [(tuple(map(float, b)), float(c)) for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        for p in t["phones"]:
            best = max((c for b, c in dets if iou(b, p) >= 0.3), default=0.0)
            found.append((name, p, best))
        for n in t["not_phones"]:
            best = max((c for b, c in dets if iou(b, n) >= 0.3), default=0.0)
            decoys.append((name, n, best))

    print(f"Model: {Path(args.model).name} @ imgsz {args.imgsz}")
    print("Real phones (confidence the model gave each; 0 = missed):")
    for name, p, c in found:
        print(f"  {name} {p}: {c:.2f}")
    print("Look-alikes (should be 0):")
    for name, n, c in decoys:
        print(f"  {name} {n}: {c:.2f}")
    print("\nThreshold | phones found | look-alikes fired")
    for th in (float(x) for x in args.thresholds.split(",")):
        print(f"  {th:.2f}    | {sum(c >= th for *_, c in found)}/{len(found)}          | "
              f"{sum(c >= th for *_, c in decoys)}/{len(decoys)}")


if __name__ == "__main__":
    main()
