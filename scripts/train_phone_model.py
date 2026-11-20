"""Train a phone detector on this site's own camera footage.

Input: data/training/phones/ after review (scripts/collect_training_data.py,
then corrected in Label Studio / CVAT):

    images/<name>.jpg
    labels/<name>.txt      YOLO format, class 0 = phone (empty = no phone)

Output: models/phone_detector.pt, then in .env:

    ACTIVITY_PHONE_MODEL=models/phone_detector.pt

and restart the camera server. The stock COCO model keeps detecting people,
monitors, keyboards and chairs; only phones come from this model.

Starts from yolov8m (fast enough to run beside the existing models on an
RTX 3060) at imgsz 1280, because phones on these cameras are ~40x25px and
shrink below what a 640/960 pass can resolve.

Usage:
    python scripts/train_phone_model.py
    python scripts/train_phone_model.py --epochs 120 --base yolov8l.pt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def group_key(img: Path) -> str:
    """Images that must land on the same side of the split: neighbouring
    camera frames (same camera, same minute -- near-duplicates), and an
    Open Images photo with its own CCTV-style copy."""
    stem = img.stem.replace("_cctv", "")
    if stem.count("_") >= 2:                 # <camera>_<YYYYmmdd>_<HHMMSS>: site frame
        return img.parent.parent.name + ":" + stem[:-2]
    return img.parent.parent.name + ":" + stem


def split(datas: list[Path], val_share: float, seed: int) -> Path:
    """<first data>/_split/{train,val}/{images,labels} as symlinks, + dataset yaml."""
    images = []
    for data in datas:
        images += sorted(p for p in (data / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(images) < 50:
        raise SystemExit(f"Only {len(images)} images -- collect at least 300 first.")
    blocks: dict[str, list[Path]] = {}
    for img in images:
        blocks.setdefault(group_key(img), []).append(img)
    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_share))
    val_keys = set(keys[:n_val])

    out = datas[0] / "_split"
    shutil.rmtree(out, ignore_errors=True)
    with_phone = 0
    for key, imgs in blocks.items():
        part = "val" if key in val_keys else "train"
        for img in imgs:
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            for sub in ("images", "labels"):
                (out / part / sub).mkdir(parents=True, exist_ok=True)
            name = f"{img.parent.parent.name}__{img.name}"      # unique across folders
            (out / part / "images" / name).symlink_to(img.resolve())
            if label.exists():
                (out / part / "labels" / f"{Path(name).stem}.txt").symlink_to(label.resolve())
                with_phone += bool(label.read_text().strip())
    print(f"{len(images)} images ({with_phone} with phones), {n_val}/{len(keys)} time blocks held out for validation")
    if with_phone < 100:
        print("WARNING: fewer than 100 images contain a phone -- accuracy will be limited.")

    yaml = out / "phones.yaml"
    yaml.write_text(f"path: {out}\ntrain: train/images\nval: val/images\nnames:\n  0: phone\n")
    return yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "training" / "phones")],
                    help="one or more folders with images/ + labels/ (YOLO, class 0 = phone)")
    ap.add_argument("--base", default="yolov8m.pt", help="starting checkpoint (downloaded if missing)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4, help="4 fits an RTX 3060 (12GB) at 1280")
    ap.add_argument("--val-share", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "models" / "phone_detector.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml = split([Path(d) for d in args.data], args.val_share, seed=0)
    model = YOLO(args.base)
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(ROOT / "runs" / "phone"), name="train", exist_ok=True,
        patience=20,
        # Phones are tilted every which way in a hand; no vertical flips (a
        # ceiling camera never sees the room upside down).
        degrees=15, fliplr=0.5, flipud=0.0, scale=0.3, mosaic=1.0,
    )
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(yaml), imgsz=args.imgsz, batch=args.batch, verbose=False)
    shutil.copy(best, args.out)
    print(f"\nPrecision {metrics.box.mp:.2f}  Recall {metrics.box.mr:.2f}  mAP50 {metrics.box.map50:.2f}")
    print(f"Saved {args.out}\nAdd to .env:  ACTIVITY_PHONE_MODEL={Path(args.out).relative_to(ROOT)}")
    print("Recall = share of real phones found; precision = share of detections that really are phones.")


if __name__ == "__main__":
    main()
