"""Train a sitting/standing classifier on this site's own camera footage.

Input: data/training/posture/reviewed/ after review
(scripts/collect_training_data.py, then wrong crops moved between folders):

    sitting/*.jpg
    standing/*.jpg

Output: models/posture_classifier.pt, then in .env:

    ACTIVITY_POSTURE_MODEL=models/posture_classifier.pt

and restart the camera server. The classifier does not replace the rules:
it adds a vote worth 2 x its confidence to them (knee/torso ratio, box shape,
chair), so when it is sure it decides, and when it is unsure the rules do.
The stand-up guard and seat memory still apply on top.

Usage:
    python scripts/train_posture_model.py
    python scripts/train_posture_model.py --epochs 60 --base yolov8m-cls.pt
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("sitting", "standing")


def split(data: Path, val_share: float, seed: int) -> Path:
    out = data.parent / "_split"
    shutil.rmtree(out, ignore_errors=True)
    for cls in CLASSES:
        imgs = sorted((data / cls).glob("*.jpg"))
        if len(imgs) < 100:
            raise SystemExit(f"Only {len(imgs)} images in {data / cls} -- aim for 500+ per class.")
        # Hold out whole time blocks: crops of the same person seconds apart
        # are near-duplicates (stem = <camera>_<YYYYmmdd_HHMMSS>_<n>).
        blocks: dict[str, list[Path]] = {}
        for img in imgs:
            blocks.setdefault(img.stem.rsplit("_", 1)[0][:-2], []).append(img)
        keys = sorted(blocks)
        random.Random(seed).shuffle(keys)
        val_keys = set(keys[: max(1, int(len(keys) * val_share))])
        for key, group in blocks.items():
            part = "val" if key in val_keys else "train"
            (out / part / cls).mkdir(parents=True, exist_ok=True)
            for img in group:
                (out / part / cls / img.name).symlink_to(img.resolve())
        print(f"{cls}: {len(imgs)} crops")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "training" / "posture" / "reviewed"))
    ap.add_argument("--base", default="yolov8s-cls.pt", help="starting checkpoint (downloaded if missing)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-share", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "models" / "posture_classifier.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    split_dir = split(Path(args.data), args.val_share, seed=0)
    model = YOLO(args.base)
    model.train(
        data=str(split_dir), epochs=args.epochs, imgsz=224, batch=64,
        project=str(ROOT / "runs" / "posture"), name="train", exist_ok=True, patience=10,
        fliplr=0.5, flipud=0.0,
    )
    best = Path(model.trainer.best)
    metrics = YOLO(str(best)).val(data=str(split_dir), imgsz=224, verbose=False)
    shutil.copy(best, args.out)
    print(f"\nValidation accuracy {metrics.top1:.3f}")
    print(f"Saved {args.out}\nAdd to .env:  ACTIVITY_POSTURE_MODEL={Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
