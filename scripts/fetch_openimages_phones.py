"""Download a phone-detection training set from Open Images V7.

Open Images (Google) has hand-drawn boxes for "Mobile phone" on real photos:
phones in hands, at ears, on desks, partly covered. Annotations are CC BY 4.0
and images are listed as CC BY 2.0 -- commercial use is allowed with
attribution (see https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

Writes, in the same layout scripts/train_phone_model.py reads:

    data/training/openimages/images/<id>.jpg      resized to max 1280px
    data/training/openimages/labels/<id>.txt      YOLO, class 0 = phone

Positives: images with at least one phone box (drawings/depictions skipped).
Hard negatives: images with bottles, remote controls, computer mice or
calculators and NO phone, so the model learns what is not a phone -- the
mistakes seen on this site's cameras.

Usage:
    python scripts/fetch_openimages_phones.py                     # 5000 phone + 1500 negative images
    python scripts/fetch_openimages_phones.py --phones 8000 --negatives 2500
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "training" / "openimages"

CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
BOX_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
PHONE = "Mobile phone"
NEGATIVES = ("Bottle", "Remote control", "Computer mouse", "Calculator")


def class_ids() -> dict[str, str]:
    rows = csv.reader(io.StringIO(requests.get(CLASSES_URL, timeout=60).text))
    return {name: mid for mid, name in rows}


def scan(split: str, phone_mid: str, neg_mids: set[str]):
    """Stream one annotation CSV (train is ~2GB) keeping only what we need.
    Returns ({image: [(xmin, ymin, xmax, ymax)]} phones, {image} negatives)."""
    phones: dict[str, list] = {}
    negs: set[str] = set()
    depicted: set[str] = set()
    with requests.get(BOX_URLS[split], stream=True, timeout=120) as r:
        r.raise_for_status()
        lines = (ln.decode("utf-8") for ln in r.iter_lines() if ln)
        header = next(lines).split(",")
        col = {name: i for i, name in enumerate(header)}
        for n, line in enumerate(lines):
            f = line.split(",")
            label = f[col["LabelName"]]
            if label != phone_mid and label not in neg_mids:
                continue
            img = f[col["ImageID"]]
            if label == phone_mid:
                if f[col["IsDepiction"]] == "1":
                    depicted.add(img)
                    continue
                box = tuple(float(f[col[k]]) for k in ("XMin", "YMin", "XMax", "YMax"))
                phones.setdefault(img, []).append(box)
            else:
                negs.add(img)
            if n and n % 2_000_000 == 0:
                print(f"  {split}: {n:,} rows scanned, {len(phones):,} phone images so far", flush=True)
    for img in depicted:
        phones.pop(img, None)   # a drawing of a phone somewhere in it: skip the image
    return phones, negs - set(phones) - depicted


def fetch(split: str, image_id: str, boxes: list, max_side: int) -> bool:
    import cv2
    import numpy as np

    img_path = OUT / "images" / f"{image_id}.jpg"
    if img_path.exists():
        return True
    try:
        r = requests.get(IMAGE_URL.format(split=split, image_id=image_id), timeout=30)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return False
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}" for x1, y1, x2, y2 in boxes]
    (OUT / "labels" / f"{image_id}.txt").write_text("\n".join(lines))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phones", type=int, default=5000, help="images with phones")
    ap.add_argument("--negatives", type=int, default=1500, help="images with bottles/remotes/mice and no phone")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    ids = class_ids()
    phone_mid = ids[PHONE]
    neg_mids = {ids[n] for n in NEGATIVES if n in ids}

    pos: list[tuple[str, str, list]] = []
    neg: list[tuple[str, str]] = []
    for split in ("validation", "test", "train"):
        print(f"Scanning {split} annotations ...", flush=True)
        phones, negs = scan(split, phone_mid, neg_mids)
        pos += [(split, i, b) for i, b in phones.items()]
        neg += [(split, i) for i in negs]
        print(f"  {split}: {len(phones):,} phone images, {len(negs):,} negative images", flush=True)

    rng = random.Random(0)
    # Prefer small phones: on these CCTV cameras a phone is ~2-5% of the frame width.
    pos.sort(key=lambda p: min((b[2] - b[0]) for b in p[2]))
    small = [p for p in pos if min(b[2] - b[0] for b in p[2]) <= 0.25]
    rest = [p for p in pos if p not in small] if len(small) < args.phones else []
    rng.shuffle(small)
    rng.shuffle(rest)
    chosen_pos = (small + rest)[: args.phones]
    rng.shuffle(neg)
    chosen_neg = neg[: args.negatives]
    print(f"Downloading {len(chosen_pos)} phone images + {len(chosen_neg)} negatives ...", flush=True)

    jobs = [(s, i, b) for s, i, b in chosen_pos] + [(s, i, []) for s, i in chosen_neg]
    ok = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch, s, i, b, args.max_side) for s, i, b in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            ok += fut.result()
            if n % 250 == 0:
                print(f"  {n}/{len(jobs)} done ({ok} ok)", flush=True)
    (OUT / "classes.txt").write_text("phone\n")
    (OUT / "ATTRIBUTION.txt").write_text(
        "Images: Open Images Dataset V7 (Google), listed as CC BY 2.0 by their authors.\n"
        "Annotations: Open Images, CC BY 4.0. https://storage.googleapis.com/openimages/web/index.html\n"
    )
    print(f"Done: {ok} images in {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
