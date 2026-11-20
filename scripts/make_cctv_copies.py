"""Make CCTV-looking copies of the Open Images phone photos.

Open Images phones are mostly big, sharp, eye-level. On this site's ceiling
cameras a phone is ~25-60px wide, soft, and heavily compressed. For every
phone photo this writes one extra training image that looks like that:

  1. the phone photo is shrunk so its phones end up ~20-70px wide and pasted
     at a random spot onto a NEGATIVE photo (bottles, remotes, mice -- no
     phones), so the small phone sits in a cluttered scene;
  2. the whole image then gets CCTV degradation: down-and-up rescale (lost
     detail), Gaussian blur, JPEG at quality 25-55, brightness / contrast /
     saturation shifts, sometimes a colour cast or near-grey.

    data/training/openimages/images/<id>_cctv.jpg (+ labels/<id>_cctv.txt)

Usage:  python scripts/make_cctv_copies.py
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "training" / "openimages"
CANVAS = (1280, 720)          # w, h -- the shape of a camera frame


def read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        parts = line.split()
        if len(parts) == 5:
            out.append(tuple(map(float, parts[1:])))
    return out


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    f = rng.uniform(0.45, 0.8)                         # lost detail
    img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.6:
        k = rng.choice((3, 5))
        img = cv2.GaussianBlur(img, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.4, 1.1)               # CCTV is often washed out
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.75, 1.2) + rng.uniform(-15, 25)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.25:                            # colour cast (white balance)
        cast = np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
        img = np.clip(img.astype(np.float32) * cast, 0, 255).astype(np.uint8)
    if rng.random() < 0.1:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 55)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> None:
    rng = random.Random(0)
    imgs = sorted(p for p in (DATA / "images").glob("*.jpg") if "_cctv" not in p.stem)
    positives = [p for p in imgs if read_labels(DATA / "labels" / f"{p.stem}.txt")]
    negatives = [p for p in imgs if not read_labels(DATA / "labels" / f"{p.stem}.txt")]
    if not positives or not negatives:
        raise SystemExit("Run scripts/fetch_openimages_phones.py first.")
    cw, ch = CANVAS
    made = 0
    for p in positives:
        out_img = DATA / "images" / f"{p.stem}_cctv.jpg"
        if out_img.exists():
            continue
        src = cv2.imread(str(p))
        bg = cv2.imread(str(rng.choice(negatives)))
        if src is None or bg is None:
            continue
        boxes = read_labels(DATA / "labels" / f"{p.stem}.txt")
        sh, sw = src.shape[:2]
        # Scale so the widest phone lands at 20-70px on the 1280px canvas.
        widest = max(b[2] for b in boxes) * sw
        scale = rng.uniform(20, 70) / max(widest, 1.0)
        scale = min(scale, cw * 0.9 / sw, ch * 0.9 / sh)
        nw, nh = max(8, int(sw * scale)), max(8, int(sh * scale))
        small = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(bg, (cw, ch), interpolation=cv2.INTER_LINEAR)
        ox, oy = rng.randint(0, cw - nw), rng.randint(0, ch - nh)
        # Feathered paste: the photo fades into the background over its outer
        # quarter, so there is no hard rectangle edge for the model to learn
        # as a "phone is here" shortcut (real camera frames have none).
        fy, fx = max(1, nh // 4), max(1, nw // 4)
        ramp_y = np.minimum(np.arange(nh), np.arange(nh)[::-1]) / fy
        ramp_x = np.minimum(np.arange(nw), np.arange(nw)[::-1]) / fx
        alpha = np.clip(np.minimum.outer(ramp_y, ramp_x), 0, 1)[..., None].astype(np.float32)
        region = canvas[oy:oy + nh, ox:ox + nw].astype(np.float32)
        canvas[oy:oy + nh, ox:ox + nw] = (alpha * small + (1 - alpha) * region).astype(np.uint8)
        lines = []
        for xc, yc, bw, bh in boxes:
            px, py, pw, ph = (ox + xc * nw) / cw, (oy + yc * nh) / ch, bw * nw / cw, bh * nh / ch
            inner = 0.2 <= xc <= 0.8 and 0.2 <= yc <= 0.8   # faded edges -> not a clean example
            if pw * cw >= 4 and ph * ch >= 4 and inner:     # too small to see -> drop the box
                lines.append(f"0 {px:.6f} {py:.6f} {pw:.6f} {ph:.6f}")
        cv2.imwrite(str(out_img), degrade(canvas, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        (DATA / "labels" / f"{p.stem}_cctv.txt").write_text("\n".join(lines))
        made += 1
    print(f"Made {made} CCTV-style copies in {DATA / 'images'}")


if __name__ == "__main__":
    main()
