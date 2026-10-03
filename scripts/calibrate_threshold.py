"""Threshold calibration: do NOT guess MATCH_THRESHOLD, measure it.

Feed this a folder of face crops of people who ARE already enrolled
(--known-dir) and a folder of face crops of people who are NOT enrolled
(--unknown-dir). Each image is detected/aligned/embedded and searched
against the live FAISS index; the top-1 cosine similarity score is recorded.

"Known" scores should cluster high (genuine matches), "unknown" scores
should cluster low (impostor/no-match). Wherever the two distributions
overlap is where mistakes happen -- this script sweeps candidate cutoffs and
recommends the one that best separates them (minimizes false-accept +
false-reject), then prints both distributions so you can judge whether the
recommended cutoff has an uncomfortably large overlap for your use case.

AdaFace's cosine-similarity distribution is NOT the same as ArcFace's or
SFace's -- do not reuse a threshold from a different project/model.

Usage:
    source .venv/bin/activate
    python scripts/calibrate_threshold.py --known-dir data/calibration/known --unknown-dir data/calibration/unknown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.attendance.alignment import align_face  # noqa: E402
from app.attendance.config import load_attendance_config  # noqa: E402
from app.attendance.detector import ScrfdDetector  # noqa: E402
from app.attendance.embedder import AdaFaceEmbedder  # noqa: E402
from app.attendance.faiss_index import FaceIndex  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def score_folder(folder: Path, detector: ScrfdDetector, embedder: AdaFaceEmbedder, index: FaceIndex) -> list[float]:
    scores = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip {path.name}: could not decode")
            continue
        faces = detector.detect(img)
        if not faces:
            print(f"  skip {path.name}: no face detected")
            continue
        face = max(faces, key=lambda f: f.width_px)
        aligned = align_face(img, face.kps)
        embedding = embedder.embed_aligned_batch([aligned])[0]
        match = index.best_match(embedding)
        if match is None:
            print(f"  skip {path.name}: FAISS index is empty (enroll some users first)")
            continue
        _, score = match
        scores.append(score)
        print(f"  {path.name}: top-1 cosine similarity = {score:.4f}")
    return scores


def print_distribution(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"{name}: no scores collected")
        return
    arr = np.array(scores)
    pct = np.percentile(arr, [5, 25, 50, 75, 95])
    print(
        f"{name}: n={len(arr)}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
        f"min={arr.min():.4f}  max={arr.max():.4f}"
    )
    print(f"  percentiles p5={pct[0]:.4f} p25={pct[1]:.4f} p50={pct[2]:.4f} p75={pct[3]:.4f} p95={pct[4]:.4f}")


def recommend_threshold(known: list[float], unknown: list[float]) -> float | None:
    if not known or not unknown:
        return None
    best_t, best_err = 0.5, float("inf")
    for t in np.arange(0.0, 1.001, 0.01):
        frr = np.mean(np.array(known) < t)  # false reject rate
        far = np.mean(np.array(unknown) >= t)  # false accept rate
        err = frr + far
        if err < best_err:
            best_err, best_t = err, t
    return float(best_t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--known-dir", required=True, type=Path)
    parser.add_argument("--unknown-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.known_dir.is_dir() or not args.unknown_dir.is_dir():
        raise SystemExit("Both --known-dir and --unknown-dir must be existing directories")

    cfg = load_attendance_config()
    print(f"Current configured MATCH_THRESHOLD = {cfg.match_threshold}\n")

    detector = ScrfdDetector(cfg)
    embedder = AdaFaceEmbedder(cfg)
    index = FaceIndex(cfg)
    if index.size == 0:
        raise SystemExit(
            "FAISS index is empty -- enroll at least the people in --known-dir via the "
            "registration UI/API first, then run this script."
        )
    print(f"FAISS index loaded: {index.size} stored embeddings\n")

    print(f"Scoring known-person crops in {args.known_dir}:")
    known_scores = score_folder(args.known_dir, detector, embedder, index)
    print(f"\nScoring unknown-person crops in {args.unknown_dir}:")
    unknown_scores = score_folder(args.unknown_dir, detector, embedder, index)

    print()
    print_distribution("KNOWN  (should score high)", known_scores)
    print_distribution("UNKNOWN (should score low)", unknown_scores)

    recommended = recommend_threshold(known_scores, unknown_scores)
    print()
    if recommended is None:
        print("Not enough data in both folders to recommend a threshold.")
        return
    frr = float(np.mean(np.array(known_scores) < recommended)) if known_scores else float("nan")
    far = float(np.mean(np.array(unknown_scores) >= recommended)) if unknown_scores else float("nan")
    print(f"RECOMMENDED MATCH_THRESHOLD = {recommended:.2f}")
    print(f"  at this cutoff: false-reject rate = {frr:.1%} (known people missed), false-accept rate = {far:.1%} (unknown people wrongly matched)")
    print("  set this in .env as MATCH_THRESHOLD=<value> -- do not hardcode it in source.")
    if known_scores and unknown_scores and max(unknown_scores) > min(known_scores):
        print(
            "  WARNING: known/unknown distributions overlap (some unknown score higher than "
            "some known) -- no threshold will be perfect here. Consider more/better enrollment "
            "angles, or a stricter camera placement, before trusting this cutoff for production."
        )


if __name__ == "__main__":
    main()
