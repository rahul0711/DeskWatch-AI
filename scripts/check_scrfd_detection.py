"""Step 3 gate: run SCRFD (insightface buffalo_l pack) on the live RTSP
stream, draw boxes + landmarks, save annotated frames so detection quality
can be confirmed visually before wiring up tracking.

Usage:
    source .venv/bin/activate
    python scripts/check_scrfd_detection.py [--camera 1] [--frames 8]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.camera.rtsp_stream import RTSPStream  # noqa: E402
from app.config import load_config  # noqa: E402

OUT_DIR = ROOT / "data" / "diagnostics" / "scrfd_check"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1, help="1-indexed camera number from .env (CAMERA_<n>_*)")
    parser.add_argument("--frames", type=int, default=8, help="number of annotated frames to save")
    parser.add_argument("--det-size", type=int, default=640, help="SCRFD detector input size (square)")
    parser.add_argument("--interval", type=float, default=0.0, help="minimum seconds between captured frames (spreads capture over real time)")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.cameras:
        raise SystemExit("No cameras configured in .env")
    idx = args.camera - 1
    if idx < 0 or idx >= len(cfg.cameras):
        raise SystemExit(f"--camera {args.camera} out of range, {len(cfg.cameras)} camera(s) configured")
    cam = cfg.cameras[idx]

    print(f"Camera: {cam.name}  URL: {cam.rtsp_url_masked}")

    from insightface.app import FaceAnalysis

    fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    active_provider = fa.models["detection"].session.get_providers()[0]
    print("SCRFD detector active provider:", active_provider)
    if active_provider != "CUDAExecutionProvider":
        raise SystemExit("CUDAExecutionProvider did not load for SCRFD -- aborting (did you `source .venv/bin/activate`?)")

    stream = RTSPStream(
        rtsp_url=cam.rtsp_url,
        transport=cam.transport,
        rtsp_url_masked=cam.rtsp_url_masked,
        name=cam.name,
    ).start()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    deadline = time.monotonic() + max(30.0, args.interval * args.frames + 10)
    last_frame_index = -1
    last_capture_time = 0.0
    print(f"Waiting for stream to connect and grabbing frames (spread over ~{args.interval * args.frames:.0f}s)...")

    while saved < args.frames and time.monotonic() < deadline:
        frame = stream.get_latest_frame()
        now = time.monotonic()
        if frame is None or frame.frame_index == last_frame_index or (now - last_capture_time) < args.interval:
            time.sleep(0.1)
            continue
        last_frame_index = frame.frame_index
        last_capture_time = now

        faces = fa.get(frame.image)
        img = frame.image.copy()
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            w = x2 - x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{f.det_score:.2f} w={w}px"
            cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if f.kps is not None:
                for (kx, ky) in f.kps.astype(int):
                    cv2.circle(img, (kx, ky), 2, (0, 0, 255), -1)

        out_path = OUT_DIR / f"frame_{saved:02d}_faces{len(faces)}.jpg"
        cv2.imwrite(str(out_path), img)
        widths = [int(f.bbox[2] - f.bbox[0]) for f in faces]
        print(f"[{saved+1}/{args.frames}] frame #{frame.frame_index}: {len(faces)} face(s), widths={widths} -> {out_path.name}")
        saved += 1

    stream.stop()

    if saved == 0:
        raise SystemExit("Never got a frame from the stream within 30s -- check camera connectivity/credentials")
    print(f"\nSaved {saved} annotated frame(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
