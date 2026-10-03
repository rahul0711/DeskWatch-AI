"""Step 2/3: smallest possible program that proves we can pull live frames
directly from the camera over RTSP (no face detection, no HUD, no threads).

Run this FIRST, before main.py, whenever setting up a new camera. If this
doesn't show a window with live video, nothing downstream (detection,
tracking) can work either -- fix the connection here first.

Usage:
    python scripts/test_connection.py              # first configured camera
    python scripts/test_connection.py --camera 2   # CAMERA_2_* from .env
(reads CAMERA_* / CAMERA_N_* from .env, same as the full app)
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from app.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", type=int, default=1,
        help="1-indexed camera to test (matches CAMERA_N_* in .env). Default: 1.",
    )
    args = parser.parse_args()

    config = load_config()
    if not config.cameras:
        print("No cameras configured. Set CAMERA_IP/CAMERA_RTSP_URL (single "
              "camera) or CAMERA_1_IP, CAMERA_2_IP, ... (multiple) in .env.")
        return 1
    if not (1 <= args.camera <= len(config.cameras)):
        print(f"--camera {args.camera} out of range: {len(config.cameras)} "
              f"camera(s) configured.")
        return 1
    camera = config.cameras[args.camera - 1]

    try:
        url = camera.rtsp_url
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 1

    print(f"Testing camera: {camera.name}")
    print(f"Connecting to: {camera.rtsp_url_masked}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("FAILED to open stream. Check IP/port/credentials/path, and that")
        print("this machine can reach the camera. Try scripts/discover_camera.py.")
        return 1

    print("[OK] Stream opened.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Reported resolution: {w}x{h}")

    frame_count = 0
    start = time.time()
    print("Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Failed to read a frame (stream may have dropped).")
            break

        frame_count += 1
        if frame_count == 1:
            print(f"[OK] First live frame received. Shape={frame.shape}")

        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"Frames: {frame_count}  FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(f"Step 2/3 - Raw RTSP connection test ({camera.name}, no AI yet)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nReceived {frame_count} frames over {time.time() - start:.1f}s.")
    return 0 if frame_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
