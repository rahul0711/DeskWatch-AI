"""Step 4 gate: SCRFD detection + ByteTrack on the live RTSP stream, shown in
a live window so track ID stability can be confirmed visually while people
walk through frame.

Usage:
    source .venv/bin/activate
    python scripts/live_tracking_check.py [--camera 2]

Press 'q' in the window, or close it, to stop. Each track ID is logged to
the console the first time it appears and when it's lost, so you can also
verify stability from the terminal output alone.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.camera.rtsp_stream import RTSPStream  # noqa: E402
from app.config import load_config  # noqa: E402

WINDOW_NAME = "Step 4: SCRFD + ByteTrack (press q to quit)"
DETECTION_FPS = 10.0
LOW_CONF_WIDTH_PX = 60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2, help="1-indexed camera number from .env (CAMERA_<n>_*)")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=600.0, help="safety auto-stop")
    args = parser.parse_args()

    cfg = load_config()
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
        raise SystemExit("CUDAExecutionProvider did not load -- aborting (did you `source .venv/bin/activate`?)")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.4,
        lost_track_buffer=int(DETECTION_FPS * 3),  # ~3s occlusion tolerance
        minimum_matching_threshold=0.8,
        frame_rate=DETECTION_FPS,
        minimum_consecutive_frames=2,  # needs 2 consecutive detections before a track counts as valid, to suppress one-off false positives
    )

    stream = RTSPStream(
        rtsp_url=cam.rtsp_url,
        transport=cam.transport,
        rtsp_url_masked=cam.rtsp_url_masked,
        name=cam.name,
    ).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    seen_ids: dict[int, float] = {}
    active_ids: set[int] = set()
    min_detection_interval = 1.0 / DETECTION_FPS
    last_detect_time = 0.0
    last_frame_index = -1
    start = time.monotonic()
    print("Window open -- walk through frame. Press 'q' in the window to stop.")

    try:
        while time.monotonic() - start < args.max_seconds:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            now = time.monotonic()
            run_detection = (frame.frame_index != last_frame_index) and (now - last_detect_time) >= min_detection_interval
            display_img = frame.image

            if run_detection:
                last_frame_index = frame.frame_index
                last_detect_time = now
                faces = fa.get(frame.image)

                if faces:
                    xyxy = np.array([f.bbox for f in faces], dtype=np.float32)
                    confidence = np.array([f.det_score for f in faces], dtype=np.float32)
                    detections = sv.Detections(xyxy=xyxy, confidence=confidence)
                else:
                    detections = sv.Detections.empty()

                tracked = tracker.update_with_detections(detections)

                current_ids = set()
                img = frame.image.copy()
                for box, conf, tid in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                    x1, y1, x2, y2 = box.astype(int)
                    w = x2 - x1
                    current_ids.add(int(tid))
                    if int(tid) not in seen_ids:
                        seen_ids[int(tid)] = now
                        print(f"[+{now - start:5.1f}s] NEW track id={tid}  width={w}px  conf={conf:.2f}")

                    low_conf = w < LOW_CONF_WIDTH_PX
                    color = (0, 165, 255) if low_conf else (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    flag = " LOW-CONF" if low_conf else ""
                    label = f"id={tid} {conf:.2f} w={w}px{flag}"
                    cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                lost = active_ids - current_ids
                for tid in lost:
                    print(f"[+{now - start:5.1f}s] LOST track id={tid}  (was tracked for {now - seen_ids[tid]:.1f}s)")
                active_ids = current_ids
                display_img = img

            cv2.imshow(WINDOW_NAME, display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal unique track IDs seen: {len(seen_ids)}")


if __name__ == "__main__":
    main()
