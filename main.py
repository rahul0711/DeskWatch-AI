"""Phase 1 entry point.

Prama CCTV camera(s) -> RTSP -> live frames -> YOLO face detection -> HUD.

Supports any number of cameras: define CAMERA_1_*, CAMERA_2_*, ... in .env
(or the legacy unprefixed CAMERA_* vars for a single camera -- see
app/config.py). Each camera gets its own capture thread and its own detection
process; all of them share one window with a button bar to switch between
"Camera 1", "Camera 2", ... and "Both"/"All".

Run:
    python main.py
Switch view:
    click the buttons at the top, or press 1..9 (one camera) / 0, a, b (all).
Quit:
    press 'q' or Esc in the window, or Ctrl+C in the terminal.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict, dataclass

from app.camera.rtsp_stream import RTSPStream
from app.config import AppConfig, CameraConfig, load_config
from app.processing.pipeline import FacePipeline
from app.ui.display import MultiCameraDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Anything a render-loop iteration takes above this is logged with a
# breakdown, so a recurring freeze can be pinned to poll() (result readback)
# vs draw_overlay() (box/HUD drawing) vs display.show() (cv2.imshow) instead
# of guessed at.
_STALL_THRESHOLD_MS = 100.0


@dataclass
class CameraSession:
    camera: CameraConfig
    stream: RTSPStream
    pipeline: FacePipeline


def _start_camera(camera: CameraConfig, config: AppConfig) -> CameraSession | None:
    try:
        rtsp_url = camera.rtsp_url
    except ValueError as exc:
        logger.error("[%s] %s", camera.name, exc)
        return None

    logger.info("[%s] RTSP URL: %s", camera.name, camera.rtsp_url_masked)

    stream = RTSPStream(
        rtsp_url=rtsp_url,
        transport=camera.transport,
        initial_reconnect_delay=config.reconnect.initial_delay,
        max_reconnect_delay=config.reconnect.max_delay,
        rtsp_url_masked=camera.rtsp_url_masked,
        name=camera.name,
    ).start()

    pipeline = FacePipeline(
        stream=stream,
        detector_kwargs=dict(
            backend=config.detection.backend,
            onnx_provider=config.detection.onnx_provider,
            det_size=config.detection.det_size,
            conf_threshold=config.detection.conf_threshold,
        ),
        max_display_fps=config.max_display_fps,
        max_detection_fps=config.max_detection_fps,
        save_crops=config.crops.enabled,
        crops_dir=config.crops.directory / camera.name.replace(" ", "_"),
        max_saved_crops=config.crops.max_saved,
        head_count_source=config.detection.head_count_source,
        person_conf_threshold=config.detection.person_conf_threshold,
        ocr_kwargs=asdict(config.ocr),
        enabled_detectors=config.detection.enabled_detectors,
    )

    return CameraSession(camera=camera, stream=stream, pipeline=pipeline)


def main() -> int:
    config = load_config()

    if not config.cameras:
        logger.error(
            "No cameras configured. Set CAMERA_IP/CAMERA_RTSP_URL (single "
            "camera) or CAMERA_1_IP, CAMERA_2_IP, ... (multiple) in .env."
        )
        return 1

    logger.info(
        "Starting %d camera(s): %s",
        len(config.cameras),
        ", ".join(c.name for c in config.cameras),
    )
    logger.info(
        "Starting face detector process(es) (first run downloads the model, "
        "~a few hundred MB)..."
    )

    sessions = [
        session
        for session in (_start_camera(camera, config) for camera in config.cameras)
        if session is not None
    ]
    if not sessions:
        logger.error("No cameras could be started.")
        return 1

    logger.info("Waiting for first frame from camera(s)...")

    display = MultiCameraDisplay(
        [session.camera.name for session in sessions],
        head_count_source=config.detection.head_count_source,
    )
    latest: list = [None] * len(sessions)

    try:
        while True:
            # Every pipeline is polled each pass so hidden cameras keep
            # draining; only a new frame from a visible one forces a redraw.
            dirty = display.consume_view_changed()
            t0 = time.monotonic()
            for i, session in enumerate(sessions):
                result = session.pipeline.poll()
                if result is None:
                    continue
                latest[i] = result
                if display.is_visible(i):
                    dirty = True
                if result.ran_detection and result.tracked_faces:
                    for t in result.tracked_faces:
                        logger.debug(
                            "[%s] Face #%s confidence=%.2f bbox=%s",
                            session.camera.name, t.track_id,
                            t.face.confidence, t.face.bbox,
                        )
            t1 = time.monotonic()

            if dirty:
                display.render(latest)
                t2 = time.monotonic()
                total_ms = (t2 - t0) * 1000
                if total_ms > _STALL_THRESHOLD_MS:
                    logger.warning(
                        "Slow render iteration: %.0fms total (poll=%.0fms render=%.0fms)",
                        total_ms, (t1 - t0) * 1000, (t2 - t1) * 1000,
                    )

            if display.poll_quit():
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        for session in sessions:
            session.stream.stop()
            session.pipeline.close()
        display.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
