"""Detection worker running in a fully separate OS process.

AsyncDetectionWorker (a background thread) keeps detection off the render
loop's call stack, but the render loop and the detector's own Python-side
work (NMS, box math, Ultralytics results wrapping) still share ONE
interpreter's GIL -- heavy Python activity inside detect() can still starve
cv2.imshow()/waitKey() for CPU time even though it's technically "off
thread". A separate process has its own GIL, so the render loop genuinely
cannot be blocked by anything detection does, no matter how much Python-level
work is involved.

Frames go to the child via a maxsize=1 queue (a new submit() drops whatever
was still waiting -- we only ever want the latest); results come back the
same way.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import numpy as np

from app.tracking.byte_tracker import ByteTrackFaceTracker
from app.tracking.simple_tracker import SimpleIouTracker, TrackedFace

logger = logging.getLogger(__name__)

# 'fork' is documented as unsafe once Cocoa (cv2.imshow's HighGUI backend) or
# Metal (PyTorch MPS) state exists in the parent process on macOS -- spawn
# gives the child a clean interpreter instead of copying that state.
_CTX = mp.get_context("spawn")


@dataclass
class DetectionResult:
    tracked_faces: list[TrackedFace]
    frame: np.ndarray | None  # only populated when return_frame=True (crop saving)
    sequence: int
    detect_ms: float


def _drain(q) -> None:
    try:
        while True:
            q.get_nowait()
    except (queue.Empty, OSError, ValueError):
        pass


def _worker_main(
    detector_class,
    detector_kwargs: dict,
    return_frame: bool,
    frame_queue,
    result_queue,
    stop_event,
) -> None:
    """Entry point run inside the child process -- owns its own detector
    instance, tracker, and GIL. detector_class can be any class exposing
    detect(frame) -> list[DetectedFace] and close() -- FaceDetector and
    PersonDetector both fit (module-level classes are picklable by
    reference, so this works fine across the spawn boundary). Exception:
    a detector with PROVIDES_TRACK_IDS = True (currently only
    ActivityDetector, which uses BoT-SORT internally) returns already-
    tracked list[TrackedFace] instead, and skips the tracker below.
    A detector with TRACKER_BACKEND = "bytetrack" (FaceDetector,
    InsightFaceAttendanceDetector) gets ByteTrackFaceTracker instead of
    the default SimpleIouTracker -- see app/tracking/byte_tracker.py for
    why that matters specifically for face identity continuity."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Construction can fail for reasons that have nothing to do with any one
    # frame -- a missing model file, or a backend that can't run on this
    # platform at all (see OCRDetector's oneDNN/Windows note). Letting that
    # propagate kills the child silently: the parent's latest_result() just
    # keeps returning the empty startup result, so the UI reads "0 detections"
    # forever and looks like an empty scene rather than a broken detector.
    try:
        detector = detector_class(**detector_kwargs)
    except Exception:
        logger.exception(
            "%s failed to initialize in its worker process -- this detection "
            "mode will report nothing until the cause is fixed and the app is "
            "restarted.",
            getattr(detector_class, "__name__", detector_class),
        )
        return

    provides_own_tracking = getattr(detector_class, "PROVIDES_TRACK_IDS", False)
    tracker_backend = getattr(detector_class, "TRACKER_BACKEND", "iou")
    if provides_own_tracking:
        tracker = None
    elif tracker_backend == "bytetrack":
        tracker = ByteTrackFaceTracker()
    else:
        tracker = SimpleIouTracker()

    recognizer = None
    is_face_detector = (getattr(detector_class, "__name__", "") in ("FaceDetector", "InsightFaceAttendanceDetector"))
    if is_face_detector:
        try:
            from app.detection.face_recognizer import FaceRecognizer
            # Was FaceRecognizer() with no args -- always defaulted to
            # CPUExecutionProvider regardless of .env's ONNX_PROVIDER.
            # detector_kwargs already carries the configured provider (it's
            # what FaceDetector/InsightFaceAttendanceDetector themselves are
            # built with above), so reuse it here instead of the class default.
            recognizer = FaceRecognizer(
                onnx_provider=detector_kwargs.get("onnx_provider", "CPUExecutionProvider")
            )
        except Exception:
            logger.exception("Failed to initialize FaceRecognizer inside process worker")

    try:
        while not stop_event.is_set():
            try:
                frame, seq = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            start = time.monotonic()
            try:
                raw = detector.detect(frame)
            except Exception:
                logger.exception("Detection failed in worker process")
                continue
            tracked = raw if provides_own_tracking else tracker.update(raw)

            # Perform face recognition in the child process worker
            if recognizer is not None and tracked:
                from app.detection.face_detector import crop_face
                for t in tracked:
                    if t.name == "Unknown":
                        face_crop = crop_face(frame, t.face.bbox)
                        if face_crop.size > 0:
                            name, dist = recognizer.recognize(face_crop)
                            if name != "Unknown":
                                t.name = name
                                t.name_confidence = max(0.0, 1.0 - dist)

            detect_ms = (time.monotonic() - start) * 1000

            _drain(result_queue)
            try:
                result_queue.put_nowait(
                    (tracked, frame if return_frame else None, seq, detect_ms)
                )
            except queue.Full:
                pass
    finally:
        detector.close()


class ProcessDetectionWorker:
    """Parent-side handle: submit frames, read back whatever the latest result is."""

    def __init__(self, detector_class, detector_kwargs: dict, return_frame: bool = False) -> None:
        self._detector_class = detector_class
        self._detector_kwargs = detector_kwargs
        self._return_frame = return_frame
        self._frame_queue = _CTX.Queue(maxsize=1)
        self._result_queue = _CTX.Queue(maxsize=1)
        self._stop_event = _CTX.Event()
        self._process: mp.process.BaseProcess | None = None
        self._submit_seq = 0
        self._result = DetectionResult(tracked_faces=[], frame=None, sequence=0, detect_ms=0.0)

    def start(self) -> "ProcessDetectionWorker":
        self._process = _CTX.Process(
            target=_worker_main,
            args=(
                self._detector_class,
                self._detector_kwargs,
                self._return_frame,
                self._frame_queue,
                self._result_queue,
                self._stop_event,
            ),
            daemon=True,
            name=f"detect-process-{getattr(self._detector_class, '__name__', 'unknown')}",
        )
        self._process.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

    def submit(self, frame: np.ndarray) -> None:
        """Hand the worker the latest frame to detect next. Never blocks."""
        self._submit_seq += 1
        try:
            _drain(self._frame_queue)
            self._frame_queue.put_nowait((frame, self._submit_seq))
        except (queue.Full, OSError, ValueError):
            pass  # child is still mid-detect or shutting down

    def latest_result(self) -> DetectionResult:
        latest = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except (queue.Empty, OSError, ValueError):
            pass
        if latest is not None:
            tracked, frame, seq, detect_ms = latest
            self._result = DetectionResult(tracked, frame, seq, detect_ms)
        return self._result
