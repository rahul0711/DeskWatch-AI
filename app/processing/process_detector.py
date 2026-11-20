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


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _match_registered(matcher, frame: np.ndarray, tracks: list[TrackedFace]) -> None:
    """Match the selected faces against registered users in one batch. A
    match names the track (and so the person, via the identity cache);
    a later non-match does not erase an earlier good one."""
    usable = [t for t in tracks if getattr(t.face, "landmarks", None) is not None and len(t.face.landmarks) == 5]
    if not usable:
        return
    try:
        results = matcher.match(frame, [t.face.landmarks for t in usable])
    except Exception:
        logger.exception("Registered-user matching failed for this pass")
        return
    for t, m in zip(usable, results):
        if m is None:
            continue
        if t.registered_user_id != m.user_id:
            logger.info("Face track %s -> registered user %s (%s, cos=%.2f)", t.track_id, m.name, m.employee_id, m.score)
        t.registered_user_id = m.user_id
        t.person_id = m.employee_id
        t.registered_score = m.score
        t.name = m.name
        t.name_confidence = m.score


class _EmbeddingScheduler:
    """Decides WHEN a tracked face gets an (expensive) InsightFace
    embedding, and caches it on the track in between.

    Runs recognition only when:
      * a new track appears, or it still has no embedding (retried, rate-limited)
      * it is still not matched to a registered user (retried every
        `unidentified_retry_seconds`, at most `unidentified_max_tries` times,
        then only on revalidation -- a stranger is not re-embedded forever)
      * the face is visible again after an occlusion gap
      * the box jumped (tracker identity uncertain -- possible id swap)
      * periodic revalidation is due
    and never for more than `max_per_pass` faces in one pass, tracks without
    any embedding first. Every other frame just carries the cached value.
    """

    def __init__(
        self,
        revalidate_seconds: float = 10.0,
        retry_seconds: float = 0.5,
        reappear_gap_seconds: float = 1.0,
        jump_iou: float = 0.2,
        max_per_pass: int = 3,
        min_face_px: int = 24,
        unidentified_retry_seconds: float = 1.0,
        unidentified_max_tries: int = 8,
    ) -> None:
        self._unid_retry = unidentified_retry_seconds
        self._unid_max = unidentified_max_tries
        self._revalidate = revalidate_seconds
        self._retry = retry_seconds
        self._gap = reappear_gap_seconds
        self._jump_iou = jump_iou
        self._max_per_pass = max_per_pass
        self._min_px = min_face_px
        # track_id -> {"last_run", "last_seen", "bbox"}
        self._state: dict[int, dict] = {}

    def reason(self, t: TrackedFace, now: float) -> str | None:
        x1, y1, x2, y2 = t.face.bbox
        if min(x2 - x1, y2 - y1) < self._min_px:
            return None
        st = self._state.get(t.track_id)
        if st is None:
            return "new_track"
        if t.embedding is None:
            return "no_embedding" if now - st["last_run"] >= self._retry else None
        if now - st["last_seen"] >= self._gap:
            st["tries"] = 0
            return "after_occlusion"
        if _box_iou(st["bbox"], t.face.bbox) < self._jump_iou:
            return "uncertain"
        if now - st["last_run"] >= self._revalidate:
            st["tries"] = 0
            return "revalidate"
        if (
            t.registered_user_id is None
            and st.get("tries", 0) < self._unid_max
            and now - st["last_run"] >= self._unid_retry
        ):
            return "unidentified"
        return None

    def select(self, tracked: list[TrackedFace], now: float) -> list[tuple[TrackedFace, str]]:
        due = [(t, r) for t in tracked if (r := self.reason(t, now)) is not None]
        due.sort(key=lambda tr: tr[0].embedding is not None)  # no-embedding first
        return due[: self._max_per_pass]

    def ran(self, t: TrackedFace, now: float) -> None:
        st = self._state.setdefault(t.track_id, {})
        st["last_run"] = now
        st["tries"] = st.get("tries", 0) + 1

    def seen(self, tracked: list[TrackedFace], now: float) -> None:
        live = set()
        for t in tracked:
            st = self._state.setdefault(t.track_id, {"last_run": -1e9})
            st["last_seen"] = now
            st["bbox"] = t.face.bbox
            live.add(t.track_id)
        # Forget tracks gone for a while (ByteTrack ids never come back).
        for tid in [k for k, v in self._state.items() if k not in live and now - v.get("last_seen", now) > 60.0]:
            del self._state[tid]


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

    scheduler = None
    if recognizer is not None:
        try:
            from app.activity.config import load_activity_config
            revalidate_s = load_activity_config().reid_revalidate_seconds
        except Exception:
            revalidate_s = 10.0
        scheduler = _EmbeddingScheduler(revalidate_seconds=revalidate_s)

    # Registered users (Register page): AdaFace + FAISS, same path as enrollment.
    matcher = None
    if recognizer is not None:
        try:
            from app.attendance.registered_matcher import RegisteredMatcher
            matcher = RegisteredMatcher()
        except Exception:
            logger.exception("Registered-user matching disabled in this worker")

    last_control = None
    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            frame, seq = item[0], item[1]
            # Optional third element: a control value from the parent
            # (ProcessDetectionWorker.set_control), e.g. the activity focus.
            control = item[2] if len(item) > 2 else None
            if control != last_control and hasattr(detector, "set_focus"):
                detector.set_focus(control)
                last_control = control

            start = time.monotonic()
            try:
                raw = detector.detect(frame)
            except Exception:
                logger.exception("Detection failed in worker process")
                continue
            tracked = raw if provides_own_tracking else tracker.update(raw)

            # Face embedding + recognition, on a schedule -- NOT every frame.
            # The cached embedding/name ride along on the (persistent)
            # TrackedFace object in between, so the parent always sees them.
            if recognizer is not None and tracked:
                from app.detection.face_detector import crop_face
                now = time.monotonic()
                selected = [t for t, _reason in scheduler.select(tracked, now)]
                for t in selected:
                    scheduler.ran(t, now)
                    face_crop = crop_face(frame, t.face.bbox)
                    if face_crop.size == 0:
                        continue
                    emb = recognizer.embed(face_crop)
                    if emb is None:
                        continue
                    t.embedding = np.asarray(emb, dtype=np.float32)
                    name, dist = recognizer.match_embedding(t.embedding)
                    if name != "Unknown" and t.registered_user_id is None:
                        t.name = name
                        t.name_confidence = max(0.0, 1.0 - dist)
                if matcher is not None:
                    _match_registered(matcher, frame, selected)
                scheduler.seen(tracked, now)

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

    def set_control(self, value) -> None:
        """Picklable value delivered to the child with the next frame; the
        detector's set_focus(value) is called when it changes."""
        self._control = value

    def submit(self, frame: np.ndarray) -> None:
        """Hand the worker the latest frame to detect next. Never blocks."""
        self._submit_seq += 1
        try:
            _drain(self._frame_queue)
            self._frame_queue.put_nowait((frame, self._submit_seq, getattr(self, "_control", None)))
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
