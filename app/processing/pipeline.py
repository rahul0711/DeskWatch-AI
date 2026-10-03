"""Orchestrates: capture -> (rate limit) -> submit every N frames to the
detection worker process -> read latest tracked result -> crop.

Detection runs in a fully separate OS process (ProcessDetectionWorker), not
just a background thread, so nothing it does -- GPU calls, NMS, box math --
can ever compete with the render loop for the GIL. The render loop always
gets the freshest captured frame immediately; detection results arrive
whenever the worker process finishes them.

Kept deliberately dumb and linear for Phase 1 (single camera, single
process). Multi-camera support (Phase 2+) means running one of these per
camera — nothing here assumes global/shared state, so that's a config change
plus a process-per-camera or asyncio-gather, not a rewrite.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.camera.rtsp_stream import Frame, RTSPStream
from app.detection.activity_detector import ActivityDetector
from app.detection.attendance_detector import InsightFaceAttendanceDetector
from app.detection.face_detector import FaceDetector, crop_face
from app.detection.ocr_detector import OCRDetector
from app.detection.person_detector import PersonDetector
from app.processing.process_detector import ProcessDetectionWorker
from app.tracking.simple_tracker import TrackedFace
from app.utils import RollingFps

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    frame: np.ndarray
    tracked_faces: list[TrackedFace]   # face detections
    tracked_heads: list[TrackedFace]   # body/head detections (PersonDetector)
    connection_state: str
    resolution: tuple[int, int]
    capture_fps: float
    process_fps: float
    latency_ms: float
    ran_detection: bool
    tracked_activity: list[TrackedFace] = field(default_factory=list)  # ActivityDetector
    tracked_ocr: list[TrackedFace] = field(default_factory=list)       # OCRDetector
    # True only on a pass that carries a NEW activity result. The activity
    # analyzer must be driven by this rather than by ran_detection: activity
    # runs at its own (slower) rate alongside the primary detector, so a
    # carried-over result re-fed to the state engine would double-count
    # elapsed time against whatever state was last observed.
    ran_activity: bool = False
    # TEMPORARY (latency probe, to be reverted): raw capture timestamp, so
    # the render loop can compute its own stage deltas instead of trusting
    # only the pre-baked latency_ms.
    frame_captured_ts: float = 0.0


class FacePipeline:
    def __init__(
        self,
        stream: RTSPStream,
        detector_kwargs: dict,
        max_display_fps: float = 25.0,
        max_detection_fps: float = 10.0,
        save_crops: bool = False,
        crops_dir: Path = Path("data/samples"),
        max_saved_crops: int = 20,
        head_count_source: str = "face",  # "face" | "person" / "head"
        person_conf_threshold: float = 0.4,
        ocr_kwargs: dict | None = None,
        enabled_detectors: tuple[str, ...] = ("face", "person", "attendance", "activity", "ocr"),
        activity_kwargs: dict | None = None,
        # When True the activity detector runs CONCURRENTLY with whatever
        # head_count_source is selected, instead of only when activity is
        # the selected mode. This is what lets face recognition (identity)
        # and activity analysis (behaviour) observe the same frames from the
        # same RTSP connection -- see app/activity/analyzer.py.
        activity_always_on: bool = False,
        activity_fps: float = 4.0,
    ) -> None:
        self._stream = stream
        # Two independent rate gates -- display and detection used to share
        # one (max_process_fps), which meant the displayed stream was
        # throttled down to the AI sampling rate instead of the camera's
        # native decode rate. See poll() for how they're applied separately.
        # DETECTION_INTERVAL's old "submit every Nth frame" role is folded
        # into max_detection_fps -- a direct rate is one knob instead of two
        # that multiply together, and this pipeline no longer takes it.
        self._min_display_interval = 1.0 / max_display_fps if max_display_fps > 0 else 0.0
        self._min_detection_interval = 1.0 / max_detection_fps if max_detection_fps > 0 else 0.0
        self._head_count_source = head_count_source

        # Initialize workers. They are lightweight when idle (consuming 0% CPU
        # when no frames are submitted to them). This allows instantaneous runtime toggling
        # between face (YOLO), body (YOLO), and attendance (InsightFace SCRFD + ArcFace) modes.
        #
        # Only workers named in enabled_detectors are actually constructed --
        # a disabled one stays None. head_count_source can still be switched
        # to a disabled mode at runtime (PATCH .../head-count-source); poll()
        # guards every worker access with "is not None" and logs once rather
        # than submitting/reading a worker that was never started.
        started, skipped = [], []

        if "face" in enabled_detectors:
            self._detect_worker = ProcessDetectionWorker(
                FaceDetector, detector_kwargs, return_frame=save_crops
            ).start()
            started.append("face")
        else:
            self._detect_worker = None
            skipped.append("face")

        if "person" in enabled_detectors:
            self._person_worker = ProcessDetectionWorker(
                PersonDetector,
                dict(conf_threshold=person_conf_threshold),
            ).start()
            started.append("person")
        else:
            self._person_worker = None
            skipped.append("person")

        if "attendance" in enabled_detectors:
            self._attendance_worker = ProcessDetectionWorker(
                InsightFaceAttendanceDetector,
                dict(
                    conf_threshold=detector_kwargs.get("conf_threshold", 0.5),
                    det_size=detector_kwargs.get("det_size", 640),
                    onnx_provider=detector_kwargs.get("onnx_provider", "CPUExecutionProvider"),
                ),
                return_frame=save_crops,
            ).start()
            started.append("attendance")
        else:
            self._attendance_worker = None
            skipped.append("attendance")

        # The activity worker is the one detector that is NOT started just
        # because it is enabled: loading YOLOv8x + YOLOv8x-pose costs ~5s
        # and a couple of GB of VRAM, so a camera that will never be sent
        # an activity frame (not in ACTIVITY_CAMERAS, and not displaying
        # activity mode) must not pay for it.
        if "activity" in enabled_detectors and (
            activity_always_on or head_count_source == "activity"
        ):
            self._activity_worker = ProcessDetectionWorker(
                ActivityDetector,
                activity_kwargs or dict(conf_threshold=person_conf_threshold),
            ).start()
            started.append("activity")
        else:
            self._activity_worker = None
            skipped.append("activity")

        self._activity_always_on = activity_always_on and self._activity_worker is not None
        self._min_activity_interval = 1.0 / activity_fps if activity_fps > 0 else 0.0
        self._last_activity_time = 0.0

        if "ocr" in enabled_detectors:
            self._ocr_worker = ProcessDetectionWorker(
                OCRDetector,
                ocr_kwargs or {},
            ).start()
            started.append("ocr")
        else:
            self._ocr_worker = None
            skipped.append("ocr")

        logger.info(
            "Detector workers started: %s -- skipped (ENABLED_DETECTORS): %s",
            started or "none", skipped or "none",
        )
        if self._head_count_source not in started and self._head_count_source not in ("head",):
            logger.warning(
                "head_count_source=%r but its worker wasn't started (not in "
                "ENABLED_DETECTORS=%s) -- that mode will report nothing until "
                "either is changed.",
                self._head_count_source, enabled_detectors,
            )

        self._save_crops = save_crops
        self._crops_dir = crops_dir
        self._max_saved_crops = max_saved_crops
        self._saved_count = 0
        if self._save_crops:
            self._crops_dir.mkdir(parents=True, exist_ok=True)

        self._last_display_time = 0.0
        self._last_detection_time = 0.0
        # process_fps keeps its existing meaning (display/render rate, what
        # the UI's "proc fps" already showed) -- ticked on every returned
        # frame, independent of how often detection actually ran.
        self._process_fps = RollingFps()
        self._last_seen_detection_seq = 0
        self._last_person_seq = 0
        self._last_attendance_seq = 0
        self._last_activity_seq = 0
        self._last_ocr_seq = 0
        self._last_tracked_heads: list[TrackedFace] = []

        # TEMPORARY (latency probe, to be reverted): shadow counter that
        # exactly mirrors whichever worker's internal ProcessDetectionWorker
        # _submit_seq, since submit() is only ever called from here, in
        # order -- lets us map a DetectionResult.sequence back to when that
        # frame was captured/pulled/submitted without touching
        # process_detector.py at all.
        self._probe_seq = 0
        self._probe_log: dict[int, tuple[float, float, float]] = {}
        self._probe_logger = logging.getLogger("latency_probe")
        self._last_probe_queue_log = 0.0

    def _log_probe(self, sequence: int) -> None:
        """TEMPORARY (latency probe, to be reverted). Logs stage 1-4 deltas
        for one detected frame: capture -> pulled-by-render-loop ->
        submitted-to-detector -> detection-result-returned."""
        entry = self._probe_log.pop(sequence, None)
        if entry is None:
            return
        capture_ts, pull_ts, submit_ts = entry
        result_ts = time.monotonic()
        self._probe_logger.info(
            "detect seq=%d cap_to_pull=%.1fms pull_to_submit=%.1fms submit_to_result=%.1fms cap_to_result=%.1fms",
            sequence,
            (pull_ts - capture_ts) * 1000,
            (submit_ts - pull_ts) * 1000,
            (result_ts - submit_ts) * 1000,
            (result_ts - capture_ts) * 1000,
        )

    @property
    def head_count_source(self) -> str:
        return self._head_count_source

    @head_count_source.setter
    def head_count_source(self, val: str) -> None:
        self._head_count_source = val.lower().strip()

    def poll(self) -> PipelineResult | None:
        """Call this in a loop. Returns None if no new frame / display isn't
        due yet.

        Detection runs in a separate process (see ProcessDetectionWorker) so
        it can never block this from returning the latest captured frame --
        the render loop stays smooth no matter how long or how variable a
        detection pass is. Display and detection are also rate-limited
        independently of each other (max_display_fps / max_detection_fps,
        Phase 2.1): the detection gate is evaluated first and unconditionally,
        so a slower display rate can never starve it, and the display gate's
        early return below can no longer suppress a detection submission that
        was due on this same call.
        """
        pull_ts = time.monotonic()  # PROBE: stage 2 -- pulled by render loop
        frame_obj: Frame | None = self._stream.get_latest_frame()
        if frame_obj is None:
            return None

        now = pull_ts

        if now - self._last_detection_time >= self._min_detection_interval:
            self._last_detection_time = now
            # Optimize: Only submit frames to the active worker.
            # The inactive workers will stay idle, preventing duplicate inference and lag.
            # A worker is None when its name wasn't in ENABLED_DETECTORS at
            # startup (see __init__) -- guarded here so switching
            # head_count_source to a disabled mode at runtime can't crash.
            active_worker = None
            if self._head_count_source == "face" and self._detect_worker is not None:
                active_worker = self._detect_worker
            elif self._head_count_source in ("person", "head") and self._person_worker is not None:
                active_worker = self._person_worker
            elif self._head_count_source == "attendance" and self._attendance_worker is not None:
                active_worker = self._attendance_worker
            elif self._head_count_source == "activity" and self._activity_worker is not None:
                active_worker = self._activity_worker
            elif self._head_count_source == "ocr" and self._ocr_worker is not None:
                active_worker = self._ocr_worker

            if active_worker is not None:
                # PROBE (to be reverted): sample queue depth ~1/s, before
                # submit() drains it, so a full-before-drain reading shows
                # up as depth=1 (max possible, given maxsize=1) rather than
                # always reading 0 right after the drain-and-replace.
                if pull_ts - self._last_probe_queue_log >= 1.0:
                    self._last_probe_queue_log = pull_ts
                    try:
                        depth = active_worker._frame_queue.qsize()
                    except NotImplementedError:
                        depth = -1  # qsize() unsupported on this platform
                    self._probe_logger.info("frame_queue depth=%d (maxsize=1)", depth)

                active_worker.submit(frame_obj.image)
                submit_ts = time.monotonic()  # PROBE: stage 3
                self._probe_seq += 1
                self._probe_log[self._probe_seq] = (frame_obj.timestamp, pull_ts, submit_ts)
                if len(self._probe_log) > 50:  # bound memory if results stop arriving
                    oldest = min(self._probe_log)
                    del self._probe_log[oldest]

        # Activity runs on its own clock, concurrently with the primary
        # detector above. Gated separately (and unconditionally, like the
        # detection gate) so the two rates never interfere: identity wants
        # frequent face passes, activity states hold for minutes and are
        # far more expensive per frame.
        if (
            self._activity_always_on
            and self._head_count_source != "activity"
            and now - self._last_activity_time >= self._min_activity_interval
        ):
            self._last_activity_time = now
            self._activity_worker.submit(frame_obj.image)

        if now - self._last_display_time < self._min_display_interval:
            return None
        self._last_display_time = now

        tracked_faces: list[TrackedFace] = []
        tracked_heads: list[TrackedFace] = []
        tracked_activity: list[TrackedFace] = []
        tracked_ocr: list[TrackedFace] = []
        ran_detection = False
        ran_activity = False

        if self._head_count_source == "face" and self._detect_worker is not None:
            detection = self._detect_worker.latest_result()
            ran_detection = detection.sequence != self._last_seen_detection_seq
            if ran_detection:
                self._last_seen_detection_seq = detection.sequence
                self._log_probe(detection.sequence)
                if self._save_crops and detection.frame is not None:
                    self._maybe_save_crops(detection.frame, detection.tracked_faces)
            tracked_faces = detection.tracked_faces
        elif self._head_count_source in ("person", "head") and self._person_worker is not None:
            person_result = self._person_worker.latest_result()
            ran_detection = person_result.sequence != self._last_person_seq
            if ran_detection:
                self._last_person_seq = person_result.sequence
            tracked_heads = person_result.tracked_faces
        elif self._head_count_source == "attendance" and self._attendance_worker is not None:
            att_result = self._attendance_worker.latest_result()
            ran_detection = att_result.sequence != self._last_attendance_seq
            if ran_detection:
                self._last_attendance_seq = att_result.sequence
                self._log_probe(att_result.sequence)
                if self._save_crops and att_result.frame is not None:
                    self._maybe_save_crops(att_result.frame, att_result.tracked_faces)
            tracked_faces = att_result.tracked_faces
        elif self._head_count_source == "activity" and self._activity_worker is not None:
            activity_result = self._activity_worker.latest_result()
            ran_detection = activity_result.sequence != self._last_activity_seq
            if ran_detection:
                self._last_activity_seq = activity_result.sequence
            ran_activity = ran_detection
            tracked_activity = activity_result.tracked_faces
        elif self._head_count_source == "ocr" and self._ocr_worker is not None:
            ocr_result = self._ocr_worker.latest_result()
            ran_detection = ocr_result.sequence != self._last_ocr_seq
            if ran_detection:
                self._last_ocr_seq = ocr_result.sequence
            tracked_ocr = ocr_result.tracked_faces

        # Concurrent activity results, read regardless of which mode is
        # driving the display/head-count path above.
        if self._activity_always_on and self._head_count_source != "activity":
            activity_result = self._activity_worker.latest_result()
            ran_activity = activity_result.sequence != self._last_activity_seq
            if ran_activity:
                self._last_activity_seq = activity_result.sequence
            tracked_activity = activity_result.tracked_faces

        self._process_fps.tick()
        latency_ms = max(0.0, (time.monotonic() - frame_obj.timestamp) * 1000)

        return PipelineResult(
            frame=frame_obj.image,
            tracked_faces=tracked_faces,
            tracked_heads=tracked_heads,
            connection_state=self._stream.state,
            resolution=self._stream.resolution,
            capture_fps=self._stream.measured_fps,
            process_fps=self._process_fps.fps,
            latency_ms=latency_ms,
            ran_detection=ran_detection,
            ran_activity=ran_activity,
            tracked_activity=tracked_activity,
            tracked_ocr=tracked_ocr,
            frame_captured_ts=frame_obj.timestamp,
        )

    def close(self) -> None:
        for worker in (
            self._detect_worker, self._person_worker, self._attendance_worker,
            self._activity_worker, self._ocr_worker,
        ):
            if worker is not None:
                worker.stop()

    def _maybe_save_crops(self, frame: np.ndarray, tracked: list[TrackedFace]) -> None:
        for t in tracked:
            if self._saved_count >= self._max_saved_crops:
                return
            crop = crop_face(frame, t.face.bbox)
            if crop.size == 0:
                continue
            path = self._crops_dir / f"face_{t.track_id}_{self._saved_count:03d}.jpg"
            cv2.imwrite(str(path), crop)
            self._saved_count += 1
            logger.info("Saved sample face crop: %s", path)
