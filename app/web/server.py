"""FastAPI web backend -- same pipeline as main.py, browser-viewable instead
of a desktop cv2 window.

Nothing about capture/detection/tracking changes: each camera still gets its
own RTSPStream (capture thread) and FacePipeline (detection in its own OS
process). The only new piece is a per-camera render thread that encodes the
latest annotated frame to JPEG and holds it in memory, which the MJPEG
endpoint below just reads and streams -- the exact same "always the latest,
never a backlog" pattern already used for capture and detection.

Run (from the project root, with the venv active):
    uvicorn app.web.server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /api/login                          -> {username, password} -> {token}
    GET  /api/cameras                        -> list of camera status objects
    GET  /api/cameras/{camera_id}/status      -> single camera status
    GET  /api/cameras/{camera_id}/stream      -> MJPEG live video (multipart)
    GET  /api/cameras/{camera_id}/head-counts -> per-minute head count history
    GET  /api/cameras/{camera_id}/recognitions -> /Recognize API call history
    GET  /api/cameras/{camera_id}/activity     -> per-person sitting/phone/computer durations
    GET  /api/cameras/{camera_id}/ocr          -> voted OCR readings (static + moving text)

All routes except /api/login require the demo token (see app/web/auth.py)
either as `Authorization: Bearer <token>` or `?token=` on the URL.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.activity import (
    ActivityRepository,
    CameraActivityAnalyzer,
    Identity,
    load_activity_config,
)
from app.camera.rtsp_stream import RTSPStream
from app.config import AppConfig, CameraConfig, load_config
from app.detection.face_detector import align_face
from app.processing.pipeline import FacePipeline, PipelineResult
from app.tracking.simple_tracker import TrackedFace
from app.ui.display import ACTIVITY_FILTERS, draw_overlay
from app.web.auth import DEMO_TOKEN, check_credentials, require_token
from app.web.recognize_client import recognize_face

# Without this, INFO-level logs (camera startup, RTSP URLs, "serving built
# frontend", ...) are silently dropped under uvicorn -- only WARNING+ shows
# via Python's default last-resort handler. main.py sets this up too; the
# subprocess detection workers configure their own copy independently.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("web")

# How often the MJPEG generator checks for a new frame to push. Independent
# of the camera's actual fps -- just how responsive the stream feels.
_STREAM_PUSH_INTERVAL_S = 1 / 20

# OCR mode: a text region only counts as "confirmed" in the results table
# once this many multi-frame votes agree -- matches TextTracker's own
# min_votes_to_confirm (app/tracking/text_tracker.py) and the overlay
# color threshold (app/ui/display.py), kept as separate small constants
# per module rather than a shared import, same as every other per-module
# threshold in this codebase.
_OCR_MIN_VOTES_TO_CONFIRM = 2

_sessions: dict[str, "CameraWebSession"] = {}
# Shared activity event writer, created in lifespan(). None until then (and
# when ACTIVITY_PERSIST_EVENTS=false), which every call site tolerates.
_activity_repo = None


def _sharpness_score(crop: np.ndarray) -> float:
    """Higher = sharper. Variance of the Laplacian is a standard cheap blur
    metric: a crisp image has strong edges (high-variance second derivative);
    a blurry/motion-smeared one is closer to flat everywhere."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frontalness_score(landmarks: np.ndarray) -> float:
    """1.0 = nose perfectly centered between the eyes (frontal); near 0.0 =
    nose sits right next to one eye (side profile). Landmark order follows
    InsightFace's 5-point convention: [left_eye, right_eye, nose, ...].

    Returns 1.0 (never filters) when landmarks aren't available -- e.g. the
    "yolo" DETECTOR_BACKEND, which doesn't produce them."""
    if landmarks is None or len(landmarks) < 3:
        return 1.0
    left_eye_x, right_eye_x, nose_x = landmarks[0][0], landmarks[1][0], landmarks[2][0]
    d1, d2 = abs(nose_x - left_eye_x), abs(nose_x - right_eye_x)
    if max(d1, d2) < 1e-3:
        return 0.0
    return float(min(d1, d2) / max(d1, d2))

# Every face detected in Face Attendance and sent to the recognition backend
# is saved to the data folder on PC (data/attendance and data/recognized_faces).
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ATTENDANCE_IMAGES_DIR = _DATA_DIR / "attendance"
_RECOGNIZE_IMAGES_DIR = _DATA_DIR / "recognized_faces"

# Cameras added at runtime via POST /api/cameras (as opposed to the ones
# .env defines at startup). Persisted here so they survive a server restart
# without anyone having to touch .env.
_CUSTOM_CAMERAS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cameras.json"


def _load_custom_cameras() -> list[dict]:
    if not _CUSTOM_CAMERAS_PATH.exists():
        return []
    try:
        return json.loads(_CUSTOM_CAMERAS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s -- ignoring", _CUSTOM_CAMERAS_PATH)
        return []


def _save_custom_cameras(cameras: list[dict]) -> None:
    _CUSTOM_CAMERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CUSTOM_CAMERAS_PATH.write_text(json.dumps(cameras, indent=2))


@dataclass
class CameraWebSession:
    id: str
    camera: CameraConfig
    stream: RTSPStream
    pipeline: FacePipeline

    # "face" | "person" | "both" -- see DetectionConfig.head_count_source.
    head_count_source: str = "face"
    show_hud: bool = False
    # What the viewer has chosen to SEE. Purely a rendering choice: both
    # the face pipeline and the activity pipeline keep running either way,
    # so switching views never loses a recognition or resets a timer.
    #   "face"     -> face/attendance boxes
    #   "activity" -> body-movement boxes, narrowed by activity_filter
    view_mode: str = "face"
    # Body-movement sub-view: all | phone | body | monitor | computer
    activity_filter: str = "all"

    # How many one-minute buckets of head-count history to keep in memory
    # per camera (120 = 2 hours). In-memory only, by design -- resets on
    # server restart, same "fake"/demo spirit as the login.
    _MAX_MINUTE_BUCKETS: ClassVar[int] = 120
    # How many /Recognize results to keep in memory per camera before the
    # oldest get dropped. Same in-memory-only, resets-on-restart spirit as
    # the head-count buckets above.
    _MAX_RECOGNITIONS: ClassVar[int] = 200
    # Minimum face size to save and send. Set to 0 so ANY face detected
    # by InsightFace in Face Attendance mode is captured and saved to the data folder.
    _RECOGNIZE_MIN_FACE_PX: ClassVar[int] = 0
    # Instead of sending the instant a track first clears the gates, keep
    # collecting candidate crops and send as soon as one is "sharp enough" --
    # measured sharpness on real captures from this camera clusters ~55-62,
    # so 55 is "good, not waiting for a miracle," not an arbitrary guess.
    _RECOGNIZE_MIN_SHARPNESS: ClassVar[float] = 55.0
    # Minimum _frontalness_score to treat a crop as "frontal enough" -- below
    # this the nose sits too close to one eye, i.e. a side-profile shot.
    _RECOGNIZE_MIN_FRONTALNESS: ClassVar[float] = 0.4
    # Hard cap so a face that never gets sharp (bad angle/lighting the whole
    # time) still gets sent eventually instead of silently never firing.
    _RECOGNIZE_MAX_BUFFER_S: ClassVar[float] = 3.0
    # Activity mode: how many per-person duration entries to keep, and how
    # long to keep one around after its track_id stops appearing (a brief
    # occlusion shouldn't reset the clock, but someone who's actually left
    # shouldn't linger forever either).
    _MAX_ACTIVITY_TRACKS: ClassVar[int] = 200
    _ACTIVITY_GRACE_S: ClassVar[float] = 15.0
    # OCR mode: track_id -> latest voted reading. Kept alive a bit longer
    # than activity tracks after last seen -- text (a sign, a badge) doesn't
    # "leave frame" the way a person's posture state does, so a brief gap
    # in detection passes shouldn't drop it from the table.
    _MAX_OCR_TRACKS: ClassVar[int] = 200
    _OCR_GRACE_S: ClassVar[float] = 20.0

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latest_jpeg: Optional[bytes] = field(default=None, repr=False)
    _latest_result: Optional[PipelineResult] = field(default=None, repr=False)
    # Incremented/decremented by _mjpeg_generator on client connect/disconnect
    # (app/web/server.py). _render_loop skips draw_overlay+imencode (the
    # expensive part of each iteration) while this is 0 -- detection and
    # /Recognize keep running either way, only the encode-for-display work
    # is skipped, since attendance shouldn't depend on someone watching.
    _viewer_count: int = field(default=0, repr=False)
    # TEMPORARY (latency probe, to be reverted): (frame_captured_ts, pull_ts,
    # encode_ts) for whatever's currently in _latest_jpeg, so
    # _mjpeg_generator can log the full display-path timing when it writes
    # those exact bytes out.
    _latest_jpeg_probe: Optional[tuple] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # minute string ("YYYY-MM-DD HH:MM") -> running sum/sample-count for
    # each signal being tracked. avg = sum / samples is "how many are
    # typically present" for that minute -- not a peak, not a unique count.
    _minute_buckets: dict = field(default_factory=dict, repr=False)
    # track_id -> already sent to /Recognize. Track IDs only ever increase
    # (see SimpleIouTracker), so this never needs to shrink or expire.
    _recognized_track_ids: set = field(default_factory=set, repr=False)
    _user_last_punched: dict = field(default_factory=dict, repr=False)
    _PUNCH_COOLDOWN_S: ClassVar[float] = 600.0
    _recognitions: list = field(default_factory=list, repr=False)
    # Activity analysis (app/activity/). None when this camera isn't in
    # ACTIVITY_CAMERAS -- every access below is guarded, so a camera without
    # activity behaves exactly as it did before this layer existed.
    _activity: Optional[CameraActivityAnalyzer] = field(default=None, repr=False)
    # face track_id -> Identity, learned from /Recognize. Read by the
    # activity analyzer to attach a name to the matching PERSON track.
    # Face track ids only ever increase, so this never needs to expire.
    _face_identities: dict = field(default_factory=dict, repr=False)
    # Pose skeleton + movement score on the overlay (ACTIVITY_DEBUG_OVERLAY).
    _activity_skeleton: bool = field(default=False, repr=False)
    _last_skip_log_time: float = field(default=0.0, repr=False)
    # track_id -> {"crop", "score", "first_seen"} while we're still waiting
    # to see if a sharper frame of this person shows up before sending.
    _pending_candidates: dict = field(default_factory=dict, repr=False)
    # Activity mode: track_id -> {"first_seen", "last_updated", "totals"}.
    # The three totals accumulate independently (seconds) -- someone can be
    # sitting AND on the phone at the same moment, they aren't mutually
    # exclusive states.
    _activity_tracks: dict = field(default_factory=dict, repr=False)
    # OCR mode: track_id -> {"text", "confidence", "votes", "confirmed",
    # "moving", "bbox", "first_seen_str", "last_seen_str", "last_updated"}.
    _ocr_tracks: dict = field(default_factory=dict, repr=False)

    def start(self) -> "CameraWebSession":
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name=f"web-render-{self.id}"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Flush open activity intervals before tearing down, so a restart
        # doesn't silently discard time people had already accumulated.
        if self._activity is not None:
            self._activity.close()
        self.stream.stop()
        self.pipeline.close()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_jpeg_with_probe(self) -> tuple[Optional[bytes], Optional[tuple]]:
        """TEMPORARY (latency probe, to be reverted)."""
        with self._lock:
            return self._latest_jpeg, self._latest_jpeg_probe

    def latest_status(self) -> dict:
        with self._lock:
            result = self._latest_result
            show_hud = self.show_hud
            view_mode = self.view_mode
            activity_filter = self.activity_filter
        base = {
            "id": self.id,
            "name": self.camera.name,
            "connection_state": self.stream.state,
            "head_count_source": self.head_count_source,
            "show_hud": show_hud,
            "view_mode": view_mode,
            "activity_filter": activity_filter,
            # Lets the UI hide the body-movement controls on a camera that
            # isn't running activity analysis at all.
            "activity_available": self._activity is not None,
        }
        if result is None:
            return {
                **base,
                "resolution": None,
                "capture_fps": 0.0,
                "process_fps": 0.0,
                "latency_ms": 0.0,
                "faces": 0,
                "heads": 0,
                "activity_people": 0,
                "ocr_regions": 0,
                "ocr_confirmed": 0,
            }
        return {
            **base,
            "connection_state": result.connection_state,
            "resolution": list(result.resolution),
            "capture_fps": round(result.capture_fps, 1),
            "process_fps": round(result.process_fps, 1),
            "latency_ms": round(result.latency_ms, 0),
            "faces": len(result.tracked_faces),
            "heads": len(result.tracked_heads),
            "activity_people": len(result.tracked_activity),
            "ocr_regions": len(result.tracked_ocr),
            "ocr_confirmed": sum(
                1 for t in result.tracked_ocr
                if t.text_votes >= _OCR_MIN_VOTES_TO_CONFIRM and t.text
            ),
        }

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            result = self.pipeline.poll()
            pull_ts = time.monotonic()  # PROBE: stage 2 (display path)
            if result is None:
                time.sleep(0.01)
                continue

            # Read the mode under lock so a concurrent PATCH doesn't cause a
            # half-updated render. The string copy is cheap.
            with self._lock:
                current_source = self.head_count_source
                current_show_hud = self.show_hud
                current_view = self.view_mode
                current_filter = self.activity_filter
                has_viewers = self._viewer_count > 0

            # draw_overlay + imencode are the expensive part of this loop
            # (full-frame draw + JPEG compress) -- skip them entirely when
            # no MJPEG client is attached (_viewer_count, tracked by
            # _mjpeg_generator). Detection/recognition below is NOT gated on
            # this: attendance must keep working whether or not anyone is
            # looking at the live feed. _latest_result still updates either
            # way since /api/cameras status doesn't need the JPEG itself.
            if has_viewers:
                show_activity = self._activity is not None and current_view == "activity"
                frame = draw_overlay(
                    result,
                    camera_name=self.camera.name,
                    head_count_source=current_source,
                    show_hud=current_show_hud,
                    draw_activity=show_activity,
                    activity_skeleton=self._activity_skeleton or current_filter == "debug",
                    activity_filter=current_filter,
                    draw_primary=not show_activity,
                )
                # Display-only quality -- separate from and does not affect
                # the /Recognize crop quality (recognize_client.py, stays 90).
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                encode_ts = time.monotonic()  # PROBE: stage 5
                with self._lock:
                    if ok:
                        self._latest_jpeg = buf.tobytes()
                        self._latest_jpeg_probe = (result.frame_captured_ts, pull_ts, encode_ts)
                    self._latest_result = result
            else:
                with self._lock:
                    self._latest_result = result

            # Only sample on an actual NEW detection pass -- between passes
            # a result is just the carried-over last one, so sampling every
            # rendered frame would repeat the same count at render-fps
            # instead of reflecting how often detection actually ran.
            # Activity analysis runs off its own "new result" flag, not
            # ran_detection: it samples at ACTIVITY_FPS concurrently with
            # whatever mode is selected, so feeding it a carried-over result
            # would double-count time against the last observed state.
            if result.ran_activity and self._activity is not None:
                try:
                    with self._lock:
                        identities = dict(self._face_identities)
                    self._activity.observe(
                        activity_tracks=result.tracked_activity,
                        face_tracks=result.tracked_faces,
                        face_identities=identities,
                    )
                except Exception:
                    # Activity analysis is strictly additive -- a bug here
                    # must never stop capture, recognition or attendance.
                    logger.exception("[%s] Activity analysis failed for this pass", self.camera.name)

            if result.ran_detection:
                if current_source == "attendance":
                    face_count = len(result.tracked_faces)
                    self._record_head_count(face_count=face_count, person_count=None, best_count=face_count)
                    self._maybe_recognize(result.frame, result.tracked_faces)
                elif current_source == "face":
                    face_count = len(result.tracked_faces)
                    self._record_head_count(face_count=face_count, person_count=None, best_count=face_count)
                elif current_source == "activity":
                    self._update_activity(result.tracked_activity)
                elif current_source == "ocr":
                    self._update_ocr(result.tracked_ocr)
                else:
                    person_count = len(result.tracked_heads)
                    self._record_head_count(face_count=None, person_count=person_count, best_count=person_count)

    def _record_head_count(
        self,
        face_count: Optional[int],
        person_count: Optional[int],
        best_count: Optional[int] = None,
    ) -> None:
        minute_key = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._lock:
            bucket = self._minute_buckets.setdefault(
                minute_key,
                {
                    "face_sum": 0, "face_samples": 0,
                    "person_sum": 0, "person_samples": 0,
                    "best_sum": 0, "best_samples": 0,
                },
            )
            if face_count is not None:
                bucket["face_sum"] += face_count
                bucket["face_samples"] += 1
            if person_count is not None:
                bucket["person_sum"] += person_count
                bucket["person_samples"] += 1
            if best_count is not None:
                bucket["best_sum"] += best_count
                bucket["best_samples"] += 1
            while len(self._minute_buckets) > self._MAX_MINUTE_BUCKETS:
                oldest_key = next(iter(self._minute_buckets))
                del self._minute_buckets[oldest_key]

    def _update_activity(self, tracked: list[TrackedFace]) -> None:
        now = time.monotonic()
        with self._lock:
            for t in tracked:
                entry = self._activity_tracks.get(t.track_id)
                if entry is None:
                    entry = {
                        "first_seen_monotonic": now,
                        "first_seen_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "last_updated": now,
                        "totals": {"sitting": 0.0, "phone": 0.0, "computer": 0.0},
                    }
                    self._activity_tracks[t.track_id] = entry

                elapsed = now - entry["last_updated"]
                if t.posture == "sitting":
                    entry["totals"]["sitting"] += elapsed
                if t.phone_in_use:
                    entry["totals"]["phone"] += elapsed
                if t.computer_in_use:
                    entry["totals"]["computer"] += elapsed
                entry["last_updated"] = now

            # Prune tracks that haven't been seen in a while, then cap total
            # count -- oldest (by first_seen) evicted first.
            stale = [
                tid for tid, e in self._activity_tracks.items()
                if now - e["last_updated"] > self._ACTIVITY_GRACE_S
            ]
            for tid in stale:
                del self._activity_tracks[tid]

            while len(self._activity_tracks) > self._MAX_ACTIVITY_TRACKS:
                oldest_id = min(
                    self._activity_tracks,
                    key=lambda tid: self._activity_tracks[tid]["first_seen_monotonic"],
                )
                del self._activity_tracks[oldest_id]

    def activity_summary(self) -> list[dict]:
        """Live per-person activity rows.

        Served by the activity analyzer when this camera has one (richer:
        identity, current activity, movement score, stationary/monitor
        timers). Falls back to the original in-memory totals otherwise, so
        a camera outside ACTIVITY_CAMERAS keeps its previous behaviour.
        """
        if self._activity is not None:
            return self._activity.rows()
        with self._lock:
            rows = []
            for track_id, entry in self._activity_tracks.items():
                totals = entry["totals"]
                rows.append({
                    "track_id": track_id,
                    "first_seen": entry["first_seen_str"],
                    "sitting_seconds": round(totals["sitting"]),
                    "phone_seconds": round(totals["phone"]),
                    "computer_seconds": round(totals["computer"]),
                })
            return rows

    def _update_ocr(self, tracked: list[TrackedFace]) -> None:
        now = time.monotonic()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            for t in tracked:
                entry = self._ocr_tracks.get(t.track_id)
                if entry is None:
                    entry = {"first_seen_str": now_str}
                    self._ocr_tracks[t.track_id] = entry
                entry.update({
                    "track_id": t.track_id,
                    "text": t.text,
                    "confidence": round(t.text_confidence, 3),
                    "votes": t.text_votes,
                    "confirmed": t.text_votes >= _OCR_MIN_VOTES_TO_CONFIRM and bool(t.text),
                    "moving": t.text_moving,
                    "bbox": list(t.face.bbox),
                    "last_seen_str": now_str,
                    "last_updated": now,
                })

            stale = [
                tid for tid, e in self._ocr_tracks.items()
                if now - e["last_updated"] > self._OCR_GRACE_S
            ]
            for tid in stale:
                del self._ocr_tracks[tid]

            while len(self._ocr_tracks) > self._MAX_OCR_TRACKS:
                oldest_id = min(self._ocr_tracks, key=lambda tid: self._ocr_tracks[tid]["last_updated"])
                del self._ocr_tracks[oldest_id]

    def ocr_results(self) -> list[dict]:
        """Latest voted reading per tracked text region, most confident
        confirmed readings first. Anonymous track ids -- reset whenever this
        camera switches away from OCR mode and back."""
        with self._lock:
            rows = [{k: v for k, v in e.items() if k != "last_updated"} for e in self._ocr_tracks.values()]
        rows.sort(key=lambda r: (not r["confirmed"], -r["confidence"]))
        return rows

    def recognitions(self) -> list[dict]:
        with self._lock:
            return list(self._recognitions)

    def _maybe_recognize(self, frame: np.ndarray, tracked: list[TrackedFace]) -> None:
        """Buffer candidate crops for each new face track, keeping only the
        sharpest one seen, and fire a background /Recognize call as soon as
        one clears _RECOGNIZE_MIN_SHARPNESS -- not on every detection pass,
        or a person standing in frame would generate dozens of identical
        requests, and not on the very first passable frame either, since
        that's often mid-stride and motion-blurred. If nothing gets sharp
        enough within _RECOGNIZE_MAX_BUFFER_S, send the best one collected
        anyway rather than never firing at all.

        A track only gets marked done once a call actually fires. If someone
        leaves frame before that happens, whatever best crop was collected
        so far gets sent immediately rather than lost.
        """
        current_ids = {t.track_id for t in tracked}
        for track_id in list(self._pending_candidates):
            if track_id not in current_ids:
                self._finalize_candidate(track_id)

        for t in tracked:
            if t.track_id in self._recognized_track_ids:
                continue

            # Ensure confident face detection before triggering recognition
            if t.face.confidence < 0.45:
                continue

            x1, y1, x2, y2 = t.face.bbox
            w, h = x2 - x1, y2 - y1
            if w < 20 or h < 20:
                continue

            # 45% margin + min 224px gives full head context to /Recognize API;
            # align_face also de-rotates for head tilt using landmarks (falls
            # back to a plain crop when the backend doesn't produce them).
            crop = align_face(frame, t.face.bbox, t.face.landmarks, margin=0.45, min_size=224)
            if crop.size == 0:
                continue

            score = _sharpness_score(crop)
            frontal = _frontalness_score(t.face.landmarks)
            now = time.monotonic()
            pending = self._pending_candidates.get(t.track_id)
            if pending is None:
                pending = {"crop": crop, "score": score, "frontal": frontal, "first_seen": now}
                self._pending_candidates[t.track_id] = pending
            else:
                # A meaningfully more frontal shot always wins (stops a sharp
                # side-profile from beating a slightly blurrier front-facing
                # one); among similarly-frontal candidates, prefer the sharper.
                is_better = (
                    frontal > pending["frontal"] + 0.1
                    or (abs(frontal - pending["frontal"]) <= 0.1 and score > pending["score"])
                )
                if is_better:
                    pending["crop"] = crop
                    pending["score"] = score
                    pending["frontal"] = frontal

            if (
                (
                    pending["frontal"] >= self._RECOGNIZE_MIN_FRONTALNESS
                    and pending["score"] >= self._RECOGNIZE_MIN_SHARPNESS
                )
                or now - pending["first_seen"] >= self._RECOGNIZE_MAX_BUFFER_S
            ):
                self._finalize_candidate(t.track_id)

    def _finalize_candidate(self, track_id: int) -> None:
        pending = self._pending_candidates.pop(track_id, None)
        if pending is None:
            return
        self._recognized_track_ids.add(track_id)
        threading.Thread(
            target=self._recognize_and_record,
            args=(pending["crop"], track_id),
            daemon=True,
            name=f"recognize-{self.id}-{track_id}",
        ).start()

    def _recognize_and_record(self, crop: np.ndarray, track_id: int) -> None:
        timestamp = datetime.now()
        response = recognize_face(crop)
        logger.info("[%s] /Recognize result for track #%s: %s", self.camera.name, track_id, response)

        # Extract employee identification if matched
        matched = False
        user_key = None
        if isinstance(response, dict):
            matched = bool(response.get("match") is True or response.get("success") is True)
            user_key = (
                response.get("EmployeeId")
                or response.get("employee_id")
                or response.get("name")
                or response.get("EmployeeName")
                or response.get("employee_name")
            )

        now_mono = time.monotonic()
        in_cooldown = False
        remaining_m = 10
        if matched and user_key:
            with self._lock:
                last_time = self._user_last_punched.get(user_key)
                if last_time is not None and (now_mono - last_time) < self._PUNCH_COOLDOWN_S:
                    in_cooldown = True
                    remaining_m = max(1, int((self._PUNCH_COOLDOWN_S - (now_mono - last_time) + 59) // 60))
                else:
                    self._user_last_punched[user_key] = now_mono

        if in_cooldown:
            logger.info(
                "[%s] Track #%s matched %s but user is in punch cooldown (%dm remaining). Not saving image.",
                self.camera.name, track_id, user_key, remaining_m,
            )
            response = dict(response)
            response["message"] = f"You're done punching for like {remaining_m} minutes"
            image_path = None
        else:
            image_path = self._save_sent_image(crop, track_id, timestamp)

        event = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": track_id,
            "response": response,
            "image_path": str(image_path) if image_path else None,
        }
        # Hand the identity to the activity layer, keyed by FACE track id.
        # identity_binder then transfers it onto whichever PERSON track
        # contains that face -- activity state belongs to the body track,
        # which survives the head turning away from the camera.
        if matched and user_key and self._activity is not None:
            display_name = None
            if isinstance(response, dict):
                display_name = (
                    response.get("EmployeeName")
                    or response.get("employee_name")
                    or response.get("name")
                )
            with self._lock:
                self._face_identities[track_id] = Identity(
                    person_id=str(user_key),
                    person_name=str(display_name or user_key),
                    confidence=1.0,
                )

        with self._lock:
            self._recognitions.append(event)
            if len(self._recognitions) > self._MAX_RECOGNITIONS:
                self._recognitions.pop(0)

    def _save_sent_image(self, crop: np.ndarray, track_id: int, timestamp: datetime) -> Optional[Path]:
        """Save detected attendance face crop directly into the PC's data/ folder."""
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_track{track_id}.jpg"

        # Save into data/attendance/<camera_id>/
        attendance_camera_dir = _ATTENDANCE_IMAGES_DIR / self.id
        attendance_camera_dir.mkdir(parents=True, exist_ok=True)
        primary_path = attendance_camera_dir / filename
        ok = cv2.imwrite(str(primary_path), crop)

        # Also mirror to data/recognized_faces/<camera_id>/
        rec_camera_dir = _RECOGNIZE_IMAGES_DIR / self.id
        rec_camera_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(rec_camera_dir / filename), crop)

        if not ok:
            logger.warning("[%s] Could not save face image to %s", self.camera.name, primary_path)
            return None
        logger.info("[%s] Saved attendance face image to %s", self.camera.name, primary_path)
        return primary_path

    def head_count_history(self) -> list[dict]:
        with self._lock:
            items = list(self._minute_buckets.items())
            source = self.head_count_source
        rows = []
        for minute, bucket in items:
            row: dict = {"minute": minute}
            if source in ("face", "attendance"):
                # Face/Attendance mode: show face count
                row["avg_faces"] = (
                    round(bucket["face_sum"] / bucket["face_samples"])
                    if bucket["face_samples"] else 0
                )
                row["face_samples"] = bucket["face_samples"]
            else:
                # Body-only mode: show best (max of face+body) as "People"
                row["avg_people"] = (
                    round(bucket["best_sum"] / bucket["best_samples"])
                    if bucket["best_samples"] else 0
                )
                row["person_samples"] = bucket.get("best_samples", 0)
            rows.append(row)
        return rows


def _slugify(name: str, index: int) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or f"camera-{index}"


def _start_camera_session(camera: CameraConfig, config: AppConfig, cam_id: str) -> Optional[CameraWebSession]:
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


    head_count_source = config.detection.head_count_source
    if head_count_source not in ("face", "head", "person", "attendance", "activity", "ocr"):
        logger.warning(
            "[%s] Unknown HEAD_COUNT_SOURCE=%r, falling back to 'face'",
            camera.name, head_count_source,
        )
        head_count_source = "face"

    # Activity analysis for this camera? Only for cameras listed in
    # ACTIVITY_CAMERAS, and only if the activity worker is actually enabled
    # -- YOLOv8x + YOLOv8x-pose is the most expensive pass in the system
    # (~236ms/frame measured at imgsz=960 on an RTX 3060), so it is opt-in
    # per camera rather than on everywhere.
    activity_cfg = load_activity_config()
    activity_enabled = (
        activity_cfg.enabled_for(cam_id)
        and "activity" in config.detection.enabled_detectors
    )
    if activity_cfg.enabled_for(cam_id) and not activity_enabled:
        logger.warning(
            "[%s] ACTIVITY_CAMERAS lists this camera but 'activity' is not in "
            "ENABLED_DETECTORS=%s -- activity analysis is off for it.",
            camera.name, config.detection.enabled_detectors,
        )

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
        head_count_source=head_count_source,
        person_conf_threshold=config.detection.person_conf_threshold,
        ocr_kwargs=asdict(config.ocr),
        enabled_detectors=config.detection.enabled_detectors,
        activity_kwargs=activity_cfg.detector_kwargs(cam_id) if activity_enabled else None,
        activity_always_on=activity_enabled,
        activity_fps=activity_cfg.fps,
    )

    analyzer = None
    if activity_enabled:
        analyzer = CameraActivityAnalyzer(
            camera_id=cam_id, config=activity_cfg, repository=_activity_repo
        )
        logger.info(
            "[%s] Activity analysis enabled (device=%s imgsz=%d fps=%.1f, models=%s + %s)",
            camera.name, activity_cfg.resolved_device(), activity_cfg.img_size,
            activity_cfg.fps, activity_cfg.obj_model.name, activity_cfg.pose_model.name,
        )

    return CameraWebSession(
        id=cam_id,
        camera=camera,
        stream=stream,
        pipeline=pipeline,
        head_count_source=head_count_source,
        _activity=analyzer,
        _activity_skeleton=activity_cfg.debug_overlay,
    ).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _activity_repo
    config = load_config()
    activity_cfg = load_activity_config()
    # One writer shared by every camera's analyzer -- a single queue and
    # thread, not one per camera.
    _activity_repo = ActivityRepository(enabled=activity_cfg.persist_events).start()
    if not config.cameras:
        logger.error(
            "No cameras configured. Set CAMERA_IP/CAMERA_RTSP_URL (single "
            "camera) or CAMERA_1_IP, CAMERA_2_IP, ... (multiple) in .env."
        )
    for i, camera in enumerate(config.cameras, start=1):
        cam_id = _slugify(camera.name, i)
        session = _start_camera_session(camera, config, cam_id)
        if session is not None:
            _sessions[cam_id] = session
            logger.info("Started camera session '%s' (%s)", cam_id, camera.name)

    for entry in _load_custom_cameras():
        camera = CameraConfig(
            name=entry.get("name") or entry["id"],
            ip=entry.get("ip", ""),
            rtsp_port=entry.get("rtsp_port", 554),
            username=entry.get("username", ""),
            password=entry.get("password", ""),
            rtsp_path=entry.get("rtsp_path", "/Streaming/Channels/102"),
            transport=entry.get("transport", "tcp"),
        )
        session = _start_camera_session(camera, config, entry["id"])
        if session is not None:
            _sessions[entry["id"]] = session
            logger.info("Started custom camera session '%s' (%s)", entry["id"], camera.name)

    yield

    for session in _sessions.values():
        session.stop()
    _sessions.clear()
    if _activity_repo is not None:
        _activity_repo.stop()


app = FastAPI(title="CCTV Face Detection API", lifespan=lifespan)

# Dev-friendly default (React dev server on a different port). Tighten this
# to the real frontend origin before deploying anywhere non-local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginRequest) -> dict:
    if not check_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": DEMO_TOKEN}


@app.get("/api/cameras")
def list_cameras(_: None = Depends(require_token)) -> list[dict]:
    return [session.latest_status() for session in _sessions.values()]


class AddCameraRequest(BaseModel):
    name: str = ""
    ip: str
    rtsp_port: int = 554
    username: str = ""
    password: str = ""
    rtsp_path: str = "/Streaming/Channels/102"
    transport: str = "tcp"


class TestCameraRequest(BaseModel):
    ip: str
    rtsp_port: int = 554
    username: str = ""
    password: str = ""
    rtsp_path: str = "/stream1"
    transport: str = "tcp"


@app.post("/api/cameras/test-connection")
def test_camera_connection(body: TestCameraRequest, _: None = Depends(require_token)) -> dict:
    """Test connecting to an RTSP camera stream (e.g. Tapo) without adding it."""
    if not body.ip.strip():
        raise HTTPException(status_code=422, detail="ip is required")
    camera = CameraConfig(
        name="Probe",
        ip=body.ip.strip(),
        rtsp_port=body.rtsp_port,
        username=body.username.strip(),
        password=body.password,
        rtsp_path=body.rtsp_path.strip() or "/stream1",
        transport=body.transport,
    )
    url = camera.rtsp_url
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return {
            "ok": False,
            "message": f"Could not connect to {camera.rtsp_url_masked}. Check IP address, port, and credentials.",
        }
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return {
            "ok": False,
            "message": f"Connected to {camera.rtsp_url_masked}, but failed to read a frame. Verify the stream path.",
        }
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {
        "ok": True,
        "message": f"Connected successfully! {w}x{h} @ {fps:.1f} fps",
        "width": w,
        "height": h,
        "fps": fps,
    }


@app.post("/api/cameras")
def add_camera(body: AddCameraRequest, _: None = Depends(require_token)) -> dict:
    """Add a camera at runtime -- no .env edit or restart required. Persisted
    to data/cameras.json so it comes back on the next server restart too."""
    if not body.ip.strip():
        raise HTTPException(status_code=422, detail="ip is required")

    config = load_config()
    cam_id = f"custom-{uuid.uuid4().hex[:8]}"
    camera = CameraConfig(
        name=body.name.strip() or f"Camera {len(_sessions) + 1}",
        ip=body.ip.strip(),
        rtsp_port=body.rtsp_port,
        username=body.username.strip(),
        password=body.password,
        rtsp_path=body.rtsp_path.strip() or "/Streaming/Channels/102",
        transport=body.transport,
    )

    session = _start_camera_session(camera, config, cam_id)
    if session is None:
        raise HTTPException(status_code=400, detail="Could not start camera -- check ip/credentials")
    _sessions[cam_id] = session

    stored = _load_custom_cameras()
    stored.append({
        "id": cam_id,
        "name": camera.name,
        "ip": camera.ip,
        "rtsp_port": camera.rtsp_port,
        "username": camera.username,
        "password": camera.password,
        "rtsp_path": camera.rtsp_path,
        "transport": camera.transport,
    })
    _save_custom_cameras(stored)

    logger.info("Added camera '%s' (%s) via API", cam_id, camera.name)
    return session.latest_status()


@app.delete("/api/cameras/{camera_id}")
def remove_camera(camera_id: str, _: None = Depends(require_token)) -> dict:
    session = _sessions.pop(camera_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    session.stop()

    stored = [c for c in _load_custom_cameras() if c["id"] != camera_id]
    _save_custom_cameras(stored)

    logger.info("Removed camera '%s'", camera_id)
    return {"camera_id": camera_id, "removed": True}


@app.get("/api/cameras/{camera_id}/status")
def camera_status(camera_id: str, _: None = Depends(require_token)) -> dict:
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return session.latest_status()


@app.get("/api/cameras/{camera_id}/head-counts")
def camera_head_counts(camera_id: str, _: None = Depends(require_token)) -> list[dict]:
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return session.head_count_history()


@app.get("/api/cameras/{camera_id}/recognitions")
def camera_recognitions(camera_id: str, _: None = Depends(require_token)) -> list[dict]:
    """Results of every /Recognize call made for this camera (one per newly
    appeared face track), most recent last."""
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return session.recognitions()


@app.get("/api/cameras/{camera_id}/activity")
def camera_activity(camera_id: str, _: None = Depends(require_token)) -> list[dict]:
    """Per-person sitting/phone/computer duration totals (Activity mode).
    Anonymous track ids -- reset whenever this camera switches away from
    Activity mode and back."""
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return session.activity_summary()


@app.get("/api/cameras/{camera_id}/activity/events")
def camera_activity_events(
    camera_id: str, limit: int = 100, _: None = Depends(require_token)
) -> list[dict]:
    """Live in-memory activity event timeline for this camera, oldest first.

    This is the "what happened just now" feed (survives only as long as the
    process). For history across restarts use /api/activity/events, which
    reads the database.
    """
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if session._activity is None:
        return []
    return session._activity.recent_events(limit=limit)


@app.get("/api/activity/events")
def activity_events(
    camera_id: Optional[str] = None,
    person_id: Optional[str] = None,
    activity_type: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 500,
    _: None = Depends(require_token),
) -> list[dict]:
    """Stored activity events, newest first. `date` is YYYY-MM-DD (UTC).

    Overlapping rows for one track are expected: SITTING, STATIONARY,
    PHONE_INTERACTION, MONITOR_INTERACTION and COMPUTER_INTERACTION are
    independent streams, each with its own start/end/duration.
    """
    try:
        return ActivityRepository.query_events(
            camera_id=camera_id, person_id=person_id,
            activity_type=activity_type, date=date, limit=limit,
        )
    except Exception as exc:
        logger.error("Activity event query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Activity history unavailable")


@app.get("/api/activity/summary")
def activity_summary(
    camera_id: Optional[str] = None,
    date: Optional[str] = None,
    _: None = Depends(require_token),
) -> list[dict]:
    """Per-person totals per activity type over the stored events."""
    try:
        return ActivityRepository.summary(camera_id=camera_id, date=date)
    except Exception as exc:
        logger.error("Activity summary query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Activity history unavailable")


@app.get("/api/cameras/{camera_id}/ocr")
def camera_ocr(camera_id: str, _: None = Depends(require_token)) -> list[dict]:
    """Latest voted OCR reading per tracked text region (OCR mode): static
    text (walls/signs/boards) and moving text (carried by tracked people/
    objects), each with confidence + how many multi-frame votes agree."""
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return session.ocr_results()


class HeadCountSourceRequest(BaseModel):
    source: str  # "face" | "head" | "person" | "attendance" | "activity"


@app.patch("/api/cameras/{camera_id}/head-count-source")
def set_head_count_source(
    camera_id: str,
    body: HeadCountSourceRequest,
    _: None = Depends(require_token),
) -> dict:
    """Change the detection mode for a camera at runtime."""
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    source = body.source.lower().strip()
    if source not in ("face", "head", "person", "attendance", "activity", "ocr"):
        raise HTTPException(
            status_code=422,
            detail="source must be one of: face, head, person, attendance, activity, ocr",
        )
    # Update the session's head_count_source and update the pipeline's active source
    with session._lock:
        session.head_count_source = source
        session.pipeline.head_count_source = source
    logger.info("[%s] head_count_source changed to %r", session.camera.name, source)
    return {"camera_id": camera_id, "head_count_source": source}


class ShowHudRequest(BaseModel):
    show_hud: bool


@app.patch("/api/cameras/{camera_id}/show-hud")
def set_show_hud(
    camera_id: str,
    body: ShowHudRequest,
    _: None = Depends(require_token),
) -> dict:
    """Toggle showing the HUD on the video stream."""
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    with session._lock:
        session.show_hud = body.show_hud
    logger.info("[%s] show_hud changed to %r", session.camera.name, body.show_hud)
    return {"camera_id": camera_id, "show_hud": body.show_hud}


class ViewModeRequest(BaseModel):
    # "face" | "activity"
    view_mode: Optional[str] = None
    # "all" | "phone" | "body" | "monitor" | "computer"
    activity_filter: Optional[str] = None


@app.patch("/api/cameras/{camera_id}/view-mode")
def set_view_mode(
    camera_id: str,
    body: ViewModeRequest,
    _: None = Depends(require_token),
) -> dict:
    """Choose what the live overlay shows: faces, or body movement.

    Rendering only. Both pipelines keep running whichever view is selected,
    so switching never drops a recognition or resets an activity timer --
    the numbers behind the Activity table are identical either way.
    """
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if body.view_mode is not None:
        view_mode = body.view_mode.lower().strip()
        if view_mode not in ("face", "activity"):
            raise HTTPException(status_code=422, detail="view_mode must be 'face' or 'activity'")
        if view_mode == "activity" and session._activity is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Activity analysis is not enabled for this camera. "
                    "Add its id to ACTIVITY_CAMERAS in .env and restart."
                ),
            )
        with session._lock:
            session.view_mode = view_mode

    if body.activity_filter is not None:
        activity_filter = body.activity_filter.lower().strip()
        if activity_filter not in ACTIVITY_FILTERS:
            raise HTTPException(
                status_code=422,
                detail=f"activity_filter must be one of: {', '.join(ACTIVITY_FILTERS)}",
            )
        with session._lock:
            session.activity_filter = activity_filter

    with session._lock:
        current = {"view_mode": session.view_mode, "activity_filter": session.activity_filter}
    logger.info("[%s] view changed to %r/%r", session.camera.name,
                current["view_mode"], current["activity_filter"])
    return {"camera_id": camera_id, **current}


_probe_logger = logging.getLogger("latency_probe")  # TEMPORARY, to be reverted


def _mjpeg_generator(session: CameraWebSession):
    boundary = b"frame"
    with session._lock:
        session._viewer_count += 1
    last_logged_probe = None
    try:
        while True:
            jpeg, probe = session.latest_jpeg_with_probe()  # PROBE (to be reverted)
            if jpeg is not None:
                write_ts = time.monotonic()  # PROBE: stage 6
                if probe is not None and probe != last_logged_probe:
                    last_logged_probe = probe
                    capture_ts, pull_ts, encode_ts = probe
                    _probe_logger.info(
                        "display cap_to_pull=%.1fms pull_to_encode=%.1fms encode_to_write=%.1fms cap_to_write=%.1fms",
                        (pull_ts - capture_ts) * 1000,
                        (encode_ts - pull_ts) * 1000,
                        (write_ts - encode_ts) * 1000,
                        (write_ts - capture_ts) * 1000,
                    )
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            time.sleep(_STREAM_PUSH_INTERVAL_S)
    finally:
        # Client disconnected (browser closed/navigated away) -- runs via
        # GeneratorExit either way, so _render_loop resumes skipping the
        # encode as soon as the last viewer leaves.
        with session._lock:
            session._viewer_count = max(0, session._viewer_count - 1)


@app.get("/api/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, _: None = Depends(require_token)) -> StreamingResponse:
    session = _sessions.get(camera_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_generator(session),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- Serve the built React app -----------------------------------------------
# `npm run build` (in frontend/) produces frontend/dist. Mounting it here,
# AFTER every /api/* route is registered, means this server alone -- no
# separate Vite dev server -- serves the whole app on one port. Mount order
# matters: Starlette matches routes in registration order, so the specific
# /api/* paths above always win over this catch-all; only requests that
# don't match anything above (index.html, JS/CSS bundles, ...) fall through
# to it.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info("Serving built frontend from %s", _FRONTEND_DIST)
else:
    logger.warning(
        "%s not found -- run `npm run build` in frontend/ to serve the web "
        "app from this server too. API routes still work without it.",
        _FRONTEND_DIST,
    )
