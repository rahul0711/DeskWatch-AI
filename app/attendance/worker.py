"""Steps 4, 6, 7 wired into one live recognition worker: SCRFD detection +
ByteTrack tracking (step 4) feed the best-frame buffer (step 5); when a
track is lost (or has lingered too long), its best crop is embedded --
batched with any other tracks finalizing in the same tick (step 6) -- and
searched against FAISS (step 7). A match above the threshold writes one
attendance row; the track ID is added to finalized_ids so it is never
processed again, satisfying "recognize once per person."

Runs on its own thread (RTSP capture + SCRFD + ByteTrack are all
synchronous); publishes events to app.attendance.events.event_bus, which a
small asyncio task in the FastAPI app fans out to WebSocket clients.
"""
from __future__ import annotations

import base64
import collections
import datetime as dt
import logging
import queue
import threading
import time
import uuid

import cv2
import numpy as np
import supervision as sv
from sqlalchemy import select

from app.attendance.best_frame import TrackBuffer, TrackBufferStore
from app.attendance.config import AttendanceConfig
from app.attendance.db import AttendanceEvent, User, get_session, to_utc_iso
from app.attendance.detector import DetectedFace, ScrfdDetector
from app.attendance.embedder import AdaFaceEmbedder
from app.attendance.events import event_bus
from app.attendance.faiss_index import FaceIndex
from app.camera.rtsp_stream import RTSPStream
from app.config import load_config

logger = logging.getLogger(__name__)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_face_to_box(faces: list[DetectedFace], box: np.ndarray, iou_threshold: float = 0.3) -> DetectedFace | None:
    best, best_iou = None, 0.0
    for f in faces:
        iou = _iou(f.bbox, box)
        if iou > best_iou:
            best_iou, best = iou, f
    return best if best_iou >= iou_threshold else None


class RecognitionWorker:
    def __init__(
        self,
        cfg: AttendanceConfig,
        detector: ScrfdDetector,
        embedder: AdaFaceEmbedder,
        index: FaceIndex,
    ) -> None:
        self._cfg = cfg
        self._detector = detector
        self._embedder = embedder
        self._index = index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None
        self._finalize_thread: threading.Thread | None = None
        self._stream: RTSPStream | None = None
        self._camera_name = "camera"

        # Finalizing a track (embed + FAISS search + SQLite write + saving
        # the crop to disk) used to run inline in the main capture loop --
        # meaning every time someone's track ended, that work blocked the
        # very loop responsible for feeding the broadcast buffer, right
        # when there was actually someone to show. Queued here and
        # processed on its own thread so recognition latency can never
        # stall the live view.
        self._finalize_queue: queue.Queue[list[tuple[int, TrackBuffer]]] = queue.Queue()

        # Backlog of pre-rendered (already boxed+resized+JPEG-encoded)
        # frames waiting to be sent to the dashboard, filled at whatever
        # rate the camera actually delivers (decoupled from the detection
        # throttle) and drained by _broadcast_loop at a steady
        # frame_broadcast_fps. This is what lets the live view keep
        # advancing through the camera's periodic keyframe stall instead of
        # visibly freezing -- see config.py's broadcast_buffer_seconds
        # docstring for the latency trade-off.
        #
        # Bounded by WALL-CLOCK TIME (trimmed in _enqueue_broadcast_frame),
        # not a fixed frame count -- an earlier version capped this at a
        # frame count sized for an assumed 25fps, but the camera's real
        # average throughput (accounting for its own stalls) runs lower
        # than that, so the same frame-count cap was quietly holding several
        # times more backlog -- and therefore several times more latency --
        # than intended. Time-based trimming is correct regardless of the
        # camera's actual fps.
        # maxlen is a structural backstop only (should never actually bind --
        # the age-based trim in both _enqueue_broadcast_frame and
        # _broadcast_loop is what normally bounds this); generous enough
        # (~4s at a busy 30fps) that it only kicks in if something upstream
        # ever stops draining entirely.
        self._broadcast_buffer: collections.deque[tuple[float, str]] = collections.deque(maxlen=120)
        self._broadcast_lock = threading.Lock()
        self._last_boxes: list[dict] = []

        # Maps user_id -> monotonic timestamp of their most recent punch.
        # Enforces punch_cooldown_seconds (default 10 min) so repeat detections
        # do not write duplicate database events or save redundant face crops.
        self._user_last_punched: dict[int, float] = {}
        self._cooldown_lock = threading.Lock()

        # Tracks unknown face crop paths with monotonic creation time: (mono_ts, Path).
        # Expired after unknown_face_ttl_seconds (default 30s) and removed from disk.
        self._unknown_crops: collections.deque[tuple[float, Path]] = collections.deque()
        self._unknown_crops_lock = threading.Lock()

    def _cleanup_expired_unknowns(self) -> None:
        """Removes unknown face crops from disk that have exceeded unknown_face_ttl_seconds (30s)."""
        now = time.monotonic()
        to_delete: list[Path] = []
        with self._unknown_crops_lock:
            while self._unknown_crops and (now - self._unknown_crops[0][0]) >= self._cfg.unknown_face_ttl_seconds:
                _, path = self._unknown_crops.popleft()
                to_delete.append(path)

        for p in to_delete:
            try:
                if p.exists():
                    p.unlink(missing_ok=True)
                    logger.info("Removed expired unknown face crop (>%0.1fs): %s", self._cfg.unknown_face_ttl_seconds, p.name)
            except Exception as exc:
                logger.warning("Failed to delete expired unknown crop %s: %s", p, exc)

    def _cleanup_orphaned_unknowns(self) -> None:
        """Cleans up any legacy unknown crops on disk older than unknown_face_ttl_seconds on startup."""
        now = time.time()
        try:
            if self._cfg.attendance_crops_dir.exists():
                for f in self._cfg.attendance_crops_dir.glob("unknown_*.jpg"):
                    if now - f.stat().st_mtime >= self._cfg.unknown_face_ttl_seconds:
                        f.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Error cleaning orphaned unknown crops: %s", exc)

    def _get_last_punch_time(self, user_id: int) -> float | None:
        """Returns monotonic timestamp of user's last punch, or None if not recent/known."""
        with self._cooldown_lock:
            if user_id in self._user_last_punched:
                return self._user_last_punched[user_id]

        # Check DB on cache miss (e.g. after server restart)
        session = get_session()
        try:
            last_event = session.execute(
                select(AttendanceEvent)
                .where(AttendanceEvent.user_id == user_id)
                .order_by(AttendanceEvent.timestamp.desc())
                .limit(1)
            ).scalars().first()
            if last_event is not None and last_event.timestamp is not None:
                ts = last_event.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                now_utc = dt.datetime.now(dt.timezone.utc)
                elapsed_s = (now_utc - ts).total_seconds()
                if 0 <= elapsed_s <= self._cfg.punch_cooldown_seconds:
                    mono_ts = time.monotonic() - elapsed_s
                    with self._cooldown_lock:
                        self._user_last_punched[user_id] = mono_ts
                    return mono_ts
        except Exception:
            logger.exception("Failed checking last punch time in DB for user %d", user_id)
        finally:
            session.close()

        return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="recognition-worker")
        self._thread.start()
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True, name="broadcast-drain")
        self._broadcast_thread.start()
        self._finalize_thread = threading.Thread(target=self._finalize_worker_loop, daemon=True, name="finalize-worker")
        self._finalize_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._broadcast_thread is not None:
            self._broadcast_thread.join(timeout=5)
        self._finalize_queue.put(None)  # wake the worker so it can see _stop_event
        if self._finalize_thread is not None:
            self._finalize_thread.join(timeout=5)

    def _finalize_worker_loop(self) -> None:
        self._cleanup_orphaned_unknowns()
        while not self._stop_event.is_set():
            try:
                batch = self._finalize_queue.get(timeout=1.0)
            except queue.Empty:
                self._cleanup_expired_unknowns()
                continue
            if batch is None:
                continue
            try:
                self._finalize_batch(batch)
            except Exception:
                logger.exception("Finalize batch failed for tracks %s", [tid for tid, _ in batch])
            finally:
                self._cleanup_expired_unknowns()

    def _run(self) -> None:
        app_cfg = load_config()
        idx = self._cfg.camera_index - 1
        if idx < 0 or idx >= len(app_cfg.cameras):
            logger.error(
                "ATTENDANCE_CAMERA_INDEX=%d out of range (%d camera(s) configured) -- "
                "recognition worker not started", self._cfg.camera_index, len(app_cfg.cameras),
            )
            return
        cam = app_cfg.cameras[idx]
        self._camera_name = cam.name
        logger.info("Recognition worker starting on camera '%s' (%s)", cam.name, cam.rtsp_url_masked)

        tracker = sv.ByteTrack(
            track_activation_threshold=0.4,
            lost_track_buffer=int(self._cfg.detection_fps * 3),
            minimum_matching_threshold=0.8,
            frame_rate=self._cfg.detection_fps,
            minimum_consecutive_frames=2,
        )
        buffers = TrackBufferStore()
        finalized_ids: set[int] = set()
        active_ids_prev: set[int] = set()

        self._stream = RTSPStream(
            rtsp_url=cam.rtsp_url,
            transport=cam.transport,
            rtsp_url_masked=cam.rtsp_url_masked,
            name=cam.name,
        ).start()

        min_detect_interval = 1.0 / self._cfg.detection_fps
        last_detect_time = 0.0
        last_detect_frame_index = -1
        last_seen_frame_index = -1

        while not self._stop_event.is_set():
            frame = self._stream.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            # Bank every genuinely new raw frame for the dashboard's live
            # view at (up to) native camera fps, using whatever detection
            # boxes were most recently computed -- independent of the
            # detection throttle below, so the broadcast buffer fills fast
            # enough to cover the camera's periodic stall.
            if frame.frame_index != last_seen_frame_index:
                last_seen_frame_index = frame.frame_index
                self._enqueue_broadcast_frame(frame.image, self._last_boxes)

            now = time.monotonic()
            if frame.frame_index == last_detect_frame_index or (now - last_detect_time) < min_detect_interval:
                time.sleep(0.005)
                continue
            last_detect_frame_index = frame.frame_index
            last_detect_time = now

            faces = self._detector.detect(frame.image)
            if faces:
                xyxy = np.array([f.bbox for f in faces], dtype=np.float32)
                confidence = np.array([f.det_score for f in faces], dtype=np.float32)
                detections = sv.Detections(xyxy=xyxy, confidence=confidence)
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)

            current_ids: set[int] = set()
            boxes_for_broadcast = []
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                tid = int(tid)
                current_ids.add(tid)
                face = _match_face_to_box(faces, box)
                if face is not None:
                    buffers.update(tid, frame.image, face)
                boxes_for_broadcast.append({
                    "track_id": tid,
                    "bbox": [int(v) for v in box],
                    "width_px": int(box[2] - box[0]),
                })
            self._last_boxes = boxes_for_broadcast

            # Collect every track that needs finalizing THIS tick -- tracks
            # that just disappeared (ByteTrack's own lost_track_buffer
            # already gives brief occlusions a few frames of grace before a
            # track actually vanishes from `tracked`), plus any that have
            # lingered past max_track_age_s. When a group walks in together
            # and several people leave frame around the same tick, this is
            # what lets step 6's batching actually kick in -- see
            # _finalize_batch below.
            to_finalize: list[tuple[int, TrackBuffer]] = []

            for tid in active_ids_prev - current_ids:
                buf = buffers.pop(tid)
                if tid not in finalized_ids and buf is not None and buf.is_ready:
                    to_finalize.append((tid, buf))
                finalized_ids.add(tid)

            for tid in current_ids - finalized_ids:
                buf = buffers.get(tid)
                if buf is not None and buf.is_ready and buf.age_seconds >= self._cfg.max_track_age_s:
                    to_finalize.append((tid, buf))
                    finalized_ids.add(tid)

            if to_finalize:
                self._finalize_queue.put(to_finalize)

            active_ids_prev = current_ids

        logger.info("Recognition worker stopped")

    def _finalize_batch(self, ready: list[tuple[int, TrackBuffer]]) -> None:
        """Step 6, for real: every track finalizing in this tick is embedded
        in ONE session.run() call, not one-at-a-time -- this is what makes a
        group arriving together (several people leaving frame in the same
        tick) cost one batched GPU call instead of N sequential ones."""
        cfg = self._cfg
        crops = [buf.best_aligned_crop for _, buf in ready]
        embeddings = self._embedder.embed_aligned_batch(crops)

        if len(ready) > 1:
            logger.info("Batch-finalizing %d tracks in one embed call: ids=%s", len(ready), [tid for tid, _ in ready])

        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        now_mono = time.monotonic()

        for (track_id, buf), embedding in zip(ready, embeddings):
            match = self._index.best_match(embedding)
            low_conf = buf.best_face_width_px < cfg.low_confidence_face_px
            logger.info(
                "Track %d finalized: width=%dpx det_score=%.2f match_score=%s low_conf=%s",
                track_id, buf.best_face_width_px, buf.best_det_score,
                f"{match[1]:.3f}" if match else "none", low_conf,
            )

            if match is not None and match[1] >= cfg.match_threshold:
                user_id, score = match
                last_punch = self._get_last_punch_time(user_id)
                if last_punch is not None and (now_mono - last_punch) < cfg.punch_cooldown_seconds:
                    remaining_s = int(cfg.punch_cooldown_seconds - (now_mono - last_punch))
                    remaining_m = max(1, (remaining_s + 59) // 60)
                    session = get_session()
                    try:
                        user = session.get(User, user_id)
                        name = user.name if user else f"User #{user_id}"
                        emp_id = user.employee_id if user else "?"
                    finally:
                        session.close()

                    logger.info(
                        "User %s (%s, id=%d) recognized but already punched within cooldown window (%ds / %dm remaining). "
                        "Suppressing punch and crop save.",
                        name, emp_id, user_id, remaining_s, remaining_m,
                    )
                    event_bus.publish({
                        "type": "cooldown",
                        "track_id": track_id,
                        "user_id": user_id,
                        "name": name,
                        "employee_id": emp_id,
                        "message": f"You're done punching for like {remaining_m} minutes",
                        "spoken_message": f"{name}, you're done punching",
                        "remaining_seconds": remaining_s,
                        "remaining_minutes": remaining_m,
                        "timestamp": now_iso,
                    })
                    # DO NOT SAVE CROP IMAGE TO DISK!
                    # DO NOT SAVE ATTENDANCE TO DATABASE!
                    continue

                # Not in cooldown -> save crop and write attendance event
                cfg.attendance_crops_dir.mkdir(parents=True, exist_ok=True)
                crop_path = cfg.attendance_crops_dir / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(crop_path), buf.best_aligned_crop)
                thumb_url = f"/data/{crop_path.relative_to(cfg.attendance_crops_dir.parent).as_posix()}"

                session = get_session()
                try:
                    user = session.get(User, user_id)
                    event = AttendanceEvent(
                        user_id=user_id,
                        camera_name=self._camera_name,
                        confidence=score,
                        face_width_px=buf.best_face_width_px,
                        low_confidence=low_conf,
                        thumbnail_path=str(crop_path),
                    )
                    session.add(event)
                    session.commit()
                    session.refresh(event)
                    with self._cooldown_lock:
                        self._user_last_punched[user_id] = now_mono

                    event_bus.publish({
                        "type": "attendance",
                        "track_id": track_id,
                        "user_id": user_id,
                        "name": user.name if user else "?",
                        "employee_id": user.employee_id if user else "?",
                        "timestamp": to_utc_iso(event.timestamp),
                        "confidence": score,
                        "face_width_px": buf.best_face_width_px,
                        "low_confidence": low_conf,
                        "thumbnail_url": thumb_url,
                        "message": f"Punch recorded for {user.name if user else 'user'}!",
                    })
                finally:
                    session.close()
            else:
                # Unknown face -> save temporary crop for enrollment (removed after unknown_face_ttl_seconds)
                cfg.attendance_crops_dir.mkdir(parents=True, exist_ok=True)
                crop_path = cfg.attendance_crops_dir / f"unknown_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(crop_path), buf.best_aligned_crop)
                with self._unknown_crops_lock:
                    self._unknown_crops.append((now_mono, crop_path))
                thumb_url = f"/data/{crop_path.relative_to(cfg.attendance_crops_dir.parent).as_posix()}"

                event_bus.publish({
                    "type": "unknown",
                    "track_id": track_id,
                    "timestamp": now_iso,
                    "face_width_px": buf.best_face_width_px,
                    "low_confidence": low_conf,
                    "best_score": match[1] if match else None,
                    "thumbnail_url": thumb_url,
                    "ttl_seconds": self._cfg.unknown_face_ttl_seconds,
                })

    # Detection/tracking/embedding all run on the full-res frame -- this cap
    # only shrinks the copy sent to the dashboard, which was previously
    # ~740KB/frame at full 1920x1080 (5.9MB/s at 8fps broadcast rate, enough
    # to make the live view itself feel laggy on top of any camera-side
    # Scaled to 1280 for high clarity in the expanded 75% camera view layout.
    _BROADCAST_MAX_WIDTH = 1280

    def _enqueue_broadcast_frame(self, image_bgr: np.ndarray, boxes: list[dict]) -> None:
        """Draw + resize + JPEG-encode one frame and bank it in the backlog.
        Called at (up to) native camera fps -- decoupled from both the
        detection throttle and the broadcast drain rate -- so the buffer
        has enough banked to smooth over the camera's periodic stall. Cheap
        enough at 960px/quality 70 to run this often given the GPU is doing
        the actual detection work, not the CPU encoding this preview."""
        img = image_bgr.copy()
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            low_conf = b["width_px"] < self._cfg.low_confidence_face_px
            color = (0, 165, 255) if low_conf else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"id={b['track_id']} w={b['width_px']}px"
            cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        h, w = img.shape[:2]
        if w > self._BROADCAST_MAX_WIDTH:
            scale = self._BROADCAST_MAX_WIDTH / w
            img = cv2.resize(img, (self._BROADCAST_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        now = time.monotonic()
        cutoff = now - self._cfg.broadcast_buffer_seconds
        with self._broadcast_lock:
            self._broadcast_buffer.append((now, f"data:image/jpeg;base64,{b64}"))
            # Trimmed here too, not just in the drain loop -- during a fast
            # capture burst this is what actually bounds backlog growth
            # tick-by-tick; the drain loop's own trim is what covers the
            # opposite case (a gap with no enqueues to trigger this).
            while self._broadcast_buffer and self._broadcast_buffer[0][0] < cutoff:
                self._broadcast_buffer.popleft()

    def _broadcast_loop(self) -> None:
        """Sends banked frames in arrival order (FIFO) at a steady
        frame_broadcast_fps rate, so motion looks continuous instead of
        jumping straight to "latest" and skipping content in between.

        Two earlier attempts at this both failed for opposite reasons:
        - Banking backlog but only trimming it when a NEW frame arrived
          meant nothing trimmed *during* a stall (no new frames = no trim
          call), so items could sit past their intended age before being
          served, and the backlog could grow without bound over time since
          capture arrives faster than an 8fps drain -- measured this
          costing 1.1-2.0s of real latency, constantly, not just during
          stalls.
        - Reacting to that by clearing the whole backlog every cycle and
          sending only the single freshest frame fixed the latency but
          reintroduced visible freezing: with nothing banked, there's
          nothing to show during the camera's own ~0.7-0.9s keyframe stall.

        The actual fix is both ends of the same buffer trimming by age --
        here AND in _enqueue_broadcast_frame -- so backlog can never exceed
        broadcast_buffer_seconds regardless of how long a gap in new
        frames lasts, while still serving oldest-first so a stall is
        smoothed by the backlog banked just before it, not skipped over.
        broadcast_buffer_seconds (1.2s) is sized just above the measured
        stall duration (~0.7-0.9s) -- enough margin to smooth it, not so
        much that the view feels behind.
        """
        interval = 1.0 / self._cfg.frame_broadcast_fps
        while not self._stop_event.is_set():
            start = time.monotonic()
            cutoff = start - self._cfg.broadcast_buffer_seconds
            with self._broadcast_lock:
                while self._broadcast_buffer and self._broadcast_buffer[0][0] < cutoff:
                    self._broadcast_buffer.popleft()
                item = self._broadcast_buffer.popleft() if self._broadcast_buffer else None
            if item is not None:
                _enqueued_at, image = item
                event_bus.publish({
                    "type": "frame",
                    "camera_name": self._camera_name,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "image": image,
                })
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))
