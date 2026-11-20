"""Persistence for activity sessions (table activity_sessions, per USER).

Writes go through a bounded queue drained by one background thread, for a
hard requirement from the spec: a database problem must never take down the
CCTV service. The render loop only ever does a non-blocking put; a dead or
slow database costs dropped event rows and a warning, not a stalled camera.

Reads go straight through a short-lived session, since they only happen on
API requests.
"""
from __future__ import annotations

import datetime as dt
import logging
import queue
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAX = 2000
_BATCH_SIZE = 100
_FLUSH_INTERVAL_S = 5.0


class ActivityRepository:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._written = 0

    # -- lifecycle ------------------------------------------------------

    def start(self) -> "ActivityRepository":
        if not self._enabled:
            logger.info("Activity event persistence disabled (ACTIVITY_PERSIST_EVENTS=false)")
            return self
        try:
            from app.attendance.db import init_db
            init_db()
        except Exception as exc:
            # A missing/broken DB disables persistence rather than killing
            # the camera pipeline.
            logger.error("Activity persistence disabled -- could not init database: %s", exc)
            self._enabled = False
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name="activity-writer")
        self._thread.start()
        logger.info("Activity event persistence started")
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=10)
        logger.info(
            "Activity persistence stopped (written=%d dropped=%d)", self._written, self._dropped
        )

    # -- write ----------------------------------------------------------

    def record(self, event) -> None:
        """Non-blocking. Drops the event (with a throttled warning) if the
        writer has fallen too far behind."""
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    "Activity event queue full -- dropped %d event(s) so far", self._dropped
                )

    def record_many(self, events: Iterable) -> None:
        for event in events:
            self.record(event)

    def _run(self) -> None:
        batch: list = []
        while not self._stop.is_set() or not self._queue.empty():
            try:
                batch.append(self._queue.get(timeout=_FLUSH_INTERVAL_S))
            except queue.Empty:
                pass
            while len(batch) < _BATCH_SIZE:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if batch:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list) -> None:
        try:
            from app.attendance.db import ActivitySessionRow, get_session
            with get_session() as session:
                session.add_all([
                    ActivitySessionRow(
                        user_id=e.person_id,
                        user_name=e.person_name,
                        camera_id=e.camera_id,
                        activity_type=e.activity_type,
                        start_time=e.start_time,
                        end_time=e.end_time,
                        duration=e.duration_seconds,
                        confidence=e.confidence,
                        track_ids=",".join(str(t) for t in (e.metadata or {}).get("track_ids", [e.track_id]))[:512],
                    )
                    for e in batch
                ])
                session.commit()
            self._written += len(batch)
        except Exception as exc:
            self._dropped += len(batch)
            logger.error("Failed writing %d activity session(s): %s", len(batch), exc)

    # -- read -----------------------------------------------------------

    @staticmethod
    def _day_bounds(date: Optional[str]):
        if not date:
            return None
        try:
            day = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            return None
        return day, day + dt.timedelta(days=1)

    @staticmethod
    def query_events(
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        """Stored activity sessions, newest first. `date` is YYYY-MM-DD (UTC)."""
        from sqlalchemy import select
        from app.attendance.db import ActivitySessionRow as R, get_session, to_utc_iso

        stmt = select(R)
        if camera_id:
            stmt = stmt.where(R.camera_id == camera_id)
        if person_id:
            stmt = stmt.where(R.user_id == person_id)
        if activity_type:
            stmt = stmt.where(R.activity_type == activity_type.upper())
        bounds = ActivityRepository._day_bounds(date)
        if bounds:
            stmt = stmt.where(R.start_time >= bounds[0], R.start_time < bounds[1])
        stmt = stmt.order_by(R.start_time.desc()).limit(max(1, min(limit, 5000)))

        with get_session() as session:
            rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "user_id": r.user_id,
                "person_id": r.user_id,
                "person_name": r.user_name or "Unknown",
                "camera_id": r.camera_id,
                "activity_type": r.activity_type,
                "start_time": to_utc_iso(r.start_time),
                "end_time": to_utc_iso(r.end_time),
                "duration_seconds": round(r.duration, 1),
                "confidence": round(r.confidence or 0.0, 3),
                "track_ids": r.track_ids,
            }
            for r in rows
        ]

    @staticmethod
    def user_totals(user_id: str, camera_id: Optional[str] = None, date: Optional[str] = None) -> dict:
        """{activity_type: total seconds} already stored for one user, for
        `date` (default: today, UTC). Seeds the live totals after a restart."""
        from sqlalchemy import func, select
        from app.attendance.db import ActivitySessionRow as R, get_session

        date = date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        stmt = select(R.activity_type, func.sum(R.duration)).where(R.user_id == user_id).group_by(R.activity_type)
        if camera_id:
            stmt = stmt.where(R.camera_id == camera_id)
        bounds = ActivityRepository._day_bounds(date)
        if bounds:
            stmt = stmt.where(R.start_time >= bounds[0], R.start_time < bounds[1])
        with get_session() as session:
            return {a: float(t or 0.0) for a, t in session.execute(stmt).all()}

    @staticmethod
    def summary(
        camera_id: Optional[str] = None,
        date: Optional[str] = None,
    ) -> list[dict]:
        """Per-USER totals per activity type (sum over all their sessions),
        for the "Rahul: sitting 5h42m, phone 24m, ..." view."""
        from sqlalchemy import func, select
        from app.attendance.db import ActivitySessionRow as R, get_session, to_utc_iso

        stmt = select(
            R.user_id,
            func.max(R.user_name),
            R.activity_type,
            func.sum(R.duration),
            func.count(R.id),
            func.min(R.start_time),
            func.max(R.end_time),
        ).group_by(R.user_id, R.activity_type)
        if camera_id:
            stmt = stmt.where(R.camera_id == camera_id)
        bounds = ActivityRepository._day_bounds(date)
        if bounds:
            stmt = stmt.where(R.start_time >= bounds[0], R.start_time < bounds[1])

        with get_session() as session:
            rows = session.execute(stmt).all()

        people: dict = {}
        for user_id, user_name, activity_type, total, count, first, last in rows:
            key = user_id or "unknown"
            entry = people.setdefault(key, {
                "user_id": user_id,
                "person_id": user_id,
                "person_name": user_name or "Unknown",
                "activities": {},
                "sessions": {},
                "first_seen": first,
                "last_seen": last,
            })
            entry["activities"][activity_type] = round(float(total or 0.0), 1)
            entry["sessions"][activity_type] = int(count or 0)
            if first is not None and (entry["first_seen"] is None or first < entry["first_seen"]):
                entry["first_seen"] = first
            if last is not None and (entry["last_seen"] is None or last > entry["last_seen"]):
                entry["last_seen"] = last

        return [
            {
                **entry,
                "first_seen": to_utc_iso(entry["first_seen"]) if entry["first_seen"] else None,
                "last_seen": to_utc_iso(entry["last_seen"]) if entry["last_seen"] else None,
            }
            for entry in people.values()
        ]
