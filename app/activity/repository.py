"""Persistence for activity events.

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
            from app.attendance.db import ActivityEventRow, get_session
            with get_session() as session:
                session.add_all([
                    ActivityEventRow(
                        camera_id=e.camera_id,
                        track_id=e.track_id,
                        person_id=e.person_id,
                        person_name=e.person_name,
                        activity_type=e.activity_type,
                        start_time=e.start_time,
                        end_time=e.end_time,
                        duration_seconds=e.duration_seconds,
                        confidence=e.confidence,
                    )
                    for e in batch
                ])
                session.commit()
            self._written += len(batch)
        except Exception as exc:
            self._dropped += len(batch)
            logger.error("Failed writing %d activity event(s): %s", len(batch), exc)

    # -- read -----------------------------------------------------------

    @staticmethod
    def query_events(
        camera_id: Optional[str] = None,
        person_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        """Chronological event rows, newest first. `date` is YYYY-MM-DD (UTC)."""
        from sqlalchemy import select
        from app.attendance.db import ActivityEventRow, get_session, to_utc_iso

        stmt = select(ActivityEventRow)
        if camera_id:
            stmt = stmt.where(ActivityEventRow.camera_id == camera_id)
        if person_id:
            stmt = stmt.where(ActivityEventRow.person_id == person_id)
        if activity_type:
            stmt = stmt.where(ActivityEventRow.activity_type == activity_type.upper())
        if date:
            try:
                day = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                day = None
            if day is not None:
                stmt = stmt.where(
                    ActivityEventRow.start_time >= day,
                    ActivityEventRow.start_time < day + dt.timedelta(days=1),
                )
        stmt = stmt.order_by(ActivityEventRow.start_time.desc()).limit(max(1, min(limit, 5000)))

        with get_session() as session:
            rows = session.execute(stmt).scalars().all()

        return [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "track_id": r.track_id,
                "person_id": r.person_id,
                "person_name": r.person_name or "Unknown",
                "activity_type": r.activity_type,
                "start_time": to_utc_iso(r.start_time),
                "end_time": to_utc_iso(r.end_time),
                "duration_seconds": round(r.duration_seconds, 1),
                "confidence": round(r.confidence or 0.0, 3),
            }
            for r in rows
        ]

    @staticmethod
    def summary(
        camera_id: Optional[str] = None,
        date: Optional[str] = None,
    ) -> list[dict]:
        """Per-person totals per activity type, for the "Rahul: sitting
        5h42m, phone 24m, ..." view."""
        from sqlalchemy import func, select
        from app.attendance.db import ActivityEventRow, get_session

        stmt = select(
            ActivityEventRow.person_id,
            ActivityEventRow.person_name,
            ActivityEventRow.activity_type,
            func.sum(ActivityEventRow.duration_seconds),
            func.min(ActivityEventRow.start_time),
            func.max(ActivityEventRow.end_time),
        ).group_by(
            ActivityEventRow.person_id,
            ActivityEventRow.person_name,
            ActivityEventRow.activity_type,
        )
        if camera_id:
            stmt = stmt.where(ActivityEventRow.camera_id == camera_id)
        if date:
            try:
                day = dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
                stmt = stmt.where(
                    ActivityEventRow.start_time >= day,
                    ActivityEventRow.start_time < day + dt.timedelta(days=1),
                )
            except ValueError:
                pass

        with get_session() as session:
            rows = session.execute(stmt).all()

        people: dict = {}
        for person_id, person_name, activity_type, total, first, last in rows:
            key = person_id or person_name or "unknown"
            entry = people.setdefault(key, {
                "person_id": person_id,
                "person_name": person_name or "Unknown",
                "activities": {},
                "first_seen": first,
                "last_seen": last,
            })
            entry["activities"][activity_type] = round(float(total or 0.0), 1)
            if first is not None and (entry["first_seen"] is None or first < entry["first_seen"]):
                entry["first_seen"] = first
            if last is not None and (entry["last_seen"] is None or last > entry["last_seen"]):
                entry["last_seen"] = last

        from app.attendance.db import to_utc_iso
        out = []
        for entry in people.values():
            out.append({
                **entry,
                "first_seen": to_utc_iso(entry["first_seen"]) if entry["first_seen"] else None,
                "last_seen": to_utc_iso(entry["last_seen"]) if entry["last_seen"] else None,
            })
        return out
