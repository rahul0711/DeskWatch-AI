"""Per-camera access control for registered users.

    Register page  -> "Allowed cameras" checkboxes per person
    camera_restrictions table -> the cameras a person is NOT allowed on
    app/web/server.py -> raises a "not welcome" alert when a registered person
                         is recognized on one of their blocked cameras

Default is allowed: no rows = allowed on every camera, including cameras
added later.
"""
from __future__ import annotations

import logging
import threading
import time

from sqlalchemy import delete, select

from app.attendance.db import CameraRestriction, User, get_session

logger = logging.getLogger(__name__)


class UserNotFound(Exception):
    pass


def blocked_cameras(user_id: int) -> list[str]:
    with get_session() as session:
        rows = session.execute(
            select(CameraRestriction.camera_id).where(CameraRestriction.user_id == user_id)
        ).scalars().all()
    return sorted(rows)


def set_blocked_cameras(user_id: int, camera_ids: list[str]) -> list[str]:
    wanted = sorted({c.strip() for c in camera_ids if c and c.strip()})
    with get_session() as session:
        if session.get(User, user_id) is None:
            raise UserNotFound(user_id)
        session.execute(delete(CameraRestriction).where(CameraRestriction.user_id == user_id))
        session.add_all([CameraRestriction(user_id=user_id, camera_id=c) for c in wanted])
        session.commit()
    logger.info("Camera access for user %s: blocked on %s", user_id, wanted or "none")
    return wanted


def all_blocked() -> dict[int, set[str]]:
    """{user_id: {blocked camera ids}} for every restricted user."""
    out: dict[int, set[str]] = {}
    with get_session() as session:
        for user_id, camera_id in session.execute(
            select(CameraRestriction.user_id, CameraRestriction.camera_id)
        ).all():
            out.setdefault(user_id, set()).add(camera_id)
    return out


class AccessPolicy:
    """Cached view of camera_restrictions for the live render loops.
    Re-reads the table at most every `refresh_seconds`, so a change in the
    Register page takes effect within a few seconds without a restart and
    without a DB query per frame."""

    def __init__(self, refresh_seconds: float = 5.0) -> None:
        self._refresh = refresh_seconds
        self._lock = threading.Lock()
        self._blocked: dict[int, set[str]] = {}
        self._loaded_at = -1e9

    def is_blocked(self, user_id: int, camera_id: str) -> bool:
        self._maybe_reload()
        with self._lock:
            return camera_id in self._blocked.get(user_id, ())

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._loaded_at < self._refresh:
            return
        self._loaded_at = now
        try:
            blocked = all_blocked()
        except Exception as exc:
            logger.warning("Could not reload camera restrictions: %s", exc)
            return
        with self._lock:
            self._blocked = blocked
