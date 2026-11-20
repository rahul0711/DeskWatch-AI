"""Which analyses each camera runs: face, body movement, both or neither.

    Admin page (Cameras)  -> PUT /api/db/cameras/{id}/features (.NET backend)
    cameras table         -> face_enabled, activity_enabled
    app/web/server.py     -> re-reads it every few seconds and restarts only
                             the camera whose switches changed

  face      face detection + recognition: attendance, identity names on the
            activity overlay, restricted-area alerts
  activity  body movement: posture, movement, phone / monitor / computer use

A camera with no row in the table (a camera added at runtime, or the table
not migrated yet) keeps the .env behaviour: face on, activity per
ACTIVITY_CAMERAS. The same fallback applies while the database is
unreachable, so a DB outage never switches a running camera off.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from app.attendance.db import get_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFeatures:
    face: bool
    activity: bool


def load_camera_features() -> Optional[dict[str, CameraFeatures]]:
    """{camera_id: CameraFeatures} from the cameras table, or None when it
    can't be read (no table, columns not added yet, database down)."""
    try:
        with get_session() as session:
            rows = session.execute(
                text("SELECT camera_id, face_enabled, activity_enabled FROM cameras")
            ).all()
    except Exception as exc:
        logger.debug("Camera features not readable (%s) -- using .env defaults", exc)
        return None
    return {str(cid).lower(): CameraFeatures(bool(face), bool(act)) for cid, face, act in rows}


def features_for(
    camera_id: str,
    table: Optional[dict[str, CameraFeatures]],
    default_activity: bool,
) -> CameraFeatures:
    if table is not None and camera_id.lower() in table:
        return table[camera_id.lower()]
    return CameraFeatures(face=True, activity=default_activity)


def load_workstations() -> Optional[dict[str, list[dict]]]:
    """{camera_id: [{"name", "seat": [x1,y1,x2,y2], "screen": [...] | None}]}
    from the workstations table (Admin -> Workstations), coordinates as
    fractions of the frame. None when it can't be read (table not created
    yet, database down) -- cameras then keep the zones they have."""
    try:
        with get_session() as session:
            rows = session.execute(text(
                "SELECT camera_id, name, seat_x1, seat_y1, seat_x2, seat_y2, "
                "screen_x1, screen_y1, screen_x2, screen_y2 FROM workstations ORDER BY id"
            )).all()
    except Exception as exc:
        logger.debug("Workstations not readable (%s)", exc)
        return None
    out: dict[str, list[dict]] = {}
    for cid, name, *v in rows:
        screen = [float(x) for x in v[4:]] if all(x is not None for x in v[4:]) else None
        out.setdefault(str(cid).lower(), []).append(
            {"name": name, "seat": [float(x) for x in v[:4]], "screen": screen})
    return out
