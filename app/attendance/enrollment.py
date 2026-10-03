"""Registration/enrollment business logic: detect -> align -> embed -> store
-> rebuild FAISS index. Used by the FastAPI routes in app/web/attendance_routes.py.

Uses the SAME align_face/normalize_for_model functions (via AdaFaceEmbedder)
as live recognition -- see app/attendance/alignment.py's module docstring for
why that matters.
"""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from app.attendance.alignment import align_face
from app.attendance.config import AttendanceConfig
from app.attendance.db import AttendanceEvent, FaceEmbedding, User, get_session, to_utc_iso
from app.attendance.pipeline import get_config, get_detector, get_embedder, get_index

logger = logging.getLogger(__name__)


@dataclass
class ImageResult:
    filename: str
    accepted: bool
    reason: str = ""  # rejection reason, empty if accepted
    face_id: int | None = None
    face_width_px: int | None = None
    thumbnail_url: str | None = None


class DuplicateEmployeeId(Exception):
    pass


class UserNotFound(Exception):
    pass


def _decode_image(raw_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _image_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


def _user_photo_url(user: User) -> str | None:
    """Registration photo shown when the user is recognized: their first stored
    full original image, falling back to the face crop for older enrollments."""
    faces = sorted(user.faces, key=lambda f: f.id)
    for f in faces:
        if f.original_path:
            return _thumbnail_url(Path(f.original_path))
    return _thumbnail_url(Path(faces[0].thumbnail_path)) if faces else None


def _thumbnail_url(path_on_disk) -> str:
    cfg = get_config()
    p = Path(path_on_disk)
    try:
        rel = p.relative_to(cfg.enrollment_dir.parent)
        return f"/data/{rel.as_posix()}"
    except (ValueError, AttributeError):
        pass
    parts = p.parts
    if "data" in parts:
        data_idx = parts.index("data")
        return f"/data/{'/'.join(parts[data_idx + 1:])}"
    return f"/data/{p.name}"


def _process_one_image(user_id: int, filename: str, raw_bytes: bytes) -> ImageResult:
    cfg = get_config()
    img = _decode_image(raw_bytes)
    if img is None:
        return ImageResult(filename=filename, accepted=False, reason="Could not decode image file")

    faces = get_detector().detect(img)
    if not faces:
        return ImageResult(filename=filename, accepted=False, reason="No face detected")

    # If multiple faces are in an enrollment photo, use the largest -- most
    # likely to be the intentional subject (a webcam selfie occasionally
    # catches a second person in the background).
    face = max(faces, key=lambda f: f.width_px)

    if face.width_px < cfg.min_register_face_px:
        return ImageResult(
            filename=filename,
            accepted=False,
            reason=f"Face too small ({face.width_px}px, need >={cfg.min_register_face_px}px) -- move closer to the camera",
        )

    aligned = align_face(img, face.kps)
    embedding = get_embedder().embed_aligned_batch([aligned])[0]

    cfg.enrollment_dir.mkdir(parents=True, exist_ok=True)
    user_dir = cfg.enrollment_dir / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    thumb_path = user_dir / f"{stem}.jpg"
    cv2.imwrite(str(thumb_path), aligned)
    original_path = user_dir / f"{stem}_original{_image_ext(filename)}"
    original_path.write_bytes(raw_bytes)

    session = get_session()
    try:
        row = FaceEmbedding(
            user_id=user_id,
            embedding=embedding.tobytes(),
            thumbnail_path=str(thumb_path),
            original_path=str(original_path),
            face_width_px=face.width_px,
            det_score=face.det_score,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return ImageResult(
            filename=filename,
            accepted=True,
            face_id=row.id,
            face_width_px=face.width_px,
            thumbnail_url=_thumbnail_url(thumb_path),
        )
    finally:
        session.close()


def register_user(name: str, employee_id: str, images: list[tuple[str, bytes]]) -> tuple[User, list[ImageResult]]:
    """images: list of (filename, raw_bytes). Returns the created user and a
    per-image accept/reject result. At least one image must be accepted or
    the user is not created."""
    session = get_session()
    try:
        existing = session.execute(select(User).where(User.employee_id == employee_id)).scalar_one_or_none()
        if existing is not None:
            raise DuplicateEmployeeId(employee_id)
        user = User(name=name, employee_id=employee_id)
        session.add(user)
        session.commit()
        session.refresh(user)
        user_id = user.id
    finally:
        session.close()

    results = [_process_one_image(user_id, fname, data) for fname, data in images]

    if not any(r.accepted for r in results):
        # roll back the empty user -- registration must have at least one usable angle
        session = get_session()
        try:
            u = session.get(User, user_id)
            if u is not None:
                session.delete(u)
                session.commit()
        finally:
            session.close()
        raise ValueError("No usable face images -- " + "; ".join(f"{r.filename}: {r.reason}" for r in results))

    get_index().rebuild()

    session = get_session()
    try:
        user = session.get(User, user_id)
        session.expunge(user)
    finally:
        session.close()
    return user, results


def add_face(user_id: int, filename: str, raw_bytes: bytes) -> ImageResult:
    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None:
            raise UserNotFound(user_id)
    finally:
        session.close()

    result = _process_one_image(user_id, filename, raw_bytes)
    if result.accepted:
        get_index().rebuild()
    return result


def delete_face(face_id: int) -> None:
    session = get_session()
    try:
        row = session.get(FaceEmbedding, face_id)
        if row is not None:
            session.delete(row)
            session.commit()
    finally:
        session.close()
    get_index().rebuild()


def rename_user(user_id: int, name: str) -> dict:
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty")
    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None:
            raise UserNotFound(user_id)
        user.name = name
        session.commit()
        return {"id": user.id, "name": user.name, "employee_id": user.employee_id}
    finally:
        session.close()


def delete_user(user_id: int) -> None:
    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None:
            raise UserNotFound(user_id)
        session.delete(user)
        session.commit()
    finally:
        session.close()
    get_index().rebuild()


def list_users() -> list[dict]:
    session = get_session()
    try:
        users = session.execute(select(User)).scalars().all()
        out = []
        for u in users:
            out.append({
                "id": u.id,
                "employee_id": u.employee_id,
                "name": u.name,
                "created_at": to_utc_iso(u.created_at),
                "faces": [
                    {
                        "id": f.id,
                        "thumbnail_url": _thumbnail_url_safe(f.thumbnail_path),
                        "original_url": _thumbnail_url_safe(f.original_path) if f.original_path else None,
                        "face_width_px": f.face_width_px,
                        "det_score": f.det_score,
                        "created_at": to_utc_iso(f.created_at),
                    }
                    for f in u.faces
                ],
            })
        return out
    finally:
        session.close()


def _thumbnail_url_safe(path_str: str) -> str:
    from pathlib import Path

    return _thumbnail_url(Path(path_str))


def recognize_and_punch(raw_bytes: bytes, source: str, punch: bool) -> dict:
    """Identify the largest face in one photo (e.g. a phone selfie) and, if
    punch=True and the user isn't inside the punch cooldown, write an
    attendance event exactly like the CCTV worker does."""
    cfg = get_config()
    img = _decode_image(raw_bytes)
    if img is None:
        raise ValueError("Could not decode image file")

    faces = get_detector().detect(img)
    if not faces:
        return {"recognized": False, "punched": False, "reason": "no_face", "message": "No face detected"}
    face = max(faces, key=lambda f: f.width_px)

    aligned = align_face(img, face.kps)
    embedding = get_embedder().embed_aligned_batch([aligned])[0]
    match = get_index().best_match(embedding)
    low_conf = face.width_px < cfg.low_confidence_face_px
    base = {"face_width_px": face.width_px, "low_confidence": low_conf}

    if match is None or match[1] < cfg.match_threshold:
        return {
            **base,
            "recognized": False,
            "punched": False,
            "reason": "unknown",
            "best_score": match[1] if match else None,
            "message": "Face not recognized",
        }

    user_id, score = match
    now = dt.datetime.now(dt.timezone.utc)
    session = get_session()
    try:
        user = session.get(User, user_id)
        result = {
            **base,
            "recognized": True,
            "confidence": score,
            "user": {
                "id": user.id,
                "name": user.name,
                "employee_id": user.employee_id,
                "photo_url": _user_photo_url(user),
            },
        }
        if not punch:
            return {**result, "punched": False, "reason": "identify_only", "message": f"Recognized {user.name}"}

        last = session.execute(
            select(AttendanceEvent)
            .where(AttendanceEvent.user_id == user_id)
            .order_by(AttendanceEvent.timestamp.desc())
            .limit(1)
        ).scalars().first()
        if last is not None:
            last_ts = last.timestamp if last.timestamp.tzinfo else last.timestamp.replace(tzinfo=dt.timezone.utc)
            elapsed = (now - last_ts).total_seconds()
            if 0 <= elapsed < cfg.punch_cooldown_seconds:
                remaining = int(cfg.punch_cooldown_seconds - elapsed)
                return {
                    **result,
                    "punched": False,
                    "reason": "cooldown",
                    "last_punch": to_utc_iso(last_ts),
                    "remaining_seconds": remaining,
                    "message": f"{user.name} already punched, try again in {max(1, (remaining + 59) // 60)} min",
                }

        cfg.attendance_crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = cfg.attendance_crops_dir / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(crop_path), aligned)
        event = AttendanceEvent(
            user_id=user_id,
            camera_name=source,
            confidence=score,
            face_width_px=face.width_px,
            low_confidence=low_conf,
            thumbnail_path=str(crop_path),
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return {
            **result,
            "punched": True,
            "reason": "punched",
            "attendance_id": event.id,
            "timestamp": to_utc_iso(event.timestamp),
            "thumbnail_url": _thumbnail_url(crop_path),
            "message": f"Punch recorded for {user.name}",
        }
    finally:
        session.close()


def list_attendance(date_str: str | None) -> list[dict]:
    """date_str: 'YYYY-MM-DD', or None for all. Deduplicates entries within the cooldown window."""
    session = get_session()
    cfg = get_config()
    cooldown_s = getattr(cfg, "punch_cooldown_seconds", 600.0)
    try:
        stmt = select(AttendanceEvent).order_by(AttendanceEvent.timestamp.asc())
        events = session.execute(stmt).scalars().all()
        out = []
        last_seen: dict[int, dt.datetime] = {}
        for e in events:
            if date_str and e.timestamp.strftime("%Y-%m-%d") != date_str:
                continue

            ts = e.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)

            # Suppress duplicate punch records within the 10-minute cooldown window
            if e.user_id in last_seen:
                prev_ts = last_seen[e.user_id]
                if (ts - prev_ts).total_seconds() < cooldown_s:
                    continue

            last_seen[e.user_id] = ts
            out.append({
                "id": e.id,
                "user_id": e.user_id,
                "name": e.user.name,
                "employee_id": e.user.employee_id,
                "camera_name": e.camera_name,
                "timestamp": to_utc_iso(e.timestamp),
                "confidence": e.confidence,
                "face_width_px": e.face_width_px,
                "low_confidence": e.low_confidence,
                "thumbnail_url": _thumbnail_url_safe(e.thumbnail_path),
            })
        return out
    finally:
        session.close()
