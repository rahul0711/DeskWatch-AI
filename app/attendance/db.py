"""SQLAlchemy schema (SQLite by default, MySQL when DATABASE_URL is set) for enrolled users, their stored face
embeddings, and attendance events.

NOT compatible with any pre-existing enrollment data from the old
ArcFace/facenet-based system -- AdaFace embeddings live in a different
vector space, so this is a fresh schema/database, not a migration.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    create_engine,
    inspect,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.attendance.config import AttendanceConfig


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(default=lambda: dt.datetime.now(dt.timezone.utc))

    faces: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    attendance_events: Mapped[list["AttendanceEvent"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class FaceEmbedding(Base):
    """One stored angle/photo for a user. embedding is a raw float32[512]
    buffer (already L2-normalized) written with ndarray.tobytes()."""

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    thumbnail_path: Mapped[str] = mapped_column(String(512))
    # Full uploaded image, byte-for-byte. NULL for faces enrolled before originals were kept.
    original_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    face_width_px: Mapped[int] = mapped_column(Integer)
    det_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(default=lambda: dt.datetime.now(dt.timezone.utc))

    user: Mapped["User"] = relationship(back_populates="faces")

    def embedding_array(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)


def to_utc_iso(ts: dt.datetime) -> str:
    """Every datetime this app writes is UTC (see the `default=` lambdas
    below), but SQLite has no real timezone-aware column type -- SQLAlchemy
    round-trips a value through it as a naive datetime, silently dropping
    the tzinfo. Serializing that naive value directly is a real bug, not a
    cosmetic one: `new Date("2026-09-16T08:21:34")` (no offset) in a
    browser assumes *local* time, not UTC, so every timestamp read back
    from the DB would show up to a full UTC-offset's worth of hours wrong
    (silently -- no error, just a wrong-looking time). Re-attach UTC before
    formatting for any datetime that came from the database; a fresh
    `dt.datetime.now(dt.timezone.utc)` that never touched the DB doesn't
    need this (it's still tz-aware), but calling it there too is harmless."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.isoformat()


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    camera_name: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[dt.datetime] = mapped_column(default=lambda: dt.datetime.now(dt.timezone.utc))
    confidence: Mapped[float] = mapped_column(Float)
    face_width_px: Mapped[int] = mapped_column(Integer)
    low_confidence: Mapped[bool] = mapped_column(default=False)
    thumbnail_path: Mapped[str] = mapped_column(String(512))

    user: Mapped["User"] = relationship(back_populates="attendance_events")


class ActivityEventRow(Base):
    """One completed interval of one activity stream for one person track.

    Written by app/activity/repository.py. Intentionally NOT foreign-keyed to
    `users`: identity here comes from the external /Recognize API, whose
    employee ids are not rows in this database, and an activity event is
    still worth keeping when nobody was recognized at all (person_id NULL,
    track_id only). Overlapping rows for the same track are expected and
    correct -- SITTING, STATIONARY, MONITOR_INTERACTION and
    PHONE_INTERACTION are independent streams, so per-activity totals can
    legitimately sum to more than the PRESENCE duration.
    """

    __tablename__ = "activity_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    # External employee id / name from /Recognize; NULL when unidentified.
    person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    person_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # PRESENCE | SITTING | STANDING | WALKING | STATIONARY |
    # PHONE_INTERACTION | MONITOR_INTERACTION -- see app/activity/state_manager.Activity
    activity_type: Mapped[str] = mapped_column(String(64), index=True)
    start_time: Mapped[dt.datetime] = mapped_column(index=True)
    end_time: Mapped[dt.datetime] = mapped_column()
    duration_seconds: Mapped[float] = mapped_column(Float)
    # Fraction of observations during the interval that agreed with the
    # reported state -- a measured agreement rate, not a model probability.
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(default=lambda: dt.datetime.now(dt.timezone.utc))

    # The query this table exists to serve is "what did person X do on
    # camera Y between these times", so that tuple gets a composite index
    # on top of the single-column ones above.
    __table_args__ = (
        Index("ix_activity_person_camera_start", "person_id", "camera_id", "start_time"),
        Index("ix_activity_type_start", "activity_type", "start_time"),
    )


_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(cfg: AttendanceConfig | None = None):
    global _engine, _SessionLocal
    cfg = cfg or AttendanceConfig()
    if cfg.database_url:
        # pre_ping/recycle: MySQL drops idle connections after wait_timeout.
        _engine = create_engine(cfg.database_url, pool_pre_ping=True, pool_recycle=3600)
    else:
        cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{cfg.db_path}")
    Base.metadata.create_all(_engine)
    # create_all never alters existing tables, so add columns introduced after a DB was created.
    cols = {c["name"] for c in inspect(_engine).get_columns("face_embeddings")}
    if "original_path" not in cols:
        with _engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE face_embeddings ADD COLUMN original_path VARCHAR(512)")
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
