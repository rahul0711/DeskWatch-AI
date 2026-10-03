import datetime as dt
import time
from pathlib import Path
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.attendance.best_frame import TrackBuffer
from app.attendance.config import AttendanceConfig
from app.attendance.db import AttendanceEvent, Base, User, init_db
from app.attendance.events import event_bus
from app.attendance.worker import RecognitionWorker


class DummyDetector:
    def detect(self, img):
        return []


class DummyEmbedder:
    def embed_aligned_batch(self, crops):
        # Return 512-dim dummy embedding for each crop
        return [np.zeros(512, dtype=np.float32) for _ in crops]


class DummyIndex:
    def __init__(self, match_result=None):
        self._match_result = match_result

    def best_match(self, embedding):
        return self._match_result


def test_config_punch_cooldown(monkeypatch):
    cfg = AttendanceConfig()
    assert cfg.punch_cooldown_seconds == 600.0

    monkeypatch.setenv("PUNCH_COOLDOWN_SECONDS", "300")
    cfg2 = AttendanceConfig()
    assert cfg2.punch_cooldown_seconds == 300.0


def test_get_last_punch_time_and_cooldown(tmp_path, monkeypatch):
    # Setup isolated test DB
    db_path = tmp_path / "test_attendance.db"
    crops_dir = tmp_path / "attendance_crops"
    cfg = AttendanceConfig(
        db_path=db_path,
        attendance_crops_dir=crops_dir,
        punch_cooldown_seconds=600.0,
        match_threshold=0.35,
    )
    init_db(cfg)

    # Insert a test user
    from app.attendance.db import get_session
    session = get_session()
    user = User(name="Test User", employee_id="EMP001")
    session.add(user)
    session.commit()
    user_id = user.id
    session.close()

    detector = DummyDetector()
    embedder = DummyEmbedder()
    index = DummyIndex(match_result=(user_id, 0.95))

    worker = RecognitionWorker(cfg, detector, embedder, index)

    # Initially, no punch recorded
    assert worker._get_last_punch_time(user_id) is None

    # Track received events on event_bus
    received_events = []
    def mock_publish(evt):
        received_events.append(evt)

    monkeypatch.setattr(event_bus, "publish", mock_publish)

    # Create dummy buffer with an aligned crop
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    buf1 = TrackBuffer(track_id=1)
    buf1.best_aligned_crop = crop
    buf1.best_face_width_px = 80
    buf1.best_det_score = 0.9

    # First finalization: should record attendance and save crop
    worker._finalize_batch([(1, buf1)])

    # Verify event published
    assert len(received_events) == 1
    assert received_events[0]["type"] == "attendance"
    assert received_events[0]["user_id"] == user_id
    assert received_events[0]["name"] == "Test User"

    # Verify 1 crop file saved on disk
    crop_files = list(crops_dir.glob("*.jpg"))
    assert len(crop_files) == 1

    # Verify 1 DB record exists
    session = get_session()
    events = session.query(AttendanceEvent).filter_by(user_id=user_id).all()
    assert len(events) == 1
    session.close()

    # Second finalization (same user, new track within cooldown):
    buf2 = TrackBuffer(track_id=2)
    buf2.best_aligned_crop = crop
    buf2.best_face_width_px = 85
    buf2.best_det_score = 0.92

    worker._finalize_batch([(2, buf2)])

    # Verify cooldown event was published, NOT another attendance event
    assert len(received_events) == 2
    assert received_events[1]["type"] == "cooldown"
    assert received_events[1]["user_id"] == user_id
    assert "done punching" in received_events[1]["message"].lower()
    assert received_events[1]["remaining_minutes"] == 10

    # Verify NO additional crop file was saved to disk
    crop_files_after = list(crops_dir.glob("*.jpg"))
    assert len(crop_files_after) == 1

    # Verify NO additional DB record was saved
    session = get_session()
    events_after = session.query(AttendanceEvent).filter_by(user_id=user_id).all()
    assert len(events_after) == 1
    session.close()


def test_cooldown_restored_from_db_on_restart(tmp_path):
    # Tests that if a user punched recently in DB, a new worker instance honors the cooldown
    db_path = tmp_path / "test_restart.db"
    crops_dir = tmp_path / "crops"
    cfg = AttendanceConfig(
        db_path=db_path,
        attendance_crops_dir=crops_dir,
        punch_cooldown_seconds=600.0,
    )
    init_db(cfg)

    from app.attendance.db import get_session
    session = get_session()
    user = User(name="Restart User", employee_id="EMP999")
    session.add(user)
    session.commit()
    user_id = user.id

    # Insert an event timestamped 2 minutes ago
    two_mins_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=2)
    evt = AttendanceEvent(
        user_id=user_id,
        camera_name="TestCam",
        confidence=0.9,
        face_width_px=80,
        low_confidence=False,
        thumbnail_path="test.jpg",
        timestamp=two_mins_ago,
    )
    session.add(evt)
    session.commit()
    session.close()

    # Create fresh worker (empty in-memory cache)
    worker = RecognitionWorker(cfg, DummyDetector(), DummyEmbedder(), DummyIndex())
    last_punch = worker._get_last_punch_time(user_id)
    assert last_punch is not None
    # Elapsed should be roughly 120s, so remaining should be ~480s
    remaining_s = cfg.punch_cooldown_seconds - (time.monotonic() - last_punch)
    assert 450 <= remaining_s <= 510


def test_unknown_face_ttl_cleanup(tmp_path, monkeypatch):
    crops_dir = tmp_path / "unknown_crops"
    cfg = AttendanceConfig(
        attendance_crops_dir=crops_dir,
        unknown_face_ttl_seconds=0.1,  # short TTL for testing fast cleanup
    )

    worker = RecognitionWorker(cfg, DummyDetector(), DummyEmbedder(), DummyIndex(match_result=None))

    events = []
    monkeypatch.setattr(event_bus, "publish", lambda evt: events.append(evt))

    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    buf = TrackBuffer(track_id=99)
    buf.best_aligned_crop = crop
    buf.best_face_width_px = 75

    # Finalize an unknown track
    worker._finalize_batch([(99, buf)])

    # Crop saved initially
    assert len(events) == 1
    assert events[0]["type"] == "unknown"
    assert events[0]["ttl_seconds"] == 0.1

    saved_files = list(crops_dir.glob("unknown_*.jpg"))
    assert len(saved_files) == 1
    crop_file = saved_files[0]
    assert crop_file.exists()

    # Before TTL expiration, cleanup should not delete it
    worker._cleanup_expired_unknowns()
    assert crop_file.exists()

    # Sleep past TTL (0.15s)
    time.sleep(0.15)
    worker._cleanup_expired_unknowns()

    # Now the file should be deleted from disk
    assert not crop_file.exists()
    assert len(list(crops_dir.glob("unknown_*.jpg"))) == 0


def test_list_attendance_deduplication(tmp_path):
    from app.attendance.enrollment import list_attendance
    from app.attendance.db import get_session
    from app.attendance.pipeline import init_pipeline

    db_path = tmp_path / "test_dedup.db"
    crops_dir = tmp_path / "crops"
    cfg = AttendanceConfig(
        db_path=db_path,
        attendance_crops_dir=crops_dir,
        punch_cooldown_seconds=600.0,
    )
    init_db(cfg)

    session = get_session()
    user = User(name="Dedup User", employee_id="EMP888")
    session.add(user)
    session.commit()
    user_id = user.id

    now = dt.datetime.now(dt.timezone.utc)
    # Insert 3 events: two 30 seconds apart, one 15 minutes later
    e1 = AttendanceEvent(user_id=user_id, camera_name="Cam", confidence=0.9, face_width_px=80, low_confidence=False, thumbnail_path="1.jpg", timestamp=now)
    e2 = AttendanceEvent(user_id=user_id, camera_name="Cam", confidence=0.88, face_width_px=82, low_confidence=False, thumbnail_path="2.jpg", timestamp=now + dt.timedelta(seconds=30))
    e3 = AttendanceEvent(user_id=user_id, camera_name="Cam", confidence=0.92, face_width_px=85, low_confidence=False, thumbnail_path="3.jpg", timestamp=now + dt.timedelta(minutes=15))
    session.add_all([e1, e2, e3])
    session.commit()
    session.close()

    today_str = now.strftime("%Y-%m-%d")
    results = list_attendance(today_str)

    # Out of 3 events, e2 is within 10 minutes of e1, so only e1 and e3 should be returned
    assert len(results) == 2
    assert results[0]["id"] == e1.id
    assert results[1]["id"] == e3.id
