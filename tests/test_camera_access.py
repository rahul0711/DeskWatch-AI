"""Per-camera access rules for registered users (Register page)."""
from __future__ import annotations

import pytest

from app.attendance import access, db
from app.attendance.config import AttendanceConfig


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cfg = AttendanceConfig()
    object.__setattr__(cfg, "database_url", None)
    object.__setattr__(cfg, "db_path", tmp_path / "test.db")
    db.init_db(cfg)
    with db.get_session() as s:
        s.add_all([db.User(employee_id="E1", name="Rahul"), db.User(employee_id="E2", name="Asha")])
        s.commit()
        ids = {u.employee_id: u.id for u in s.query(db.User).all()}
    yield ids
    db._engine = None
    db._SessionLocal = None


def test_default_is_allowed_everywhere(fresh_db):
    assert access.blocked_cameras(fresh_db["E1"]) == []
    assert access.AccessPolicy().is_blocked(fresh_db["E1"], "corridor") is False


def test_blocking_a_camera_only_blocks_that_camera(fresh_db):
    uid = fresh_db["E1"]
    assert access.set_blocked_cameras(uid, ["corridor", "corridor", " "]) == ["corridor"]
    policy = access.AccessPolicy()
    assert policy.is_blocked(uid, "corridor") is True
    assert policy.is_blocked(uid, "camera-01") is False
    assert policy.is_blocked(fresh_db["E2"], "corridor") is False


def test_unblocking_replaces_the_previous_list(fresh_db):
    uid = fresh_db["E1"]
    access.set_blocked_cameras(uid, ["corridor", "inside"])
    access.set_blocked_cameras(uid, ["inside"])
    assert access.blocked_cameras(uid) == ["inside"]


def test_unknown_user_is_rejected(fresh_db):
    with pytest.raises(access.UserNotFound):
        access.set_blocked_cameras(9999, ["corridor"])


def test_deleting_a_user_removes_their_restrictions(fresh_db):
    uid = fresh_db["E1"]
    access.set_blocked_cameras(uid, ["corridor"])
    with db.get_session() as s:
        s.delete(s.get(db.User, uid))
        s.commit()
    assert access.all_blocked() == {}
