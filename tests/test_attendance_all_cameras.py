"""Attendance from every camera: one punch per person per cooldown."""
import threading
import time
from types import SimpleNamespace

import numpy as np

from app.web import server


def face(uid, score=0.6, bbox=(100, 100, 160, 170)):
    return SimpleNamespace(registered_user_id=uid, registered_score=score, face=SimpleNamespace(bbox=bbox))


def recorder(monkeypatch):
    rec = server._AttendanceRecorder()
    rec._enabled = True
    punches = []
    done = threading.Event()

    def fake_punch(uid, camera, crop, score, face_w):
        punches.append((uid, camera))
        with rec._lock:
            rec._last[uid] = time.time()
            rec._busy.discard(uid)
        done.set()

    monkeypatch.setattr(rec, "_punch", fake_punch)
    return rec, punches, done


def test_registered_face_on_any_camera_is_punched_once(monkeypatch):
    rec, punches, done = recorder(monkeypatch)
    frame = np.zeros((480, 640, 3), np.uint8)
    rec.offer("Camera 01", [face(130)], frame)
    done.wait(2)
    for _ in range(5):                                  # seen again, same and other camera
        rec.offer("Camera 01", [face(130)], frame)
        rec.offer("Corridor", [face(130)], frame)
    time.sleep(0.2)
    assert punches == [(130, "Camera 01")]


def test_unregistered_faces_are_never_punched(monkeypatch):
    rec, punches, _ = recorder(monkeypatch)
    rec.offer("Camera 01", [face(None)], np.zeros((480, 640, 3), np.uint8))
    time.sleep(0.2)
    assert punches == []
