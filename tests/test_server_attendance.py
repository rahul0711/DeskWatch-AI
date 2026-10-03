import tempfile
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.web.auth import DEMO_TOKEN
from app.web.server import app, _save_custom_cameras, _load_custom_cameras, _sessions, CameraWebSession
from app.config import CameraConfig, load_config
from app.processing.pipeline import FacePipeline


def test_login_and_token():
    client = TestClient(app)
    resp = client.post("/api/login", json={"username": "AIFace", "password": "India@AI"})
    assert resp.status_code == 200
    assert resp.json()["token"] == DEMO_TOKEN


def test_head_count_source_validation(monkeypatch):
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {DEMO_TOKEN}"}

    # Register a mock session in _sessions
    cam = CameraConfig(name="Mock Cam", ip="127.0.0.1")
    lock = threading.Lock()
    mock_pipeline = type("MockPipeline", (), {"head_count_source": "face"})()
    mock_session = type("MockSession", (), {
        "id": "mock-cam",
        "camera": cam,
        "head_count_source": "face",
        "pipeline": mock_pipeline,
        "_lock": lock,
    })()

    monkeypatch.setitem(_sessions, "mock-cam", mock_session)

    # Test invalid source rejected with 422
    resp = client.patch("/api/cameras/mock-cam/head-count-source", json={"source": "invalid_mode"}, headers=headers)
    assert resp.status_code == 422

    # Test attendance is accepted
    resp = client.patch("/api/cameras/mock-cam/head-count-source", json={"source": "attendance"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["head_count_source"] == "attendance"
    assert mock_session.head_count_source == "attendance"
    assert mock_pipeline.head_count_source == "attendance"


def test_save_sent_image_writes_to_data(tmp_path, monkeypatch):
    import app.web.server as server_module
    monkeypatch.setattr(server_module, "_ATTENDANCE_IMAGES_DIR", tmp_path / "attendance")
    monkeypatch.setattr(server_module, "_RECOGNIZE_IMAGES_DIR", tmp_path / "recognized_faces")

    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    now = datetime.now()

    # Create dummy CameraConfig
    cam = CameraConfig(name="Test Cam", ip="127.0.0.1")
    # Mock class instance method directly
    session_stub = type("SessionStub", (), {
        "id": "test-cam",
        "camera": cam,
        "_save_sent_image": CameraWebSession._save_sent_image,
    })()

    path = session_stub._save_sent_image(crop, track_id=42, timestamp=now)
    assert path is not None
    assert path.exists()
    assert "track42" in path.name
    assert path.parent.name == "test-cam"
    assert (tmp_path / "recognized_faces" / "test-cam" / path.name).exists()


def test_session_attendance_head_count_history():
    cam = CameraConfig(name="Test Cam", ip="127.0.0.1")
    minute_key = datetime.now().strftime("%Y-%m-%d %H:%M")
    lock = threading.Lock()
    session_stub = type("SessionStub", (), {
        "id": "test-cam",
        "camera": cam,
        "head_count_source": "attendance",
        "_lock": lock,
        "_minute_buckets": {
            minute_key: {
                "face_sum": 6, "face_samples": 2,
                "person_sum": 0, "person_samples": 0,
                "best_sum": 6, "best_samples": 2,
            }
        },
        "head_count_history": CameraWebSession.head_count_history,
    })()

    history = session_stub.head_count_history()
    assert len(history) == 1
    assert history[0]["minute"] == minute_key
    assert history[0]["avg_faces"] == 3


def test_tapo_camera_email_config():
    cam = CameraConfig(
        name="Tapo Living Room",
        ip="192.168.1.50",
        rtsp_port=554,
        username="user@example.com",
        password="tapopassword",
        rtsp_path="/stream1",
    )
    assert cam.rtsp_url == "rtsp://user%40example.com:tapopassword@192.168.1.50:554/stream1"
    assert "tapopassword" not in cam.rtsp_url_masked
    assert "***" in cam.rtsp_url_masked


def test_test_camera_connection_validation():
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {DEMO_TOKEN}"}
    resp = client.post(
        "/api/cameras/test-connection",
        json={"ip": "", "username": "admin", "password": "123"},
        headers=headers,
    )
    assert resp.status_code == 422


def test_server_cooldown_suppression(monkeypatch, tmp_path):
    import app.web.server as server_module
    monkeypatch.setattr(server_module, "_ATTENDANCE_IMAGES_DIR", tmp_path / "attendance")
    monkeypatch.setattr(server_module, "_RECOGNIZE_IMAGES_DIR", tmp_path / "recognized_faces")

    # Mock recognize_face to return a successful punch
    monkeypatch.setattr(
        server_module,
        "recognize_face",
        lambda crop: {"success": True, "name": "Alice", "EmployeeId": "E123", "message": "Punch OK"},
    )

    cam = CameraConfig(name="Test Cam", ip="127.0.0.1")
    lock = threading.Lock()
    session_stub = type("SessionStub", (), {
        "id": "test-cam",
        "camera": cam,
        "_lock": lock,
        "_user_last_punched": {},
        "_PUNCH_COOLDOWN_S": 600.0,
        "_recognitions": [],
        "_MAX_RECOGNITIONS": 50,
        # Activity analysis off for this camera -- the identity hand-off to
        # app/activity/ is skipped, which is the path a camera outside
        # ACTIVITY_CAMERAS takes.
        "_activity": None,
        "_face_identities": {},
        "_save_sent_image": CameraWebSession._save_sent_image,
        "_recognize_and_record": CameraWebSession._recognize_and_record,
    })()

    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    # First punch: image saved, response has original message
    session_stub._recognize_and_record(crop, track_id=1)
    assert len(session_stub._recognitions) == 1
    assert session_stub._recognitions[0]["image_path"] is not None
    assert session_stub._recognitions[0]["response"]["message"] == "Punch OK"

    # Second punch within cooldown: no image saved, message updated to cooldown notice
    session_stub._recognize_and_record(crop, track_id=2)
    assert len(session_stub._recognitions) == 2
    assert session_stub._recognitions[1]["image_path"] is None
    assert "done punching" in session_stub._recognitions[1]["response"]["message"]



def test_recognize_hands_identity_to_activity_layer(tmp_path, monkeypatch):
    """A matched /Recognize result must be published for the activity layer,
    keyed by FACE track id, so identity_binder can transfer it onto the
    person track that owns that face."""
    import threading

    import numpy as np

    from app.web import server as server_module
    from app.web.server import CameraWebSession
    from app.config import CameraConfig

    monkeypatch.setattr(server_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(server_module, "_ATTENDANCE_IMAGES_DIR", tmp_path / "attendance")
    monkeypatch.setattr(
        server_module,
        "recognize_face",
        lambda crop: {"success": True, "EmployeeName": "Rahul", "EmployeeId": "E123"},
    )

    cam = CameraConfig(name="Test Cam", ip="127.0.0.1")
    session_stub = type("SessionStub", (), {
        "id": "test-cam",
        "camera": cam,
        "_lock": threading.Lock(),
        "_user_last_punched": {},
        "_PUNCH_COOLDOWN_S": 600.0,
        "_recognitions": [],
        "_MAX_RECOGNITIONS": 50,
        "_activity": object(),  # truthy: activity enabled for this camera
        "_face_identities": {},
        "_save_sent_image": CameraWebSession._save_sent_image,
        "_recognize_and_record": CameraWebSession._recognize_and_record,
    })()

    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    session_stub._recognize_and_record(crop, track_id=7)

    assert 7 in session_stub._face_identities
    identity = session_stub._face_identities[7]
    assert identity.person_id == "E123"
    assert identity.person_name == "Rahul"
