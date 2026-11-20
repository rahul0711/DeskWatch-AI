"""Per-camera face / body-movement switches (app/web/camera_features.py)."""
from app.web.camera_features import CameraFeatures, features_for


def test_switches_from_the_table_win():
    table = {"camera-01": CameraFeatures(face=False, activity=True)}
    assert features_for("Camera-01", table, default_activity=False) == CameraFeatures(False, True)


def test_camera_missing_from_table_keeps_env_defaults():
    table = {"camera-01": CameraFeatures(face=False, activity=False)}
    assert features_for("custom-ab12", table, default_activity=True) == CameraFeatures(True, True)


def test_unreadable_table_never_switches_a_camera_off():
    assert features_for("inside", None, default_activity=False) == CameraFeatures(True, False)
