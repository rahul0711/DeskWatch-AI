import cv2
import numpy as np
import pytest
from app.detection.attendance_detector import InsightFaceAttendanceDetector
from app.detection.face_detector import DetectedFace


def test_insightface_attendance_detector_init():
    detector = InsightFaceAttendanceDetector(conf_threshold=0.5, det_size=320)
    assert detector._conf == 0.5
    assert detector._det_size == (320, 320)


def test_insightface_attendance_detector_detect_blank():
    detector = InsightFaceAttendanceDetector(conf_threshold=0.5, det_size=320)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.detect(blank_frame)
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_insightface_attendance_detector_output_format():
    detector = InsightFaceAttendanceDetector(conf_threshold=0.3, det_size=320)
    # Synthetic frame test
    test_img = np.full((320, 320, 3), 128, dtype=np.uint8)
    faces = detector.detect(test_img)
    assert isinstance(faces, list)
    for f in faces:
        assert isinstance(f, DetectedFace)
        assert len(f.bbox) == 4
        assert 0.0 <= f.confidence <= 1.0
