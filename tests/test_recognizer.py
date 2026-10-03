import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2
from app.detection.face_recognizer import FaceRecognizer

def test_face_recognizer_init():
    # Test initialization with a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        recognizer = FaceRecognizer(known_faces_dir=tmp_dir)
        assert recognizer.known_faces_dir == Path(tmp_dir)
        assert len(recognizer.known_embeddings) == 0

def test_face_recognizer_unknown():
    # Test recognizing on empty database
    with tempfile.TemporaryDirectory() as tmp_dir:
        recognizer = FaceRecognizer(known_faces_dir=tmp_dir)
        dummy_face = np.zeros((160, 160, 3), dtype=np.uint8)
        name, dist = recognizer.recognize(dummy_face)
        assert name == "Unknown"
        assert dist == 1.0
