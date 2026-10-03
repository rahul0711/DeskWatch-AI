import numpy as np

from app.detection.face_detector import DetectedFace
from app.tracking.simple_tracker import SimpleIouTracker


def _face(bbox):
    return DetectedFace(bbox=bbox, confidence=0.9, landmarks=np.zeros((5, 2)))


def test_same_position_keeps_same_id():
    tracker = SimpleIouTracker()
    t1 = tracker.update([_face((10, 10, 60, 60))])
    t2 = tracker.update([_face((12, 11, 62, 61))])  # nearly identical position
    assert t1[0].track_id == t2[0].track_id


def test_new_face_gets_new_id():
    tracker = SimpleIouTracker()
    t1 = tracker.update([_face((10, 10, 60, 60))])
    t2 = tracker.update([_face((10, 10, 60, 60)), _face((500, 500, 550, 550))])
    ids = {t.track_id for t in t2}
    assert t1[0].track_id in ids
    assert len(ids) == 2


def test_disappearing_face_frees_up_gracefully():
    tracker = SimpleIouTracker(max_missed_frames=1)
    tracker.update([_face((10, 10, 60, 60))])
    result = tracker.update([])
    assert result == []
