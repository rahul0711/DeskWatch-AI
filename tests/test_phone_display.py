"""The Phone view draws a phone only while it was recently seen, though the
recorded session keeps its longer grace (one phone use = one session)."""
from dataclasses import dataclass, field

from app.activity.analyzer import CameraActivityAnalyzer
from app.activity.config import ActivityConfig


@dataclass
class Box:
    bbox: tuple = (100, 100, 300, 600)
    confidence: float = 0.9


@dataclass
class Track:
    track_id: int = 1
    posture: str = "sitting"
    phone_in_use: bool = False
    computer_in_use: bool = False
    monitor_facing: bool = False
    monitor_bbox: tuple | None = None
    keypoints: object = None
    keypoints_conf: object = None
    name: str = "Unknown"
    face: Box = field(default_factory=Box)


def test_phone_label_drops_soon_after_the_phone_is_put_down():
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=None, enabled_activities=("PHONE_INTERACTION",))
    t = 0.0
    for _ in range(12):                          # 3s on the phone
        tr = Track(phone_in_use=True); a.observe([tr], now=t); t += 0.25
    assert tr.phone_in_use
    shown = []
    for _ in range(24):                          # 6s drinking water, no phone seen
        tr = Track(phone_in_use=False); a.observe([tr], now=t); shown.append(tr.phone_in_use); t += 0.25
    assert shown[0] is True, "a brief miss is bridged"
    assert shown[-1] is False, "gone from screen within ~3s"
    state = a._states.state_for_track(1)
    assert state.is_on("phone"), "the recorded session is still open (10s grace)"


def test_confirmed_phone_use_is_a_live_violation_with_confidence():
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=None, enabled_activities=("PHONE_INTERACTION",))
    t = 0.0
    assert a.phone_violations(now=t) == []
    for _ in range(8):                           # 2s with a phone in hand
        tr = Track(phone_in_use=True)
        tr.phone_use_confidence = 0.8
        a.observe([tr], now=t); t += 0.25
    v = a.phone_violations(now=t)
    assert len(v) == 1 and v[0]["name"] == "Unidentified person"
    assert v[0]["seconds"] > 1.0 and v[0]["confidence"] == 0.8


def test_low_confidence_phone_use_is_not_a_violation():
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=None, enabled_activities=("PHONE_INTERACTION",))
    t = 0.0
    for _ in range(8):
        tr = Track(phone_in_use=True)
        tr.phone_use_confidence = 0.3
        a.observe([tr], now=t); t += 0.25
    assert a.phone_violations(now=t) == []


def test_violation_alert_clears_soon_after_the_phone_is_put_down():
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=None, enabled_activities=("PHONE_INTERACTION",))
    t = 0.0
    for _ in range(8):
        tr = Track(phone_in_use=True)
        tr.phone_use_confidence = 0.8
        a.observe([tr], now=t); t += 0.25
    assert a.phone_violations(now=t)
    for _ in range(20):                          # 5s without the phone
        tr = Track(phone_in_use=False)
        a.observe([tr], now=t); t += 0.25
    assert a.phone_violations(now=t) == []


class Repo:
    def __init__(self):
        self.events = []

    def record_many(self, events):
        self.events.extend(events)


def _use_then_stop(conf):
    repo = Repo()
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=repo, enabled_activities=("PHONE_INTERACTION",))
    t = 0.0
    for _ in range(12):                          # 3s phone use
        tr = Track(phone_in_use=True); tr.phone_use_confidence = conf
        a.observe([tr], now=t); t += 0.25
    for _ in range(60):                          # 15s without -> session closes
        a.observe([Track(phone_in_use=False)], now=t); t += 0.25
    return [e for e in repo.events if e.activity_type == "PHONE_INTERACTION"]


def test_confident_phone_use_is_stored_doubtful_one_is_not():
    assert len(_use_then_stop(0.8)) == 1
    assert _use_then_stop(0.3) == []



def test_main_label_drops_phone_once_it_is_no_longer_seen():
    """All view: a person who put the phone down (session still in its
    recording grace) must not keep showing PHONE as their activity."""
    a = CameraActivityAnalyzer("camera-01", ActivityConfig(), repository=None)
    t = 0.0
    for _ in range(12):
        tr = Track(phone_in_use=True); tr.phone_use_confidence = 0.8
        a.observe([tr], now=t); t += 0.25
    assert tr.activity == "PHONE_INTERACTION"
    for _ in range(20):                          # 5s, hands behind the head, no phone
        tr = Track(phone_in_use=False)
        a.observe([tr], now=t); t += 0.25
    assert a._states.state_for_track(1).is_on("phone"), "still inside the recording grace"
    assert tr.activity != "PHONE_INTERACTION"
