"""Looking at a screen: remembered screens + head direction with hysteresis."""
import numpy as np

from app.detection.activity_detector import (
    _CLASS_TV, _GAZE_ENTER_DEG, _STATIC_CONFIRM_HITS, _ScreenGaze, _StaticObjects, _head_direction,
    _looking_at_hands,
)

PERSON = (100, 100, 300, 500)            # head around (200, 160)
SCREEN_AHEAD = ((350, 120, 450, 200), _CLASS_TV)    # to the image-right of the head
SCREEN_BEHIND = ((0, 120, 60, 200), _CLASS_TV)      # to the image-left


def pose_facing(dx, dy, nose=(200, 160)):
    """Ears behind the nose, so the face points along (dx, dy)."""
    kxy = np.zeros((17, 2), np.float32)
    kconf = np.zeros(17, np.float32)
    kxy[0] = nose
    kxy[3] = kxy[4] = (nose[0] - 20 * dx, nose[1] - 20 * dy)
    kconf[0] = kconf[3] = kconf[4] = 0.9
    return kxy, kconf


def test_screen_is_remembered_through_missed_detections():
    objs = _StaticObjects()
    box = SCREEN_AHEAD[0]
    for k in range(_STATIC_CONFIRM_HITS):
        out = objs.update([(box, _CLASS_TV)], now=float(k))
    assert out and out[0][1] == _CLASS_TV
    assert objs.update([], now=30.0), "missed by the detector -- still there"


def test_one_off_detection_is_not_a_screen():
    objs = _StaticObjects()
    assert objs.update([((10, 10, 50, 50), _CLASS_TV)], now=0.0) == []


def test_looking_only_counts_toward_the_screen():
    g = _ScreenGaze()
    screens = [SCREEN_AHEAD, SCREEN_BEHIND]
    head = _head_direction(pose_facing(1, 0))
    screen, looking = g.update(1, head, screens, PERSON, now=0.0, max_distance_factor=1.5)
    assert looking and screen == SCREEN_AHEAD[0]
    head = _head_direction(pose_facing(0, 1))          # turned to look down/away
    _screen, looking = g.update(1, head, screens, PERSON, now=1.0, max_distance_factor=1.5)
    assert not looking


def test_small_head_moves_do_not_flicker():
    g = _ScreenGaze()
    screens = [SCREEN_AHEAD]
    g.update(1, _head_direction(pose_facing(1, 0)), screens, PERSON, now=0.0, max_distance_factor=1.5)
    # Turn ~50 deg: past ENTER but inside EXIT -> still looking.
    a = np.radians(_GAZE_ENTER_DEG + 10)
    _s, looking = g.update(1, _head_direction(pose_facing(np.cos(a), np.sin(a))), screens, PERSON,
                           now=1.0, max_distance_factor=1.5)
    assert looking
    # A new person turned the same 50 deg has not started looking.
    _s, looking = g.update(2, _head_direction(pose_facing(np.cos(a), np.sin(a))), screens, PERSON,
                           now=1.0, max_distance_factor=1.5)
    assert not looking


def test_unreadable_head_keeps_the_decision_briefly():
    g = _ScreenGaze()
    g.update(1, _head_direction(pose_facing(1, 0)), [SCREEN_AHEAD], PERSON, now=0.0, max_distance_factor=1.5)
    assert g.update(1, None, [SCREEN_AHEAD], PERSON, now=2.0, max_distance_factor=1.5)[1]
    assert not g.update(1, None, [SCREEN_AHEAD], PERSON, now=5.0, max_distance_factor=1.5)[1]


def with_wrists(pose, left, right):
    kxy, kconf = pose
    kxy[9], kxy[10] = left, right
    kconf[9] = kconf[10] = 0.9
    return kxy, kconf


def test_hands_held_together_in_line_with_the_screen_means_not_the_screen():
    """Phone held low in both hands, between the face and the screen."""
    pose = with_wrists(pose_facing(1, 0.3), (262, 178), (270, 182))
    head = _head_direction(pose)
    assert _looking_at_hands(head, pose, PERSON, SCREEN_AHEAD[0])


def test_typing_hands_apart_still_count_as_the_screen():
    pose = with_wrists(pose_facing(1, 0.3), (250, 300), (330, 290))
    head = _head_direction(pose)
    assert not _looking_at_hands(head, pose, PERSON, SCREEN_AHEAD[0])
