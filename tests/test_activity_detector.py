import numpy as np
import pytest

from app.detection.activity_detector import (
    _CLASS_KEYBOARD,
    _CLASS_LAPTOP,
    _CLASS_MOUSE,
    _classify_posture,
    _DeskPhones,
    _hand_zoom_windows,
    _skin_share,
    _ZOOM_MAX_SKIN,
    ActivityDetector,
    _PhoneHold,
    _StandUpGuard,
    _containment_ratio,
    _expand_box,
    _iou,
    _is_computer_in_use,
    _PHONE_OWNER_MIN_SCORE,
    _phone_person_score,
    _posture_from_bbox,
)


def test_containment_ratio_and_iou():
    box_a = (10, 10, 50, 50)  # area = 1600
    box_b = (10, 10, 100, 100)  # area = 8100, completely contains box_a

    # box_a inside box_b -> containment is 1.0
    assert _containment_ratio(box_a, box_b) == pytest.approx(1.0)
    # box_b inside box_a -> containment is 1600 / 8100
    assert _containment_ratio(box_b, box_a) == pytest.approx(1600 / 8100)

    # Disjoint boxes
    box_c = (200, 200, 250, 250)
    assert _containment_ratio(box_a, box_c) == 0.0
    assert _iou(box_a, box_c) == 0.0


def test_expand_box():
    bbox = (100, 100, 200, 200)  # w=100, h=100
    expanded = _expand_box(bbox, mx=0.10, my=0.15, w=1920, h=1080)
    assert expanded == (90, 85, 210, 215)


def test_posture_from_bbox():
    # Standing (tall/narrow): w=100, h=200 -> h/w = 2.0 >= 1.6
    assert _posture_from_bbox((0, 0, 100, 200)) == "standing"
    # Sitting (shorter/wider): w=150, h=150 -> h/w = 1.0 < 1.6
    assert _posture_from_bbox((0, 0, 150, 150)) == "sitting"


def _pose(shoulders, hips, knees):
    """17 COCO keypoints with just shoulders/hips/knees set: each arg is
    ((x_left, y), (x_right, y))."""
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    for (l, r), (il, ir) in zip((shoulders, hips, knees), ((5, 6), (11, 12), (13, 14))):
        kxy[il], kxy[ir] = l, r
        kconf[il] = kconf[ir] = 0.9
    return kxy, kconf


def test_seated_person_facing_camera_is_not_standing():
    """High camera, person facing it: thighs project straight down, so the
    knee/torso ratio reads 'standing'. Short box + chair under hips win."""
    kxy, kconf = _pose(((200, 450), (300, 450)), ((210, 540), (290, 540)), ((215, 620), (285, 620)))
    bbox = (140, 400, 420, 665)          # wider than tall
    chair = [(150, 430, 400, 700)]
    assert _classify_posture(kxy, kconf, bbox, chairs=chair) == "sitting"
    assert _classify_posture(kxy, kconf, bbox) == "sitting"  # box shape alone outvotes the half vote


def test_standing_person_is_still_standing():
    kxy, kconf = _pose(((1500, 600), (1560, 600)), ((1505, 700), (1555, 700)), ((1505, 790), (1555, 790)))
    bbox = (1440, 565, 1620, 880)        # tall/narrow
    assert _classify_posture(kxy, kconf, bbox) == "standing"
    # An empty chair beside them must not flip it.
    assert _classify_posture(kxy, kconf, bbox, chairs=[(1340, 690, 1500, 850)]) == "standing"


def test_side_view_seated_person_is_sitting():
    """Thigh runs sideways: knee barely below the hip."""
    kxy, kconf = _pose(((300, 300), (320, 300)), ((300, 400), (320, 400)), ((400, 420), (420, 420)))
    assert _classify_posture(kxy, kconf, (260, 260, 460, 560)) == "sitting"


def test_guessed_knees_under_desk_do_not_vote_standing():
    """Legs under the desk: YOLO-pose still guesses knees (low confidence),
    far below the hips. Box is in-between, person side-on, no chair seen."""
    kxy, kconf = _pose(((300, 300), (330, 300)), ((300, 400), (330, 400)), ((305, 520), (325, 520)))
    kconf[13] = kconf[14] = 0.35
    assert _classify_posture(kxy, kconf, (250, 260, 450, 520)) == "sitting"


def test_stand_up_guard_needs_shoulders_to_rise():
    g = _StandUpGuard()
    seated = _StandUpGuard.anchors((200, 300, 400, 600), _pose(((260, 380), (340, 380)), ((270, 480), (330, 480)), ((270, 560), (330, 560))))
    for _ in range(5):
        assert g.apply(1, "sitting", seated) == "sitting"
    # A misread while still in the chair: shoulders where they were.
    assert g.apply(1, "standing", seated) == "sitting"
    # Actually standing up: shoulders 40px higher (> 0.4 x 80px width).
    up = _StandUpGuard.anchors((200, 240, 400, 600), _pose(((260, 340), (340, 340)), ((270, 450), (330, 450)), ((270, 560), (330, 560))))
    assert g.apply(1, "standing", up) == "standing"
    assert g.apply(1, "standing", up) == "standing", "stays standing after the guard let it through"


def test_stand_up_guard_lets_new_tracks_and_walkers_through():
    g = _StandUpGuard()
    box = _StandUpGuard.anchors((200, 300, 400, 600), None)
    assert g.apply(7, "standing", box) == "standing", "no seated history -> nothing to guard"
    g.apply(8, "sitting", box)
    walked = _StandUpGuard.anchors((420, 300, 620, 600), None)  # moved > 1 half-width sideways
    assert g.apply(8, "standing", walked) == "standing"


def _wrist_pose(x, y):
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kxy[10] = [x, y]
    kconf[10] = 0.9
    return kxy, kconf


def test_phone_is_held_through_short_misses_while_hand_stays():
    hold = _PhoneHold(hold_seconds=4.0)
    person, phone = (200, 200, 400, 600), (290, 350, 320, 400)
    assert hold.update(1, (phone, 0.5, 0.9), person, _wrist_pose(305, 380), now=0.0) == (phone, 0.5)
    # Missed by the detector, hand still there -> still a phone.
    assert hold.update(1, None, person, _wrist_pose(308, 384), now=1.2) == (phone, 0.5)
    # Hand moved away (phone put down) -> gone.
    assert hold.update(1, None, person, _wrist_pose(380, 560), now=1.6) is None


def test_phone_hold_runs_out():
    hold = _PhoneHold(hold_seconds=4.0)
    person, phone = (200, 200, 400, 600), (290, 350, 320, 400)
    hold.update(1, (phone, 0.5, 0.9), person, _wrist_pose(305, 380), now=0.0)
    assert hold.update(1, None, person, _wrist_pose(305, 380), now=4.5) is None


def test_track_born_at_a_known_seat_starts_sitting():
    g = _StandUpGuard(seat_learn_passes=5)
    seated = _StandUpGuard.anchors((200, 300, 400, 600), None)
    for k in range(6):
        g.apply(1, "sitting", seated, now=k)
    g.keep_only(())  # track 1 lost (occlusion)
    # New track id, same seated head position, misread as standing.
    assert g.apply(2, "standing", _StandUpGuard.anchors((205, 305, 400, 600), None), now=10) == "sitting"
    # Someone standing at that seat: head clearly higher -> not the seat.
    assert g.apply(3, "standing", _StandUpGuard.anchors((200, 180, 400, 600), None), now=11) == "standing"


def test_seats_are_forgotten_after_long_disuse():
    g = _StandUpGuard(seat_learn_passes=2, seat_forget_seconds=100)
    seated = _StandUpGuard.anchors((200, 300, 400, 600), None)
    for k in range(3):
        g.apply(1, "sitting", seated, now=k)
    g.keep_only(())
    assert g.apply(2, "standing", seated, now=500) == "standing"


def test_confident_site_model_outvotes_the_rules():
    kxy, kconf = _pose(((200, 450), (300, 450)), ((210, 540), (290, 540)), ((215, 620), (285, 620)))
    bbox = (140, 300, 300, 665)  # tall box: rules alone say standing
    assert _classify_posture(kxy, kconf, bbox) == "standing"
    assert _classify_posture(kxy, kconf, bbox, learned=("sitting", 0.95)) == "sitting"
    # An unsure model does not override clear rules.
    assert _classify_posture(kxy, kconf, bbox, learned=("sitting", 0.4)) == "standing"


def test_site_model_decides_when_no_keypoints():
    empty = (np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert _classify_posture(*empty, (0, 0, 100, 250), learned=("sitting", 0.9)) == "sitting"


def test_phone_person_score():
    person_bbox = (200, 200, 400, 600)  # w=200, h=400 (torso/lap around 240..550)
    phone_held = (280, 350, 320, 420)

    # No pose: body + zone alone must never be enough to own a phone
    assert _phone_person_score(phone_held, person_bbox, None, 1920, 1080) < _PHONE_OWNER_MIN_SCORE

    # Phone far away across the room
    phone_far = (800, 350, 840, 420)
    assert _phone_person_score(phone_far, person_bbox, None, 1920, 1080) < _PHONE_OWNER_MIN_SCORE

    # Phone held with wrist keypoint detected right next to it
    # Keypoint 9 (left wrist), 10 (right wrist)
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kxy[9] = [300, 370]
    kconf[9] = 0.8
    assert _phone_person_score(phone_held, person_bbox, (kxy, kconf), 1920, 1080) >= _PHONE_OWNER_MIN_SCORE

    # Neighbour whose box overlaps the phone but whose wrists are elsewhere
    # scores lower than the person actually holding it
    neighbour_bbox = (250, 200, 450, 600)
    kxy_n = np.zeros((17, 2), dtype=np.float32)
    kxy_n[9] = [440, 580]
    kconf_n = kconf.copy()
    assert _phone_person_score(phone_held, neighbour_bbox, (kxy_n, kconf_n), 1920, 1080) < \
        _phone_person_score(phone_held, person_bbox, (kxy, kconf), 1920, 1080)


def test_phone_beside_a_typing_hand_is_not_in_use():
    """Hand on the keyboard, phone lying ~0.2 body heights away on the desk."""
    person = (200, 200, 400, 600)            # body height 400
    phone_on_desk = (360, 500, 400, 525)
    assert _phone_person_score(phone_on_desk, person, _wrist_pose(300, 470), 1920, 1080) < _PHONE_OWNER_MIN_SCORE
    # Same phone with the hand on it: in use.
    assert _phone_person_score(phone_on_desk, person, _wrist_pose(372, 520), 1920, 1080) >= _PHONE_OWNER_MIN_SCORE


def _phone_at(cx, cy, w=40, h=25):
    return ((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), 0.6)


def test_desk_phone_never_counts_even_with_a_hand_on_it():
    """Box steady to a pixel (the measured desk phone): never in use."""
    desk = _DeskPhones()
    for k in range(20):
        jitter = k % 2                       # detector noise, 1px
        assert desk.filter([_phone_at(500 + jitter, 500)], [(500, 505, 400.0)], now=k * 0.5) == []


def test_phone_in_hand_counts_once_it_moves():
    desk = _DeskPhones()
    assert desk.filter([_phone_at(500, 500)], [], now=0.0) == [], "not proven yet"
    held = _phone_at(507, 503)               # hand shifts it ~7px
    assert desk.filter([held], [], now=0.4) == [held]


def test_held_phone_keeps_counting_with_natural_hand_motion():
    desk = _DeskPhones()
    out = []
    for k in range(30):                      # 15s of reading, hand drifting a few px
        p = _phone_at(500 + (k % 3) * 4, 500 + (k % 2) * 3)
        out = desk.filter([p], [], now=k * 0.5)
    assert out != []


def test_phone_put_down_stops_counting_after_three_seconds():
    desk = _DeskPhones()
    desk.filter([_phone_at(500, 500)], [], now=0.0)
    desk.filter([_phone_at(508, 500)], [], now=0.5)          # in hand
    assert desk.filter([_phone_at(508, 500)], [], now=2.0) != []
    assert desk.filter([_phone_at(508, 500)], [], now=4.0) == [], "still for 3.5s -> put down"
    assert desk.filter([_phone_at(516, 494)], [], now=4.5) != [], "picked up again"


def test_skin_rule_drops_a_fist_but_keeps_a_phone_in_hand():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:80, 20:80] = (120, 150, 210)        # BGR skin tone: a fist
    frame[120:180, 20:80] = (40, 30, 30)         # dark phone case
    frame[120:180, 20:28] = (120, 150, 210)      # fingers along one edge
    assert _skin_share(frame, (20, 20, 80, 80)) > _ZOOM_MAX_SKIN
    assert _skin_share(frame, (20, 120, 80, 180)) < _ZOOM_MAX_SKIN


def test_box_too_big_for_a_phone_is_never_owned():
    person = (200, 200, 400, 600)             # body height 400
    huge = (300, 400, 388, 521)               # 88 x 121
    assert _phone_person_score(huge, person, _wrist_pose(340, 450), 1920, 1080) == 0.0


def test_is_computer_in_use_rejection_of_side_keyboard():
    person_bbox = (300, 200, 500, 600)  # person seated: x in [300, 500]

    # Side table keyboard (to the left of person, e.g. x in [150, 280])
    side_keyboard = (150, 350, 280, 400)

    # Person holding phone -> should definitely NOT flag side keyboard as computer in use
    assert _is_computer_in_use(
        side_keyboard, _CLASS_KEYBOARD, person_bbox, None, phone_detected=True, img_w=1920, img_h=1080
    ) is False

    # Even without phone, side keyboard that is mostly outside workspace zone should not flag
    assert _is_computer_in_use(
        side_keyboard, _CLASS_KEYBOARD, person_bbox, None, phone_detected=False, img_w=1920, img_h=1080
    ) is False


def test_is_computer_in_use_active_laptop_and_typing():
    person_bbox = (300, 200, 500, 600)

    # Laptop directly in front of person
    front_laptop = (330, 350, 470, 480)
    assert _is_computer_in_use(
        front_laptop, _CLASS_LAPTOP, person_bbox, None, phone_detected=False, img_w=1920, img_h=1080
    ) is True

    # Active typing with wrists on keyboard
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kxy[9] = [360, 400]  # wrist directly on laptop/keyboard
    kconf[9] = 0.9
    assert _is_computer_in_use(
        front_laptop, _CLASS_KEYBOARD, person_bbox, (kxy, kconf), phone_detected=False, img_w=1920, img_h=1080
    ) is True


def test_phone_use_confidence_builds_up_and_falls_away():
    from app.detection.activity_detector import _PhoneUseConfidence
    c = _PhoneUseConfidence()
    strong = ((0, 0, 10, 20), 0.6, 1.0)          # detected, sure, firmly in hand
    levels = [c.update(1, _PhoneUseConfidence.evidence(strong, held=False)) for _ in range(6)]
    assert levels[0] < 0.5 < levels[-1], "climbs over a few passes, not at once"
    assert levels[-1] > 0.85
    after = [c.update(1, _PhoneUseConfidence.evidence(None, held=False)) for _ in range(6)]
    assert after[-1] < 0.15, "falls once the phone is gone"
    assert _PhoneUseConfidence.evidence(None, held=True) == 0.5



def test_slivers_of_a_desk_phone_do_not_make_it_move():
    """Measured on Camera 01: the full box of a desk phone, now and then a
    5-15px sliver of it. It must still be parked as lying on the desk."""
    desk = _DeskPhones()
    full = ((500, 366, 553, 395), 0.4)
    slivers = [((500, 376, 515, 394), 0.23), ((500, 378, 509, 394), 0.23), ((500, 380, 505, 390), 0.25)]
    t, out = 0.0, None
    for k in range(20):
        obs = [full] + ([slivers[k % 3]] if k % 3 == 0 else [])
        out = desk.filter(obs, [], now=t)
        t += 0.8
    assert out == [], "a phone that never moved is on the desk"



def _pose_hand_shoulders(hand, shoulders_y):
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kxy[10] = hand
    kxy[5], kxy[6] = (180, shoulders_y), (260, shoulders_y)
    kconf[10] = kconf[5] = kconf[6] = 0.9
    return kxy, kconf


def test_phone_on_the_table_beside_a_typing_hand_is_not_held():
    """Measured reference case: hand 0.078 body heights from the phone's edge,
    phone 0.22 body heights below the shoulders (desk level)."""
    person = (100, 100, 400, 600)                       # body height 500
    phone = (250, 380, 290, 420)                        # centre y 400 -> 0.22 below shoulders at 290
    hand = (250, 380 - 39)                              # 39px = 0.078 * 500 above the phone
    assert _phone_person_score(phone, person, _pose_hand_shoulders(hand, 290), 1920, 1080) < _PHONE_OWNER_MIN_SCORE


def test_phone_held_out_at_chest_height_still_counts():
    person = (100, 100, 400, 600)
    phone = (250, 280, 290, 320)                        # centre y 300, at shoulder level
    hand = (250, 280 - 39)
    assert _phone_person_score(phone, person, _pose_hand_shoulders(hand, 290), 1920, 1080) >= _PHONE_OWNER_MIN_SCORE


def test_phone_in_the_lap_with_the_hand_on_it_counts():
    person = (100, 100, 400, 600)
    phone = (250, 430, 290, 470)                        # low, in the lap
    hand = (270, 450)                                   # on it
    assert _phone_person_score(phone, person, _pose_hand_shoulders(hand, 290), 1920, 1080) >= _PHONE_OWNER_MIN_SCORE
