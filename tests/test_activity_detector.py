import numpy as np
import pytest

from app.detection.activity_detector import (
    _CLASS_KEYBOARD,
    _CLASS_LAPTOP,
    _CLASS_MOUSE,
    _containment_ratio,
    _expand_box,
    _iou,
    _is_computer_in_use,
    _is_phone_in_use,
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


def test_is_phone_in_use():
    person_bbox = (200, 200, 400, 600)  # w=200, h=400 (torso/lap around 240..550)

    # Phone held in hands in front of chest/lap
    phone_held = (280, 350, 320, 420)
    assert _is_phone_in_use(phone_held, person_bbox, None, 1920, 1080) is True

    # Phone far away across the room
    phone_far = (800, 350, 840, 420)
    assert _is_phone_in_use(phone_far, person_bbox, None, 1920, 1080) is False

    # Phone held with wrist keypoint detected
    # Keypoint 9 (left wrist), 10 (right wrist)
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kxy[9] = [300, 370]
    kconf[9] = 0.8
    assert _is_phone_in_use(phone_held, person_bbox, (kxy, kconf), 1920, 1080) is True


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
