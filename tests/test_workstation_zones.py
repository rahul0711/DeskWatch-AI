"""Workstation zones (Admin -> Workstations): the seat box votes for sitting,
and the person in it is matched to that seat's screen."""
import numpy as np

from app.detection.activity_detector import _classify_posture, _seat_zone_for

FRONT = {"name": "Front", "seat": (370, 330, 520, 560), "screen": (585, 330, 650, 470)}
BACK = {"name": "Back", "seat": (310, 220, 420, 370), "screen": (545, 180, 605, 320)}


def _pose(shoulders, hips, knees=None):
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    pairs = [(shoulders, (5, 6)), (hips, (11, 12))] + ([(knees, (13, 14))] if knees else [])
    for (l, r), (il, ir) in pairs:
        kxy[il], kxy[ir] = l, r
        kconf[il] = kconf[ir] = 0.9
    return kxy, kconf


def test_no_zones_means_no_seat():
    assert _seat_zone_for((380, 300, 500, 600), None, []) is None


def test_hips_pick_the_seat():
    pose = _pose(((400, 360), (480, 360)), ((410, 450), (470, 450)))
    assert _seat_zone_for((380, 300, 500, 600), pose, [FRONT, BACK])["name"] == "Front"


def test_box_point_used_when_hips_hidden():
    # no hip keypoints: anchor is 60% down the box -> (440, 480), inside Front
    assert _seat_zone_for((380, 300, 500, 600), None, [FRONT, BACK])["name"] == "Front"
    # person far from both seats
    assert _seat_zone_for((1200, 300, 1300, 600), None, [FRONT, BACK]) is None


def test_overlapping_seats_pick_nearest_centre():
    # hips at (400, 345): inside both Back (ends y=370) and Front (starts y=330);
    # Back's centre (365, 295) is nearer than Front's (445, 445)
    pose = _pose(((380, 260), (420, 260)), ((390, 345), (410, 345)))
    assert _seat_zone_for((360, 230, 440, 400), pose, [FRONT, BACK])["name"] == "Back"


def test_seat_zone_outvotes_tall_box_when_knees_hidden():
    """Desk hides the legs and the person sits upright: the box is tall and
    narrow (reads standing). In their seat it is sitting."""
    kxy, kconf = _pose(((400, 340), (460, 340)), ((405, 450), (455, 450)))
    bbox = (390, 300, 470, 520)  # h/w = 2.75
    assert _classify_posture(kxy, kconf, bbox) == "standing"
    assert _classify_posture(kxy, kconf, bbox, in_seat=True) == "sitting"


def test_seat_zone_does_not_overrule_clear_standing():
    """Long straight legs fully visible: standing next to (or in) the seat zone."""
    kxy, kconf = _pose(((400, 200), (460, 200)), ((405, 330), (455, 330)), ((405, 470), (455, 470)))
    bbox = (390, 170, 470, 600)
    assert _classify_posture(kxy, kconf, bbox, in_seat=True) == "standing"
