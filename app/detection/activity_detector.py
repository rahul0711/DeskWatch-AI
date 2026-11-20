"""Per-person activity observation: posture, phone/computer/monitor interaction.

Two models per frame, both stock/pretrained (no fine-tuning, no new dataset):

  1. Object detection + tracking -- a COCO checkpoint (YOLOv8x by default,
     see app/activity/config.py) looking for person/laptop/mouse/keyboard/
     cell-phone/tv classes, run via Ultralytics' built-in .track()
     (BoT-SORT) so each person keeps a stable ID across frames without
     needing SimpleIouTracker.
  2. Pose estimation (YOLOv8x-pose by default) -- a separate, untracked,
     per-frame pass giving 17 COCO keypoints per person, used to classify
     sitting vs standing, to decide phone/computer/monitor interaction, and
     (in the parent process) to measure per-joint movement.

Unlike every other detector in this package, detect() returns already-tracked
list[TrackedFace] (BoT-SORT assigns the track_id), not list[DetectedFace] --
see PROVIDES_TRACK_IDS below and app/processing/process_detector.py, which
checks it to skip wrapping this detector's output in another tracker.

Deliberately STATELESS across frames apart from the short anti-flicker
buffers below: every timer, duration, movement score, state transition and
event lives in app/activity/ in the parent process, because this class runs
in a worker subprocess that can be restarted independently and must not own
anything whose loss would reset a person's accumulated durations.

Model paths and every threshold arrive as __init__ kwargs (from
ActivityConfig.detector_kwargs) rather than being read from the environment
here -- this class is constructed inside a spawn()ed subprocess, so its
arguments have to be plain picklable values.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.detection.face_detector import DetectedFace
from app.tracking.byte_tracker import ByteTrackFaceTracker
from app.tracking.simple_tracker import TrackedFace

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
# Defaults kept for direct/legacy construction; the configured paths come in
# through __init__ (see app/activity/config.py).
_OBJ_MODEL = _MODEL_DIR / "yolov8x.pt"
_POSE_MODEL = _MODEL_DIR / "yolov8x-pose.pt"
_TRACKER_CFG = _MODEL_DIR / "activity_botsort.yaml"
_PERSON_TRACKER_CFG = _MODEL_DIR / "activity_person_bytetrack.yaml"

# COCO class ids (stock Ultralytics checkpoint): person, tv/monitor, laptop,
# mouse, keyboard, cell phone.
_CLASS_PERSON = 0
_CLASS_CHAIR = 56
_CLASS_TV = 62
_CLASS_LAPTOP = 63
_CLASS_MOUSE = 64
_CLASS_KEYBOARD = 66
_CLASS_PHONE = 67
_ACTIVITY_CLASSES = [
    _CLASS_PERSON, _CLASS_TV, _CLASS_LAPTOP, _CLASS_MOUSE, _CLASS_KEYBOARD, _CLASS_PHONE,
]
_COMPUTER_CLASSES = {_CLASS_LAPTOP, _CLASS_MOUSE, _CLASS_KEYBOARD}

# COCO-pose keypoint indices.
_KP_NOSE = 0
_KP_EYE_L, _KP_EYE_R = 1, 2
_KP_EAR_L, _KP_EAR_R = 3, 4
_KP_SHOULDER_L, _KP_SHOULDER_R = 5, 6
_KP_ELBOW_L, _KP_ELBOW_R = 7, 8
_KP_WRIST_L, _KP_WRIST_R = 9, 10
_KP_HIP_L, _KP_HIP_R = 11, 12
_KP_KNEE_L, _KP_KNEE_R = 13, 14
_KP_ANKLE_L, _KP_ANKLE_R = 15, 16
_KP_MIN_CONF = 0.25

_POSE_IOU_MATCH_THRESHOLD = 0.3

# Confidence thresholds per object class (defaults; overridable via __init__)
_PHONE_CONF_THRESHOLD = 0.12        # Cell phones are small and hand-occluded; lower threshold catches them
_COMPUTER_CONF_THRESHOLD = 0.25     # Keyboards/mice/laptops on desks
_MONITOR_CONF_THRESHOLD = 0.30      # tv/monitor class
_PROXIMITY_PHONE_CONTAINMENT = 0.20 # Phone containment ratio inside upper-body/hand zone
_PROXIMITY_COMPUTER_CONTAINMENT = 0.35 # Computer peripheral containment inside active workspace zone
# Minimum _phone_person_score() for a phone to be assigned to anyone. Body +
# zone alone tops out at 0.35, so this requires a visible wrist near the
# phone -- body containment is never sufficient by itself.
_PHONE_OWNER_MIN_SCORE = 0.40

# Posture defaults (overridable via __init__ / ActivityConfig)
_SITTING_KNEE_TORSO_RATIO = 0.5
_STANDING_BBOX_ASPECT = 1.6
# Chairs are only a posture cue (a person whose hips are on one is seated),
# so a fairly confident detection is enough.
_CHAIR_CONF_THRESHOLD = 0.30
# Shoulder width / torso length above which a person counts as facing the
# camera -- the view where the knee/torso ratio cannot tell sitting from
# standing (see _classify_posture).
_FRONTAL_SHOULDER_TORSO = 0.5
# Knees hidden under a desk still come back from YOLO-pose as guesses with
# moderate confidence (often 0.3-0.5), placed far below the hips -- exactly
# what a standing person looks like. Knees must be clearly seen to vote.
_KNEE_MIN_CONF = 0.5
# Stand-up guard (_StandUpGuard): a seated person is only accepted as
# standing once their shoulders rise by this many shoulder-widths above
# where they sat, or they move this many shoulder-widths away from it.
_STAND_UP_MIN_RISE = 0.4
_STAND_UP_MIN_MOVE = 1.0
# Seat memory (_StandUpGuard): a track that sat this many passes in a row
# teaches the guard where a seated head is; seats unused this long are
# forgotten (chairs get moved).
_SEAT_LEARN_PASSES = 25
_SEAT_FORGET_SECONDS = 4 * 3600
_MAX_SEATS = 60

# Phone hold (_PhoneHold): a phone in use is often missed for a pass or two
# (tilted, covered by the thumb, motion blur). Keep it while a wrist stays
# where the phone was, for at most this long after the last real sighting.
_PHONE_HOLD_SECONDS = 10.0
_PHONE_HOLD_MAX_DIST = 0.10   # wrist -> last phone centre, in body heights

# A phone counts as IN HAND only with a wrist within this distance of its
# edge, in body heights (_phone_person_score).
_PHONE_IN_HAND_DIST = 0.10
# A phone a hand is only NEAR (not on) counts when its centre is at most this
# far below the shoulders, in body heights (chest/face level, held out).
_PHONE_RAISED_BELOW_SHOULDER = 0.12
# Desk phones (_DeskPhones), judged by the phone's OWN motion. Measured on
# Camera 01: a phone lying beside a typing hand kept its box within 1px for
# 10s+; a phone in a hand never holds that still.
_DESK_PHONE_STILL_SECONDS = 3.0   # pixel-still this long -> lying on the desk
_DESK_PHONE_MOVE_EPS = 0.12       # "moved" = centre shifted by this many phone widths
_DESK_PHONE_MATCH = 0.6           # same phone if within this many widths of its anchor
_DESK_PHONE_FORGET_SECONDS = 30.0
# A phone box longer than this share of the person's height is not a phone
# (a mouse pad, a hand, a book).
_PHONE_MAX_BODY_SHARE = 0.30

# Hand zoom check (_hand_zoom_windows): a square around each visible wrist,
# this many body heights wide, re-checked for phones at zoom. Measured on
# Camera 01, a phone held to the ear: nothing in the full frame at any
# resolution, 0.28 in a 0.6-body-height window around the hand.
_HAND_ZOOM_WINDOW = 0.6
_HAND_ZOOM_MIN_PX = 96
_HAND_ZOOM_UPSCALE_TO = 384     # each window is resized to this before the pass
_HAND_ZOOM_MAX_WINDOWS = 12
# Windows checked per pass (YOLOv8x on each costs about a third of a full
# frame pass), most promising first -- see ActivityDetector._pick_zoom_windows.
_HAND_ZOOM_PER_PASS = 3
# The zoom check's typical false hit is a fist or fingers at the face. Shape
# cannot tell them apart (a phone held toward the face at an angle is as
# square as a fist), colour can: measured on Camera 01, two real phones in
# hand (at the ear, in front of the face) had 0% skin inside their box, two
# hands at the chin 69% and 97%. Zoom-only finds with more skin than this
# are hands, not phones.
_ZOOM_MAX_SKIN = 0.45
# For RF-DETR finds, a looser limit: a phone wrapped in both hands is partly
# skin. Measured: phones 0.00, 0.00, 0.55 (held in both hands); fists at the
# chin / mouth 0.69, 0.97.
_RFDETR_MAX_SKIN = 0.62
# Phone hints this faint in the full frame are not phones, but they mark a
# hand worth zooming into first.
_PHONE_HINT_CONF = 0.03

# Monitor-interaction defaults
_MONITOR_MAX_DISTANCE_FACTOR = 1.5
_MONITOR_FACING_TOLERANCE = 0.55

# Fixed objects (_StaticObjects): monitors, laptops, keyboards, mice do not
# move, so once seen in the same place a few times they are remembered and
# used every pass -- a missed detection (screen glare, an arm in front) no
# longer makes them vanish. Forgotten after this long unseen (desk moved).
_STATIC_CONFIRM_HITS = 4
_STATIC_FORGET_SECONDS = 600.0
_STATIC_MATCH_IOU = 0.3
# Looking at a screen (_ScreenGaze): the head's direction (ears -> nose) must
# point at the screen within ENTER degrees to start looking, and keeps
# looking until it turns past EXIT degrees (no flicker on small head moves).
_GAZE_ENTER_DEG = 40.0
_GAZE_EXIT_DEG = 60.0
# Head direction briefly unreadable (hand over face, turned away from the
# camera): keep the last decision this long rather than dropping it.
_GAZE_HOLD_SECONDS = 3.0

# Ultralytics' default resizes the long side to 640px before inference. On a
# 1920x1080+ main-stream frame that's a ~3x downscale -- fine for a
# person-sized box, but it can shrink a phone or mouse past the point YOLO
# can find it at all. Measured on this camera: at 640 the phone and mouse
# classes never appeared; at 960 both did.
_IMG_SIZE = 960

# Anti-flicker: how many recent frames to keep per track, and how many of
# them must agree before a raw per-frame hit is reported as an interaction.
# This is cheap local hysteresis -- the real temporal confirmation (and all
# duration accounting) happens in app/activity/state_manager.py.
_SMOOTH_WINDOW = 5
_PHONE_MIN_VOTES = 3    # phone seen in 3 of the last 5 passes (temporal confirmation)
_COMPUTER_MIN_VOTES = 3
_MONITOR_MIN_VOTES = 2


def _containment_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0
    ix1c, iy1c = max(ix1, ox1), max(iy1, oy1)
    ix2c, iy2c = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, ix2c - ix1c) * max(0, iy2c - iy1c)
    return inter / inner_area


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _expand_box(
    bbox: tuple[int, int, int, int],
    mx: float,
    my: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * mx), int(bh * my)
    return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))


def _posture_from_bbox(
    bbox: tuple[int, int, int, int],
    standing_aspect: float = _STANDING_BBOX_ASPECT,
) -> str:
    """Fallback when no usable pose keypoints are available: a tall/narrow
    person box reads as standing, a shorter/wider one as sitting.

    Also considers vertical position in frame: a person whose box top is in
    the upper third of the frame is more likely standing (further from the
    ceiling camera, so their full body is visible), while someone whose box
    top is in the lower two-thirds is closer/seated at a desk.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return "unknown"
    aspect = h / w
    # Primary: aspect ratio
    if aspect >= standing_aspect:
        return "standing"
    # Heuristic: on a ceiling camera, a person whose bbox top is high in
    # frame (small y1) with moderate aspect is typically standing further
    # back. Lower the aspect requirement for boxes in the upper portion.
    if aspect >= standing_aspect * 0.75 and y1 < 250:
        return "standing"
    return "sitting"


# Weight of "this person is inside a drawn seat zone" in the posture vote:
# outweighs any single cue and the ambiguous frontal reading (0.5 + 1), but
# straight legs seen from the side plus a tall box (1 + 1) still say
# standing -- someone standing at their own desk. Same 24/24 on the site
# posture set as a stronger 2.5, which no rule could ever overrule.
_SEAT_ZONE_VOTE = 1.5


def _seat_zone_for(bbox, pose, zones_px, kp_min_conf: float = _KP_MIN_CONF):
    """The workstation zone this person sits in, or None. Anchor: the hips
    when visible, else a point 60% down the person box (where hips are)."""
    if not zones_px:
        return None
    x1, y1, x2, y2 = bbox
    ax, ay = (x1 + x2) / 2, y1 + 0.6 * (y2 - y1)
    if pose is not None:
        kxy, kconf = pose
        if len(kconf) > _KP_HIP_R and min(kconf[_KP_HIP_L], kconf[_KP_HIP_R]) >= kp_min_conf:
            ax = float(kxy[_KP_HIP_L][0] + kxy[_KP_HIP_R][0]) / 2
            ay = float(kxy[_KP_HIP_L][1] + kxy[_KP_HIP_R][1]) / 2
    inside = [z for z in zones_px if z["seat"][0] <= ax <= z["seat"][2] and z["seat"][1] <= ay <= z["seat"][3]]
    if not inside:
        return None
    # Neighbouring seats drawn with a little overlap: the seat whose centre
    # is nearest the hips is this person's.
    return min(inside, key=lambda z: np.hypot((z["seat"][0] + z["seat"][2]) / 2 - ax, (z["seat"][1] + z["seat"][3]) / 2 - ay))


def _hips_on_chair(
    hips: tuple[float, float] | None,
    bbox: tuple[int, int, int, int],
    chairs: list[tuple[int, int, int, int]],
) -> bool:
    """True when a detected chair sits under this person's hips. Uses the hip
    keypoint when there is one, else the lower-middle of the person box --
    and the chair box shrunk a little, so a standing person merely next to
    or in front of an empty chair does not count."""
    if not chairs:
        return False
    if hips is None:
        x1, y1, x2, y2 = bbox
        hips = ((x1 + x2) / 2, y1 + 0.65 * (y2 - y1))
    hx, hy = hips
    for cx1, cy1, cx2, cy2 in chairs:
        mx, my = 0.15 * (cx2 - cx1), 0.15 * (cy2 - cy1)
        if cx1 + mx <= hx <= cx2 - mx and cy1 + my <= hy <= cy2 - my:
            return True
    return False


def _classify_posture(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    bbox: tuple[int, int, int, int],
    knee_torso_ratio: float = _SITTING_KNEE_TORSO_RATIO,
    standing_aspect: float = _STANDING_BBOX_ASPECT,
    kp_min_conf: float = _KP_MIN_CONF,
    chairs: list[tuple[int, int, int, int]] | None = None,
    learned: tuple[str, float] | None = None,
    in_seat: bool = False,
) -> str:
    """Sitting/standing from several weak cues, voted.

    This camera is high and looks down. For someone seen from the SIDE, the
    knee/torso ratio works: a seated person's thigh runs sideways in the
    image, so the knee is barely below the hip. For someone FACING the
    camera it does not: their thighs point toward the lens, which from above
    projects straight down the image -- hip, knee and ankle line up exactly
    as they do for a standing person. So a "standing" reading from the
    ratio is only half a vote for a frontal person, and two other cues weigh
    in:

      box shape     tall/narrow box -> standing, short/wide -> sitting
      chair         a chair under the hips -> sitting
      learned       a classifier trained on this site's own footage
                    (scripts/train_posture_model.py), when configured:
                    2 x its probability, so it outweighs the rules
                    whenever it is confident

    Standing needs a positive total; anything else is sitting.
    """
    chairs = chairs or []
    needed = (_KP_SHOULDER_L, _KP_SHOULDER_R, _KP_HIP_L, _KP_HIP_R, _KP_KNEE_L, _KP_KNEE_R)
    knee_min = max(kp_min_conf, _KNEE_MIN_CONF)
    have_kp = keypoints_conf.shape[0] > max(needed) and all(
        keypoints_conf[i] >= (knee_min if i in (_KP_KNEE_L, _KP_KNEE_R) else kp_min_conf) for i in needed
    )

    hips = None
    if keypoints_conf.shape[0] > _KP_HIP_R and min(keypoints_conf[_KP_HIP_L], keypoints_conf[_KP_HIP_R]) >= kp_min_conf:
        hips = (
            float(keypoints_xy[_KP_HIP_L][0] + keypoints_xy[_KP_HIP_R][0]) / 2,
            float(keypoints_xy[_KP_HIP_L][1] + keypoints_xy[_KP_HIP_R][1]) / 2,
        )
    on_chair = _hips_on_chair(hips, bbox, chairs)

    learned_vote = 0.0
    if learned is not None and learned[0] in ("sitting", "standing"):
        learned_vote = (2.0 if learned[0] == "standing" else -2.0) * float(learned[1])

    # A seat zone an admin drew around this workstation's chair (Admin ->
    # Workstations): being in it is strong evidence of sitting -- it settles
    # the views 2D geometry cannot (facing the camera, legs under the desk).
    # Clear standing (legs visible from the side + tall box) still wins.
    seat_vote = -_SEAT_ZONE_VOTE if in_seat else 0.0

    def rules_only_fallback() -> str:
        if in_seat:
            return "sitting" if learned_vote < _SEAT_ZONE_VOTE else "standing"
        if on_chair:
            base = -1.0
        else:
            by_box = _posture_from_bbox(bbox, standing_aspect)
            if by_box == "unknown" and not learned_vote:
                return "unknown"
            base = 1.0 if by_box == "standing" else -1.0
        return "standing" if base + learned_vote > 0 else "sitting"

    if not have_kp:
        return rules_only_fallback()

    shoulder_y = (keypoints_xy[_KP_SHOULDER_L][1] + keypoints_xy[_KP_SHOULDER_R][1]) / 2
    hip_y = (keypoints_xy[_KP_HIP_L][1] + keypoints_xy[_KP_HIP_R][1]) / 2
    knee_y = (keypoints_xy[_KP_KNEE_L][1] + keypoints_xy[_KP_KNEE_R][1]) / 2
    torso_len = abs(hip_y - shoulder_y)
    if torso_len < 1:
        return rules_only_fallback()
    ratio = abs(knee_y - hip_y) / torso_len
    shoulder_w = abs(keypoints_xy[_KP_SHOULDER_L][0] - keypoints_xy[_KP_SHOULDER_R][0])
    frontal = shoulder_w / torso_len >= _FRONTAL_SHOULDER_TORSO

    score = 0.0
    if ratio < knee_torso_ratio:
        score -= 1.0
    else:
        score += 0.5 if frontal else 1.0

    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1:
        aspect = (y2 - y1) / (x2 - x1)
        if aspect >= standing_aspect:
            score += 1.0
        elif aspect < standing_aspect * 0.8:
            score -= 1.0

    if on_chair:
        score -= 1.0
    score += learned_vote + seat_vote

    return "standing" if score > 0 else "sitting"


def posture_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.10) -> np.ndarray | None:
    """The person crop the posture classifier sees -- the same in training
    (scripts/collect_training_data.py) and live, so they never drift apart."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 24 or y2 - y1 < 48:
        return None
    return frame[y1:y2, x1:x2]


class _StandUpGuard:
    """Posture can only go from sitting to standing by actually standing up.

    Per-frame posture is a guess from 2D geometry, and some seated views
    are hard to tell from standing (legs under a desk with guessed knees,
    a person facing the camera). Standing up, though, always lifts the
    shoulders. So while a track sits, remember where its shoulders are; a
    later "standing" is accepted only if they rose clearly above that, or
    the person moved away from the seat (walked off). Otherwise it stays
    sitting. Standing -> sitting is never blocked.

    Seat memory covers the one case that leaves: a track that is BORN
    seated (tracker id changed after an occlusion, or the person sat down
    off-screen) has no seated history. Every place someone sat steadily is
    remembered as a seated head position; a new track whose head is at a
    remembered seat starts as sitting, and from then on must stand up like
    anyone else. A person standing at that seat has their head clearly
    higher, so they do not match it.

    Anchors are the shoulder midpoint scaled by shoulder width when both
    shoulders are visible, else the person box's top-centre scaled by half
    its width; a check only compares an anchor with the same kind of
    reference. Seats use the box anchor, which is always available.
    """

    def __init__(self, min_rise: float = _STAND_UP_MIN_RISE, min_move: float = _STAND_UP_MIN_MOVE,
                 alpha: float = 0.2, seat_learn_passes: int = _SEAT_LEARN_PASSES,
                 seat_forget_seconds: float = _SEAT_FORGET_SECONDS) -> None:
        self._min_rise = min_rise
        self._min_move = min_move
        self._alpha = alpha
        self._ref: dict[int, dict[str, tuple[float, float, float]]] = {}
        self._seated_passes: dict[int, int] = {}
        self._seat_learn = seat_learn_passes
        self._seat_forget = seat_forget_seconds
        # (x, y, scale, last_used) of a seated head, image coordinates.
        self._seats: list[list[float]] = []

    @staticmethod
    def anchors(bbox, pose, kp_min_conf: float = _KP_MIN_CONF) -> dict[str, tuple[float, float, float]]:
        x1, y1, x2, y2 = bbox
        out = {"box": ((x1 + x2) / 2, float(y1), max(10.0, (x2 - x1) / 2))}
        if pose is not None:
            kxy, kconf = pose
            if kconf.shape[0] > _KP_SHOULDER_R and min(kconf[_KP_SHOULDER_L], kconf[_KP_SHOULDER_R]) >= kp_min_conf:
                (lx, ly), (rx, ry) = kxy[_KP_SHOULDER_L], kxy[_KP_SHOULDER_R]
                width = abs(float(lx) - float(rx))
                if width >= 10:
                    out["kp"] = ((lx + rx) / 2, (ly + ry) / 2, width)
        return out

    def _seat_at(self, anchor: tuple[float, float, float]) -> Optional[list[float]]:
        x, y, sc = anchor
        for seat in self._seats:
            sx, sy, ss, _ = seat
            scale = max(ss, sc)
            if abs(x - sx) <= 0.6 * scale and abs(y - sy) <= self._min_rise * scale:
                return seat
        return None

    def _learn_seat(self, anchor: tuple[float, float, float], now: float) -> None:
        seat = self._seat_at(anchor)
        if seat is not None:
            a = self._alpha
            seat[0] += a * (anchor[0] - seat[0])
            seat[1] += a * (anchor[1] - seat[1])
            seat[2] += a * (anchor[2] - seat[2])
            seat[3] = now
            return
        self._seats.append([*anchor, now])
        if len(self._seats) > _MAX_SEATS:
            self._seats.sort(key=lambda s_: s_[3])
            del self._seats[0]

    def apply(self, track_id: int, raw: str, anchors: dict[str, tuple[float, float, float]],
              now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        self._seats = [s_ for s_ in self._seats if now - s_[3] <= self._seat_forget]
        ref = self._ref.get(track_id)

        if raw == "standing" and not ref and "box" in anchors:
            seat = self._seat_at(anchors["box"])
            if seat is not None:
                # Born seated at a known seat: sitting, and guarded from here on.
                seat[3] = now
                raw = "sitting"

        if raw == "sitting":
            ref = self._ref.setdefault(track_id, {})
            for kind, (x, y, sc) in anchors.items():
                if kind in ref:
                    a = self._alpha
                    rx, ry, rs = ref[kind]
                    ref[kind] = (rx + a * (x - rx), ry + a * (y - ry), rs + a * (sc - rs))
                else:
                    ref[kind] = (float(x), float(y), float(sc))
            passes = self._seated_passes.get(track_id, 0) + 1
            self._seated_passes[track_id] = passes
            if passes >= self._seat_learn and "box" in ref:
                self._learn_seat(ref["box"], now)
            return raw

        self._seated_passes.pop(track_id, None)
        if raw != "standing" or not ref:
            return raw
        kind = "kp" if "kp" in anchors and "kp" in ref else "box"
        if kind not in anchors or kind not in ref:
            return "sitting"
        x, y, _ = anchors[kind]
        rx, ry, rs = ref[kind]
        if (ry - y) / rs >= self._min_rise or abs(x - rx) / rs >= self._min_move:
            self._ref.pop(track_id, None)
            return "standing"
        return "sitting"

    def keep_only(self, track_ids) -> None:
        self._ref = {tid: r for tid, r in self._ref.items() if tid in track_ids}
        self._seated_passes = {tid: n for tid, n in self._seated_passes.items() if tid in track_ids}


# The hand -- where a phone actually sits -- is past the wrist joint, along
# the forearm: wrist + this share of the elbow->wrist vector (forearm ~25cm,
# palm centre ~8cm beyond the wrist). Measured on the Inside camera: a phone
# held forward was 0.19 body heights from the wrist joint but in the fingers.
_HAND_BEYOND_WRIST = 0.35
_KP_ELBOW_OF = {_KP_WRIST_L: _KP_ELBOW_L, _KP_WRIST_R: _KP_ELBOW_R}


def _hand_points(kxy: np.ndarray, kconf: np.ndarray, kp_min_conf: float = _KP_MIN_CONF) -> list[list[tuple[float, float]]]:
    """For each visible wrist: [wrist, hand centre] (hand centre = wrist
    extended along the forearm when the elbow is visible, else just the
    wrist). A phone near EITHER point is in that hand."""
    out = []
    for wi in (_KP_WRIST_L, _KP_WRIST_R):
        if wi >= len(kconf) or kconf[wi] < kp_min_conf:
            continue
        wx, wy = float(kxy[wi][0]), float(kxy[wi][1])
        pts = [(wx, wy)]
        ei = _KP_ELBOW_OF[wi]
        if ei < len(kconf) and kconf[ei] >= kp_min_conf:
            ex, ey = float(kxy[ei][0]), float(kxy[ei][1])
            pts.append((wx + _HAND_BEYOND_WRIST * (wx - ex), wy + _HAND_BEYOND_WRIST * (wy - ey)))
        out.append(pts)
    return out


def _skin_share(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 2) -> float:
    """Share of skin-coloured pixels inside a box (YCrCb skin range), with a
    small inset so the hand holding a phone at its edge does not count."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 + pad), max(0, y1 + pad)
    x2, y2 = min(w, x2 - pad), min(h, y2 - pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    ycc = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2YCrCb)
    return float(cv2.inRange(ycc, (0, 135, 85), (255, 180, 135)).mean() / 255.0)


def _hand_zoom_windows(
    person_boxes: list, pose_by_track: dict, img_w: int, img_h: int, kp_min_conf: float = _KP_MIN_CONF,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    """(window, raised) for each person's visible wrists, for the hand zoom
    check. raised = the hand is in the upper part of the body (face, chest:
    calls and reading). Two wrists of one person close together (both hands
    on a phone) share one window centred between them."""
    out = []
    for tid, (x1, y1, x2, y2), _s in person_boxes:
        pose = pose_by_track.get(tid)
        if pose is None:
            continue
        kxy, kconf = pose
        side = max(_HAND_ZOOM_MIN_PX, _HAND_ZOOM_WINDOW * (y2 - y1))
        wrists = [pts[-1] for pts in _hand_points(kxy, kconf, kp_min_conf)]   # hand centres
        if len(wrists) == 2 and np.hypot(wrists[0][0] - wrists[1][0], wrists[0][1] - wrists[1][1]) < side * 0.5:
            wrists = [((wrists[0][0] + wrists[1][0]) / 2, (wrists[0][1] + wrists[1][1]) / 2)]
        for cx, cy in wrists:
            wx1, wy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            wx2, wy2 = int(min(img_w, cx + side / 2)), int(min(img_h, cy + side / 2))
            if wx2 - wx1 >= 16 and wy2 - wy1 >= 16:
                out.append(((wx1, wy1, wx2, wy2), cy < y1 + 0.45 * (y2 - y1)))
    return out[:_HAND_ZOOM_MAX_WINDOWS]


class _DeskPhones:
    """Tells a phone IN A HAND from a phone LYING ON A DESK by its own motion.

    A hand is never perfectly still: a held phone's box shifts by a few
    pixels from moment to moment. A phone on a desk does not move at all
    (its box stays within a pixel or two, detector noise). So:

      * a phone counts only once it has MOVED since it was first seen
        (centre shift > _DESK_PHONE_MOVE_EPS of its width) -- a held phone
        does within a second, a desk phone never does;
      * a phone that then stays pixel-still for _DESK_PHONE_STILL_SECONDS is
        put down (parked) and stops counting until it moves again.

    This does not depend on where the hands are, so a phone right next to a
    typing or mousing hand -- wrist practically on it -- is still a desk
    phone. The flip side: a phone resting on the desk while someone scrolls
    it with one finger is "on the desk", not "in hand".

    Only an observation of about the same SIZE can move a phone: a partial
    box (a sliver of the phone, or part hidden under a typing hand) is the
    same phone where it was, not the phone somewhere new. Measured on
    Camera 01: such slivers made a desk phone "move" every few passes, so it
    was never parked and was counted as phone use for minutes.
    """

    def __init__(self, still_seconds: float = _DESK_PHONE_STILL_SECONDS,
                 move_eps: float = _DESK_PHONE_MOVE_EPS, forget_seconds: float = _DESK_PHONE_FORGET_SECONDS) -> None:
        self._still = still_seconds
        self._eps = move_eps
        self._forget = forget_seconds
        # [anchor_x, anchor_y, width, anchored_at, has_moved, last_seen]
        self._phones: list[list] = []

    def filter(self, phones: list, wrists=None, now: float = 0.0) -> list:
        """phones: [(bbox, conf)]. Returns the phones that look hand-held.
        (wrists is accepted for call-site compatibility and not used.)"""
        self._phones = [p for p in self._phones if now - p[5] <= self._forget]
        out = []
        for bbox, conf in phones:
            x1, y1, x2, y2 = bbox
            cx, cy, pw = (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, float(x2 - x1))
            rec = min(
                (p for p in self._phones if np.hypot(cx - p[0], cy - p[1]) <= _DESK_PHONE_MATCH * max(pw, p[2])),
                key=lambda p: np.hypot(cx - p[0], cy - p[1]), default=None,
            )
            if rec is None:
                rec = [cx, cy, pw, now, False, now]
                self._phones.append(rec)
            else:
                rec[5] = now
                same_size = 0.6 <= pw / max(rec[2], 1.0) <= 1.67
                if same_size and np.hypot(cx - rec[0], cy - rec[1]) > self._eps * max(pw, rec[2]):
                    rec[0], rec[1], rec[2], rec[3], rec[4] = cx, cy, pw, now, True
            if rec[4] and now - rec[3] < self._still:
                out.append((bbox, conf))
        return out


class _PhoneUseConfidence:
    """Per person, how sure we are they are using a phone (0..1).

    Each pass gives evidence:
      phone detected in their hand   0.4 .. 1.0  (detector confidence and
                                                  how firmly it is in the hand)
      phone briefly missed, hand
      still where it was (held)      0.5
      no phone                       0.0
    and the confidence moves part of the way toward it (exponential average),
    so it climbs over a few passes and one odd frame does not swing it.
    """

    def __init__(self, alpha: float = 0.35) -> None:
        self._alpha = alpha
        self._level: dict[int, float] = {}

    @staticmethod
    def evidence(detected: tuple | None, held: bool) -> float:
        if detected is not None:
            _bbox, conf, owner_score = detected
            detector = min(1.0, conf / 0.5)
            in_hand = min(1.0, max(0.0, (owner_score - _PHONE_OWNER_MIN_SCORE) / (1.0 - _PHONE_OWNER_MIN_SCORE)))
            return 0.4 + 0.6 * (0.5 * detector + 0.5 * in_hand)
        return 0.5 if held else 0.0

    def update(self, track_id: int, evidence: float) -> float:
        level = self._level.get(track_id, 0.0)
        level += self._alpha * (evidence - level)
        self._level[track_id] = level
        return level

    def keep_only(self, track_ids) -> None:
        self._level = {t: v for t, v in self._level.items() if t in track_ids}


class _PhoneHold:
    """Keeps a phone "in hand" through short detector misses.

    A phone being used is missed for a pass or two all the time -- tilted
    edge-on, covered by the thumb, motion blur -- which made phone sessions
    blink off and on. While a wrist of the same person stays where the
    phone last was, the phone is assumed still there, for at most
    hold_seconds after the last real sighting. Putting the phone down
    moves the hand away (or the hold simply runs out), so a held phone
    never outlives the evidence by more than a few seconds.
    """

    def __init__(self, hold_seconds: float = _PHONE_HOLD_SECONDS, max_dist: float = _PHONE_HOLD_MAX_DIST) -> None:
        self._hold = hold_seconds
        self._max_dist = max_dist
        # track_id -> (phone bbox, phone conf, time of last real sighting)
        self._last: dict[int, tuple[tuple[int, int, int, int], float, float]] = {}

    def update(self, track_id: int, owned, person_bbox, pose, now: float,
               kp_min_conf: float = _KP_MIN_CONF):
        """Return (phone_bbox, conf) for this pass: the real detection if
        there is one, a held one if the hand is still there, else None."""
        if owned is not None:
            self._last[track_id] = (owned[0], owned[1], now)
            return owned[0], owned[1]
        last = self._last.get(track_id)
        if last is None or pose is None:
            return None
        bbox, conf, seen_at = last
        if now - seen_at > self._hold:
            self._last.pop(track_id, None)
            return None
        ph = max(1, person_bbox[3] - person_bbox[1])
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        kxy, kconf = pose
        for pts in _hand_points(kxy, kconf, kp_min_conf):
            if any(np.hypot(cx - hx, cy - hy) / ph <= self._max_dist for hx, hy in pts):
                return bbox, conf
        return None

    def keep_only(self, track_ids) -> None:
        self._last = {tid: v for tid, v in self._last.items() if tid in track_ids}


def _phone_person_score(
    phone_bbox: tuple[int, int, int, int],
    person_bbox: tuple[int, int, int, int],
    pose: tuple[np.ndarray, np.ndarray] | None,
    img_w: int,
    img_h: int,
) -> float:
    """
    Calculate how strongly a phone belongs to this person.

    IMPORTANT:
    This function does NOT decide phone ownership.
    The caller must compare this score against ALL people
    and assign the phone to only the highest-scoring person.
    """

    px1, py1, px2, py2 = person_bbox
    pw = px2 - px1
    ph = py2 - py1

    if pw <= 0 or ph <= 0:
        return 0.0

    ox1, oy1, ox2, oy2 = phone_bbox
    if max(ox2 - ox1, oy2 - oy1) > _PHONE_MAX_BODY_SHARE * ph:
        return 0.0   # far too big to be a phone in this person's hand
    phone_cx = (ox1 + ox2) / 2.0
    phone_cy = (oy1 + oy2) / 2.0

    best_wrist_score = 0.0

    # ---------------------------------------------------------
    # 1. WRIST PROXIMITY
    # ---------------------------------------------------------

    if pose is not None:
        kxy, kconf = pose

        # Distance from the hand (wrist joint, or the palm centre beyond it
        # along the forearm -- _hand_points) to the phone's EDGE, 0 when
        # over the phone, normalised by body height.
        #   ON the phone (<= half of _PHONE_IN_HAND_DIST): in the hand --
        #     texting, a call, a phone held in the lap.
        #   NEAR it (up to _PHONE_IN_HAND_DIST): only if the phone is RAISED
        #     to chest/face height (held out in front). Near a hand at desk
        #     level it is lying on the table beside a typing hand. Measured:
        #     phone on a table beside the keyboard 0.078 from the hand and
        #     0.22 body heights below the shoulders; texting 0.000; a call
        #     0.023 and above the shoulders.
        sh = [kxy[i] for i in (_KP_SHOULDER_L, _KP_SHOULDER_R) if i < len(kconf) and kconf[i] >= _KP_MIN_CONF]
        raised = bool(sh) and (phone_cy - float(np.mean([p_[1] for p_ in sh]))) / max(ph, 1) <= _PHONE_RAISED_BELOW_SHOULDER
        for pts in _hand_points(kxy, kconf, _KP_MIN_CONF):
            for wx, wy in pts:
                dx = max(ox1 - wx, 0.0, wx - ox2)
                dy = max(oy1 - wy, 0.0, wy - oy2)
                normalized_distance = float(np.hypot(dx, dy)) / max(ph, 1)

                if normalized_distance <= _PHONE_IN_HAND_DIST * 0.5:
                    score = 1.0
                elif normalized_distance <= _PHONE_IN_HAND_DIST and raised:
                    score = 0.75
                else:
                    score = 0.0

                best_wrist_score = max(best_wrist_score, score)

    # ---------------------------------------------------------
    # 2. PERSON-BODY CONTAINMENT
    # ---------------------------------------------------------

    expanded = _expand_box(
        person_bbox,
        mx=0.05,
        my=0.05,
        w=img_w,
        h=img_h,
    )

    containment = _containment_ratio(
        phone_bbox,
        expanded
    )

    # Give body containment only a SMALL contribution.
    # It must never be sufficient by itself.
    body_score = min(containment, 1.0) * 0.20

    # ---------------------------------------------------------
    # 3. VERTICAL HAND/TORSO REGION
    # ---------------------------------------------------------

    # Phone should generally be around torso/hand region,
    # not above the head or far outside the body.

    relative_y = (phone_cy - py1) / max(ph, 1)

    if 0.30 <= relative_y <= 1.10:
        zone_score = 0.15
    else:
        zone_score = 0.0

    # ---------------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------------

    score = (
        best_wrist_score * 0.65 +
        body_score +
        zone_score
    )

    return float(min(score, 1.0))

def _is_computer_in_use(
    comp_bbox: tuple[int, int, int, int],
    cls: int,
    person_bbox: tuple[int, int, int, int],
    pose: tuple[np.ndarray, np.ndarray] | None,
    phone_detected: bool,
    img_w: int,
    img_h: int,
) -> bool:
    """Check if a computer peripheral (laptop/keyboard/mouse) is actively in use."""
    px1, py1, px2, py2 = person_bbox
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return False

    ox1, oy1, ox2, oy2 = comp_bbox
    ox_c = (ox1 + ox2) / 2
    oy_c = (oy1 + oy2) / 2

    # Check 1: Keypoint proximity (hands/wrists on keyboard/mouse/laptop)
    if pose is not None:
        kxy, kconf = pose
        wrists_valid = [
            idx for idx in (_KP_WRIST_L, _KP_WRIST_R)
            if idx < len(kconf) and kconf[idx] >= _KP_MIN_CONF
        ]
        if wrists_valid:
            for idx in wrists_valid:
                wx, wy = kxy[idx]
                if (ox1 - 30 <= wx <= ox2 + 30) and (oy1 - 30 <= wy <= oy2 + 30):
                    return True

    # Check 2: If phone is actively in use, ignore idle side-table keyboards
    if phone_detected:
        comp_expanded = _expand_box(person_bbox, mx=0.05, my=0.10, w=img_w, h=img_h)
        ratio = _containment_ratio(comp_bbox, comp_expanded)
        return (cls == _CLASS_LAPTOP and ratio >= 0.50) or ratio >= 0.65

    # Check 3: Active working workspace zone
    # Tighter horizontal margin (0.08) prevents idle side-table keyboards from false-triggering
    comp_expanded = _expand_box(person_bbox, mx=0.08, my=0.15, w=img_w, h=img_h)
    ratio = _containment_ratio(comp_bbox, comp_expanded)
    if ratio >= _PROXIMITY_COMPUTER_CONTAINMENT:
        if (px1 - 0.10 * pw <= ox_c <= px2 + 0.10 * pw) and (py1 + 0.20 * ph <= oy_c <= py2 + 0.20 * ph):
            return True

    return False


def _head_yaw_proxy(pose: tuple[np.ndarray, np.ndarray] | None, kp_min_conf: float = _KP_MIN_CONF):
    """Rough horizontal head-turn estimate from 2D keypoints, or None.

    Returns (offset_ratio, shoulder_mid_x, shoulder_mid_y, shoulder_width).
    offset_ratio is the nose's horizontal displacement from the shoulder
    midpoint divided by shoulder width: ~0 looking straight along the body
    axis, positive looking to image-right, negative to image-left.

    This is a proxy, not head-pose estimation -- it needs no extra model and
    degrades to None when the shoulders or nose aren't visible, which is the
    honest outcome on a top-down camera rather than a fabricated angle.
    """
    if pose is None:
        return None
    kxy, kconf = pose
    needed = (_KP_NOSE, _KP_SHOULDER_L, _KP_SHOULDER_R)
    if len(kconf) <= max(needed) or any(kconf[i] < kp_min_conf for i in needed):
        return None
    sl, sr = kxy[_KP_SHOULDER_L], kxy[_KP_SHOULDER_R]
    shoulder_w = float(abs(sr[0] - sl[0]))
    if shoulder_w < 1:
        return None
    mid_x = float((sl[0] + sr[0]) / 2)
    mid_y = float((sl[1] + sr[1]) / 2)
    offset = (float(kxy[_KP_NOSE][0]) - mid_x) / shoulder_w
    return offset, mid_x, mid_y, shoulder_w


_YAW_THRESHOLD = 0.35
_PITCH_DOWN_THRESHOLD = 0.30


def _classify_head_orientation(
    pose: tuple[np.ndarray, np.ndarray] | None,
    kp_min_conf: float = _KP_MIN_CONF,
) -> tuple[str, float, float]:
    """Discrete head orientation from 2D keypoints — INDEPENDENT of any
    monitor or interaction state. Never returns TOWARD_MONITOR; that
    inference belongs to the interaction layer, not here.

    Returns (label, yaw, pitch) where yaw/pitch are signed floats
    (positive = image-right / looking down) and label is one of
    FORWARD, LEFT, RIGHT, DOWN, UP, UNKNOWN.
    """
    if pose is None:
        return "UNKNOWN", 0.0, 0.0
    kxy, kconf = pose

    yaw_info = _head_yaw_proxy(pose, kp_min_conf)
    if yaw_info is None:
        return "UNKNOWN", 0.0, 0.0

    yaw_ratio, mid_x, mid_y, shoulder_w = yaw_info

    # Reject obviously wrong keypoint geometry (ceiling camera perspective
    # distortion can push nose far from the shoulder midpoint).
    if abs(yaw_ratio) > 2.0:
        return "UNKNOWN", 0.0, 0.0

    pitch = 0.0
    nose_y = float(kxy[_KP_NOSE][1])
    eye_indices = [i for i in (_KP_EYE_L, _KP_EYE_R) if i < len(kconf) and kconf[i] >= kp_min_conf]
    if eye_indices:
        eye_y = float(np.mean([kxy[i][1] for i in eye_indices]))
        eye_nose_dy = nose_y - eye_y
        pitch = eye_nose_dy / max(shoulder_w, 1.0)
    else:
        nose_shoulder_dy = nose_y - mid_y
        pitch = nose_shoulder_dy / max(shoulder_w, 1.0)

    if abs(pitch) > 2.0:
        return "UNKNOWN", 0.0, 0.0

    if pitch > _PITCH_DOWN_THRESHOLD:
        label = "DOWN"
    elif pitch < -_PITCH_DOWN_THRESHOLD:
        label = "UP"
    elif yaw_ratio > _YAW_THRESHOLD:
        label = "RIGHT"
    elif yaw_ratio < -_YAW_THRESHOLD:
        label = "LEFT"
    else:
        label = "FORWARD"

    return label, yaw_ratio, pitch


def _is_monitor_facing(
    monitor_bbox: tuple[int, int, int, int],
    person_bbox: tuple[int, int, int, int],
    pose: tuple[np.ndarray, np.ndarray] | None,
    posture: str,
    requires_sitting: bool = True,
    max_distance_factor: float = _MONITOR_MAX_DISTANCE_FACTOR,
    facing_tolerance: float = _MONITOR_FACING_TOLERANCE,
    kp_min_conf: float = _KP_MIN_CONF,
) -> bool:
    """Whether this person plausibly faces/interacts with this monitor.

    Deliberately named facing/interaction, never "working": the evidence
    here is proximity plus a 2D head-turn proxy plus posture, which cannot
    distinguish attention from a blank stare, and the caller reports it with
    a measured confidence rather than as fact.
    """
    px1, py1, px2, py2 = person_bbox
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return False
    if requires_sitting and posture == "standing":
        return False

    diag = float(np.hypot(pw, ph))
    if diag <= 0:
        return False

    mx_c = (monitor_bbox[0] + monitor_bbox[2]) / 2
    my_c = (monitor_bbox[1] + monitor_bbox[3]) / 2

    yaw = _head_yaw_proxy(pose, kp_min_conf)
    if yaw is not None:
        offset, mid_x, mid_y, _ = yaw
        anchor_x, anchor_y = mid_x, mid_y
    else:
        # No usable pose: anchor on the upper third of the person box, which
        # is roughly head/shoulder height for both seated and standing.
        offset = None
        anchor_x, anchor_y = (px1 + px2) / 2, py1 + 0.25 * ph

    if float(np.hypot(mx_c - anchor_x, my_c - anchor_y)) > max_distance_factor * diag:
        return False

    if offset is None:
        # No usable keypoints for head orientation — we cannot determine
        # whether the person is facing the monitor. Do NOT fall back to
        # proximity alone, which grants false positives to everyone near
        # any screen.
        return False

    # Reject absurd yaw values from ceiling-camera perspective distortion.
    # A real head turn produces |offset| < ~1.5 shoulder-widths.
    if abs(offset) > 2.0:
        return False

    # Facing when the head is roughly along the body axis, or turned toward
    # the monitor's side rather than away from it.
    if abs(offset) <= facing_tolerance:
        return True
    return (mx_c - anchor_x) * offset > 0


class _StaticObjects:
    """Remembers objects that do not move -- screens and computer peripherals.

    Each detection is matched (IoU) to a remembered object of the same kind;
    its box is smoothed, and after _STATIC_CONFIRM_HITS sightings it is
    "confirmed" and returned on every pass until unseen for
    _STATIC_FORGET_SECONDS. Fixed regions (ACTIVITY_MONITOR_REGIONS) are
    always confirmed.
    """

    def __init__(self, fixed: list | None = None) -> None:
        # [bbox(list of 4 floats), cls, hits, last_seen]
        self._objs: list[list] = []
        self._fixed = [(tuple(r), _CLASS_TV) for r in (fixed or [])]

    def update(self, detections: list[tuple[tuple[int, int, int, int], int]], now: float) -> list:
        """detections: [(bbox, cls)]. Returns confirmed [(bbox, cls)]."""
        for bbox, cls in detections:
            best, best_iou = None, _STATIC_MATCH_IOU
            for o in self._objs:
                if o[1] != cls:
                    continue
                v = _iou(bbox, tuple(int(x) for x in o[0]))
                if v >= best_iou:
                    best, best_iou = o, v
            if best is None:
                self._objs.append([list(map(float, bbox)), cls, 1, now])
            else:
                best[0] = [0.8 * a + 0.2 * b for a, b in zip(best[0], bbox)]
                best[2] += 1
                best[3] = now
        self._objs = [o for o in self._objs if now - o[3] <= _STATIC_FORGET_SECONDS]
        confirmed = [(tuple(int(x) for x in o[0]), o[1]) for o in self._objs if o[2] >= _STATIC_CONFIRM_HITS]
        return self._fixed + confirmed


def _head_direction(pose, kp_min_conf: float = _KP_MIN_CONF):
    """(nose_x, nose_y, unit dx, unit dy): which way the face points in the
    image, from behind the face (ears, else eyes) to the nose. Seen from a
    high camera this is the person's heading. None when not readable."""
    if pose is None:
        return None
    kxy, kconf = pose
    if len(kconf) <= _KP_EAR_R or kconf[_KP_NOSE] < kp_min_conf:
        return None
    nx, ny = float(kxy[_KP_NOSE][0]), float(kxy[_KP_NOSE][1])
    back = [i for i in (_KP_EAR_L, _KP_EAR_R) if kconf[i] >= kp_min_conf]
    if not back:
        back = [i for i in (_KP_EYE_L, _KP_EYE_R) if kconf[i] >= kp_min_conf]
    if not back:
        return None
    bx = float(np.mean([kxy[i][0] for i in back]))
    by = float(np.mean([kxy[i][1] for i in back]))
    dx, dy = nx - bx, ny - by
    n = float(np.hypot(dx, dy))
    if n < 3.0:                       # nose on top of the ears: no direction
        return None
    return nx, ny, dx / n, dy / n


def _angle_to(head, target_xy) -> float:
    """Degrees between the head direction and the line head -> target."""
    nx, ny, dx, dy = head
    tx, ty = target_xy[0] - nx, target_xy[1] - ny
    n = float(np.hypot(tx, ty))
    if n < 1e-6:
        return 0.0
    cos = max(-1.0, min(1.0, (dx * tx + dy * ty) / n))
    return float(np.degrees(np.arccos(cos)))


# Looking at their own hands (a phone or paper held in both hands): wrists
# this close together, in body heights, with the head pointing at them
# within this angle -- typing hands are apart on keyboard and mouse.
_HANDS_TOGETHER = 0.18
_HANDS_GAZE_DEG = 35.0


def _looking_at_hands(head, pose, person_bbox, screen, kp_min_conf: float = _KP_MIN_CONF) -> bool:
    """True when the person is looking at something held in both hands
    rather than at the screen: hands together, head pointing at them, and
    the hands nearer to the face than the screen is. Catches a phone held
    in the lap that the phone detector misses."""
    if head is None or pose is None or screen is None:
        return False
    kxy, kconf = pose
    if min(kconf[_KP_WRIST_L], kconf[_KP_WRIST_R]) < kp_min_conf:
        return False
    ph = max(1, person_bbox[3] - person_bbox[1])
    (lx, ly), (rx, ry) = kxy[_KP_WRIST_L], kxy[_KP_WRIST_R]
    if np.hypot(lx - rx, ly - ry) / ph > _HANDS_TOGETHER:
        return False
    mid = ((lx + rx) / 2, (ly + ry) / 2)
    if _angle_to(head, mid) > _HANDS_GAZE_DEG:
        return False
    sc = ((screen[0] + screen[2]) / 2, (screen[1] + screen[3]) / 2)
    return np.hypot(mid[0] - head[0], mid[1] - head[1]) < np.hypot(sc[0] - head[0], sc[1] - head[1])


class _ScreenGaze:
    """Per person: which screen they are looking at, with hysteresis.

    Each pass, among the confirmed screens within reach, the one the head
    points at most directly is the candidate. Looking starts below
    _GAZE_ENTER_DEG and, for the SAME screen, continues until the angle
    passes _GAZE_EXIT_DEG. If the head direction cannot be read this pass,
    the last decision is kept for _GAZE_HOLD_SECONDS.
    """

    def __init__(self) -> None:
        # tid -> (screen bbox or None, looking, time of last readable head)
        self._state: dict[int, tuple] = {}

    def update(self, tid: int, head, screens: list, person_bbox, now: float, max_distance_factor: float):
        prev_screen, prev_looking, seen_at = self._state.get(tid, (None, False, now))
        if head is None:
            if prev_looking and now - seen_at <= _GAZE_HOLD_SECONDS:
                return prev_screen, True
            self._state[tid] = (prev_screen, False, seen_at)
            return prev_screen, False
        px1, py1, px2, py2 = person_bbox
        reach = max_distance_factor * float(np.hypot(px2 - px1, py2 - py1))
        best, best_angle = None, 180.0
        for sb, _cls in screens:
            centre = ((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2)
            if np.hypot(centre[0] - head[0], centre[1] - head[1]) > reach:
                continue
            a = _angle_to(head, centre)
            if prev_looking and prev_screen is not None and _iou(sb, prev_screen) > 0.5:
                a -= (_GAZE_EXIT_DEG - _GAZE_ENTER_DEG)      # sticky: favour the screen they were on
            if a < best_angle:
                best, best_angle = sb, a
        looking = best is not None and best_angle <= _GAZE_ENTER_DEG
        self._state[tid] = (best if looking else prev_screen, looking, now)
        return (best if looking else None), looking

    def keep_only(self, track_ids) -> None:
        self._state = {t: v for t, v in self._state.items() if t in track_ids}


class ActivityDetector:
    """Stock YOLOv8x (COCO, object+track) + YOLOv8x-pose.

    Not for identity -- track ids are anonymous and reset whenever this
    worker stops receiving frames for a while (e.g. switching to another
    mode and back), since BoT-SORT's internal state goes stale relative to
    wall-clock time during the gap. Identity is attached in the parent
    process by app/activity/identity_binder.py from the face pipeline's own
    recognition result.
    """

    PROVIDES_TRACK_IDS = True

    def __init__(
        self,
        conf_threshold: float = 0.35,
        pose_conf_threshold: float = 0.35,
        obj_model: str | Path = _OBJ_MODEL,
        pose_model: str | Path = _POSE_MODEL,
        tracker_cfg: str | Path = _TRACKER_CFG,
        img_size: int = _IMG_SIZE,
        pose_img_size: int = 640,
        device: str = "auto",
        phone_conf: float = _PHONE_CONF_THRESHOLD,
        computer_conf: float = _COMPUTER_CONF_THRESHOLD,
        monitor_conf: float = _MONITOR_CONF_THRESHOLD,
        keypoint_min_conf: float = _KP_MIN_CONF,
        sitting_knee_torso_ratio: float = _SITTING_KNEE_TORSO_RATIO,
        standing_bbox_aspect: float = _STANDING_BBOX_ASPECT,
        monitor_max_distance_factor: float = _MONITOR_MAX_DISTANCE_FACTOR,
        monitor_requires_sitting: bool = True,
        monitor_facing_tolerance: float = _MONITOR_FACING_TOLERANCE,
        monitor_regions: tuple = (),
        person_tracker_cfg: str | Path = _PERSON_TRACKER_CFG,
        activity_fps: float = 4.0,
        phone_crop_pass: bool = True,
        phone_crop_upscale: float = 2.0,
        phone_crop_imgsz: int = 640,
        use_fp16: bool = True,
        enabled_activities: tuple = (),
        phone_model: str = "",
        phone_model_conf: float = 0.35,
        phone_model_imgsz: int = 1280,
        posture_model: str = "",
        focus_phone_imgsz: int = 1280,
        phone_zoom_always: bool = False,
        phone_zoom_conf: float = 0.20,
        phone_detector: str = "rfdetr",
        rfdetr_model: str = "",
        rfdetr_conf: float = 0.25,
    ) -> None:
        self._rfdetr = None   # RF-DETR phone detector, loaded below when configured
        import torch
        from ultralytics import YOLO

        obj_path, pose_path = Path(obj_model), Path(pose_model)
        if not obj_path.exists():
            raise FileNotFoundError(
                f"Activity object model not found at {obj_path}. "
                "Download with: curl -L -o models/yolov8x.pt "
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
            )
        if not pose_path.exists():
            raise FileNotFoundError(
                f"Activity pose model not found at {pose_path}. "
                "Download with: curl -L -o models/yolov8x-pose.pt "
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-pose.pt"
            )

        self._conf = conf_threshold
        self._pose_conf = pose_conf_threshold
        self._phone_conf = phone_conf
        self._computer_conf = computer_conf
        self._monitor_conf = monitor_conf
        self._kp_min_conf = keypoint_min_conf
        self._knee_torso_ratio = sitting_knee_torso_ratio
        self._standing_aspect = standing_bbox_aspect
        self._monitor_max_distance_factor = monitor_max_distance_factor
        self._monitor_requires_sitting = monitor_requires_sitting
        self._monitor_facing_tolerance = monitor_facing_tolerance
        self._monitor_regions = [tuple(r) for r in (monitor_regions or ())]
        self._img_size = img_size
        self._pose_img_size = pose_img_size
        self._tracker_cfg = str(tracker_cfg)
        self._phone_crop_pass = phone_crop_pass
        self._phone_crop_upscale = max(1.0, phone_crop_upscale)
        self._phone_crop_imgsz = phone_crop_imgsz

        # Admin-selected activities (app/activity/settings.py). Empty = all.
        # Analysis for an unselected interaction is skipped entirely -- its
        # YOLO class is not even requested. Persons + pose always run:
        # posture, movement and identity binding all need them.
        enabled = set(enabled_activities or ())
        self._base_wants = (
            not enabled or "PHONE_INTERACTION" in enabled,
            not enabled or "MONITOR_INTERACTION" in enabled,
            not enabled or "COMPUTER_INTERACTION" in enabled,
        )
        self._focus: Optional[str] = None
        self._apply_wants()
        self._phone_crop_pass = self._phone_crop_pass and self._want_phone

        if device and device != "auto":
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        if self._device == "cpu":
            logger.warning(
                "Activity models running on CPU -- no CUDA/MPS device available. "
                "Expect well under 1 fps with YOLOv8x; set ACTIVITY_OBJ_MODEL/"
                "ACTIVITY_POSE_MODEL to a smaller checkpoint (e.g. yolo11m.pt) "
                "or lower ACTIVITY_IMG_SIZE."
            )

        self._track_smooth: dict[int, dict[str, list[bool]]] = {}
        self._base_img_size = self._img_size
        self._focus_phone_imgsz = focus_phone_imgsz
        self._phone_zoom_always = phone_zoom_always
        self._phone_zoom = phone_zoom_always
        self._phone_zoom_conf = phone_zoom_conf
        self._zoom_turn = 0
        self._stand_up_guard = _StandUpGuard()
        self._zones: list[dict] = []         # workstations (Admin -> Workstations), 0..1 coords
        self._screens = _StaticObjects(fixed=self._monitor_regions)
        self._peripherals = _StaticObjects()
        self._gaze = _ScreenGaze()
        self._phone_hold = _PhoneHold()
        self._phone_use = _PhoneUseConfidence()
        self._desk_phones = _DeskPhones()
        self._use_fp16 = use_fp16 and self._device == "cuda"

        logger.info("Loading Activity object+track model (%s) on device=%s ...", obj_path.name, self._device)
        self._obj_model = YOLO(str(obj_path))
        logger.info("Loading Activity pose model (%s) on device=%s ...", pose_path.name, self._device)
        self._pose_model = YOLO(str(pose_path))

        # Site-trained models (scripts/train_*_model.py), both optional.
        self._phone_model = None
        if phone_model and self._want_phone:
            if Path(phone_model).exists():
                logger.info("Loading site-trained phone model (%s) ...", Path(phone_model).name)
                self._phone_model = YOLO(phone_model)
                self._phone_model_conf = phone_model_conf
                self._phone_model_imgsz = phone_model_imgsz
            else:
                logger.warning("ACTIVITY_PHONE_MODEL %s not found -- using the stock phone class", phone_model)
        # RF-DETR on native-resolution upper-body crops (app/detection/rfdetr_phone.py):
        # on this site's test frames 7/9 phones vs 1-2/9 for YOLO on the full frame.
        if (phone_detector or "").lower() == "rfdetr" and self._phone_model is None:
            if rfdetr_model and Path(rfdetr_model).exists():
                from app.detection.rfdetr_phone import RFDETRPhoneDetector
                provider = "CUDAExecutionProvider" if self._device == "cuda" else "CPUExecutionProvider"
                self._rfdetr = RFDETRPhoneDetector(rfdetr_model, conf=rfdetr_conf, provider=provider)
                self._apply_wants()          # the YOLO phone class is no longer needed
            else:
                logger.warning("RF-DETR model %s not found -- phones from YOLO", rfdetr_model or "(unset)")
        self._posture_model = None
        if posture_model:
            if Path(posture_model).exists():
                logger.info("Loading site-trained posture model (%s) ...", Path(posture_model).name)
                self._posture_model = YOLO(posture_model)
            else:
                logger.warning("ACTIVITY_POSTURE_MODEL %s not found -- posture from rules only", posture_model)

        if self._use_fp16:
            self._obj_model.model.half()
            self._pose_model.model.half()
            logger.info("FP16 (half precision) enabled for both models -- ~1.8x speedup on RTX 3060")

        # Base track confidence: lowest of the per-class floors, so a class
        # with a low threshold still survives the tracker's own filter.
        self._track_conf = min(self._conf, self._phone_conf, self._computer_conf, self._monitor_conf)

        # track_buffer in the person config is expressed in FRAMES at
        # ACTIVITY_FPS (not at 30fps video rate) -- see that file.
        self._person_tracker = ByteTrackFaceTracker(tracker_cfg=person_tracker_cfg)

        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        self._obj_model.predict(
            blank, imgsz=self._img_size, classes=self._classes,
            conf=self._track_conf, device=self._device, verbose=False,
        )
        self._pose_model.predict(
            blank, imgsz=self._pose_img_size, conf=self._pose_conf,
            device=self._device, verbose=False,
        )
        logger.info(
            "Activity detector ready (device=%s fp16=%s imgsz=%d conf=%.2f pose_conf=%.2f "
            "phone=%.2f computer=%.2f monitor=%.2f, %d configured monitor region(s), "
            "hand-zone phone pass=%s)",
            self._device, self._use_fp16, self._img_size, conf_threshold, pose_conf_threshold,
            self._phone_conf, self._computer_conf, self._monitor_conf,
            len(self._monitor_regions), "on" if self._phone_crop_pass else "off",
        )

    def _hand_zone_phone_boxes(
        self,
        frame: np.ndarray,
        person_boxes: list[tuple[int, tuple[int, int, int, int], float]],
    ) -> list[tuple[int, int, int, int]]:
        """Second, zoomed pass for phones inside each person's upper body.

        A phone occupies ~50x27px in a 1920x1080 frame at this camera's
        distance, which the main pass shrinks to ~25x13px at imgsz=960 --
        right at the edge of what the detector can resolve. Re-running the
        detector on an upscaled crop of just the upper body roughly doubles
        phone recall (measured: 22 -> 45 hits over the same 11 frames) and
        raises peak confidence from 0.34 to 0.70.

        Crops are inferred as ONE batch, so this costs a single extra
        forward pass regardless of how many people are in frame.
        """
        if not person_boxes:
            return []
        h, w = frame.shape[:2]
        crops, origins = [], []
        for _tid, (x1, y1, x2, y2), _score in person_boxes:
            bh = y2 - y1
            # Upper ~70% of the body: hands held at or above waist height,
            # which is where a phone in use is. Widened slightly because an
            # outstretched arm leaves the body box.
            cx1 = max(0, int(x1 - 0.10 * (x2 - x1)))
            cx2 = min(w, int(x2 + 0.10 * (x2 - x1)))
            cy1, cy2 = max(0, y1), min(h, int(y1 + 0.70 * bh))
            if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]
            crops.append(cv2.resize(
                crop, (int(cw * self._phone_crop_upscale), int(ch * self._phone_crop_upscale)),
                interpolation=cv2.INTER_CUBIC,
            ))
            origins.append((cx1, cy1))
        if not crops:
            return []

        results = self._obj_model.predict(
            crops, imgsz=self._phone_crop_imgsz, conf=self._phone_conf,
            classes=[_CLASS_PHONE], device=self._device, verbose=False,
        )
        out: list[tuple[int, int, int, int]] = []
        for res, (ox, oy) in zip(results, origins):
            if res.boxes is None or not len(res.boxes):
                continue
            for b in res.boxes.xyxy.cpu().numpy():
                out.append((
                    max(0, int(ox + b[0] / self._phone_crop_upscale)),
                    max(0, int(oy + b[1] / self._phone_crop_upscale)),
                    min(w, int(ox + b[2] / self._phone_crop_upscale)),
                    min(h, int(oy + b[3] / self._phone_crop_upscale)),
                ))
        return out

    def _apply_wants(self) -> None:
        """What this pass analyses: the admin's selection, narrowed to the
        live-view focus (Phone / Monitor / Computer / Body) when there is one.
        A class nobody needs is not even requested from YOLO."""
        phone, monitor, computer = self._base_wants
        f = self._focus
        self._want_phone = phone and f in (None, "phone")
        self._want_monitor = monitor and f in (None, "monitor")
        self._want_computer = computer and f in (None, "computer")
        self._want_posture = f in (None, "body")
        self._classes = [_CLASS_PERSON]
        if self._want_posture:
            self._classes.append(_CLASS_CHAIR)   # chairs are only a posture cue
        if self._want_monitor:
            self._classes.append(_CLASS_TV)
        if self._want_computer:
            self._classes += [_CLASS_LAPTOP, _CLASS_MOUSE, _CLASS_KEYBOARD]
        if (self._want_phone or self._want_monitor) and self._rfdetr is None:
            # In the Monitor view phones are looked for only as a veto: a
            # phone in someone's hand means they are looking at the phone,
            # not the screen behind it (from above, both can lie in the same
            # direction from the head). Never recorded or drawn there.
            self._classes.append(_CLASS_PHONE)

    def set_focus(self, focus) -> None:
        """Control from the parent process: a focus string (legacy) or
        {"focus": str|None, "zones": [workstation dicts in 0..1 coords]}."""
        if isinstance(focus, dict):
            self._zones = list(focus.get("zones") or [])
            focus = focus.get("focus")
        self._set_focus(focus)

    def _set_focus(self, focus: Optional[str]) -> None:
        """Called by the worker when the live view focuses this camera
        (app/web/server.py's focus coordinator). In the Phone view the GPU
        freed by pausing the other cameras is spent on resolution: phones
        here are ~40x25px, and a 1280 pass resolves far more of them than
        960. Every other view keeps the configured size."""
        if focus != self._focus:
            self._focus = focus
            self._apply_wants()
            # Smoothing history from before the switch must not carry over
            # (it kept e.g. "facing a monitor" alive in the Phone view).
            self._track_smooth.clear()
            logger.info("Activity focus=%s -> analysing %s", focus or "everything", self._classes)
        # The resolution boost and hand zoom are the YOLO path's way to find small
        # phones; RF-DETR already looks at each person at native resolution.
        boost = focus == "phone" and self._want_phone and self._rfdetr is None
        size = self._focus_phone_imgsz if boost else self._base_img_size
        if size != self._img_size:
            logger.info("Activity focus=%s -> object pass at imgsz %d", focus or "none", size)
            self._img_size = size
        zoom = self._rfdetr is None and (self._phone_zoom_always or boost)
        if zoom != self._phone_zoom:
            logger.info("Hand zoom check for phones: %s", "on" if zoom else "off")
            self._phone_zoom = zoom

    def _pick_zoom_windows(self, windows: list, hints: list) -> list[tuple[int, int, int, int]]:
        """At most _HAND_ZOOM_PER_PASS windows: hands holding a faint phone
        hint first, then raised hands (calls, reading), then the rest taking
        turns pass by pass, so every hand is still looked at regularly."""
        def has_hint(win):
            x1, y1, x2, y2 = win
            return any(x1 <= (h[0] + h[2]) / 2 <= x2 and y1 <= (h[1] + h[3]) / 2 <= y2 for h in hints)

        def rotated(items):
            if not items:
                return items
            # Advance a full pass's worth, so N hands all get a turn within
            # ceil(N / _HAND_ZOOM_PER_PASS) passes.
            k = (self._zoom_turn * _HAND_ZOOM_PER_PASS) % len(items)
            return items[k:] + items[:k]

        hinted = [w_ for w_, _r in windows if has_hint(w_)]
        raised = rotated([w_ for w_, r in windows if r and w_ not in hinted])
        rest = rotated([w_ for w_, r in windows if not r and w_ not in hinted])
        self._zoom_turn += 1
        return (hinted + raised + rest)[:_HAND_ZOOM_PER_PASS]

    def _hand_zoom_phones(self, frame: np.ndarray, windows: list) -> list[tuple[tuple[int, int, int, int], float]]:
        """Re-check a zoomed square around each hand for phones -- one batched
        pass. Catches phones the full frame misses: at the ear, half hidden
        by fingers, tilted edge-on."""
        if not windows:
            return []
        crops, scales = [], []
        for x1, y1, x2, y2 in windows:
            crop = frame[y1:y2, x1:x2]
            scale = _HAND_ZOOM_UPSCALE_TO / max(crop.shape[:2])
            crops.append(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
            scales.append(scale)
        model = self._phone_model if self._phone_model is not None else self._obj_model
        classes = None if self._phone_model is not None else [_CLASS_PHONE]
        results = model.predict(crops, imgsz=_HAND_ZOOM_UPSCALE_TO, conf=self._phone_zoom_conf, classes=classes,
                                device=self._device, verbose=False)
        out = []
        for res, (x1, y1, _x2, _y2), scale in zip(results, windows, scales):
            if res.boxes is None:
                continue
            for b, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                out.append(((int(x1 + b[0] / scale), int(y1 + b[1] / scale),
                             int(x1 + b[2] / scale), int(y1 + b[3] / scale)), float(sc)))
        return out

    def _site_phone_boxes(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = frame.shape[:2]
        res = self._phone_model.predict(
            frame, imgsz=self._phone_model_imgsz, conf=self._phone_model_conf,
            device=self._device, verbose=False,
        )[0]
        out = []
        if res.boxes is not None:
            for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                out.append(((max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))), float(score)))
        return out

    def _learned_postures(
        self, frame: np.ndarray, person_boxes: list[tuple[int, tuple[int, int, int, int], float]],
    ) -> dict[int, tuple[str, float]]:
        """{track_id: (label, probability)} from the site-trained posture
        classifier -- one batched pass over every person crop."""
        if self._posture_model is None or not person_boxes:
            return {}
        crops, ids = [], []
        for tid, bbox, _score in person_boxes:
            crop = posture_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)
                ids.append(tid)
        if not crops:
            return {}
        results = self._posture_model.predict(crops, imgsz=224, device=self._device, verbose=False)
        names = self._posture_model.names
        out = {}
        for tid, res in zip(ids, results):
            if res.probs is None:
                continue
            top = int(res.probs.top1)
            out[tid] = (str(names[top]).lower(), float(res.probs.top1conf))
        return out

    def detect(self, frame: np.ndarray) -> list[TrackedFace]:
        h, w = frame.shape[:2]

        # predict(), NOT track(): Ultralytics' .track() returns only boxes
        # the tracker confirmed and silently drops the rest. Measured on
        # this camera that discarded 100% of phone detections (9 of 9 over
        # 40 frames) -- small, flickering, fast-moving objects never survive
        # track confirmation. Objects need no identity, only presence; they
        # are matched to a person geometrically every frame. Only PERSON
        # boxes are tracked, below.
        obj_results = self._obj_model.predict(
            frame, imgsz=self._img_size, classes=self._classes,
            conf=min(self._track_conf, _PHONE_HINT_CONF) if self._phone_zoom else self._track_conf,
            device=self._device, verbose=False,
        )
        boxes = obj_results[0].boxes
        if boxes is None or not len(boxes):
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        raw_persons: list[DetectedFace] = []
        phone_boxes: list[tuple[tuple[int, int, int, int], float]] = []  # (bbox, conf)
        phone_hints: list[tuple[int, int, int, int]] = []  # too faint to count; zoom there first
        computer_boxes: list[tuple[int, tuple[int, int, int, int]]] = []  # (cls, bbox)
        chair_boxes: list[tuple[int, int, int, int]] = []
        # This pass's raw monitor detections; fixed regions and remembered
        # screens are added by _StaticObjects below.
        monitor_boxes: list[tuple[tuple[int, int, int, int], float]] = []  # (bbox, conf)

        for (x1f, y1f, x2f, y2f), score, cls in zip(xyxy, confs, clss):
            x1, y1 = max(0, int(x1f)), max(0, int(y1f))
            x2, y2 = min(w, int(x2f)), min(h, int(y2f))
            bbox = (x1, y1, x2, y2)

            if cls == _CLASS_PERSON:
                if score >= self._conf:
                    raw_persons.append(DetectedFace(
                        bbox=bbox, confidence=float(score),
                        landmarks=np.empty((0, 2), dtype=np.float32),
                    ))
            elif cls == _CLASS_CHAIR:
                if score >= _CHAIR_CONF_THRESHOLD:
                    chair_boxes.append(bbox)
            elif cls == _CLASS_PHONE:
                if score >= self._phone_conf:
                    phone_boxes.append((bbox, float(score)))
                else:
                    phone_hints.append(bbox)
            elif cls == _CLASS_TV:
                if score >= self._monitor_conf:
                    monitor_boxes.append((bbox, float(score)))
            elif cls in _COMPUTER_CLASSES:
                if score >= self._computer_conf:
                    computer_boxes.append((cls, bbox))

        # Persons get stable ids from ByteTrack over the raw boxes.
        tracked_persons = self._person_tracker.update(raw_persons)
        person_boxes: list[tuple[int, tuple[int, int, int, int], float]] = [
            (t.track_id, t.face.bbox, t.face.confidence) for t in tracked_persons
        ]

        if not person_boxes:
            return []

        if self._phone_crop_pass:
            for crop_bbox in self._hand_zone_phone_boxes(frame, person_boxes):
                phone_boxes.append((crop_bbox, self._phone_conf))

        if self._phone_model is not None:
            # Trained on this camera's own footage: its phones replace the
            # stock COCO class, which misses phones seen from above and
            # fires on chargers and routers.
            phone_boxes = self._site_phone_boxes(frame)
        elif self._rfdetr is not None and (self._want_phone or self._want_monitor):
            # Per person: upper body + hands cropped at native resolution ->
            # RF-DETR. A find that is mostly skin is a hand, not a phone.
            phone_boxes = [
                (b, c) for b, c in self._rfdetr.detect_in_people(frame, [pb for _t, pb, _s in person_boxes])
                if _skin_share(frame, b) <= _RFDETR_MAX_SKIN
            ]
            phone_hints = []

        pose_results = self._pose_model.predict(
            frame, imgsz=self._pose_img_size, conf=self._pose_conf,
            device=self._device, verbose=False,
        )
        pose_boxes: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        pr = pose_results[0]
        if pr.boxes is not None and pr.keypoints is not None and len(pr.boxes) > 0:
            pxyxy = pr.boxes.xyxy.cpu().numpy()
            kxy = pr.keypoints.xy.cpu().numpy()
            kconf = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else None
            for i, (px1f, py1f, px2f, py2f) in enumerate(pxyxy):
                pbbox = (
                    max(0, int(px1f)), max(0, int(py1f)),
                    min(w, int(px2f)), min(h, int(py2f)),
                )
                conf_row = kconf[i] if kconf is not None else np.ones(kxy.shape[1], dtype=np.float32)
                pose_boxes.append((pbbox, kxy[i], conf_row))

        # Match each tracked person to its pose (needed by phone ownership
        # below, which compares every phone against every person).
        pose_by_track: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        for track_id, bbox, _score in person_boxes:
            best_iou, best_pose = _POSE_IOU_MATCH_THRESHOLD, None
            for pbbox, kxy, kconf in pose_boxes:
                score_iou = _iou(bbox, pbbox)
                if score_iou > best_iou:
                    best_iou, best_pose = score_iou, (kxy, kconf)
            pose_by_track[track_id] = best_pose

        learned_posture = self._learned_postures(frame, person_boxes) if self._want_posture else {}

        # Phone ownership: each phone goes to the ONE highest-scoring person
        # (see _phone_person_score), so a phone between two neighbours is
        # never counted for both. A person keeps their best-scoring phone.
        phone_owner: dict[int, tuple[tuple[int, int, int, int], float, float]] = {}  # tid -> (bbox, conf, score)
        wrists = []
        for _tid, (bx1, by1, bx2, by2), _s in person_boxes:
            pose = pose_by_track[_tid]
            if pose is None:
                continue
            for wi in (_KP_WRIST_L, _KP_WRIST_R):
                if wi < len(pose[1]) and pose[1][wi] >= self._kp_min_conf:
                    wrists.append((float(pose[0][wi][0]), float(pose[0][wi][1]), float(by2 - by1)))
        if self._phone_zoom and self._want_phone:
            for z_bbox, z_conf in self._hand_zoom_phones(
                frame, self._pick_zoom_windows(
                    _hand_zoom_windows(person_boxes, pose_by_track, w, h, self._kp_min_conf), phone_hints,
                ),
            ):
                # Skip what the full frame (or another window) already found,
                # and hands mistaken for phones (a fist at the chin).
                if _skin_share(frame, z_bbox) > _ZOOM_MAX_SKIN:
                    continue
                # A zoom find inside (or mostly inside) a phone already found
                # is a sliver of that same phone -- adding it made a desk
                # phone look like it was moving (see _DeskPhones).
                if all(_iou(z_bbox, b) < 0.3 and _containment_ratio(z_bbox, b) < 0.6 for b, _c in phone_boxes):
                    phone_boxes.append((z_bbox, z_conf))
        phone_boxes = self._desk_phones.filter(phone_boxes, wrists, time.monotonic())
        for p_bbox, p_conf in phone_boxes:
            owner_tid, owner_score = None, _PHONE_OWNER_MIN_SCORE
            for track_id, bbox, _score in person_boxes:
                s = _phone_person_score(p_bbox, bbox, pose_by_track[track_id], w, h)
                if s >= owner_score:
                    owner_tid, owner_score = track_id, s
            if owner_tid is not None and owner_score > phone_owner.get(owner_tid, (None, 0.0, 0.0))[2]:
                phone_owner[owner_tid] = (p_bbox, p_conf, owner_score)

        # Fixed objects: remembered across passes, so a missed detection does
        # not make a screen or keyboard disappear. Laptops are screens too.
        now_m = time.monotonic()
        screens = self._screens.update(
            [(b, _CLASS_TV) for b, _c in monitor_boxes]
            + [(b, _CLASS_LAPTOP) for c, b in computer_boxes if c == _CLASS_LAPTOP],
            now_m,
        ) if self._want_monitor else []
        if self._want_computer:
            seen = list(computer_boxes)
            computer_boxes = [(c, b) for b, c in self._peripherals.update([(b, c) for c, b in seen], now_m)]
            for c, b in seen:                 # this pass's own finds count at once
                if all(_iou(b, kb) < _STATIC_MATCH_IOU for _kc, kb in computer_boxes):
                    computer_boxes.append((c, b))

        tracked: list[TrackedFace] = []
        active_track_ids = set()
        zones_px = [
            {"name": z.get("name"),
             "seat": tuple(int(v * d) for v, d in zip(z["seat"], (w, h, w, h))),
             "screen": tuple(int(v * d) for v, d in zip(z["screen"], (w, h, w, h))) if z.get("screen") else None}
            for z in self._zones if z.get("seat")
        ]

        for track_id, bbox, score in person_boxes:
            active_track_ids.add(track_id)
            best_pose = pose_by_track[track_id]
            seat_zone = _seat_zone_for(bbox, best_pose, zones_px, self._kp_min_conf)

            posture = (
                _classify_posture(
                    best_pose[0], best_pose[1], bbox,
                    self._knee_torso_ratio, self._standing_aspect, self._kp_min_conf,
                    chair_boxes, learned_posture.get(track_id), seat_zone is not None,
                )
                if best_pose is not None
                else _classify_posture(
                    np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), bbox,
                    self._knee_torso_ratio, self._standing_aspect, self._kp_min_conf,
                    chair_boxes, learned_posture.get(track_id), seat_zone is not None,
                )
            )
            posture = self._stand_up_guard.apply(
                track_id, posture, _StandUpGuard.anchors(bbox, best_pose, self._kp_min_conf),
            )

            # --- Raw phone measurement ---
            owned = phone_owner.get(track_id)
            detected_phone = owned                     # a real detection this pass, before the hold
            held = self._phone_hold.update(track_id, owned, bbox, best_pose, time.monotonic(), self._kp_min_conf)
            if owned is None and held is not None:
                # A phone carried over a missed pass is re-checked like a fresh
                # find: a hand now covering that spot (chin resting on a fist)
                # is not a phone.
                if _skin_share(frame, held[0]) <= _RFDETR_MAX_SKIN:
                    owned = (held[0], held[1], 0.0)
            raw_phone = owned is not None
            best_phone_bbox, best_phone_conf = (owned[0], owned[1]) if owned else (None, 0.0)
            best_phone_dist = float("inf")
            if owned is not None and best_pose is not None:
                ox_c = (owned[0][0] + owned[0][2]) / 2
                oy_c = (owned[0][1] + owned[0][3]) / 2
                kxy, kconf = best_pose
                for wi in (_KP_WRIST_L, _KP_WRIST_R):
                    if wi < len(kconf) and kconf[wi] >= self._kp_min_conf:
                        d = float(np.hypot(ox_c - kxy[wi][0], oy_c - kxy[wi][1]))
                        best_phone_dist = min(best_phone_dist, d)
            if best_phone_dist == float("inf"):
                best_phone_dist = -1.0

            # Check computer use
            raw_computer = any(
                _is_computer_in_use(c_bbox, c_cls, bbox, best_pose, raw_phone, w, h)
                for c_cls, c_bbox in computer_boxes
            )

            # --- Looking at a screen (monitor or laptop) ---
            # Which way the face points (ears -> nose), and whether that is
            # at one of the remembered screens -- with hysteresis, so a small
            # head move does not flip it (_ScreenGaze).
            nearest_monitor_bbox, nearest_monitor_conf = None, 0.0
            nearest_monitor_dist = -1.0
            raw_monitor = False
            if self._want_monitor:
                head = _head_direction(best_pose, self._kp_min_conf)
                # At a drawn workstation, its own screen is the one that counts
                # (a neighbour's screen in the same direction does not).
                own = [(seat_zone["screen"], _CLASS_TV)] if seat_zone is not None and seat_zone["screen"] else None
                screen, looking = self._gaze.update(
                    track_id, head, own or screens, bbox, now_m, self._monitor_max_distance_factor,
                )
                if not looking and head is None and own and posture != "standing":
                    # Head not readable (seen from behind, ~30% of the time on
                    # these cameras): seated at a workstation means facing its
                    # screen -- chairs face their desks. The phone and
                    # hands-together checks below still overrule it.
                    screen, looking = own[0][0], True
                if looking and self._monitor_requires_sitting and posture == "standing":
                    looking = False
                if looking and owned is not None:
                    looking = False          # phone in hand: looking at the phone
                if looking and _looking_at_hands(head, best_pose, bbox, screen, self._kp_min_conf):
                    looking = False          # both hands held together, eyes on them
                raw_monitor = looking
                if screen is not None:
                    nearest_monitor_bbox, nearest_monitor_conf = screen, 1.0
                    if head is not None:
                        nearest_monitor_dist = float(np.hypot(
                            (screen[0] + screen[2]) / 2 - head[0], (screen[1] + screen[3]) / 2 - head[1],
                        ))

            # Multi-frame smoothing (anti-flicker)
            history = self._track_smooth.setdefault(
                track_id, {"phone": [], "computer": [], "monitor": []}
            )
            for key, raw in (("phone", raw_phone), ("computer", raw_computer), ("monitor", raw_monitor)):
                history[key].append(raw)
                if len(history[key]) > _SMOOTH_WINDOW:
                    history[key].pop(0)

            # Output states with short hysteresis
            phone_use_conf = self._phone_use.update(
                track_id, _PhoneUseConfidence.evidence(detected_phone, held=raw_phone and detected_phone is None),
            ) if self._want_phone else 0.0

            # Temporal confirmation: a phone (seen, or held through a miss by
            # _PhoneHold) in 3 of the last 5 passes. One stray detection
            # never shows as phone use; a real one is confirmed within ~1s.
            phone_in_use = sum(history["phone"]) >= _PHONE_MIN_VOTES
            computer_in_use = sum(history["computer"]) >= _COMPUTER_MIN_VOTES or (
                raw_computer and len(history["computer"]) < 3
            )
            monitor_facing = sum(history["monitor"]) >= _MONITOR_MIN_VOTES or (
                raw_monitor and len(history["monitor"]) < 3
            )

            # Anything not selected (live-view focus or the admin's list) is
            # off, whatever older passes voted.
            phone_in_use = phone_in_use and self._want_phone
            computer_in_use = computer_in_use and self._want_computer
            monitor_facing = monitor_facing and self._want_monitor

            # If phone is actively detected in hands, suppress false computer triggers from side peripherals
            if phone_in_use and not any(cls == _CLASS_LAPTOP for cls, _ in computer_boxes):
                computer_in_use = False

            head_label, head_yaw, head_pitch = _classify_head_orientation(
                best_pose, kp_min_conf=self._kp_min_conf,
            )

            tracked.append(
                TrackedFace(
                    track_id=track_id,
                    face=DetectedFace(
                        bbox=bbox,
                        confidence=score,
                        landmarks=np.empty((0, 2), dtype=np.float32),
                    ),
                    posture=posture,
                    phone_in_use=phone_in_use,
                    computer_in_use=computer_in_use,
                    monitor_facing=monitor_facing,
                    monitor_bbox=nearest_monitor_bbox,
                    monitor_distance=nearest_monitor_dist,
                    monitor_confidence=nearest_monitor_conf,
                    phone_hand_distance=best_phone_dist,
                    phone_bbox=best_phone_bbox,
                    phone_confidence=best_phone_conf,
                    phone_use_confidence=round(phone_use_conf, 3),
                    keypoints=best_pose[0] if best_pose is not None else None,
                    keypoints_conf=best_pose[1] if best_pose is not None else None,
                    head_orientation=head_label,
                    head_yaw=head_yaw,
                    head_pitch=head_pitch,
                )
            )

        # Cleanup stale smoothed tracks
        self._track_smooth = {tid: hist for tid, hist in self._track_smooth.items() if tid in active_track_ids}
        self._stand_up_guard.keep_only(active_track_ids)
        self._gaze.keep_only(active_track_ids)
        self._phone_hold.keep_only(active_track_ids)
        self._phone_use.keep_only(active_track_ids)

        return tracked

    def close(self) -> None:
        self._track_smooth.clear()
        self._stand_up_guard.keep_only(())
        self._phone_hold.keep_only(())
