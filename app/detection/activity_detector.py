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
from pathlib import Path

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

# Posture defaults (overridable via __init__ / ActivityConfig)
_SITTING_KNEE_TORSO_RATIO = 0.5
_STANDING_BBOX_ASPECT = 1.6

# Monitor-interaction defaults
_MONITOR_MAX_DISTANCE_FACTOR = 1.5
_MONITOR_FACING_TOLERANCE = 0.55

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
_PHONE_MIN_VOTES = 2
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


def _classify_posture(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    bbox: tuple[int, int, int, int],
    knee_torso_ratio: float = _SITTING_KNEE_TORSO_RATIO,
    standing_aspect: float = _STANDING_BBOX_ASPECT,
    kp_min_conf: float = _KP_MIN_CONF,
) -> str:
    """Heuristic sitting/standing classifier off hip/knee/shoulder keypoints.

    This camera is ceiling-mounted/top-down (see face_detector.py's module
    docstring), so hip/knee/ankle keypoints will often be occluded by desks
    and monitors -- not a rare edge case. When keypoint confidence is too low
    to trust, fall back to the person box's aspect ratio.
    """
    needed = (_KP_SHOULDER_L, _KP_SHOULDER_R, _KP_HIP_L, _KP_HIP_R, _KP_KNEE_L, _KP_KNEE_R)
    if keypoints_conf.shape[0] <= max(needed) or any(keypoints_conf[i] < kp_min_conf for i in needed):
        return _posture_from_bbox(bbox, standing_aspect)

    shoulder_y = (keypoints_xy[_KP_SHOULDER_L][1] + keypoints_xy[_KP_SHOULDER_R][1]) / 2
    hip_y = (keypoints_xy[_KP_HIP_L][1] + keypoints_xy[_KP_HIP_R][1]) / 2
    knee_y = (keypoints_xy[_KP_KNEE_L][1] + keypoints_xy[_KP_KNEE_R][1]) / 2
    torso_len = abs(hip_y - shoulder_y)
    lower_leg_len = abs(knee_y - hip_y)
    if torso_len < 1:
        return _posture_from_bbox(bbox, standing_aspect)
    ratio = lower_leg_len / torso_len
    return "sitting" if ratio < knee_torso_ratio else "standing"


def _is_phone_in_use(
    phone_bbox: tuple[int, int, int, int],
    person_bbox: tuple[int, int, int, int],
    pose: tuple[np.ndarray, np.ndarray] | None,
    img_w: int,
    img_h: int,
) -> bool:
    """Check if a phone detection belongs to and is being used by this person.

    A phone merely *present* in frame (face down on a desk, in a pocket) is
    not usage -- that distinction is the whole point of this function, and
    the caller never treats a bare phone detection as interaction.
    """
    px1, py1, px2, py2 = person_bbox
    pw, ph = px2 - px1, py2 - py1
    if pw <= 0 or ph <= 0:
        return False

    ox1, oy1, ox2, oy2 = phone_bbox
    ox_c = (ox1 + ox2) / 2
    oy_c = (oy1 + oy2) / 2

    # Check 1: Keypoint proximity (wrists / chest)
    if pose is not None:
        kxy, kconf = pose
        for wrist_idx in (_KP_WRIST_L, _KP_WRIST_R):
            if wrist_idx < len(kconf) and kconf[wrist_idx] >= _KP_MIN_CONF:
                wx, wy = kxy[wrist_idx]
                dist = np.hypot(ox_c - wx, oy_c - wy)
                if dist <= max(45, ph * 0.35):
                    return True

    # Check 2: Phone is in front torso / hand zone of the person
    phone_expanded = _expand_box(person_bbox, mx=0.15, my=0.10, w=img_w, h=img_h)
    ratio = _containment_ratio(phone_bbox, phone_expanded)
    if ratio >= _PROXIMITY_PHONE_CONTAINMENT:
        if py1 + 0.10 * ph <= oy_c <= py2 + 0.15 * ph:
            return True

    return False


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
    ) -> None:
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
        self._use_fp16 = use_fp16 and self._device == "cuda"

        logger.info("Loading Activity object+track model (%s) on device=%s ...", obj_path.name, self._device)
        self._obj_model = YOLO(str(obj_path))
        logger.info("Loading Activity pose model (%s) on device=%s ...", pose_path.name, self._device)
        self._pose_model = YOLO(str(pose_path))

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
            blank, imgsz=self._img_size, classes=_ACTIVITY_CLASSES,
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
            frame, imgsz=self._img_size, classes=_ACTIVITY_CLASSES,
            conf=self._track_conf, device=self._device, verbose=False,
        )
        boxes = obj_results[0].boxes
        if boxes is None or not len(boxes):
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        raw_persons: list[DetectedFace] = []
        phone_boxes: list[tuple[tuple[int, int, int, int], float]] = []  # (bbox, conf)
        computer_boxes: list[tuple[int, tuple[int, int, int, int]]] = []  # (cls, bbox)
        monitor_boxes: list[tuple[tuple[int, int, int, int], float]] = [
            (r, 1.0) for r in self._monitor_regions
        ]  # (bbox, conf)

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
            elif cls == _CLASS_PHONE:
                if score >= self._phone_conf:
                    phone_boxes.append((bbox, float(score)))
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

        tracked: list[TrackedFace] = []
        active_track_ids = set()

        for track_id, bbox, score in person_boxes:
            active_track_ids.add(track_id)

            # Match pose
            best_iou, best_pose = _POSE_IOU_MATCH_THRESHOLD, None
            for pbbox, kxy, kconf in pose_boxes:
                score_iou = _iou(bbox, pbbox)
                if score_iou > best_iou:
                    best_iou, best_pose = score_iou, (kxy, kconf)

            posture = (
                _classify_posture(
                    best_pose[0], best_pose[1], bbox,
                    self._knee_torso_ratio, self._standing_aspect, self._kp_min_conf,
                )
                if best_pose is not None
                else _posture_from_bbox(bbox, self._standing_aspect)
            )

            # --- Raw phone measurement ---
            best_phone_dist = float("inf")
            best_phone_bbox, best_phone_conf = None, 0.0
            raw_phone = False
            for p_bbox, p_conf in phone_boxes:
                if _is_phone_in_use(p_bbox, bbox, best_pose, w, h):
                    ox_c = (p_bbox[0] + p_bbox[2]) / 2
                    oy_c = (p_bbox[1] + p_bbox[3]) / 2
                    wrist_dist = float("inf")
                    if best_pose is not None:
                        kxy, kconf = best_pose
                        for wi in (_KP_WRIST_L, _KP_WRIST_R):
                            if wi < len(kconf) and kconf[wi] >= self._kp_min_conf:
                                d = float(np.hypot(ox_c - kxy[wi][0], oy_c - kxy[wi][1]))
                                wrist_dist = min(wrist_dist, d)
                    if wrist_dist < best_phone_dist:
                        best_phone_dist = wrist_dist
                        best_phone_bbox = p_bbox
                        best_phone_conf = p_conf
                        raw_phone = True
            if best_phone_dist == float("inf"):
                best_phone_dist = -1.0

            # Check computer use
            raw_computer = any(
                _is_computer_in_use(c_bbox, c_cls, bbox, best_pose, raw_phone, w, h)
                for c_cls, c_bbox in computer_boxes
            )

            # --- Raw monitor measurement ---
            # Find nearest monitor with distance, then decide interaction.
            nearest_monitor_bbox, nearest_monitor_conf = None, 0.0
            nearest_monitor_dist = float("inf")
            raw_monitor = False
            cx, cy = (bbox[0] + bbox[2]) / 2, bbox[1] + 0.25 * (bbox[3] - bbox[1])
            for m_bbox, m_conf in monitor_boxes:
                mdx = (m_bbox[0] + m_bbox[2]) / 2 - cx
                mdy = (m_bbox[1] + m_bbox[3]) / 2 - cy
                dist = float(np.hypot(mdx, mdy))
                if dist < nearest_monitor_dist:
                    nearest_monitor_dist = dist
                    nearest_monitor_bbox = m_bbox
                    nearest_monitor_conf = m_conf

            # Monitor interaction: require ALL conditions
            if nearest_monitor_bbox is not None:
                if _is_monitor_facing(
                    nearest_monitor_bbox, bbox, best_pose, posture,
                    self._monitor_requires_sitting,
                    self._monitor_max_distance_factor,
                    self._monitor_facing_tolerance,
                    self._kp_min_conf,
                ):
                    raw_monitor = True
            if nearest_monitor_dist == float("inf"):
                nearest_monitor_dist = -1.0

            # Multi-frame smoothing (anti-flicker)
            history = self._track_smooth.setdefault(
                track_id, {"phone": [], "computer": [], "monitor": []}
            )
            for key, raw in (("phone", raw_phone), ("computer", raw_computer), ("monitor", raw_monitor)):
                history[key].append(raw)
                if len(history[key]) > _SMOOTH_WINDOW:
                    history[key].pop(0)

            # Output states with short hysteresis
            phone_in_use = sum(history["phone"]) >= _PHONE_MIN_VOTES or (
                raw_phone and len(history["phone"]) < 3
            )
            computer_in_use = sum(history["computer"]) >= _COMPUTER_MIN_VOTES or (
                raw_computer and len(history["computer"]) < 3
            )
            monitor_facing = sum(history["monitor"]) >= _MONITOR_MIN_VOTES or (
                raw_monitor and len(history["monitor"]) < 3
            )

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
                    keypoints=best_pose[0] if best_pose is not None else None,
                    keypoints_conf=best_pose[1] if best_pose is not None else None,
                    head_orientation=head_label,
                    head_yaw=head_yaw,
                    head_pitch=head_pitch,
                )
            )

        # Cleanup stale smoothed tracks
        self._track_smooth = {tid: hist for tid, hist in self._track_smooth.items() if tid in active_track_ids}

        return tracked

    def close(self) -> None:
        self._track_smooth.clear()
