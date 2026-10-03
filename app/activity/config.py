"""Config for the human activity analysis layer (app/activity/).

Follows the same env-driven, frozen-dataclass pattern as
app/attendance/config.py -- every threshold that affects a posture/movement/
interaction decision is read from the environment rather than hardcoded at
its use site, so tuning this against a real camera never means editing
Python.

Nothing here is read at import time by the detector module itself: the
detector takes plain kwargs (so it stays picklable for the spawn-based
ProcessDetectionWorker), and server.py/main.py build those kwargs from this
config. See app/activity/README-ish notes in analyzer.py for the data flow.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# app/config.py also does this, but relying on it means every ACTIVITY_*
# value silently falls back to its default whenever this module is imported
# first (a test, a script, a worker subprocess that only needs activity
# config). load_dotenv is idempotent, so doing it here too is free.
load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent.parent


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    val = os.getenv(name)
    if not val:
        return default
    return tuple(v.strip().lower() for v in val.split(",") if v.strip())


def _parse_monitor_regions(raw: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """ACTIVITY_MONITOR_REGIONS -- optional fixed workstation/monitor zones.

    Format: "camera-id:x1,y1,x2,y2; camera-id:x1,y1,x2,y2; other-cam:..."
    Multiple regions for one camera are allowed (repeat the camera id).

    Entirely optional: with no regions configured, monitor interaction is
    decided purely off YOLO's own detected monitor boxes. A configured
    region is additive -- it is treated as a monitor that is always present,
    which is how you pin a workstation whose screen YOLO keeps missing
    (dark screen, extreme angle).
    """
    regions: dict[str, list[tuple[int, int, int, int]]] = {}
    if not raw:
        return regions
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        cam_id, _, coords = chunk.partition(":")
        parts = [p.strip() for p in coords.split(",")]
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(float(p)) for p in parts)
        except ValueError:
            continue
        regions.setdefault(cam_id.strip().lower(), []).append((x1, y1, x2, y2))
    return regions


@dataclass(frozen=True)
class ActivityConfig:
    # --- Which cameras run activity analysis ---------------------------------
    # Comma-separated camera ids (the slug server.py derives from the camera
    # name, e.g. "camera-01"). Activity adds a YOLOv8x + YOLOv8x-pose pass
    # per camera, so this deliberately defaults to one camera rather than
    # every configured camera.
    cameras: tuple[str, ...] = field(
        default_factory=lambda: _str_tuple("ACTIVITY_CAMERAS", ("camera-01",))
    )

    # --- Models --------------------------------------------------------------
    # YOLOv8x / YOLOv8x-pose per spec. Both are configurable because they
    # dominate the frame budget: measured on this RTX 3060 at imgsz=960,
    # yolov8x + yolov8x-pose costs ~236ms/frame (~4.2 fps ceiling), while
    # the lighter yolo11m/yolo11n-pose pair runs several times faster at
    # some accuracy cost. Swap via .env, no code change.
    obj_model: Path = field(
        default_factory=lambda: ROOT / os.getenv("ACTIVITY_OBJ_MODEL", "models/yolov8x.pt")
    )
    pose_model: Path = field(
        default_factory=lambda: ROOT / os.getenv("ACTIVITY_POSE_MODEL", "models/yolov8x-pose.pt")
    )
    tracker_cfg: Path = field(
        default_factory=lambda: ROOT / os.getenv("ACTIVITY_TRACKER_CFG", "models/activity_botsort.yaml")
    )
    # Persons are tracked separately from the object pass -- see
    # models/activity_person_bytetrack.yaml for why.
    person_tracker_cfg: Path = field(
        default_factory=lambda: ROOT / os.getenv(
            "ACTIVITY_PERSON_TRACKER_CFG", "models/activity_person_bytetrack.yaml"
        )
    )
    # Ultralytics resizes the long side to this before inference. 960 is not
    # arbitrary: at 640 the phone and mouse classes went undetected entirely
    # on this camera's 1920x1080 frames, at 960 both appear. Raising it to
    # 1280 roughly doubles cost for marginal gain.
    img_size: int = field(default_factory=lambda: _int("ACTIVITY_IMG_SIZE", 960))
    # Pose gets its own, smaller resolution: bodies are large and easy, so
    # the pose pass gains nothing from the resolution the object pass needs
    # for small phones. Measured: 640 costs 138ms vs 202ms at 960, with no
    # observable difference in keypoint quality at this camera distance.
    pose_img_size: int = field(default_factory=lambda: _int("ACTIVITY_POSE_IMG_SIZE", 640))
    # "auto" -> cuda if available, else mps, else cpu. Force with "cuda"/"cpu".
    device: str = field(default_factory=lambda: os.getenv("ACTIVITY_DEVICE", "auto").strip().lower())

    # --- Inference rate ------------------------------------------------------
    # How often a frame is handed to the activity worker. Independent of
    # both display fps and the face/attendance detection rate -- activity
    # states persist for minutes, so sampling them slower than faces is
    # free accuracy-wise and keeps the GPU available for recognition.
    fps: float = field(default_factory=lambda: _float("ACTIVITY_FPS", 4.0))

    # --- Detection confidences ----------------------------------------------
    person_conf: float = field(default_factory=lambda: _float("ACTIVITY_PERSON_CONF", 0.35))
    # Phones are small and usually partly occluded by a hand; a high floor
    # here means never seeing one at all.
    phone_conf: float = field(default_factory=lambda: _float("ACTIVITY_PHONE_CONF", 0.12))
    computer_conf: float = field(default_factory=lambda: _float("ACTIVITY_COMPUTER_CONF", 0.25))
    monitor_conf: float = field(default_factory=lambda: _float("ACTIVITY_MONITOR_CONF", 0.30))
    pose_conf: float = field(default_factory=lambda: _float("ACTIVITY_POSE_CONF", 0.35))
    keypoint_min_conf: float = field(default_factory=lambda: _float("ACTIVITY_KEYPOINT_MIN_CONF", 0.25))

    # --- Hand-zone phone pass (OFF by default) -------------------------------
    # Re-runs the detector on an upscaled crop of each person's upper body,
    # purely to find phones, on the theory that a phone is only ~25x13px
    # after the main pass downscales the frame.
    #
    # Measured on this camera and left OFF: it costs ~610ms/frame (4 person
    # crops through YOLOv8x) and did NOT beat the plain full-frame pass on
    # recall -- full-frame@960 found phone boxes in 17 of 25 frames, the
    # crop pass in 6 of 25. It produced higher peak confidence in one of
    # two trials, so it is kept available for cameras where people sit much
    # further away, but it is not worth 3x the frame budget here.
    phone_crop_pass: bool = field(default_factory=lambda: _bool("ACTIVITY_PHONE_CROP_PASS", False))
    phone_crop_upscale: float = field(default_factory=lambda: _float("ACTIVITY_PHONE_CROP_UPSCALE", 2.0))
    phone_crop_imgsz: int = field(default_factory=lambda: _int("ACTIVITY_PHONE_CROP_IMGSZ", 640))

    # --- Posture -------------------------------------------------------------
    # lower-leg-length : torso-length ratio below which a person reads as
    # seated. Determine experimentally for your camera height/angle: set
    # ACTIVITY_DEBUG_OVERLAY=true and watch the per-track posture label on
    # the live stream while someone sits and stands.
    sitting_knee_torso_ratio: float = field(
        default_factory=lambda: _float("ACTIVITY_SITTING_KNEE_TORSO_RATIO", 0.5)
    )
    # Fallback when hips/knees are occluded (common on a ceiling-mounted
    # camera with desks in the way): person-box height:width at or above
    # this reads as standing.
    standing_bbox_aspect: float = field(
        default_factory=lambda: _float("ACTIVITY_STANDING_BBOX_ASPECT", 1.6)
    )

    # --- Movement ------------------------------------------------------------
    # Movement score is normalized by the person's bbox diagonal, so a
    # distant small figure and a close large one are comparable, then
    # exponentially smoothed. Thresholds are in those normalized units
    # (fraction of body diagonal moved per second).
    movement_ema_alpha: float = field(default_factory=lambda: _float("ACTIVITY_MOVEMENT_EMA_ALPHA", 0.4))
    walking_threshold: float = field(default_factory=lambda: _float("ACTIVITY_WALKING_THRESHOLD", 0.25))
    stationary_threshold: float = field(default_factory=lambda: _float("ACTIVITY_STATIONARY_THRESHOLD", 0.06))
    # How long movement must stay under stationary_threshold before the
    # person is reported stationary (the timer then runs until real
    # movement resumes).
    stationary_min_seconds: float = field(
        default_factory=lambda: _float("ACTIVITY_STATIONARY_MIN_SECONDS", 3.0)
    )
    # Per-keypoint-group weights for the movement score. Wrists/elbows are
    # weighted up because hand motion while seated is the signal that
    # separates "at the desk working" from "asleep at the desk".
    movement_weights: tuple[float, ...] = field(
        default_factory=lambda: (
            _float("ACTIVITY_W_CENTER", 1.0),
            _float("ACTIVITY_W_NOSE", 0.8),
            _float("ACTIVITY_W_SHOULDER", 0.8),
            _float("ACTIVITY_W_ELBOW", 1.0),
            _float("ACTIVITY_W_WRIST", 1.2),
            _float("ACTIVITY_W_HIP", 0.6),
            _float("ACTIVITY_W_KNEE", 0.6),
        )
    )

    # --- Monitor interaction -------------------------------------------------
    # A monitor counts as "this person's" when its centre lies within this
    # many body-diagonals of the person's head/upper body.
    monitor_max_distance_factor: float = field(
        default_factory=lambda: _float("ACTIVITY_MONITOR_MAX_DISTANCE_FACTOR", 1.5)
    )
    # Require a seated posture before reporting monitor interaction. On a
    # desk-facing camera this removes most false hits from people walking
    # past a screen.
    monitor_requires_sitting: bool = field(
        default_factory=lambda: _bool("ACTIVITY_MONITOR_REQUIRES_SITTING", True)
    )
    # How far the head may be turned away from the monitor direction and
    # still count as facing it, as a fraction of shoulder width. Derived
    # from nose offset relative to the shoulder midpoint -- a proxy for
    # head yaw that needs no extra model.
    monitor_facing_tolerance: float = field(
        default_factory=lambda: _float("ACTIVITY_MONITOR_FACING_TOLERANCE", 0.55)
    )
    monitor_regions: dict[str, list[tuple[int, int, int, int]]] = field(
        default_factory=lambda: _parse_monitor_regions(os.getenv("ACTIVITY_MONITOR_REGIONS", ""))
    )

    # --- Temporal state machine ---------------------------------------------
    # A candidate state must hold for this long before it replaces the
    # current one. This is what stops sitting->standing->sitting flicker
    # from a single bad pose frame.
    confirmation_seconds: float = field(
        default_factory=lambda: _float("ACTIVITY_CONFIRMATION_SECONDS", 2.0)
    )
    # Keep the last known identity on a track for this long after face
    # recognition stops confirming it (head turned, occlusion, blur).
    identity_grace_seconds: float = field(
        default_factory=lambda: _float("ACTIVITY_IDENTITY_GRACE_SECONDS", 5.0)
    )
    # How long a track may go unseen before it is treated as having left
    # (closes its open event and emits PERSON_LEFT). Must comfortably
    # exceed the tracker's own track_buffer in wall-clock terms.
    track_grace_seconds: float = field(
        default_factory=lambda: _float("ACTIVITY_TRACK_GRACE_SECONDS", 10.0)
    )
    max_tracks: int = field(default_factory=lambda: _int("ACTIVITY_MAX_TRACKS", 200))

    # --- Persistence ---------------------------------------------------------
    persist_events: bool = field(default_factory=lambda: _bool("ACTIVITY_PERSIST_EVENTS", True))
    # Events shorter than this are dropped rather than written -- keeps the
    # table free of sub-second transition noise.
    min_event_seconds: float = field(default_factory=lambda: _float("ACTIVITY_MIN_EVENT_SECONDS", 2.0))
    # A track must have been seen in at least this many detection passes
    # before any of its events are stored or broadcast. At ACTIVITY_FPS=4
    # the default is ~2s of continuous presence. This is what keeps ghost
    # tracks (the person detector momentarily firing on a chair or a
    # reflection) out of the event table.
    min_track_observations: int = field(
        default_factory=lambda: _int("ACTIVITY_MIN_TRACK_OBSERVATIONS", 8)
    )
    # How many finished events to keep in memory per camera for the API's
    # timeline view (the database keeps all of them).
    max_recent_events: int = field(default_factory=lambda: _int("ACTIVITY_MAX_RECENT_EVENTS", 500))

    # --- GPU acceleration ----------------------------------------------------
    # FP16 (half precision) halves memory bandwidth and nearly doubles
    # throughput on the RTX 3060's Tensor Cores. Measured: combined
    # obj+pose drops from ~313ms to ~171ms per frame (1.8x). No observable
    # accuracy loss for detection/pose at these resolutions.
    use_fp16: bool = field(default_factory=lambda: _bool("ACTIVITY_FP16", True))

    # --- Debug ---------------------------------------------------------------
    # Draws the pose skeleton, movement score and live timers on the video
    # overlay. Costs real render time on a 1080p frame, so it is off by
    # default and toggleable per camera at runtime via the existing
    # PATCH /api/cameras/{id}/show-hud-style endpoints.
    debug_overlay: bool = field(default_factory=lambda: _bool("ACTIVITY_DEBUG_OVERLAY", False))

    def resolved_device(self) -> str:
        """'auto' -> the best available backend, else whatever was forced.

        Never raises on a missing GPU: a clear warning plus CPU fallback is
        the documented behaviour, since activity analysis going slow is far
        better than the whole CCTV service refusing to start.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
        except Exception:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detector_kwargs(self, camera_id: str) -> dict:
        """Plain picklable kwargs for ActivityDetector in the worker process."""
        return dict(
            obj_model=str(self.obj_model),
            pose_model=str(self.pose_model),
            tracker_cfg=str(self.tracker_cfg),
            img_size=self.img_size,
            pose_img_size=self.pose_img_size,
            device=self.resolved_device(),
            conf_threshold=self.person_conf,
            phone_conf=self.phone_conf,
            computer_conf=self.computer_conf,
            monitor_conf=self.monitor_conf,
            pose_conf_threshold=self.pose_conf,
            keypoint_min_conf=self.keypoint_min_conf,
            sitting_knee_torso_ratio=self.sitting_knee_torso_ratio,
            standing_bbox_aspect=self.standing_bbox_aspect,
            monitor_max_distance_factor=self.monitor_max_distance_factor,
            monitor_requires_sitting=self.monitor_requires_sitting,
            monitor_facing_tolerance=self.monitor_facing_tolerance,
            monitor_regions=tuple(self.monitor_regions.get(camera_id.lower(), ())),
            person_tracker_cfg=str(self.person_tracker_cfg),
            activity_fps=self.fps,
            phone_crop_pass=self.phone_crop_pass,
            phone_crop_upscale=self.phone_crop_upscale,
            phone_crop_imgsz=self.phone_crop_imgsz,
            use_fp16=self.use_fp16,
        )

    def enabled_for(self, camera_id: str) -> bool:
        return camera_id.lower() in self.cameras


def load_activity_config() -> ActivityConfig:
    return ActivityConfig()
