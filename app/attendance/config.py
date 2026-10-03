"""Config for the AdaFace/SCRFD/ByteTrack/FAISS attendance pipeline.

Every tunable that affects match accuracy is read from the environment (or
this dataclass's defaults) rather than hardcoded where it's used -- see
README's "assumptions that would silently break accuracy" section for why
this matters for MATCH_THRESHOLD in particular.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class AttendanceConfig:
    onnx_provider: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

    adaface_model_path: Path = field(
        default_factory=lambda: ROOT / os.getenv("ADAFACE_ONNX_PATH", "models/adaface_ir101_webface12m.onnx")
    )
    scrfd_det_size: int = field(default_factory=lambda: _int("SCRFD_DET_SIZE", 640))

    # Below this many pixels of face width, a registration image is rejected
    # outright -- per spec, an uploaded/captured enrollment photo must be a
    # clear, close, deliberate shot.
    min_register_face_px: int = field(default_factory=lambda: _int("MIN_REGISTER_FACE_PX", 100))
    # Below this width, a *live* recognition attempt is still attempted and
    # logged, but flagged low-confidence -- see README.
    low_confidence_face_px: int = field(default_factory=lambda: _int("LOW_CONFIDENCE_FACE_PX", 60))

    # Cosine similarity cutoff for a FAISS match to count as a recognition.
    # NOT a number to guess -- run scripts/calibrate_threshold.py against
    # real known/unknown crops from your camera and use its recommendation.
    # AdaFace's score distribution is NOT the same as ArcFace's; don't copy
    # a threshold from another project's ArcFace/SFace setup.
    match_threshold: float = field(default_factory=lambda: _float("MATCH_THRESHOLD", 0.35))

    db_path: Path = field(default_factory=lambda: ROOT / os.getenv("ATTENDANCE_DB_PATH", "data/attendance.db"))
    # SQLAlchemy URL, e.g. mysql+pymysql://user:pass@127.0.0.1:3306/attendance.
    # Unset/empty = SQLite at db_path.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    faiss_index_path: Path = field(
        default_factory=lambda: ROOT / os.getenv("FAISS_INDEX_PATH", "data/attendance_index.faiss")
    )
    enrollment_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("ENROLLMENT_DIR", "data/enrollment")
    )
    attendance_crops_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("ATTENDANCE_CROPS_DIR", "data/attendance_crops")
    )

    embedding_dim: int = 512

    # 1-indexed camera number from .env (CAMERA_<n>_*) -- the CP Plus unit
    # confirmed viable during live testing (see README).
    camera_index: int = field(default_factory=lambda: _int("ATTENDANCE_CAMERA_INDEX", 2))
    detection_fps: float = field(default_factory=lambda: _float("ATTENDANCE_DETECTION_FPS", 10.0))
    # How long a track may stay alive without being lost before we force a
    # finalize (embed + match + write attendance) -- stops someone who
    # lingers in frame from never getting marked.
    max_track_age_s: float = field(default_factory=lambda: _float("MAX_TRACK_AGE_S", 6.0))
    # Presentation rate of the dashboard's live view. Raised from an
    # earlier 8fps: each frame is a small (~960px, quality 70) JPEG, cheap
    # enough over a LAN/localhost that there's no bandwidth reason to
    # throttle this low, and a higher rate reads as visibly smoother.
    frame_broadcast_fps: float = field(default_factory=lambda: _float("FRAME_BROADCAST_FPS", 20.0))
    # Max age of a frame the dashboard's live view will still show, and the
    # size of the backlog banked to smooth over the camera's own periodic
    # keyframe stall (measured at ~0.7-0.9s, see README). Sized just above
    # that so a stall is covered by frames banked just before it, without
    # the view feeling behind the rest of the time.
    #
    # This value only trades latency for smoothness if the buffer is
    # correctly bounded -- an earlier version wasn't (trimmed old frames
    # only when a new one arrived, so a gap in arrivals meant nothing
    # trimmed), and 1.5s here quietly became 1.5-2s+ of *permanent* latency
    # because capture arrives faster than the dashboard drains it. Fixed by
    # trimming by age on both the fill and drain side (see worker.py) --
    # with that fix, this number means what it says.
    broadcast_buffer_seconds: float = field(default_factory=lambda: _float("BROADCAST_BUFFER_SECONDS", 1.2))

    # Cooldown period in seconds before a recognized user can punch attendance again.
    # Defaults to 600 seconds (10 minutes). During this window, repeat visits do not
    # write new attendance rows or save new crop images to disk.
    punch_cooldown_seconds: float = field(default_factory=lambda: _float("PUNCH_COOLDOWN_SECONDS", 600.0))

    # Time-to-live in seconds for storing unknown face crops on disk.
    # Defaults to 30.0 seconds. After 30s, unrecognized face crops are removed.
    unknown_face_ttl_seconds: float = field(default_factory=lambda: _float("UNKNOWN_FACE_TTL_SECONDS", 30.0))


def load_attendance_config() -> AttendanceConfig:
    return AttendanceConfig()
