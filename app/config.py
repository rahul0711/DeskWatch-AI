"""Central configuration, loaded entirely from environment / .env.

No credentials or camera details live in source code. Every value here can be
overridden without touching a single line of Python.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv(override=True)


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


_ALL_DETECTOR_NAMES = ("face", "person", "attendance", "activity", "ocr")


def _detector_set(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated list env var -> tuple of lowercased, stripped names.
    Absent/empty = default (the full set), so this is a no-op unless set."""
    val = os.getenv(name)
    if not val:
        return default
    return tuple(v.strip().lower() for v in val.split(",") if v.strip())


@dataclass(frozen=True)
class CameraConfig:
    # Defaults read the legacy unprefixed CAMERA_* vars, so CameraConfig()
    # with no args keeps working exactly as before for single-camera setups
    # and existing tests. Multi-camera configs go through from_env() below,
    # which passes every field explicitly and so never touches these.
    name: str = field(default_factory=lambda: os.getenv("CAMERA_NAME", "Camera 01"))
    ip: str = field(default_factory=lambda: os.getenv("CAMERA_IP", ""))
    rtsp_port: int = field(default_factory=lambda: _int("CAMERA_RTSP_PORT", 554))
    username: str = field(default_factory=lambda: os.getenv("CAMERA_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("CAMERA_PASSWORD", ""))
    rtsp_path: str = field(
        default_factory=lambda: os.getenv("CAMERA_RTSP_PATH", "/Streaming/Channels/102")
    )
    explicit_url: str = field(default_factory=lambda: os.getenv("CAMERA_RTSP_URL", ""))
    transport: str = field(
        default_factory=lambda: os.getenv("CAMERA_RTSP_TRANSPORT", "tcp")
    )
    onvif_port: int = field(default_factory=lambda: _int("ONVIF_PORT", 80))

    @staticmethod
    def from_env(prefix: str, default_name: str) -> "CameraConfig":
        """Build one camera's config from env vars under the given prefix
        (e.g. "CAMERA_1_" or the legacy unprefixed "CAMERA_")."""
        return CameraConfig(
            name=os.getenv(f"{prefix}NAME", default_name),
            ip=os.getenv(f"{prefix}IP", ""),
            rtsp_port=_int(f"{prefix}RTSP_PORT", 554),
            username=os.getenv(f"{prefix}USERNAME", ""),
            password=os.getenv(f"{prefix}PASSWORD", ""),
            rtsp_path=os.getenv(f"{prefix}RTSP_PATH", "/Streaming/Channels/102"),
            explicit_url=os.getenv(f"{prefix}RTSP_URL", ""),
            transport=os.getenv(f"{prefix}RTSP_TRANSPORT", "tcp"),
            onvif_port=_int(f"{prefix}ONVIF_PORT", 80),
        )

    @property
    def rtsp_url(self) -> str:
        """The resolved RTSP URL, never logged/printed with credentials in the clear."""
        if self.explicit_url:
            return self.explicit_url
        if not self.ip:
            raise ValueError(
                "No CAMERA_RTSP_URL and no CAMERA_IP configured. "
                "Set one of these in your .env file (see .env.example)."
            )
        auth = ""
        if self.username:
            # Credentials go through URL quoting: an '@' or ':' inside the
            # password would otherwise be mistaken for the userinfo/host
            # separator by the RTSP URL parser.
            user = quote(self.username, safe="")
            pwd = quote(self.password, safe="")
            auth = f"{user}:{pwd}@"
        path = self.rtsp_path if self.rtsp_path.startswith("/") else f"/{self.rtsp_path}"
        return f"rtsp://{auth}{self.ip}:{self.rtsp_port}{path}"

    @property
    def rtsp_url_masked(self) -> str:
        if self.explicit_url:
            # Best-effort masking for a user-supplied full URL: hide whatever
            # sits between ':' and '@' in the userinfo section, if present.
            if "@" in self.explicit_url and "://" in self.explicit_url:
                scheme, rest = self.explicit_url.split("://", 1)
                userinfo, _, hostpart = rest.partition("@")
                if ":" in userinfo:
                    user = userinfo.split(":", 1)[0]
                    return f"{scheme}://{user}:***@{hostpart}"
            return self.explicit_url
        url = self.rtsp_url
        if self.username and self.password:
            user = quote(self.username, safe="")
            pwd = quote(self.password, safe="")
            return url.replace(f"{user}:{pwd}@", f"{user}:***@")
        return url


def _load_camera_configs() -> list[CameraConfig]:
    """Discover camera configs from the environment.

    Multi-camera: CAMERA_1_IP / CAMERA_1_RTSP_URL, CAMERA_2_IP, ... -- scanned
    until the first gap, so adding a camera is just adding another numbered
    block to .env, no code change.

    Single-camera (legacy): if no CAMERA_N_* vars are set at all, falls back
    to the original unprefixed CAMERA_* vars, so existing .env files keep
    working unmodified.
    """
    cameras: list[CameraConfig] = []
    n = 1
    while os.getenv(f"CAMERA_{n}_IP") or os.getenv(f"CAMERA_{n}_RTSP_URL"):
        cameras.append(CameraConfig.from_env(f"CAMERA_{n}_", default_name=f"Camera {n:02d}"))
        n += 1
    if cameras:
        return cameras

    if os.getenv("CAMERA_IP") or os.getenv("CAMERA_RTSP_URL"):
        return [CameraConfig.from_env("CAMERA_", default_name="Camera 01")]

    return []


@dataclass(frozen=True)
class DetectionConfig:
    # "yolo" (default) | "insightface"/"scrfd" | "mtcnn" -- see
    # app/detection/face_detector.py's module docstring for tradeoffs.
    backend: str = field(
        default_factory=lambda: os.getenv("DETECTOR_BACKEND", "yolo").lower().strip()
    )
    onnx_provider: str = field(
        default_factory=lambda: os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")
    )
    det_size: int = field(default_factory=lambda: _int("DETECTION_SIZE", 320))
    conf_threshold: float = field(
        default_factory=lambda: _float("DETECTION_CONF_THRESHOLD", 0.5)
    )
    # Vestigial as of Phase 2.1 -- folded into AppConfig.max_detection_fps
    # (a direct rate instead of a "every Nth frame" counter). Still parsed
    # so an old .env doesn't error, but FacePipeline no longer reads it.
    interval: int = field(default_factory=lambda: max(1, _int("DETECTION_INTERVAL", 3)))

    # What the web app's "head count per minute" table counts by:
    #   face       -- count tracked faces only (YOLO face detector)
    #   person     -- count detected people/bodies instead (YOLO person detector)
    #   attendance -- Face Attendance mode using InsightFace (SCRFD + ArcFace)
    #   activity   -- sitting/phone/computer tracking (YOLO + BoT-SORT + pose)
    #   ocr        -- scene-text OCR, static + moving (see app/detection/ocr_detector.py)
    head_count_source: str = field(
        default_factory=lambda: os.getenv("HEAD_COUNT_SOURCE", "face").lower().strip()
    )
    person_conf_threshold: float = field(
        default_factory=lambda: _float("PERSON_CONF_THRESHOLD", 0.4)
    )
    # Which of the 5 per-camera detector worker subprocesses to actually
    # start (face, person, attendance, activity, ocr) -- see FacePipeline
    # in app/processing/pipeline.py. Defaults to all 5 (current behavior)
    # so an absent/unset ENABLED_DETECTORS changes nothing.
    enabled_detectors: tuple[str, ...] = field(
        default_factory=lambda: _detector_set("ENABLED_DETECTORS", _ALL_DETECTOR_NAMES)
    )


@dataclass(frozen=True)
class OcrConfig:
    """Scene-text OCR mode (HEAD_COUNT_SOURCE=ocr) -- see
    app/detection/ocr_detector.py's module docstring for the full pipeline.
    """
    det_limit_side_len: int = field(default_factory=lambda: _int("OCR_DET_LIMIT_SIDE_LEN", 1920))
    det_thresh: float = field(default_factory=lambda: _float("OCR_DET_THRESH", 0.3))
    det_box_thresh: float = field(default_factory=lambda: _float("OCR_DET_BOX_THRESH", 0.5))
    det_unclip_ratio: float = field(default_factory=lambda: _float("OCR_DET_UNCLIP_RATIO", 1.6))
    # Default to the "mobile" det/rec models -- measured on this CPU-only
    # Apple Silicon Mac (no GPU acceleration available for paddlepaddle
    # here): ~1.3s/frame at native 1080p vs ~7-9s/frame for PaddleOCR's own
    # default "medium" models, for a small accuracy cost (still correctly
    # read both a wall-sign-style and small-print test string at >0.96
    # confidence). Set OCR_DET_MODEL/OCR_REC_MODEL to e.g. PP-OCRv6_medium_det
    # / PP-OCRv6_medium_rec (or *_server_* on a GPU box) for higher accuracy
    # at that cost, or "" to fall back to PaddleOCR's own current default.
    det_model_name: str = field(default_factory=lambda: os.getenv("OCR_DET_MODEL", "PP-OCRv5_mobile_det"))
    rec_model_name: str = field(default_factory=lambda: os.getenv("OCR_REC_MODEL", "PP-OCRv5_mobile_rec"))
    rec_min_confidence: float = field(default_factory=lambda: _float("OCR_REC_MIN_CONFIDENCE", 0.35))
    min_crop_height_px: int = field(default_factory=lambda: _int("OCR_MIN_CROP_HEIGHT_PX", 10))
    target_rec_height: int = field(default_factory=lambda: _int("OCR_TARGET_REC_HEIGHT", 48))
    max_upscale: float = field(default_factory=lambda: _float("OCR_MAX_UPSCALE", 6.0))
    max_regions_per_frame: int = field(default_factory=lambda: _int("OCR_MAX_REGIONS_PER_FRAME", 40))
    carrier_conf_threshold: float = field(default_factory=lambda: _float("OCR_CARRIER_CONF_THRESHOLD", 0.35))
    max_carriers_per_frame: int = field(default_factory=lambda: _int("OCR_MAX_CARRIERS_PER_FRAME", 6))
    carrier_crop_max_side: int = field(default_factory=lambda: _int("OCR_CARRIER_CROP_MAX_SIDE", 800))
    carrier_upscale: float = field(default_factory=lambda: _float("OCR_CARRIER_UPSCALE", 2.0))
    enable_carrier_pass: bool = field(default_factory=lambda: _bool("OCR_ENABLE_CARRIER_PASS", True))
    # PaddlePaddle's oneDNN CPU kernels crash these models on Windows (see
    # OCRDetector.__init__), which would make OCR mode silently read nothing.
    # None = let the detector pick per-platform: off on Windows, on elsewhere.
    enable_mkldnn: bool | None = field(
        default_factory=lambda: _bool("OCR_ENABLE_MKLDNN", True)
        if os.getenv("OCR_ENABLE_MKLDNN") else None
    )


@dataclass(frozen=True)
class CropSavingConfig:
    enabled: bool = field(default_factory=lambda: _bool("SAVE_FACE_CROPS", False))
    directory: Path = field(
        default_factory=lambda: Path(os.getenv("SAVE_CROPS_DIR", "data/samples"))
    )
    max_saved: int = field(default_factory=lambda: _int("SAVE_CROPS_MAX", 20))


@dataclass(frozen=True)
class ReconnectConfig:
    initial_delay: float = field(
        default_factory=lambda: _float("RECONNECT_INITIAL_DELAY", 1.0)
    )
    max_delay: float = field(default_factory=lambda: _float("RECONNECT_MAX_DELAY", 30.0))


def _resolve_fps(specific_name: str, specific_default: float) -> float:
    """MAX_DISPLAY_FPS / MAX_DETECTION_FPS, each falling back to the legacy
    MAX_PROCESS_FPS (which used to gate both at once -- see FacePipeline.poll)
    if set and the specific var isn't, else the given default. So an old
    .env with only MAX_PROCESS_FPS set keeps behaving as before (one rate
    for both); setting either new var overrides just that one."""
    specific_val = os.getenv(specific_name)
    if specific_val:
        return float(specific_val)
    legacy_val = os.getenv("MAX_PROCESS_FPS")
    if legacy_val:
        return float(legacy_val)
    return specific_default


@dataclass(frozen=True)
class AppConfig:
    cameras: list[CameraConfig] = field(default_factory=_load_camera_configs)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    crops: CropSavingConfig = field(default_factory=CropSavingConfig)
    reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)
    # Caps how often poll() returns a frame for rendering -- the displayed
    # stream's rate, never throttled by detection. See FacePipeline.poll()
    # in app/processing/pipeline.py.
    max_display_fps: float = field(default_factory=lambda: _resolve_fps("MAX_DISPLAY_FPS", 25.0))
    # Caps how often a frame is submitted to the detector -- independent of
    # display. DETECTION_INTERVAL is now vestigial (folded into this, see
    # FacePipeline.poll()'s docstring) -- kept parsed below for anyone still
    # setting it, but it no longer affects anything.
    max_detection_fps: float = field(default_factory=lambda: _resolve_fps("MAX_DETECTION_FPS", 10.0))

    @property
    def camera(self) -> CameraConfig:
        """First configured camera -- convenience for single-camera tools
        (scripts/test_connection.py, scripts/benchmark.py) that don't need
        to handle a multi-camera setup."""
        if not self.cameras:
            raise ValueError(
                "No cameras configured. Set CAMERA_IP/CAMERA_RTSP_URL (single "
                "camera) or CAMERA_1_IP, CAMERA_2_IP, ... (multiple) in .env."
            )
        return self.cameras[0]


def load_config() -> AppConfig:
    return AppConfig()
