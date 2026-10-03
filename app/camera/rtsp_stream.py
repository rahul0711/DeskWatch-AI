"""Threaded RTSP capture with automatic reconnect.

Design notes
------------
- OpenCV's VideoCapture buffers frames internally; if we read at a slower
  pace than the camera produces them, latency grows unbounded. We run
  capture in its own thread that continuously grabs frames and always
  exposes only the *latest* one, so the main/processing loop never falls
  behind the live edge of the stream.
- Connection failures (camera reboot, network blip, wrong credentials at
  runtime) are handled by an exponential-backoff reconnect loop rather than
  crashing the process.
- This class knows nothing about faces or display — it only turns an RTSP
  URL into a stream of timestamped frames plus connection/health state.
  That separation is what lets Phase 2+ swap in GStreamer/DeepStream or a
  multi-camera manager without touching detection code.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.utils import RollingFps

logger = logging.getLogger(__name__)


class ConnectionState:
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float  # time.monotonic() when this frame was grabbed
    frame_index: int


class RTSPStream:
    """Continuously pulls frames from an RTSP URL on a background thread."""

    def __init__(
        self,
        rtsp_url: str,
        transport: str = "tcp",
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        rtsp_url_masked: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self._url = rtsp_url
        self._url_masked = rtsp_url_masked or rtsp_url
        self._name = name or self._url_masked
        self._transport = transport
        self._initial_delay = initial_reconnect_delay
        self._max_delay = max_reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[Frame] = None
        self._frame_index = 0
        self._state = ConnectionState.CONNECTING
        self._stop_event = threading.Event()

        self.resolution: tuple[int, int] = (0, 0)
        self.reported_fps: float = 0.0
        self._measured_fps = RollingFps()
        self._last_read_ms: float = 0.0
        # Anything slower than this is logged as a stall candidate -- tuned
        # to be well above a normal delta-frame read but well below a
        # keyframe decode, so we only see the events actually worth chasing.
        self._stall_threshold_ms: float = 100.0

        # Force FFmpeg to use the requested RTSP transport (tcp avoids the
        # packet loss / green-frame artifacts UDP suffers on many networks).
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self._transport}"
        )

    @property
    def state(self) -> str:
        return self._state

    def start(self) -> "RTSPStream":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rtsp-reader")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._cap is not None:
            self._cap.release()
        self._state = ConnectionState.STOPPED

    def get_latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._latest_frame

    @property
    def measured_fps(self) -> float:
        return self._measured_fps.fps

    @property
    def last_read_ms(self) -> float:
        """Wall time of the most recent cap.read() call, in milliseconds."""
        return self._last_read_ms

    # -- internal ------------------------------------------------------

    def _open(self) -> bool:
        logger.info("[%s] Connecting to RTSP stream: %s", self._name, self._url_masked)
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            # Let FFmpeg pick a hardware decoder (VideoToolbox on macOS) if
            # one is available, so decoding an I-frame doesn't stall this
            # thread on CPU software decode. No-op if unsupported by this
            # OpenCV/FFmpeg build.
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        self.resolution = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return True

    def _run(self) -> None:
        delay = self._initial_delay
        while not self._stop_event.is_set():
            self._state = ConnectionState.CONNECTING
            if not self._open():
                self._state = ConnectionState.RECONNECTING
                logger.warning(
                    "Failed to open RTSP stream, retrying in %.1fs", delay
                )
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 2, self._max_delay)
                continue

            delay = self._initial_delay  # reset backoff after a successful open
            self._state = ConnectionState.CONNECTED
            self._read_loop()
            # _read_loop only returns on failure/stop
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if not self._stop_event.is_set():
                self._state = ConnectionState.RECONNECTING

    def _read_loop(self) -> None:
        assert self._cap is not None
        consecutive_failures = 0
        while not self._stop_event.is_set():
            read_start = time.monotonic()
            ok, image = self._cap.read()
            read_ms = (time.monotonic() - read_start) * 1000
            self._last_read_ms = read_ms
            if read_ms > self._stall_threshold_ms:
                logger.warning(
                    "[%s] Slow RTSP frame read: %.0fms (frame #%s) -- likely a "
                    "keyframe decode or network hiccup, not detection",
                    self._name, read_ms, self._frame_index + 1,
                )
            if not ok or image is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    logger.warning("[%s] Lost RTSP stream (%s consecutive failed reads)", self._name, consecutive_failures)
                    return
                continue
            consecutive_failures = 0
            self._frame_index += 1
            frame = Frame(image=image, timestamp=time.monotonic(), frame_index=self._frame_index)
            with self._lock:
                self._latest_frame = frame
            self._measured_fps.tick()
