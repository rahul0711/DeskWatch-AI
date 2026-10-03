"""Small shared helpers used by both the camera and processing layers."""
from __future__ import annotations

import time


class RollingFps:
    """Rolling FPS counter based on recent tick timestamps."""

    def __init__(self, window: float = 3.0) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        now = time.monotonic()
        self._timestamps.append(now)
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0
